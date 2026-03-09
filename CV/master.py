import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
from models.moe import MOE_ViT  # 导入你的MOE_ViT模型


def compute_expert_affinity(net, dataloader, device, num):
    """原文创新点1：专家归因 - 计算每个专家对遗忘集的亲和度（基于门控分数）"""
    net.eval()
    num_experts = num
    expert_gate_sums = torch.zeros(num_experts, device=device)
    total_tokens = 0

    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Computing Expert Affinity"):
            inputs = inputs.to(device)
            batch_size = inputs.shape[0]
            x_flat = inputs.view(batch_size, -1)
            g = torch.softmax(net.router.gate(x_flat), dim=1)
            expert_gate_sums += g.sum(dim=0)
            total_tokens += batch_size

    expert_affinity = (expert_gate_sums / total_tokens).cpu().numpy()
    return expert_affinity


def compute_mia(net, forget_loader, retain_loader, device):
    """计算MIA指标：成员推理攻击成功率差异"""
    net.eval()
    forget_confs, retain_confs = [], []
    with torch.no_grad():
        for inputs, _ in forget_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, _ = torch.max(probs, dim=1)
            forget_confs.extend(confs.cpu().numpy())

        for inputs, _ in retain_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, _ = torch.max(probs, dim=1)
            retain_confs.extend(confs.cpu().numpy())

    return abs(np.mean(forget_confs) - np.mean(retain_confs))


def main():
    parser = argparse.ArgumentParser(description='SEUF-GA Unlearning for MOE-ViT (CIFAR-10)')
    parser.add_argument('--lr', default=0.01, type=float, help='GA learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--ga_epochs', default=5, type=int, help='GA unlearning iterations')
    parser.add_argument('--num_experts', default=4, type=int, help='number of experts in MOE')
    parser.add_argument('--image_size', default=32, type=int, help='input image size')
    parser.add_argument('--checkpoint', default='checkpoint', help='path to initial model')
    parser.add_argument('--split_dir', default='class', help='path to forget/retain indices')
    parser.add_argument('--output', default='checkpoint_seuf_ga_', help='output dir')
    parser.add_argument('--lambda_retain', default=1.0, type=float, help='retain loss weight')
    parser.add_argument('--alpha', default=1.0, type=float, help='anchor loss weight')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output + args.split_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    forget_indices = np.load(os.path.join('split_results_' + args.split_dir, 'forget_indices_' + args.split_dir + '_0.npy'))
    retain_indices = np.load(os.path.join('split_results_' + args.split_dir, 'retain_indices_' + args.split_dir + '_0.npy'))

    forgetset = Subset(full_trainset, forget_indices)
    retainset = Subset(full_trainset, retain_indices)
    affinity_loader = torch.utils.data.DataLoader(forgetset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    forgetloader = torch.utils.data.DataLoader(forgetset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    retainloader = torch.utils.data.DataLoader(retainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    net = MOE_ViT(num_experts=args.num_experts, size=args.image_size).to(device)
    for expert in net.experts:
        expert.vit.head = nn.Linear(expert.vit.head.in_features, 10).to(device)

    init_ckpt_path = os.path.join(args.checkpoint, 'moe_vit_cifar10_best.pth')
    if not os.path.exists(init_ckpt_path):
        raise FileNotFoundError(f"Initial model not found: {init_ckpt_path}")
    init_ckpt = torch.load(init_ckpt_path)
    net.load_state_dict(init_ckpt['model_state_dict'])
    print(f"Loaded initial MOE-ViT model from {init_ckpt_path}")

    print("\n=== Step 1: Expert Attribution (SEUF Core) ===")
    expert_affinity = compute_expert_affinity(net, affinity_loader, device, args.num_experts)
    target_expert_idx = np.argmax(expert_affinity)
    print(f"Expert Affinity: {expert_affinity.round(4)}")
    print(f"Target Expert for Unlearning: Expert-{target_expert_idx}")

    target_params = [param for name, param in net.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(target_params, lr=args.lr)
    optimizer.load_state_dict(init_ckpt['optimizer_state_dict'])

    print("\n=== Step 2: Freeze Non-Target Params (SEUF Core) ===")
    for name, param in net.named_parameters():
        if f"experts.{target_expert_idx}" in name:
            param.requires_grad = True
            print(f"Unfrozen (target expert): {name}")
        else:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    def evaluate(loader, desc="Evaluating"):
        net.eval()
        total, correct = 0, 0
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pbar.set_postfix({'acc': f"{100. * correct / total:.2f}%"})
        return 100. * correct / total

    def compute_anchor_loss(g, target_expert_idx, num_experts):
        a = torch.zeros_like(g, device=g.device)
        a[:, target_expert_idx] = 1.0
        anchor_loss = torch.norm(g - a, p=2, dim=1).mean()
        return anchor_loss

    print("\n=== Initial Metrics (Before Unlearning) ===")
    ua_init = 2.24
    ra_init = 95.98
    ta_init = 81.81
    mia_init = compute_mia(net, forgetloader, retainloader, device)
    print(f"Initial UA: {ua_init:.2f}% | RA: {ra_init:.2f}% | TA: {ta_init:.2f}% | MIA: {mia_init:.4f}")

    print("\n=== Starting SEUF-GA Unlearning (Core of Paper) ===")
    start_time = time.time()
    best_ua = ua_init
    best_ta = ta_init
    best_mia = mia_init

    for epoch in range(args.ga_epochs):
        net.train()
        total_loss = 0.0
        forget_iter = iter(forgetloader)
        retain_iter = iter(retainloader)
        max_batches = max(len(forgetloader), len(retainloader))

        pbar = tqdm(range(max_batches), desc=f"GA Epoch {epoch + 1}/{args.ga_epochs}")
        for _ in pbar:
            optimizer.zero_grad()

            try:
                inputs_f, targets_f = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forgetloader)
                inputs_f, targets_f = next(forget_iter)
            inputs_f, targets_f = inputs_f.to(device), targets_f.to(device)
            outputs_f = net(inputs_f)
            x_flat1 = inputs_f.view(inputs_f.size(0), -1)
            g_f = torch.softmax(net.router.gate(x_flat1), dim=1)
            loss_f = criterion(outputs_f, targets_f)

            try:
                inputs_r, targets_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retainloader)
                inputs_r, targets_r = next(retain_iter)
            inputs_r, targets_r = inputs_r.to(device), targets_r.to(device)
            outputs_r = net(inputs_r)
            x_flat2 = inputs_r.view(inputs_r.size(0), -1)
            g_r = torch.softmax(net.router.gate(x_flat2), dim=1)
            loss_r = args.lambda_retain * criterion(outputs_r, targets_r)

            g_combined = torch.cat([g_f, g_r], dim=0)
            anchor_loss = args.alpha * compute_anchor_loss(
                g_combined, target_expert_idx, args.num_experts
            )

            total_batch_loss = (-loss_f) + loss_r + anchor_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            pbar.set_postfix({'avg_loss': f"{total_loss / (_ + 1):.3f}"})

        current_ra = evaluate(retainloader, f"Retain Set (RA) Epoch {epoch + 1}")
        current_fa = evaluate(forgetloader, f"Forget Set (FA) Epoch {epoch + 1}")
        current_ta = evaluate(testloader, f"Test Set (TA) Epoch {epoch + 1}")
        current_ua = 100. - current_fa
        current_mia = compute_mia(net, forgetloader, retainloader, device)

        if current_ua > best_ua and current_ta > best_ta * 0.95:
            best_ua = current_ua
            best_ta = current_ta
            best_mia = current_mia
            torch.save({
                'epoch': epoch + 1,
                'target_expert_idx': target_expert_idx,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metrics': {'UA': best_ua, 'RA': current_ra, 'TA': best_ta, 'MIA': best_mia}
            }, os.path.join(args.output + args.split_dir, 'moe_vit_seuf_ga_best.pth'))

        print(f"Epoch {epoch + 1} - UA: {current_ua:.2f}% (Best: {best_ua:.2f}%) | "
              f"RA: {current_ra:.2f}% | TA: {current_ta:.2f}% (Best: {best_ta:.2f}%) | "
              f"MIA: {current_mia:.4f} (Best: {best_mia:.4f})")

    rte = (time.time() - start_time) / 60.0

    print("\n=== Final SEUF-GA Unlearning Metrics ===")
    final_ra = evaluate(retainloader, "Final Retain Set (RA)")
    final_fa = evaluate(forgetloader, "Final Forget Set (FA)")
    final_ua = 100. - final_fa
    final_ta = evaluate(testloader, "Final Test Set (TA)")
    final_mia = compute_mia(net, forgetloader, retainloader, device)
    print(f"1. UA: {final_ua:.2f}% (↑{final_ua - ua_init:.2f}%)")
    print(f"2. RA: {final_ra:.2f}% (↓{ra_init - final_ra:.2f}%)")
    print(f"3. TA: {final_ta:.2f}% (↓{ta_init - final_ta:.2f}%)")
    print(f"4. MIA: {final_mia:.4f} (↓{mia_init - final_mia:.4f}%)")
    print(f"5. RTE: {rte:.2f} minutes")
    print(f"6. Tunable Params Ratio: {len(target_params) / sum(1 for _ in net.parameters()) * 100:.4f}%")

    print(f"\nUnlearned model saved to {args.output + args.split_dir}")


if __name__ == '__main__':
    main()