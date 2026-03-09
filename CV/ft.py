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
    parser = argparse.ArgumentParser(description='Unlearning with Finetuning (FT) Algorithm')
    parser.add_argument('--model_path', default='checkpoint', help='Path to saved model checkpoint')
    parser.add_argument('--split_dir', default='class', help='path to forget/retain indices')
    parser.add_argument('--output_dir', default='FT_unlearned_models', help='Directory to save unlearned model')
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate for finetuning')
    parser.add_argument('--epochs', default=10, type=int, help='Number of finetuning epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载数据集
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # 加载划分索引
    forget_indices = np.load(
        os.path.join('split_results_' + args.split_dir, 'forget_indices_' + args.split_dir + '_0.npy'))
    retain_indices = np.load(
        os.path.join('split_results_' + args.split_dir, 'retain_indices_' + args.split_dir + '_0.npy'))

    # 创建DataLoader（FT方法仅使用保留集进行微调）
    retainset = Subset(full_trainset, retain_indices)
    forgetset = Subset(full_trainset, forget_indices)

    retain_loader = torch.utils.data.DataLoader(
        retainset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    forget_loader = torch.utils.data.DataLoader(
        forgetset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # 加载模型
    from models.moe import MOE_ViT
    init_ckpt_path = os.path.join(args.model_path, 'moe_vit_cifar10_best.pth')
    checkpoint = torch.load(init_ckpt_path)

    net = MOE_ViT(num_experts=4, size=32).to(device)
    for expert in net.experts:
        expert.vit.head = nn.Linear(expert.vit.head.in_features, 10).to(device)

    net.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path} (epoch {checkpoint['epoch']})")

    # 优化器（微调通常使用较小的学习率）
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 评估函数
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

    # 计算所有指标
    def compute_metrics():
        fa = evaluate(forget_loader, "Forget Set")
        ra = evaluate(retain_loader, "Retain Set")
        ta = evaluate(test_loader, "Test Set")
        ua = 100.0 - fa
        mia = compute_mia(net, forget_loader, retain_loader, device)
        return ua, ra, ta, fa, mia

    # 初始指标
    print("\n=== Initial Metrics ===")
    ua_init = 2.24
    ra_init = 95.98
    ta_init = 81.81
    print(f"UA: {ua_init:.2f}% | RA: {ra_init:.2f}% | TA: {ta_init:.2f}% ")

    # FT微调主循环（仅使用保留集）
    print("\n=== Starting Finetuning (FT) Unlearning ===")
    start_time = time.time()
    best_ua = ua_init
    best_ta = ta_init

    for epoch in range(args.epochs):
        net.train()
        total_loss = 0.0

        pbar = tqdm(retain_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            optimizer.zero_grad()

            # 仅在保留集上进行微调
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)  # 仅最小化保留集损失

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"})

        # 每轮指标
        ua, ra, ta, fa, mia = compute_metrics()

        # 保存最佳模型
        if ua > best_ua and ta > best_ta * 0.9:
            best_ua, best_ta = ua, ta
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {'UA': ua, 'RA': ra, 'TA': ta, 'MIA': mia}
            }, os.path.join(args.output_dir, 'unlearned_best.pth'))

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"UA: {ua:.2f}% (Best: {best_ua:.2f}%) | RA: {ra:.2f}% | TA: {ta:.2f}% | MIA: {mia:.4f}")

    # 最终指标
    rte = (time.time() - start_time) / 60.0
    print("\n=== Final Unlearning Metrics ===")
    final_ua, final_ra, final_ta, final_fa, final_mia = compute_metrics()
    print(f"UA: {final_ua:.2f}% (↑{final_ua - ua_init:.2f}%)")
    print(f"RA: {final_ra:.2f}% (↓{ra_init - final_ra:.2f}%)")
    print(f"TA: {final_ta:.2f}% (↓{ta_init - final_ta:.2f}%)")
    print(f"MIA: {final_mia:.4f}")
    print(f"RTE: {rte:.2f} minutes")


if __name__ == '__main__':
    main()
