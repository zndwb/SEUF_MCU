import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torchvision.transforms import RandAugment
import time
import os
from try_moevit import MoEVisionTransformer  # 请替换为你的MoEVisionTransformer路径

ga_epoches = 10
lr = 0.00005
ckpt_path = "origin2.pth"
zl = 'class'

# ===================== 数据增强 =====================
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    RandAugment(num_ops=2, magnitude=9),  # 强化增强
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.25)  # 随机擦除
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# ===================== MIA 计算 =====================
def compute_mia(net, forget_loader, retain_loader, device):
    net.eval()
    forget_confs, retain_confs = [], []
    with torch.no_grad():
        for inputs, _ in forget_loader:
            inputs = inputs.to(device)
            probs = torch.softmax(net(inputs), dim=1)
            confs, _ = torch.max(probs, dim=1)
            forget_confs.extend(confs.cpu().numpy())
        for inputs, _ in retain_loader:
            inputs = inputs.to(device)
            probs = torch.softmax(net(inputs), dim=1)
            confs, _ = torch.max(probs, dim=1)
            retain_confs.extend(confs.cpu().numpy())
    return abs(np.mean(forget_confs) - np.mean(retain_confs))

# ===================== 专家亲和度 =====================
def compute_expert_affinity_per_layer(net, layer_idx, dataloader, device):
    """
    返回每层专家的 softmax 平均值和 top-1 被选中比例
    """
    blk = net.blocks[layer_idx]
    blk.eval()
    num_experts = len(blk.mlp.experts)
    expert_softmax_sum = torch.zeros(num_experts, device=device)
    expert_top1_count = torch.zeros(num_experts, device=device)
    total_tokens = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            x = net.patch_embed(inputs)
            for i, b in enumerate(net.blocks[:layer_idx]):
                x = b(x)

            gate_logits = blk.mlp.gate(x)  # [B, N, num_experts]
            gate_softmax = torch.softmax(gate_logits, dim=-1)
            expert_softmax_sum += gate_softmax.sum(dim=(0, 1))

            # top-1 激活统计
            top1_idx = torch.argmax(gate_logits, dim=-1)  # [B, N]
            for expert_id in range(num_experts):
                expert_top1_count[expert_id] += (top1_idx == expert_id).sum().float()

            total_tokens += x.shape[0] * x.shape[1]

    softmax_avg = (expert_softmax_sum / total_tokens).cpu().numpy()
    top1_ratio = (expert_top1_count / total_tokens).cpu().numpy()
    return softmax_avg, top1_ratio

def compute_anchor_loss(g, target_expert_idx, num_experts):
    """
    g: gating softmax [B*N, num_experts]
    target_expert_idx: 目标专家索引
    """
    a = torch.zeros_like(g, device=g.device)
    a[:, target_expert_idx] = 1.0
    anchor_loss = torch.norm(g - a, p=2, dim=1).mean()
    return anchor_loss
# ===================== 主函数 =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==== 数据集划分 ====
    split_dir = "split_results_" + zl
    forget_file_train = 'forget_indices_' + zl + '_0_train.npy'
    retain_file_train = 'retain_indices_' + zl + '_0_train.npy'
    forget_file_test = 'forget_indices_' + zl + '_0_test.npy'
    retain_file_test = 'retain_indices_' + zl + '_0_test.npy'

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    forget_train = Subset(full_trainset, np.load(os.path.join(split_dir, forget_file_train)))
    retain_train = Subset(full_trainset, np.load(os.path.join(split_dir, retain_file_train)))
    forget_test = Subset(testset, np.load(os.path.join(split_dir, forget_file_test)))
    retain_test = Subset(testset, np.load(os.path.join(split_dir, retain_file_test)))

    forget_train_loader = DataLoader(forget_train, batch_size=128, shuffle=True, num_workers=2)
    retain_train_loader = DataLoader(retain_train, batch_size=128, shuffle=True, num_workers=2)
    forget_test_loader = DataLoader(forget_test, batch_size=128, shuffle=False, num_workers=2)
    retain_test_loader = DataLoader(retain_test, batch_size=128, shuffle=False, num_workers=2)

    # ==== 模型加载 ====
    model = MoEVisionTransformer(
        img_size=32, patch_size=4, embed_dim=128, depth=8, num_heads=4, num_classes=10
    ).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    print(f"Loaded initial model from {ckpt_path}")

    # ==== 评估函数 ====
    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = out.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        return 100. * correct / total

    # ==== 遗忘训练前评估 ====
    print("=== Pre-GA Evaluation ===")
    train_ra = evaluate(retain_train_loader)
    train_fa = evaluate(forget_train_loader)
    test_ra = evaluate(retain_test_loader)
    test_fa = evaluate(forget_test_loader)
    train_ua = 100. - train_fa
    test_ua = 100. - test_fa
    mia = compute_mia(model, forget_train_loader, retain_train_loader, device)
    print(f"Before GA Training | "
          f"Train RA: {train_ra:.2f}% | Train UA: {train_ua:.2f}% | "
          f"Test RA: {test_ra:.2f}% | Test UA: {test_ua:.2f}% | MIA: {mia:.4f}")

    # ===================== 专家归因 =====================
    target_experts_per_layer = {}
    print("=== Expert Affinity per MoE Layer ===")
    for i, blk in enumerate(model.blocks):
        if hasattr(blk.mlp, 'experts'):
            softmax_avg, top1_ratio = compute_expert_affinity_per_layer(model, i, forget_train_loader, device)
            target_idx = int(np.argmax(top1_ratio))  # 用 top1_ratio 选目标专家更合理
            target_experts_per_layer[i] = target_idx
            softmax_str = ", ".join([f"{s:.4f}" for s in softmax_avg])
            top1_str = ", ".join([f"{t:.4f}" for t in top1_ratio])
            print(f"Layer {i} | Softmax Avg: {softmax_str} | Top-1 Ratio: {top1_str} | Target Expert -> {target_idx}")

    # ===================== 冻结非目标专家 =====================
    for i, blk in enumerate(model.blocks):
        if hasattr(blk.mlp, 'experts'):
            tgt = target_experts_per_layer[i]
            for j, expert in enumerate(blk.mlp.experts):
                requires_grad = (j == tgt)
                for p in expert.parameters():
                    p.requires_grad = requires_grad

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ===================== 训练函数 =====================
    def train_one_epoch():
        model.train()
        total_loss = 0
        for (x_r, y_r), (x_f, y_f) in zip(retain_train_loader, forget_train_loader):
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)

            optimizer.zero_grad()
            out_r = model(x_r)
            out_f = model(x_f)

            loss_r = criterion(out_r, y_r)
            loss_f = -criterion(out_f, y_f)  # SEUF-GA 对遗忘集负梯度

            # 计算 anchor loss（所有 MoE 层）
            anchor_losses = []
            for i, blk in enumerate(model.blocks):
                if hasattr(blk.mlp, 'experts'):
                    tgt_idx = target_experts_per_layer[i]
                    gate_logits = blk.mlp.gate(x_f)
                    gate_softmax = torch.softmax(gate_logits, dim=-1)
                    anchor_losses.append(compute_anchor_loss(gate_softmax, tgt_idx, len(blk.mlp.experts)))

            if len(anchor_losses) > 0:
                loss_anchor = torch.stack(anchor_losses).mean()
            else:
                loss_anchor = torch.tensor(0.0, device=device)

            loss = loss_r + loss_f + 0.1 * loss_anchor

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(retain_train_loader), len(forget_train_loader))

    # ===================== 主训练循环 =====================
    start_time = time.time()
    for epoch in range(ga_epoches):
        loss = train_one_epoch()
        train_ra = evaluate(retain_train_loader)
        train_fa = evaluate(forget_train_loader)
        test_ra = evaluate(retain_test_loader)
        test_fa = evaluate(forget_test_loader)
        train_ua = 100. - train_fa
        test_ua = 100. - test_fa
        mia = compute_mia(model, forget_train_loader, retain_train_loader, device)
        print(f"Epoch {epoch + 1}/{ga_epoches} | Loss: {loss:.4f} | "
              f"Train RA: {train_ra:.2f}% | Train UA: {train_ua:.2f}% | "
              f"Test RA: {test_ra:.2f}% | Test UA: {test_ua:.2f}% | MIA: {mia:.4f}")

    rte = (time.time() - start_time) / 60.0
    print(f"=== Final Metrics ===")
    print(f"Train Retain Acc (RA): {train_ra:.2f}%")
    print(f"Train Forget Acc (UA): {train_ua:.2f}%")
    print(f"Test Retain Acc (RA): {test_ra:.2f}%")
    print(f"Test Forget Acc (UA): {test_ua:.2f}%")
    print(f"MIA: {mia:.4f}")
    print(f"RTE: {rte:.2f} minutes")


if __name__ == "__main__":
    main()
