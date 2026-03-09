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
from try_moevit import MoEVisionTransformer  # **请确保该模型支持 forward(x, return_gate=True) -> (logits, gates_list)**

# ----------------------------
# Hyperparams
# ----------------------------
ga_epoches = 10
lr = 5e-5
ckpt_path = "origin2.pth"
zl = 'class'
anchor_lambda = 0.1  # anchor loss 权重（超参，可调）

# ===================== 数据增强 =====================
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    RandAugment(num_ops=2, magnitude=9),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.25)
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# ===================== MIA 计算 =====================
def compute_mia(net, forget_loader, retain_loader, device):
    """使用 logits 计算 MIA 差异（使用 model 的 logits，适配 model 返回 (logits, gates) 或单 logits）"""
    net.eval()
    forget_confs, retain_confs = [], []
    with torch.no_grad():
        for inputs, _ in forget_loader:
            inputs = inputs.to(device)
            out = net(inputs)  # 可能返回 logits 或 (logits, gates)
            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            probs = torch.softmax(logits, dim=1)
            confs, _ = torch.max(probs, dim=1)
            forget_confs.extend(confs.cpu().numpy())
        for inputs, _ in retain_loader:
            inputs = inputs.to(device)
            out = net(inputs)
            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            probs = torch.softmax(logits, dim=1)
            confs, _ = torch.max(probs, dim=1)
            retain_confs.extend(confs.cpu().numpy())
    return abs(np.mean(forget_confs) - np.mean(retain_confs))


# ===================== 专家亲和度 =====================
def compute_expert_affinity_per_layer(net, layer_idx, dataloader, device):
    """
    通过 model(inputs, return_gate=True) 累加该 MoE 层的 gate softmax 和 top1 counts
    返回 (softmax_avg, top1_ratio)
    注意：layer_idx 表示 model.blocks 中的索引（例如 1,3,5...），函数会映射到 gates_list 中的位置。
    """
    net.eval()
    # 先找出所有带 MoE 的层索引（blocks 中 use_moe=True 的索引）
    moelayer_indices = [i for i, blk in enumerate(net.blocks) if hasattr(blk, 'use_moe') and blk.use_moe]
    if layer_idx not in moelayer_indices:
        raise ValueError(f"layer_idx {layer_idx} is not a MoE layer. MoE layers: {moelayer_indices}")
    gate_pos = moelayer_indices.index(layer_idx)  # 在 gates_list 中的位置

    # 获取 num_experts from that block
    blk = net.blocks[layer_idx]
    num_experts = len(blk.mlp.experts)

    expert_softmax_sum = torch.zeros(num_experts, device=device)
    expert_top1_count = torch.zeros(num_experts, device=device)
    total_tokens = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            # **必须用 return_gate=True，从模型一次性得到 gates**
            out = net(inputs, return_gate=True)
            if not (isinstance(out, (tuple, list)) and len(out) == 2):
                raise RuntimeError("Model.forward must support return_gate=True and return (logits, gates_list).")
            _, gates_list = out
            # gates_list is in order of MoE blocks; select gate for our target layer
            gate = gates_list[gate_pos]  # [B, N, E]  (we assume model returns softmax already)
            # some implementations may return logits; safe approach: treat as softmax if sums ~1
            if not torch.allclose(gate.sum(dim=-1).mean(), torch.tensor(1.0, device=device), atol=1e-3):
                # try softmax
                gate_softmax = torch.softmax(gate, dim=-1)
            else:
                gate_softmax = gate

            expert_softmax_sum += gate_softmax.sum(dim=(0, 1))  # sum over B and N
            top1_idx = torch.argmax(gate_softmax, dim=-1)  # [B, N]
            for expert_id in range(num_experts):
                expert_top1_count[expert_id] += (top1_idx == expert_id).sum().float()

            total_tokens += gate_softmax.shape[0] * gate_softmax.shape[1]

    softmax_avg = (expert_softmax_sum / total_tokens).cpu().numpy()
    top1_ratio = (expert_top1_count / total_tokens).cpu().numpy()
    return softmax_avg, top1_ratio


# ===================== Anchor Loss =====================
def compute_anchor_loss(g_flat, target_expert_idx):
    """
    g_flat: [M, E] (M = B*N flattened), 每行是 token 在所有 experts 上的 softmax 概率
    target_expert_idx: int
    返回 L2 mean over M samples
    """
    # g_flat should be float tensor [M, E]
    assert g_flat.ndim == 2, "g_flat must be 2D [M, E]"
    M, E = g_flat.shape
    a = torch.zeros_like(g_flat, device=g_flat.device)
    a[:, target_expert_idx] = 1.0
    # L2 per token then mean
    anchor_loss = torch.norm(g_flat - a, p=2, dim=1).mean()
    return anchor_loss


# ===================== Main =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data split files
    split_dir = "split_results_" + zl
    forget_file_train = 'forget_indices_' + zl + '_0_train.npy'
    retain_file_train = 'retain_indices_' + zl + '_0_train.npy'
    forget_file_test = 'forget_indices_' + zl + '_0_test.npy'
    retain_file_test = 'retain_indices_' + zl + '_0_test.npy'

    # datasets
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

    # load model (must support return_gate=True)
    model = MoEVisionTransformer(
        img_size=32, patch_size=4, embed_dim=128, depth=8, num_heads=4, num_classes=10
    ).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    # load state dict; the checkpoint might be a dict or pure state_dict
    ck = torch.load(ckpt_path, map_location=device)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        model.load_state_dict(ck['model_state_dict'])
    else:
        model.load_state_dict(ck)
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    # evaluate helper (accommodate model returning logits or (logits,gates))
    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)  # might be logits or (logits, gates)
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                else:
                    logits = out
                _, pred = logits.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        return 100. * correct / total

    # ==== Pre-GA evaluation ====
    print("=== Pre-GA Evaluation ===")
    train_ra = evaluate(retain_train_loader)
    train_fa = evaluate(forget_train_loader)
    test_ra = evaluate(retain_test_loader)
    test_fa = evaluate(forget_test_loader)
    train_ua = 100. - train_fa
    test_ua = 100. - test_fa
    mia = compute_mia(model, forget_train_loader, retain_train_loader, device)
    print(f"Before GA Training | Train RA: {train_ra:.2f}% | Train UA: {train_ua:.2f}% | "
          f"Test RA: {test_ra:.2f}% | Test UA: {test_ua:.2f}% | MIA: {mia:.4f}")

    # ===================== Expert affinity (per MoE layer) =====================
    target_experts_per_layer = {}
    print("=== Expert Affinity per MoE Layer ===")
    # we iterate blocks and test the ones that are MoE
    for i, blk in enumerate(model.blocks):
        if hasattr(blk, 'use_moe') and blk.use_moe:
            softmax_avg, top1_ratio = compute_expert_affinity_per_layer(model, i, forget_train_loader, device)
            target_idx = int(np.argmax(top1_ratio))
            target_experts_per_layer[i] = target_idx
            softmax_str = ", ".join([f"{s:.4f}" for s in softmax_avg])
            top1_str = ", ".join([f"{t:.4f}" for t in top1_ratio])
            print(f"Layer {i} | Softmax Avg: {softmax_str} | Top-1 Ratio: {top1_str} | Target Expert -> {target_idx}")

    # freeze non-target experts (only allow selected experts' params to require_grad)
    for i, blk in enumerate(model.blocks):
        if hasattr(blk, 'use_moe') and blk.use_moe:
            tgt = target_experts_per_layer.get(i, None)
            for j, expert in enumerate(blk.mlp.experts):
                requires_grad = (j == tgt)
                for p in expert.parameters():
                    p.requires_grad = requires_grad
            # keep gate params trainable (choice: you can also freeze gates if desired)
            for p in blk.mlp.gate.parameters():
                p.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # training step: use model(x, return_gate=True) to get gates for anchor loss
    def train_one_epoch():
        model.train()
        total_loss = 0.0
        # simple pairing (zip) — if you want to iterate over both full loaders, consider cycling shorter one.
        for (x_r, y_r), (x_f, y_f) in zip(retain_train_loader, forget_train_loader):
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)

            optimizer.zero_grad()
            # **get logits and gates from model at once**
            out_r = model(x_r)  # maybe logits or (logits, gates)
            if isinstance(out_r, (tuple, list)):
                logits_r, gates_r = out_r
            else:
                # model didn't return gates on this call; attempt return_gate=True
                try:
                    logits_r, gates_r = model(x_r, return_gate=True)
                except TypeError:
                    raise RuntimeError("Model must support `return_gate=True` for anchor loss computation.")

            out_f = model(x_f)
            if isinstance(out_f, (tuple, list)):
                logits_f, gates_f = out_f
            else:
                try:
                    logits_f, gates_f = model(x_f, return_gate=True)
                except TypeError:
                    raise RuntimeError("Model must support `return_gate=True` for anchor loss computation.")

            # classification losses
            loss_r = criterion(logits_r, y_r)
            loss_f = -criterion(logits_f, y_f)  # gradient ascent on forget set

            # compute anchor losses for each MoE layer using gates_f (list)
            anchor_losses = []
            # build list of MoE layer indices to map to gates_f
            moelayer_indices = [ii for ii, bb in enumerate(model.blocks) if hasattr(bb, 'use_moe') and bb.use_moe]
            # gates_f is list in same order
            for gate_pos, layer_idx in enumerate(moelayer_indices):
                tgt_idx = target_experts_per_layer.get(layer_idx, None)
                if tgt_idx is None:
                    continue
                g = gates_f[gate_pos]  # [B, N, E] as returned by model
                # flatten to [M, E]
                g_flat = g.reshape(-1, g.shape[-1])
                anchor_losses.append(compute_anchor_loss(g_flat, tgt_idx))
            if len(anchor_losses) > 0:
                loss_anchor = torch.stack(anchor_losses).mean()
            else:
                loss_anchor = torch.tensor(0.0, device=device)

            loss = loss_r + loss_f + anchor_lambda * loss_anchor
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(retain_train_loader), len(forget_train_loader))

    # ===================== main GA loop =====================
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
    print("=== Final Metrics ===")
    print(f"Train Retain Acc (RA): {train_ra:.2f}%")
    print(f"Train Forget Acc (UA): {train_ua:.2f}%")
    print(f"Test Retain Acc (RA): {test_ra:.2f}%")
    print(f"Test Forget Acc (UA): {test_fa:.2f}%")
    print(f"MIA: {mia:.4f}")
    print(f"RTE: {rte:.2f} minutes")


if __name__ == "__main__":
    main()
