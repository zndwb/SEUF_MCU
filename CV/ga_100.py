import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import os
import pickle
from try_moevit import MoEVisionTransformer

# ----------------------------
# Hyperparams
# ----------------------------
ga_epochs = 10
lr = 1e-4
ckpt_path = "moevit_cifar100_super_best.pth"
zl = 'fine'   # 数据划分模式，对应你的 split 目录

# ===================== 数据增强 =====================
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# ===================== MIA 计算 =====================
def compute_mia(net, forget_loader, retain_loader, device):
    net.eval()
    forget_confs, retain_confs = [], []
    with torch.no_grad():
        for inputs, _ in forget_loader:
            inputs = inputs.to(device)
            out, _ = net(inputs)
            probs = torch.softmax(out, dim=1)
            confs, _ = torch.max(probs, dim=1)
            forget_confs.extend(confs.cpu().numpy())
        for inputs, _ in retain_loader:
            inputs = inputs.to(device)
            out, _ = net(inputs)
            probs = torch.softmax(out, dim=1)
            confs, _ = torch.max(probs, dim=1)
            retain_confs.extend(confs.cpu().numpy())
    return abs(np.mean(forget_confs) - np.mean(retain_confs))

# ===================== 主函数 =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==== 数据划分加载 ====
    split_dir = "split_results_cifar100_" + zl
    forget_file_train = 'forget_indices_' + zl + '_train.npy'
    retain_file_train = 'retain_indices_' + zl + '_train.npy'
    forget_file_test = 'forget_indices_' + zl + '_test.npy'
    retain_file_test = 'retain_indices_' + zl + '_test.npy'

    # ==== 加载 CIFAR100 coarse labels ====
    try:
        full_trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True,
                                                      transform=transform_train, target_type='coarse')
        testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True,
                                                transform=transform_test, target_type='coarse')
    except TypeError:
        # fallback to fine->coarse 手动映射
        with open('./data/cifar-100-python/meta', 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        with open('./data/cifar-100-python/train', 'rb') as f:
            train_dict = pickle.load(f, encoding='latin1')
        fine2coarse = {fin: coar for fin, coar in zip(train_dict['fine_labels'], train_dict['coarse_labels'])}

        class CIFAR100_Coarse(torchvision.datasets.CIFAR100):
            def __init__(self, *args, fine2coarse_map=None, **kwargs):
                super().__init__(*args, **kwargs)
                self._fine2coarse = fine2coarse_map
            def __getitem__(self, index):
                img, fine_target = super().__getitem__(index)
                return img, self._fine2coarse[fine_target]

        full_trainset = CIFAR100_Coarse(root="./data", train=True, transform=transform_train,
                                        fine2coarse_map=fine2coarse)
        testset = CIFAR100_Coarse(root="./data", train=False, transform=transform_test,
                                  fine2coarse_map=fine2coarse)

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
        img_size=32, patch_size=4, embed_dim=128, depth=8,
        num_heads=4, num_classes=20
    ).to(device)

    ck = torch.load(ckpt_path, map_location=device)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        model.load_state_dict(ck['model_state_dict'])
    else:
        model.load_state_dict(ck)
    model.eval()
    print(f"Loaded pretrained model from {ckpt_path}")

    # ==== 评估函数 ====
    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out, _ = model(x)
                _, pred = out.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        return 100. * correct / total

    # ==== 预评估 ====
    print("=== Pre-GA Evaluation ===")
    train_ra = evaluate(retain_train_loader)
    train_fa = evaluate(forget_train_loader)
    test_ra = evaluate(retain_test_loader)
    test_fa = evaluate(forget_test_loader)
    print(f"Before GA | Train RA: {train_ra:.2f}% | Train UA: {100-train_fa:.2f}% | "
          f"Test RA: {test_ra:.2f}% | Test UA: {100-test_fa:.2f}% | MIA: {compute_mia(model, forget_train_loader, retain_train_loader, device):.4f}")

    # ==== 优化器 ====
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ==== 训练 ====
    def train_one_epoch():
        model.train()
        total_loss = 0.0
        for (x_r, y_r), (x_f, y_f) in zip(retain_train_loader, forget_train_loader):
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)

            optimizer.zero_grad()

            # 对保留集执行正常反向传播
            out_r, _ = model(x_r)
            loss_r = criterion(out_r, y_r)

            # 对遗忘集执行梯度上升（负损失）
            out_f, _ = model(x_f)
            loss_f = -criterion(out_f, y_f)

            loss = loss_r + loss_f
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(retain_train_loader), len(forget_train_loader))

    # ==== 主训练循环 ====
    start_time = time.time()
    for epoch in range(ga_epochs):
        loss = train_one_epoch()
        train_ra = evaluate(retain_train_loader)
        train_fa = evaluate(forget_train_loader)
        test_ra = evaluate(retain_test_loader)
        test_fa = evaluate(forget_test_loader)
        mia = compute_mia(model, forget_train_loader, retain_train_loader, device)
        print(f"Epoch {epoch+1}/{ga_epochs} | Loss: {loss:.4f} | "
              f"Train RA: {train_ra:.2f}% | Train UA: {100-train_fa:.2f}% | "
              f"Test RA: {test_ra:.2f}% | Test UA: {100-test_fa:.2f}% | MIA: {mia:.4f}")

    print(f"Training finished in {(time.time()-start_time)/60:.2f} min")

    # ==== 保存模型 ====
    torch.save(model.state_dict(), f"moevit_cifar100_GAforget_{zl}.pth")
    print("✅ Model saved.")


if __name__ == "__main__":
    main()
