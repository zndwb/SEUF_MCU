import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from try_moevit import MoEVisionTransformer  # 假设内部 MoEMLP 已返回 load_balance_loss
from torchvision.transforms import RandAugment
import random
import numpy as np
import time

# ---------------- reproducibility ----------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 数据增强 ====
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# ==== 数据集：使用原始 fine labels（100类） ====
trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform_test
)

# ==== DataLoader ====
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

# ==== 模型：输出类别数改为 100 ====
num_classes = 100
model = MoEVisionTransformer(
    img_size=32,
    patch_size=4,
    embed_dim=256,
    depth=12,
    num_heads=8,
    num_classes=num_classes,
).to(device)

# print basic diagnostics
def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
print("Device:", device)
print("Total params: {:.2f}M".format(count_params(model)/1e6))

# ==== criterion / optimizer / scheduler ====
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

base_lr = 5e-4
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

epochs = 250
warmup_epochs = 10
alpha = 0.01  # balance loss weight for aux_loss (if model returns it)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

best_acc = 0.0
start_time = time.time()

for epoch in range(epochs):
    # ---- warmup lr ----
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * warmup_factor

    current_lr = optimizer.param_groups[0]['lr']

    # ---- 训练阶段 ----
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs, aux_loss = model(inputs)
        loss = criterion(outputs, labels) # + alpha * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = 100. * correct / total

    # ---- 测试阶段 ----
    model.eval()
    test_total, test_correct, test_loss_agg = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs, aux_loss = model(inputs)
            loss = criterion(outputs, labels) # + alpha * aux_loss
            test_loss_agg += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_loss = test_loss_agg / test_total
    test_acc = 100. * test_correct / test_total

    if epoch >= warmup_epochs:
        scheduler.step()

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} | lr={current_lr:.6f} | "
          f"Train loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
          f"Test loss={test_loss:.4f}, Test Acc={test_acc:.2f}% | elapsed={elapsed/60:.2f}min")

    # ==== 保存最佳模型 ====
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, "moevit_cifar100_best.pth")
        print(f"==> 保存最佳模型，Test Acc: {best_acc:.2f}%")

print(f"训练完成，最佳测试集保留准确率: {best_acc:.2f}%")
