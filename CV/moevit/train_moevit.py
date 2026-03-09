import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from moevit import MoEVisionTransformer  # 引入模型

# ===============================
# 1. 基础配置
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
num_workers = 4
num_epochs = 100
learning_rate = 3e-4
num_classes = 100

# ===============================
# 2. CIFAR-100 数据增强与加载
# ===============================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

train_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ===============================
# 3. 初始化模型
# ===============================
model = MoEVisionTransformer(
    img_size=32,
    patch_size=4,
    in_chans=3,
    num_classes=num_classes,
    embed_dim=192,
    depth=8,
    num_heads=3,
    num_experts=4,
).to(device)

# ===============================
# 4. 优化器与损失函数
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ===============================
# 5. 训练与测试函数
# ===============================
def train_one_epoch(epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f"Epoch [{epoch}] Train Loss: {total_loss/total:.4f}, Acc: {acc:.2f}%")

def test(epoch):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"Epoch [{epoch}] Test Loss: {total_loss/total:.4f}, Acc: {acc:.2f}%")
    return acc

# ===============================
# 6. 主训练循环
# ===============================
best_acc = 0
for epoch in range(1, num_epochs + 1):
    train_one_epoch(epoch)
    acc = test(epoch)
    scheduler.step()

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "moevit_best.pth")
        print(f"✅ Saved new best model at epoch {epoch} | Acc: {best_acc:.2f}%")

print(f"\n🎯 Training Finished! Best Test Accuracy: {best_acc:.2f}%")
