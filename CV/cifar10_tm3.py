import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import RandAugment
from try_moevit import MoEVisionTransformer

# ======================
# 环境与设备配置
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ======================
# 数据增强与预处理
# ======================
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    RandAugment(num_ops=2, magnitude=9),          # 强化数据增强
    transforms.RandomHorizontalFlip(),            # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.25)              # 随机擦除
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# ======================
# 数据集加载
# ======================
trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# ======================
# 模型定义
# ======================
model = MoEVisionTransformer(
    img_size=32,
    patch_size=4,
    embed_dim=128,
    depth=8,
    num_heads=4,
    num_classes=100,       # CIFAR-10 分类数
).to(device)

# ======================
# 损失函数与优化器
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

# ======================
# 训练配置
# ======================
epochs = 200
best_acc = 0.0

# ======================
# 训练与测试流程
# ======================
for epoch in range(epochs):
    # ---- 训练阶段 ----
    model.train()
    running_loss, total, correct = 0.0, 0, 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs,_ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total

    # ---- 测试阶段 ----
    model.eval()
    test_total, test_correct = 0, 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs,_ = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total

    # ---- 输出结果 ----
    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Loss: {running_loss/len(trainloader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # ---- 保存最佳模型 ----
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "moevit_cifar10_best.pth")
        print(f"==> 保存最佳模型，Test Acc: {best_acc:.2f}%")

print(f"训练完成，最佳测试集准确率: {best_acc:.2f}%")
