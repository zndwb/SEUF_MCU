import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import time
from tqdm import tqdm
from try_moevit import MoEVisionTransformer
from torchvision.transforms import RandAugment
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 数据增强 ====
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

# ==== 加载完整数据集 ====
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

zl="class"

split_dir = 'split_results_'+zl
forget_file_train = 'forget_indices_'+zl+'_0_train.npy'
retain_file_train = 'retain_indices_'+zl+'_0_train.npy'
forget_file_test = 'forget_indices_'+zl+'_0_test.npy'
retain_file_test = 'retain_indices_'+zl+'_0_test.npy'

# ==== 加载划分索引 ====
forget_train_idx = np.load(os.path.join(split_dir, forget_file_train))
retain_train_idx = np.load(os.path.join(split_dir, retain_file_train))
forget_test_idx = np.load(os.path.join(split_dir, forget_file_test))
retain_test_idx = np.load(os.path.join(split_dir, retain_file_test))

# ==== 创建子集 ====
forget_train_set = Subset(trainset, forget_train_idx)
retain_train_set = Subset(trainset, retain_train_idx)
forget_test_set = Subset(testset, forget_test_idx)
retain_test_set = Subset(testset, retain_test_idx)

trainloader = DataLoader(retain_train_set, batch_size=128, shuffle=True, num_workers=2)
retain_train_loader = DataLoader(retain_train_set, batch_size=128, shuffle=False, num_workers=2)
forget_train_loader = DataLoader(forget_train_set, batch_size=128, shuffle=False, num_workers=2)
retain_test_loader = DataLoader(retain_test_set, batch_size=128, shuffle=False, num_workers=2)
forget_test_loader = DataLoader(forget_test_set, batch_size=128, shuffle=False, num_workers=2)

# ==== 模型 ====
model = MoEVisionTransformer(
    img_size=32,
    patch_size=4,
    embed_dim=128,
    depth=8,
    num_heads=4,
    num_classes=10
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

epochs = 200
best_acc = 0.0  # 保存测试集保留部分最好的模型
start_time = time.time()

def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def compute_mia(forget_loader, retain_loader):
    model.eval()
    forget_confs, retain_confs = [], []
    with torch.no_grad():
        for inputs, labels in forget_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            probs = torch.softmax(model(inputs), dim=1)
            confs, _ = torch.max(probs, dim=1)
            forget_confs.extend(confs.cpu().numpy())
        for inputs, labels in retain_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            probs = torch.softmax(model(inputs), dim=1)
            confs, _ = torch.max(probs, dim=1)
            retain_confs.extend(confs.cpu().numpy())
    return abs(np.mean(forget_confs) - np.mean(retain_confs))

for epoch in range(epochs):
    # ---- 训练阶段 ----
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_acc = 100. * correct / total

    # ---- 评估阶段 ----
    ua = 100.0 - evaluate(forget_train_loader)   # 训练集遗忘准确率
    ra = evaluate(retain_train_loader)           # 训练集保留准确率
    tau = 100.0 - evaluate(forget_test_loader)   # 测试集遗忘准确率
    tar = evaluate(retain_test_loader)           # 测试集保留准确率
    mia = compute_mia(forget_train_loader, retain_train_loader)

    # 保存测试集保留准确率最好的模型
    if tar > best_acc:
        best_acc = tar
        torch.save(model.state_dict(), "moevit_cifar10_unlearn_best_"+zl+".pth")

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}, "
          f"UA: {ua:.2f}%, RA: {ra:.2f}%, TUA: {tau:.2f}%, TRA: {tar:.2f}%, MIA: {mia:.4f}")

end_time = time.time()
rte = (end_time - start_time)/60.0
print(f"\n训练完成，最佳测试集保留准确率: {best_acc:.2f}%")
print(f"运行时间 RTE: {rte:.2f} 分钟")
