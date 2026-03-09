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
import pickle
import warnings

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

# ==== 尝试加载返回 coarse labels 的 CIFAR100 数据集（优先） ====
use_coarse_target = False
try:
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True,
                                             transform=transform_train, target_type='coarse')
    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True,
                                            transform=transform_test, target_type='coarse')
    use_coarse_target = True
    print("Using torchvision CIFAR100 with target_type='coarse'.")
except TypeError:
    # 旧版 torchvision 不支持 target_type 参数 -> 回退到 manual mapping
    warnings.warn("torchvision CIFAR100 does not support target_type. Falling back to fine->coarse mapping.")
    # 加载原始 fine-label 数据集（带 transform）
    trainset_fine = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    testset_fine = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    # 读取 cifar-100-python/meta 以及 train/test 文件构建 fine -> coarse 映射
    cifar_root = trainset_fine.root
    meta_path = os.path.join(cifar_root, 'cifar-100-python', 'meta')
    train_pickle = os.path.join(cifar_root, 'cifar-100-python', 'train')
    test_pickle = os.path.join(cifar_root, 'cifar-100-python', 'test')

    try:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        # meta contains 'fine_label_names' and 'coarse_label_names'
        with open(train_pickle, 'rb') as f:
            train_dict = pickle.load(f, encoding='latin1')
        with open(test_pickle, 'rb') as f:
            test_dict = pickle.load(f, encoding='latin1')

        fine_to_coarse = {}
        for fin, coar in zip(train_dict['fine_labels'], train_dict['coarse_labels']):
            fine_to_coarse[fin] = coar
        for fin, coar in zip(test_dict['fine_labels'], test_dict['coarse_labels']):
            fine_to_coarse.setdefault(fin, coar)

        # 确保映射覆盖 0..99
        if len(fine_to_coarse) != 100:
            warnings.warn(f"Expected mapping for 100 fine labels but got {len(fine_to_coarse)} entries.")

        class CIFAR100_Coarse(torchvision.datasets.CIFAR100):
            def __init__(self, *args, fine2coarse_map=None, **kwargs):
                super().__init__(*args, **kwargs)
                self._fine2coarse = fine2coarse_map

            def __getitem__(self, index):
                img, fine_target = super().__getitem__(index)  # this returns (img, fine_label)
                coarse_target = self._fine2coarse[fine_target]
                return img, coarse_target

        trainset = CIFAR100_Coarse(root=trainset_fine.root, train=True, download=False,
                                   transform=transform_train, fine2coarse_map=fine_to_coarse)
        testset = CIFAR100_Coarse(root=testset_fine.root, train=False, download=False,
                                  transform=transform_test, fine2coarse_map=fine_to_coarse)
        use_coarse_target = True
        print("Using manual fine->coarse mapping for CIFAR100 (20 super classes).")
    except Exception as e:
        raise RuntimeError("Failed to construct fine->coarse mapping. Ensure 'cifar-100-python' files exist under ./data/") from e

# ==== 注意：你的 split 索引（Subset）是基于样本索引，不受 fine/coarse label 表示影响 ==== #
zl = "super"  # 保留原先变量，若需要可改为 'super'（仅用于文件命名）
split_dir = 'split_results_cifar100_' + zl
forget_file_train = 'forget_indices_' + zl + '_train.npy'
retain_file_train = 'retain_indices_' + zl + '_train.npy'
forget_file_test = 'forget_indices_' + zl + '_test.npy'
retain_file_test = 'retain_indices_' + zl + '_test.npy'

# ==== 加载划分索引 ==== #
forget_train_idx = np.load(os.path.join(split_dir, forget_file_train))
retain_train_idx = np.load(os.path.join(split_dir, retain_file_train))
forget_test_idx = np.load(os.path.join(split_dir, forget_file_test))
retain_test_idx = np.load(os.path.join(split_dir, retain_file_test))

# ==== 创建子集（Subsets 用样本索引，与标签空间无关） ==== #
forget_train_set = Subset(trainset, forget_train_idx)
retain_train_set = Subset(trainset, retain_train_idx)
forget_test_set = Subset(testset, forget_test_idx)
retain_test_set = Subset(testset, retain_test_idx)

trainloader = DataLoader(retain_train_set, batch_size=128, shuffle=True, num_workers=2)
retain_train_loader = DataLoader(retain_train_set, batch_size=128, shuffle=False, num_workers=2)
forget_train_loader = DataLoader(forget_train_set, batch_size=128, shuffle=False, num_workers=2)
retain_test_loader = DataLoader(retain_test_set, batch_size=128, shuffle=False, num_workers=2)
forget_test_loader = DataLoader(forget_test_set, batch_size=128, shuffle=False, num_workers=2)

# ==== 模型：输出类别数改为 20（super classes） ==== #
model = MoEVisionTransformer(
    img_size=32,
    patch_size=4,
    embed_dim=128,
    depth=8,
    num_heads=4,
    num_classes=20,   # <- 改为 20
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

epochs = 250
best_acc = 0.0  # 保存测试集保留部分最好的模型
start_time = time.time()

def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs,_ = model(inputs)
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
            p,_=model(inputs)
            probs = torch.softmax(p, dim=1)
            confs, _ = torch.max(probs, dim=1)
            forget_confs.extend(confs.cpu().numpy())
        for inputs, labels in retain_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            p2,_=model(inputs)
            probs = torch.softmax(p2, dim=1)
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
        outputs,_ = model(inputs)
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

    # 保存测试集保留准确率最好的模型（改名以示是 super-label 版本）
    if tar > best_acc:
        best_acc = tar
        torch.save(model.state_dict(), "moevit_cifar100_unlearn_best_super.pth")

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}, "
          f"UA: {ua:.2f}%, RA: {ra:.2f}%, TUA: {tau:.2f}%, TRA: {tar:.2f}%, MIA: {mia:.4f}")

end_time = time.time()
rte = (end_time - start_time)/60.0
print(f"\n训练完成，最佳测试集保留准确率: {best_acc:.2f}%")
print(f"运行时间 RTE: {rte:.2f} 分钟")
