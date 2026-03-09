import torchvision
import torchvision.transforms as transforms
import os
import numpy as np


def load_cifar10():
    """加载 CIFAR-10 训练集和测试集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    return trainset, testset


def split_by_class(dataset, class_to_forget, save_dir, split_name="train"):
    """
    按类别划分：遗忘指定类别，保留其他所有类别

    参数:
        dataset: CIFAR-10 数据集（训练集或测试集）
        class_to_forget: 要遗忘的类别索引 (0-9)
        save_dir: 保存结果的目录
        split_name: "train" 或 "test" 用于区分保存
    """
    os.makedirs(save_dir, exist_ok=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"[{split_name}] 按类别划分：遗忘类别 - {classes[class_to_forget]} (索引 {class_to_forget})")

    forget_indices, retain_indices = [], []
    for i, (_, label) in enumerate(dataset):
        if label == class_to_forget:
            forget_indices.append(i)
        else:
            retain_indices.append(i)

    # 保存索引
    np.save(os.path.join(save_dir, f'forget_indices_class_{class_to_forget}_{split_name}.npy'), forget_indices)
    np.save(os.path.join(save_dir, f'retain_indices_class_{class_to_forget}_{split_name}.npy'), retain_indices)

    print(f"[{split_name}] 划分完成：遗忘集大小 {len(forget_indices)}, 保留集大小 {len(retain_indices)}")
    print(f"结果保存在 {save_dir} 目录下\n")

    return forget_indices, retain_indices


if __name__ == '__main__':
    # 加载 CIFAR-10 训练集和测试集
    trainset, testset = load_cifar10()
    print(f"成功加载 CIFAR-10：训练集 {len(trainset)} 样本, 测试集 {len(testset)} 样本")

    # 例如遗忘类别 0: 飞机
    class_to_forget = 0
    save_dir = "split_results_class"

    # 划分训练集
    split_by_class(trainset, class_to_forget, save_dir, split_name="train")
    # 划分测试集
    split_by_class(testset, class_to_forget, save_dir, split_name="test")
