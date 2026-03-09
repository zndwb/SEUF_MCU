import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from sklearn.cluster import KMeans


def load_cifar10():
    """加载 CIFAR-10 训练集和测试集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    return trainset, testset


def split_by_subclass(trainset, testset, class_to_forget=0, num_clusters=5,
                      save_dir="split_results_subclass"):
    """
    使用 KMeans 在 train+test 的 classX 样本上聚类，选最大簇作为遗忘集

    参数:
        trainset: 训练集
        testset: 测试集
        class_to_forget: 要划分的类别 (0-9)
        num_clusters: KMeans 聚类簇数
        save_dir: 保存路径
    """
    os.makedirs(save_dir, exist_ok=True)

    # 取出 train/test 中 classX 的索引
    train_idx = [i for i, (_, label) in enumerate(trainset) if label == class_to_forget]
    test_idx = [i for i, (_, label) in enumerate(testset) if label == class_to_forget]

    print(f"训练集中 class{class_to_forget} 数量: {len(train_idx)}")
    print(f"测试集中 class{class_to_forget} 数量: {len(test_idx)}")

    if len(train_idx) + len(test_idx) == 0:
        print("该类别没有样本！")
        return

    # 提取图像数据（转 numpy）
    X_train = torch.stack([trainset[i][0] for i in train_idx]).numpy()
    X_test = torch.stack([testset[i][0] for i in test_idx]).numpy()

    # 展平处理 (N, C*H*W)
    X_train = X_train.reshape(len(train_idx), -1)
    X_test = X_test.reshape(len(test_idx), -1)

    # 合并
    X_all = np.vstack([X_train, X_test])

    print(f"合并后的 class{class_to_forget} 总样本数: {len(X_all)}")

    # 进行 KMeans 聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_all)

    # 找到最大簇
    unique, counts = np.unique(cluster_labels, return_counts=True)
    largest_cluster = unique[np.argmax(counts)]
    print(f"最大 subclass ID: {largest_cluster}, 样本数: {counts.max()}")

    # 找出最大簇的索引
    all_idx = np.array(train_idx + test_idx)  # 全局索引（先train再test）
    largest_idx = np.where(cluster_labels == largest_cluster)[0]

    # 分回 train/test
    forget_train = [train_idx[i] for i in largest_idx if i < len(train_idx)]
    forget_test = [test_idx[i - len(train_idx)] for i in largest_idx if i >= len(train_idx)]

    retain_train = list(set(range(len(trainset))) - set(forget_train))
    retain_test = list(set(range(len(testset))) - set(forget_test))

    # 保存结果
    np.save(os.path.join(save_dir, "forget_indices_subclass_train.npy"), forget_train)
    np.save(os.path.join(save_dir, "retain_indices_subclass_train.npy"), retain_train)
    np.save(os.path.join(save_dir, "forget_indices_subclass_test.npy"), forget_test)
    np.save(os.path.join(save_dir, "retain_indices_subclass_test.npy"), retain_test)

    # 打印结果
    print("\n=== 划分结果 ===")
    print(f"训练集遗忘集大小: {len(forget_train)}, 保留集大小: {len(retain_train)}")
    print(f"测试集遗忘集大小: {len(forget_test)}, 保留集大小: {len(retain_test)}")
    print(f"结果保存到 {save_dir}")


if __name__ == '__main__':
    import torch

    # 加载 CIFAR-10
    trainset, testset = load_cifar10()

    # 例如对 class0 (飞机) 做 subclass 划分
    split_by_subclass(trainset, testset, class_to_forget=0, num_clusters=3)
