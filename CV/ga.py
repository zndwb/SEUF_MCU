import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
import pickle
import warnings

def compute_mia(net, forget_loader, retain_loader, device):
    """
    计算MIA指标：成员推理攻击成功率差异
    返回：abs(平均置信度差)，越接近0越好
    """
    net.eval()
    forget_confs, retain_confs = [], []
    with torch.no_grad():
        for inputs, _ in forget_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, _ = torch.max(probs, dim=1)
            forget_confs.extend(confs.cpu().numpy())

        for inputs, _ in retain_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, _ = torch.max(probs, dim=1)
            retain_confs.extend(confs.cpu().numpy())

    return abs(np.mean(forget_confs) - np.mean(retain_confs))


def main():
    parser = argparse.ArgumentParser(description='Unlearning with Basic GA Algorithm')
    parser.add_argument('--model_path', default='checkpoint', help='Path to saved model checkpoint')
    parser.add_argument('--split_dir', default='fine', help='path to forget/retain indices')
    parser.add_argument('--output_dir', default='GA_unlearned_models', help='Directory to save unlearned model')
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate for GA')
    parser.add_argument('--epochs', default=5, type=int, help='Number of unlearning epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lambda_retain', default=1.0, type=float, help='Weight for retain loss')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),  # 强化随机变换
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))  # 保留 RandomErasing
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
        full_trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True,
                                                      transform=transform_train, target_type='coarse')
        testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True,
                                                transform=transform_test, target_type='coarse')
        use_coarse_target = True
        print("Using torchvision CIFAR100 with target_type='coarse'.")
    except TypeError:
        # 旧版 torchvision 不支持 target_type 参数 -> 回退到 manual mapping
        warnings.warn("torchvision CIFAR100 does not support target_type. Falling back to fine->coarse mapping.")
        # 加载原始 fine-label 数据集（带 transform）
        trainset_fine = torchvision.datasets.CIFAR100(root="./data", train=True, download=True,
                                                      transform=transform_train)
        testset_fine = torchvision.datasets.CIFAR100(root="./data", train=False, download=True,
                                                     transform=transform_test)

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

            full_trainset = CIFAR100_Coarse(root=trainset_fine.root, train=True, download=False,
                                            transform=transform_train, fine2coarse_map=fine_to_coarse)
            testset = CIFAR100_Coarse(root=testset_fine.root, train=False, download=False,
                                      transform=transform_test, fine2coarse_map=fine_to_coarse)
            use_coarse_target = True
            print("Using manual fine->coarse mapping for CIFAR100 (20 super classes).")
        except Exception as e:
            raise RuntimeError(
                "Failed to construct fine->coarse mapping. Ensure 'cifar-100-python' files exist under ./data/") from e

    # 加载划分索引
    forget_indices = np.load(
        os.path.join('split_results_' + args.split_dir, 'forget_indices_' + args.split_dir + '_0.npy'))
    retain_indices = np.load(
        os.path.join('split_results_' + args.split_dir, 'retain_indices_' + args.split_dir + '_0.npy'))

    # 创建DataLoader
    forgetset = Subset(full_trainset, forget_indices)
    retainset = Subset(full_trainset, retain_indices)

    forget_loader = torch.utils.data.DataLoader(
        forgetset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    retain_loader = torch.utils.data.DataLoader(
        retainset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # 加载模型
    from models.moe import MOE_ViT
    init_ckpt_path = os.path.join(args.model_path, 'moe_vit_cifar10_best.pth')
    checkpoint = torch.load(init_ckpt_path)

    net = MOE_ViT(num_experts=4, size=32).to(device)
    for expert in net.experts:
        expert.vit.head = nn.Linear(expert.vit.head.in_features, 10).to(device)

    net.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path} (epoch {checkpoint['epoch']})")

    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 评估函数
    def evaluate(loader, desc="Evaluating"):
        net.eval()
        total, correct = 0, 0
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pbar.set_postfix({'acc': f"{100. * correct / total:.2f}%"})
        return 100. * correct / total

    # 计算所有指标
    def compute_metrics():
        fa = evaluate(forget_loader, "Forget Set")
        ra = evaluate(retain_loader, "Retain Set")
        ta = evaluate(test_loader, "Test Set")
        ua = 100.0 - fa
        mia = compute_mia(net, forget_loader, retain_loader, device)
        return ua, ra, ta, fa, mia

    # 初始指标
    print("\n=== Initial Metrics ===")
    fa_init = 2.24
    ra_init = 95.98
    ta_init = 81.81
    ua_init = 100.0 - fa_init
    mia_init = 0.0137
    print(f"UA: {ua_init:.2f}% | RA: {ra_init:.2f}% | TA: {ta_init:.2f}% | MIA: {mia_init:.4f}")

    # GA遗忘主循环
    print("\n=== Starting GA Unlearning ===")
    start_time = time.time()
    best_ua = ua_init
    best_ta = ta_init

    for epoch in range(args.epochs):
        net.train()
        total_loss = 0.0

        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        max_batches = max(len(forget_loader), len(retain_loader))

        pbar = tqdm(range(max_batches), desc=f"Epoch {epoch + 1}/{args.epochs}")
        for _ in pbar:
            optimizer.zero_grad()

            # 遗忘集：梯度上升
            try:
                inputs_f, targets_f = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                inputs_f, targets_f = next(forget_iter)
            inputs_f, targets_f = inputs_f.to(device), targets_f.to(device)
            outputs_f = net(inputs_f)
            loss_forget = criterion(outputs_f, targets_f)

            # 保留集：梯度下降
            try:
                inputs_r, targets_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                inputs_r, targets_r = next(retain_iter)
            inputs_r, targets_r = inputs_r.to(device), targets_r.to(device)
            outputs_r = net(inputs_r)
            loss_retain = criterion(outputs_r, targets_r)

            # 总损失
            total_batch_loss = (-loss_forget) + args.lambda_retain * loss_retain
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            pbar.set_postfix({'avg_loss': f"{total_loss / (_ + 1):.4f}"})

        # 每轮指标
        ua, ra, ta, fa, mia = compute_metrics()

        # 保存最佳模型
        if ua > best_ua and ta > best_ta * 0.9:
            best_ua, best_ta = ua, ta
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {'UA': ua, 'RA': ra, 'TA': ta, 'MIA': mia}
            }, os.path.join(args.output_dir, 'unlearned_best.pth'))

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"UA: {ua:.2f}% (Best: {best_ua:.2f}%) | RA: {ra:.2f}% | TA: {ta:.2f}% | MIA: {mia:.4f}")

    # 最终指标
    rte = (time.time() - start_time) / 60.0
    print("\n=== Final Unlearning Metrics ===")
    final_ua, final_ra, final_ta, final_fa, final_mia = compute_metrics()
    print(f"UA: {final_ua:.2f}% (↑{final_ua - ua_init:.2f}%)")
    print(f"RA: {final_ra:.2f}% (↓{ra_init - final_ra:.2f}%)")
    print(f"TA: {final_ta:.2f}% (↓{ta_init - final_ta:.2f}%)")
    print(f"MIA: {final_mia:.4f}")
    print(f"RTE: {rte:.2f} minutes")


if __name__ == '__main__':
    main()