import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

# CIFAR-100 大类划分
CIFAR100_SUPERCLASS = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple', 'oak', 'palm', 'pine', 'willow'],
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}


def load_cifar100():
    """加载 CIFAR-100 数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    return trainset, testset


def split_by_superclass(dataset, superclass_name, save_dir, split_name="train"):
    """
    按大类划分：遗忘指定大类（superclass），保留其他类别
    """
    os.makedirs(save_dir, exist_ok=True)

    fine_to_idx = dataset.class_to_idx
    forget_class_names = CIFAR100_SUPERCLASS[superclass_name]
    forget_class_indices = [fine_to_idx[name] for name in forget_class_names]

    print(f"[{split_name}] 遗忘大类：{superclass_name} -> {forget_class_names}")

    forget_indices, retain_indices = [], []
    for i, (_, label) in enumerate(dataset):
        if label in forget_class_indices:
            forget_indices.append(i)
        else:
            retain_indices.append(i)

    # 保存索引
    np.save(os.path.join(save_dir, f'forget_indices_{superclass_name}_{split_name}.npy'), forget_indices)
    np.save(os.path.join(save_dir, f'retain_indices_{superclass_name}_{split_name}.npy'), retain_indices)

    print(f"[{split_name}] 划分完成：遗忘集 {len(forget_indices)}，保留集 {len(retain_indices)}")
    print(f"结果保存在 {save_dir}\n")

    return forget_indices, retain_indices


if __name__ == '__main__':
    trainset, testset = load_cifar100()
    print(f"成功加载 CIFAR-100：训练集 {len(trainset)}，测试集 {len(testset)}")

    superclass_to_forget = 'vehicles_1'  # 🚗 示例：遗忘“大型交通工具”
    save_dir = "split_results_cifar100"

    split_by_superclass(trainset, superclass_to_forget, save_dir, split_name="train")
    split_by_superclass(testset, superclass_to_forget, save_dir, split_name="test")
