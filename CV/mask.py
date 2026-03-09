import os
import torch
import models
import data
import utils
from arg_parser import parse_args
from MCU.generate_weight_mask import save_gradient_ratio  # 复用原掩码生成核心函数


def generate_mask_for_rt_model():
    parser = argparse.ArgumentParser(description='MASK)')
    parser.add_argument('--lr', default=0.01, type=float, help='GA learning rate (原文建议小学习率)')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--ga_epochs', default=5, type=int, help='GA unlearning iterations (原文20轮最优)')
    parser.add_argument('--num_experts', default=4, type=int, help='number of experts in MOE (原文适配4专家)')
    parser.add_argument('--image_size', default=32, type=int, help='input image size (CIFAR-10为32)')
    parser.add_argument('--checkpoint', default='checkpoint', help='path to initial model (moe_vit_cifar10_best.pth)')
    parser.add_argument('--split_dir', default='class', help='path to forget/retain indices')
    parser.add_argument('--output', default='checkpoint_seuf_ga_', help='output dir for unlearned model')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 固定随机种子确保一致性
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 加载数据集（与训练时参数匹配）
    loaders, num_classes = data.loaders(
        args.dataset,
        args.unlearn_type,
        args.forget_ratio,
        args.data_path,  # 原始数据在data文件夹，通过命令行指定
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test
    )

    # 加载RT模型（moe_vit）
    architecture = getattr(models, args.model)  # 需指定为'moe_vit'
    model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    model.to(device)

    # 加载RT模型权重（路径固定为RT_class/moe_vit_cifar10_unlearn_best.pth）
    rt_model_path = 'RT_class/moe_vit_cifar10_unlearn_best.pth'
    checkpoint_rt = torch.load(rt_model_path, weights_only=True)

    # 适配权重格式（去除可能的分布式训练前缀）
    new_state_dict = {}
    for k, v in checkpoint_rt['model_state'].items():
        new_key = k.replace("module.", "") if "module." in k else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    print(f"已加载RT模型: {rt_model_path}")

    # 配置掩码保存目录（可自定义）
    save_dir = 'moe_vit_cifar10_mask/'
    os.makedirs(save_dir, exist_ok=True)
    print(f"掩码将保存至: {save_dir}")

    # 调用原文件的掩码生成函数
    criterion = torch.nn.CrossEntropyLoss()
    save_gradient_ratio(
        data_loaders=loaders,
        model=model,
        criterion=criterion,
        args=args,
        save_dir=save_dir
    )
    print("掩码生成完成")


if __name__ == "__main__":
    generate_mask_for_rt_model()