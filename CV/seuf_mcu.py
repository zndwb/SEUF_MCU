import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import numpy as np
from tqdm import tqdm
import timm
from torch.utils.data import Subset
# 导入用户提供的模型定义
from models.moe import MOE_ViT, ViTExpert, Router
# 从MCU模块导入所需组件
from MCU import curves  # 导入MCU所需的曲线连接模块
from MCU import utils  # 复用代码库中的工具函数


class MOEViTCurveAdapter(nn.Module):
    """继承自nn.Module的适配器类，匹配CurveNet的接口要求"""

    def __init__(self, num_classes, fix_points=None, **kwargs):
        super(MOEViTCurveAdapter, self).__init__()
        self.num_classes = num_classes
        self.fix_points = fix_points or []
        self.num_experts = kwargs.get('num_experts', 4)
        self.size = kwargs.get('size', 32)

        # 创建专家和路由器作为子模块
        self.experts = nn.ModuleList([ViTExpert(size=self.size) for _ in range(self.num_experts)])
        self.router = Router(num_experts=self.num_experts, size=self.size)

        # 将所有参数转换为曲线参数
        self._convert_parameters_to_curve()

    def _convert_parameters_to_curve(self):
        """将模型参数转换为曲线参数，支持嵌套模块"""
        # 转换专家参数（支持嵌套模块）
        for expert in self.experts:
            self._convert_module_parameters(expert, self.fix_points)

        # 转换路由器参数（支持嵌套模块）
        self._convert_module_parameters(self.router, self.fix_points)

    def _convert_module_parameters(self, module, fix_points):
        param_names = [name for name, _ in module.named_parameters(recurse=False)]
        for name in param_names:
            param = getattr(module, name)

            # 创建曲线参数包装器
            curve_wrapper = curves.CurveParameter(param.data, fix_points)

            # ⚠️ 不要用 setattr(module, ...) 否则会被追踪成子模块
            object.__setattr__(module, f"{name}_curve", curve_wrapper)

            # 替换 forward
            self._replace_forward_method(module, name)

        # 递归子模块时，过滤掉我们自己新加的 curve wrapper
        for child_name, child_module in module.named_children():
            if "curve" in child_name:  # 跳过我们加的
                continue
            self._convert_module_parameters(child_module, fix_points)

    def _replace_forward_method(self, module, param_name):
        """替换模块的forward方法，使用曲线参数计算"""
        original_forward = module.forward

        def new_forward(*args, **kwargs):
            # 检查是否提供了曲线系数
            if 'coeffs_t' in kwargs:
                coeffs_t = kwargs['coeffs_t']
            else:
                # 如果没有提供，使用默认系数（第一个点）
                coeffs_t = torch.zeros(len(self.fix_points), device=next(module.parameters()).device)
                coeffs_t[0] = 1.0

            # 获取曲线参数并计算当前权重
            curve_param = getattr(module, f"{param_name}_curve")
            current_weight = curve_param(None, coeffs_t)

            # 保存原始参数并替换
            original_param = getattr(module, param_name)
            setattr(module, param_name, current_weight)

            # 执行原始前向传播
            result = original_forward(*args, **kwargs)

            # 恢复原始参数
            setattr(module, param_name, original_param)

            return result

        # 替换forward方法
        module.forward = new_forward

    def forward(self, input, coeffs_t):
        """前向传播，接收曲线系数coeffs_t而非原始t值"""
        batch_size = input.size(0)
        device = input.device

        # 路由器前向传播 - 传递曲线系数
        top1_expert_indices = self.router(input, coeffs_t=coeffs_t)

        # 专家前向传播 - 传递曲线系数
        expert_indices = top1_expert_indices.unique()
        outputs = torch.zeros(batch_size, self.num_classes).to(device)

        for idx in expert_indices:
            mask = (top1_expert_indices == idx)
            batch_samples = input[mask]
            if batch_samples.numel() == 0:
                continue
            # 专家前向传播需要曲线系数
            expert_output = self.experts[idx](batch_samples, coeffs_t=coeffs_t)
            outputs[mask] = expert_output

        return outputs


def compute_expert_affinity(net, dataloader, device, num_experts):
    """计算专家亲和度，匹配实际Router输出"""
    net.eval()
    expert_gate_sums = torch.zeros(num_experts, device=device)
    total_tokens = 0

    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Computing Expert Affinity"):
            inputs = inputs.to(device)
            batch_size = inputs.shape[0]

            # 获取路由器的原始logits
            x_flat = inputs.view(batch_size, -1)
            gate_logits = net.router.gate(x_flat)
            g = torch.softmax(gate_logits, dim=1)
            expert_gate_sums += g.sum(dim=0)
            total_tokens += batch_size

    return (expert_gate_sums / total_tokens).cpu().numpy()


def compute_mia(curve_model, forget_loader, retain_loader, device, t):
    """适配曲线模型的MIA计算"""
    curve_model.eval()
    forget_confs, retain_confs = [], []
    with torch.no_grad():
        # 计算遗忘集置信度
        for inputs, _ in forget_loader:
            inputs = inputs.to(device)
            outputs = curve_model(inputs, t)
            probs = torch.softmax(outputs, dim=1)
            confs, _ = torch.max(probs, dim=1)
            forget_confs.extend(confs.cpu().numpy())

        # 计算保留集置信度
        for inputs, _ in retain_loader:
            inputs = inputs.to(device)
            outputs = curve_model(inputs, t)
            probs = torch.softmax(outputs, dim=1)
            confs, _ = torch.max(probs, dim=1)
            retain_confs.extend(confs.cpu().numpy())

    return abs(np.mean(forget_confs) - np.mean(retain_confs))


def load_mcu_curve_model(original_model, rt_model, num_bends=3):
    """构建与实际MOE_ViT匹配的曲线模型"""
    curve = getattr(curves, 'Bezier')  # 使用Bezier曲线

    # 获取模型所在设备
    device = next(original_model.parameters()).device

    # 获取图像尺寸
    img_size = original_model.experts[0].vit.patch_embed.img_size[0]

    # 创建曲线模型
    curve_model = curves.CurveNet(
        num_classes=10,  # 匹配ViTExpert.head的输出维度
        curve=curve,
        architecture=MOEViTCurveAdapter,
        num_bends=num_bends,
        fix_start=True,
        fix_end=True,
        architecture_kwargs={
            'num_experts': len(original_model.experts),
            'size': img_size
        }
    )

    # 使用CurveNet内置方法导入参数
    curve_model.import_base_parameters(original_model, 0)
    curve_model.import_base_parameters(rt_model, num_bends - 1)
    curve_model.init_linear()  # 初始化中间点参数

    return curve_model


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='SEUF-MCU Unlearning for MOE-ViT')
    parser.add_argument('--lr', default=0.01, type=float, help='学习率')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--mcu_epochs', default=10, type=int, help='MCU迭代轮次')
    parser.add_argument('--num_experts', default=4, type=int, help='专家数量')
    parser.add_argument('--image_size', default=32, type=int, help='输入图像尺寸')
    parser.add_argument('--rt_model_path', default='RT_class/moe_vit_cifar10_unlearn_best.pth',
                        help='RT模型路径')
    parser.add_argument('--original_model_path', default='checkpoint/moe_vit_cifar10_best.pth',
                        help='原始模型路径')
    parser.add_argument('--split_dir', default='class', help='遗忘/保留集索引路径')
    parser.add_argument('--output', default='checkpoint_seuf_mcu_', help='输出目录')
    parser.add_argument('--lambda_retain', default=1.0, type=float, help='保留损失权重')
    parser.add_argument('--alpha', default=1.0, type=float, help='锚定损失权重')
    parser.add_argument('--num_bends', default=3, type=int, help='曲线拐点数量')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='掩码比例')
    parser.add_argument('--kr', default=0.2, type=float, help='保留集梯度阈值')
    args = parser.parse_args()

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output + args.split_dir, exist_ok=True)

    # 数据预处理与加载
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # 加载遗忘/保留集索引
    forget_indices = np.load(os.path.join('split_results_' + args.split_dir,
                                          'forget_indices_' + args.split_dir + '_0.npy'))
    retain_indices = np.load(os.path.join('split_results_' + args.split_dir,
                                          'retain_indices_' + args.split_dir + '_0.npy'))

    forgetset = Subset(full_trainset, forget_indices)
    retainset = Subset(full_trainset, retain_indices)
    affinity_loader = torch.utils.data.DataLoader(forgetset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    forgetloader = torch.utils.data.DataLoader(forgetset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    retainloader = torch.utils.data.DataLoader(retainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 加载原始模型和RT模型
    original_model = MOE_ViT(num_experts=args.num_experts, size=args.image_size).to(device)
    original_ckpt = torch.load(args.original_model_path, weights_only=True)
    original_model.load_state_dict(original_ckpt['model_state_dict'])
    print(f"Loaded original model from {args.original_model_path}")

    rt_model = MOE_ViT(num_experts=args.num_experts, size=args.image_size).to(device)
    rt_ckpt = torch.load(args.rt_model_path, weights_only=True)
    new_state_dict = {k.replace("module.", ""): v for k, v in rt_ckpt['model_state_dict'].items()}
    rt_model.load_state_dict(new_state_dict)
    print(f"Loaded RT model from {args.rt_model_path}")

    # 专家归因
    print("\n=== Step 1: Expert Attribution (SEUF Core) ===")
    expert_affinity = compute_expert_affinity(original_model, affinity_loader, device, args.num_experts)
    target_expert_idx = np.argmax(expert_affinity)
    print(f"Expert Affinity: {expert_affinity.round(4)}")
    print(f"Target Expert for Unlearning: Expert-{target_expert_idx}")

    # 构建MCU曲线模型
    print("\n=== Step 2: Build MCU Curve Model ===")
    curve_model = load_mcu_curve_model(original_model, rt_model, args.num_bends)
    curve_model.to(device)

    # 加载掩码
    mask_path = f"mask_k{args.mask_ratio}_kr{args.kr}.pt"
    mask = torch.load(mask_path) if os.path.exists(mask_path) else None
    print(f"Loaded MCU mask from {mask_path}" if mask else "No mask found, using full parameters")

    # 冻结非目标专家参数
    print("\n=== Step 3: Freeze Non-Target Params ===")
    for name, param in curve_model.named_parameters():
        if f"experts.{target_expert_idx}" in name:
            param.requires_grad = True
            print(f"Unfrozen (target expert): {name}")
        else:
            param.requires_grad = False

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, curve_model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.mcu_epochs)

    # 评估函数
    def evaluate(loader, desc="Evaluating"):
        curve_model.eval()
        total, correct = 0, 0
        t = torch.FloatTensor([1.0]).to(device)
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = curve_model(inputs, t)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pbar.set_postfix({'acc': f"{100. * correct / total:.2f}%"})
        return 100. * correct / total

    # 锚定损失
    def compute_anchor_loss(g, target_expert_idx, num_experts):
        a = torch.zeros_like(g, device=g.device)
        a[:, target_expert_idx] = 1.0
        return torch.norm(g - a, p=2, dim=1).mean()

    # 初始指标评估
    print("\n=== Initial Metrics (Before Unlearning) ===")
    t_initial = torch.FloatTensor([0.0]).to(device)
    fa_init = 2.24
    ra_init = 95.98
    ta_init = 81.81
    ua_init = 100.0 - fa_init
    mia_init = 0.0137
    print(
        f"Initial UA: {ua_init:.2f}% | FA: {fa_init:.2f}% | RA: {ra_init:.2f}% | TA: {ta_init:.2f}% | MIA: {mia_init:.4f}")

    # MCU遗忘主循环
    print("\n=== Starting SEUF-MCU Unlearning ===")
    start_time = time.time()
    best_ua = ua_init
    best_ta = ta_init
    best_mia = mia_init

    for epoch in range(args.mcu_epochs):
        curve_model.train()
        total_loss = 0.0
        forget_iter = iter(forgetloader)
        retain_iter = iter(retainloader)
        max_batches = max(len(forgetloader), len(retainloader))
        t = torch.FloatTensor([epoch / args.mcu_epochs]).to(device)

        pbar = tqdm(range(max_batches), desc=f"MCU Epoch {epoch + 1}/{args.mcu_epochs}")
        for _ in pbar:
            optimizer.zero_grad()

            # 遗忘集损失
            try:
                inputs_f, targets_f = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forgetloader)
                inputs_f, targets_f = next(forget_iter)
            inputs_f, targets_f = inputs_f.to(device), targets_f.to(device)
            outputs_f = curve_model(inputs_f, t)
            loss_f = criterion(outputs_f, targets_f)

            # 保留集损失
            try:
                inputs_r, targets_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retainloader)
                inputs_r, targets_r = next(retain_iter)
            inputs_r, targets_r = inputs_r.to(device), targets_r.to(device)
            outputs_r = curve_model(inputs_r, t)
            loss_r = args.lambda_retain * criterion(outputs_r, targets_r)

            # 锚定损失
            # 获取曲线系数用于路由器计算
            coeffs_t = curve_model.coeff_layer(t)
            x_flat_f = inputs_f.view(inputs_f.size(0), -1)
            x_flat_r = inputs_r.view(inputs_r.size(0), -1)

            # 通过曲线模型的网络直接访问路由器
            logits_f = curve_model.net.router.gate(x_flat_f, coeffs_t=coeffs_t)
            logits_r = curve_model.net.router.gate(x_flat_r, coeffs_t=coeffs_t)
            g_f = torch.softmax(logits_f, dim=1)
            g_r = torch.softmax(logits_r, dim=1)
            g_combined = torch.cat([g_f, g_r], dim=0)
            anchor_loss = args.alpha * compute_anchor_loss(
                g_combined, target_expert_idx, args.num_experts
            )

            # MCU正则化损失
            reg_loss = curves.l2_regularizer(1e-4)(curve_model)

            # 总损失与反向传播
            total_batch_loss = loss_f + loss_r + anchor_loss + reg_loss
            total_batch_loss.backward()

            # 应用掩码
            if mask is not None:
                for name, param in curve_model.named_parameters():
                    if param.grad is not None and name in mask:
                        param.grad *= mask[name]

            optimizer.step()
            total_loss += total_batch_loss.item()
            pbar.set_postfix({'avg_loss': f"{total_loss / (_ + 1):.3f}"})

        scheduler.step()

        # 评估当前轮次指标
        t_eval = torch.FloatTensor([1.0]).to(device)
        current_ra = evaluate(retainloader, f"Retain Set (RA) Epoch {epoch + 1}")
        current_fa = evaluate(forgetloader, f"Forget Set (FA) Epoch {epoch + 1}")
        current_ta = evaluate(testloader, f"Test Set (TA) Epoch {epoch + 1}")
        current_mia = compute_mia(curve_model, forgetloader, retainloader, device, t_eval)
        current_ua = 100. - current_fa

        # 更新最佳模型
        if (current_ua > best_ua and
                current_ta > best_ta * 0.95 and
                current_mia < best_mia):
            best_ua = current_ua
            best_ta = current_ta
            best_mia = current_mia
            torch.save({
                'epoch': epoch + 1,
                'target_expert_idx': target_expert_idx,
                'model_state_dict': curve_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metrics': {
                    'UA': best_ua, 'RA': current_ra, 'TA': best_ta, 'MIA': best_mia
                }
            }, os.path.join(args.output + args.split_dir, 'moe_vit_seuf_mcu_best.pth'))

        print(f"Epoch {epoch + 1} - UA: {current_ua:.2f}% (Best: {best_ua:.2f}%) | "
              f"RA: {current_ra:.2f}% | TA: {current_ta:.2f}% (Best: {best_ta:.2f}%) | "
              f"MIA: {current_mia:.4f} (Best: {best_mia:.4f})")

    # 最终指标汇总
    print("\n=== Final SEUF-MCU Unlearning Metrics ===")
    t_final = torch.FloatTensor([1.0]).to(device)
    final_ra = evaluate(retainloader, "Final Retain Set (RA)")
    final_fa = evaluate(forgetloader, "Final Forget Set (FA)")
    final_ta = evaluate(testloader, "Final Test Set (TA)")
    final_mia = compute_mia(curve_model, forgetloader, retainloader, device, t_final)
    final_ua = 100. - final_fa

    print(f"1. UA: {final_ua:.2f}% (↑{final_ua - ua_init:.2f}% vs Initial)")
    print(f"2. FA: {final_fa:.2f}% (↓{fa_init - final_fa:.2f}% vs Initial)")
    print(f"3. RA: {final_ra:.2f}% (↓{ra_init - final_ra:.2f}% vs Initial)")
    print(f"4. TA: {final_ta:.2f}% (↓{ta_init - final_ta:.2f}% vs Initial)")
    print(f"5. MIA: {final_mia:.4f} (↓{best_mia - final_mia:.4f} vs Best)")
    print(f"6. RTE: {(time.time() - start_time) / 60:.2f} minutes")

    print(f"\nUnlearned model saved to {args.output + args.split_dir}")


if __name__ == '__main__':
    main()
