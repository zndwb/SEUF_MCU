import os
import torch
import curves
import models
import utils
from arg_parser import parse_args
import data

# --------------------------
# 初始化MCU相关组件
# --------------------------
# 1. 解析参数（可根据实际配置调整）
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 加载数据集（与GA方法共享数据加载器）
loaders, num_classes = data.loaders(
    args.dataset,
    args.unlearn_type,
    args.forget_ratio,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)
forgetloader = loaders['train_forget']
retainloader = loaders['train_retain']

# 3. 定义曲线模型（MCU核心：连接原始模型和目标retain模型）
architecture = getattr(models, args.model)  # 假设模型为MoE-ViT
curve = getattr(curves, args.curve)  # 如Bezier曲线
model = curves.CurveNet(
    num_classes,
    curve,
    architecture.curve,
    args.num_bends,  # 曲线弯曲点数，如3
    fix_start=True,  # 固定起点（原始模型）
    fix_end=True,  # 固定终点（retain模型）
    architecture_kwargs=architecture.kwargs
)
model.to(device)

# 4. 加载端点模型
# 4.1 原始模型（起点）
original_ckpt = torch.load(args.original_pth, weights_only=True)
base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
base_model.load_state_dict(original_ckpt['model_state'])
model.import_base_parameters(base_model, k=0)  # k=0表示起点

# 4.2 Retain模型（终点，目标模型）
retain_ckpt = torch.load('RT_class/moe_vit_cifar10_unlearn_best.pth', weights_only=True)
base_model.load_state_dict(retain_ckpt['model_state_dict'])  # 对应保存的键
model.import_base_parameters(base_model, k=args.num_bends - 1)  # 最后一个点为终点

# 5. 生成并加载掩码（使用generate_weight_mask.py逻辑）
# 5.1 生成掩码（如需动态生成，可直接调用generate_weight_mask.py的main函数）
mask_dir = 'vit_cifar10_class_20/mask/'  # 掩码保存目录
os.makedirs(mask_dir, exist_ok=True)
# 假设已通过generate_weight_mask.py生成掩码，这里直接加载
mask_path = os.path.join(mask_dir, 'mask_k0.5_kr0.2.pt')  # 选择合适阈值的掩码
mask = torch.load(mask_path)

# 5.2 适配曲线模型参数命名（曲线模型参数格式为"net.{name}_{i}"）
curve_mask = {}
for name in mask:
    for i in range(args.num_bends):
        curve_name = f'net.{name}_{i}'
        # 处理MoE层特殊命名（如router.gate）
        curve_name = curve_name.replace("router.gate", "router.gate_{i}")
        curve_mask[curve_name] = mask[name]
mask = curve_mask

# 6. 配置优化器（仅优化掩码标记的重要参数）
params_to_train = [
    param for name, param in model.named_parameters()
    if name in mask and mask[name] == 1
]
optimizer = torch.optim.SGD(
    params_to_train,
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
criterion = torch.nn.CrossEntropyLoss()

# --------------------------
# MCU核心训练逻辑（替换GA的更新步骤）
# --------------------------
# 混合数据集（保留集+遗忘集，与GA方法的LossData2一致）
mixed_dataset = utils.LossData2(
    forget_data=forgetloader.dataset,
    retain_data=retainloader.dataset
)
train_loader = torch.utils.data.DataLoader(
    mixed_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True
)

# 训练循环
for epoch in range(args.epochs):
    model.train()
    for batch in train_loader:
        inputs, targets, labels = batch  # labels=1（遗忘）/0（保留）
        inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)

        # 路径参数t（初始为0.5，可随训练调整）
        t = torch.tensor([0.5], device=device, requires_grad=True)
        outputs = model(inputs, t=t)  # 曲线模型输出

        # 分离保留集和遗忘集损失
        retain_mask = (labels == 0)
        forget_mask = (labels == 1)

        loss_retain = criterion(outputs[retain_mask], targets[retain_mask])
        loss_forget = criterion(outputs[forget_mask], targets[forget_mask])

        # MCU损失函数：最小化保留集损失，最大化遗忘集损失
        loss = loss_retain - args.beta * loss_forget

        # 梯度更新（仅更新掩码标记的参数）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# --------------------------
# 保存MCU优化后的模型
# --------------------------
checkpoint_path = 'RT_class/moe_vit_cifar10_mcu_best.pth'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),  # 保存曲线模型参数
    'optimizer_state_dict': optimizer.state_dict(),
    'acc': utils.evaluate_acc(model, loaders['test_retain'], device)  # 保留集精度
}, checkpoint_path)