import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# === 引入 MoEViT 模型 ===
from origin import VisionTransformerMoE  # 你自己的模型文件

# ------------------------------
#        参数配置
# ------------------------------
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.0001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 100
SAVE_PATH = './checkpoints'
os.makedirs(SAVE_PATH, exist_ok=True)

# ------------------------------
#        数据加载
# ------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)

# ------------------------------
#        模型定义
# ------------------------------
model = VisionTransformerMoE(
    model_name="vit_small_patch16_224",
    img_size=(32,32),
    patch_size=4,
    embed_dim=384,       # 取决于你的模型设置
    depth=12,
    num_heads=6,
    num_classes=NUM_CLASSES,
    mlp_ratio=4.0,
    drop_rate=0.1,
    pos_embed_interp=True
)
model.to(DEVICE)

# ------------------------------
#        优化器与损失
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ------------------------------
#        训练与验证函数
# ------------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(trainloader, total=len(trainloader), desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        # if isinstance(outputs, tuple):  # 如果模型返回额外项，如 gates_loss
        #     # outputs, gate_loss = outputs
        #     # loss = criterion(outputs, labels) + 0.01 * gate_loss
        #     outputs = outputs
        #     loss = criterion(outputs, labels)
        # else:
        #     loss = criterion(outputs, labels)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        acc = 100. * correct / total

        loop.set_postfix(loss=running_loss / (len(loop)), acc=acc)

    return running_loss / len(trainloader), acc


def test(epoch):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"[Test] Epoch {epoch+1}: Acc={acc:.2f}%, Loss={test_loss / len(testloader):.4f}")
    return acc


# ------------------------------
#        主训练循环
# ------------------------------
best_acc = 0.0
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(epoch)
    test_acc = test(epoch)
    scheduler.step()

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), f"{SAVE_PATH}/moevit_best.pth")
        print(f"✅ Best model updated! Test Acc: {best_acc:.2f}%")

print(f"🎯 Training Finished! Best Test Acc = {best_acc:.2f}%")
