# AlexNet 训练实验

本章将带你通过 PyTorch 实现一个完整的 AlexNet 训练流程，从数据准备到模型推理。这是一个端到端的深度学习实验，帮助你理解经典 CNN 架构的实际应用。

## 实验准备

在开始实验之前，请确保已挂载[数据目录](../../sandbox.md)并下载 Tiny ImageNet 数据集。
```bash
# 选择 "下载数据集" -> 选择 "Tiny ImageNet 200"
dmla data
```

## 第一阶段：数据准备

首先，我们验证数据集是否已正确下载，并检查其结构。Tiny ImageNet 包含 200 个类别，共 12 万张图像。训练前需要确认数据集完整下载、目录结构正确，否则后续 DataLoader 会因为找不到文件而报错。

```python runnable gpu
import os

# 检查数据目录是否存在（/data/ 是 Docker 沙箱中挂载宿主机数据卷的固定路径）
data_dir = '/data/datasets/tiny-imagenet-200'

if os.path.exists(data_dir):
    print("数据集目录已存在")
    
    # 检查子目录结构
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if os.path.exists(train_dir):
        train_classes = os.listdir(train_dir)
        print(f"训练集类别数: {len(train_classes)}")
        print(f"示例类别: {train_classes[:5]}")
    
    if os.path.exists(val_dir):
        val_files = os.listdir(val_dir)
        print(f"验证集文件数: {len(val_files)}")
else:
    print("数据集未下载，请运行 'dmla data' 下载数据集")
```

## 第二阶段：数据预处理

接下来，我们创建 PyTorch DataLoader，对图像进行预处理和数据增强。

> **目的：** PyTorch 训练需要 `Dataset` 和 `DataLoader` 两个组件。`Dataset` 负责把磁盘上的图像文件和标签映射成 `(图像, 标签)` 对，`DataLoader` 负责批量加载、打乱顺序、多线程读取。
>
> **为什么需要数据增强：** Tiny ImageNet 每个类别只有 500 张训练图，数据量有限。通过随机翻转、裁剪、颜色抖动等变换，可以人工增加训练数据的多样性，防止模型过拟合。
>
> **代码做了什么：**
> 1. **`TinyImageNetDataset` 类：** 继承 `torch.utils.data.Dataset`。训练集按类别子目录读取（每个文件夹名就是类别标签），验证集从 `val_annotations.txt` 中解析标签（验证集所有图片混放在一个目录中，标签写在标注文件里）
> 2. **`train_transform`（训练集预处理）：**
>    - `Resize(224, 224)`：AlexNet 要求输入为 224×224 的图像
>    - `RandomHorizontalFlip`：50% 概率水平翻转，增加姿态变化
>    - `RandomCrop(224, padding=4)`：先四周各填充 4 像素再随机裁剪 224×224，模拟尺度变化
>    - `ColorJitter`：随机调整亮度和对比度，增强对光照变化的鲁棒性
>    - `ToTensor`：将 PIL 图像转为 PyTorch Tensor（像素值从 [0, 255] 缩放到 [0, 1]）
>    - `Normalize`：使用 ImageNet 的统计均值和标准差进行标准化，使输入分布与预训练权重一致（便于后续迁移学习）
> 3. **`val_transform`（验证集预处理）：** 仅做 Resize + ToTensor + Normalize，不做数据增强（验证集需要保持稳定，才能客观评估模型性能）
> 4. **`DataLoader`：** `batch_size=64` 表示每批读取 64 张图，`shuffle=True` 在训练时打乱顺序以避免模型记住数据顺序

```python runnable gpu timeout=120
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np

# 自定义 Tiny ImageNet Dataset
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        self.samples = []
        self.classes = []
        
        if is_train:
            train_dir = os.path.join(root_dir, 'train')
            self.classes = sorted(os.listdir(train_dir))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            
            for cls in self.classes:
                cls_dir = os.path.join(train_dir, cls)
                images_dir = os.path.join(cls_dir, 'images')
                if os.path.exists(images_dir):
                    for img_name in os.listdir(images_dir):
                        if img_name.endswith('.JPEG'):
                            self.samples.append((
                                os.path.join(images_dir, img_name),
                                self.class_to_idx[cls]
                            ))
        else:
            # 验证集处理
            val_dir = os.path.join(root_dir, 'val')
            val_images_dir = os.path.join(val_dir, 'images')
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            
            if os.path.exists(val_annotations):
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0]
                            cls = parts[1]
                            if cls not in self.classes:
                                self.classes.append(cls)
                            self.samples.append((
                                os.path.join(val_images_dir, img_name),
                                self.classes.index(cls)
                            ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 数据预处理和增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # AlexNet 输入尺寸
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建 Dataset 和 DataLoader
data_dir = '/data/datasets/tiny-imagenet-200'

try:
    train_dataset = TinyImageNetDataset(data_dir, transform=train_transform, is_train=True)
    val_dataset = TinyImageNetDataset(data_dir, transform=val_transform, is_train=False)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"类别数: {len(train_dataset.classes)}")
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"\nDataLoader 创建成功")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
except Exception as e:
    print(f"数据加载失败: {e}")
    print("请确保数据集已正确下载")
```

## 第三阶段：模型定义

下面实现 AlexNet 网络结构，适配 Tiny ImageNet 的 200 个类别（而非原始的 1000 类）。

> **背景：** AlexNet 是 2012 年 ImageNet 竞赛冠军网络，包含 5 个卷积层和 3 个全连接层，共约 6000 万参数。它的成功证明了深度 CNN 在大规模图像分类上的强大能力。
>
> **代码做了什么：**
> 1. **`features`（特征提取层）：** 5 个卷积层交替叠加，逐层提取从低级（边缘、纹理）到高级（物体部件）的特征。卷积层之间的 `MaxPool2d` 负责下采样，逐步缩小空间尺寸。`AdaptiveAvgPool2d((6, 6))` 确保无论输入图像经过前面的卷积池化后尺寸如何，输出始终固定为 6×6，这样后续全连接层的输入维度就固定了
> 2. **`classifier`（分类层）：** 3 个全连接层。前两层使用 `Dropout(p=0.5)` 随机丢弃 50% 的神经元激活，防止过拟合（AlexNet 的标志性设计）。最后将 4096 维特征映射到 200 个类别的 logits
> 3. **为什么要改成 200 类：** 原始 AlexNet 最后一层输出 1000 类（对应完整 ImageNet），Tiny ImageNet 只有 200 类，所以 `num_classes=200`
> 4. **测试前向传播：** 用随机输入 `torch.randn(1, 3, 224, 224)` 跑一遍网络，验证输出形状为 `[1, 200]`，确认网络结构没有维度错误

```python runnable gpu extract-class="AlexNet"
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet 网络结构
    适配 Tiny ImageNet 200 类分类任务
    
    原始 AlexNet 为 1000 类，这里修改最后一层为 200 类
    使用 AdaptiveAvgPool2d 确保输出尺寸固定为 6x6
    """
    def __init__(self, num_classes=200):
        super(AlexNet, self).__init__()
        
        # 特征提取层 (5 个卷积层)
        self.features = nn.Sequential(
            # Conv1: 11x11 卷积，步长 4，输出 96 通道
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 5x5 卷积，输出 256 通道
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 3x3 卷积，输出 384 通道
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 3x3 卷积，输出 384 通道
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 3x3 卷积，输出 256 通道
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 自适应池化，确保输出固定为 6x6
            nn.AdaptiveAvgPool2d((6, 6))
        )
        
        # 分类层 (3 个全连接层)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 创建模型实例
model = AlexNet(num_classes=200)

# 打印模型结构
print("AlexNet 模型结构:")
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,}")

# 测试前向传播
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f"输入形状: {dummy_input.shape}")
print(f"输出形状: {output.shape}")
```

## 第四阶段：模型训练

本阶段执行完整的训练流程，使用 ProgressReporter 报告进度。

> **目的：** 这是整个实验的核心，模型通过多轮（epoch）迭代学习，逐步提升分类准确率。
>
> **训练流程概述：** 每个 epoch 中，模型遍历全部训练数据（前向传播 → 计算损失 → 反向传播 → 更新权重），然后在验证集上评估泛化能力。
>
> **代码做了什么：**
> 1. **设备选择：** 检测 GPU 是否可用，GPU 训练速度比 CPU 快 10-100 倍
> 2. **损失函数 `CrossEntropyLoss`：** 多分类任务的标准损失函数，对 logits 计算 softmax 后与真实标签求交叉熵
> 3. **优化器 `SGD`：** 随机梯度下降，`lr=0.01` 是初始学习率，`momentum=0.9` 加速收敛并抑制震荡，`weight_decay=0.0005`（L2 正则化）防止过拟合
> 4. **学习率调度 `StepLR`：** 每 10 个 epoch 将学习率乘以 0.1。训练初期用大学习率快速下降，后期用小学习率精细调优
> 5. **`train_one_epoch`：** 一个 epoch 的训练逻辑。每个 batch 执行 `zero_grad() → forward → loss → backward() → step()` 的标准反向传播流程
> 6. **`validate`：** 在验证集上评估模型，使用 `torch.no_grad()` 关闭梯度计算以节省显存和计算量
> 7. **模型保存策略：** 保存验证准确率最高的模型（`best_model.pth`），并每 5 个 epoch 保存一次 checkpoint，防止训练中断后可恢复

```python runnable gpu timeout=unlimited
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 检查 CUDA 可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024} MB")

# 创建模型
model = AlexNet(num_classes=200).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练参数
num_epochs = 20
save_dir = '/data/models/alexnet/checkpoints'
os.makedirs(save_dir, exist_ok=True)

# 创建进度报告器
progress = ProgressReporter(total_steps=num_epochs, description="训练 AlexNet on Tiny ImageNet")

# 训练函数
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

# 主训练循环
print("\n开始训练...")
best_acc = 0.0

try:
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更学习率
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% "
              f"Time: {epoch_time:.1f}s")
        
        # 更新进度报告
        progress.update(
            epoch + 1,
            message=f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%"
        )
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  -> 保存最佳模型 (准确率: {best_acc:.2f}%)")
        
        # 每 5 个 epoch 保存 checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(save_dir, f'epoch_{epoch+1}.pth'))
    
    # 训练完成
    progress.complete(message=f"训练完成，最佳验证准确率: {best_acc:.2f}%")
    
    # 保存最终模型
    final_path = '/data/models/alexnet/final/alexnet_tiny_imagenet.pth'
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"\n最终模型已保存: {final_path}")
    
except Exception as e:
    progress.error(message=f"训练出错: {str(e)}")
    print(f"\n训练出错: {e}")
```

## 第五阶段：模型推理

使用训练好的模型对新图像进行分类预测。

> **目的：** 训练完成后，验证模型的实际分类效果，展示模型"学到了什么"。
>
> **代码做了什么：**
> 1. **模型加载：** 优先加载验证准确率最高的 checkpoint（`best_model.pth`），其次加载最终模型。如果都找不到，则使用未训练的随机权重模型（仅供测试，预测结果无意义）
> 2. **`model.eval()`：** 切换到推理模式，关闭 Dropout 和 BatchNorm 的训练行为，确保每次推理结果一致
> 3. **推理预处理：** 与验证集预处理相同（Resize → ToTensor → Normalize），不做数据增强。输入图像的预处理方式必须与训练时一致
> 4. **类别名称映射：** Tiny ImageNet 的类别标签是 WordNet ID（如 `n01675725`），通过 `wnids.txt` 和 `words.txt` 映射为可读的英文描述（如 `turtle, tortoise`）
> 5. **`predict_image` 函数：**
>    - 读取图像 → 预处理 → 送入模型
>    - 使用 `softmax` 将 logits 转为概率（0-100%）
>    - `topk(5)` 取概率最高的 5 个类别，输出 Top-5 预测结果
>    - Top-5 是图像分类常用的评估指标：只要正确答案在前 5 个预测中，就认为模型"猜对了"

```python runnable gpu
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 加载训练好的模型
model_path = '/data/models/alexnet/final/alexnet_tiny_imagenet.pth'
checkpoint_path = '/data/models/alexnet/checkpoints/best_model.pth'

# 选择加载路径
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = AlexNet(num_classes=200)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载最佳模型 (Epoch {checkpoint['epoch']}, 准确率 {checkpoint['best_acc']:.2f}%)")
elif os.path.exists(model_path):
    model = AlexNet(num_classes=200)
    model.load_state_dict(torch.load(model_path))
    print("加载最终模型")
else:
    print("未找到训练好的模型，使用未训练的模型（预测结果将随机）")
    model = AlexNet(num_classes=200)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载类别名称（从 wnids.txt）
wnids_path = '/data/datasets/tiny-imagenet-200/wnids.txt'
words_path = '/data/datasets/tiny-imagenet-200/words.txt'

class_names = {}
if os.path.exists(wnids_path) and os.path.exists(words_path):
    with open(wnids_path, 'r') as f:
        wnids = [line.strip() for line in f.readlines()]
    
    with open(words_path, 'r') as f:
        word_lines = f.readlines()
        for line in word_lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                class_names[parts[0]] = parts[1]

def predict_image(image_path, model, transform, device, class_names, wnids):
    """对单张图像进行预测"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top5_prob, top5_idx = probabilities.topk(5)
    
    results = []
    for i in range(5):
        idx = top5_idx[0][i].item()
        prob = top5_prob[0][i].item() * 100
        
        if idx < len(wnids):
            wnid = wnids[idx]
            name = class_names.get(wnid, wnid)
        else:
            name = f"Class {idx}"
        
        results.append((name, prob))
    
    return results

# 使用验证集中的一张图片进行测试
val_images_dir = '/data/datasets/tiny-imagenet-200/val/images'

if os.path.exists(val_images_dir):
    test_images = os.listdir(val_images_dir)[:5]
    
    print("\n预测示例:")
    print("=" * 60)
    
    for img_name in test_images:
        img_path = os.path.join(val_images_dir, img_name)
        
        if os.path.exists(img_path):
            predictions = predict_image(img_path, model, transform, device, class_names, wnids if 'wnids' in dir() else [])
            
            print(f"\n图像: {img_name}")
            print("Top-5 预测:")
            for rank, (name, prob) in enumerate(predictions, 1):
                print(f"  {rank}. {name}: {prob:.2f}%")
else:
    print("验证集目录不存在，无法进行推理测试")

# 也可以使用自定义图片
print("\n" + "=" * 60)
print("提示: 您可以将自己的图片放到 /data/datasets/custom/ 目录进行测试")
print("使用方法: predict_image('/data/datasets/custom/your_image.jpg', model, transform, device, class_names, wnids)")
```

## 实验总结

本实验完整展示了 AlexNet 的训练流程：

| 阶段 | 关键步骤 | 代码块 |
|:-----|:---------|:-------|
| 数据准备 | 检查/下载数据集 | `timeout=300` |
| 数据预处理 | DataLoader 创建 | `timeout=120` |
| 模型定义 | AlexNet 类定义 | `extract-class` |
| 模型训练 | 训练循环 + 进度报告 | `timeout=unlimited` |
| 模型推理 | 加载模型预测 | 常规 |

### 保存的文件

训练完成后，以下文件将保存到数据目录：

- `/data/models/alexnet/checkpoints/best_model.pth` - 最佳验证准确率的模型
- `/data/models/alexnet/checkpoints/epoch_*.pth` - 每 5 epoch 的 checkpoint
- `/data/models/alexnet/final/alexnet_tiny_imagenet.pth` - 最终模型权重
