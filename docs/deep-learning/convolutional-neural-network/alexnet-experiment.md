# AlexNet 训练实验

本节将带你通过 PyTorch 实现一个完整的 AlexNet 训练流程，从数据准备到模型推理，进行第一个端到端的深度学习实验，帮助你理解经典 CNN 架构与现代机器学习框架下如何定义一个模型应用。

## 实验准备

在开始实验之前，请确保已挂载[数据目录](../../sandbox.md)并下载 Tiny ImageNet 数据集：
```bash
# 选择 "下载数据集" -> 选择 "Tiny ImageNet 200"
dmla data
```

## 第一阶段：数据准备

首先，验证数据集是否已正确下载，并检查其结构。Tiny ImageNet 包含 200 个类别，共 12 万张图像。训练前需要确认数据集完整下载、目录结构正确，否则后续 DataLoader 会因为找不到文件而报错。

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

## 第二阶段：数据预处理与缓存

接下来，我们创建 PyTorch DataLoader 对图像进行预处理和数据增强。Tiny ImageNet 200 数据集每个类别只有 500 张训练图，相对于模型参数量而言，数量十分有限。数据增强通过随机翻转、裁剪、颜色抖动等变换，可以人工增加训练数据的多样性，防止模型过拟合。AlexNet 参加比赛时使用的是 ImageNet 1K 数据集，尽管比 200 数据集来说要大不少，但仍然需进行数据预处理增强。

本阶段借助了 PyTorch 中十分常用的 `Dataset` 和 `DataLoader` 两个组件。`Dataset` 负责把磁盘上的图像文件和标签映射成 `(图像, 标签)` 对，`DataLoader` 负责批量加载、打乱顺序、多线程读取。

::: tip 性能优化：预处理缓存
原始 Tiny ImageNet 图片尺寸为 64×64，需要放大到 224×224 才能输入 AlexNet。这个 Resize 操作在 CPU 上非常耗时，会导致 GPU 利用率极低（约 15%）。

**解决方案：** 本阶段会将预处理好的数据缓存到磁盘（`.pt` 文件），训练时直接加载缓存，跳过预处理步骤，让 GPU 利用率提升到 80%+。

- 缓存位置：`/data/cache/preprocessing/tiny-imagenet-224/`
- 首次运行：执行预处理并保存（约 5-10 分钟）
- 后续运行：直接加载缓存（几秒钟）
:::

**预处理缓存流程：**

1. **检查缓存目录：** 如果 `/data/cache/preprocessing/tiny-imagenet-224/` 已存在，跳过预处理
2. **训练集预处理：** 每个类别一个 `.pt` 文件，包含该类别所有图片的预处理 tensor
3. **验证集预处理：** 所有验证图片打包为 `val.pt`
4. **预处理内容：**
   - `Resize(224, 224)`：从 64×64 放大到 224×224（最耗时，缓存后跳过）
   - `ToTensor`：转换为 PyTorch tensor
   - `Normalize`：使用 ImageNet 均值和标准差标准化
5. **数据增强保留：** RandomHorizontalFlip、RandomCrop 等在训练时实时执行（不缓存）

```python runnable gpu timeout=unlimited
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import time

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 预处理缓存目录
CACHE_DIR = '/data/cache/preprocessing/tiny-imagenet-224'
TRAIN_CACHE = os.path.join(CACHE_DIR, 'train')
VAL_CACHE = os.path.join(CACHE_DIR, 'val.pt')

# 原始数据目录
DATA_DIR = '/data/datasets/tiny-imagenet-200'

# 基础预处理（将被缓存）
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 64→224 放大
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_and_cache():
    """预处理数据集并缓存到磁盘"""
    print("开始预处理并缓存数据集...")
    start_time = time.time()
    
    # 创建进度报告器（训练集 200 个类别）
    progress = ProgressReporter(total_steps=200, description="预处理训练集")
    
    os.makedirs(TRAIN_CACHE, exist_ok=True)
    
    # 处理训练集（按类别分组）
    train_dir = os.path.join(DATA_DIR, 'train')
    classes = sorted(os.listdir(train_dir))
    total_samples = 0
    
    for cls_idx, cls in enumerate(classes):
        cls_dir = os.path.join(train_dir, cls)
        images_dir = os.path.join(cls_dir, 'images')
        
        if not os.path.exists(images_dir):
            continue
        
        # 收集该类别所有图片
        images = []
        labels = []
        
        for img_name in os.listdir(images_dir):
            if img_name.endswith('.JPEG'):
                img_path = os.path.join(images_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                img_tensor = base_transform(img)
                images.append(img_tensor)
                labels.append(cls_idx)
                total_samples += 1
        
        # 保存为 .pt 文件
        if images:
            cls_data = {
                'images': torch.stack(images),  # [N, 3, 224, 224]
                'labels': torch.tensor(labels),  # [N]
                'class_name': cls
            }
            cache_path = os.path.join(TRAIN_CACHE, f'{cls}.pt')
            torch.save(cls_data, cache_path)
        
        progress.update(cls_idx + 1, message=f"预处理类别 {cls_idx+1}/200: {cls} ({len(images)} 张)")
    
    # 训练集预处理完成
    progress.complete(message=f"训练集预处理完成: {total_samples} 张图片")
    
    # 处理验证集
    val_dir = os.path.join(DATA_DIR, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')
    
    # 读取类别映射
    wnids_path = os.path.join(DATA_DIR, 'wnids.txt')
    with open(wnids_path, 'r') as f:
        wnids = [line.strip() for line in f.readlines()]
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
    
    # 先读取标注文件获取总数
    with open(val_annotations, 'r') as f:
        val_lines = f.readlines()
    total_val = len(val_lines)
    
    # 重置进度报告器用于验证集处理
    progress.reset(total_steps=total_val, description="预处理验证集")
    
    val_images = []
    val_labels = []
    
    try:
        for line_idx, line in enumerate(val_lines):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name = parts[0]
                cls = parts[1]
                img_path = os.path.join(val_images_dir, img_name)
                
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = base_transform(img)
                    val_images.append(img_tensor)
                    val_labels.append(class_to_idx.get(cls, 0))
            
            # 每 100 张图片更新进度
            if (line_idx + 1) % 100 == 0 or line_idx == total_val - 1:
                progress.update(
                    line_idx + 1,
                    message=f"预处理验证集 {line_idx+1}/{total_val} 张图片"
                )
        
        # 分段保存验证集（避免一次性处理过大数据）
        # 将10000张图片分成5个批次保存，每个批次2000张
        print(f"验证集处理完成，开始分段保存...")
        num_batches = 5
        batch_size = len(val_images) // num_batches
        
        val_cache_dir = os.path.join(CACHE_DIR, 'val')
        os.makedirs(val_cache_dir, exist_ok=True)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(val_images))
            
            batch_images = val_images[start_idx:end_idx]
            batch_labels = val_labels[start_idx:end_idx]
            
            if batch_images:
                batch_data = {
                    'images': torch.stack(batch_images),
                    'labels': torch.tensor(batch_labels)
                }
                batch_path = os.path.join(val_cache_dir, f'batch_{batch_idx}.pt')
                torch.save(batch_data, batch_path)
                print(f"保存批次 {batch_idx+1}/{num_batches}: {len(batch_images)} 张图片")
        
        # 合并所有批次并保存最终文件
        print("合并验证集批次...")
        all_batch_images = []
        all_batch_labels = []
        
        for batch_idx in range(num_batches):
            batch_path = os.path.join(val_cache_dir, f'batch_{batch_idx}.pt')
            if os.path.exists(batch_path):
                batch_data = torch.load(batch_path)
                all_batch_images.append(batch_data['images'])
                all_batch_labels.append(batch_data['labels'])
        
        if all_batch_images:
            val_data = {
                'images': torch.cat(all_batch_images, dim=0),
                'labels': torch.cat(all_batch_labels, dim=0)
            }
            torch.save(val_data, VAL_CACHE)
            print(f"验证集保存完成: {val_data['images'].shape[0]} 张图片")
        
        # 清理临时批次文件
        for batch_idx in range(num_batches):
            batch_path = os.path.join(val_cache_dir, f'batch_{batch_idx}.pt')
            if os.path.exists(batch_path):
                os.remove(batch_path)
        os.rmdir(val_cache_dir)
        
        elapsed = time.time() - start_time
        progress.complete(message=f"预处理完成: 训练集 {total_samples} 张, 验证集 {len(val_images)} 张, 耗时 {elapsed:.1f}s")
        
    except Exception as e:
        progress.error(message=f"验证集处理出错: {str(e)}")
        print(f"验证集处理出错: {e}")
        raise
    
    return total_samples, len(val_images)

# 检查缓存是否存在
if os.path.exists(TRAIN_CACHE) and os.path.exists(VAL_CACHE):
    # 统计缓存数据量
    train_files = [f for f in os.listdir(TRAIN_CACHE) if f.endswith('.pt')]
    train_count = sum(torch.load(os.path.join(TRAIN_CACHE, f))['images'].shape[0] for f in train_files)
    val_data = torch.load(VAL_CACHE)
    val_count = val_data['images'].shape[0]
    
    print(f"缓存已存在，跳过预处理")
    print(f"训练集缓存: {train_count} 张图片 ({len(train_files)} 个类别文件)")
    print(f"验证集缓存: {val_count} 张图片")
    print(f"缓存目录: {CACHE_DIR}")
else:
    # 执行预处理
    if not os.path.exists(DATA_DIR):
        print("错误: 数据集未下载，请先运行 'dmla data' 下载数据集")
    else:
        train_count, val_count = preprocess_and_cache()
```

### 数据缓存结构

预处理完成后，缓存目录结构如下：

```
/data/cache/preprocessing/tiny-imagenet-224/
├── train/
│   ├── n01443537.pt    # 类别 0 的所有预处理图片 [500, 3, 224, 224]
│   ├── n01641515.pt    # 类别 1 的所有预处理图片
│   └── ...             # 共 200 个类别文件
└── val.pt              # 验证集所有预处理图片 [10000, 3, 224, 224]
```

每个 `.pt` 文件包含：
- `images`: 预处理好的 tensor `[N, 3, 224, 224]`
- `labels`: 标签 tensor `[N]`
- `class_name`: 类别名称（仅训练集）

## 第三阶段：模型定义

以下代码实现 [AlexNet 网络结构](alexnet.md#网络结构)，除因输出分类部分要适配 Tiny ImageNet 的 200 个类别外（而非原始的 1000 类），其余网络定义与原版的 AlexNet 保持完全一致。

1. **`features`（特征提取层）：** 5 个卷积层交替叠加，逐层提取从低级（边缘、纹理）到高级（物体部件）的特征。卷积层之间的 `MaxPool2d` 负责下采样，逐步缩小空间尺寸。`AdaptiveAvgPool2d((6, 6))` 确保无论输入图像经过前面的卷积池化后尺寸如何，输出始终固定为 6×6
2. **`classifier`（分类层）：** 3 个全连接层。前两层使用 `Dropout(p=0.5)` 随机丢弃 50% 的神经元激活，防止过拟合，这是 AlexNet 的标志性设计。最后将 4096 维特征映射到 200 个类别的 Softmax 分类器
3. **输出从 1000 类改成 200 类：** 原始 AlexNet 最后一层输出 1000 类（对应完整 ImageNet），Tiny ImageNet 只有 200 类，所以 `num_classes=200`

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

本阶段执行完整的训练流程，这是整个实验的核心，模型通过多轮（epoch）迭代学习，逐步提升分类准确率。这一部分需要在 GPU 下进行，使用 ProgressReporter 报告训练进度。

::: tip 性能优化：从缓存加载预处理数据
本阶段直接从第二阶段预处理好的缓存文件加载数据，跳过耗时的 Resize、ToTensor、Normalize 操作，让 GPU 利用率从 15% 提升到 80%+。

- 训练集加载：从 `/data/cache/preprocessing/tiny-imagenet-224/train/*.pt` 加载
- 验证集加载：从 `/data/cache/preprocessing/tiny-imagenet-224/val.pt` 加载
- 数据增强：RandomHorizontalFlip、RandomCrop 等在训练时实时执行（GPU 上很快）
:::

**训练流程：** 每个 epoch 中，模型遍历全部训练数据（前向传播 → 计算损失 → 反向传播 → 更新权重），然后在验证集上评估泛化能力。

**训练关键点与超参数：**

1. **设备选择：** 检测 GPU 是否可用，GPU 训练速度比 CPU 快 10-100 倍，使用 CPU 完成本实验将非常耗时
2. **缓存数据加载：** 直接加载预处理好的 tensor，跳过 Resize 等耗时操作，大幅提升 GPU 利用率
3. **实时数据增强：** 在 GPU 上执行 RandomHorizontalFlip 和 RandomCrop（比 CPU 快得多）
4. **损失函数 `CrossEntropyLoss`：** 多分类任务的标准损失函数，计算 Softmax 后与真实标签求交叉熵
5. **优化器 `SGD`：** 随机梯度下降，`lr=0.01` 是初始学习率，`momentum=0.9` 加速收敛并抑制震荡，`weight_decay=0.0005`（L2 正则化）防止过拟合
6. **学习率调度 `StepLR`：** 每 10 个 epoch 将学习率乘以 0.1。训练初期用大学习率快速下降，后期用小学习率精细调优
7. **模型保存策略：** 保存验证准确率最高的模型（`best_model.pth`），并每 5 个 epoch 保存一次 checkpoint，防止训练中断后可恢复

```python runnable gpu timeout=unlimited
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import time

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 导入共享模块中的 AlexNet
from shared.cnn.alex_net import AlexNet

# 创建进度报告器
progress = ProgressReporter(total_steps=100, description="准备训练环境")
progress.update(0, message="正在导入模块...")

# 预处理缓存目录
CACHE_DIR = '/data/cache/preprocessing/tiny-imagenet-224'
TRAIN_CACHE = os.path.join(CACHE_DIR, 'train')
VAL_CACHE = os.path.join(CACHE_DIR, 'val.pt')

# 检查缓存是否存在
progress.update(10, message="检查预处理缓存...")
if not os.path.exists(TRAIN_CACHE) or not os.path.exists(VAL_CACHE):
    print("错误: 预处理缓存不存在，请先执行第二阶段的预处理代码")
    progress.error(message="缓存不存在")
else:
    print("预处理缓存已存在，开始加载...")

    # 加载训练集缓存
    progress.update(20, message="正在加载训练集缓存...")
    train_files = sorted([f for f in os.listdir(TRAIN_CACHE) if f.endswith('.pt')])
    
    all_train_images = []
    all_train_labels = []
    
    for i, f in enumerate(train_files):
        cls_data = torch.load(os.path.join(TRAIN_CACHE, f))
        all_train_images.append(cls_data['images'])
        all_train_labels.append(cls_data['labels'])
        
        if (i + 1) % 20 == 0 or i == len(train_files) - 1:
            progress.update(20 + int(30 * (i + 1) / len(train_files)), 
                          message=f"加载训练集缓存 {i+1}/{len(train_files)} 个类别文件")
    
    train_images = torch.cat(all_train_images, dim=0)  # [N, 3, 224, 224]
    train_labels = torch.cat(all_train_labels, dim=0)  # [N]
    
    # 加载验证集缓存
    progress.update(60, message="正在加载验证集缓存...")
    val_data = torch.load(VAL_CACHE)
    val_images = val_data['images']  # [N, 3, 224, 224]
    val_labels = val_data['labels']  # [N]
    
    progress.update(70, message=f"数据加载完成: {train_images.shape[0]} 训练样本, {val_images.shape[0]} 验证样本")
    print(f"数据加载完成: 训练集 {train_images.shape[0]} 样本, 验证集 {val_images.shape[0]} 样本")

# 检查 CUDA 可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")

# 实时数据增强（在 GPU 上执行，比 CPU 快得多）
class CachedDatasetWithAugmentation(Dataset):
    """从缓存加载的数据集，支持实时数据增强"""
    def __init__(self, images, labels, device, augment=True):
        self.images = images
        self.labels = labels
        self.device = device
        self.augment = augment
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].to(self.device)
        label = self.labels[idx].to(self.device)
        
        if self.augment:
            # RandomHorizontalFlip (50% 概率)
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[2])  # 水平翻转
            
            # RandomCrop 224 with padding=4
            if image.shape[1] == 224:  # 只有已经是 224x224 才能做 crop
                # 先 padding 4 像素
                padded = torch.nn.functional.pad(image, (4, 4, 4, 4), mode='reflect')
                # 随机裁剪 224x224
                top = torch.randint(0, 9, (1,)).item()
                left = torch.randint(0, 9, (1,)).item()
                image = padded[:, top:top+224, left:left+224]
        
        return image, label

# 创建数据集（训练集带增强，验证集不带增强）
progress.update(80, message="正在创建数据集...")
train_dataset = CachedDatasetWithAugmentation(train_images, train_labels, device, augment=True)
val_dataset = CachedDatasetWithAugmentation(val_images, val_labels, device, augment=False)

# 创建 DataLoader（数据已在 GPU 上，无需 pin_memory）
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

progress.update(85, message="正在创建模型...")
# 创建模型
model = AlexNet(num_classes=200).to(device)

progress.update(90, message="正在配置优化器...")
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 准备阶段完成，切换到训练阶段
progress.update(95, message="准备阶段完成，开始训练...")
total_batches = len(train_loader)
num_epochs = 20
progress.reset(total_steps=num_epochs * total_batches, description="训练 AlexNet on Tiny ImageNet")
current_batch = 0

# 训练函数
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, progress, current_batch, total_batches, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 数据已经在 GPU 上（由 CachedDatasetWithAugmentation 处理）
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        batch_count += 1

        # 每个 batch 更新进度
        current_batch += 1
        batch_acc = 100. * correct / total if total > 0 else 0
        progress.update(
            current_batch,
            message=f"Epoch {epoch+1}/{num_epochs} Batch {batch_idx+1}/{total_batches}: Loss={loss.item():.4f}, Acc={batch_acc:.2f}%"
        )

    return running_loss / batch_count, 100. * correct / total, current_batch

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            # 数据已经在 GPU 上
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

# 主训练循环
print("\n开始训练...")
print(f"每个 epoch 约 {total_batches} 个 batch, batch_size=128")
best_acc = 0.0
save_dir = '/data/models/alexnet/checkpoints'
os.makedirs(save_dir, exist_ok=True)

try:
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # 训练一个 epoch
        train_loss, train_acc, current_batch = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, progress, current_batch, total_batches, num_epochs
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 打印 epoch 结果
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% "
              f"Time: {epoch_time:.1f}s")
        
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

### 性能对比

| 指标 | 原方案（实时预处理） | 新方案（缓存加载） |
|------|----------------------|-------------------|
| GPU 利用率 | ~15% | ~80% |
| 吞吐量 | 370 images/sec | 预计 2000+ images/sec |
| 每 epoch 时间 | ~30 分钟 | 预计 ~5 分钟 |
| batch_size | 64 | 128（显存充足） |
| 数据增强 | CPU（慢） | GPU（快） |

## 第五阶段：模型推理

使用训练好的模型对新图像进行分类预测。训练完成后，验证模型的实际分类效果，展示模型"学到了什么"。

1. **模型加载：** 优先加载验证准确率最高的 checkpoint（`best_model.pth`），其次加载最终模型。如果都找不到，则使用未训练的随机权重模型（仅供测试，预测结果无意义）
2. **`model.eval()`：** 切换到推理模式，关闭 Dropout 和 BatchNorm 的训练行为，确保每次推理结果一致
3. **推理预处理：** 与验证集预处理相同（Resize → ToTensor → Normalize），不做数据增强。输入图像的预处理方式必须与训练时一致
4. **类别名称映射：** Tiny ImageNet 的类别标签是 WordNet ID（如 `n01675725`），通过 `wnids.txt` 和 `words.txt` 映射为可读的英文描述（如 `turtle, tortoise`）
5. **`predict_image`：** 预测图像，判断 Top-5 错误率结果
   - 读取图像 → 预处理 → 送入模型
   - 使用 `softmax` 将 logits 转为概率（0-100%）
   - `topk(5)` 取概率最高的 5 个类别，输出 Top-5 预测结果
   - Top-5 ILSVRC 图像分类的默认评估指标，只要正确答案在前 5 个预测中，就认为模型正确分类了

```python runnable gpu
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 导入共享模块中的 AlexNet
from shared.cnn.alex_net import AlexNet

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

本实验完整展示了 AlexNet 的训练流程，采用预处理缓存策略优化 GPU 利用率：

| 阶段 | 关键步骤 | 代码块 | 执行时间 |
|:-----|:---------|:-------|:---------|
| 数据准备 | 检查/下载数据集 | 常规 | - |
| 数据预处理 | 预处理并缓存到磁盘 | `timeout=unlimited` | 首次约 5-10 分钟，后续秒级加载 |
| 模型定义 | AlexNet 类定义 | `extract-class` | - |
| 模型训练 | 从缓存加载 + 训练循环 | `timeout=unlimited` | 每 epoch 约 5 分钟 |
| 模型推理 | 加载模型预测 | 常规 | - |

**性能优化要点：**

1. **预处理缓存：** Resize(64→224) 操作预先完成并保存，避免每次训练时重复处理
2. **GPU 数据增强：** RandomHorizontalFlip 和 RandomCrop 在 GPU 上执行，比 CPU 快得多
3. **增大 batch_size：** 从 64 提升到 128，充分利用显存（RTX 4060 只用了约 1.4GB）
4. **GPU 利用率：** 从约 15% 提升到 80%+，训练速度提升 5-10 倍

## 生成的文件

训练完成后，以下文件将保存到数据目录：

**预处理缓存：**
- `/data/cache/preprocessing/tiny-imagenet-224/train/*.pt` - 训练集预处理缓存（200 个类别文件）
- `/data/cache/preprocessing/tiny-imagenet-224/val.pt` - 验证集预处理缓存

**模型文件：**
- `/data/models/alexnet/checkpoints/best_model.pth` - 最佳验证准确率的模型
- `/data/models/alexnet/checkpoints/epoch_*.pth` - 每 5 epoch 的 checkpoint
- `/data/models/alexnet/final/alexnet_tiny_imagenet.pth` - 最终模型权重
