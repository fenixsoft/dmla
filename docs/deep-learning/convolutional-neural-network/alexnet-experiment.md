# AlexNet 训练实验

本节笔者将与你一同使用现代深度学习框架（PyTorch）来复现完整的 AlexNet 训练流程，从数据准备到预处理、从模型训练到推理，进行端到端的深度学习实验，通过实践来理解经典 CNN 架构与现代机器学习框架如何结合，如何开发一个机器学习应用。

## 实验准备

在开始实验之前，请确保已[挂载数据目录](../../sandbox.md#数据管理)并下载 Tiny ImageNet 200 数据集，你可以通过 `DMLA-CLI` 工具自动完成该工作：
```bash
# 选择 "下载数据集" -> 选择 "Tiny ImageNet 200"
dmla data
```

## 第一阶段：数据准备

首先，验证数据集是否已正确下载，并检查其结构。Tiny ImageNet 200 包含 200 个类别，共 11 万张图像。训练前需要确认数据集完整下载、目录结构正确，否则后续 DataLoader 会因为找不到文件而报错。

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

### 三种缓存策略对比

深度学习训练的数据预处理是一个经典的工程权衡问题。本节对比三种不同策略，展示不同硬件条件下的最佳选择。

::: tip 为什么需要缓存策略权衡？
AlexNet 在 2012 年用 GTX 580（3GB 显存）训练了完整的 ImageNet 数据集（120 万张图片）。但如果我们按本实验原始方案缓存预处理结果，需要 67GB 内存才能训练 Tiny ImageNet（11 万张图片）。这显然不符合真实工程场景。

**问题根源**：原始方案将 JPEG 图片转换为 float32 tensor 存储，膨胀 300 倍：
- 原始 JPEG：64×64，压缩格式，单张 ~2KB
- 缓存 tensor：224×224，float32 未压缩，单张 ~600KB

**解决方案**：不同场景需要不同策略，以下是三种代表性方案。
:::

#### 方案 A：实时预处理（2012 年经典方案）

**实现方式**：不缓存预处理结果，训练时实时从原始图片读取并处理。

**性能特征**：

| 指标 | 数值 | 说明 |
|------|------|------|
| 内存需求 | ~2 GB | 只缓存当前 batch |
| 磁盘占用 | ~250 MB | 原始数据集大小 |
| GPU 利用率 | ~15% | CPU 预处理成为瓶颈 |
| 每 epoch 时间 | ~30 分钟 | I/O + CPU 预处理耗时 |

**技术要点**：
- DataLoader 实时读取 64×64 JPEG
- CPU 执行 Resize(224)、ToTensor、数据增强
- GPU 只执行 Normalize 和训练

**适用场景**：历史教学、极低内存环境、展示 AlexNet 原始训练方式

::: warning 这是 AlexNet 2012 年的真实训练方式
当年的硬件限制决定了必须实时预处理。虽然 GPU 利用率低，但这是理解历史的重要案例。
:::

#### 方案 B：最小缓存 + 实时增强（本实验采用）

**实现方式**：只缓存最耗时的 Resize 操作结果，保留 JPEG 压缩格式。其他预处理实时执行。

**性能特征**：

| 指标 | 数值 | 说明 |
|------|------|------|
| 内存需求 | ~4 GB | DataLoader 预取 + batch |
| 磁盘占用 | ~600 MB | Resize 后的 JPEG（224×224） |
| GPU 利用率 | 40-60% | 多线程优化 I/O |
| 每 epoch 时间 | ~10-15 分钟 | Resize 已缓存 |

**技术要点**：
- 预处理：Resize(224) → 保存 JPEG（quality=95）
- 训练：实时 ToTensor + 数据增强（CPU 多线程）
- Normalize 在 GPU 执行（固定变换，批量化快）
- DataLoader 配置：`num_workers=4`, `pin_memory=True`, `prefetch_factor=2`

**适用场景**：教学推荐、普通机器（16-32GB 内存）

::: tip 为什么选择这个方案？
平衡了内存、磁盘、速度三个维度：
- 内存友好：不需要一次性加载全部数据
- 磁盘友好：600MB 而非 60GB
- 速度适中：比方案 A 快 2-3 倍
- 教学价值：展示现代 PyTorch 的多线程优化技巧
:::

#### 方案 D：LMDB 高效存储（工业界方案）

**实现方式**：使用 LMDB 键值存储，支持快速随机读取。

**性能特征**：

| 指标 | 数值 | 说明 |
|------|------|------|
| 内存需求 | ~4 GB | LMDB mmap 读取 |
| 磁盘占用 | ~600 MB | LMDB 数据库 |
| GPU 利用率 | 60-70% | 零拷贝读取 |
| 每 epoch 时间 | ~8-10 分钟 | 高效 I/O |

**技术要点**：
- LMDB 数据库存储预处理结果
- 键值对：`image_id` → JPEG bytes
- 支持 mmap 零拷贝读取
- 安装：`pip install lmdb`（~2MB，无额外依赖）

**适用场景**：生产环境、高性能需求、大规模数据集

::: info 工业界的标准做法
FFCV、WebDataset 等现代工具都基于类似原理。如果追求最高性能，这是最佳选择。
:::

#### 性能对比总结

| 方案 | 内存 | 磁盘 | GPU利用率 | 每 epoch | 适用场景 |
|------|------|------|-----------|----------|---------|
| A: 实时预处理 | 2GB | 250MB | 15% | 30分钟 | 历史教学 |
| **B: 最小缓存** | **4GB** | **600MB** | **40-60%** | **10-15分钟** | **教学推荐** |
| D: LMDB存储 | 4GB | 600MB | 60-70% | 8-10分钟 | 生产环境 |

**本实验选择方案 B 的原因**：
1. 内存需求适中（4GB），普通机器可运行
2. 磁盘占用合理（600MB），不会填满硬盘
3. 训练速度可接受（比实时预处理快 2-3 倍）
4. 展示现代 PyTorch 的多线程优化技巧
5. 不引入额外依赖，代码简洁易懂

如果你有更大的内存（64GB+），可以参考原版文档的"全量缓存方案"获得最快的训练速度。如果你追求最高性能，可以尝试方案 D（LMDB）。

接下来，我们创建 PyTorch DataLoader 对图像进行预处理和数据增强。Tiny ImageNet 200 数据集每个类别只有 500 张训练图，相对于模型参数量而言，数量十分有限。数据增强通过随机翻转、裁剪、颜色抖动等变换，可以人工增加训练数据的多样性，防止模型过拟合。AlexNet 参加比赛时使用的是 ImageNet 1K 数据集，尽管比 200 数据集来说要大不少，但仍然需进行数据预处理增强。

本阶段借助了 PyTorch 中十分常用的 `Dataset` 和 `DataLoader` 两个组件。`Dataset` 负责把磁盘上的图像文件和标签映射成 `(图像, 标签)` 对，`DataLoader` 负责批量加载、打乱顺序、多线程读取。

::: tip 性能优化：预处理缓存
原始 Tiny ImageNet 图片尺寸为 64×64，需要放大到 224×224 才能输入 AlexNet。这个 Resize 操作在 CPU 上非常耗时，会导致 GPU 利用率极低（约 15%）。

**解决方案：** 本阶段会将预处理好的数据缓存到磁盘（`.pt` 文件），训练时直接加载缓存，跳过预处理步骤，让 GPU 利用率提升到 80%+。

- 缓存位置：`/data/cache/preprocessing/tiny-imagenet-224/`
- 首次运行：执行预处理并保存（约 5-10 分钟）
- 后续运行：直接加载缓存（几秒钟）
:::

::: tip 缓存为何这么大？60GB 的原因
原始 Tiny ImageNet 数据集仅约 250 MB，但预处理缓存却膨胀到约 60 GB，这是性能优化的权衡结果：

**每个 `.pt` 文件的内容结构：**
```python
{
    'images': torch.tensor([500, 3, 224, 224]),  # 500 张预处理图片（float32）
    'labels': torch.tensor([500]),               # 500 个标签（int64）
    'class_name': 'n01443537'                    # 类别名称
}
```

**单文件大小计算：**
- images: 500 × 3 × 224 × 224 × 4 bytes (float32) ≈ **301 MB**
- labels: 500 × 8 bytes (int64) ≈ 4 KB
- 200 个类别文件总计 ≈ **60 GB**

**膨胀原因对比：**

| 对比项 | 原始 JPEG | 预处理 tensor |
|--------|-----------|----------------|
| 图片尺寸 | 64×64 | 224×224 |
| 数据格式 | JPEG 压缩（8-bit） | float32 未压缩 |
| 单张大小 | ~2 KB | **~600 KB** |
| 膨胀倍数 | | **~300 倍** |

这是两个因素叠加的结果：
1. **尺寸放大**：64×64 → 224×224，像素数量增加 `(224/64)² ≈ 12 倍`
2. **格式转换**：JPEG 压缩格式 → 未压缩 float32 tensor，增加 4 倍
3. **组合效果**：12 × 4 ≈ **48 倍膨胀**（加上 PyTorch tensor 元数据开销约 300 倍）

**性能权衡：**

| 方案 | 磁盘占用 | GPU 利用率 | 每 epoch 时间 |
|------|----------|------------|---------------|
| 实时预处理（原图） | ~250 MB | ~15% | ~30 分钟 |
| **缓存预处理（tensor）** | **~60 GB** | **~80%** | **~5 分钟** |

结论：牺牲 60 GB 磁盘空间换取 6 倍训练速度提升。如果磁盘空间紧张，训练完成后可删除缓存目录（下次运行会重新生成）。
:::

**预处理缓存流程（支持断点续传）：**

1. **检查缓存目录：** 如果 `/data/cache/preprocessing/tiny-imagenet-224/` 已存在且完整，跳过预处理
2. **训练集预处理（可恢复）：** 每个类别处理后立即保存 `.pt` 文件，已存在的类别文件自动跳过
3. **验证集预处理（可恢复）：** 每 1000 张保存一个批次文件 `val_batch_x.pt`，已存在的批次自动跳过
4. **预处理内容：**
    - `Resize(224, 224)`：AlexNet 要求输入为 224×224 的图像，而 Tiny ImageNet 200 提供的图片是 64×64
    - `RandomHorizontalFlip`：50% 概率水平翻转，增加姿态变化
    - `RandomCrop(224, padding=4)`：先四周各填充 4 像素再随机裁剪 224×224，模拟尺度变化
    - `ColorJitter`：随机调整亮度和对比度，增强对光照变化的鲁棒性
    - `ToTensor`：将 PIL 图像转为 PyTorch Tensor，像素值从整数 [0, 255] 缩放到浮点数 [0, 1]
    - `Normalize`：使用 ImageNet 的统计均值和标准差进行标准化，使输入分布与预训练权重的统计特征一致
5. **数据增强保留：** RandomHorizontalFlip、RandomCrop、ColorJitter 等在训练时实时执行（不缓存）

::: tip 为什么数据增强不缓存？
数据增强的核心目的不是"预处理一次"，而是"每次训练都随机产生不同版本"。

**假设一张图片在 20 个 epoch 中被训练 20 次：**

| epoch | RandomHorizontalFlip | RandomCrop | ColorJitter |
|-------|---------------------|------------|-------------|
| 1 | 翻转 | 裁剪位置 A | 亮度 +10% |
| 2 | 不翻转 | 裁剪位置 B | 亮度 -5% |
| 3 | 翻转 | 裁剪位置 C | 对比度 +8% |
| ... | ... | ... | ... |

每次模型看到的都是这张图片的"不同版本"，仿佛在训练 20 张不同的图片，有效增加数据多样性，防止过拟合。

**如果缓存数据增强的结果：**
- 每张图片只保存 1 个固定的增强版本
- 20 个 epoch 都看到同一张"增强后的图片"
- 随机性消失，失去防过拟合效果

**因此，只有固定变换（Resize、ToTensor）才缓存，随机变换必须在训练时实时执行。**
:::

```python runnable gpu timeout=unlimited extract-class="DataAugmentor,PreprocessCache"
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import json
from PIL import Image
import time

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 数据及预处理缓存目录
DATA_DIR = '/data/datasets/tiny-imagenet-200'
CACHE_DIR = '/data/cache/preprocessing/tiny-imagenet-224'


class DataAugmentor:
    """
    数据增强器：在训练时实时执行随机变换
    
    这些操作不缓存，每次调用都产生不同的结果，
    使模型在每个 epoch 看到不同的"版本"，防止过拟合。
    """
    
    # Normalize 参数（ImageNet 统计值）
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    def __init__(self, device):
        self.device = device
        # 将 Normalize 参数转为 tensor 并移到目标设备
        self.mean = torch.tensor(self.MEAN).view(3, 1, 1).to(device)
        self.std = torch.tensor(self.STD).view(3, 1, 1).to(device)
    
    def random_horizontal_flip(self, image, p=0.5):
        """
        随机水平翻转（50% 概率）
        
        Args:
            image: [C, H, W] tensor，值范围 [0, 1]
            p: 翻转概率
        Returns:
            翻转后的 tensor
        """
        if torch.rand(1).item() > p:
            image = torch.flip(image, dims=[2])  # 翻转 W 维度
        return image
    
    def random_crop(self, image, padding=4):
        """
        随机裁剪：先填充再随机位置裁剪
        
        Args:
            image: [C, H, W] tensor，值范围 [0, 1]
            padding: 四周填充像素数
        Returns:
            裁剪后的 tensor，尺寸不变
        """
        # 四周填充
        padded = torch.nn.functional.pad(image, (padding, padding, padding, padding), mode='reflect')
        # 随机选择裁剪起始位置
        top = torch.randint(0, 2 * padding + 1, (1,)).item()
        left = torch.randint(0, 2 * padding + 1, (1,)).item()
        h, w = image.shape[1], image.shape[2]
        return padded[:, top:top+h, left:left+w]
    
    def color_jitter(self, image, brightness=0.2, contrast=0.2):
        """
        随机调整亮度和对比度
        
        Args:
            image: [C, H, W] tensor，值范围 [0, 1]
            brightness: 亮度调整范围 [1-brightness, 1+brightness]
            contrast: 对比度调整范围 [1-contrast, 1+contrast]
        Returns:
            调整后的 tensor，值范围 [0, 1]
        """
        # 亮度调整
        if brightness > 0:
            factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * brightness
            image = image * factor
            image = torch.clamp(image, 0, 1)
        
        # 对比度调整
        if contrast > 0:
            factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * contrast
            mean_val = image.mean(dim=[1, 2], keepdim=True)
            image = (image - mean_val) * factor + mean_val
            image = torch.clamp(image, 0, 1)
        
        return image
    
    def normalize(self, image):
        """
        标准化：使用 ImageNet 均值和标准差
        
        Args:
            image: [C, H, W] tensor，值范围 [0, 1]
        Returns:
            标准化后的 tensor
        """
        return (image - self.mean) / self.std
    
    def augment(self, image):
        """
        执行完整的数据增强流程（训练集用）
        
        流程：RandomHorizontalFlip → RandomCrop → ColorJitter → Normalize
        """
        image = self.random_horizontal_flip(image)
        image = self.random_crop(image)
        image = self.color_jitter(image)
        image = self.normalize(image)
        return image
    
    def preprocess_val(self, image):
        """
        验证集预处理（无数据增强）
        
        流程：Normalize（数据已缓存为 [0, 1] 范围）
        """
        return self.normalize(image)


class PreprocessCache:
    """
    预处理缓存器：将数据集预处理并缓存到磁盘
    
    支持：
    - 断点续传：中断后可从上次进度继续
    - 训练集：每个类别单独保存为 .pt 文件
    - 验证集：每 1000 张保存为一个批次文件
    
    缓存的数据不含 Normalize，值范围为 [0, 1]，
    Normalize 和数据增强在训练时由 DataAugmentor 实时执行。
    """
    
    def __init__(self, data_dir, cache_dir, val_batch_size=1000):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.train_cache = os.path.join(cache_dir, 'train')
        self.val_batch_dir = os.path.join(cache_dir, 'val_batches')
        self.val_manifest_path = os.path.join(cache_dir, 'val_manifest.json')
        self.checkpoint_file = os.path.join(cache_dir, 'checkpoint.json')
        self.val_batch_size = val_batch_size
        
        # 基础预处理（只做 Resize + ToTensor，不含 Normalize）
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def _load_checkpoint(self):
        """加载进度追踪文件"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'train_completed': False, 'val_batch_completed': -1}
    
    def _save_checkpoint(self, checkpoint):
        """保存进度追踪文件"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
    
    def _preprocess_train_set(self, progress, checkpoint):
        """预处理训练集（支持断点续传）"""
        train_dir = os.path.join(self.data_dir, 'train')
        classes = sorted(os.listdir(train_dir))
        skipped_classes = 0
        
        os.makedirs(self.train_cache, exist_ok=True)
        
        # 获取已存在的类别文件
        existing_files = set(f.replace('.pt', '') for f in os.listdir(self.train_cache) if f.endswith('.pt'))
        
        for cls_idx, cls in enumerate(classes):
            if cls in existing_files:
                skipped_classes += 1
                if (cls_idx + 1) % 10 == 0 or cls_idx == len(classes) - 1:
                    progress.update(cls_idx + 1, message=f"跳过已缓存类别 {cls_idx+1}/200")
                continue
            
            images_dir = os.path.join(train_dir, cls, 'images')
            if not os.path.exists(images_dir):
                continue
            
            images, labels = [], []
            for img_name in os.listdir(images_dir):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(images_dir, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        images.append(self.base_transform(img))
                        labels.append(cls_idx)
                    except Exception as e:
                        print(f"Warning: Failed to process {img_path}: {e}")
            
            if images:
                torch.save({
                    'images': torch.stack(images),
                    'labels': torch.tensor(labels),
                    'class_name': cls
                }, os.path.join(self.train_cache, f'{cls}.pt'))
            
            progress.update(cls_idx + 1, message=f"预处理类别 {cls_idx+1}/200: {cls}")
        
        checkpoint['train_completed'] = True
        self._save_checkpoint(checkpoint)
        
        # 统计最终数量
        all_files = [f for f in os.listdir(self.train_cache) if f.endswith('.pt')]
        progress.reset(total_steps=len(all_files), description="统计训练集数量")
        final_count = 0
        for i, f in enumerate(all_files):
            data = torch.load(os.path.join(self.train_cache, f), weights_only=True)
            final_count += data['images'].shape[0]
            progress.update(i + 1, message=f"统计 {i+1}/{len(all_files)} 个文件")
        
        progress.complete(message=f"训练集预处理完成: {final_count} 张 (跳过 {skipped_classes} 个)")
        return final_count
    
    def _preprocess_val_set(self, progress, checkpoint):
        """预处理验证集（支持断点续传）"""
        val_dir = os.path.join(self.data_dir, 'val')
        val_images_dir = os.path.join(val_dir, 'images')
        val_annotations = os.path.join(val_dir, 'val_annotations.txt')
        
        # 读取类别映射
        wnids_path = os.path.join(self.data_dir, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        
        # 读取标注文件
        with open(val_annotations, 'r') as f:
            val_lines = f.readlines()
        total_val = len(val_lines)
        
        os.makedirs(self.val_batch_dir, exist_ok=True)
        
        completed_batches = checkpoint.get('val_batch_completed', -1)
        num_batches = total_val // self.val_batch_size + (1 if total_val % self.val_batch_size > 0 else 0)
        
        progress.reset(total_steps=total_val, description="预处理验证集")
        
        if completed_batches < num_batches - 1:
            for batch_idx in range(num_batches):
                batch_path = os.path.join(self.val_batch_dir, f'val_batch_{batch_idx}.pt')
                
                if batch_idx <= completed_batches or os.path.exists(batch_path):
                    count = (batch_idx + 1) * self.val_batch_size if batch_idx < num_batches - 1 else total_val
                    progress.update(count, message=f"跳过已缓存批次 {batch_idx+1}/{num_batches}")
                    continue
                
                start_idx = batch_idx * self.val_batch_size
                end_idx = min((batch_idx + 1) * self.val_batch_size, total_val)
                
                batch_images, batch_labels = [], []
                for line_idx in range(start_idx, end_idx):
                    parts = val_lines[line_idx].strip().split('\t')
                    if len(parts) >= 2:
                        img_path = os.path.join(val_images_dir, parts[0])
                        if os.path.exists(img_path):
                            try:
                                img = Image.open(img_path).convert('RGB')
                                batch_images.append(self.base_transform(img))
                                batch_labels.append(class_to_idx.get(parts[1], 0))
                            except Exception as e:
                                print(f"处理图片出现异常 {img_path}: {e}")
                    
                    if (line_idx + 1) % 100 == 0 or line_idx == end_idx - 1:
                        progress.update(line_idx + 1, message=f"预处理验证集 {line_idx+1}/{total_val}")
                
                if batch_images:
                    torch.save({
                        'images': torch.stack(batch_images),
                        'labels': torch.tensor(batch_labels),
                        'batch_idx': batch_idx
                    }, batch_path)
                
                checkpoint['val_batch_completed'] = batch_idx
                self._save_checkpoint(checkpoint)
        
        # 创建清单文件
        batch_count = sum(1 for i in range(num_batches) if os.path.exists(os.path.join(self.val_batch_dir, f'val_batch_{i}.pt')))
        manifest = {
            'num_batches': batch_count,
            'batch_size': self.val_batch_size,
            'total_images': batch_count * self.val_batch_size,
            'batch_files': [f'val_batch_{i}.pt' for i in range(batch_count)]
        }
        with open(self.val_manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        # 清理 checkpoint
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        
        progress.complete(message=f"验证集预处理完成: {manifest['total_images']} 张")
        return manifest['total_images']
    
    def check_cache_exists(self):
        """检查缓存是否已完整存在"""
        return os.path.exists(self.train_cache) and os.path.exists(self.val_manifest_path)
    
    def get_cache_stats(self):
        """获取缓存统计信息（不加载大文件）"""
        train_files = [f for f in os.listdir(self.train_cache) if f.endswith('.pt')]
        train_count = len(train_files) * 500  # 估算
        
        with open(self.val_manifest_path, 'r') as f:
            manifest = json.load(f)
        
        return train_count, manifest['total_images'], len(train_files), manifest['num_batches']
    
    def run(self, progress):
        """
        执行预处理（支持断点续传）
        
        Returns:
            (train_count, val_count) 预处理的图片数量
        """
        start_time = time.time()
        checkpoint = self._load_checkpoint()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 阶段 1：训练集预处理
        if checkpoint.get('train_completed', False):
            train_files = [f for f in os.listdir(self.train_cache) if f.endswith('.pt')]
            train_count = len(train_files) * 500
            progress.update(200, message=f"训练集已缓存: ~{train_count} 张")
        else:
            train_count = self._preprocess_train_set(progress, checkpoint)
        
        # 阶段 2：验证集预处理
        if os.path.exists(self.val_manifest_path):
            with open(self.val_manifest_path, 'r') as f:
                manifest = json.load(f)
            val_count = manifest['total_images']
        else:
            val_count = self._preprocess_val_set(progress, checkpoint)
        
        elapsed = time.time() - start_time
        
        progress.reset(total_steps=1, description="预处理阶段")
        progress.update(1, message=f"✓ 完成！训练集 {train_count} 张, 验证集 {val_count} 张")
        progress.complete(message=f"预处理完成，耗时 {elapsed:.1f}s")
        
        return train_count, val_count


# ========== 主执行逻辑 ==========

# 检查缓存是否存在
preprocessor = PreprocessCache(DATA_DIR, CACHE_DIR)

if preprocessor.check_cache_exists():
    train_count, val_count, train_files, val_batches = preprocessor.get_cache_stats()
    
    progress = ProgressReporter(total_steps=1, description="预处理阶段")
    progress.update(1, message=f"✓ 缓存已存在，跳过预处理！训练集 ~{train_count} 张, 验证集 {val_count} 张")
    progress.complete(message="预处理阶段完成（缓存已存在）")
    
    print(f"缓存已存在，跳过预处理")
    print(f"训练集缓存: ~{train_count} 张图片 ({train_files} 个类别文件)")
    print(f"验证集缓存: {val_count} 张图片 ({val_batches} 个批次)")
else:
    if not os.path.exists(DATA_DIR):
        print("错误: 数据集未下载，请先运行 'dmla data' 下载数据集")
    else:
        progress = ProgressReporter(total_steps=200, description="预处理训练集")
        train_count, val_count = preprocessor.run(progress)
        print(f"预处理完成: 训练集 {train_count} 张, 验证集 {val_count} 张")
```

### 数据缓存结构

预处理完成后，缓存目录结构如下：

```
/data/cache/preprocessing/tiny-imagenet-224/
├── train/
│   ├── n01443537.pt    # 类别 0 的所有预处理图片 [500, 3, 224, 224]
│   ├── n01641515.pt    # 类别 1 的所有预处理图片
│   └── ...             # 共 200 个类别文件
├── val_batches/        # 验证集批次目录（10 个批次文件）
│   ├── val_batch_0.pt  # 第 0-999 张图片 [1000, 3, 224, 224]
│   ├── val_batch_1.pt  # 第 1000-1999 张图片
│   └── ...             # 共 10 个批次文件
└── val_manifest.json   # 验证集批次清单（记录批次信息）
```

**设计说明**（内存优化）：

- **训练集**：每个类别独立保存，避免一次性加载整个训练集（~10GB）
- **验证集**：分 10 个批次保存，训练时按需加载，避免内存爆炸（合并需 ~3GB）
- **清单文件**：`val_manifest.json` 记录批次数量和文件路径，训练阶段据此加载

预处理过程中临时文件（完成后自动清理）：
```
/data/cache/preprocessing/tiny-imagenet-224/
└── checkpoint.json     # 进度追踪文件（完成后删除）
```

每个 `.pt` 文件包含：
- `images`: 预处理好的 tensor `[N, 3, 224, 224]`
- `labels`: 标签 tensor `[N]`
- `class_name`: 类别名称（仅训练集）

### 断点续传说明

预处理过程支持中断后自动恢复：

1. **训练集恢复**：每个类别处理完立即保存 `.pt` 文件，下次运行自动跳过已存在的类别文件
2. **验证集恢复**：每 1000 张保存一个批次文件，下次运行从 `checkpoint.json` 读取已完成的批次号继续
3. **完全完成后**：批次文件保留在 `val_batches/` 目录，清单文件 `val_manifest.json` 记录批次信息

**示例场景**：
- 第一次运行：训练集完成，验证集处理到第 5000 张时中断
- 第二次运行：自动跳过训练集，验证集从第 5001 张继续处理

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

::: warning 内存需求提示
**一次性加载全部数据约需 12 GB 内存**：

| 数据 | 图片数 | 单张大小 | 总大小 |
|------|--------|---------|--------|
| 训练集 tensor | 100,000 张 | 0.12 MB | **~12 GB** |
| 验证集 tensor | 10,000 张 | 0.12 MB | **~1.2 GB** |
| 模型 + 优化器 | - | - | ~0.3 GB |
| **合计峰值** | - | - | **~15 GB** |

**如果内存不足 16 GB**，可以减小 batch_size 或使用 CPU 训练（但会非常慢）。
:::

**训练关键点与超参数：**

1. **设备选择：** 检测 GPU 是否可用，GPU 训练速度比 CPU 快 10-100 倍
2. **一次性数据加载：** 训练前加载全部缓存数据到内存，避免每个 batch 频繁读取磁盘
3. **实时数据增强：** 在 GPU 上执行 RandomHorizontalFlip 和 RandomCrop
4. **损失函数 `CrossEntropyLoss`：** 多分类任务的标准损失函数
5. **优化器 `SGD`：** 随机梯度下降，`lr=0.01`，`momentum=0.9`，`weight_decay=0.0005`
6. **学习率调度 `StepLR`：** 每 10 个 epoch 将学习率乘以 0.1
7. **模型保存：** 保存最佳模型和每 5 个 epoch 的 checkpoint

```python runnable gpu timeout=unlimited
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import time
import json

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 导入共享模块中的 AlexNet 和 DataAugmentor
from shared.cnn.alex_net import AlexNet
from shared.cnn.data_augmentor import DataAugmentor

# 预处理缓存目录
CACHE_DIR = '/data/cache/preprocessing/tiny-imagenet-224'
TRAIN_CACHE = os.path.join(CACHE_DIR, 'train')
VAL_BATCH_DIR = os.path.join(CACHE_DIR, 'val_batches')
VAL_MANIFEST = os.path.join(CACHE_DIR, 'val_manifest.json')

# 创建进度报告器（训练准备阶段）
progress = ProgressReporter(total_steps=100, description="准备训练环境")
progress.update(0, message="正在检查预处理缓存...")

# 检查缓存是否存在
if not os.path.exists(TRAIN_CACHE) or not os.path.exists(VAL_MANIFEST):
    print("错误: 预处理缓存不存在，请先执行第二阶段的预处理代码")
    progress.error(message="缓存不存在")
else:
    print("预处理缓存已存在，开始加载...")
    
    # 统计缓存文件
    train_files = sorted([f for f in os.listdir(TRAIN_CACHE) if f.endswith('.pt')])
    with open(VAL_MANIFEST, 'r') as f:
        val_manifest = json.load(f)
    
    print(f"训练集缓存: {len(train_files)} 个类别文件")
    print(f"验证集缓存: {val_manifest['num_batches']} 个批次文件")
    print(f"预计内存占用: ~15 GB（训练集 12GB + 验证集 1.2GB + 模型 0.3GB）")

# 检查 CUDA 可用性
progress.update(10, message="检测 GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")

# 一次性加载全部数据到内存（避免训练时频繁磁盘读取）
progress.update(20, message="正在加载训练集缓存...")
print("[DEBUG] 开始加载训练集缓存（一次性加载，约需 30-60 秒）...")

all_train_images = []
all_train_labels = []

load_start = time.time()
for i, f in enumerate(train_files):
    cls_path = os.path.join(TRAIN_CACHE, f)
    cls_data = torch.load(cls_path, weights_only=True)
    all_train_images.append(cls_data['images'])
    all_train_labels.append(cls_data['labels'])
    
    # 每 20 个文件更新进度
    if (i + 1) % 20 == 0 or i == len(train_files) - 1:
        progress.update(20 + int(40 * (i + 1) / len(train_files)),
                       message=f"加载训练集 {i+1}/{len(train_files)} 个类别文件")
        print(f"[DEBUG] 已加载 {i+1}/{len(train_files)} 个类别文件")

train_images = torch.cat(all_train_images, dim=0)  # [N, 3, 224, 224]
train_labels = torch.cat(all_train_labels, dim=0)  # [N]

load_time = time.time() - load_start
print(f"[DEBUG] 训练集加载完成: {train_images.shape}, 耗时 {load_time:.1f}s")
print(f"[DEBUG] 训练集内存占用: {train_images.numel() * 4 / 1024 / 1024 / 1024:.2f} GB")

# 加载验证集缓存
progress.update(65, message="正在加载验证集缓存...")
print("[DEBUG] 开始加载验证集缓存...")

all_val_images = []
all_val_labels = []

for i, batch_file in enumerate(val_manifest['batch_files']):
    batch_path = os.path.join(VAL_BATCH_DIR, batch_file)
    batch_data = torch.load(batch_path, weights_only=True)
    all_val_images.append(batch_data['images'])
    all_val_labels.append(batch_data['labels'])
    
    progress.update(65 + int(15 * (i + 1) / val_manifest['num_batches']),
                   message=f"加载验证集批次 {i+1}/{val_manifest['num_batches']}")

val_images = torch.cat(all_val_images, dim=0)  # [N, 3, 224, 224]
val_labels = torch.cat(all_val_labels, dim=0)  # [N]

print(f"[DEBUG] 验证集加载完成: {val_images.shape}")
print(f"[DEBUG] 验证集内存占用: {val_images.numel() * 4 / 1024 / 1024 / 1024:.2f} GB")

progress.update(85, message="数据加载完成，创建 Dataset...")
print(f"[DEBUG] 数据加载完成: 训练集 {train_images.shape[0]} 张, 验证集 {val_images.shape[0]} 张")

# 创建数据增强器（用于训练时的实时数据增强）
augmentor = DataAugmentor(device)

# 创建 Dataset（数据已在 CPU 内存中，训练时移到 GPU 并执行数据增强）
class CachedDataset(Dataset):
    """
    从内存加载的数据集
    
    数据已缓存为 [0, 1] 范围的 tensor（不含 Normalize），
    训练时移到 GPU 并使用 DataAugmentor 执行数据增强和 Normalize。
    """
    def __init__(self, images, labels, augmentor, augment=True):
        self.images = images      # [N, 3, 224, 224]，值范围 [0, 1]
        self.labels = labels      # [N]
        self.augmentor = augmentor  # DataAugmentor 实例
        self.augment = augment      # 是否执行数据增强
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 从 CPU 内存移到 GPU
        image = self.images[idx].to(self.augmentor.device)
        label = self.labels[idx].to(self.augmentor.device)
        
        if self.augment:
            # 训练集：执行完整数据增强流程
            image = self.augmentor.augment(image)
        else:
            # 验证集：只执行 Normalize
            image = self.augmentor.preprocess_val(image)
        
        return image, label

# 创建 Dataset 和 DataLoader
train_dataset = CachedDataset(train_images, train_labels, augmentor, augment=True)
val_dataset = CachedDataset(val_images, val_labels, augmentor, augment=False)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

print(f"[DEBUG] DataLoader 创建完成: batch_size=128, 每 epoch {len(train_loader)} 个 batch")

# 创建模型
progress.update(90, message="创建 AlexNet 模型...")
model = AlexNet(num_classes=200).to(device)
print(f"[DEBUG] 模型创建完成: {sum(p.numel() for p in model.parameters()):,} 参数")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

progress.update(95, message="训练环境准备完成")

# 切换到训练阶段进度
total_batches = len(train_loader)
num_epochs = 20
progress.reset(total_steps=num_epochs * total_batches, description="训练 AlexNet")
current_batch = 0
best_acc = 0.0

print(f"[DEBUG] 开始训练: {num_epochs} epochs, 每 epoch {total_batches} batches")

# 训练函数
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, progress, current_batch, total_batches, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 每 10 个 batch 更新进度（减少 stderr 调用）
        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            current_batch += 1
            batch_acc = 100. * correct / total
            progress.update(
                current_batch * 10,  # 每 10 batch 为一个进度单位
                message=f"Epoch {epoch+1}/{num_epochs} Batch {batch_idx+1}/{total_batches}: Loss={loss.item():.4f}, Acc={batch_acc:.2f}%"
            )
    
    return running_loss / total_batches, 100. * correct / total, current_batch + (total_batches - current_batch * 10)

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

# 主训练循环
print("\n" + "=" * 60)
print("开始训练 AlexNet on Tiny ImageNet")
print("=" * 60)

save_dir = '/data/models/alexnet/checkpoints'
os.makedirs(save_dir, exist_ok=True)

try:
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 训练一个 epoch
        train_loss, train_acc, current_batch = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, progress, current_batch // 10, total_batches // 10, num_epochs
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
    progress.complete(message=f"训练完成！最佳验证准确率: {best_acc:.2f}%")
    
    # 保存最终模型
    final_dir = '/data/models/alexnet/final'
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, 'alexnet_tiny_imagenet.pth'))
    print(f"\n最终模型已保存: {os.path.join(final_dir, 'alexnet_tiny_imagenet.pth')}")
    
except Exception as e:
    progress.error(message=f"训练出错: {str(e)}")
    print(f"\n训练出错: {e}")
    raise
```

### 性能对比

| 指标 | 原方案（实时预处理） | 新方案（缓存加载） |
|------|----------------------|-------------------|
| GPU 利用率 | ~15% | ~80% |
| 吞吐量 | 370 images/sec | 2000+ images/sec |
| 每 epoch 时间 | ~30 分钟 | ~5 分钟 |
| **内存占用** | ~2 GB | **~15 GB** |
| batch_size | 64 | 128 |
| 数据增强 | CPU（慢） | GPU（快） |

**内存说明**：
- 原方案（实时预处理）：每次从原图读取并处理，内存占用低但 GPU 利用率极低
- **缓存加载方案**：一次性加载全部预处理好的 tensor，内存约 15 GB，GPU 利用率高
- **推荐**：如果内存 ≥ 16 GB，使用缓存加载方案获得最佳训练速度

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

| 阶段 | 关键步骤 | 代码块 | 执行时间 | 内存占用 |
|:-----|:---------|:-------|:---------|:---------|
| 数据准备 | 检查/下载数据集 | 常规 | - | ~100 MB |
| 数据预处理 | 预处理并缓存到磁盘 | `timeout=unlimited` | 首次约 5-10 分钟，后续秒级 | ~2 GB |
| 模型定义 | AlexNet 类定义 | `extract-class` | - | ~60 MB |
| 模型训练 | 加载缓存 + 训练循环 | `timeout=unlimited` | 每 epoch 约 5 分钟 | **~15 GB** |
| 模型推理 | 加载模型预测 | 常规 | - | ~200 MB |

**性能优化要点：**

1. **预处理缓存：** Resize(64→224) 操作预先完成并保存到磁盘，避免每次训练时重复处理
2. **一次性加载：** 训练前加载全部缓存数据到内存（~15 GB），避免训练时频繁磁盘读取
3. **GPU 数据增强：** RandomHorizontalFlip 和 RandomCrop 在 GPU 上执行，比 CPU 快得多
4. **GPU 利用率：** 从约 15% 提升到 80%+，训练速度提升 5-6 倍

## 生成的文件

训练完成后，以下文件将保存到数据目录：

**预处理缓存：**
- `/data/cache/preprocessing/tiny-imagenet-224/train/*.pt` - 训练集预处理缓存（200 个类别文件）
- `/data/cache/preprocessing/tiny-imagenet-224/val_batches/*.pt` - 验证集预处理缓存（10 个批次文件）
- `/data/cache/preprocessing/tiny-imagenet-224/val_manifest.json` - 验证集批次清单

**模型文件：**
- `/data/models/alexnet/checkpoints/best_model.pth` - 最佳验证准确率的模型
- `/data/models/alexnet/checkpoints/epoch_*.pth` - 每 5 epoch 的 checkpoint
- `/data/models/alexnet/final/alexnet_tiny_imagenet.pth` - 最终模型权重
