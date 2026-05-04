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
- DataLoader 配置：`num_workers=0`, `pin_memory=True`（单线程避免 Docker shm 限制）

**适用场景**：教学推荐、普通机器（16-32GB 内存）

::: tip 为什么选择这个方案？
平衡了内存、磁盘、速度三个维度：
- 内存友好：不需要一次性加载全部数据
- 磁盘友好：600MB 而非 60GB
- 速度适中：比方案 A 快 2-3 倍
- 教学价值：展示现代 PyTorch 的多线程优化技巧
:::

#### 方案 D：LMDB + PyTorch DataLoader 多线程（推荐方案）

**实现方式**：使用 LMDB 键值存储 + PyTorch DataLoader 多线程数据加载。通过 Docker shm 配置启用多线程解码：

**性能特征**：

| 指标 | 数值 | 说明 |
|------|------|------|
| 内存需求 | ~2 GB | GPU batch + 模型 |
| 磁盘占用 | ~600 MB | LMDB 数据库 |
| GPU 利用率 | ~40-60% | 多线程 DataLoader 消除 I/O 瓶颈 |
| 每 epoch 时间 | ~1-2 分钟 | DataLoader num_workers=4 |

**技术要点**：
- LMDB 数据库存储预处理结果（mmap 零拷贝读取）
- PyTorch DataLoader 多线程 JPEG 解码（num_workers=4）
- Docker shm 配置 1GB（`dmla start --gpu` 自动设置）
- 数据增强在 CPU 上并行执行（RandomFlip、Crop、Normalize）
- 跨平台兼容：Windows 和 Linux Docker 均可运行
- 安装：`pip install lmdb`

**适用场景**：教学实验、生产环境、高性能需求

::: info Docker shm 配置
PyTorch DataLoader 多线程需要足够的共享内存（shm）。默认 Docker shm 只有 64MB，会导致 `unable to allocate shared memory` 错误。解决方案：`dmla start --gpu` 自动设置 1GB shm，无需手动配置。
:::

#### 性能对比总结

| 方案 | 内存 | 磁盘 | GPU利用率 | 每 epoch | 适用场景 |
|------|------|------|-----------|----------|---------|
| A: 实时预处理 | 2GB | 250MB | 15% | 30分钟 | 历史教学 |
| B: 最小缓存 | 4GB | 600MB | 40-60% | 10-15分钟 | 教学推荐 |
| **D: LMDB+多线程** | **2GB** | **600MB** | **40-60%** | **~1-2分钟** | **推荐方案** |

**本实验选择方案 D 的原因**：
1. 内存需求最低（仅 GPU batch + 模型）
2. GPU 利用率达 ~40-60%，多线程 DataLoader 消除 I/O 瓶颈
3. 跨平台兼容：Windows 和 Linux Docker 均可运行
4. 配置简单：`dmla start --gpu` 自动设置 shm
5. 原方案的问题（67GB 内存需求）已彻底解决

如果你有更大的内存（64GB+），可以参考原版文档的"全量缓存方案"获得最快的训练速度。如果你追求平衡的性能和内存效率，方案 D（LMDB + DataLoader 多线程）是最佳选择。

接下来，我们创建 PyTorch DataLoader 对图像进行预处理和数据增强。Tiny ImageNet 200 数据集每个类别只有 500 张训练图，相对于模型参数量而言，数量十分有限。数据增强通过随机翻转、裁剪、颜色抖动等变换，可以人工增加训练数据的多样性，防止模型过拟合。AlexNet 参加比赛时使用的是 ImageNet 1K 数据集，尽管比 200 数据集来说要大不少，但仍然需进行数据预处理增强。

本阶段借助了 PyTorch 中十分常用的 `Dataset` 和 `DataLoader` 两个组件。`Dataset` 负责把磁盘上的图像文件和标签映射成 `(图像, 标签)` 对，`DataLoader` 负责批量加载、打乱顺序、多线程读取。

::: tip 性能优化：预处理缓存
原始 Tiny ImageNet 图片尺寸为 64×64，需要放大到 224×224 才能输入 AlexNet。这个 Resize 操作在 CPU 上非常耗时，会导致 GPU 利用率极低（约 15%）。

**解决方案：** 本阶段只缓存 Resize 结果为 JPEG 格式，训练时实时执行数据增强。磁盘占用仅 600MB（而非原方案的 60GB）。

- 缓存位置：`/data/cache/preprocessing/tiny-imagenet-224-minimal/`
- 首次运行：执行 Resize 并保存为 JPEG（约 2-3 分钟）
- 后续运行：直接跳过已缓存的类别
:::

**预处理缓存流程（方案 B：最小缓存 + 实时增强）：**

1. **检查缓存目录：** 如果 `/data/cache/preprocessing/tiny-imagenet-224-minimal/` 已存在且完整，跳过预处理
2. **训练集预处理（可恢复）：** 保持原有目录结构，每张图片 Resize(224) 后保存为 JPEG
3. **验证集预处理（可恢复）：** 扁平化保存到 `val/` 目录，文件名 `val_<idx>.JPEG`
4. **预处理内容：**
    - `Resize(224, 224)`：AlexNet 要求输入为 224×224 的图像，Tiny ImageNet 提供的是 64×64
    - 保存为 JPEG 格式（quality=95）：保留压缩，减少磁盘占用
5. **训练时实时执行：** ToTensor、RandomFlip、RandomCrop、ColorJitter、Normalize

::: tip 缓存大小对比
| 对比项 | 原方案（全量缓存） | 新方案（最小缓存） |
|--------|------------------|------------------|
| 存储格式 | float32 tensor | JPEG（压缩） |
| 单张大小 | ~600 KB | ~6 KB |
| 总磁盘占用 | **~60 GB** | **~600 MB** |
| 内存需求 | ~67 GB | ~4 GB |
:::

```python runnable gpu timeout=unlimited extract-class="LMDBPreprocessCache"
import os
import io
import json
import lmdb
import struct
from PIL import Image
import time

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 数据及缓存目录
DATA_DIR = '/data/datasets/tiny-imagenet-200'
CACHE_DIR = '/data/cache/preprocessing/tiny-imagenet-224-lmdb'

class LMDBPreprocessCache:
    """
    LMDB 缓存策略：将预处理结果存储到 LMDB 数据库
    
    优势：
    - 单个大文件，避免大量小文件的随机 I/O
    - 内存映射（mmap），零拷贝读取
    - 多进程友好（无锁读取）
    
    数据结构：
    - 键：图片索引（uint64，8字节）
    - 值：label（int32，4字节） + JPEG bytes
    """
    def __init__(self, data_dir, cache_dir, map_size=10*1024*1024*1024):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.map_size = map_size  # LMDB 最大容量（10GB）
        self.train_lmdb_path = os.path.join(cache_dir, 'train.lmdb')
        self.val_lmdb_path = os.path.join(cache_dir, 'val.lmdb')
        self.manifest_path = os.path.join(cache_dir, 'manifest.json')
        
    def preprocess_image(self, img_path):
        """单张图片预处理：Resize(224) → JPEG bytes"""
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=95)
        return buf.getvalue()
    
    def check_cache_exists(self):
        """检查缓存是否已完整存在"""
        return os.path.exists(self.manifest_path)
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            return manifest.get('train_count', 0), manifest.get('val_count', 0)
        return 0, 0
    
    def _preprocess_train_set(self, progress):
        """预处理训练集到 LMDB"""
        train_dir = os.path.join(self.data_dir, 'train')
        classes = sorted(os.listdir(train_dir))
        
        # 读取类别映射
        wnids_path = os.path.join(self.data_dir, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 创建 LMDB 环境
        env = lmdb.open(
            self.train_lmdb_path,
            map_size=self.map_size,
            writemap=True,
            lock=True
        )
        
        total_count = 0
        idx = 0
        
        with env.begin(write=True) as txn:
            for cls_idx, cls in enumerate(classes):
                images_dir = os.path.join(train_dir, cls, 'images')
                if not os.path.exists(images_dir):
                    continue
                
                label = class_to_idx.get(cls, cls_idx)
                
                for img_name in os.listdir(images_dir):
                    if img_name.endswith('.JPEG'):
                        img_path = os.path.join(images_dir, img_name)
                        
                        try:
                            jpeg_bytes = self.preprocess_image(img_path)
                            # 存储格式：键=idx(uint64)，值=label(int32) + JPEG bytes
                            key = struct.pack('>Q', idx)
                            value = struct.pack('>i', label) + jpeg_bytes
                            txn.put(key, value)
                            
                            idx += 1
                            total_count += 1
                        except Exception as e:
                            print(f"Warning: Failed to process {img_path}: {e}")
                
                progress.update(cls_idx + 1, message=f"预处理类别 {cls_idx+1}/200: {cls}")
        
        env.close()
        return total_count
    
    def _preprocess_val_set(self, progress):
        """预处理验证集到 LMDB"""
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
        
        # 重置进度条用于验证集处理
        progress.reset(total_steps=total_val, description="预处理验证集")
        
        # 创建 LMDB 环境
        env = lmdb.open(
            self.val_lmdb_path,
            map_size=self.map_size,
            writemap=True,
            lock=True
        )
        
        labels = []
        idx = 0
        
        with env.begin(write=True) as txn:
            for line_idx, line in enumerate(val_lines):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    img_path = os.path.join(val_images_dir, img_name)
                    label = class_to_idx.get(parts[1], 0)
                    
                    if os.path.exists(img_path):
                        try:
                            jpeg_bytes = self.preprocess_image(img_path)
                            key = struct.pack('>Q', idx)
                            value = struct.pack('>i', label) + jpeg_bytes
                            txn.put(key, value)
                            
                            labels.append(label)
                            idx += 1
                        except Exception as e:
                            print(f"Warning: Failed to process {img_path}: {e}")
                    
                    if (line_idx + 1) % 100 == 0 or line_idx == total_val - 1:
                        progress.update(line_idx + 1, message=f"预处理验证集 {line_idx+1}/{total_val}")
        
        env.close()
        return idx, labels
    
    def run(self, progress):
        """执行预处理"""
        start_time = time.time()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        train_count = self._preprocess_train_set(progress)
        val_count, val_labels = self._preprocess_val_set(progress)
        
        # 保存清单文件
        manifest = {
            'train_count': train_count,
            'val_count': val_count,
            'val_labels': val_labels,
            'format': 'lmdb',
            'key_format': 'uint64',
            'value_format': 'int32_label + jpeg_bytes'
        }
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        elapsed = time.time() - start_time
        progress.complete(message=f"预处理完成: 训练集 {train_count} 张, 验证集 {val_count} 张, 耗时 {elapsed:.1f}s")
        
        return train_count, val_count

preprocessor = LMDBPreprocessCache(DATA_DIR, CACHE_DIR)

if preprocessor.check_cache_exists():
    train_count, val_count = preprocessor.get_cache_stats()
    
    progress = ProgressReporter(total_steps=1, description="预处理阶段")
    progress.update(1, message=f"✓ LMDB 缓存已存在，跳过预处理！训练集 {train_count} 张, 验证集 {val_count} 张")
    progress.complete(message="预处理阶段完成（LMDB 缓存已存在）")
    
    print(f"LMDB 缓存已存在，跳过预处理")
    print(f"训练集: {train_count} 张图片（train.lmdb）")
    print(f"验证集: {val_count} 张图片（val.lmdb）")
else:
    if not os.path.exists(DATA_DIR):
        print("错误: 数据集未下载，请先运行 'dmla data' 下载数据集")
    else:
        progress = ProgressReporter(total_steps=200, description="预处理训练集")
        train_count, val_count = preprocessor.run(progress)
        print(f"预处理完成: 训练集 {train_count} 张, 验证集 {val_count} 张")
```

### 数据缓存结构（方案 D：LMDB 存储）

预处理完成后，缓存目录结构如下：

```
/data/cache/preprocessing/tiny-imagenet-224-lmdb/
├── train.lmdb              # 训练集 LMDB 数据库（单文件，约 600MB）
│   └── data.mdb            # 实际数据文件
│   └── lock.mdb            # 锁文件
├── val.lmdb                # 验证集 LMDB 数据库（单文件，约 60MB）
│   └── data.mdb
│   └── lock.mdb
└── manifest.json           # 元数据（总数量、格式说明）
```

**LMDB 数据存储格式**：

| 项目 | 格式 |
|------|------|
| 键（Key） | uint64（8字节），图片索引 |
| 值（Value） | int32 label（4字节） + JPEG bytes |
| 数据库大小 | train.lmdb ~600MB，val.lmdb ~60MB |

**与其他方案对比**：

| 对比项 | 方案 A（实时） | 方案 B（JPEG 缓存） | 方案 D（LMDB） |
|--------|--------------|-------------------|---------------|
| 存储格式 | 不缓存 | JPEG 文件 | LMDB 数据库 |
| 文件数量 | 0 | 110,000 个文件 | 2 个数据库 |
| 单文件大小 | N/A | ~6 KB | 600 MB |
| 总磁盘占用 | ~250 MB | ~600 MB | ~600 MB |
| 加载方式 | Image.open() | Image.open() | LMDB mmap |
| 内存需求 | ~2 GB | ~4 GB | ~4 GB |
| GPU 利用率 | ~15% | ~40% | ~60-70% |

**设计说明**：
- **单文件存储**：避免大量小文件的随机 I/O 开销
- **内存映射**：LMDB 使用 mmap，零拷贝读取，CPU 开销更低
- **多进程友好**：无锁读取设计，适合 DataLoader 多线程

### LMDB 读取原理

LMDB（Lightning Memory-Mapped Database）是一种高效的键值存储：

1. **内存映射（mmap）**：数据库文件直接映射到进程内存空间，读取时无需拷贝
2. **B+树结构**：按键有序存储，支持快速随机访问
3. **零拷贝读取**：JPEG bytes 直接从 mmap 区域读取，无需额外缓冲区
4. **多进程安全**：读取操作无锁，多个 DataLoader worker 可并发读取

### LMDB + DataLoader 多线程原理

**LMDB mmap 零拷贝读取**：
- LMDB 使用内存映射（mmap）直接访问磁盘数据
- 无需将数据复制到用户空间内存，直接读取
- 数据库结构紧凑，避免大量小文件 I/O

**PyTorch DataLoader 多线程**：
- `num_workers=4`：4 个子进程并行读取数据
- 每个子进程独立解码 JPEG、执行数据增强
- 主进程收集 batch 后传输到 GPU

**Docker shm 配置**：
- PyTorch DataLoader 多线程需要共享内存（shm）
- 默认 Docker shm 只有 64MB，不够用
- `dmla start --gpu` 自动设置 1GB shm
- 解决 `unable to allocate shared memory` 错误

**性能对比**：
| 配置 | num_workers | 128 batch 耗时 | GPU 利用率 |
|------|------------|---------------|-----------|
| shm=64MB | 0（单线程） | ~200ms | ~1-3% |
| shm=1GB | 4（多线程） | ~40-50ms | ~40-60% |

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

本阶段执行完整的训练流程，采用方案 D（LMDB 存储 + 多线程 DataLoader）策略。通过 CLI 的 `--shm-size` 参数配置 Docker 共享内存，支持 DataLoader 多线程并行加载。

::: tip Docker shm 配置
**启动服务时配置共享内存**：
```bash
# GPU 模式自动使用 1GB shm
dmla start --gpu

# 或手动指定 shm 大小
dmla start --gpu --shm-size 1024
```

**为什么需要 shm 配置**：
- PyTorch DataLoader 多线程模式需要共享内存存储数据批次
- Docker 默认 shm 只有 64MB，无法支持 num_workers > 0
- GPU 模式默认 1GB shm，支持 4-8 个 DataLoader worker

**性能提升**：DataLoader 多线程后，GPU 利用率从 3% 提升至 40-60%
:::

**训练关键点：**

1. **设备选择：** 检测 GPU 是否可用
2. **LMDB 读取：** mmap 零拷贝读取，多线程友好
3. **DataLoader 多线程：** num_workers=4，并行 JPEG 解码
4. **Normalize 在 GPU 执行：** 批量化操作更快
5. **损失函数 `CrossEntropyLoss`：** 多分类任务标准损失
6. **优化器 `SGD`：** lr=0.01, momentum=0.9, weight_decay=0.0005
7. **学习率调度 `StepLR`：** 每 10 epoch 乘以 0.1

```python runnable gpu timeout=unlimited
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 导入共享模块
from shared.cnn.alexnet import AlexNet
from shared.cnn.lmdb_dataset import LMDBDataset, LMDBValDataset

# LMDB 缓存目录（方案 D）
LMDB_DIR = '/data/cache/preprocessing/tiny-imagenet-224-lmdb'

# 创建进度报告器
progress = ProgressReporter(total_steps=100, description="准备训练环境")
progress.update(0, message="检查 LMDB 缓存...")

# 检查 LMDB 缓存
manifest_path = os.path.join(LMDB_DIR, 'manifest.json')
train_lmdb_path = os.path.join(LMDB_DIR, 'train.lmdb')
val_lmdb_path = os.path.join(LMDB_DIR, 'val.lmdb')

if not os.path.exists(manifest_path) or not os.path.exists(train_lmdb_path):
    print("错误: LMDB 缓存不存在，请先执行第二阶段的预处理代码")
    progress.error(message="LMDB 缓存不存在")
else:
    import json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    print(f"LMDB 缓存已存在: 训练集 {manifest['train_count']} 张, 验证集 {manifest['val_count']} 张")

# 检测 GPU
progress.update(10, message="检测 GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")

# 创建 LMDB Dataset
progress.update(20, message="创建 LMDB Dataset...")
print("[DEBUG] 创建 LMDB Dataset（mmap 零拷贝读取）...")

train_dataset = LMDBDataset(train_lmdb_path, augment=True, normalize_on_gpu=True)
val_dataset = LMDBValDataset(val_lmdb_path, normalize_on_gpu=True)

print(f"[DEBUG] Dataset 创建完成: 训练集 {len(train_dataset)} 张, 验证集 {len(val_dataset)} 张")

# 创建 DataLoader（多线程模式，shm 已通过 CLI 配置）
progress.update(30, message="创建 DataLoader...")
num_workers = 4  # 多线程 DataLoader（需要 shm >= 512MB）

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True  # 固定内存，加速 GPU 传输
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

print(f"[DEBUG] DataLoader 配置: num_workers={num_workers}, pin_memory=True")
print(f"[DEBUG] 每 epoch {len(train_loader)} 个 batch")

# 创建模型
progress.update(50, message="创建 AlexNet 模型...")
model = AlexNet(num_classes=200).to(device)
print(f"[DEBUG] 模型创建完成: {sum(p.numel() for p in model.parameters()):,} 参数")

# 获取 Normalize 参数（在 GPU 执行）
mean, std = train_dataset.get_normalize_params()
mean = mean.to(device)
std = std.to(device)

def normalize_batch(batch):
    """在 GPU 执行 Normalize（批量化更快）"""
    return (batch - mean) / std

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

progress.update(60, message="训练环境准备完成")

# 创建性能日志文件
perf_log_path = '/data/models/alexnet/performance_log.txt'
os.makedirs('/data/models/alexnet', exist_ok=True)
perf_log = open(perf_log_path, 'w')
perf_log.write("batch_idx,data_load_ms,transfer_ms,normalize_ms,forward_ms,backward_ms,optimizer_ms,total_ms\n")
print(f"[DEBUG] 性能日志文件: {perf_log_path}")

# 切换到训练阶段进度
total_batches = len(train_loader)
num_epochs = 1  # 测试运行：只执行 1 个 epoch
progress.reset(total_steps=num_epochs * total_batches, description="训练 AlexNet（多线程 DataLoader）")
current_batch = 0
best_acc = 0.0

print(f"[DEBUG] 开始训练: {num_epochs} epochs, 每 epoch {total_batches} batches")
print(f"[DEBUG] DataLoader 多线程: {num_workers} workers")

# 训练函数（带详细耗时日志）
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, progress, current_batch, perf_log):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    prev_batch_end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 1. 测量数据加载耗时（DataLoader 迭代时间）
        data_load_time = time.time() - prev_batch_end
        batch_start = time.time()

        # 2. 测量传输到GPU耗时
        transfer_start = time.time()
        inputs_gpu = inputs.to(device)
        targets_gpu = targets.to(device)
        transfer_time = time.time() - transfer_start
        
        # 3. 测量Normalize耗时（在GPU执行）
        normalize_start = time.time()
        inputs_norm = normalize_batch(inputs_gpu)
        normalize_time = time.time() - normalize_start
        
        # 4. 测量前向传播耗时
        forward_start = time.time()
        optimizer.zero_grad()
        outputs = model(inputs_norm)
        forward_time = time.time() - forward_start
        
        # 5. 测量损失计算耗时
        loss = criterion(outputs, targets_gpu)
        
        # 6. 测量反向传播耗时
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # 7. 测量优化器step耗时
        optimizer_start = time.time()
        optimizer.step()
        optimizer_time = time.time() - optimizer_start
        
        batch_total = time.time() - batch_start

        # 写入日志（每10个batch写一次）
        if batch_idx % 10 == 0:
            perf_log.write(f"{batch_idx},{data_load_time*1000:.1f},{transfer_time*1000:.1f},{normalize_time*1000:.1f},{forward_time*1000:.1f},{backward_time*1000:.1f},{optimizer_time*1000:.1f},{batch_total*1000:.1f}\n")

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets_gpu.size(0)
        correct += predicted.eq(targets_gpu).sum().item()

        prev_batch_end = time.time()

        # 每 10 个 batch 更新进度
        if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            current_batch += 1
            batch_acc = 100. * correct / total
            progress.update(
                current_batch * 10,
                message=f"Epoch {epoch+1}/{num_epochs} Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={batch_acc:.2f}%"
            )
    
    return running_loss / len(train_loader), 100. * correct / total, current_batch

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = normalize_batch(inputs.to(device))
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

# 主训练循环
print("\n" + "=" * 60)
print("开始训练 AlexNet on Tiny ImageNet（方案 D: LMDB + 多线程 DataLoader）")
print("=" * 60)

save_dir = '/data/models/alexnet/checkpoints'
os.makedirs(save_dir, exist_ok=True)

try:
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        train_loss, train_acc, current_batch = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, progress, current_batch // 10, perf_log
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% "
              f"Time: {epoch_time:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  -> 保存最佳模型 (准确率: {best_acc:.2f}%)")
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(save_dir, f'epoch_{epoch+1}.pth'))
    
    progress.complete(message=f"训练完成！最佳验证准确率: {best_acc:.2f}%")
    
    # 关闭性能日志
    perf_log.close()
    print(f"\n性能日志已保存: {perf_log_path}")
    
    final_dir = '/data/models/alexnet/final'
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, 'alexnet_tiny_imagenet.pth'))
    print(f"\n最终模型已保存: {os.path.join(final_dir, 'alexnet_tiny_imagenet.pth')}")
    
except Exception as e:
    perf_log.close()
    progress.error(message=f"训练出错: {str(e)}")
    print(f"\n训练出错: {e}")
    print(f"\n性能日志已保存: {perf_log_path}（可用于分析瓶颈）")
    raise
```

### 性能对比

| 指标 | 方案 A | 方案 B | 方案 D | 原方案 |
|------|-------|-------|--------|--------|
| 内存占用 | ~2 GB | ~4 GB | **~2 GB** | ~67 GB |
| 磁盘缓存 | 250 MB | 600 MB | **600 MB** | 60 GB |
| GPU 利用率 | ~15% | ~40-60% | **~40-60%** | ~80% |
| 每 epoch 时间 | ~30 分钟 | ~10-15 分钟 | **~1-2 分钟** | ~5 分钟 |
| JPEG 解码 | CPU (单线程) | CPU (单线程) | **CPU (多线程)** | N/A |
| DataLoader workers | 0 | 0 | **4** | 4 |

**方案选择建议**：
- **内存 < 16 GB**：使用方案 A（实时预处理）
- **内存 16-32 GB**：使用方案 B（最小缓存）
- **追求最高性能**：使用方案 D（LMDB + 多线程 DataLoader）—— **本实验推荐**
- **内存 > 64 GB**：使用原方案（全量缓存）

**方案 D 的优势**：
- GPU 利用率高：多线程 DataLoader 消除 I/O 瓶颈
- 内存最友好：LMDB mmap 零拷贝读取
- 配置简单：通过 `dmla start --gpu` 自动设置 shm
- 跨平台兼容：Windows 和 Linux Docker 均可运行

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

本实验完整展示了 AlexNet 的训练流程，采用三种缓存策略对比的教学方法：

| 阶段 | 关键步骤 | 代码块 | 执行时间 | 内存占用 |
|:-----|:---------|:-------|:---------|:---------|
| 数据准备 | 检查/下载数据集 | 常规 | - | ~100 MB |
| 数据预处理 | Resize → LMDB 存储 | `timeout=unlimited` | 首次约 2-3 分钟 | ~2 GB |
| 模型定义 | AlexNet 类定义 | `extract-class` | - | ~60 MB |
| 模型训练 | LMDB + DataLoader (num_workers=4) | `timeout=unlimited` | ~40-50ms/batch | **~2 GB** |
| 模型推理 | 加载模型预测 | 常规 | - | ~200 MB |

**性能优化要点（方案 D）**：

1. **LMDB 高效存储**：mmap 零拷贝读取，避免大量小文件 I/O
2. **PyTorch DataLoader 多线程**：`num_workers=4` 利用多核 CPU 并行解码
3. **Docker shm 配置**：`dmla start --gpu` 自动设置 1GB shm，支持多线程 DataLoader
4. **跨平台兼容**：Windows 和 Linux Docker 均可运行，无需区分操作系统
5. **内存最低**：LMDB mmap 按需加载，适合所有机器

**四种方案对比总结**：

| 方案 | 内存 | 磁盘 | GPU利用率 | 每 epoch | 适用场景 |
|------|------|------|-----------|----------|---------|
| A: 实时预处理 | 2GB | 250MB | 15% | 30分钟 | 历史教学 |
| B: 最小缓存 | 4GB | 600MB | 40-60% | 10-15分钟 | 普通机器 |
| **D: LMDB+多线程** | **2GB** | **600MB** | **40-60%** | **~1-2分钟** | **推荐方案** |
| 原方案: 全量缓存 | 67GB | 60GB | 80% | 5分钟 | 高内存机器 |

## 生成的文件

训练完成后，以下文件将保存到数据目录：

**预处理缓存（方案 D：LMDB 存储）**：
- `/data/cache/preprocessing/tiny-imagenet-224-lmdb/train.lmdb/` - 训练集 LMDB 数据库（约 600MB）
- `/data/cache/preprocessing/tiny-imagenet-224-lmdb/val.lmdb/` - 验证集 LMDB 数据库（约 60MB）
- `/data/cache/preprocessing/tiny-imagenet-224-lmdb/manifest.json` - 缓存清单（数量、格式说明）

**模型文件**：
- `/data/models/alexnet/checkpoints/best_model.pth` - 最佳验证准确率的模型
- `/data/models/alexnet/checkpoints/epoch_*.pth` - 每 5 epoch 的 checkpoint
- `/data/models/alexnet/final/alexnet_tiny_imagenet.pth` - 最终模型权重

**性能日志**：
- `/data/models/alexnet/performance_log.txt` - 详细耗时日志（用于分析瓶颈）
