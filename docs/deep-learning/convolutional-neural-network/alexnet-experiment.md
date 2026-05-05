# AlexNet 复现训练

本次工程实训中，笔者将与你一同使用现代深度学习框架（PyTorch）来复现完整的 AlexNet 训练流程，从数据准备到预处理增强、从模型训练到推理，通过实践来理解经典 CNN 架构与现代机器学习框架如何结合，并讨论在工程上如何权衡考量机器学习应用的性能、鲁棒、环境约束、资源消耗等因素。

## 实验准备

在开始实验之前，请确保已[挂载数据目录](../../sandbox.md#数据管理)并下载好 Tiny ImageNet 200 数据集，你可以通过 `DMLA-CLI` 工具自动完成该工作：
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

## 第二阶段：数据预处理

接下来，我们创建 PyTorch DataLoader 对图像进行预处理和数据增强。

AlexNet 参加比赛时使用的是 ImageNet 1K 数据集，包含的训练集约 120 万张图像，验证集约 5 万张图像，测试集约 10 万张图像。这个数据规模已经不小，但与 6000 万模型参数和 1000 类的分类输出来说依然十分有限，因此需要在数据预处理阶段进行随机翻转、裁剪、颜色抖动等变换，人工增加训练数据的多样性，防止模型过拟合。

本次训练我们使用的是 [Tiny ImageNet 200](https://cs231n.stanford.edu/reports/2016/pdfs/401_Report.pdf) 数据集，Tiny 是指图片尺寸被缩小为 64 × 64 的 JPEG 格式，200 是指数据集包含有 200 类别。我们在数据预处理进行的唯一增强是将图片插值放大至 224 × 224 的尺寸，以便对接原版 AlexNet 的网络结构。

数据预处理代码借助了 PyTorch 中十分常用的 `Dataset` 和 `DataLoader` 两个组件。`Dataset` 负责把磁盘上的图像文件和标签映射成 `(图像, 标签)` 对，`DataLoader` 负责批量加载、打乱顺序、多线程读取。

从学术角度看，数据预处理无外乎填充、去噪、归一化这些事情，但从工程角度来看，数据预处理对整个模型的训练效果与训练效率都有巨大影响。以本次实验中的预处理缓存这一个点为例，如果完全不采用缓存，进行实时预处理，那么每一个 epoch 都要进行十万次文件访问、JPEG 解码、Resize 等操作，将产生大量重复的计算。另一方面，如果要将计算结果直接缓存也未必是合适的决策，原始 64 × 64 压缩格式单张图片约 2KB，转成 224 × 224 的 FP32 张量后，要膨胀 300 倍来到一张 600KB 左右，这样总数据量将超过 60GB，放到内存中需要工作站级别的硬件，放到磁盘会又会带来高昂的 I/O 负担。

::: warning 这是 AlexNet 2012 年的真实训练方式
AlexNet 在 2012 年用 GTX 580（3GB 显存）训练了完整的 ImageNet 数据集。当年的硬件限制决定了只能进行实时预处理，包括 Resize 、Clip、Normalize 等操作实际上只能在 CPU 上进行，根据 AlexNet 团队自己提供的信息，GPU 利用率其实只有 10%，一次训练要耗费了 5 天时间才能完成。
:::

针对预处理的缓存场景，在本次实验的核心工程决策如下：

- 负担最重的两个操作是 JPEG 解码和 Resize，它们都会扩大数据，但从 64 到 224 的 Resize 操作是扩大几倍，而 JPEG 解码转成 FP32 的张量则是扩大上百倍。因此将 Resize 后的结果不解码原样保存。这节省掉其中一个重量级操作，大约会让数据集从 250MB 膨胀到 1GB 左右（按 Quality=95 估算）。
- 使用 LMDB（Lightning Memory-Mapped Database）代替文件系统存储预处理结果，LMDB 通过内存映射文件（mmap）将数据文件直接映射到进程的虚拟地址空间，实现零拷贝读取，大幅提升了 I/O 效率。
- JPEG 解码使用 NVIDIA DALI 库的 nvJPEG 算子，移动到 GPU 中完成，避免了显存和内存的来回复制，大幅度提升解码效率（Windows 环境不适用）。
- 使用多线程 DataLoader（`num_workers=4`）的批量操作尽可能消除 I/O 瓶颈，提升处理效率（Windows 环境不适用）。

在此方案下，内存消耗在 4GB 左右，形成的预处理结果为 2.3GB（LMDB 的预设存储空间为训练集 2GB + 验证集 300MB），最终数据预处理代码如下：

```python runnable gpuonly timeout=unlimited extract-class="LMDBPreprocessCache"
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
    def __init__(self, data_dir, cache_dir, map_size=2*1024*1024*1024):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.map_size = map_size  # LMDB 最大容量（2GB，足够存储 600MB JPEG）
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
        
        # 创建 LMDB 环境（验证集使用较小的 map_size）
        env = lmdb.open(
            self.val_lmdb_path,
            map_size=256*1024*1024,  # 256MB（验证集约 60MB）
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

## 第三阶段：模型定义

以下二三十行代码就实现了 [AlexNet 网络结构](alexnet.md#网络结构)，除因输出分类部分要适配 Tiny ImageNet 的 200 个类别外（而非原始的 1000 类），其余网络定义与原版的 AlexNet 完全保持一致。从代码结构可见，神经网络模型的常用部件，如卷积层、池化层、激活函数、Dropout 正则化等在现代机器学习框架都有标准件提供。模型的难点在于合理设计与高效训练，将设计转化为代码实现并不困难。

1. `features`（特征提取层）：5 个卷积层交替叠加，逐层提取从低级（边缘、纹理）到高级（物体部件）的特征。卷积层之间的 `MaxPool2d` 负责下采样，逐步缩小空间尺寸。`AdaptiveAvgPool2d((6, 6))` 确保无论输入图像经过前面的卷积池化后尺寸如何，输出始终固定为 6×6
2. `classifier`（分类层）：3 个全连接层。前两层使用 `Dropout(p=0.5)` 随机丢弃 50% 的神经元激活，防止过拟合，这是 AlexNet 的标志性设计。最后将 4096 维特征映射到 200 个类别的 Softmax 分类器
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

模型训练是深度学习最核心的环节，通过反向传播算法不断调整网络参数，使模型逐步学会从图像中提取特征并进行分类。训练流程包含三个关键步骤：前向传播计算预测值与损失、反向传播计算梯度、优化器更新参数。这个过程看似简单，但工程上却涉及诸多效率考量。训练一个 epoch 需要处理 10 万张图片，假设每张图片处理耗时 10ms，那么一轮训练就需要十多分钟。如果数据加载、预处理、GPU 计算之间存在串行等待，这个时间会更长。

本阶段的工程决策围绕"如何消除数据加载瓶颈"展开：

- **JPEG 解码位置选择**：解码是 CPU 密集型操作，单线程 PIL 解码每张图片约 1-3ms。使用 NVIDIA DALI 库的 nvJPEG 算子可以将解码移到 GPU 执行，速度提升 10 倍以上。但 Windows 宿主环境下，Docker 由于 NVML 限制无法使用 GPU nvJPEG，只能使用 DALI 的 CPU 多线程解码（仍比单线程快 2-3 倍）。
- **数据增强位置选择**：随机翻转、裁剪、归一化等操作如果放在 CPU 执行，会产生额外的 CPU-GPU 数据传输。DALI 将这些操作全部移到 GPU 执行，数据在 GPU 显存中流转，无需传输回 CPU。
- **LMDB mmap 零拷贝读取**：第二阶段已经将预处理结果存入 LMDB，本阶段通过内存映射直接读取 JPEG bytes，避免了额外的文件 I/O 操作。
- **环境自适应设计**：通过检测 `/proc/version` 内容判断宿主操作系统，Windows 自动切换为 CPU 多线程解码模式，确保兼容运行；Linux 使用 GPU nvJPEG 解码模式，获得最大效率。

```python runnable gpuonly timeout=unlimited
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import struct
import numpy as np

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 导入 DALI
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# 导入共享模块中的 AlexNet
from shared.cnn.alexnet import AlexNet

# LMDB 缓存目录
LMDB_DIR = '/data/cache/preprocessing/tiny-imagenet-224-lmdb'

def detect_host_os():
    """检测宿主操作系统"""
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            if 'microsoft' in version_info or 'wsl' in version_info:
                return 'windows'
            return 'linux'
    except:
        return 'linux'

class DALILMDBReader:
    """
    DALI External Source - LMDB JPEG Reader
    
    从 LMDB 数据库读取 JPEG bytes，供 DALI Pipeline 使用
    """
    def __init__(self, lmdb_path, batch_size, shuffle=True):
        import lmdb
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 获取数据数量
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
        
        self.indices = np.arange(self.length)
        self._reset()
    
    def _reset(self):
        """重置迭代器"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        self._position = 0
    
    def __call__(self):
        """DALI external_source 需要一个 callable，每次调用返回一个 batch"""
        if self._position >= self.length:
            self._reset()
            return None, None
        
        batch_jpegs = []
        batch_labels = []
        end_idx = min(self._position + self.batch_size, self.length)
        with self.env.begin() as txn:
            for i in range(self._position, end_idx):
                idx = self.indices[i]
                key = struct.pack('>Q', idx)
                value = txn.get(key)
                if value is not None:
                    label = struct.unpack('>i', value[:4])[0]
                    jpeg_bytes = np.frombuffer(value[4:], dtype=np.uint8)
                    batch_jpegs.append(jpeg_bytes)
                    batch_labels.append(label)
        self._position = end_idx
        return batch_jpegs, np.array(batch_labels, dtype=np.int32)
    
    def __len__(self):
        return self.length

@pipeline_def
def create_train_pipeline(data_source, decode_device='cpu'):
    """
    DALI 训练 Pipeline
    
    decode_device:
    - 'cpu': Windows Docker (NVML 限制，使用 CPU 多线程解码)
    - 'mixed': Linux Docker (GPU nvJPEG 解码)
    """
    jpegs, labels = fn.external_source(
        source=data_source,
        num_outputs=2,
        dtype=[types.UINT8, types.INT32],
        batch=True
    )
    
    # JPEG 解码
    images = fn.decoders.image(
        jpegs,
        device=decode_device,
        output_type=types.RGB
    )
    
    # 如果是 CPU 解码，传输到 GPU
    if decode_device == 'cpu':
        images = images.gpu()
    
    # GPU 数据增强 + Normalize
    images = fn.crop_mirror_normalize(
        images,
        device='gpu',
        dtype=types.FLOAT,
        output_layout='CHW',
        crop=(224, 224),
        mirror=fn.random.coin_flip(probability=0.5),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )
    
    labels = labels.gpu()
    
    return images, labels

@pipeline_def
def create_val_pipeline(data_source, decode_device='cpu'):
    """DALI 验证 Pipeline（无数据增强）"""
    jpegs, labels = fn.external_source(
        source=data_source,
        num_outputs=2,
        dtype=[types.UINT8, types.INT32],
        batch=True
    )
    
    images = fn.decoders.image(
        jpegs,
        device=decode_device,
        output_type=types.RGB
    )
    
    if decode_device == 'cpu':
        images = images.gpu()
    
    images = fn.crop_mirror_normalize(
        images.gpu(),
        device='gpu',
        dtype=types.FLOAT,
        output_layout='CHW',
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )
    
    labels = labels.gpu()
    
    return images, labels

# 主训练代码
progress = ProgressReporter(total_steps=100, description="准备训练环境")
progress.update(0, message="检测运行环境...")
host_os = detect_host_os()
decode_device = 'cpu' if host_os == 'windows' else 'mixed'
print(f"[环境检测] 宿主操作系统: {host_os.upper()}")
print(f"[环境检测] DALI 解码设备: {decode_device}")
if host_os == 'windows':
    print("[环境检测] Windows Docker: CPU 多线程 JPEG 解码")
else:
    print("[环境检测] Linux Docker: GPU nvJPEG 解码")

# 检查 LMDB 缓存
progress.update(5, message="检查 LMDB 缓存...")
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
    device_id = torch.cuda.current_device()
else:
    print("警告: 未检测到 GPU，DALI 需要 GPU")
    device_id = 0

# 创建 DALI Pipeline
progress.update(20, message="创建 DALI Pipeline...")
batch_size = 128
train_reader = DALILMDBReader(train_lmdb_path, batch_size, shuffle=True)
val_reader = DALILMDBReader(val_lmdb_path, batch_size, shuffle=False)

train_pipe = create_train_pipeline(
    data_source=train_reader,
    decode_device=decode_device,
    batch_size=batch_size,
    num_threads=4,
    device_id=device_id
)
val_pipe = create_val_pipeline(
    data_source=val_reader,
    decode_device=decode_device,
    batch_size=batch_size,
    num_threads=4,
    device_id=device_id
)

train_pipe.build()
val_pipe.build()

print(f"DALI Pipeline 创建完成 ({host_os} 模式)")
print(f"训练集: {len(train_reader)} 张, 每 epoch {len(train_reader) // batch_size} batches")

# 创建模型
progress.update(50, message="创建 AlexNet 模型...")
model = AlexNet(num_classes=200).to(device)
print(f"模型创建完成: {sum(p.numel() for p in model.parameters()):,} 参数")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

progress.update(60, message="训练环境准备完成")

# 创建性能日志
perf_log_path = '/data/models/alexnet/performance_log.txt'
os.makedirs('/data/models/alexnet', exist_ok=True)
perf_log = open(perf_log_path, 'w')
perf_log.write("batch_idx,decode_ms,transfer_ms,forward_ms,backward_ms,optimizer_ms,total_ms\n")

# 切换到训练进度
total_batches = len(train_reader) // batch_size
num_epochs = 20
progress.reset(total_steps=num_epochs * total_batches, description=f"训练 AlexNet (DALI {host_os})")
best_acc = 0.0

print(f"开始训练: {num_epochs} epochs, 每 epoch {total_batches} batches")

# 训练函数
def train_one_epoch_dali(model, train_reader, train_pipe, criterion, optimizer, device, perf_log):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_reader._reset()
    
    for batch_idx in range(total_batches):
        pipe_start = time.time()
        outputs = train_pipe.run()
        decode_time = time.time() - pipe_start
        batch_start = time.time()
        
        # 从 DALI TensorList 获取 PyTorch tensor
        images = outputs[0].as_tensor()
        labels = outputs[1].as_tensor()
        inputs = torch.from_dlpack(images)
        targets = torch.from_dlpack(labels).long()
        
        # Forward
        forward_start = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        forward_time = time.time() - forward_start
        
        # Backward
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # Optimizer
        optimizer_start = time.time()
        optimizer.step()
        optimizer_time = time.time() - optimizer_start
        total_time = time.time() - batch_start
        perf_log.write(f"{batch_idx},{decode_time*1000:.1f},0,{forward_time*1000:.1f},{backward_time*1000:.1f},{optimizer_time*1000:.1f},{total_time*1000:.1f}\n")
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 50 == 0:
            progress.update(batch_idx, message=f"Batch {batch_idx}/{total_batches}")
    return running_loss / total_batches, 100. * correct / total

# 验证函数
def validate_dali(model, val_reader, val_pipe, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    val_reader._reset()
    val_batches = len(val_reader) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(val_batches):
            outputs = val_pipe.run()
            images = outputs[0].as_tensor()
            labels = outputs[1].as_tensor()
            inputs = torch.from_dlpack(images)
            targets = torch.from_dlpack(labels).long()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / val_batches, 100. * correct / total

try:
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch_dali(model, train_reader, train_pipe, criterion, optimizer, device, perf_log)
        val_loss, val_acc = validate_dali(model, val_reader, val_pipe, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% Time: {epoch_time:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_dir = '/data/models/alexnet/checkpoints'
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  -> 保存最佳模型 (准确率: {best_acc:.2f}%)")
        
        # 每 4 epoch 保存 checkpoint
        if (epoch + 1) % 4 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(save_dir, f'epoch_{epoch+1}.pth'))
            print(f"  -> 保存 epoch {epoch+1} checkpoint")
    
    progress.complete(message=f"训练完成！最佳准确率: {best_acc:.2f}%")
    
    perf_log.close()
    print(f"\n性能日志已保存: {perf_log_path}")
    
    final_dir = '/data/models/alexnet/final'
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, 'alexnet_tiny_imagenet.pth'))
    print(f"最终模型已保存: {os.path.join(final_dir, 'alexnet_tiny_imagenet.pth')}")
    
except Exception as e:
    perf_log.close()
    progress.error(message=f"训练出错: {str(e)}")
    print(f"\n训练出错: {e}")
    print(f"\n性能日志已保存: {perf_log_path}")
    raise
```

## 第五阶段：推理评估

使用训练好的模型对新图像进行分类预测。训练完成后，验证模型的实际分类效果，展示模型"学到了什么"。关键设计点有：

1. **模型加载：** 优先加载验证准确率最高的 checkpoint（`best_model.pth`），其次加载最终模型。如果都找不到，则使用未训练的随机权重模型（仅供测试，预测结果无意义）。
2. **推理预处理**：与验证集预处理相同（Resize → ToTensor → Normalize），不做其他数据增强。输入图像的预处理方式与训练时一致。
3. **类别名称映射**：Tiny ImageNet 的类别标签是 WordNet ID（如 `n01675725`），通过 `wnids.txt` 和 `words.txt` 映射为可读的英文描述（如 `turtle, tortoise`）。
4. **预测图像**：判断 Top-5 错误率结果，基本逻辑是：读取图像 → 预处理 → 送入模型 → 使用 `softmax` 将 logits 转为概率（0-100%） → `topk(5)` 取概率最高的 5 个类别，输出 Top-5 预测结果。Top-5 是 ILSVRC 图像分类的默认评估指标，只要正确答案在前 5 个预测中，就认为模型正确分类了。

```python runnable gpuonly
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

## 实验结果

本实验完整展示了 AlexNet 的训练流程，训练完成后，以下生成的文件将保存到数据目录：

- **模型文件**：
    - `/data/models/alexnet/checkpoints/best_model.pth` - 最佳验证准确率的模型
    - `/data/models/alexnet/checkpoints/epoch_*.pth` - 每 5 epoch 的 checkpoint
    - `/data/models/alexnet/final/alexnet_tiny_imagenet.pth` - 最终模型权重
- **预处理缓存**：
    - `/data/cache/preprocessing/tiny-imagenet-224-lmdb/train.lmdb/` - 训练集 LMDB 数据库（约 2GB）
    - `/data/cache/preprocessing/tiny-imagenet-224-lmdb/val.lmdb/` - 验证集 LMDB 数据库（约 300MB）
    - `/data/cache/preprocessing/tiny-imagenet-224-lmdb/manifest.json` - 缓存清单（数量、格式说明）
- **性能日志**：
    - `/data/models/alexnet/performance_log.txt` - 详细耗时日志（用于分析瓶颈）
