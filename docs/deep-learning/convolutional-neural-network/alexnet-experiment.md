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

**预处理缓存流程（支持断点续传）：**

1. **检查缓存目录：** 如果 `/data/cache/preprocessing/tiny-imagenet-224/` 已存在且完整，跳过预处理
2. **训练集预处理（可恢复）：** 每个类别处理后立即保存 `.pt` 文件，已存在的类别文件自动跳过
3. **验证集预处理（可恢复）：** 每 1000 张保存一个批次文件 `val_batch_x.pt`，已存在的批次自动跳过
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
import json
from PIL import Image
import time

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 预处理缓存目录1
CACHE_DIR = '/data/cache/preprocessing/tiny-imagenet-224'
TRAIN_CACHE = os.path.join(CACHE_DIR, 'train')
VAL_BATCH_DIR = os.path.join(CACHE_DIR, 'val_batches')  # 验证集批次目录
VAL_MANIFEST = os.path.join(CACHE_DIR, 'val_manifest.json')  # 验证集批次清单
CHECKPOINT_FILE = os.path.join(CACHE_DIR, 'checkpoint.json')  # 进度追踪文件

# 原始数据目录
DATA_DIR = '/data/datasets/tiny-imagenet-200'

# 基础预处理（将被缓存）
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 64→224 放大
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集批次大小（每 1000 张保存一次，支持断点续传）
VAL_BATCH_SIZE = 1000


def load_checkpoint():
    """加载进度追踪文件"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {'train_completed': False, 'val_batch_completed': -1}


def save_checkpoint(checkpoint):
    """保存进度追踪文件"""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")


def preprocess_train_set(progress, checkpoint):
    """
    预处理训练集（支持断点续传）
    
    每个类别处理后立即保存，已存在的类别文件自动跳过
    """
    print("[DEBUG] preprocess_train_set 开始执行")
    
    train_dir = os.path.join(DATA_DIR, 'train')
    classes = sorted(os.listdir(train_dir))
    total_samples = 0
    skipped_classes = 0
    
    print(f"[DEBUG] 训练集类别数: {len(classes)}")
    
    os.makedirs(TRAIN_CACHE, exist_ok=True)
    
    # 获取已存在的类别文件
    existing_cache_files = set(f.replace('.pt', '') for f in os.listdir(TRAIN_CACHE) if f.endswith('.pt'))
    print(f"[DEBUG] 已存在缓存文件数: {len(existing_cache_files)}")
    
    print("[DEBUG] 开始遍历类别...")
    for cls_idx, cls in enumerate(classes):
        # 断点续传：跳过已处理的类别
        if cls in existing_cache_files:
            skipped_classes += 1
            # 每 10 个类别输出一次进度，减少 stderr 调用次数
            if (cls_idx + 1) % 10 == 0 or cls_idx == len(classes) - 1:
                progress.update(cls_idx + 1, message=f"跳过已缓存类别 {cls_idx+1}/200: {cls}")
            continue
        
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
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = base_transform(img)
                    images.append(img_tensor)
                    labels.append(cls_idx)
                except Exception as e:
                    print(f"Warning: Failed to process {img_path}: {e}")
        
        # 立即保存该类别（断点续传的关键）
        if images:
            cls_data = {
                'images': torch.stack(images),
                'labels': torch.tensor(labels),
                'class_name': cls
            }
            cache_path = os.path.join(TRAIN_CACHE, f'{cls}.pt')
            torch.save(cls_data, cache_path)
            total_samples += len(images)
        
        progress.update(cls_idx + 1, message=f"预处理类别 {cls_idx+1}/200: {cls} ({len(images)} 张)")
    
    print(f"[DEBUG] 类别遍历完成, 跳过 {skipped_classes} 个, 处理 {len(classes) - skipped_classes} 个")
    
    # 训练集完成，更新 checkpoint
    print("[DEBUG] 正在更新 checkpoint...")
    checkpoint['train_completed'] = True
    save_checkpoint(checkpoint)
    print("[DEBUG] checkpoint 已保存")
    
    # 统计最终数量
    print("[DEBUG] 正在统计训练集数量...")
    all_cache_files = [f for f in os.listdir(TRAIN_CACHE) if f.endswith('.pt')]
    final_count = sum(torch.load(os.path.join(TRAIN_CACHE, f))['images'].shape[0] for f in all_cache_files)
    print(f"[DEBUG] 训练集总数: {final_count}")
    
    print("[DEBUG] 正在调用 progress.complete()...")
    progress.complete(message=f"训练集预处理完成: {final_count} 张图片 (跳过 {skipped_classes} 个已缓存类别)")
    print("[DEBUG] progress.complete() 完成")
    
    return final_count


def preprocess_val_set(progress, checkpoint):
    """
    预处理验证集（支持断点续传）
    
    每 1000 张保存一个批次文件，已存在的批次自动跳过
    最后保存批次清单，供训练阶段使用
    """
    print("[DEBUG] preprocess_val_set 开始执行")
    
    val_dir = os.path.join(DATA_DIR, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')
    
    print(f"[DEBUG] val_dir: {val_dir}")
    print(f"[DEBUG] val_images_dir: {val_images_dir}")
    print(f"[DEBUG] val_annotations: {val_annotations}")
    
    # 读取类别映射
    print("[DEBUG] 正在读取 wnids.txt...")
    wnids_path = os.path.join(DATA_DIR, 'wnids.txt')
    with open(wnids_path, 'r') as f:
        wnids = [line.strip() for line in f.readlines()]
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
    print(f"[DEBUG] 类别映射完成: {len(class_to_idx)} 个类别")
    
    # 读取标注文件
    print("[DEBUG] 正在读取 val_annotations.txt...")
    with open(val_annotations, 'r') as f:
        val_lines = f.readlines()
    total_val = len(val_lines)
    print(f"[DEBUG] 验证集总数: {total_val} 张")
    
    print("[DEBUG] 正在创建批次目录...")
    os.makedirs(VAL_BATCH_DIR, exist_ok=True)
    print(f"[DEBUG] 批次目录: {VAL_BATCH_DIR}")
    
    # 断点续传：检查已完成的批次
    completed_batches = checkpoint.get('val_batch_completed', -1)
    num_batches = total_val // VAL_BATCH_SIZE + (1 if total_val % VAL_BATCH_SIZE > 0 else 0)
    print(f"[DEBUG] 已完成批次: {completed_batches}, 总批次数: {num_batches}")
    
    print("[DEBUG] 正在重置 ProgressReporter...")
    progress.reset(total_steps=total_val, description="预处理验证集")
    print("[DEBUG] ProgressReporter 重置完成")
    
    # 如果所有批次都已完成，直接跳到合并阶段
    if completed_batches >= num_batches - 1:
        print("[DEBUG] 所有批次已完成，跳到合并阶段")
        progress.update(total_val, message="验证集批次已完成，开始合并...")
    else:
        print(f"[DEBUG] 开始处理批次 {completed_batches + 1} 到 {num_batches - 1}")
        # 处理未完成的批次
        for batch_idx in range(num_batches):
            batch_path = os.path.join(VAL_BATCH_DIR, f'val_batch_{batch_idx}.pt')
            print(f"[DEBUG] 检查批次 {batch_idx}: {batch_path}")
            
            # 断点续传：跳过已存在的批次
            if batch_idx <= completed_batches or os.path.exists(batch_path):
                batch_count = (batch_idx + 1) * VAL_BATCH_SIZE if batch_idx < num_batches - 1 else total_val
                print(f"[DEBUG] 跳过批次 {batch_idx} (已完成或文件存在)")
                progress.update(batch_count, message=f"跳过已缓存批次 {batch_idx+1}/{num_batches}")
                continue
            
            print(f"[DEBUG] 开始处理批次 {batch_idx}...")
            # 处理当前批次
            start_idx = batch_idx * VAL_BATCH_SIZE
            end_idx = min((batch_idx + 1) * VAL_BATCH_SIZE, total_val)
            
            batch_images = []
            batch_labels = []
            
            for line_idx in range(start_idx, end_idx):
                line = val_lines[line_idx]
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    cls = parts[1]
                    img_path = os.path.join(val_images_dir, img_name)
                    
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path).convert('RGB')
                            img_tensor = base_transform(img)
                            batch_images.append(img_tensor)
                            batch_labels.append(class_to_idx.get(cls, 0))
                        except Exception as e:
                            print(f"Warning: Failed to process {img_path}: {e}")
                
                # 每 100 张更新进度
                if (line_idx + 1) % 100 == 0 or line_idx == end_idx - 1:
                    progress.update(line_idx + 1, message=f"预处理验证集 {line_idx+1}/{total_val} 张图片")
            
            # 立即保存当前批次（断点续传的关键）
            if batch_images:
                batch_data = {
                    'images': torch.stack(batch_images),
                    'labels': torch.tensor(batch_labels),
                    'batch_idx': batch_idx
                }
                torch.save(batch_data, batch_path)
                print(f"保存批次 {batch_idx+1}/{num_batches}: {len(batch_images)} 张图片")
            
            # 更新 checkpoint
            checkpoint['val_batch_completed'] = batch_idx
            save_checkpoint(checkpoint)
    
    # 统计验证集数量（不合并批次，避免内存爆炸）
    print("[DEBUG] 统计验证集批次...")
    batch_count = 0
    for batch_idx in range(num_batches):
        batch_path = os.path.join(VAL_BATCH_DIR, f'val_batch_{batch_idx}.pt')
        if os.path.exists(batch_path):
            batch_count += 1
    
    # 创建 val_manifest.json 记录批次信息（供训练阶段使用）
    manifest = {
        'num_batches': batch_count,
        'batch_size': VAL_BATCH_SIZE,
        'total_images': batch_count * VAL_BATCH_SIZE,
        'batch_files': [f'val_batch_{i}.pt' for i in range(batch_count)]
    }
    manifest_path = os.path.join(CACHE_DIR, 'val_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    print(f"[DEBUG] 验证集清单已保存: {manifest_path}")
    
    # 不合并批次文件，直接保留在 val_batches/ 目录
    # 训练阶段会根据 val_manifest.json 加载批次
    print(f"验证集预处理完成: {batch_count} 个批次文件 ({manifest['total_images']} 张图片)")
    
    # 清理 checkpoint 文件（保留批次文件供训练使用）
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    progress.complete(message=f"验证集预处理完成: {manifest['total_images']} 张图片 ({batch_count} 个批次)")
    return manifest['total_images']


def preprocess_and_cache():
    """
    预处理数据集并缓存到磁盘（支持断点续传）
    
    特性：
    - 训练集：每个类别处理后立即保存，中断后可从断点继续
    - 验证集：每 1000 张保存一个批次，中断后可从断点继续
    - 使用 checkpoint.json 追踪进度
    """
    print("[DEBUG] 开始预处理并缓存数据集（支持断点续传）...")
    start_time = time.time()
    
    # 加载进度追踪
    print("[DEBUG] 正在加载 checkpoint...")
    checkpoint = load_checkpoint()
    print(f"[DEBUG] checkpoint 内容: {checkpoint}")
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[DEBUG] 缓存目录已创建: {CACHE_DIR}")
    
    # 创建进度报告器
    print("[DEBUG] 正在创建 ProgressReporter (训练集)...")
    progress = ProgressReporter(total_steps=200, description="预处理训练集")
    print("[DEBUG] ProgressReporter 创建完成")
    
    # 阶段 1：训练集预处理
    if checkpoint.get('train_completed', False):
        print("[DEBUG] checkpoint 显示训练集已完成，跳过预处理")
        # 统计已缓存的训练集数量（只统计文件数，避免加载大文件）
        train_files = [f for f in os.listdir(TRAIN_CACHE) if f.endswith('.pt')]
        print(f"[DEBUG] 训练集缓存文件数: {len(train_files)}")
        # 使用估算值：每个类别约 500 张图片（避免 torch.load 200 个大文件）
        train_count = len(train_files) * 500  # Tiny ImageNet 每个类别约 500 张
        print(f"[DEBUG] 训练集估算图片总数: {train_count}")
        # 只输出进度，不调用 progress.complete()（避免额外的 stderr 写入）
        progress.update(200, message=f"训练集已缓存: ~{train_count} 张图片")
        print("[DEBUG] 训练集跳过处理完成")
    else:
        print("[DEBUG] checkpoint 显示训练集未完成，开始预处理...")
        train_count = preprocess_train_set(progress, checkpoint)
        print(f"[DEBUG] 训练集预处理完成: {train_count} 张")
    
    # 阶段 2：验证集预处理
    print("[DEBUG] 开始验证集预处理阶段...")
    if os.path.exists(VAL_MANIFEST):
        print(f"[DEBUG] VAL_MANIFEST 已存在: {VAL_MANIFEST}")
        with open(VAL_MANIFEST, 'r') as f:
            manifest = json.load(f)
        val_count = manifest['total_images']
        print(f"[DEBUG] 验证集已缓存: {val_count} 张图片 ({manifest['num_batches']} 个批次)，跳过预处理")
    else:
        print("[DEBUG] VAL_MANIFEST 不存在，开始预处理验证集...")
        print("[DEBUG] 正在重置 ProgressReporter...")
        val_count = preprocess_val_set(progress, checkpoint)
        print(f"[DEBUG] 验证集预处理完成: {val_count} 张")
    
    elapsed = time.time() - start_time
    print(f"[DEBUG] 预处理完成: 训练集 {train_count} 张, 验证集 {val_count} 张, 耗时 {elapsed:.1f}s")
    
    # 更新进度条标题，明确告知预处理全部完成
    progress.reset(total_steps=1, description="预处理阶段")
    progress.update(1, message=f"✓ 预处理全部完成！训练集 {train_count} 张, 验证集 {val_count} 张, 可开始训练")
    progress.complete(message=f"预处理阶段完成，耗时 {elapsed:.1f}s")
    
    return train_count, val_count


# 检查缓存是否存在
print("[DEBUG] 开始检查缓存状态...")
print(f"[DEBUG] TRAIN_CACHE: {TRAIN_CACHE}, 存在: {os.path.exists(TRAIN_CACHE)}")
print(f"[DEBUG] VAL_MANIFEST: {VAL_MANIFEST}, 存在: {os.path.exists(VAL_MANIFEST)}")

# 新的缓存检查逻辑：训练集目录存在 + 验证集清单存在
if os.path.exists(TRAIN_CACHE) and os.path.exists(VAL_MANIFEST):
    print("[DEBUG] 缓存已完整存在，跳过预处理")
    # 统计缓存数据量（只统计文件数，避免加载大文件）
    train_files = [f for f in os.listdir(TRAIN_CACHE) if f.endswith('.pt')]
    train_count = len(train_files) * 500  # Tiny ImageNet 每个类别约 500 张
    
    # 从 manifest 读取验证集数量
    with open(VAL_MANIFEST, 'r') as f:
        manifest = json.load(f)
    val_count = manifest['total_images']
    
    # 创建进度条告知用户预处理阶段完成
    progress = ProgressReporter(total_steps=1, description="预处理阶段")
    progress.update(1, message=f"✓ 缓存已存在，跳过预处理！训练集 {train_count} 张, 验证集 {val_count} 张")
    progress.complete(message=f"预处理阶段完成（缓存已存在）")
    
    print(f"缓存已存在，跳过预处理")
    print(f"训练集缓存: ~{train_count} 张图片 ({len(train_files)} 个类别文件)")
    print(f"验证集缓存: {val_count} 张图片 ({manifest['num_batches']} 个批次)")
    print(f"缓存目录: {CACHE_DIR}")
else:
    print("[DEBUG] 缓存不完整，需要预处理")
    # 执行预处理（支持断点续传）
    if not os.path.exists(DATA_DIR):
        print("错误: 数据集未下载，请先运行 'dmla data' 下载数据集")
    else:
        print(f"[DEBUG] DATA_DIR 存在: {DATA_DIR}")
        train_count, val_count = preprocess_and_cache()
        print(f"[DEBUG] 预处理函数返回: train={train_count}, val={val_count}")
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

# 导入共享模块中的 AlexNet
from shared.cnn.alex_net import AlexNet

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

# 创建 Dataset（数据已在 CPU 内存中，训练时移到 GPU）
class CachedDataset(Dataset):
    """从内存加载的数据集，训练时将数据移到 GPU"""
    def __init__(self, images, labels, device, augment=True):
        self.images = images
        self.labels = labels
        self.device = device
        self.augment = augment
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 从 CPU 内存移到 GPU
        image = self.images[idx].to(self.device)
        label = self.labels[idx].to(self.device)
        
        if self.augment:
            # RandomHorizontalFlip (50% 概率)
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[2])
            
            # RandomCrop with padding=4
            padded = torch.nn.functional.pad(image, (4, 4, 4, 4), mode='reflect')
            top = torch.randint(0, 9, (1,)).item()
            left = torch.randint(0, 9, (1,)).item()
            image = padded[:, top:top+224, left:left+224]
        
        return image, label

# 创建 Dataset 和 DataLoader
train_dataset = CachedDataset(train_images, train_labels, device, augment=True)
val_dataset = CachedDataset(val_images, val_labels, device, augment=False)

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
