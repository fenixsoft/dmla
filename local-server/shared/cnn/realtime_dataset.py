# 实时数据增强 Dataset 类
# 从缓存读取 JPEG，实时执行数据增强

import os
import json
import time
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 性能日志文件（全局）
PERF_LOG_PATH = '/data/models/alexnet/dataset_perf_log.txt'
_perf_log_file = None
_perf_counter = 0

def _get_perf_log():
    """获取性能日志文件（懒加载）"""
    global _perf_log_file
    if _perf_log_file is None:
        os.makedirs('/data/models/alexnet', exist_ok=True)
        _perf_log_file = open(PERF_LOG_PATH, 'w')
        _perf_log_file.write("idx,jpeg_decode_ms,to_tensor_ms,augment_ms,total_ms\n")
    return _perf_log_file


class RealtimeAugmentDataset(Dataset):
    """
    实时执行数据增强的 Dataset

    流程：
    1. 从缓存读取 JPEG（224×224）
    2. CPU 执行 ToTensor + RandomFlip + RandomCrop + ColorJitter
    3. Normalize 参数提供，可在 GPU 执行

    特点：
    - 内存占用低：只缓存当前 batch
    - 数据增强随机性：每次 epoch 看到不同版本
    - 多线程友好：配合 DataLoader num_workers
    """

    # ImageNet Normalize 参数
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, cache_dir, augment=True, normalize_on_gpu=False):
        """
        Args:
            cache_dir: 缓存目录路径
            augment: 是否执行数据增强（训练集 True，验证集 False）
            normalize_on_gpu: 是否将 Normalize 移到 GPU 执行
        """
        self.cache_dir = cache_dir
        self.augment = augment
        self.normalize_on_gpu = normalize_on_gpu

        # 加载清单文件
        manifest_path = os.path.join(cache_dir, 'manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {'val_labels': []}

        # 加载图片路径和标签
        self.image_paths = []
        self.labels = []

        train_cache = os.path.join(cache_dir, 'train')
        if os.path.exists(train_cache):
            # 加载类别映射
            wnids_path = '/data/datasets/tiny-imagenet-200/wnids.txt'
            if os.path.exists(wnids_path):
                with open(wnids_path, 'r') as f:
                    wnids = [line.strip() for line in f.readlines()]
                class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
            else:
                class_to_idx = {}

            classes = sorted(os.listdir(train_cache))
            for cls_idx, cls in enumerate(classes):
                cls_dir = os.path.join(train_cache, cls)
                if os.path.isdir(cls_dir):
                    for img_name in os.listdir(cls_dir):
                        if img_name.endswith('.JPEG'):
                            self.image_paths.append(os.path.join(cls_dir, img_name))
                            self.labels.append(class_to_idx.get(cls, cls_idx))

        # CPU 数据增强（训练集）- 分离各个变换以便测量耗时
        if augment:
            self.to_tensor = transforms.ToTensor()
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(224, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            # 验证集预处理（无增强）
            self.to_tensor = transforms.ToTensor()
            self.augment_transform = None

        # Normalize（可在 CPU 或 GPU 执行）
        self.normalize = None if normalize_on_gpu else transforms.Normalize(mean=self.MEAN, std=self.STD)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取单张图片（带详细耗时测量）

        Returns:
            image: tensor [3, 224, 224]
            label: int
        """
        global _perf_counter

        start_time = time.time()
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 1. JPEG 解码耗时
        jpeg_start = time.time()
        image = Image.open(img_path).convert('RGB')
        jpeg_time = time.time() - jpeg_start

        # 2. ToTensor 耗时
        to_tensor_start = time.time()
        image = self.to_tensor(image)
        to_tensor_time = time.time() - to_tensor_start

        # 3. 数据增强耗时（如果有）
        augment_time = 0.0
        if self.augment_transform:
            augment_start = time.time()
            image = self.augment_transform(image)
            augment_time = time.time() - augment_start

        # 4. Normalize 耗时（如果在 CPU 执行）
        if self.normalize:
            image = self.normalize(image)

        total_time = time.time() - start_time

        # 采样记录日志（每100张记录一次，避免日志过大）
        _perf_counter += 1
        if _perf_counter % 100 == 0:
            log = _get_perf_log()
            log.write(f"{idx},{jpeg_time*1000:.1f},{to_tensor_time*1000:.1f},{augment_time*1000:.1f},{total_time*1000:.1f}\n")
            log.flush()

        return image, label

    def get_normalize_params(self):
        """获取 Normalize 参数（供 GPU 执行时使用）"""
        return torch.tensor(self.MEAN).view(3, 1, 1), torch.tensor(self.STD).view(3, 1, 1)


class RealtimeValDataset(Dataset):
    """
    验证集 Dataset（从缓存读取）

    扁平化结构：val/val_<idx>.JPEG
    """

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, cache_dir, normalize_on_gpu=False):
        self.cache_dir = cache_dir
        self.val_cache = os.path.join(cache_dir, 'val')
        self.normalize_on_gpu = normalize_on_gpu

        # 加载清单文件获取标签
        manifest_path = os.path.join(cache_dir, 'manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                self.labels = manifest.get('val_labels', [])
        else:
            self.labels = []

        # 构建图片路径列表
        self.image_paths = []
        if os.path.exists(self.val_cache):
            for i in range(len(self.labels)):
                img_path = os.path.join(self.val_cache, f'val_{i}.JPEG')
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)

        # 验证集变换（无增强）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if not normalize_on_gpu:
            self.transform = transforms.Compose([
                self.transform,
                transforms.Normalize(mean=self.MEAN, std=self.STD)
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, label

    def get_normalize_params(self):
        return torch.tensor(self.MEAN).view(3, 1, 1), torch.tensor(self.STD).view(3, 1, 1)