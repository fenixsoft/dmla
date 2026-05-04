# LMDB 数据集加载类
# 使用 LMDB 键值存储实现高效随机读取

import os
import io
import json
import lmdb
import struct
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class LMDBDataset(Dataset):
    """
    LMDB 存储的数据集

    优势：
    - 单个大文件，避免大量小文件的随机 I/O
    - 内存映射（mmap），零拷贝读取
    - 多进程友好（无锁读取）

    数据结构：
    - 键：图片索引（uint64，8字节）
    - 值：JPEG bytes + label（前 4 字节为 label，后面为 JPEG 数据）
    """

    # ImageNet Normalize 参数
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, lmdb_path, augment=True, normalize_on_gpu=False):
        """
        Args:
            lmdb_path: LMDB 数据库路径
            augment: 是否执行数据增强（训练集 True，验证集 False）
            normalize_on_gpu: 是否将 Normalize 移到 GPU 执行
        """
        self.lmdb_path = lmdb_path
        self.augment = augment
        self.normalize_on_gpu = normalize_on_gpu

        # 打开 LMDB 环境（只读模式）
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256
        )

        # 获取数据数量
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']

        # 加载 manifest（标签信息）
        manifest_path = os.path.join(os.path.dirname(lmdb_path), 'manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

        # 数据变换
        self.to_tensor = transforms.ToTensor()

        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(224, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.augment_transform = None

        self.normalize = None if normalize_on_gpu else transforms.Normalize(mean=self.MEAN, std=self.STD)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        获取单张图片

        Returns:
            image: tensor [3, 224, 224]
            label: int
        """
        # LMDB 读取（零拷贝）
        with self.env.begin() as txn:
            # 键为 8 字节的 uint64
            key = struct.pack('>Q', idx)
            value = txn.get(key)

            if value is None:
                raise IndexError(f"Index {idx} out of range")

            # 解析值：前 4 字节为 label（int32），后面为 JPEG 数据
            label = struct.unpack('>i', value[:4])[0]
            jpeg_data = value[4:]

        # JPEG 解码
        image = Image.open(io.BytesIO(jpeg_data)).convert('RGB')

        # ToTensor
        image = self.to_tensor(image)

        # 数据增强
        if self.augment_transform:
            image = self.augment_transform(image)

        # Normalize
        if self.normalize:
            image = self.normalize(image)

        return image, label

    def get_normalize_params(self):
        """获取 Normalize 参数（供 GPU 执行时使用）"""
        return torch.tensor(self.MEAN).view(3, 1, 1), torch.tensor(self.STD).view(3, 1, 1)

    def close(self):
        """关闭 LMDB 环境"""
        self.env.close()


class LMDBValDataset(Dataset):
    """
    LMDB 验证集

    与训练集结构相同，但无数据增强
    """

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, lmdb_path, normalize_on_gpu=False):
        self.lmdb_path = lmdb_path
        self.normalize_on_gpu = normalize_on_gpu

        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256
        )

        with self.env.begin() as txn:
            self.length = txn.stat()['entries']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if not normalize_on_gpu:
            self.transform = transforms.Compose([
                self.transform,
                transforms.Normalize(mean=self.MEAN, std=self.STD)
            ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = struct.pack('>Q', idx)
            value = txn.get(key)

            if value is None:
                raise IndexError(f"Index {idx} out of range")

            label = struct.unpack('>i', value[:4])[0]
            jpeg_data = value[4:]

        image = Image.open(io.BytesIO(jpeg_data)).convert('RGB')
        image = self.transform(image)

        return image, label

    def get_normalize_params(self):
        return torch.tensor(self.MEAN).view(3, 1, 1), torch.tensor(self.STD).view(3, 1, 1)

    def close(self):
        self.env.close()


# 需要导入 io 模块（已在文件开头导入）