# 最小缓存预处理类
# 只执行 Resize(224)，保存为 JPEG 格式

import os
import json
from PIL import Image
import time

class MinimalPreprocessCache:
    """
    最小缓存策略：只执行 Resize，保存为 JPEG 格式

    与原 PreprocessCache 的区别：
    - 原方案：Resize + ToTensor → float32 tensor → 60GB
    - 新方案：Resize → JPEG → 600MB

    性能权衡：
    - 磁盘：600MB vs 60GB（减少 100 倍）
    - 加载：需解码 JPEG（增加 CPU 开销）
    - 增强：实时执行（每次不同）
    """

    def __init__(self, data_dir, cache_dir):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.train_cache = os.path.join(cache_dir, 'train')
        self.val_cache = os.path.join(cache_dir, 'val')
        self.manifest_path = os.path.join(cache_dir, 'manifest.json')

    def preprocess_image(self, img_path, save_path):
        """
        单张图片预处理：Resize → JPEG

        Args:
            img_path: 原始图片路径
            save_path: 缓存保存路径
        """
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        img.save(save_path, 'JPEG', quality=95)

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
        """
        预处理训练集（支持断点续传）

        保持原有目录结构：train/<class_name>/<image>.JPEG
        """
        train_dir = os.path.join(self.data_dir, 'train')
        classes = sorted(os.listdir(train_dir))

        os.makedirs(self.train_cache, exist_ok=True)

        total_count = 0

        for cls_idx, cls in enumerate(classes):
            cls_cache_dir = os.path.join(self.train_cache, cls)

            # 断点续传：检查已存在的类别目录
            if os.path.exists(cls_cache_dir):
                existing_files = [f for f in os.listdir(cls_cache_dir) if f.endswith('.JPEG')]
                if len(existing_files) >= 500:  # 每类约 500 张
                    total_count += len(existing_files)
                    progress.update(cls_idx + 1, message=f"跳过已缓存类别 {cls_idx+1}/200: {cls}")
                    continue

            os.makedirs(cls_cache_dir, exist_ok=True)

            images_dir = os.path.join(train_dir, cls, 'images')
            if not os.path.exists(images_dir):
                continue

            count = 0
            for img_name in os.listdir(images_dir):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(images_dir, img_name)
                    save_path = os.path.join(cls_cache_dir, img_name)

                    try:
                        self.preprocess_image(img_path, save_path)
                        count += 1
                        total_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to process {img_path}: {e}")

            progress.update(cls_idx + 1, message=f"预处理类别 {cls_idx+1}/200: {cls} ({count} 张)")

        return total_count

    def _preprocess_val_set(self, progress):
        """
        预处理验证集（支持断点续传）

        扁平化保存：val/val_<idx>.JPEG
        """
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

        os.makedirs(self.val_cache, exist_ok=True)

        # 断点续传：检查已存在的文件数
        existing_files = [f for f in os.listdir(self.val_cache) if f.endswith('.JPEG')]
        start_idx = len(existing_files)

        if start_idx >= total_val:
            progress.update(total_val, message=f"验证集已缓存: {total_val} 张")
            return total_val, []

        labels = []
        for line_idx in range(start_idx, total_val):
            parts = val_lines[line_idx].strip().split('\t')
            if len(parts) >= 2:
                img_name = parts[0]
                img_path = os.path.join(val_images_dir, img_name)
                save_path = os.path.join(self.val_cache, f'val_{line_idx}.JPEG')

                if os.path.exists(img_path):
                    try:
                        self.preprocess_image(img_path, save_path)
                        labels.append(class_to_idx.get(parts[1], 0))
                    except Exception as e:
                        print(f"处理图片出现异常 {img_path}: {e}")

                if (line_idx + 1) % 100 == 0 or line_idx == total_val - 1:
                    progress.update(line_idx + 1, message=f"预处理验证集 {line_idx+1}/{total_val}")

        return total_val, labels

    def run(self, progress):
        """
        执行预处理（支持断点续传）

        Returns:
            (train_count, val_count) 预处理的图片数量
        """
        start_time = time.time()
        os.makedirs(self.cache_dir, exist_ok=True)

        # 阶段 1：训练集预处理
        train_count = self._preprocess_train_set(progress)

        # 阶段 2：验证集预处理
        val_count, val_labels = self._preprocess_val_set(progress)

        # 保存清单文件
        manifest = {
            'train_count': train_count,
            'val_count': val_count,
            'cache_size_mb': self._estimate_cache_size(),
            'val_labels': val_labels if val_labels else self._load_existing_val_labels()
        }
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f)

        elapsed = time.time() - start_time
        progress.complete(message=f"预处理完成: 训练集 {train_count} 张, 验证集 {val_count} 张, 耗时 {elapsed:.1f}s")

        return train_count, val_count

    def _estimate_cache_size(self):
        """估算缓存大小（MB）"""
        total_size = 0
        for root, dirs, files in os.walk(self.cache_dir):
            for f in files:
                if f.endswith('.JPEG'):
                    total_size += os.path.getsize(os.path.join(root, f))
        return total_size / 1024 / 1024

    def _load_existing_val_labels(self):
        """加载已有的验证集标签"""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
                return manifest.get('val_labels', [])
        return []