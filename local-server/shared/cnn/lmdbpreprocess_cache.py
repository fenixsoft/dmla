# LMDBPreprocessCache 类定义
# 从文档自动提取生成

import os
from PIL import Image

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
