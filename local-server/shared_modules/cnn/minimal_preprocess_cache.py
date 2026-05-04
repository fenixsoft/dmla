# MinimalPreprocessCache 类定义
# 从文档自动提取生成

import os
from PIL import Image

class MinimalPreprocessCache:
    """
    最小缓存策略：只执行 Resize，保存为 JPEG 格式
    
    缓存大小：约 600MB（而非原方案的 60GB）
    内存需求：训练时约 4GB（而非原方案的 67GB）
    """
    
    def __init__(self, data_dir, cache_dir):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.train_cache = os.path.join(cache_dir, 'train')
        self.val_cache = os.path.join(cache_dir, 'val')
        self.manifest_path = os.path.join(cache_dir, 'manifest.json')
        
    def preprocess_image(self, img_path, save_path):
        """单张图片预处理：Resize(224) → JPEG"""
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
        """预处理训练集（支持断点续传）"""
        train_dir = os.path.join(self.data_dir, 'train')
        classes = sorted(os.listdir(train_dir))
        
        os.makedirs(self.train_cache, exist_ok=True)
        total_count = 0
        
        for cls_idx, cls in enumerate(classes):
            cls_cache_dir = os.path.join(self.train_cache, cls)
            
            # 断点续传：检查已存在的类别目录
            if os.path.exists(cls_cache_dir):
                existing_files = [f for f in os.listdir(cls_cache_dir) if f.endswith('.JPEG')]
                if len(existing_files) >= 500:
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
        
        os.makedirs(self.val_cache, exist_ok=True)
        
        # 断点续传
        existing_files = [f for f in os.listdir(self.val_cache) if f.endswith('.JPEG')]
        start_idx = len(existing_files)
        
        if start_idx >= total_val:
            progress.update(total_val, message=f"验证集已缓存: {total_val} 张")
            return total_val, []
        
        labels = []
        progress.reset(total_steps=total_val, description="预处理验证集")
        
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
        """执行预处理（支持断点续传）"""
        start_time = time.time()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        train_count = self._preprocess_train_set(progress)
        val_count, val_labels = self._preprocess_val_set(progress)
        
        # 保存清单文件
        manifest = {
            'train_count': train_count,
            'val_count': val_count,
            'val_labels': val_labels if val_labels else self._load_existing_val_labels()
        }
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        elapsed = time.time() - start_time
        progress.complete(message=f"预处理完成: 训练集 {train_count} 张, 验证集 {val_count} 张, 耗时 {elapsed:.1f}s")
        
        return train_count, val_count

    def _load_existing_val_labels(self):
        """加载已有的验证集标签"""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
                return manifest.get('val_labels', [])
        return []
