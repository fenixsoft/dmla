# TinyImageNetDataset 类定义
# 从文档自动提取生成

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet 200 数据集加载器

    训练集按类别子目录读取，验证集从标注文件解析标签。
    支持自定义预处理变换，适配 AlexNet 训练需求。
    """
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        self.samples = []
        self.classes = []

        if is_train:
            train_dir = os.path.join(root_dir, 'train')
            self.classes = sorted(os.listdir(train_dir))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

            for cls in self.classes:
                cls_dir = os.path.join(train_dir, cls)
                images_dir = os.path.join(cls_dir, 'images')
                if os.path.exists(images_dir):
                    for img_name in os.listdir(images_dir):
                        if img_name.endswith('.JPEG'):
                            self.samples.append((
                                os.path.join(images_dir, img_name),
                                self.class_to_idx[cls]
                            ))
        else:
            val_dir = os.path.join(root_dir, 'val')
            val_images_dir = os.path.join(val_dir, 'images')
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')

            if os.path.exists(val_annotations):
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0]
                            cls = parts[1]
                            if cls not in self.classes:
                                self.classes.append(cls)
                            self.samples.append((
                                os.path.join(val_images_dir, img_name),
                                self.classes.index(cls)
                            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
