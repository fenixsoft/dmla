---
title: Python 知识库
---

# Python 知识库

欢迎来到 Python 知识库！这里收集了 Python 编程相关的学习笔记和教程。

## 目录

- [装饰器详解](./decorators.md) - 深入理解 Python 装饰器

## 在线运行代码

本地部署模式下，代码块支持在线运行：

```python runnable
# 点击下方 "Run" 按钮运行此代码
print("Hello, IdeaSpaces!")
```

```python runnable gpu
# GPU 加速示例 (需要本地部署 + NVIDIA GPU)
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(100, 100).cuda()
    print(f"Tensor shape: {x.shape}")
else:
    print("CUDA 不可用，请检查 GPU 配置")
```