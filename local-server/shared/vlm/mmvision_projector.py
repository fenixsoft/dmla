# MMVisionProjector 定义
# 从文档自动提取生成

import torch
import torch.nn as nn

class MMVisionProjector(nn.Module):
    """视觉-语言投影层：将视觉编码器的输出映射到语言模型的嵌入空间"""
    def __init__(self, in_dim=768, out_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)
