# ImageVAE 定义
# 从文档自动提取生成

import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from dmla_progress import ProgressReporter
from PIL import Image

class ImageVAE(nn.Module):
    """
    用于 MNIST 图像生成的 VAE

    网络结构:
    - 编码器: 784 → 512 → 256 → (μ, σ)
    - 解码器: z → 256 → 512 → 784

    潜在空间维度: 20
    """
    def __init__(self, latent_dim=20):
        super().__init__()

        # 编码器（更深的网络，提取更丰富的特征）
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # 解码器（对称结构）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()  # 输出像素概率
        )

        self.latent_dim = latent_dim

    def encode(self, x):
        """编码过程"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """重参数化"""
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        """解码过程"""
        return self.decoder(z)

    def forward(self, x):
        """完整流程"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, num_samples):
        """生成新样本"""
        z = torch.randn(num_samples, self.latent_dim)
        return self.decode(z)
