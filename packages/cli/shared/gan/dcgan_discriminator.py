import torch
import torch.nn as nn


class DCGANDiscriminator(nn.Module):
    """
    DCGAN 判别器

    输入: 64×64×3 RGB 图像 (值域 [-1, 1])
    输出: 真假概率 [0, 1]

    架构: 卷积逐步下采样
    64×64 → 32×32 → 16×16 → 8×8 → 4×4 → 1×1
    """
    def __init__(self, img_channels=3):
        super(DCGANDiscriminator, self).__init__()

        self.main = nn.Sequential(
            # 3 × 64 × 64 → 64 × 32 × 32 (无 BatchNorm)
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 × 32 × 32 → 128 × 16 × 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 × 16 × 16 → 256 × 8 × 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 × 8 × 8 → 512 × 4 × 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 512 × 4 × 4 → 1 × 1 × 1
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.main(img).view(-1)