```python runnable gpu extract-class="DCGANGenerator"
import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    """
    DCGAN 生成器
    
    输入: 噪声向量 z (latent_dim 维)
    输出: 64×64×3 RGB 图像 (值域 [-1, 1])
    
    架构: 转置卷积逐步上采样
    1×1 → 4×4 → 8×8 → 16×16 → 32×32 → 64×64
    """
    def __init__(self, latent_dim=100, img_channels=3):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # 输入: latent_dim × 1 × 1 → 512 × 4 × 4
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 512 × 4 × 4 → 256 × 8 × 8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 256 × 8 × 8 → 128 × 16 × 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 128 × 16 × 16 → 64 × 32 × 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64 × 32 × 32 → 3 × 64 × 64
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        # 将噪声向量 reshape 为 4D 张量: (batch, latent_dim, 1, 1)
        return self.main(z.view(z.size(0), z.size(1), 1, 1))

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

# 创建生成器实例
generator = DCGANGenerator(latent_dim=100)

# 打印模型结构
print("DCGAN 生成器结构:")
print(generator)

# 计算参数量
total_params = sum(p.numel() for p in generator.parameters())
print(f"\n生成器参数量: {total_params:,}")

# 测试前向传播
noise = torch.randn(16, 100)
fake_images = generator(noise)
print(f"输入噪声形状: {noise.shape}")
print(f"生成图像形状: {fake_images.shape}")
print(f"输出值域: [{fake_images.min():.2f}, {fake_images.max():.2f}]")

# 创建判别器实例
discriminator = DCGANDiscriminator()

# 打印模型结构
print("DCGAN 判别器结构:")
print(discriminator)

# 计算参数量
total_params = sum(p.numel() for p in discriminator.parameters())
print(f"\n判别器参数量: {total_params:,}")

# 测试前向传播
fake_images = torch.randn(16, 3, 64, 64)
output = discriminator(fake_images)
print(f"输入图像形状: {fake_images.shape}")
print(f"判别输出形状: {output.shape}")
print(f"输出值域: [{output.min():.4f}, {output.max():.4f}]")
```
