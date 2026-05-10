# GAN 图像生成实验

本次工程实训中，笔者将与你一同使用 PyTorch 实现 DCGAN（Deep Convolutional GAN）的完整训练流程，从数据预处理到模型定义、从对抗训练到图像生成，通过实践来理解生成式对抗网络的训练机制与工程考量，并最终训练出能够生成卡通头像的生成器模型。

## 实验准备

在开始实验之前，请确保已[挂载数据目录](../../sandbox.md#数据管理)并下载好 Cartoon Face 数据集，你可以通过 `DMLA-CLI` 工具自动完成该工作：

```bash
# 选择 "下载数据集" -> 选择 "Cartoon Face"
dmla data
```

验证数据集是否已正确下载，并检查其结构。Cartoon Face 数据集包含数万张卡通头像图片，是用于人脸生成任务的数据集，图片涵盖了多种风格和表情的卡通人物面部。

```python runnable gpu
import os

# 检查数据目录是否存在（DATA_DIR 由 kernel 自动注入）
data_dir = os.path.join(DATA_DIR, 'datasets', 'cartoon-face')

if os.path.exists(data_dir):
    print("数据集目录已存在")
    
    # 递归统计图片数量
    image_count = 0
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(image_extensions) and not f.startswith('.'):
                image_count += 1
    
    print(f"图片总数: {image_count}")
    
    # 检查几张样本图片的尺寸信息
    from PIL import Image
    sample_images = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(image_extensions) and not f.startswith('.'):
                sample_images.append(os.path.join(root, f))
                if len(sample_images) >= 3:
                    break
        if len(sample_images) >= 3:
            break
    
    if sample_images:
        for img_path in sample_images:
            img = Image.open(img_path)
            print(f"样本图片: {os.path.basename(img_path)}, 尺寸: {img.size}, 格式: {img.mode}")
else:
    print("数据集未下载，请运行 'dmla data' 下载数据集")
```

## 第一阶段：数据预处理

GAN 训练的数据预处理与分类任务有显著差异。分类任务中数据增强是为了增加训练多样性防止过拟合，而 GAN 的数据预处理首要考虑的是将图像归一化到与生成器输出范围一致的区间。因为 DCGAN 生成器最后一层使用 $\tanh$ 激活函数，输出值域为 $[-1, 1]$，所以真实图像也需要归一化到相同范围，否则判别器将无法有效区分真实与生成图像。

本阶段的工程决策围绕以下几点展开：

- **图像尺寸选择**：DCGAN 原论文使用 64×64 的图像尺寸，本实验也采用这一规格。选择 64×64 而不是更高分辨率（如 128×128 或 256×256）是基于训练效率的考量，而非简单照搬论文参数。GAN 需要同时训练两个网络，且训练轮数通常远多于分类任务（本实验使用 100 个 epoch），每个 epoch 的处理时间直接决定了总训练时间。从计算量角度看，128×128 的图像面积是 64×64 的 4 倍，每层卷积的计算量至少翻 4 倍，整个网络的训练时间可能增加 8-10 倍。64×64 提供了最快的反馈循环，可以在 15-20 分钟内看到训练效果，适合首次实践 GAN 的读者。
- **归一化范围**：使用 `Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` 将图像像素从 $[0, 1]$ 映射到 $[-1, 1]$。这是 GAN 训练的标准归一化方式，与分类任务常用的 ImageNet 归一化不同。分类任务用 per-channel 均值和方差归一化是为了让各通道特征分布相似；GAN 用固定值归一化是为了匹配 $\tanh$ 的输出范围。如果使用 ImageNet 归一化（`mean=[0.485, 0.456, 0.406]`），真实图像和生成图像的数值范围不匹配，判别器将无法学习有效的鉴别特征。
- **数据增强**：仅使用随机水平翻转。GAN 不需要分类任务中的丰富数据增强（裁剪、颜色扰动等），因为生成器已经在学习数据分布的全部特征。过多的数据增强会改变真实数据的分布，反而让生成器学习一个被扰动过的分布，生成质量下降。水平翻转是唯一对 GAN 训练安全且有益的增强方式，因为人脸在水平方向上大致对称。
- **不需要缓存**：与 AlexNet 实验不同，GAN 实验不需要 LMDB 缓存。64×64 的图像加载和解码非常快，PIL 解码一张 64×64 JPEG 图片约 0.1ms，加上 Resize 和 Normalize 的开销也不超过 1ms。数万张图片的 DataLoader 预取足以覆盖 I/O 等待，无需像 224×224 的 AlexNet 那样使用 LMDB 或 DALI 优化。

```python runnable gpu
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CartoonFaceDataset(Dataset):
    """Cartoon Face 数据集
    
    递归扫描目录下的所有图片文件，
    自动适配不同的目录结构
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(image_extensions) and not f.startswith('.'):
                    self.image_paths.append(os.path.join(root, f))
        
        print(f"找到 {len(self.image_paths)} 张图片")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 数据预处理变换
transform = transforms.Compose([
    transforms.Resize((64, 64)),            # 缩放至 64×64
    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
    transforms.ToTensor(),                  # 转为张量 [0, 1]
    transforms.Normalize(                   # 归一化至 [-1, 1]
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# 创建数据集和数据加载器（DATA_DIR 由 kernel 自动注入）
data_dir = os.path.join(DATA_DIR, 'datasets', 'cartoon-face')
if os.path.exists(data_dir):
    dataset = CartoonFaceDataset(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"批次数: {len(dataloader)}")
    print(f"批次大小: 128")
    
    # 检查一个批次的数据
    sample_batch = next(iter(dataloader))
    print(f"批次张量形状: {sample_batch.shape}")
    print(f"数值范围: [{sample_batch.min():.2f}, {sample_batch.max():.2f}]")
else:
    print("数据集未下载，请先运行 'dmla data' 下载数据集")
```

## 第二阶段：模型定义

DCGAN 是将卷积神经网络系统性地引入 GAN 的里程碑式改进，我们在[GAN 理论章节](gan.md#GAN 变体)中已经介绍过它的核心思想。原始 GAN 使用 MLP（多层感知器）结构，无法有效捕捉图像的空间结构特征。DCGAN 的核心改进是使用卷积层替代全连接层：生成器使用转置卷积（Transposed Convolution）逐步上采样，从低维噪声向量扩展到高维图像；判别器使用标准卷积逐步下采样，从高维图像压缩到真假判断。

DCGAN 论文给出了一系列经过大量实验验证的架构设计准则，本实验严格遵循这些准则：

1. **使用卷积替代全连接层**：全连接层忽略了图像的空间结构，卷积层天然适合处理二维图像的局部特征。
2. **生成器使用转置卷积上采样**：避免使用上池化（Upsampling）+ 卷积的组合，转置卷积直接学习上采样的方式，生成质量更好。
3. **判别器使用步长卷积下采样**：避免使用池化层（MaxPool/AvgPool），步长卷积让网络自己学习下采样的方式，判别能力更强。
4. **批量归一化（Batch Normalization）**：生成器除输出层外都使用，判别器除输入层外都使用。生成器输出层不使用 BN 是因为 BN 会将输出分布强制归一化，削弱 $\tanh$ 的表达能力；判别器输入层不使用 BN 是因为 BN 会破坏输入数据的原始分布特征，影响对真实样本的鉴别能力。
5. **激活函数选择**：生成器中间层使用 ReLU，输出层使用 $\tanh$（输出范围 $[-1, 1]$，匹配归一化后的真实图像）；判别器中间层使用 LeakyReLU（斜率 0.2），输出层使用 Sigmoid（输出概率 $[0, 1]$）。LeakyReLU 比 ReLU 更适合判别器，因为它在负值区间保留了小梯度（$\alpha = 0.2$），防止梯度完全消失，这对判别器学习"假样本的特征"至关重要。
6. **去掉卷积层的 bias**：使用 BN 的层不需要 bias，因为 BN 本身有偏移参数 $\beta$，两者功能重叠。

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
```

```python runnable gpu extract-class="DCGANDiscriminator"
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

DCGAN 的权重初始化需要特别关注。GAN 的训练稳定性高度依赖初始权重，错误的初始化可能导致训练初期梯度消失或模式崩溃。DCGAN 论文推荐以下初始化策略：卷积层和转置卷积层的权重从正态分布 $\mathcal{N}(0, 0.02)$ 采样，标准差 0.02 比默认的 0.01 略大，提供足够的初始梯度信号又不至于导致梯度爆炸；BatchNorm 层的缩放参数 $\gamma$ 初始化为 1，偏移参数 $\beta$ 初始化为 0，这是 PyTorch 的默认设置，无需额外修改。

```python runnable gpu
import torch.nn as nn

def weights_init_normal(m):
    """DCGAN 权重初始化
    
    卷积/转置卷积层: N(0, 0.02)
    BatchNorm 层: weight=1, bias=0
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 导入共享模块中的模型
from shared.gan.dcgan_generator import DCGANGenerator
from shared.gan.dcgan_discriminator import DCGANDiscriminator

generator = DCGANGenerator(latent_dim=100)
discriminator = DCGANDiscriminator()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# 验证初始化结果
print("权重初始化验证:")
for name, param in generator.named_parameters():
    if 'weight' in name and param.requires_grad:
        print(f"  {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
```

## 第三阶段：模型训练

GAN 训练是本实验最关键也最具挑战性的环节。与 AlexNet 等分类网络的训练相比，GAN 训练有两个本质区别：第一，GAN 需要同时优化两个相互对抗的网络，而非单一目标；第二，GAN 的训练目标不是最小化某个明确的损失值，而是让两个网络在对抗中达到动态平衡。这两个区别使得 GAN 训练远比分类网络训练不稳定，也是 GAN 训练被称为"黑魔法"的原因。

本阶段的工程决策围绕训练稳定性展开：

- **损失函数选择**：使用二元交叉熵损失（Binary Cross Entropy Loss，BCE Loss）。这是 GAN 训练的标准损失函数，将判别器视为二分类器，对真实样本标签为 1，对生成样本标签为 0。BCE Loss 的梯度特性恰好匹配 GAN 的训练需求——当判别器输出接近目标时梯度较小（稳定），远离目标时梯度较大（快速学习）。

- **标签平滑**：将真实样本的目标标签从 1.0 降为 0.9，即"单侧标签平滑"。这个工程技巧的原理是：判别器对真实样本输出 1.0 表示"绝对确信这是真图"，这种极端置信度会导致梯度信号消失（判别器已经完美，生成器无法从它获得学习信号）和过拟合真实样本的特定细节。将目标降为 0.9 后，判别器保留了一定的不确定性，为生成器留出梯度空间。注意只平滑真实标签而不平滑假标签（仍为 0.0），因为平滑假标签会让判别器认为"假图也有点真实"，削弱判别能力。

- **优化器参数**：使用 Adam 优化器，但参数与分类任务不同。学习率 0.0002（分类任务常用 0.01），$\beta_1 = 0.5$（分类任务常用 0.9）。降低 $\beta_1$ 是 DCGAN 论文的关键发现：$\beta_1$ 控制动量项的衰减率，较高的 $\beta_1$（如 0.9）会让优化器记住太多历史梯度方向，导致训练震荡；降低到 0.5 减少了动量的影响，让每步更新更依赖当前梯度，训练更稳定。

- **训练比例**：判别器和生成器各训练 1 步（1:1 比例）。这是最简单的训练策略，也是最常用的比例。有些实践建议使用 5:1 的比例（判别器训练 5 步，生成器训练 1 步），让判别器保持适度优势以提供更好的梯度信号，但对于 DCGAN 在 64×64 图像上的训练，1:1 比例通常足够稳定，且训练效率更高。

```python runnable gpuonly timeout=unlimited
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 导入共享模块中的模型
from shared.gan.dcgan_generator import DCGANGenerator
from shared.gan.dcgan_discriminator import DCGANDiscriminator

# 导入数据集（DATA_DIR 由 kernel 自动注入）
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CartoonFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(image_extensions) and not f.startswith('.'):
                    self.image_paths.append(os.path.join(root, f))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 权重初始化
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# === 训练配置 ===
latent_dim = 100
batch_size = 128
num_epochs = 100
lr = 0.0002
beta1 = 0.5
real_label_smooth = 0.9  # 单侧标签平滑

# === 创建模型 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")

generator = DCGANGenerator(latent_dim=latent_dim).to(device)
discriminator = DCGANDiscriminator().to(device)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

print(f"生成器参数量: {sum(p.numel() for p in generator.parameters()):,}")
print(f"判别器参数量: {sum(p.numel() for p in discriminator.parameters()):,}")

# === 损失函数与优化器 ===
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# 创建固定噪声用于追踪训练进度（同一组噪声在每个阶段生成样本，便于对比训练效果）
fixed_noise = torch.randn(64, latent_dim, device=device)

# === 创建数据加载器 ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir = os.path.join(DATA_DIR, 'datasets', 'cartoon-face')
if not os.path.exists(data_dir):
    print("错误: 数据集未下载，请先运行 'dmla data' 下载数据集")
else:
    dataset = CartoonFaceDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # 创建输出目录
    sample_dir = os.path.join(DATA_DIR, 'outputs', 'training_samples')
    os.makedirs(sample_dir, exist_ok=True)
    model_dir = os.path.join(DATA_DIR, 'models', 'gan', 'dcgan')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    final_dir = os.path.join(model_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    
    # 训练进度
    progress = ProgressReporter(total_steps=num_epochs, description="训练 DCGAN")
    
    # 训练日志
    log_path = os.path.join(model_dir, 'training_log.txt')
    log_entries = []
    
    print(f"开始训练: {num_epochs} epochs, {len(dataloader)} batches/epoch")
    print(f"训练配置: lr={lr}, beta1={beta1}, batch_size={batch_size}, label_smooth={real_label_smooth}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)
            
            # ========== 训练判别器 ==========
            optimizer_D.zero_grad()
            
            # 真实样本: 目标标签使用标签平滑 (0.9 而非 1.0)
            real_labels = torch.full((batch_size_actual,), real_label_smooth, device=device)
            real_output = discriminator(real_images)
            loss_D_real = criterion(real_output, real_labels)
            
            # 生成样本: 目标标签为 0
            noise = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_images = generator(noise)
            fake_labels = torch.zeros(batch_size_actual, device=device)
            # detach() 阻止梯度流向生成器，判别器只更新自己的参数
            fake_output = discriminator(fake_images.detach())
            loss_D_fake = criterion(fake_output, fake_labels)
            
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()
            
            # ========== 训练生成器 ==========
            optimizer_G.zero_grad()
            
            # 重新计算判别器对假样本的输出（使用更新后的判别器权重）
            # 生成器希望判别器将假样本判为真实（目标标签 1.0，不平滑）
            fake_output_for_G = discriminator(fake_images)
            target_labels = torch.full((batch_size_actual,), 1.0, device=device)
            loss_G = criterion(fake_output_for_G, target_labels)
            loss_G.backward()
            optimizer_G.step()
        
        epoch_time = time.time() - epoch_start
        
        # 记录本 epoch 损失
        log_entries.append({
            'epoch': epoch + 1,
            'G_loss': loss_G.item(),
            'D_loss': loss_D.item(),
            'time': epoch_time
        })
        
        progress.update(epoch + 1, message=f"Epoch {epoch+1}/{num_epochs}: G_loss={loss_G.item():.4f}, D_loss={loss_D.item():.4f}, time={epoch_time:.1f}s")
        print(f"Epoch [{epoch+1}/{num_epochs}] G_loss: {loss_G.item():.4f} D_loss: {loss_D.item():.4f} Time: {epoch_time:.1f}s")
        
        # 每 10 epoch 保存训练样本图片
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_samples = generator(fixed_noise)
            # 反归一化 [-1, 1] → [0, 1]
            fake_samples = (fake_samples + 1) / 2.0
            from torchvision.utils import save_image
            save_image(fake_samples, os.path.join(sample_dir, f'epoch_{epoch+1}.png'), nrow=8, padding=2)
            print(f"  -> 保存训练样本: epoch_{epoch+1}.png")
        
        # 每 20 epoch 保存 checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'G_loss': loss_G.item(),
                'D_loss': loss_D.item(),
            }, os.path.join(checkpoints_dir, f'epoch_{epoch+1}.pth'))
            print(f"  -> 保存 checkpoint: epoch_{epoch+1}.pth")
    
    # 保存最终模型
    torch.save(generator.state_dict(), os.path.join(final_dir, 'dcgan_generator_cartoon_face.pth'))
    progress.complete(message=f"训练完成！G_loss: {loss_G.item():.4f}, D_loss: {loss_D.item():.4f}")
    
    # 保存训练日志
    with open(log_path, 'w') as f:
        f.write("epoch,g_loss,d_loss,time\n")
        for entry in log_entries:
            f.write(f"{entry['epoch']},{entry['G_loss']:.4f},{entry['D_loss']:.4f},{entry['time']:.1f}\n")
    print(f"训练日志已保存: {log_path}")
    print(f"最终模型已保存: {os.path.join(final_dir, 'dcgan_generator_cartoon_face.pth')}")
```

## 第四阶段：推理评估

训练完成后，使用生成器从随机噪声生成卡通头像图像。GAN 的推理阶段与分类模型截然不同：分类模型需要输入一张真实图像并输出类别，GAN 生成器只需要输入一个随机噪声向量就能生成全新图像，无需任何真实数据输入。这正是生成式模型的魅力所在——模型学会的不是"识别"而是"创造"。

推理阶段的工程要点如下：

1. **模型加载**：优先加载最终模型，其次加载 checkpoint。如果都找不到，则使用未训练的随机权重模型（仅供测试，生成结果将是无意义的噪声图像）。
2. **噪声维度**：必须与训练时的 `latent_dim` 一致（本实验为 100 维），否则模型无法正确处理输入。
3. **反归一化**：生成器输出值域为 $[-1, 1]$（$\tanh$ 激活函数），显示和保存图像时需要反归一化到 $[0, 1]$，即 $(x + 1) / 2$。
4. **生成数量**：生成 10 张卡通头像，展示模型在不同随机输入下的生成多样性。

```python runnable gpuonly
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image

# 导入共享模块中的生成器
from shared.gan.dcgan_generator import DCGANGenerator

# 加载训练好的模型（DATA_DIR 由 kernel 自动注入）
latent_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = DCGANGenerator(latent_dim=latent_dim).to(device)

model_path = os.path.join(DATA_DIR, 'models', 'gan', 'dcgan', 'final', 'dcgan_generator_cartoon_face.pth')
checkpoint_dir = os.path.join(DATA_DIR, 'models', 'gan', 'dcgan', 'checkpoints')

if os.path.exists(model_path):
    generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("加载最终生成器模型")
elif os.path.exists(checkpoint_dir):
    # 尝试加载最新的 checkpoint
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        generator.load_state_dict(ckpt['generator_state_dict'])
        print(f"加载最新 checkpoint (Epoch {ckpt['epoch']})")
    else:
        print("未找到训练好的模型，使用未训练的模型（生成结果将无意义）")
else:
    print("未找到训练好的模型，使用未训练的模型（生成结果将无意义）")

generator.eval()

# 生成 10 张卡通头像
num_images = 10
noise = torch.randn(num_images, latent_dim, device=device)

with torch.no_grad():
    generated_images = generator(noise)

# 反归一化: [-1, 1] → [0, 1]
generated_images = (generated_images + 1) / 2.0

# 保存生成的图片到输出目录
output_dir = os.path.join(DATA_DIR, 'outputs', 'visualizations')
os.makedirs(output_dir, exist_ok=True)

for i in range(num_images):
    img_path = os.path.join(output_dir, f'generated_face_{i+1}.png')
    save_image(generated_images[i], img_path)
    print(f"保存: generated_face_{i+1}.png")

# 保存 2×5 网格图
grid_path = os.path.join(output_dir, 'generated_faces_grid.png')
save_image(generated_images, grid_path, nrow=5, padding=4, pad_value=1.0)

# 用 matplotlib 显示生成结果
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('DCGAN 生成的卡通头像', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < num_images:
        img = generated_images[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img.clip(0, 1))
        ax.set_title(f'Face {i+1}')
    ax.axis('off')

plt.tight_layout()
display_path = os.path.join(output_dir, 'generated_faces_display.png')
plt.savefig(display_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n生成完成！共生成 {num_images} 张卡通头像")
print(f"网格图: {grid_path}")
print(f"显示图: {display_path}")
print(f"单张图片目录: {output_dir}")

# 显示生成结果
display_img = Image.open(display_path)
display(display_img)
```

## 实验结论

DCGAN 在 Cartoon Face 数据集上的训练结果，需要从 GAN 训练的特殊性来理解。GAN 的训练目标不是像分类任务那样有明确的"准确率越高越好"指标，而是追求生成器与判别器之间的动态平衡。这种特殊性使得 GAN 的训练评估不能简单看损失曲线，而需要从多个维度综合判断。

1. **训练损失的解读**：GAN 的损失曲线与分类网络的损失曲线完全不同。分类网络的训练损失和验证损失单调下降代表训练成功，而 GAN 的 G_loss 和 D_loss 在训练过程中会持续震荡，这是正常现象。因为对抗博弈的动态特性——生成器能力提升 → 判别器损失上升 → 判别器能力提升 → 生成器损失上升 → 循环往复。损失震荡不代表训练失败，只有当一方损失持续为零或持续上升不收敛时才代表训练失败。判断 GAN 训练是否成功的最可靠方法是直接观察生成图像的质量，而非分析损失曲线。

2. **生成质量评估**：100 个 epoch 的训练通常能生成有一定结构性的图像，但质量远不如真实数据。这与 AlexNet 实验中 45% 的准确率类似，是资源约束下的合理结果。GAN 文献中报告的"照片级质量"通常来自数百甚至数千个 epoch 的训练，配合更复杂的架构（如 Progressive GAN、StyleGAN）和更精细的训练技巧（如梯度惩罚、特征匹配损失）。本实验的目标是通过完整实现 DCGAN 训练流程来理解对抗训练的机制，而非追求竞赛级别的生成质量。

3. **训练稳定性挑战**：GAN 训练的不稳定是初学者最常见的困扰，以下是几个典型问题及其应对策略：

    | 问题 | 表现 | 常见应对策略 |
    |------|------|------------|
    | 模式崩溃 | 所有生成图像几乎相同 | 增加 D 训练次数（5:1），使用特征匹配损失 |
    | 判别器过强 | G_loss 不下降，D_loss 接近零 | 单侧标签平滑（real=0.9），降低 D 学习率 |
    | 训练震荡 | 损失剧烈波动，不收敛 | 降低学习率，增加 BatchNorm |
    | 生成模糊 | 图像缺少细节特征 | 增加训练 epoch，使用更大模型 |

4. **工程改进方向**：如果希望进一步提升生成质量，可以考虑以下方向：

    - **增加训练轮数**：将 epoch 从 100 增加到 200-500，这是最直接的改进方式。GAN 的收敛速度比分类网络慢得多，100 epoch 往往不够。
    - **使用更复杂的架构**：DCGAN 是 2015 年的架构，现代 GAN 有更好的设计。例如 WGAN-GP 使用梯度惩罚替代权重裁剪，PGGAN 使用渐进式训练策略，StyleGAN 使用风格注入机制。这些架构的改进目标是解决训练稳定性问题。
    - **调整训练比例**：尝试判别器训练 5 步、生成器训练 1 步的策略，让判别器保持适度优势，提供更有效的梯度信号。
    - **使用更高分辨率**：将图像从 64×64 提升到 128×128 或更高，配合更深的网络结构，可以生成更多细节的图像，但训练时间会成倍增加。

## 数据结果

本实验完整展示了 DCGAN 的训练流程，训练完成后，以下文件将保存到数据目录：

- **模型文件**：
    - `<DATA_DIR>/models/gan/dcgan/checkpoints/epoch_20.pth` ~ `epoch_100.pth` - 每 20 epoch 的 checkpoint
    - `<DATA_DIR>/models/gan/dcgan/final/dcgan_generator_cartoon_face.pth` - 最终生成器权重
- **训练样本**：
    - `<DATA_DIR>/outputs/training_samples/epoch_10.png` ~ `epoch_100.png` - 每 10 epoch 的 8×8 样本网格
- **推理结果**：
    - `<DATA_DIR>/outputs/visualizations/generated_face_1.png` ~ `generated_face_10.png` - 10 张生成的卡通头像
    - `<DATA_DIR>/outputs/visualizations/generated_faces_grid.png` - 2×5 网格图
    - `<DATA_DIR>/outputs/visualizations/generated_faces_display.png` - matplotlib 显示图
- **训练日志**：
    - `<DATA_DIR>/models/gan/dcgan/training_log.txt` - 每 epoch 的损失和时间记录