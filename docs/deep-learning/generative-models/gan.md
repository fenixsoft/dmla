---
title: "生成对抗网络"
date: 2026-05-07
tags: [deep-learning, gan, generative-models]
series:
  name: "深度学习经典模型"
  chapter: 7
  order: 2
---

# 生成对抗网络

上一篇文章介绍了 VAE——一种通过变分推断实现生成的模型。VAE 的生成过程是：学习数据的潜在分布，从分布采样，通过解码器生成新样本。

VAE 的优势是生成稳定、潜在空间可控，但生成的样本通常模糊。这是因为 VAE 的训练目标是重建损失（MSE 或交叉熵），会倾向于生成"平均化"的样本——模糊但覆盖所有特征。

**生成对抗网络**（Generative Adversarial Network, GAN）采用完全不同的思路：不学习数据的显式分布，而是通过"对抗训练"让生成器产生与真实数据难以区分的样本。

2014 年，Goodfellow 等人提出的 GAN 是生成模型的重要突破。GAN 的核心思想：

- **生成器**（Generator）：从随机噪声生成"假"样本
- **判别器**（Discriminator）：判断样本是真实的还是生成的
- **对抗训练**：生成器努力欺骗判别器，判别器努力识破生成器

这种"博弈对抗"的设计让 GAN 生成的样本通常比 VAE 更清晰、更真实。但代价是训练不稳定——生成器和判别器的博弈难以收敛，可能出现"崩溃"（Collapse）现象。

GAN 的提出开启了生成模型的新时代，催生了 DCGAN、WGAN、StyleGAN 等一系列改进版本，成为现代生成模型的基础架构。

本文将介绍 GAN 的原理、训练方法、稳定性问题和常见变体。

## GAN 架构设计

### 生成器-判别器架构

GAN 由两个神经网络组成：

```
GAN 架构:

真实数据 x_real
              ↓
      ┌───────────────────────┐
      │       判别器          │ ← 二分类网络：判断真假
      │   D(x) → 0/1         │    输出: 1=真，0=假
      └───────────────────────┘
              ↑
              │ 判断
              │
      ┌───────────────────────┐
      │       生成器          │ ← 生成网络：噪声→样本
      │   G(z) → x_fake      │    输入: 随机噪声
      └───────────────────────┘
              ↑
随机噪声 z ~ N(0, 1)
```

**生成器 $G$**：

- 输入：随机噪声 $z$（通常从标准正态分布采样）
- 输出：生成的样本 $x_{fake} = G(z)$
- 目标：生成尽可能真实的样本，欺骗判别器

**判别器 $D$**：

- 输入：样本 $x$（可能是真实数据 $x_{real}$ 或生成数据 $x_{fake}$）
- 输出：概率 $D(x) \in [0, 1]$（判断样本是真还是假）
- 目标：准确判断样本真假，不被生成器欺骗

**对抗关系**：

生成器和判别器是零和博弈：
- 生成器希望 $D(G(z))$ 接近 1（判别器认为生成的样本是真的）
- 判别器希望 $D(x_{real})$ 接近 1（正确识别真实样本），$D(G(z))$ 接近 0（正确识别生成样本）

### GAN 的数学表示

GAN 的训练目标可以形式化描述：

**判别器的目标**：

最大化判断正确率：

$$\max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]$$

- $\mathbb{E}_{x \sim p_{data}}[\log D(x)]$：对真实样本，希望 $D(x)$ 接近 1
- $\mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]$：对生成样本，希望 $D(G(z))$ 接近 0

**生成器的目标**：

最小化判别器的判断正确率：

$$\min_G \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]$$

生成器希望 $D(G(z))$ 接近 1，让判别器认为生成样本是真的。

**综合目标函数**：

GAN 的训练是 minimax 博弈：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]$$

判别器最大化 $V$，生成器最小化 $V$。

### Nash 均衡理解

GAN 的理想训练结果：生成器和判别器达到 **Nash 均衡**（纳什均衡）。

**Nash 均衡的定义**：

在博弈论中，Nash 均衡是一种状态：
- 每个玩家的策略是最优的（给定其他玩家的策略）
- 任何玩家单独改变策略都不会获益

在 GAN 中，Nash 均衡对应：

```
Nash 均衡状态:

生成器: G 生成的样本分布 p_G 与真实数据分布 p_data 完全相同
        p_G = p_data

判别器: D 无法区分真实和生成样本
        D(x) = 0.5 (对所有 x，输出概率都是 0.5)
```

**证明 Nash 均衡**：

当 $p_G = p_data$ 时：

$$V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{x \sim p_G}[\log (1 - D(x)))]$$

$$= \mathbb{E}_{x \sim p_{data}}[\log D(x) + \log (1 - D(x))]$$

最优判别器满足：

$$\frac{d}{dD(x)}[\log D(x) + \log (1 - D(x))] = 0$$

$$\frac{1}{D(x)} - \frac{1}{1 - D(x)} = 0$$

解得：$D(x) = 0.5$

此时 $V(D, G) = \log 0.5 + \log 0.5 = -\log 4$（全局最优）

**Nash 均衡的含义**：

- 生成器已经学到真实数据的分布，生成的样本与真实样本完全相同
- 判别器无法分辨真假，对所有样本输出 0.5（随机猜测）
- 任何一方单独改变策略都不会获益（生成器已经最优，判别器无法改进）

这是 GAN 训练的理想终点。

### 与 VAE 的对比

| 特性 | VAE | GAN |
|:-----|:-----|:-----|
| 生成原理 | 学习数据分布，从分布采样 | 对抗训练，无显式分布 |
| 训练目标 | 重建损失 + KL 散度 | 对抗博弈损失 |
| 生成质量 | 通常模糊，但稳定 | 通常清晰，但不稳定 |
| 潜在空间 | 有结构，可解释 | 无显式结构 |
| 训练稳定性 | 稳定，容易收敛 | 不稳定，可能崩溃 |
| 计算开销 | 较低（单网络训练） | 较高（两网络交替训练） |

GAN 的优势在于生成质量，劣势在于训练不稳定。这催生了大量改进版本，解决稳定性问题。

## 生成器与判别器对抗训练

### 训练流程

GAN 的训练采用交替训练策略：

```
GAN 训练流程:

Step 1: 训练判别器（固定生成器）
  - 采样真实数据 x_real
  - 采样噪声 z，生成假数据 x_fake = G(z)
  - 计算判别器损失:
    L_D = -[log D(x_real) + log(1 - D(x_fake))]
  - 更新判别器参数

Step 2: 训练生成器（固定判别器）
  - 采样噪声 z
  - 计算生成器损失:
    L_G = -log D(G(z))  ← 希望 D 认为生成的样本是真的
  - 更新生成器参数

重复 Step 1 和 Step 2 直到收敛
```

**交替训练的原因**：

不能同时训练生成器和判别器：
- 判别器在训练初期可能远强于生成器（$D(x)$ 快速接近最优）
- 如果同时训练，生成器可能无法追赶判别器的进步
- 交替训练让生成器和判别器都有机会提升

**训练比例**：

通常每个 epoch 训练判别器 $k$ 次，训练生成器 1 次（$k = 1 \sim 5$）。判别器需要更强的能力，才能提供有效的梯度信号给生成器。

### 判别器损失函数

判别器是二分类网络，使用二元交叉熵损失：

$$L_D = -\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]$$

实际实现：

```python
# 判别器损失
d_loss_real = -torch.log(D(x_real))      # 真样本希望 D 输出 1
d_loss_fake = -torch.log(1 - D(x_fake))  # 假样本希望 D 输出 0
d_loss = d_loss_real + d_loss_fake
```

### 生成器损失函数

生成器的损失函数有两种形式：

**形式 1：最小化 $\log (1 - D(G(z)))]$**

$$L_G = \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]$$

这直接对应 minimax 博弈的目标。

**形式 2：最大化 $\log D(G(z)))]$（更常用）**

$$L_G = -\mathbb{E}_{z \sim p_z}[\log D(G(z)))]$$

两种形式在 Nash 均衡点相同，但训练初期行为不同：
- 形式 1：训练初期 $D(G(z))$ 很小，$\log(1-D(G(z)))$ 接近 0，梯度很小
- 形式 2：训练初期梯度较大，更有利于生成器学习

实际实现常用形式 2：

```python
# 生成器损失（形式 2）
g_loss = -torch.log(D(G(z)))  # 希望 D 认为生成的样本是真的
```

### PyTorch 实现

```python runnable
import torch
import torch.nn as nn
import torch.optim as optim

# 定义生成器
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()  # 输出像素值（0-1）
        )
    
    def forward(self, z):
        return self.model(z)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出概率（0-1）
        )
    
    def forward(self, x):
        return self.model(x)

# 创建 GAN
noise_dim = 100
data_dim = 784  # MNIST 图像维度

G = Generator(noise_dim, data_dim)
D = Discriminator(data_dim)

# 优化器
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)

# 模拟训练流程
batch_size = 32
print("GAN 模型构建:")
print(f"  生成器: 输入噪声 {noise_dim} 维 → 输出图像 {data_dim} 维")
print(f"  判别器: 输入图像 {data_dim} 维 → 输出概率 1 维")

# 模拟训练一步
z = torch.randn(batch_size, noise_dim)
x_fake = G(z)
x_real = torch.rand(batch_size, data_dim)  # 模拟真实数据

# 判别器判断
d_real = D(x_real)
d_fake = D(x_fake)

print(f"\n判别器输出:")
print(f"  真样本: {d_real.mean().item():.3f} (希望接近 1)")
print(f"  假样本: {d_fake.mean().item():.3f} (希望接近 0)")

# 计算损失
d_loss = -torch.log(d_real).mean() - torch.log(1 - d_fake).mean()
g_loss = -torch.log(d_fake).mean()  # 生成器希望 D 认为假样本是真的

print(f"\n损失函数:")
print(f"  判别器损失: {d_loss.item():.3f}")
print(f"  生成器损失: {g_loss.item():.3f}")

print("\nGAN 架构构建成功，可以开始对抗训练")
```

### 完整训练示例

```python runnable
import torch
import torch.nn as nn
import torch.optim as optim

# 定义简化的 GAN
class SimpleGAN(nn.Module):
    def __init__(self, noise_dim=100, data_dim=784):
        super().__init__()
        
        # 生成器
        self.G = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim),
            nn.Sigmoid()
        )
        
        # 判别器
        self.D = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.noise_dim = noise_dim
    
    def generate(self, num_samples):
        """生成样本"""
        z = torch.randn(num_samples, self.noise_dim)
        return self.G(z)
    
    def discriminate(self, x):
        """判断真假"""
        return self.D(x)

# 创建模型
gan = SimpleGAN()
g_optimizer = optim.Adam(gan.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(gan.D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练参数
num_epochs = 50
batch_size = 64
k_steps = 1  # 每训练生成器 1 次，训练判别器 k 次

print("开始 GAN 对抗训练...")
g_losses = []
d_losses = []

for epoch in range(num_epochs):
    epoch_g_loss = 0
    epoch_d_loss = 0
    
    # 训练判别器 k 次
    for _ in range(k_steps):
        # 采样真实数据（模拟 MNIST）
        x_real = torch.rand(batch_size, 784)
        
        # 生成假数据
        z = torch.randn(batch_size, gan.noise_dim)
        x_fake = gan.generate(batch_size)
        
        # 判别器判断
        d_real = gan.discriminate(x_real)
        d_fake = gan.discriminate(x_fake.detach())  # detach 防止梯度传给 G
        
        # 判别器损失
        d_loss = -torch.log(d_real).mean() - torch.log(1 - d_fake).mean()
        
        # 更新判别器
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        epoch_d_loss += d_loss.item()
    
    # 训练生成器 1 次
    z = torch.randn(batch_size, gan.noise_dim)
    x_fake = gan.generate(batch_size)
    d_fake = gan.discriminate(x_fake)
    
    # 生成器损失（希望判别器认为假样本是真的）
    g_loss = -torch.log(d_fake).mean()
    
    # 更新生成器
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    
    epoch_g_loss += g_loss.item()
    
    g_losses.append(epoch_g_loss)
    d_losses.append(epoch_d_loss / k_steps)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: D_loss={epoch_d_loss/k_steps:.3f}, G_loss={epoch_g_loss:.3f}")

# 测试生成
print("\n测试生成能力:")
gan.eval()
with torch.no_grad():
    generated = gan.generate(10)
    print(f"生成 10 个样本，形状: {generated.shape}")
    print(f"像素范围: {generated.min().item():.3f} ~ {generated.max().item():.3f}")
    
    # 测试判别器判断
    x_real = torch.rand(10, 784)
    x_fake = gan.generate(10)
    d_real = gan.discriminate(x_real)
    d_fake = gan.discriminate(x_fake)
    print(f"\n判别器判断:")
    print(f"  真样本平均概率: {d_real.mean().item():.3f}")
    print(f"  假样本平均概率: {d_fake.mean().item():.3f}")
    print("  (理想均衡: 两者都接近 0.5)")

print("\nGAN 对抗训练完成")
```

## 训练稳定性问题

### GAN 的核心难题

GAN 的训练不稳定是主要挑战，常见问题：

**问题 1：模式崩溃（Mode Collapse）**

生成器只学会生成少数几种样本，多样性丧失：

```
模式崩溃示例:

训练前: G 可以生成多种数字
  z_1 → "1"
  z_2 → "2"
  z_3 → "3"

模式崩溃后: G 只生成一种数字
  z_1 → "1"
  z_2 → "1"
  z_3 → "1"

生成器发现: 只要生成 "1"，判别器就无法分辨
         → 只优化生成 "1"，放弃其他模式
```

模式崩溃的原因：
- 生成器的优化目标只是"欺骗判别器"，而非"覆盖所有数据分布"
- 如果生成一种样本就能欺骗判别器，生成器没有动力生成其他样本

**问题 2：判别器过强**

判别器训练过快，远强于生成器：

```
判别器过强:

Epoch 10: D 已经能完美分辨真假
         D(x_real) = 1.0
         D(x_fake) = 0.0

生成器的梯度:
         ∂L_G/∂G = ∂[-log D(G(z))]/∂G
         当 D(G(z)) ≈ 0 时:
         ∂L_G/∂G ≈ ∂[-log 0]/∂G ≈ 很小的梯度

结果: 生成器无法学习（梯度消失）
```

判别器过强导致梯度消失：$D(G(z)) \approx 0$，$\log D(G(z)) \approx -\infty$，梯度趋近于零。

**问题 3：训练震荡**

生成器和判别器交替占据优势，无法收敛：

```
训练震荡:

Epoch 100: D 很强 → G 梯度消失
Epoch 200: G 提升 → D 无法分辨
Epoch 300: D 重新变强 → G 又梯度消失
...

无法达到 Nash 均衡（D=0.5, p_G=p_data）
```

### 解决方案

**方案 1：调整训练比例**

增加判别器训练次数，让判别器保持适度优势：

```python
# 常用比例：判别器训练 5 次，生成器训练 1 次
for epoch in range(epochs):
    for _ in range(5):
        train_discriminator()  # 判别器多训练
    train_generator()          # 生成器少训练
```

但比例过高可能导致判别器过强（梯度消失），需要平衡。

**方案 2：标签平滑**

降低判别器对真实样本的目标值：

```python
# 标准：D(x_real) → 1.0
# 标签平滑：D(x_real) → 0.9

d_target_real = 0.9  # 不是 1.0
d_loss_real = -torch.log(D(x_real)) * d_target_real
```

标签平滑防止判别器过于自信，保留梯度信号。

**方案 3：梯度惩罚**

约束判别器的梯度范数：

$$L_{GP} = \lambda (\|\nabla_x D(x)\|_2 - 1)^2$$

这防止判别器梯度过大或过小，保持稳定的梯度信号。

**方案 4： Wasserstein 距离（WGAN）**

用 Wasserstein 距离替代 JS 散度，提供稳定的梯度。这是最重要的改进，下一节详细介绍。

### Wasserstein GAN

2017 年，Arjovsky 等人提出 WGAN，用 Wasserstein 距离（推土机距离 Earth Mover's Distance）替代原始 GAN 的 JS 散度。

**Wasserstein 距离的定义**：

$$W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

直观理解：将分布 $P_r$ 的"土"移动到分布 $P_g$ 的最小代价。

**WGAN 的关键改进**：

1. 判别器改为"评论家"（Critic），输出分数而非概率：
   - 标准 GAN：$D(x) \in [0, 1]$（概率）
   - WGAN：$C(x) \in \mathbb{R}$（分数，越大越真实）

2. 损失函数：
   - 评论家：最大化 $C(x_{real}) - C(x_{fake})$
   - 生成器：最大化 $C(x_{fake})$

3. 梯度惩罚：强制评论家满足 Lipschitz 连续性（梯度范数 $\leq 1$）

**WGAN 的优势**：

- Wasserstein 距离在分布不重叠时仍提供有效梯度
- 训练稳定，收敛可靠
- 损失函数值可以反映生成质量（可以作为训练指标）

**WGAN-GP 实现**：

```python runnable
import torch
import torch.nn as nn
import torch.optim as optim

# WGAN-GP 评论家（不输出概率，输出分数）
class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出分数，无 sigmoid
        )
    
    def forward(self, x):
        return self.model(x)

# 生成器（与标准 GAN 相同）
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.model(z)

# 梯度惩罚函数
def gradient_penalty(critic, real_data, fake_data):
    """计算梯度惩罚"""
    batch_size = real_data.size(0)
    
    # 随机插值点
    epsilon = torch.rand(batch_size, 1)
    epsilon = epsilon.expand_as(real_data)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    
    # 计算评论家在插值点的梯度
    interpolated.requires_grad_(True)
    critic_output = critic(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 梯度范数
    grad_norm = gradients.norm(2, dim=1)
    
    # 梯度惩罚（强制范数接近 1）
    penalty = ((grad_norm - 1) ** 2).mean()
    
    return penalty

# 创建 WGAN-GP
noise_dim = 100
data_dim = 784

G = Generator(noise_dim, data_dim)
C = Critic(data_dim)

g_optimizer = optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.9))
c_optimizer = optim.Adam(C.parameters(), lr=0.0001, betas=(0.0, 0.9))

print("WGAN-GP 架构:")
print(f"  生成器: 噪声 {noise_dim} 维 → 图像 {data_dim} 维")
print(f"  评论家: 图像 {data_dim} 维 → 分数（无 sigmoid）")
print(f"  梯度惩罚: 强制 Lipschitz 连续性")

# 模拟训练一步
batch_size = 32
z = torch.randn(batch_size, noise_dim)
x_fake = G(z)
x_real = torch.rand(batch_size, data_dim)

# 评论家分数
c_real = C(x_real)
c_fake = C(x_fake)

print(f"\n评论家输出:")
print(f"  真样本分数: {c_real.mean().item():.3f}")
print(f"  假样本分数: {c_fake.mean().item():.3f}")
print("  (分数越大越真实，无 sigmoid)")

# 梯度惩罚
gp = gradient_penalty(C, x_real, x_fake)
print(f"\n梯度惩罚: {gp.item():.3f} (希望接近 0)")

# 评论家损失
c_loss = -(c_real.mean() - c_fake.mean()) + 10 * gp

# 生成器损失
g_loss = -c_fake.mean()

print(f"\n损失函数:")
print(f"  评论家损失: {c_loss.item():.3f}")
print(f"  生成器损失: {g_loss.item():.3f}")

print("\nWGAN-GP 构建成功，训练更稳定")
```

## GAN 变体介绍

GAN 的提出催生了大量变体，解决不同问题：

### DCGAN

DCGAN（Deep Convolutional GAN）将 CNN 结构引入 GAN：

**改进要点**：
- 生成器使用转置卷积（上采样）
- 判别器使用卷积（下采样）
- 去除全连接层，使用卷积层
- 使用 Batch Normalization 稳定训练
- 生成器输出层用 Tanh，判别器输出层用 Sigmoid

**架构示意**：

```
DCGAN 生成器:

噪声 z (100 维)
    ↓ reshape → (1, 1, 100)
    ↓ ConvTranspose2d (stride=4)
    ↓ BatchNorm + ReLU
    ↓ ConvTranspose2d (stride=2)
    ↓ BatchNorm + ReLU
    ↓ ConvTranspose2d (stride=2)
    ↓ Tanh
输出图像 (64×64×3)
```

DCGAN 是图像生成的基础架构，后续 StyleGAN 等都是在 DCGAN 基础上改进。

### StyleGAN

StyleGAN（2019）由 NVIDIA 提出，实现了高质量的人脸生成：

**创新要点**：
- **风格注入**：将潜在编码 $w$ 注入到生成器的不同层，控制不同尺度的特征
- **AdaIN**（Adaptive Instance Normalization）：动态调整每层的风格
- **噪声注入**：在每层添加随机噪声，增加细节变化
- **渐进式生成**：从低分辨率到高分辨率，逐层生成

**效果**：
- 生成的图像达到照片级质量
- 可以精确控制生成图像的特征（年龄、性别、表情等）
- 潜在空间有良好的语义结构

StyleGAN 是当前最先进的图像生成架构之一。

### Progressive GAN

Progressive GAN（Progressive Growing of GANs）采用渐进式训练：

**训练策略**：
- 从低分辨率开始训练（4×4）
- 稳定后增加分辨率（8×8, 16×16, ..., 1024×1024）
- 逐层添加生成器和判别器的卷积层

**优势**：
- 低分辨率容易训练，提供稳定基础
- 高分辨率在稳定基础上逐步学习
- 避免直接训练高分辨率的困难

### CycleGAN

CycleGAN 用于无配对的图像翻译（如照片→油画）：

**架构**：
- 两个生成器：$G: X \rightarrow Y$, $F: Y \rightarrow X$
- 两个判别器：$D_Y$（判断 $Y$ 域图像），$D_X$（判断 $X$ 域图像）
- 循环一致性损失：$F(G(x)) \approx x$, $G(F(y)) \approx y$

**应用**：
- 照片→油画风格转换
- 夏天→冬天景色转换
- 马→斑马转换

CycleGAN 解决了无配对数据的翻译问题。

### Pix2Pix

Pix2Pix 用于有配对的图像翻译（如草图→照片）：

**架构**：
- 生成器：U-Net 结构，保留细节
- 判别器：条件判别器，判断配对是否合理
- 损失函数：对抗损失 + L1 重建损失

**应用**：
- 草图→照片
- 语义图→真实图像
- 白天→夜景

Pix2Pix 是条件生成模型的代表。

## 小结

本文介绍了生成对抗网络（GAN）的原理和实现：

**GAN 架构**：
- 生成器：从随机噪声生成假样本
- 判别器：判断样本真假
- 对抗训练：生成器欺骗判别器，判别器识破生成器

**数学表示**：
- minimax 博弈：$\min_G \max_D V(D, G)$
- Nash 均衡：$p_G = p_{data}$，$D(x) = 0.5$

**训练流程**：
- 交替训练：判别器和生成器轮流更新
- 损失函数：判别器用交叉熵，生成器欺骗判别器

**稳定性问题**：
- 模式崩溃：生成器只生成少数模式
- 判别器过强：梯度消失
- 训练震荡：无法收敛

**解决方案**：
- WGAN：Wasserstein 距离替代 JS 散度
- 梯度惩罚：强制 Lipschitz 连续性
- 标签平滑：防止判别器过于自信

**GAN 变体**：
- DCGAN：卷积结构，图像生成基础
- StyleGAN：高质量人脸生成
- CycleGAN：无配对图像翻译
- Pix2Pix：有配对图像翻译

GAN 开启了生成模型的新时代，成为现代图像生成、风格转换、数据增强等任务的基础架构。

---

## 练习题

**1. Nash 均衡分析**

分析 GAN 的 Nash 均衡状态：
- 当 $p_G = p_{data}$ 时，证明最优判别器 $D(x) = 0.5$
- 为什么这是 Nash 均衡（双方都不愿单独改变策略）

**2. 梯度消失分析**

分析判别器过强导致的梯度消失：
- 当 $D(G(z)) \approx 0$ 时，计算生成器的梯度
- 为什么梯度会消失
- WGAN 如何解决这个问题

**3. 模式崩溃分析**

分析模式崩溃现象：
- 为什么生成器只生成少数模式
- 标签平滑如何缓解这个问题
- 设计实验验证模式崩溃

**4. 编程实现**

实现一个用于 MNIST 的 GAN：
- 生成器：MLP 结构，输入 100 维噪声
- 判别器：MLP 结构，输出真/假概率
- 训练并可视化生成样本的演化过程
- 对比标准 GAN 和 WGAN 的训练稳定性

---

## 参考资料

1. **GAN 原始论文**: "Generative Adversarial Nets" (Goodfellow et al., 2014)
2. **DCGAN 论文**: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (Radford et al., 2015)
3. **WGAN 论文**: "Wasserstein GAN" (Arjovsky et al., 2017)
4. **WGAN-GP 论文**: "Improved Training of Wasserstein GANs" (Gulrajani et al., 2017)
5. **StyleGAN 论文**: "A Style-Based Generator Architecture for Generative Adversarial Networks" (Karras et al., 2019)
6. **GAN 训练技巧**: "GAN Hacks: How to Train a GAN" (经验总结)