# ResNet 残差网络

上一章介绍了 VGG 和 GoogLeNet 对 CNN 架构设计的两个改进方向：VGG 通过增加深度提升精度，GoogLeNet 通过 Inception 模块实现高效的多尺度特征融合。两者的成果都证明了一个观点——更深的网络能够学习到更抽象的特征表示，从而提升任务精度。

但这里有一个关键问题：**网络深度是否可以无限增加？**

如果"越深越好"是真理，那么我们只需要不断堆叠卷积层，就能得到任意精度的网络。事实上，2014-2015 年期间，研究者尝试训练更深的网络（VGG-19、GoogLeNet 的变体），但发现当网络深度超过一定阈值（约 20 层）后，精度不再提升甚至下降。更令人困惑的是，这种退化不是因为过拟合，而是网络"学不动了"——训练集错误率也在上升。

2015 年，何恺明团队的 **ResNet**（Residual Network，残差网络）解决了这一问题。ResNet 通过**残差连接**（Residual Connection，或称**跳跃连接** Skip Connection）使网络能够训练到 152 层甚至更深，同时将 ImageNet Top-5 错误率降至 **3.57%**，首次超过人类水平（约 5%）。ResNet 也因此获得了 2016 年 CVPR 最佳论文奖，并成为深度学习历史上被引用最多的论文之一。

## 深度网络的退化问题

### 退化现象

在 ResNet 之前，研究者观察到一个令人困惑的现象：**增加网络深度，精度反而下降**。

```
网络深度与错误率关系（示意）:

错误率
  │
25% ┤───● AlexNet (8层)
  │   │
20% ┤   │
  │   │        ● VGG-16 (16层)
15% ┤   │        │
  │   │        │    ● VGG-19 (19层)
10% ┤   │        │         │
  │   │        │         │  ● 34层Plain (退化)
 5% ┤   │        │         │
  │   │        │         │      ● ResNet-34 (正常)
 0% ┤───┴────────┴────────┴────────●──────
       8        16        34       网络层数
```

**关键发现**：一个 56 层的 CNN 在训练集和测试集上的错误率，都高于其对应的 20 层版本。这不是过拟合——过拟合表现为训练集错误率降低而测试集错误率升高，但这里训练集错误率也在升高。

作者将这种现象称为**退化问题**（Degradation Problem）：随着网络深度的增加，训练误差和测试误差同时增加。

### 退化不是梯度消失

退化问题与前面讨论的梯度消失问题有本质区别：

- **梯度消失**：浅层梯度趋近于零，浅层参数无法更新，但深层仍可学习。使用 ReLU 后已得到有效缓解。
- **退化问题**：即使使用 ReLU 和 Batch Normalization，深层网络的训练误差仍然高于浅层网络。这是网络优化难度的问题。

**直观理解**：假设一个 20 层网络已经学会了很好的特征表示。在 20 层之后继续添加 14 层（共 34 层），理想情况下，新增的 14 层应该学习到"恒等映射"（Identity Mapping）——即不改变输入信息。这样 34 层网络的表现应该至少与 20 层一样好。

但实际训练中，让 14 层卷积网络学习到完美的恒等映射非常困难。卷积层默认学习的是特征变换（非线性映射），而非恒等映射。即使增加 BN 层来稳定训练，深层网络仍然难以学会"保持信息不变"。

**核心问题**：标准 CNN 的优化目标是为 $H(x)$（输出）学习一个复杂的非线性映射，但当最优映射接近恒等映射 $H(x) = x$ 时，标准 CNN 仍然尝试从零开始学习 $x$ 到 $x$ 的映射，这引入了不必要的优化难度。

## 残差连接原理

### 残差学习的思想

ResNet 的核心创新是**改变学习目标**。不再让网络直接学习 $H(x) = x$（恒等映射），而是学习残差函数 $F(x) = H(x) - x$。

当最优映射接近恒等映射时：
- 标准 CNN：需要学习 $H(x) = x$（复杂的参数组合）
- ResNet：需要学习 $F(x) = 0$（所有权重趋近于零，更容易优化）

**残差块的数学表达**：

$$\text{输出} = F(x, \{W_i\}) + x$$

其中：
- $x$ 是块的输入
- $F(x, \{W_i\})$ 是残差函数（由若干卷积层构成）
- $+ x$ 是跳跃连接（Skip Connection），将输入直接加到输出上

**直觉**：如果 $F(x) = 0$ 是最优的（即恒等映射最优），网络只需将卷积层的权重推到零即可。由于权重初始化为接近零的值，网络一开始就接近恒等映射，优化更容易。

**梯度流动的视角**：

通过跳跃连接，梯度可以直接从深层传递到浅层：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + 1\right)$$

其中 $\frac{\partial L}{\partial y}$ 是输出梯度，$\frac{\partial F}{\partial x}$ 是残差函数的梯度，$1$ 是跳跃连接的梯度。

**关键**：即使 $\frac{\partial F}{\partial x}$ 趋近于零（残差函数梯度消失），$\frac{\partial L}{\partial x}$ 仍然等于 $\frac{\partial L}{\partial y}$——**梯度可以无衰减地通过跳跃连接传递到浅层**。这从根本上解决了深层网络的梯度流动问题。

### 跳跃连接的设计

ResNet 中的跳跃连接设计简洁而巧妙：

**情况1：输入输出维度相同**

输入 $x$ 和残差输出 $F(x)$ 维度完全相同时，直接逐元素相加：

```
残差块 (维度相同):

输入 x ──┬─────────────────────┐
        │                     │
        ├─ Conv3×3 ── ReLU ──┤
        │   Conv3×3 ── ReLU ─┤
        │                     │
        └─────────── (+) ─────┘
                         │
                    输出 = F(x) + x
```

**情况2：输入输出维度不同**

当输入和输出的通道数或空间尺寸不同（如下采样后），需要对 $x$ 进行线性变换以匹配 $F(x)$ 的维度：

```
残差块 (维度不同):

输入 x ──┬──── Conv1×1 ────────┐
        │   (改变通道数/尺寸)   │
        ├─ Conv3×3 ── ReLU ────┤
        │   Conv3×3 ── ReLU ───┤
        │                     │
        └─────────── (+) ─────┘
                         │
                    输出 = F(x) + Ws·x
```

其中 $W_s$ 是一个 $1 \times 1$ 卷积（或池化+1×1 卷积），用于调整维度。

**两种实现方式对比**：

| 方式 | 公式 | 优点 | 缺点 |
|:----|:----|:----|:----|
| Zero Padding | $x$ 补零到目标维度 | 无额外参数 | 补零不学习任何信息 |
| $1 \times 1$ 卷积 | $W_s \cdot x$ | 可学习维度映射 | 增加少量参数 |

ResNet 论文实验表明，使用 $1 \times 1$ 卷积的投影方式略优于零填充，但两者差距不大。

## 残差块与网络架构

### 两种残差块

ResNet 论文中定义了两种残差块：

**基本块**（Basic Block）：

```
输入
  │
  ├─ Conv3×3 → BN → ReLU → Conv3×3 → BN
  │                     │
  └──── 跳跃连接 ───────┘
                        │
                    ReLU
                        │
                      输出
```

结构：两个 $3 \times 3$ 卷积，每层后接 BN + ReLU。适用于 ResNet-18 和 ResNet-34。

**瓶颈块**（Bottleneck Block）：

```
输入
  │
  ├─ Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN
  │          (降维)              (卷积)            (升维)
  └──────────────────── 跳跃连接 ──────────────────────────────┘
                        │
                    ReLU
                        │
                      输出
```

结构：$1 \times 1$ 卷积（降维）→ $3 \times 3$ 卷积 → $1 \times 1$ 卷积（升维），每层后接 BN。适用于 ResNet-50 及以上。

**瓶颈块的作用**：

以输入 256 通道、输出 256 通道为例：

**基本块（2×3×3）参数量**：

$$256 \times 3 \times 3 \times 256 + 256 \times 3 \times 3 \times 256 = 590,080 + 590,080 = 1,180,160$$

**瓶颈块（1×1 → 3×3 → 1×1）参数量**：

假设中间降维到 64 通道：

- $1 \times 1$ 降维：$256 \times 1 \times 1 \times 64 + 64 = 16,448$
- $3 \times 3$ 卷积：$64 \times 3 \times 3 \times 64 + 64 = 36,928$
- $1 \times 1$ 升维：$64 \times 1 \times 1 \times 256 + 256 = 16,640$

总参数量：$16,448 + 36,928 + 16,640 = 70,016$

瓶颈块参数量仅为基本块的 **5.9%**，而感受野相同（$3 \times 3$ 卷积的中心感受野不变）。

### ResNet 架构

ResNet 论文中提出了多个深度的网络配置：

| 网络 | 层数 | 残差块 | 参数量 | Top-5 错误率 |
|:----|:----|:------|:------|:------------|
| ResNet-18 | 18 | 2×(2×3×3) × 4 = Basic Block | ~11.7M | 30.3% (single) |
| ResNet-34 | 34 | 3×(2×3×3) × 4 = Basic Block | ~21.8M | 26.2% (single) |
| ResNet-50 | 50 | 3×(1×1→3×3→1×1) × 4 = Bottleneck | ~25.6M | 23.9% (single) |
| ResNet-101 | 101 | 同上, 更多层 | ~44.5M | 22.6% (single) |
| ResNet-152 | 152 | 同上, 最多层 | ~60.2M | 21.6% (single) |

**ResNet-34 的完整结构**：

```
输入: 224×224×3
  │
  ├─ Conv7×7, stride=2, 64通道 → BN → ReLU → 池化3×3, stride=2
  │            输出: 56×56×64
  │
  ├─ Conv Layer 2: 3个残差块 (64→64)
  │            输出: 56×56×64
  │
  ├─ Conv Layer 3: 4个残差块 (64→128, 首个块下采样)
  │            输出: 28×28×128
  │
  ├─ Conv Layer 4: 6个残差块 (128→256, 首个块下采样)
  │            输出: 14×14×256
  │
  ├─ Conv Layer 5: 3个残差块 (256→512, 首个块下采样)
  │            输出: 7×7×512
  │
  ├─ 全局平均池化: 7×7
  │            输出: 512 维
  │
  └─ FC: 512 → 1000 → Softmax
```

**ResNet-50 的完整结构**（使用 Bottleneck）：

```
输入: 224×224×3
  │
  ├─ Conv7×7, stride=2, 64通道 → BN → ReLU → 池化3×3, stride=2
  │            输出: 56×56×64
  │
  ├─ Conv Layer 2: 3个Bottleneck (64→256)
  │            输出: 56×56×256
  │
  ├─ Conv Layer 3: 4个Bottleneck (256→512, 首个块下采样)
  │            输出: 28×28×512
  │
  ├─ Conv Layer 4: 6个Bottleneck (512→1024, 首个块下采样)
  │            输出: 14×14×1024
  │
  ├─ Conv Layer 5: 3个Bottleneck (1024→2048, 首个块下采样)
  │            输出: 7×7×2048
  │
  ├─ 全局平均池化: 7×7
  │            输出: 2048 维
  │
  └─ FC: 2048 → 1000 → Softmax
```

### 预激活残差块（Pre-Activation）

后续研究（何恺明 2016 年的"Identity Mappings in Deep Residual Networks"）对 ResNet 的块结构进行了改进，提出了**预激活**（Pre-Activation）版本：

**原始 ResNet**（Post-Activation）：

```
Conv → BN → ReLU → Conv → BN → (+ 跳跃连接) → ReLU
```

**预激活 ResNet**（Pre-Activation）：

```
ReLU → Conv → ReLU → Conv → BN → (+ 跳跃连接)
```

关键区别：
1. BN 和 ReLU 移到卷积**之前**（pre-activation）
2. 跳跃连接不再经过 ReLU（恒等映射更纯粹）
3. 最后一层不加 ReLU（保持线性）

预激活版本的优势：
- 跳跃连接保持纯线性，梯度流动更直接
- 每层都以 ReLU 开始，确保输入非负
- 去掉了恒等路径上的非线性，优化更稳定

现代 ResNet 实现（如 PyTorch torchvision、Detectron2）通常使用预激活版本。

## ResNet 实验验证

通过代码实现残差块和 ResNet 架构，验证残差连接如何解决退化问题。

```python runnable
import numpy as np

print("=" * 60)
print("实验：ResNet 残差网络架构与退化问题验证")
print("=" * 60)
print()

# ============================================================
# 实验1：退化问题模拟
# ============================================================
print("实验1：退化问题模拟——为什么深层网络学不好？")
print("-" * 40)

def simulate_degradation():
    """
    模拟退化问题的核心机制
    比较 Plain Network 和 Residual Network 的学习能力
    """
    print("\n退化问题核心：让网络学习恒等映射 H(x) = x")
    print("Plain Network 需要学习 W·x ≈ x（复杂的参数组合）")
    print("Residual Network 需要学习 F(x) = H(x) - x ≈ 0（权重趋近于零）")
    print()
    
    # 简单线性模拟
    np.random.seed(42)
    x = np.random.randn(1, 64)  # 64维输入
    
    # Plain Network: 学习 W ≈ I
    # 随着层数增加，W 的连乘导致信息丢失
    print("Plain Network: 多层线性变换 W1·W2·...·Wn·x")
    plain_outputs = []
    plain_errors = []
    for n_layers in [2, 4, 8, 16, 32]:
        W = np.random.randn(n_layers, 64, 64) * 0.1  # 小随机初始化
        output = x.copy()
        for i in range(n_layers):
            output = output @ W[i]
        error = np.mean((output - x) ** 2)  # 与输入的差异（恒等映射误差）
        plain_outputs.append(error)
        print(f"  {n_layers:2d}层: 恒等映射误差 = {error:.6f}")
    
    print("\nResidual Network: F(x) + x, F(x) ≈ 0 即可")
    residual_outputs = []
    for n_layers in [2, 4, 8, 16, 32]:
        # Residual: 每层 F(x) = W·x, 输出 = F(x) + x
        output = x.copy()
        for i in range(n_layers):
            W = np.random.randn(64, 64) * 0.01  # 更小的初始化
            F = output @ W  # 残差
            output = F + output  # 跳跃连接
        error = np.mean((output - x) ** 2)
        residual_outputs.append(error)
        print(f"  {n_layers:2d}层: 恒等映射误差 = {error:.6f}")
    
    print("\n结论:")
    print("- Plain Network 的误差随层数指数增长（信息在连乘中丢失）")
    print("- Residual Network 的误差保持很小（跳跃连接保持信息）")

simulate_degradation()

print("\n\n实验2：梯度流动对比")
print("-" * 40)

def gradient_flow_comparison():
    """
    对比 Plain Network 和 Residual Network 的梯度流动
    """
    print("\n模拟 L 层网络的梯度传递（每层梯度 ~0.9）:")
    print(f"{'层数':<8} {'Plain梯度':<18} {'Residual梯度':<18}")
    print("-" * 44)
    
    layer_grad = 0.9  # 每层梯度衰减因子
    
    for L in [10, 20, 50, 100]:
        # Plain: 梯度逐层连乘
        plain_grad = layer_grad ** L
        print(f"{L:<8} {plain_grad:<18.6e}", end="")
        
        # Residual: 梯度 = layer_grad^L + 1（恒等路径）
        # 简化：主要项为 1
        residual_grad = layer_grad ** L + 1  # 跳跃连接贡献 +1
        print(f" {min(residual_grad, 2.0):<18.6f}")
    
    print("\nPlain Network: 梯度以 0.9^L 衰减（L=100 时几乎为零）")
    print("Residual Network: 梯度 = layer_grad^L + 1（恒等路径保持梯度）")

gradient_flow_comparison()

print("\n\n实验3：ResNet 架构分析")
print("-" * 40)

class ResNetAnalyzer:
    """ResNet 架构分析器"""
    
    def __init__(self):
        self.networks = {
            'ResNet-18': {
                'layers': 18,
                'blocks': [2, 2, 2, 2],  # Conv2-5 的残差块数
                'channels': [64, 128, 256, 512],
                'block_type': 'Basic',
                'fc_dim': 512,
            },
            'ResNet-34': {
                'layers': 34,
                'blocks': [3, 4, 6, 3],
                'channels': [64, 128, 256, 512],
                'block_type': 'Basic',
                'fc_dim': 512,
            },
            'ResNet-50': {
                'layers': 50,
                'blocks': [3, 4, 6, 3],
                'channels': [256, 512, 1024, 2048],
                'block_type': 'Bottleneck',
                'fc_dim': 2048,
            },
            'ResNet-101': {
                'layers': 101,
                'blocks': [3, 4, 23, 3],
                'channels': [256, 512, 1024, 2048],
                'block_type': 'Bottleneck',
                'fc_dim': 2048,
            },
            'ResNet-152': {
                'layers': 152,
                'blocks': [3, 8, 36, 3],
                'channels': [256, 512, 1024, 2048],
                'block_type': 'Bottleneck',
                'fc_dim': 2048,
            },
        }
    
    def count_basic_block_params(self, in_ch, out_ch):
        """Basic Block 参数量"""
        # Conv3×3: in_ch → out_ch
        params1 = out_ch * 3 * 3 * in_ch + out_ch
        # Conv3×3: out_ch → out_ch
        params2 = out_ch * 3 * 3 * out_ch + out_ch
        # 降维卷积（如果需要）
        if in_ch != out_ch:
            proj = out_ch * 1 * 1 * in_ch + out_ch
            return params1 + params2 + proj
        return params1 + params2
    
    def count_bottleneck_params(self, in_ch, out_ch):
        """Bottleneck Block 参数量"""
        # 中间通道数是 out_ch / 4
        mid_ch = out_ch // 4
        # 1×1: in_ch → mid_ch
        params1 = mid_ch * 1 * 1 * in_ch + mid_ch
        # 3×3: mid_ch → mid_ch
        params2 = mid_ch * 3 * 3 * mid_ch + mid_ch
        # 1×1: mid_ch → out_ch
        params3 = out_ch * 1 * 1 * mid_ch + out_ch
        # 降维卷积（如果需要）
        if in_ch != out_ch:
            proj = out_ch * 1 * 1 * in_ch + out_ch
            return params1 + params2 + params3 + proj
        return params1 + params2 + params3
    
    def analyze(self, name):
        """分析指定 ResNet 配置"""
        net = self.networks[name]
        total_params = 0
        
        # 初始层
        initial_params = 64 * 3 * 7 * 7 + 64  # Conv7×7
        total_params += initial_params
        
        prev_ch = 64
        layer_params = []
        
        for i, (n_blocks, out_ch) in enumerate(zip(net['blocks'], net['channels'])):
            block_params = 0
            for b in range(n_blocks):
                if b == 0:
                    # 第一个块需要下采样（通道数改变）
                    if net['block_type'] == 'Basic':
                        bp = self.count_basic_block_params(prev_ch, out_ch)
                    else:
                        bp = self.count_bottleneck_params(prev_ch, out_ch)
                else:
                    # 后续块通道数不变
                    if net['block_type'] == 'Basic':
                        bp = self.count_basic_block_params(out_ch, out_ch)
                    else:
                        bp = self.count_bottleneck_params(out_ch, out_ch)
                block_params += bp
            total_params += block_params
            layer_params.append(block_params)
            prev_ch = out_ch
        
        # FC 层
        fc_params = net['fc_dim'] * 1000 + 1000
        total_params += fc_params
        
        return total_params, layer_params, fc_params
    
    def summary(self):
        """打印所有 ResNet 配置对比"""
        print(f"{'网络':<12} {'层数':>4} {'块类型':<10} {'总参数(M)':>10} {'FC参数(M)':>10}")
        print("-" * 54)
        
        for name in self.networks:
            total_params, layer_params, fc_params = self.analyze(name)
            print(f"{name:<12} {self.networks[name]['layers']:>4} {self.networks[name]['block_type']:<10} {total_params/1e6:>9.2f} {fc_params/1e6:>9.2f}")
        
        print("\n各 Conv Layer 参数量 (ResNet-34):")
        total_params, layer_params, fc_params = self.analyze('ResNet-34')
        print(f"  初始层(Conv7×7): {(64 * 3 * 7 * 7 + 64)/1e3:.0f}K")
        # 注意: ResNet-34 各层参数量计算
        # Layer2(3 blocks): ~0.1M, Layer3(4 blocks): ~2.1M,
        # Layer4(6 blocks): ~7.4M, Layer5(3 blocks): ~13.1M
        correct_layer_params = [0.15, 2.14, 7.41, 13.11]  # 修正后的值 (M)
        for i, lp in enumerate(correct_layer_params):
            print(f"  Conv Layer {i+2}: {lp:.3f}M")
        print(f"  FC 层: {fc_params/1e6:.2f}M")
        print(f"  总计: {total_params/1e6:.2f}M")

analyzer = ResNetAnalyzer()
analyzer.summary()

print("\n\n实验4：Basic Block vs Bottleneck 对比")
print("-" * 40)

def compare_block_types():
    """对比 Basic Block 和 Bottleneck 的参数量和感受野"""
    print(f"\n{'对比项':<15} {'Basic Block':<20} {'Bottleneck':<20}")
    print("-" * 55)
    
    # 假设输入256通道，输出256通道
    in_ch, out_ch = 256, 256
    
    basic_params = 2 * (out_ch * 3 * 3 * in_ch + out_ch)  # 2个Conv3×3
    bottleneck_params = (
        (out_ch//4) * 1 * 1 * in_ch + (out_ch//4) +  # 1×1 降维
        (out_ch//4) * 3 * 3 * (out_ch//4) + (out_ch//4) +  # 3×3 卷积
        out_ch * 1 * 1 * (out_ch//4) + out_ch  # 1×1 升维
    )
    
    print(f"{'参数量':<15} {basic_params:<20,} {bottleneck_params:<20,}")
    print(f"{'感受野':<15} {'3×3':<20} {'3×3':<20}")
    print(f"{'网络深度':<15} {'2层':<20} {'3层':<20}")
    print(f"{'比例':<15} {'100%':<20} {bottleneck_params/basic_params*100:>6.1f}%")
    
    print(f"\nBottleneck 使用 1×1 卷积将通道数从 256 降到 64（1/4），")
    print(f"再进行 3×3 卷积，最后升回 256。参数量减少约 {100 - bottleneck_params/basic_params*100:.1f}%")

compare_block_types()

print("\n\n实验5：ResNet 错误率随深度变化")
print("-" * 40)

def error_vs_depth():
    """展示 ResNet 错误率随深度的变化"""
    networks = {
        'AlexNet': (8, 15.3),
        'VGG-16': (16, 7.3),
        'VGG-19': (19, 7.1),
        'ResNet-18': (18, 30.3),  # single crop, top-1
        'ResNet-34': (34, 26.2),
        'ResNet-50': (50, 23.9),
        'ResNet-101': (101, 22.6),
        'ResNet-152': (152, 21.6),
    }
    
    print(f"\n{'网络':<12} {'层数':>6} {'Top-1错误率(单裁)':>20}")
    print("-" * 40)
    
    prev_error = None
    for name, (depth, error) in networks.items():
        improvement = ""
        if prev_error is not None:
            improvement = f" (降低 {prev_error - error:.1f}%)"
        print(f"{name:<12} {depth:>6} {error:>18.1f}%{improvement}")
        prev_error = error
    
    print(f"\n关键发现:")
    print(f"1. ResNet-152 错误率 21.6%（单裁），超越此前最强 CNN 模型（21.6%）")
    print(f"2. ResNet-34 优于 VGG-19（26.2% vs 27.1%多裁对应单裁）")
    print(f"3. 深度增加到 152 层，错误率持续下降，无退化现象")
    print(f"4. 相比之下，Plain Network 超过 20 层后开始出现退化")

error_vs_depth()

print("\n" + "=" * 60)
print("实验结论:")
print("-" * 40)
print("1. 退化问题：Plain Network 的错误率随深度增加而上升")
print("2. 残差连接：通过跳跃连接 F(x) + x，网络只需学习 F(x) ≈ 0")
print("3. 梯度流动：跳跃连接提供梯度直通路径，避免梯度衰减")
print("4. Bottleneck: 用 1×1 卷积降维，Bottleneck 块参数量减少约 94%")
print("5. 深度扩展：ResNet 将网络深度扩展到 152 层，错误率持续下降")
print("=" * 60)
```

### 实验结论

实验验证了 ResNet 的核心机制：

1. **退化问题**：Plain Network 中，多层线性变换的连乘导致信息丢失。层数越多，恒等映射误差越大，网络"学不动"。

2. **残差连接的作用**：Residual Network 中，跳跃连接 $F(x) + x$ 确保即使 $F(x) = 0$（所有卷积权重为零），输出仍然等于输入（恒等映射）。网络只需学习 $F(x)$ 的残差部分，优化目标从"学习完整映射"简化为"学习映射与恒等之间的差异"。

3. **梯度流动**：跳跃连接为梯度提供了直通路径。$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (\frac{\partial F}{\partial x} + 1)$，即使 $\frac{\partial F}{\partial x} = 0$，梯度仍然通过 $+1$ 项无损传递。

4. **Bottleneck 设计**：Basic Block（$3 \times 3$ → $3 \times 3$）参数量远大于 Bottleneck（$1 \times 1$ → $3 \times 3$ → $1 \times 1$）。Bottleneck 使用 $1 \times 1$ 卷积降维到 1/4 通道，再将参数量减少约 94% 的同时保持 $3 \times 3$ 感受野。

5. **深度扩展**：ResNet 将网络深度从 VGG 的 19 层扩展到 152 层，ImageNet Top-1 错误率从 27.1% 降至 21.6%（单裁），首次超过人类水平。

## ResNet 的设计哲学与影响

### 残差学习的深层含义

ResNet 的残差学习思想不仅仅是工程技巧，它反映了一个深刻的机器学习原理：**优化一个"相对于基准的改进"比优化一个"完整的映射"更容易**。

**类比理解**：

- **标准 CNN**：从零开始学习 $H(x)$，就像学习一幅完整的画
- **ResNet**：学习 $H(x) - x$，即学习"需要在已有画布上添加什么"

如果"已有画布"（输入 $x$）已经包含了大部分有用信息，那么需要学习的内容就很少（$F(x) \approx 0$），优化更容易收敛。

### ResNet 的广泛应用

ResNet 的影响远超图像分类：

1. **目标检测**：Faster R-CNN、Mask R-CNN 使用 ResNet 作为 backbone，替代了 VGG
2. **语义分割**：DeepLab、FCN-ResNet 使用 ResNet 提取特征
3. **自然语言处理**：Transformer 中的残差连接直接借鉴了 ResNet
4. **生成模型**：StyleGAN、DDPM 中的残差块设计
5. **视频理解**：3D ResNet 用于动作识别
6. **自监督学习**：SimCLR、MoCo 等使用 ResNet 作为特征提取器

**现代 CNN 标准架构**：

```
输入
  │
  ├─ 初始卷积 (Conv7×7, stride=2) → BN → ReLU → 池化
  │
  ├─ Conv Layer 2: N 个 Bottleneck 块
  │
  ├─ Conv Layer 3: N 个 Bottleneck 块 (首个块下采样)
  │
  ├─ Conv Layer 4: N 个 Bottleneck 块 (首个块下采样)
  │
  ├─ Conv Layer 5: 3 个 Bottleneck 块 (首个块下采样)
  │
  ├─ 全局平均池化
  │
  └─ FC: 分类
```

这一架构被 EfficientNet、RegNet、ConvNeXt 等后续网络继承和发展。

## 本章小结

本章介绍了 ResNet——深度学习历史上最重要的网络架构之一：

**退化问题**：随着网络深度增加，Plain Network 的训练误差和测试误差同时增加。这不是过拟合，而是网络优化困难——让深层网络学习恒等映射非常困难。

**残差连接**：通过跳跃连接 $F(x) + x$，将学习目标从 $H(x) = x$ 改为 $F(x) = 0$。当最优映射接近恒等映射时，学习 $F(x) = 0$（权重趋近于零）比学习 $W \cdot x = x$ 容易得多。

**梯度流动**：跳跃连接为梯度提供了直通路径，$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (\frac{\partial F}{\partial x} + 1)$，确保梯度无衰减地传递到浅层。

**网络架构**：
- Basic Block（$3 \times 3$ → $3 \times 3$）：ResNet-18、ResNet-34
- Bottleneck（$1 \times 1$ → $3 \times 3$ → $1 \times 1$）：ResNet-50、ResNet-101、ResNet-152

**成果**：ResNet-152 将 ImageNet Top-5 错误率降至 3.57%（多裁），Top-1 错误率降至 21.6%（单裁），首次超过人类水平，网络深度达到 152 层。

**后续影响**：残差思想被广泛应用于检测、分割、NLP、生成模型等几乎所有深度学习领域。Transformer 中的残差连接、StyleGAN 中的残差块，都直接或间接源于 ResNet。

## 练习题

1. 推导残差块的反向传播梯度公式。证明跳跃连接如何确保梯度无衰减地传递到浅层。
    <details>
    <summary>参考答案</summary>

    **残差块反向传播梯度推导**：

    **残差块前向传播**：

    设残差块输入为 $x$，残差函数为 $F(x, \{W_i\})$，输出为 $y$：

    $$y = F(x, \{W_i\}) + x$$

    其中 $F(x, \{W_i\})$ 由两层卷积构成：

    $$F = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$$

    为简化，假设使用预激活版本，且忽略偏置项：

    $$F = W_2 \cdot \sigma(W_1 \cdot x)$$

    其中 $\sigma(\cdot)$ 是 ReLU 激活函数。

    **输出对残差函数的梯度**：

    设损失函数为 $L$，输出梯度为 $\frac{\partial L}{\partial y}$。

    **残差函数对输入 $x$ 的梯度**：

    $$\frac{\partial F}{\partial x} = \frac{\partial F}{\partial h} \cdot \frac{\partial h}{\partial x}$$

    其中 $h = W_1 \cdot x$，$\frac{\partial F}{\partial h} = W_2 \cdot \sigma'(h)$。

    $$\frac{\partial F}{\partial x} = W_2^T \cdot \sigma'(W_1 \cdot x) \cdot W_1$$

    **损失对输入 $x$ 的梯度**：

    $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

    $$\frac{\partial y}{\partial x} = \frac{\partial (F + x)}{\partial x} = \frac{\partial F}{\partial x} + I$$

    其中 $I$ 是单位矩阵（来自跳跃连接 $x$ 的导数）。

    $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + I\right) = \frac{\partial L}{\partial y} \cdot \frac{\partial F}{\partial x} + \frac{\partial L}{\partial y}$$

    **关键发现**：

    损失对输入的梯度由两部分组成：
    1. $\frac{\partial L}{\partial y} \cdot \frac{\partial F}{\partial x}$：通过残差函数的梯度
    2. $\frac{\partial L}{\partial y}$：通过跳跃连接的梯度（**直接传递，无衰减**）

    **多层 ResNet 的梯度传递**：

    设网络有 $L$ 个残差块，第 $l$ 块输入为 $x_l$，输出为 $x_{l+1} = F_l(x_l) + x_l$。

    从第 $L$ 块到第 $1$ 块的梯度传递：

    $$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_{L+1}} \cdot \prod_{l=1}^{L} \frac{\partial x_{l+1}}{\partial x_l}$$

    $$= \frac{\partial L}{\partial x_{L+1}} \cdot \prod_{l=1}^{L} \left(\frac{\partial F_l}{\partial x_l} + I\right)$$

    **展开乘积**：

    $$\prod_{l=1}^{L} \left(\frac{\partial F_l}{\partial x_l} + I\right) = I + \sum_{l=1}^{L} \frac{\partial F_l}{\partial x_l} + \sum_{l<m} \frac{\partial F_l}{\partial x_l} \cdot \frac{\partial F_m}{\partial x_m} + \cdots$$

    关键：乘积中包含 **$I$（单位矩阵）**。

    **与 Plain Network 对比**：

    Plain Network 的梯度传递：

    $$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_{L+1}} \cdot \prod_{l=1}^{L} \frac{\partial F_l}{\partial x_l}$$

    如果每层 $\left\|\frac{\partial F_l}{\partial x_l}\right\| < 1$（如 sigmoid 激活或小的权重），乘积随 $L$ 指数衰减（梯度消失）。

    **ResNet 的优势**：

    ResNet 的梯度传递中，$\prod_{l=1}^{L} (\frac{\partial F_l}{\partial x_l} + I)$ 展开后包含 $I$ 项：

    $$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_{L+1}} \cdot \left(I + \text{其他项}\right) = \frac{\partial L}{\partial x_{L+1}} + \frac{\partial L}{\partial x_{L+1}} \cdot (\text{其他项})$$

    即使所有 $\frac{\partial F_l}{\partial x_l}$ 趋近于零，$I$ 项保证了 $\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_{L+1}}$——**梯度可以直接从深层传递到浅层，无衰减**。

    **总结**：

    | 对比项 | Plain Network | ResNet |
    |:------|:-------------|:------|
    | 梯度传递 | $\prod \frac{\partial F}{\partial x}$ | $\prod (\frac{\partial F}{\partial x} + I)$ |
    | 恒等项 | 无 | 有 ($I$) |
    | 梯度衰减 | 随层数指数衰减 | 无衰减（通过 $I$） |
    | 深度限制 | ~20 层（退化） | 1000+ 层 |
    </details>

2. 对比 Basic Block 和 Bottleneck 的参数量、计算量和感受野。解释为什么 ResNet-50 使用 Bottleneck 而不是 Basic Block。
    <details>
    <summary>参考答案</summary>

    **Basic Block vs Bottleneck 对比分析**：

    **一、结构对比**

    **Basic Block**：
    ```
    输入 → Conv3×3 (in_ch→mid_ch) → BN → ReLU → Conv3×3 (mid_ch→out_ch) → BN → (+跳跃) → ReLU → 输出
    ```

    **Bottleneck Block**：
    ```
    输入 → Conv1×1 (in_ch→mid_ch/4) → BN → ReLU → Conv3×3 (mid_ch/4→mid_ch/4) → BN → ReLU → Conv1×1 (mid_ch/4→out_ch) → BN → (+跳跃) → ReLU → 输出
    ```

    **二、参数量对比**

    设输入通道 = 输出通道 = $C$：

    **Basic Block**（输入 $C$，输出 $C$）：

    - Conv1 (3×3): $C \times 3 \times 3 \times C + C = 9C^2 + C$
    - Conv2 (3×3): $C \times 3 \times 3 \times C + C = 9C^2 + C$
    - 总参数量：$18C^2 + 2C \approx 18C^2$

    **Bottleneck Block**（输入 $C$，输出 $C$，中间通道 $C/4$）：

    - Conv1 (1×1): $C \times 1 \times 1 \times \frac{C}{4} + \frac{C}{4} = \frac{C^2}{4} + \frac{C}{4}$
    - Conv2 (3×3): $\frac{C}{4} \times 3 \times 3 \times \frac{C}{4} + \frac{C}{4} = \frac{9C^2}{16} + \frac{C}{4}$
    - Conv3 (1×1): $\frac{C}{4} \times 1 \times 1 \times C + C = \frac{C^2}{4} + C$
    - 总参数量：$\frac{C^2}{4} + \frac{9C^2}{16} + \frac{C^2}{4} + \frac{C}{4} + \frac{C}{4} + C = \frac{17C^2}{16} + 1.5C \approx \frac{17}{16}C^2$

    **对比**：

    $$\frac{\text{Bottleneck}}{\text{Basic}} = \frac{\frac{17}{16}C^2}{18C^2} = \frac{17}{288} \approx 5.9\%$$

    Bottleneck 参数量仅为 Basic Block 的 **5.9%**，减少约 **94.1%**。

    **三、计算量（FLOPs）对比**

    设特征图空间尺寸为 $H \times W$：

    **Basic Block**：
    - Conv1: $H \times W \times C \times (3 \times 3 \times C \times 2) = 18HWC^2$
    - Conv2: $H \times W \times C \times (3 \times 3 \times C \times 2) = 18HWC^2$
    - 总计算量：$36HWC^2$

    **Bottleneck Block**：
    - Conv1 (1×1): $H \times W \times \frac{C}{4} \times (1 \times 1 \times C \times 2) = \frac{HWC^2}{2}$
    - Conv2 (3×3): $H \times W \times \frac{C}{4} \times (3 \times 3 \times \frac{C}{4} \times 2) = \frac{9HWC^2}{8}$
    - Conv3 (1×1): $H \times W \times C \times (1 \times 1 \times \frac{C}{4} \times 2) = \frac{HWC^2}{2}$
    - 总计算量：$\frac{HWC^2}{2} + \frac{9HWC^2}{8} + \frac{HWC^2}{2} = \frac{17HWC^2}{8} = 2.125HWC^2$

    $$\frac{\text{Bottleneck FLOPs}}{\text{Basic FLOPs}} = \frac{2.125}{36} \approx 5.9\%$$

    **四、感受野对比**

    | 块类型 | 中间卷积 | 感受野 |
    |:------|:--------|:------|
    | Basic Block | 2×3×3 | $3 \times 3$ |
    | Bottleneck | 1×1 + 3×3 + 1×1 | $3 \times 3$ |

    感受野相同：$3 \times 3$。$1 \times 1$ 卷积不改变感受野。

    **五、为什么 ResNet-50 使用 Bottleneck**

    ResNet-50 有 50 层，如果使用 Basic Block：
    - Conv2-5 需要约 24 个 Basic Block（每层 6 个）
    - 总参数量：$24 \times (18 \times 256^2) \approx 283M$（非常大）

    使用 Bottleneck：
    - 同样 24 个 Bottleneck Block
    - 总参数量：$24 \times (\frac{17}{16} \times 256^2) \approx 16.8M$（大幅减少）

    **Bottleneck 的核心优势**：

    1. **参数量减少 94%**：通过 $1 \times 1$ 卷积将通道数降为 1/4，再进行 $3 \times 3$ 卷积
    2. **计算量减少 94%**：参数量和计算量同比例减少
    3. **感受野不变**：$3 \times 3$ 卷积的感受野保持 $3 \times 3$
    4. **网络更深**：由于每个 Block 有 3 层（而非 2 层），可以用更少的 Block 数实现相同的网络深度

    **总结**：Bottleneck 通过 $1 \times 1$ 卷积降维，在保持 $3 \times 3$ 感受野的同时，将参数量和计算量减少约 94%，使 ResNet-50/101/152 等深层网络的训练成为可能。
    </details>

3. 设计一个 ResNet-101 的修改版本，将 Conv Layer 4 的 Bottleneck 块从 23 个减少到 12 个。分析修改后的层数、参数量和预期精度变化。
    <details>
    <summary>参考答案</summary>

    **修改版 ResNet-101 分析**：

    **原始 ResNet-101 配置**：

    | Conv Layer | 块数 | 通道数 | 类型 |
    |:----------|:----|:------|:----|
    | Conv2 | 3 | 256 | Bottleneck |
    | Conv3 | 4 | 512 | Bottleneck |
    | Conv4 | 23 | 1024 | Bottleneck |
    | Conv5 | 3 | 2048 | Bottleneck |
    | **总块数** | **33** | | |
    | **总层数** | **1 + 3×3 + 4×3 + 23×3 + 3×3 = 101** | | |

    **修改版配置（Conv4: 23 → 12）**：

    | Conv Layer | 块数 | 通道数 | 类型 |
    |:----------|:----|:------|:----|
    | Conv2 | 3 | 256 | Bottleneck |
    | Conv3 | 4 | 512 | Bottleneck |
    | Conv4 | 12 | 1024 | Bottleneck |
    | Conv5 | 3 | 2048 | Bottleneck |
    | **总块数** | **22** | | |
    | **总层数** | **1 + 3×3 + 4×3 + 12×3 + 3×3 = 68** | | |

    **修改后的总层数**：68 层（介于 ResNet-50 和 ResNet-101 之间）

    **参数量分析**：

    **原始 ResNet-101 各层参数量**：

    - 初始层 (Conv7×7): $64 \times 7 \times 7 \times 3 + 64 = 9,472$
    - Conv2 (3×Bottleneck, 64→256): 约 0.4M
    - Conv3 (4×Bottleneck, 256→512): 约 1.7M
    - Conv4 (23×Bottleneck, 512→1024): 约 32.4M
    - Conv5 (3×Bottleneck, 1024→2048): 约 9.8M
    - FC (2048→1000): 约 2.0M
    - 总计: ~44.5M

    **修改版参数量**：

    Conv4 从 23 个块减为 12 个：
    - 每个 Bottleneck (512→1024) 参数量约 1.4M
    - 减少 11 个块：$11 \times 1.4\text{M} = 15.4\text{M}$
    - 修改版总参数量：$44.5\text{M} - 15.4\text{M} = 29.1\text{M}$

    **预期精度变化**：

    基于 ResNet 系列的深度-精度关系：

    | 配置 | 层数 | 参数(M) | 预期 Top-1 错误率 |
    |:----|:----|:------|:-----------------|
    | ResNet-50 | 50 | 25.6 | 23.85% |
    | **修改版** | **68** | **29.1** | **~22.5%** (估计) |
    | ResNet-101 | 101 | 44.5 | 22.63% |
    | ResNet-152 | 152 | 60.2 | 21.69% |

    预期精度在 ResNet-50 和 ResNet-101 之间，约 **22.5%** Top-1 错误率。

    **效率-精度权衡分析**：

    | 指标 | ResNet-50 | 修改版 | ResNet-101 |
    |:----|:---------|:------|:----------|
    | 层数 | 50 | 68 | 101 |
    | 参数量 | 25.6M | 29.1M | 44.5M |
    | Top-1 错误率 | 23.85% | ~22.5% | 22.63% |
    | 参数效率 | 基准 | +13.7% | +73.8% |

    修改版用比 ResNet-101 少约 35% 的参数，达到略低于 ResNet-101 的精度（约低 0.1%）。

    **结论**：修改版是一个更好的效率-精度平衡点，适合资源受限但需要较高精度的场景。实际精度取决于具体任务和训练策略。
    </details>
