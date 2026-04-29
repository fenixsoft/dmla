# VGG与GoogLeNet

上一章介绍了 AlexNet——2012 年 ImageNet 挑战赛的突破性模型。AlexNet 的成功验证了深度卷积神经网络在大规模视觉任务上的有效性，但 AlexNet 的架构设计仍然相对粗糙：8 层网络中使用大卷积核（$11 \times 11$、$5 \times 5$），全连接层参数量占总参数的 94%。

2014 年（arXiv 预印本），两篇重要的研究工作从不同方向改进了 AlexNet 的设计：**VGGNet**（牛津大学 Visual Geometry Group）证明"更深"是提升精度的有效途径，而 **GoogLeNet**（Google）提出了 Inception 模块，用更高效的架构设计同时降低了参数量和错误率。

这两项工作代表了 CNN 架构设计的两个核心问题：**网络深度与宽度的权衡**，以及**如何设计高效的卷积结构**。

## VGGNet：深度探索

### VGG 的动机

AlexNet 之后，研究者面临一个关键问题：**网络性能的提升究竟来自哪里？** 是网络深度、宽度、卷积核大小、训练技巧，还是数据增强？

VGG 的核心思想非常简洁：**在宽度和其他超参数不变的情况下，增加网络深度可以提升精度**。VGG 将网络深度从 AlexNet 的 8 层扩展到 16-19 层，同时将错误率从 AlexNet 的 15.3% 降至 **7.3%**（Top-5）。

### VGG 架构设计

VGG 的架构具有极强的规律性——全部使用 $3 \times 3$ 卷积核，通道数逐层递增，每经过一次池化空间尺寸减半。这种高度统一的风格使其易于实现和理解。

**VGG-16 网络结构**（最常用版本）：

```
输入: 224×224×3 (RGB图像)
  │
  ├─ Block 1: Conv3×3, 64 → Conv3×3, 64 → 池化2×2, stride=2
  │           输出: 112×112×64
  │
  ├─ Block 2: Conv3×3, 128 → Conv3×3, 128 → 池化2×2, stride=2
  │           输出: 56×56×128
  │
  ├─ Block 3: Conv3×3, 256 → Conv3×3, 256 → Conv3×3, 256 → 池化2×2, stride=2
  │           输出: 28×28×256
  │
  ├─ Block 4: Conv3×3, 512 → Conv3×3, 512 → Conv3×3, 512 → 池化2×2, stride=2
  │           输出: 14×14×512
  │
  ├─ Block 5: Conv3×3, 512 → Conv3×3, 512 → Conv3×3, 512 → 池化2×2, stride=2
  │           输出: 7×7×512
  │
  ├─ FC6:    4096 → ReLU → Dropout(0.5)
  │
  ├─ FC7:    4096 → ReLU → Dropout(0.5)
  │
  └─ FC8:    1000 → Softmax
```

**命名说明**：VGG-16 中的 "16" 表示 16 个带权重的层（13 个卷积层 + 3 个全连接层）。VGG 系列包含多个变体：

| 配置 | 层数 | 卷积块 | 参数量 | Top-5 错误率 |
|:----|:----|:------|:------|:------------|
| VGG-A | 8 | 2-2-2-2-2 | ~133M | 23.1% |
| VGG-B | 11 | 2-2-2-2-2 | ~132M | 22.4% |
| VGG-C (VGG-16) | 13+3 | 2-2-3-3-3 | ~138M | **7.2%** |
| VGG-D (VGG-16) | 13+3 | 2-2-3-3-3 | ~138M | **7.3%** |
| VGG-E (VGG-19) | 16+3 | 2-2-4-4-4 | ~144M | **7.0%** |

VGG-C 和 VGG-D 参数量相近，VGG-C 多了 4 个 $1 \times 1$ 卷积层（增加非线性）。VGG-D 是最常用的版本。

### $3 \times 3$ 卷积核的优势

VGG 的核心设计选择是**全部使用 $3 \times 3$ 卷积核**（步长 1，填充 1），而非 AlexNet 中的 $11 \times 11$、$5 \times 5$ 大卷积核。这一选择有三个关键优势：

**1. 感受野等价，但参数量更少**

两层 $3 \times 3$ 卷积的级联（堆叠）等价于一个 $5 \times 5$ 卷积的感受野：

$$\text{感受野} = 3 + (3 - 1) \times 1 = 5$$

三层 $3 \times 3$ 卷积的级联等价于一个 $7 \times 7$ 卷积：

$$\text{感受野} = 5 + (3 - 1) \times 1 = 7$$

**参数量对比**（设输入通道 $C$，输出通道 $C$）：

| 方案 | 参数量 | 说明 |
|:----|:------|:----|
| 一个 $5 \times 5$ 卷积 | $5 \times 5 \times C \times C = 25C^2$ | 单层 |
| 两个 $3 \times 3$ 卷积 | $2 \times (3 \times 3 \times C \times C) = 18C^2$ | 两层，感受野=5 |
| 一个 $7 \times 7$ 卷积 | $7 \times 7 \times C \times C = 49C^2$ | 单层 |
| 三个 $3 \times 3$ 卷积 | $3 \times (3 \times 3 \times C \times C) = 27C^2$ | 三层，感受野=7 |

两层 $3 \times 3$ 代替一个 $5 \times 5$，参数量减少 **28%**。三个 $3 \times 3$ 代替一个 $7 \times 7$，参数量减少 **45%**。

**2. 更强的非线性表达能力**

两层 $3 \times 3$ 卷积包含**两个 ReLU 激活函数**，而一个 $5 \times 5$ 卷积只有一个：

```
方案A（单层大卷积）:
  输入 → Conv(5×5) → ReLU → 输出
  非线性变换次数: 1

方案B（多层小卷积）:
  输入 → Conv(3×3) → ReLU → Conv(3×3) → ReLU → 输出
  非线性变换次数: 2
```

更多的 ReLU 激活意味着更强的非线性拟合能力。两个 $3 \times 3$ 卷积构成的函数空间，包含一个 $5 \times 5$ 卷积的函数空间（严格更大），因为中间多了一层非线性变换。

**3. 更深的网络**

使用多个小卷积核替代大卷积核，自然地增加了网络深度。更深的网络能够学习到更抽象的特征表示。VGG 通过这种设计，将网络深度从 AlexNet 的 8 层扩展到 16-19 层。

### VGG 的参数量分析

VGG-16 的总参数量约 **1.38 亿**，远超 AlexNet 的 6200 万：

**卷积层参数量**：

| 层 | 输入通道 | 输出通道 | 卷积核 | 参数量 |
|:--|:--------|:--------|:------|:------|
| Block1-Conv1 | 3 | 64 | $3 \times 3$ | $64 \times 3 \times 3 \times 3 + 64 = 1,792$ |
| Block1-Conv2 | 64 | 64 | $3 \times 3$ | $64 \times 3 \times 3 \times 64 + 64 = 36,928$ |
| Block2-Conv1 | 64 | 128 | $3 \times 3$ | $128 \times 3 \times 3 \times 64 + 128 = 73,856$ |
| Block2-Conv2 | 128 | 128 | $3 \times 3$ | $128 \times 3 \times 3 \times 128 + 128 = 147,584$ |
| Block3-Conv1 | 128 | 256 | $3 \times 3$ | $256 \times 3 \times 3 \times 128 + 256 = 295,168$ |
| Block3-Conv2 | 256 | 256 | $3 \times 3$ | $256 \times 3 \times 3 \times 256 + 256 = 590,080$ |
| Block3-Conv3 | 256 | 256 | $3 \times 3$ | $590,080$ |
| Block4-Conv1 | 256 | 512 | $3 \times 3$ | $512 \times 3 \times 3 \times 256 + 512 = 1,180,160$ |
| Block4-Conv2 | 512 | 512 | $3 \times 3$ | $512 \times 3 \times 3 \times 512 + 512 = 2,359,808$ |
| Block4-Conv3 | 512 | 512 | $3 \times 3$ | $2,359,808$ |
| Block5-Conv1 | 512 | 512 | $3 \times 3$ | $2,359,808$ |
| Block5-Conv2 | 512 | 512 | $3 \times 3$ | $2,359,808$ |
| Block5-Conv3 | 512 | 512 | $3 \times 3$ | $2,359,808$ |

卷积层总参数量：约 **1470 万**。

**全连接层参数量**：

Conv5 池化后输出 $7 \times 7 \times 512 = 25,088$ 维。

| 层 | 输入 | 输出 | 参数量 |
|:--|:----|:----|:------|
| FC6 | 25,088 | 4,096 | $25,088 \times 4,096 + 4,096 = 102,769,664$ |
| FC7 | 4,096 | 4,096 | $4,096 \times 4,096 + 4,096 = 16,781,312$ |
| FC8 | 4,096 | 1,000 | $4,096 \times 1,000 + 1,000 = 4,097,000$ |

全连接层总参数量：约 **1.236 亿**。

**VGG-16 总参数量**：$14,700,000 + 123,600,000 \approx 1.38$ 亿。

**与 AlexNet 对比**：

| 指标 | AlexNet | VGG-16 |
|:----|:--------|:------|
| 卷积层参数量 | 375 万 | 1470 万 |
| 全连接层参数量 | 5863 万 | 1.236 亿 |
| 总参数量 | 6200 万 | ~1.38 亿 |
| Top-5 错误率 | 15.3% | 7.3% |
| 网络深度 | 8 层 | 16 层 |

VGG 的错误率减半，但参数量翻倍，其中大部分参数仍在全连接层。这引出了一个问题：**能否用更少的参数达到同样的精度？**

## Inception 模块设计

### 从多尺度特征到 Inception

VGG 通过增加深度来提升精度。与之不同，GoogLeNet 从一个不同的问题出发：**什么是最优的局部连接模式？**

在传统 CNN 中，每一层使用固定尺寸的卷积核（如 $3 \times 3$ 或 $5 \times 5$）。但不同大小的特征（如小目标和大目标）可能需要不同感受野的卷积核来提取。GoogLeNet 的**核心思想**是：与其选择单一卷积核尺寸，不如同时使用多种尺寸的卷积核，让网络自己学习哪些尺度最有信息量。

### Inception 模块

**Inception 模块**（Inception Module）是 GoogLeNet 的核心构建块。其基本形式如下：

```
Inception 模块基本结构:

输入 (H×W×C_in)
  │
  ├────→ [1×1 卷积, C₁ 个] → ReLU ────→ ─────┐
  │                                            │
  ├────→ [1×1 卷积, C₂ 个] → ReLU             │
  │     → [3×3 卷积, C₃ 个] → ReLU ────→ ─────┤
  │                                            │
  ├────→ [1×1 卷积, C₄ 个] → ReLU             │
  │     → [5×5 卷积, C₅ 个] → ReLU ────→ ─────┤ → 拼接 (Concat) → 输出
  │                                            │
  └────→ [3×3 最大池化] → [1×1 卷积, C₆ 个] → ─┘
```

每个 Inception 模块包含四条并行的路径：

1. **$1 \times 1$ 卷积分支**：直接提取 $1 \times 1$ 尺度的特征
2. **$3 \times 3$ 卷积分支**：先用 $1 \times 1$ 卷积降维，再用 $3 \times 3$ 卷积
3. **$5 \times 5$ 卷积分支**：先用 $1 \times 1$ 卷积降维，再用 $5 \times 5$ 卷积
4. **池化分支**：最大池化后接 $1 \times 1$ 卷积调整通道数

四路输出在通道维度上拼接（Concatenate），得到模块的最终输出。

### $1 \times 1$ 卷积的关键作用

Inception 模块中大量使用了 **$1 \times 1$ 卷积**，这是 GoogLeNet 的另一个关键设计。前面（CNN 基础章节）曾提到 $1 \times 1$ 卷积的三个作用：

1. **通道降维**（减少参数）
2. **增加非线性**
3. **跨通道信息融合**

在 Inception 模块中，$1 \times 1$ 卷积的**降维作用是核心**。

**为什么需要降维？**

假设输入 $28 \times 28 \times 192$（Inception 模块的常见输入尺寸），如果直接应用 $5 \times 5$ 卷积产生 32 个输出通道：

$$\text{参数量} = 5 \times 5 \times 192 \times 32 + 32 = 153,632$$

$$\text{输出尺寸} = 28 \times 28 \times 32$$

但如果先使用 $1 \times 1$ 卷积将通道数从 192 降到 16，再应用 $5 \times 5$ 卷积产生 32 个输出通道：

$$\text{参数量} = (1 \times 1 \times 192 \times 16 + 16) + (5 \times 5 \times 16 \times 32 + 32)$$
$$= 3,088 + 12,832 = 15,920$$

参数量从 153,632 降至 15,920，减少约 **90%**，而输出尺寸保持不变（$28 \times 28 \times 32$）。

**降维的直观理解**：

$1 \times 1$ 卷积相当于在每个空间位置上对通道维度做一个线性组合，相当于一个"信息瓶颈"——将 192 维信息压缩到 16 维，再用这个 16 维表示进行后续的 $5 \times 5$ 卷积。由于同空间位置的信息通常是高度冗余的，这种压缩不会丢失关键信息。

### GoogLeNet 完整架构

GoogLeNet 由 9 个 Inception 模块（Inception blocks）堆叠而成，每个 block 包含多个 Inception 模块。网络中间还加入了两个**辅助分类器**（Auxiliary Classifier），用于缓解梯度消失问题。

```
GoogLeNet 网络结构:

输入: 224×224×3
  │
  ├─ 初始卷积: Conv 7×7, stride=2, padding=3 → 64 通道
  │            → 池化3×3, stride=2
  │            → Conv 1×1, stride=1 → 64 通道
  │            → Conv 3×3, stride=1 → 192 通道
  │            → 池化3×3, stride=2
  │            输出: 56×56×192
  │
  ├─ Inception Block 1: 3a → 3b
  │            输出: 28×28×256+480
  │
  ├─ Inception Block 2: 4a → 4b → 4c → 4d → 4e
  │            辅助分类器1 (在 4e 后)
  │            输出: 14×14×512+512+512
  │
  ├─ Inception Block 3: 5a → 5b
  │            辅助分类器2 (在 5b 后)
  │            输出: 7×7×572+572+1280
  │
  ├─ 全局平均池化: 7×7, stride=1
  │            输出: 1×1×(572+572+1280) = 2424 维
  │
  ├─ Dropout: 0.4
  │
  └─ FC: 2424 → 1000 → Softmax
```

**关键特征**：

> 注：通道数标注 "256+480" 表示多个 Inception 模块输出通道的拼接（如 3a 输出 256 通道，3b 输出 480 通道，拼接后共 736 通道）。

1. **无全连接层**（除了最后的分类层）：GoogLeNet 用全局平均池化替代了 AlexNet/VGG 的三个全连接层，大幅减少了参数量
2. **辅助分类器**：在网络中间位置加入两个额外的分类损失，训练时与主损失加权求和（权重各为 0.3），确保中间层也能学到有用特征
3. **多尺度特征融合**：每个 Inception 模块同时提取 $1 \times 1$、$3 \times 3$、$5 \times 5$ 和池化特征

**GoogLeNet 参数量**：约 **700 万**，不到 AlexNet 的 1/8。

### 辅助分类器的作用

辅助分类器（Auxiliary Classifier）是 GoogLeNet 的一个独特设计。在网络中间两个位置（Inception 4e 和 Inception 5b 后）各放置一个分类头：

```
辅助分类器结构:

Inception 输出
  │
  ├─ 平均池化: 5×5, stride=3 (减小尺寸)
  │
  ├─ 1×1 卷积: 降维到 128 通道
  │
  ├─ Flatten
  │
  ├─ FC: → 1024 → ReLU → Dropout(0.7)
  │
  └─ FC: → 1000 (Softmax 在推理时被忽略)
```

**训练时的损失函数**：

$$L = L_{main} + 0.3 \times L_{aux1} + 0.3 \times L_{aux2}$$

其中 $L_{main}$ 是主分类损失，$L_{aux1}$ 和 $L_{aux2}$ 是两个辅助分类器的损失。

**作用**：

1. **缓解梯度消失**：辅助分类器在中间层直接计算损失，提供额外的梯度信号，确保浅层网络也能获得足够的梯度
2. **正则化效果**：辅助分类器迫使中间层学习到判别性特征，防止中间层"偷懒"（学习无意义的表示）

**推理时的处理**：辅助分类器在推理时被丢弃，只保留主分类器的输出。

### GoogLeNet vs VGG 参数量对比

| 网络 | 总参数量 | 卷积层参数 | 全连接层参数 | Top-5 错误率 |
|:----|:--------|:----------|:------------|:------------|
| AlexNet | 62M | 3.75M | 58.63M | 15.3% |
| VGG-16 | 138M | 14.7M | 123.3M | 7.3% |
| GoogLeNet | ~7M | ~4.6M | ~2.4M | **6.7%** |

GoogLeNet 用不到 VGG 1/20 的参数量，达到了更低的错误率。这是架构设计的胜利——Inception 模块通过多尺度特征融合和 $1 \times 1$ 卷积降维，实现了参数效率的大幅提升。

## 网络深度与宽度的权衡

### 深度与宽度的定义

在 CNN 设计中，**深度**（Depth）指网络的层数，**宽度**（Width）指每层的通道数（特征数量）。

```
深度增加:                    宽度增加:

输入 →                        输入 →
Conv (64) →                   Conv (64 → 128) →
Conv (64) →                   Conv (128 → 256) →
Conv (64) →                   Conv (256 → 512) →
Conv (128) →                  池化 (2×2 空间尺寸)
Conv (128) →
Conv (128) →
池化                          池化
```

### 深度 vs 宽度的效果

**VGG 选择了"深度优先"**：

- 将 AlexNet 的 8 层扩展到 16-19 层
- 通道数从 64 递增到 512
- 错误率从 15.3% 降至 7.3%
- 代价：参数量从 62M 增加到 138M

**GoogLeNet 选择了"宽度优先"**：

- 保持相对较浅的层数（约 22 层，但 Inception 模块内是并行而非串行）
- 每个模块的输出通道数（即网络宽度）较大（多个分支拼接）
- 错误率降至 6.7%
- 代价：计算量（FLOPs）较大，但参数量仅 7M

### 感受野与深度的关系

网络深度直接影响感受野——层数越多，感受野越大。

**VGG 的感受野**：

VGG-16 每层使用 $3 \times 3$ 卷积，步长 1，填充 1（尺寸不变），每经过一次池化（步长 2）尺寸减半。

**池化对感受野的影响**：池化步长 2 使输出空间尺寸减半，这意味着后续每层卷积的每个输出位置对应的输入区域扩大了 2 倍。感受野的递推公式中，$\prod_{i=1}^{l-1} s_i$（步长累积）在每次池化后翻倍，因此后续层感受野的增长量也翻倍。例如，池化前的感受野增量为 $k-1$，池化后变为 $2(k-1)$。

```
Layer    空间尺寸    感受野    说明
输入      224×224    1        -
Block1    112×112    3        Conv3+Conv3 (2层)
Block2     56×56    6        Pool2 使后续感受野翻倍
Block3     28×28    12       Pool2 使感受野翻倍
Block4     14×14    24       Pool2 使感受野翻倍
Block5      7×7     48       Pool2 使感受野翻倍
```

Block 5 的每个位置对应输入 $48 \times 48$ 区域。对于 $224 \times 224$ 的输入，这个感受野相对较小——深层特征无法"看到"整个图像。这是 VGG 的一个设计限制。

**感受野的改进**：后续网络（如 DeepLab 系列）使用空洞卷积（Atrous/Dilated Convolution）在不增加参数量的情况下扩大感受野，解决了这个问题。

### 计算量 vs 参数量的权衡

在评估网络效率时，需要考虑两个指标：

- **参数量**（Parameters）：影响内存占用
- **计算量**（FLOPs，Floating Point Operations）：影响推理速度

VGG 和 GoogLeNet 在这两个指标上展现了不同的权衡：

| 网络 | 参数量 | 计算量 (FLOPs) | Top-5 错误率 |
|:----|:------|:--------------|:------------|
| VGG-16 | ~138M | ~15.5B | 7.3% |
| GoogLeNet | ~7M | ~1.4B | 6.7% |

GoogLeNet 在参数量和计算量上都远优于 VGG，同时达到了更低的错误率。这证明了 Inception 模块的设计效率——通过 $1 \times 1$ 卷积降维和多尺度特征融合，在减少资源消耗的同时提升了精度。

## VGG 与 GoogLeNet 实验验证

通过代码实现 VGG 简化版和 Inception 模块，验证其架构设计和计算特性。

```python runnable
import numpy as np

print("=" * 60)
print("实验：VGG 与 GoogLeNet 架构对比")
print("=" * 60)
print()

# ============================================================
# 实验1：VGG 架构分析
# ============================================================
print("实验1：VGG-16 架构分析")
print("-" * 40)

class VGG16Analyzer:
    """VGG-16 架构分析器"""
    
    def __init__(self):
        # VGG-16 配置: 每个Block的卷积层数
        self.cfg = [64, 64, 'M',    # Block 1
                    128, 128, 'M',   # Block 2
                    256, 256, 256, 'M',  # Block 3
                    512, 512, 512, 'M',  # Block 4
                    512, 512, 512, 'M']  # Block 5
    
    def build_and_analyze(self):
        """构建并分析 VGG-16 架构"""
        h, w, c = 224, 224, 3
        total_params = 0
        conv_params = 0
        fc_params = 0
        
        print(f"{'层':<10} {'类型':<6} {'输入尺寸':<15} {'输出尺寸':<15} {'参数量':<15}")
        print("-" * 65)
        
        layer_idx = 0
        for item in self.cfg:
            if item == 'M':
                # 最大池化
                h = (h - 2) // 2 + 1
                w = (w - 2) // 2 + 1
                print(f"{'Pool'+str(layer_idx):<10} {'Pool':<6} {f'{h*2}x{w*2}x{c}':<15} {f'{h}x{w}x{c}':<15} {'0':<15}")
            else:
                layer_idx += 1
                # 卷积层: 3x3, stride=1, padding=1
                in_ch = c
                out_ch = item
                params = out_ch * 3 * 3 * in_ch + out_ch
                total_params += params
                conv_params += params
                
                in_str = f"{h}x{w}x{in_ch}" if layer_idx > 1 else f"{h}x{w}x{c}"
                c = out_ch
                
                # 如果是Block的第一层，输入尺寸可能是池化后的
                if layer_idx in [1, 3, 6, 9, 12]:
                    in_str = f"{h}x{w}x{in_ch}"
                
                print(f"{'Conv'+str(layer_idx):<10} {'Conv':<6} {in_str:<15} {f'{h}x{w}x{c}':<15} {f'{params/1e6:.3f}M':<15}")
        
        # 全连接层
        flatten_dim = h * w * c
        fc_layers = [(flatten_dim, 4096), (4096, 4096), (4096, 1000)]
        for in_dim, out_dim in fc_layers:
            params = in_dim * out_dim + out_dim
            total_params += params
            fc_params += params
            print(f"{'FC':<10} {'FC':<6} {str(in_dim):<15} {str(out_dim):<15} {f'{params/1e6:.2f}M':<15}")
        
        print("-" * 65)
        print(f"\n参数量汇总:")
        print(f"  卷积层: {conv_params/1e6:.2f}M ({conv_params/total_params*100:.1f}%)")
        print(f"  全连接层: {fc_params/1e6:.2f}M ({fc_params/total_params*100:.1f}%)")
        print(f"  总计: {total_params/1e6:.2f}M")
        print(f"  最终输出尺寸: {h}x{w}x{c}")
        
        return total_params, conv_params, fc_params

vgg = VGG16Analyzer()
vgg_total, vgg_conv, vgg_fc = vgg.build_and_analyze()

print("\n\n实验2：感受野深度对比")
print("-" * 40)

def compute_receptive_field(layers):
    """
    计算感受野
    layers: 列表 [(kernel_size, stride), ...]
    """
    rf = 1
    stride_prod = 1
    result = [(1, 1, 1)]  # (layer, rf, stride_prod)
    
    for i, (k, s) in enumerate(layers):
        rf = rf + (k - 1) * stride_prod
        stride_prod = stride_prod * s
        result.append((i + 2, rf, stride_prod))
    
    return result

# VGG 风格
print("VGG-16 感受野:")
print(f"{'层':<8} {'卷积核':<10} {'感受野':<12} {'步长累积':<10}")
print("-" * 42)

vgg_layers = [
    (3, 1), (3, 1),   # Block 1
    (2, 2),             # Pool 1
    (3, 1), (3, 1),   # Block 2
    (2, 2),             # Pool 2
    (3, 1), (3, 1), (3, 1),  # Block 3
    (2, 2),             # Pool 3
    (3, 1), (3, 1), (3, 1),  # Block 4
    (2, 2),             # Pool 4
    (3, 1), (3, 1), (3, 1),  # Block 5
    (2, 2),             # Pool 5
]

rf = 1
sp = 1
print(f"{'输入':<8} {'-':<10} {rf:<12} {sp:<10}")

block_names = {3: 'Pool1', 6: 'Pool2', 10: 'Pool3', 14: 'Pool4', 18: 'Pool5'}
for i, (k, s) in enumerate(vgg_layers):
    if i > 0:
        rf = rf + (k - 1) * sp
        sp = sp * s
    layer_name = f"L{i+1}"
    if i + 1 in block_names:
        layer_name = block_names[i + 1]
    print(f"{layer_name:<8} {f'{k}x{k}' if k > 1 else '2x2':<10} {rf:<12} {sp:<10}")

print(f"\nVGG-16 最终感受野: {rf}x{rf}")
print(f"输入图像: 224x224")
print(f"最终空间尺寸: 7x7")
print(f"每个输出位置覆盖输入: {rf}x{rf} = {rf**2} 像素")
print(f"覆盖比例: {rf**2 / (224**2) * 100:.2f}%")

print("\n\n实验3：3×3 vs 5×5 vs 7×7 参数量对比")
print("-" * 40)

def compare_kernel_params(in_ch, out_ch):
    """对比不同卷积核尺寸的参数量"""
    print(f"\n输入通道: {in_ch}, 输出通道: {out_ch}")
    print(f"{'卷积核':<10} {'参数量':<15} {'相对大小':<10}")
    print("-" * 38)
    
    params_3x3 = 3 * 3 * in_ch * out_ch + out_ch
    params_5x5 = 5 * 5 * in_ch * out_ch + out_ch
    params_7x7 = 7 * 7 * in_ch * out_ch + out_ch
    
    print(f"{'3×3':<10} {params_3x3:<15,} {'1.0x':<10}")
    print(f"{'5×5':<10} {params_5x5:<15,} {params_5x5/params_3x3:.1f}x")
    print(f"{'7×7':<10} {params_7x7:<15,} {params_7x7/params_3x3:.1f}x")
    
    return params_3x3, params_5x5, params_7x7

# VGG Block 1
print("VGG Block 1 (64→64):")
compare_kernel_params(64, 64)

# VGG Block 3
print("\nVGG Block 3 (256→256):")
compare_kernel_params(256, 256)

print("\n\n实验4：1×1 卷积降维效果")
print("-" * 40)

def inception_with_1x1_reduction(input_ch, bottleneck_ch, output_ch):
    """计算使用 1×1 卷积降维后的参数量"""
    # 1×1 降维 + 5×5 卷积
    params_1x1 = 1 * 1 * input_ch * bottleneck_ch + bottleneck_ch
    params_5x5 = 5 * 5 * bottleneck_ch * output_ch + output_ch
    return params_1x1 + params_5x5

def inception_without_reduction(input_ch, output_ch):
    """不使用 1×1 卷积降维，直接 5×5"""
    return 5 * 5 * input_ch * output_ch + output_ch

print("Inception 模块中 1×1 卷积降维效果:")
print(f"\n{'场景':<12} {'输入通道':>8} {'1×1输出':>8} {'5×5输出':>8} {'无降维':>12} {'降维后':>12} {'节省':>8}")
print("-" * 72)

scenarios = [
    (192, 16, 32),   # Inception 3a
    (256, 16, 32),   # Inception 3b
    (480, 32, 64),   # Inception 4a
    (832, 64, 128),  # Inception 5a
]

for inp, bot, out in scenarios:
    params_no_red = inception_without_reduction(inp, out)
    params_with_red = inception_with_1x1_reduction(inp, bot, out)
    savings = (1 - params_with_red / params_no_red) * 100
    print(f"{'Inception':<12} {inp:>8} {bot:>8} {out:>8} {params_no_red:>12,} {params_with_red:>12,} {savings:>7.1f}%")

print("\n\n实验5：网络对比汇总")
print("-" * 40)

networks = {
    'AlexNet': {'params': 62, 'conv': 3.75, 'fc': 58.63, 'depth': 8, 'top5_error': 15.3, 'year': 2012},
    'VGG-16': {'params': 138, 'conv': 14.7, 'fc': 123.3, 'depth': 16, 'top5_error': 7.3, 'year': 2014},
    'GoogLeNet': {'params': 7, 'conv': 6.7, 'fc': 0.7, 'depth': 22, 'top5_error': 6.7, 'year': 2014},
}

print(f"\n{'网络':<12} {'年份':>4} {'深度':>4} {'总参数(M)':>10} {'卷积(M)':>9} {'FC(M)':>8} {'错误率':>6}")
print("-" * 56)

for name, info in networks.items():
    print(f"{name:<12} {info['year']:>4} {info['depth']:>4} {info['params']:>10.1f} {info['conv']:>9.1f} {info['fc']:>8.1f} {info['top5_error']:>5.1f}%")

print(f"\n关键发现:")
print("1. VGG-16 通过增加深度，将错误率降至 7.3%（AlexNet 的约一半）")
print("2. GoogLeNet 通过 Inception 模块，用更少的参数达到更低错误率（6.7%）")
print("3. $1 \\times 1$ 卷积降维可节省 80-90% 的参数量")
print("4. $3 \\times 3$ 卷积核相比大卷积核，参数量减少 28-45%")
print("=" * 60)
```

### 实验结论

实验对比了 VGG 和 GoogLeNet 的关键设计：

1. **VGG-16 参数量**：约 1.38 亿，全连接层占 89%，卷积层占 11%。参数量远超 AlexNet（6200 万），但通过增加深度实现了精度的显著提升。

2. **感受野**：VGG-16 经过 5 次池化（每次步长 2），最终空间尺寸为 $7 \times 7$。最终感受野约 $48 \times 48$，仅覆盖输入图像的 4.6%。这说明 VGG 的深层感受野相对不足，限制了全局信息的整合。

3. **卷积核尺寸**：$3 \times 3$ 卷积核的参数量仅为 $5 \times 5$ 的 36%、$7 \times 7$ 的 18%。两层 $3 \times 3$ 卷积感受野等价于一个 $5 \times 5$ 卷积，但参数量减少 28%，且多一层非线性变换。

4. **$1 \times 1$ 卷积降维**：在 Inception 模块中，$1 \times 1$ 卷积降维可将参数量减少 80-90%，是 GoogLeNet 低参数量的关键。

5. **网络对比**：从 AlexNet 到 GoogLeNet，错误率从 15.3% 降至 6.7%，而参数量从 62M 降至 7M。这体现了架构设计优化的巨大价值。

## 历史意义与后续影响

### VGG 的影响

1. **深度优先设计**：VGG 证明了"更深"是提升精度的有效途径。后续几乎所有网络都采用了更深的架构（ResNet-152、DenseNet-161 等）。

2. **$3 \times 3$ 卷积核**：VGG 确立了 $3 \times 3$ 卷积核作为 CNN 的基本构建单元。后续网络几乎全部使用 $3 \times 3$ 卷积（或等价的结构），大卷积核逐渐被淘汰。

3. **简单规律的结构**：VGG 的高度规律性（Block 重复结构）使其易于实现和复现，成为后续研究的基准模型。即使在今天，VGG-16 的特征提取仍常用于图像风格迁移等任务。

### GoogLeNet 的影响

1. **Inception 模块**：Inception 思想影响了后续大量网络设计。GoogLeNet 之后，Inception v2/v3（Batch Normalization + 分解大卷积）、v4 等变体相继出现。

2. **$1 \times 1$ 卷积**：GoogLeNet 确立了 $1 \times 1$ 卷积在 CNN 中的核心地位——通道降维、跨通道信息融合、增加非线性。这一思想被 ResNet（瓶颈结构）、EfficientNet 等后续网络广泛采用。

3. **全局平均池化**：GoogLeNet 用全局平均池化替代全连接层，减少了参数量和过拟合风险。这一设计成为现代 CNN 的标准做法。

4. **辅助分类器**：虽然辅助分类器在后续网络中被逐渐抛弃（被 Batch Normalization 等更有效的技术替代），但这一思想启发了后来的"跳跃连接"和"多尺度监督"技术。

## 本章小结

本章介绍了 2014 年两篇重要的 CNN 改进工作：

**VGGNet**：
- 核心思想：增加网络深度可以提升精度
- 架构特点：全部使用 $3 \times 3$ 卷积核，通道数逐层递增
- 参数量：~1.38 亿（全连接层占 89%）
- Top-5 错误率：7.3%
- 贡献：确立了 $3 \times 3$ 卷积核和深度优先设计

**GoogLeNet**：
- 核心思想：同时使用多尺度卷积核，让网络学习最优尺度
- 架构特点：Inception 模块，$1 \times 1$ 卷积降维，全局平均池化
- 参数量：~7M（VGG 的 1/20）
- Top-5 错误率：6.7%
- 贡献：Inception 模块、$1 \times 1$ 卷积降维、全局平均池化

**深度与宽度的权衡**：VGG 选择了增加深度，GoogLeNet 选择了增加宽度（通过并行多尺度分支）。两者都取得了显著进步，但 GoogLeNet 以更低的参数量实现了更高的精度。

下一章将介绍 ResNet——通过残差连接解决深层网络的退化问题，将网络深度进一步推进到 152 层甚至更深。

## 练习题

1. 推导 $3 \times 3$ 卷积核级联的感受野与参数量。证明两层 $3 \times 3$ 卷积等价于一个 $5 \times 5$ 卷积的感受野，且参数量减少 28%（输入通道等于输出通道）。
    <details>
    <summary>参考答案</summary>

    **$3 \times 3$ 卷积核级联分析**：

    **一、感受野推导**

    感受野递推公式：
    $$R_l = R_{l-1} + (k_l - 1) \times \prod_{i=1}^{l-1} s_i$$

    两层 $3 \times 3$ 卷积，步长均为 1：

    $$R_1 = 1 + (3 - 1) \times 1 = 3$$

    $$R_2 = 3 + (3 - 1) \times 1 = 5$$

    两层 $3 \times 3$ 卷积的级联感受野为 $5 \times 5$，等价于一个 $5 \times 5$ 卷积。

    **二、参数量推导**

    设输入通道数 = 输出通道数 = $C$。

    **一个 $5 \times 5$ 卷积**：

    $$\text{参数量}_{5\times5} = 5 \times 5 \times C \times C + C = 25C^2 + C$$

    **两个 $3 \times 3$ 卷积**：

    第一个 $3 \times 3$ 卷积（输入 $C$ 通道，输出 $C$ 通道）：

    $$\text{参数量}_1 = 3 \times 3 \times C \times C + C = 9C^2 + C$$

    第二个 $3 \times 3$ 卷积（输入 $C$ 通道，输出 $C$ 通道）：

    $$\text{参数量}_2 = 3 \times 3 \times C \times C + C = 9C^2 + C$$

    总参数量：

    $$\text{参数量}_{3\times3 \times 2} = 18C^2 + 2C$$

    **对比**：

    参数量之比：

    $$\frac{\text{参数量}_{3\times3 \times 2}}{\text{参数量}_{5\times5}} = \frac{18C^2 + 2C}{25C^2 + C} \approx \frac{18}{25} = 0.72$$

    即两层 $3 \times 3$ 卷积的参数量约为一个 $5 \times 5$ 卷积的 **72%**，减少约 **28%**。

    **数值验证**（$C = 256$）：

    - $5 \times 5$ 卷积：$25 \times 256^2 + 256 = 1,638,656$
    - 两个 $3 \times 3$ 卷积：$18 \times 256^2 + 2 \times 256 = 1,179,904$
    - 减少：$(1,638,656 - 1,179,904) / 1,638,656 = 28.0\%$

    **三、非线性能力对比**

    一个 $5 \times 5$ 卷积包含 1 个 ReLU 激活：
    $$f(x) = \text{ReLU}(W_5 * x + b_5)$$

    两个 $3 \times 3$ 卷积包含 2 个 ReLU 激活：
    $$f(x) = \text{ReLU}(W_3 * \text{ReLU}(W_3 * x + b_{3a}) + b_{3b})$$

    两层 ReLU 激活使得两层 $3 \times 3$ 卷积能够拟合更复杂的函数。从函数空间的角度，两层 $3 \times 3$ 卷积的函数空间包含一个 $5 \times 5$ 卷积的函数空间（严格更大）。

    **四、结论**

    两层 $3 \times 3$ 卷积相比一个 $5 \times 5$ 卷积的优势：

    | 对比项 | $5 \times 5$ | $3 \times 3 \times 2$ | 优势 |
    |:------|:------------|:---------------------|:----|
    | 感受野 | $5 \times 5$ | $5 \times 5$ | 相同 |
    | 参数量 | $25C^2 + C$ | $18C^2 + 2C$ | 减少 28% |
    | ReLU 数量 | 1 | 2 | 更强非线性 |
    | 网络深度 | 1 层 | 2 层 | 更深 |
    </details>

2. 分析 Inception 模块中 $1 \times 1$ 卷积的降维原理。计算一个具体 Inception 模块使用和不使用 $1 \times 1$ 降维的参数量差异。
    <details>
    <summary>参考答案</summary>

    **Inception 模块中 $1 \times 1$ 卷积降维分析**：

    **一、Inception 模块结构**

    考虑一个典型的 Inception 模块，输入通道 $C_{in} = 192$，四路输出：

    | 分支 | 配置 | 输出通道 |
    |:----|:-----|:--------|
    | 1×1 卷积分支 | $1 \times 1$ | $C_1 = 64$ |
    | 3×3 卷积分支 | $1 \times 1 \to 3 \times 3$ | $C_3 = 128$ |
    | 5×5 卷积分支 | $1 \times 1 \to 5 \times 5$ | $C_5 = 32$ |
    | 池化分支 | Pool → $1 \times 1$ | $C_p = 32$ |

    总输出通道数：$64 + 128 + 32 + 32 = 256$

    **二、不使用 $1 \times 1$ 降维**

    如果直接应用 $3 \times 3$ 和 $5 \times 5$ 卷积（不做降维）：

    | 分支 | 计算 | 参数量 |
    |:----|:----|:------|
    | 1×1 卷积分支 | $1 \times 1 \times 192 \times 64 + 64$ | 12,352 |
    | 3×3 卷积分支 | $3 \times 3 \times 192 \times 128 + 128$ | 221,312 |
    | 5×5 卷积分支 | $5 \times 5 \times 192 \times 32 + 32$ | 153,632 |
    | 池化分支 | $1 \times 1 \times 192 \times 32 + 32$ | 6,176 |
    | **总计** | | **393,472** |

    **三、使用 $1 \times 1$ 降维**

    对 $3 \times 3$ 和 $5 \times 5$ 卷积分支，先用 $1 \times 1$ 卷积降维：

    假设 $3 \times 3$ 分支的 $1 \times 1$ 降维到 96 通道，$5 \times 5$ 分支的 $1 \times 1$ 降维到 16 通道：

    | 分支 | 计算 | 参数量 |
    |:----|:----|:------|
    | 1×1 卷积分支 | $1 \times 1 \times 192 \times 64 + 64$ | 12,352 |
    | 3×3 分支 ($1\times1\to3\times3$) | $(1\times1\times192\times96+96) + (3\times3\times96\times128+128)$ | 18,528 + 110,656 = 129,184 |
    | 5×5 分支 ($1\times1\to5\times5$) | $(1\times1\times192\times16+16) + (5\times5\times16\times32+32)$ | 3,088 + 12,832 = 15,920 |
    | 池化分支 | $1 \times 1 \times 192 \times 32 + 32$ | 6,176 |
    | **总计** | | **163,632** |

    **四、对比**

    | 方案 | 参数量 | 输出尺寸 |
    |:----|:------|:--------|
    | 无降维 | 393,472 | $H \times W \times 256$ |
    | 有降维 | 163,632 | $H \times W \times 256$ |
    | **节省** | **58.4%** | 相同 |

    **五、$1 \times 1$ 卷积降维的数学原理**

    设输入 $X \in \mathbb{R}^{H \times W \times C_{in}}$，$1 \times 1$ 卷积将其投影到 $C_{bottleneck}$ 维：

    $$Y = X \cdot W + b$$

    其中 $W \in \mathbb{R}^{C_{in} \times C_{bottleneck}}$，$b \in \mathbb{R}^{C_{bottleneck}}$。

    这相当于在每个空间位置 $(h, w)$ 上，对 $C_{in}$ 维向量做线性变换：

    $$y_{h,w} = W^T x_{h,w} + b$$

    其中 $x_{h,w} \in \mathbb{R}^{C_{in}}$，$y_{h,w} \in \mathbb{R}^{C_{bottleneck}}$。

    **降维的有效性基于**：同空间位置的通道信息通常高度冗余。192 维的特征向量中，许多通道包含相似信息（如多个通道检测相似特征）。$1 \times 1$ 卷积学习一个 $C_{in} \to C_{bottleneck}$ 的线性投影，保留了最重要的信息，丢弃了冗余。

    **六、总结**

    $1 \times 1$ 卷积降维在 Inception 模块中的核心作用：

    1. **参数量减少**：本例中从 393K 降至 164K，节省 58.4%
    2. **计算量减少**：后续 $3 \times 3$ 和 $5 \times 5$ 卷积的输入通道减少，计算量同比例减少
    3. **增加非线性**：降维后加 ReLU，增加了非线性变换
    4. **跨通道融合**：在降维过程中融合不同通道的信息
    </details>

3. 假设输入 $28 \times 28 \times 256$，分别用 VGG 风格的连续 $3 \times 3$ 卷积和 GoogLeNet 风格的 Inception 模块处理，输出 $28 \times 28 \times 512$，计算并对比两种方案的参数量、计算量和感受野。
    <details>
    <summary>参考答案</summary>

    **VGG vs GoogLeNet 方案对比（输入 $28 \times 28 \times 256$ → 输出 $28 \times 28 \times 512$）**：

    **一、VGG 风格方案**

    VGG 风格使用连续 $3 \times 3$ 卷积，步长 1，填充 1（空间尺寸不变），通道数逐层增加。

    **设计**：3 层 $3 \times 3$ 卷积，通道数递增：

    ```
    256 → 384 (3×3) → ReLU → 384 → 512 (3×3) → ReLU
    ```

    但通道数从 256 直接到 512 需要合理的过渡。实际设计中，通常分阶段递增：

    **方案 A**：3 层递增（256 → 384 → 384 → 512）

    | 层 | 输入通道 | 输出通道 | 参数量 |
    |:--|:--------|:--------|:------|
    | Conv1 | 256 | 384 | $384 \times 3 \times 3 \times 256 + 384 = 885,120$ |
    | Conv2 | 384 | 384 | $384 \times 3 \times 3 \times 384 + 384 = 1,327,488$ |
    | Conv3 | 384 | 512 | $512 \times 3 \times 3 \times 384 + 512 = 1,769,984$ |
    | **总计** | | | **3,982,592** |

    **感受野**：3 层 $3 \times 3$ 卷积级联

    $$R = 3 + (3 - 1) \times 1 + (3 - 1) \times 1 = 7$$

    感受野：$7 \times 7$

    **计算量**：

    卷积层 FLOPs 标准公式：$\text{FLOPs} = H \times W \times C_{out} \times (k \times k \times C_{in} \times 2)$

    **VGG 三层块 FLOPs**：

    - Conv1 (256→384, 3×3)：$28 \times 28 \times 384 \times (3 \times 3 \times 256 \times 2) = 109,663,872 \approx 109.7\text{M}$
    - Conv2 (384→384, 3×3)：$28 \times 28 \times 384 \times (3 \times 3 \times 384 \times 2) = 164,495,808 \approx 164.5\text{M}$
    - Conv3 (384→512, 3×3)：$28 \times 28 \times 512 \times (3 \times 3 \times 384 \times 2) = 219,327,744 \approx 219.3\text{M}$

    VGG 总 FLOPs：$109.7\text{M} + 164.5\text{M} + 219.3\text{M} = 493.5\text{M}$

    **Inception 模块 FLOPs**：

    - 1×1 分支 (256→128)：$28 \times 28 \times 128 \times (1 \times 1 \times 256 \times 2) = 50,331,648 \approx 50.3\text{M}$
    - 3×3 分支：先 1×1 降维 (256→128)，再 3×3 (128→256)
      - 1×1：$28 \times 28 \times 128 \times (1 \times 1 \times 256 \times 2) = 50.3\text{M}$
      - 3×3：$28 \times 28 \times 256 \times (3 \times 3 \times 128 \times 2) = 105,646,080 \approx 105.6\text{M}$
      - 小计：$50.3\text{M} + 105.6\text{M} = 155.9\text{M}$
    - 5×5 分支：先 1×1 降维 (256→32)，再 5×5 (32→96)
      - 1×1：$28 \times 28 \times 32 \times (1 \times 1 \times 256 \times 2) = 12,582,912 \approx 12.6\text{M}$
      - 5×5：$28 \times 28 \times 96 \times (5 \times 5 \times 32 \times 2) = 42,467,328 \approx 42.5\text{M}$
      - 小计：$12.6\text{M} + 42.5\text{M} = 55.1\text{M}$
    - Pool 分支：$28 \times 28 \times 32 \times (1 \times 1 \times 256 \times 2) = 12.6\text{M}$

    Inception 总 FLOPs：$50.3\text{M} + 155.9\text{M} + 55.1\text{M} + 12.6\text{M} = 273.9\text{M}$

    **二、GoogLeNet 风格 Inception 模块**

    设计一个 Inception 模块，输入 $28 \times 28 \times 256$，输出 $28 \times 28 \times 512$：

    ```
    输入 (28×28×256)
      │
      ├→ [1×1→128] → 输出128通道
      ├→ [1×1→96] → [3×3→192] → 输出192通道
      ├→ [1×1→16] → [5×5→48] → 输出48通道
      └→ [Pool] → [1×1→64] → 输出64通道
        总计: 128 + 192 + 48 + 64 = 432（调整到512）
    ```

    调整使总输出为 512：

    | 分支 | 降维通道 | 主卷积输出 | 参数量 |
    |:----|:--------|:----------|:------|
    | 1×1 | - | 128 | $1\times1\times256\times128+128 = 32,896$ |
    | 3×3 | 96 | 192 | $(1\times1\times256\times96+96) + (3\times3\times96\times192+192) = 24,672 + 166,080 = 190,752$ |
    | 5×5 | 16 | 48 | $(1\times1\times256\times16+16) + (5\times5\times16\times48+48) = 4,112 + 19,248 = 23,360$ |
    | Pool | - | 64 | $1\times1\times256\times64+64 = 16,448$ |
    | **总计** | | $128+192+48+64=432$ | **263,456** |

    输出通道 432 与目标 512 有差距。调整为 512：

    | 分支 | 降维 | 主输出 | 参数量 |
    |:----|:----|:------|:------|
    | 1×1 | - | 128 | 32,896 |
    | 3×3 | 128 | 256 | $(1\times1\times256\times128+128) + (3\times3\times128\times256+256) = 32,896 + 295,168 = 328,064$ |
    | 5×5 | 32 | 96 | $(1\times1\times256\times32+32) + (5\times5\times32\times96+96) = 8,224 + 76,896 = 85,120$ |
    | Pool | - | 32 | $1\times1\times256\times32+32 = 8,224$ |
    | **总计** | | $128+256+96+32=512$ | **454,304** |

    **感受野**：Inception 模块中最大感受野来自 $5 \times 5$ 分支 = $5 \times 5$。

    **计算量**：已在上方详细计算，Inception 总 FLOPs = $273.9\text{M}$

    **三、对比总结**

    | 对比项 | VGG (3层) | Inception | 优势 |
    |:------|:---------|:---------|:----|
    | 参数量 | 3,982,592 | 454,304 | **Inception 减少 88.6%** |
    | 计算量 (FLOPs) | 493.5M | 273.9M | **Inception 减少 44.5%** |
    | 感受野 | $7 \times 7$ | $5 \times 5$ | **VGG 更大** |
    | 网络深度 | 3 层串行 | 4 路并行 | **VGG 更深** |
    | 输出通道 | 512 | 512 | 相同 |

    **四、分析**

    **1. 参数量差异**：

    VGG 方案参数量是 Inception 的约 **8.8 倍**。主要差异在于：

    - VGG 的 Conv3（384→512）：$512 \times 3 \times 3 \times 384 + 512 = 1,769,984$
    - Inception 最大的 3×3 分支（128→256）：$3 \times 3 \times 128 \times 256 + 256 = 295,168$

    Inception 通过 $1 \times 1$ 降维将通道数压缩后再做大卷积，参数量大幅减少。

    **2. 计算量差异**：

    Inception 计算量比 VGG 少约 **44.5%**（493.5M vs 273.9M FLOPs）。主要节省来自：
    - 3×3 分支通过 1×1 降维到 128 通道，3×3 卷积的输入通道从 384 降至 128
    - 5×5 分支通过 1×1 降维到 32 通道，5×5 卷积的输入通道从 384 降至 32
    - 1×1 卷积分支直接替换部分 3×3 卷积

    **3. 感受野差异**：

    VGG 的 $7 \times 7$ 感受野大于 Inception 的 $5 \times 5$。但 Inception 的多尺度设计（同时包含 $1 \times 1$、$3 \times 3$、$5 \times 5$）使得网络能够同时捕捉不同尺度的特征，这在一定程度上补偿了感受野的差异。

    **4. 适用场景**：

    - **VGG 风格**：适合需要大感受野的任务（如语义分割），结构简单易于训练
    - **Inception 风格**：适合参数量和计算量受限的场景（如移动设备部署），多尺度特征融合效果好

    **4. 结论**：

    在输入 $28 \times 28 \times 256$、输出 $28 \times 28 \times 512$ 的条件下：
    - Inception 参数量减少 88.6%，计算量减少 44.5%
    - VGG 感受野更大（$7 \times 7$ vs $5 \times 5$）
    - Inception 的多尺度设计提供更丰富的特征表示
    - 选择取决于具体需求：精度优先（VGG）还是效率优先（Inception）
    </details>
