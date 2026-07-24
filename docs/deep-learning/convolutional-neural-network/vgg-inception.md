# VGG 与 GoogLeNet

2012 年 AlexNet 的成功验证了深度卷积神经网络在大规模视觉任务上的有效性，作为开山之作，AlexNet 的架构设计难免有些粗糙，如 8 层网络中使用大卷积核（$11 \times 11$、$5 \times 5$）粗暴降采样，卷积层与全连接层参数量不平衡（全连接层占总参数的 94%）等问题。所有人都好奇深度神经网络的潜力上限在哪里，性能提升的源泉是什么。是网络更深、更宽，还是卷积核更小？是训练技巧改进了，还是数据增强更好了？如果不搞清楚这些问题，后续的架构设计就如同盲人摸象，每次改进都可能只是在某个维度上偶然碰对了方向。

2014 年，两篇里程碑式的研究工作从不同方向部分回答了这些问题。牛津大学视觉几何组（Visual Geometry Group，VGG）在论文《[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)》中证明，将网络深度从 8 层推进到 16-19 层可以显著提升精度，后来据此实现了著名的 **VGGNet**。不久之后，Google 的克里斯蒂安·塞格迪（Christian Szegedy）在论文《[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)》中提出了 **GoogLeNet**（也称 Inception-v1），通过精心设计的 Inception 模块实现多尺度特征融合，仅用 700 万参数就达到了比 VGG 更低的错误率。

这两项工作代表了 CNN 架构设计的两个核心探索方向：VGG 选择了深度优先，通过堆叠小卷积核增加层数，证明更深的网络能学到更抽象的特征；GoogLeNet 选择了宽度优先，通过并行多尺度分支增加有效宽度，让网络自己决定哪些尺度的特征最有价值。网络应该更深还是更宽、参数效率如何提升，这些探索开启了现代 CNN 架构设计的黄金时代。

## 深度探索：VGGNet

VGG 组的研究者决定用最朴素的实验方法来回答"深度学习是否越深越有效"这个问题，他们尝试固定模型的其他因素，只改变深度，看看会发生什么。实验的核心假设是在保持卷积核尺寸、宽度（通道数）等其他超参数不变的情况下，增加网络深度应该能提升精度。这个假设背后的逻辑很明确：更深的网络有更多的非线性变换层，每层都能学到比前一层更抽象的特征。从像素到边缘，从边缘到纹理，从纹理到部件，从部件到物体，这种层次化的特征提取，正是深度学习区别于传统机器学习的核心优势。

从实验结果可以看到，层数越多，错误率越低，VGG-19 的 Top-5 错误率 7.0% 比 VGG-11 的 23.1% 降低了约 70%，但参数量增长相对有限（133M→144M），这说明深度增加带来的精度提升是"划算的"，深度是提升 CNN 性能的有效途径。

| 配置 | 带权重层数 | Block 结构 | 参数量 | Top-5 错误率 |
|:----|:----------|:----------|:------|:------------|
| VGG-A | 11 层 | 2-2-2-2-2 | ~133M | 23.1% |
| VGG-B | 13 层 | 2-2-2-2-2 | ~132M | 22.4% |
| VGG-D（VGG-16） | 16 层 | 2-2-3-3-3 | ~138M | 7.3% |
| VGG-E （VGG-19） | 19 层 | 2-2-4-4-4 | ~144M | 7.0% |

基于"深度有效"的假设，VGG 采用了一个极度简洁、高度规律的网络结构。以最常见的 VGG-16 为例：全部使用 $3 \times 3$ 小卷积核，每个 Block 后接 $2 \times 2$ 最大池化层；通道数每经过一个 Block 翻倍（64→128→256→512），空间尺寸每经过一次池化减半（$224 \times 224$ → $7 \times 7$）。这种复制粘贴式的模块化设计谈不上深奥，却蕴含着精妙的工程智慧，其结构可以用以下架构图表示：

```nn-arch width=900
name: VGG-16 流程架构图（13 个卷积层 + 3 个全连接层）
layout: horizontal

sections:
  - name: 特征提取器（5 个Block，13 个卷积层）
    layers: ["Input", "block1", "block2", "block3", "block4", "block5"]
    row_label: "Flatten: 25088"
  - name: 分类器（3 个全连接层）
    layers: ["FC6", "FC7", "FC8", "Output"]

layers:
  - {name: "Input", type: input, size: "224×224×3"}
  - {id: block1, name: "2×Conv3×3", type: conv, kernel: 3, channels: 64, out: "112×112×64", pool: true}
  - {id: block2, name: "2×Conv3×3", type: conv, kernel: 3, channels: 128, out: "56×56×128", pool: true}
  - {id: block3, name: "3×Conv3×3", type: conv, kernel: 3, channels: 256, out: "28×28×256", pool: true}
  - {id: block4, name: "3×Conv3×3", type: conv, kernel: 3, channels: 512, out: "14×14×512", pool: true}
  - {id: block5, name: "3×Conv3×3", type: conv, kernel: 3, channels: 512, out: "7×7×512", pool: true}
  - {name: "FC6", type: fc, size: "4096 维", act: ReLU, dropout: true}
  - {name: "FC7", type: fc, size: "4096 维", act: ReLU, dropout: true}
  - {name: "FC8", type: fc, size: "1000 维"}
  - {name: "Output", type: output, size: 1000, act: Softmax}
```

*图：VGG-16 流程架构图*

VGG 网络架构上最有特色的设计决策是全部使用 $3 \times 3$ 小卷积核，完全抛弃了 AlexNet 中的 $11 \times 11$ 和 $5 \times 5$ 大卷积核。这背后有三个考量：

1. **感受野等价，参数量更少**：直觉上，感受野越大，能"看到"的输入信息越多，越能捕捉全局特征。一个 $5 \times 5$ 卷积核的感受野是 $5 \times 5$，两层 $3 \times 3$ 卷积的堆叠，感受野也是 $5 \times 5$，但是两层 $3 \times 3$ 卷积的参数量仅为一个 $5 \times 5$ 卷积的 72%。同理，三个 $3 \times 3$ 卷积（感受野 $7 \times 7$）的参数量仅为一个 $7 \times 7$ 卷积的 55%。

2. **更强的非线性表达能力**：参数量减少只是表面优势，更深层的好处在于非线性变换次数的增加。两层 $3 \times 3$ 卷积包含两个 ReLU 激活函数，意味着两次非线性变换。从数学角度看，两层 $3 \times 3$ 卷积能表示的函数空间，严格包含了一个 $5 \times 5$ 卷积能表示的函数空间——堆叠小卷积核能拟合更复杂的函数关系。

3. **自然增加网络深度**：使用多个小卷积核替代大卷积核，自然而然地增加了网络深度。更深意味着更多的特征层次，从低级特征逐步抽象到高级语义，这契合了层次化特征学习的核心思想。

VGG 用 $3 \times 3$ 小卷积核替代大卷积核的设计，后来被许多现代 CNN 借鉴，$3 \times 3$ 卷积核从此成为 CNN 架构的"标准件"。不过，VGG 也继承了 AlexNet 的设计缺陷，甚至愈发严重：参数量巨大且分布极不均衡，全连接层几乎成了参数的"黑洞"。通过[卷积网络基础](cnn-basics.md)中的参数量公式（$\text{参数量}_{conv} = k \times k \times C_{in} \times C_{out} + C_{out}$），可以计算出每一层卷积层的参数量：

| 层 | 输入通道 | 输出通道 | 参数量计算 | 参数量 |
|:--|:--------|:--------|:----------|:------|
| Block1：Conv1 | 3 | 64 | $3 \times 3 \times 3 \times 64 + 64$ | 1,792 |
| Block1：Conv2 | 64 | 64 | $3 \times 3 \times 64 \times 64 + 64$ | 36,928 |
| Block2：Conv1 | 64 | 128 | $3 \times 3 \times 64 \times 128 + 128$ | 73,856 |
| Block2：Conv2 | 128 | 128 | $3 \times 3 \times 128 \times 128 + 128$ | 147,584 |
| Block3：Conv1 | 128 | 256 | $3 \times 3 \times 128 \times 256 + 256$ | 295,168 |
| Block3：Conv2 | 256 | 256 | $3 \times 3 \times 256 \times 256 + 256$ | 590,080 |
| Block3：Conv3 | 256 | 256 | $3 \times 3 \times 256 \times 256 + 256$ | 590,080 |
| Block4：Conv1 | 256 | 512 | $3 \times 3 \times 256 \times 512 + 512$ | 1,180,160 |
| Block4：Conv2、Conv3 | 512 | 512 | $(3 \times 3 \times 512 \times 512 + 512) \times 2$ | 4,719,616 |
| Block5：Conv1、Conv2、Conv3 | 512 | 512 | $(3 \times 3 \times 512 \times 512 + 512) \times 3$ | 7,079,424 |

卷积层部分的总参数量约为 1470 万。卷积层最终输出尺寸为 $7 \times 7 \times 512 = 25,088$ 维，这也是全连接层的输入，全连接层部分的总参数量约为 1.236 亿，具体如下：

| 层 | 输入维度 | 输出维度 | 参数量计算 | 参数量 |
|:--|:--------|:--------|:----------|:------|
| FC6 | 25,088 | 4,096 | $25,088 \times 4,096 + 4,096$ | 102,764,544 |
| FC7 | 4,096 | 4,096 | $4,096 \times 4,096 + 4,096$ | 16,781,312 |
| FC8 | 4,096 | 1,000 | $4,096 \times 1,000 + 1,000$ | 4,097,000 |

对比 AlexNet，VGG 成功让 Top-5 错误率减半（15.3% 到 7.3%），却也付出了参数量翻倍的成本，且约 90% 参数集中在全连接层，不仅占用大量内存，还容易导致过拟合。解决这个问题就交到了 GoogLeNet 手上。

## 宽度探索：GoogLeNet

Google 的研究团队从另一个角度思考深度网络的发展：如果不能让网络更深，能不能让它更强大？由此催生了 Inception 模块，这是一个用多尺度并行结构替代串行堆叠的创新设计。

在实际视觉任务中，图像中的目标大小差异很大。一张照片里可能同时包含一只小鸟（占几十个像素）和一座大楼（占几百个像素）。如果只用固定尺寸的卷积核（譬如 $3 \times 3$），小目标可能淹没在卷积核内，大目标则感受野不足。不同大小的卷积核能够捕捉不同的特征，$1 \times 1$ 能捕捉点特征，$3 \times 3$ 能捕捉局部纹理，$5 \times 5$ 能捕捉更大范围的结构。与其人为选择某一种，不如让所有尺寸并行工作，让网络自己决定哪些尺度的特征最有用。多尺度并行提取、特征融合决策正是 Inception 模块的核心思想，它的名字"Inception"来源于电影《盗梦空间》（Inception）的一句台词"We need to go deeper"，寓意"我们要去往更深处"，但这里的"深"不是层数的堆叠，而是特征空间的深度探索。

Inception 模块是 GoogLeNet 的组成部件，每个 Inception 的基本结构都是四路并行、末端拼接的网络，每条路径负责提取不同尺度的特征，具体如下图所示：

```nn-arch width=680
name: Inception 模块（4 路并行网络）
layout: vertical

layers:
  - {id: input, name: "输入特征图", type: input, size: "H×W×C"}

blocks:
  - name: Inception
    type: parallel
    branches:
      # 1x1 conv 分支
      - {id: branch_1x1, name: "1×1卷积",type: conv, kernel: 1, channels: 64, act: ReLU}
      # 3x3 conv 分支 (先 1x1 reduce)
      - [
          {id: branch_3x3_reduce, name: "1×1降维", type: conv, kernel: 1, channels: 96, act: ReLU},
          {id: branch_3x3, name: "3×3卷积", type: conv, kernel: 3, pad: 1, channels: 128, act: ReLU}
        ]
      # 5x5 conv 分支 (先 1x1 reduce)
      - [
          {id: branch_5x5_reduce, name: "1×1降维", type: conv, kernel: 1, channels: 16, act: ReLU},
          {id: branch_5x5, name: "5×5卷积", type: conv, kernel: 5, pad: 2, channels: 32, act: ReLU}
        ]
      # pool 分支
      - [
          {id: branch_pool, name: "3×3最大池化", type: pool, kernel: 3, stride: 1, pad: 1},
          {id: branch_pool_proj, name: "1×1卷积", type: conv, kernel: 1, channels: 32, act: ReLU}
        ]
    merge: concat

layers_after_blocks:
  - {id: concat, name: "通道拼接", type: note, label: "Concatenate" }  
  - {id: output, name: "输入特征图", type: output, size: "H×W×C'"}
```
*图：Inception 模块*

从这个结构图可以清晰地看到 Inception 模块包含四条并行的路径分支，每条分支都有独立的设计意图：

- **$1 \times 1$ 卷积分支**：直接提取点状特征，无空间扩展，感受野最小。
- **$3 \times 3$ 卷积分支**：先用 $1 \times 1$ 降维（减少通道数），再用 $3 \times 3$ 卷积提取中等尺度特征。
- **$5 \times 5$ 卷积分支**：同样先降维，再用 $5 \times 5$ 卷积提取大尺度特征。
- **池化分支**：最大池化保留显著特征，后接 $1 \times 1$ 卷积调整通道数。

四路输出在通道维度上拼接（Concatenate，即输出的通道数 C' 是四路通道数之和），形成最终的输出特征图。输出的空间尺寸保持不变（通过适当的填充），只有通道数改变。这意味着后续模块还可以继续以相同的方式堆叠。

降低参数量的贡献来自于 Inception 模块的 $1 \times 1$ 卷积，它构造了一种"瓶颈"（Bottleneck）结构。在[ CNN 基础](cnn-basics.md#cnn-架构设计原则)中曾提到$1 \times 1$ 卷积具有三个作用：跨通道信息融合、增加非线性、通道降维。在 Inception 模块中，降维作用是关键，解决了多尺度并行结构带来的参数爆炸问题。

举个具体例子，假设 Inception 模块的输入是 $28 \times 28 \times 192$（高度 28，宽度 28，通道 192），卷积核尺寸为 $5 \times 5$，输出为 $28 \times 28 \times 32$（输出通道数 32，空间尺寸保持 28×28 不变）。如果不降维，直接对 192 通道输入做 $5 \times 5$ 卷积则$\text{参数量} = 5 \times 5 \times 192 \times 32 + 32 = 153,632$；如果先用 $1 \times 1$ 卷积将通道从 192 降到 16（称为瓶颈通道数），再对 16 通道做 $5 \times 5$ 卷积，则$\text{参数量}_{降维} = (1 \times 1 \times 192 \times 16 + 16) + (5 \times 5 \times 16 \times 32 + 32) = 3,088 + 12,832 = 15,920$。参数量从 153,632 降至 15,920，减少了约 90%，而输出尺寸保持完全相同，都是 $28 \times 28 \times 32$。

$1 \times 1$ 卷积在每个空间位置上对通道维度做一个线性组合，相当于一个信息压缩器，将 192 维的通道信息压缩到 16 维。由于同一空间位置的通道信息通常是高度冗余的，多个通道可能检测相似的特征，因此这种压缩不会丢失关键信息，却能大幅减少后续计算的成本。

理解了 Inception 模块的设计原理，再来看它如何组装成一个完整的分类网络。GoogLeNet 的结构比 AlexNet 和 VGG 的串行网络结构复杂很多，由 9 个 Inception 模块按顺序堆叠而成，中间穿插了若干常规卷积层和池化层。整体架构可以分为三个部分：初始特征提取、Inception 堆叠、分类输出，如下所示：

```nn-arch width=900
name: GoogLeNet 网络架构
layout: horizontal

sections:
  - name: 初始特征提取
    layers: [input, conv1, pool1, conv2, pool2]
  - name: Inception 堆叠
    layers: [Inception_3a, Inception_3b, pool3, Inception_4a, Inception_4b, Inception_4c, Inception_4d, Inception_4e, pool4, Inception_5a, Inception_5b]
  - name: 分类输出
    layers: [pool5, fc1, fc2, output]

layers:
  - {id: input, name: Input, type: input, size: "224x224x3"}
  - {id: conv1, name: Conv1, type: conv, kernel: 7, stride: 2, channels: 64, out: "112x112x64", act: ReLU}
  - {id: pool1, name: Pool1, type: pool, kernel: 3, stride: 2, out: "56x56x64"}
  - {id: conv2, name: Conv2, type: conv, kernel: 3, channels: 192, act: ReLU}
  - {id: pool2, name: Pool2, type: pool, kernel: 3, stride: 2, out: "28x28x192"}

blocks:
  - name: Inception_3a
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc3a_1x1, name: "1x1", type: conv, kernel: 1, channels: 64, act: ReLU}
      - [{id: inc3a_3x3r, name: "reduce", type: conv, kernel: 1, channels: 96, act: ReLU},
         {id: inc3a_3x3, name: "3x3", type: conv, kernel: 3, channels: 128, act: ReLU}]
      - [{id: inc3a_5x5r, name: "reduce", type: conv, kernel: 1, channels: 16, act: ReLU},
         {id: inc3a_5x5, name: "5x5", type: conv, kernel: 5, channels: 32, act: ReLU}]
      - [{id: inc3a_pool, name: "pool", type: pool, kernel: 3},
         {id: inc3a_proj, name: "proj", type: conv, kernel: 1, channels: 32, act: ReLU}]
    merge: concat

  - name: Inception_3b
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc3b_1x1, name: "1x1", type: conv, kernel: 1, channels: 128, act: ReLU}
      - [{id: inc3b_3x3r, name: "reduce", type: conv, kernel: 1, channels: 128, act: ReLU},
         {id: inc3b_3x3, name: "3x3", type: conv, kernel: 3, channels: 192, act: ReLU}]
      - [{id: inc3b_5x5r, name: "reduce", type: conv, kernel: 1, channels: 32, act: ReLU},
         {id: inc3b_5x5, name: "5x5", type: conv, kernel: 5, channels: 96, act: ReLU}]
      - [{id: inc3b_pool, name: "pool", type: pool, kernel: 3},
         {id: inc3b_proj, name: "proj", type: conv, kernel: 1, channels: 64, act: ReLU}]
    merge: concat

  - {id: pool3, name: Pool3, type: pool, kernel: 3, stride: 2}

  - name: Inception_4a
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc4a_1x1, name: "1x1", type: conv, kernel: 1, channels: 192, act: ReLU}
      - [{id: inc4a_3x3r, name: "reduce", type: conv, kernel: 1, channels: 96, act: ReLU},
         {id: inc4a_3x3, name: "3x3", type: conv, kernel: 3, channels: 208, act: ReLU}]
      - [{id: inc4a_5x5r, name: "reduce", type: conv, kernel: 1, channels: 16, act: ReLU},
         {id: inc4a_5x5, name: "5x5", type: conv, kernel: 5, channels: 48, act: ReLU}]
      - [{id: inc4a_pool, name: "pool", type: pool, kernel: 3},
         {id: inc4a_proj, name: "proj", type: conv, kernel: 1, channels: 64, act: ReLU}]
    merge: concat

  - name: Inception_4b
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc4b_1x1, name: "1x1", type: conv, kernel: 1, channels: 160, act: ReLU}
      - [{id: inc4b_3x3r, name: "reduce", type: conv, kernel: 1, channels: 112, act: ReLU},
         {id: inc4b_3x3, name: "3x3", type: conv, kernel: 3, channels: 224, act: ReLU}]
      - [{id: inc4b_5x5r, name: "reduce", type: conv, kernel: 1, channels: 24, act: ReLU},
         {id: inc4b_5x5, name: "5x5", type: conv, kernel: 5, channels: 64, act: ReLU}]
      - [{id: inc4b_pool, name: "pool", type: pool, kernel: 3},
         {id: inc4b_proj, name: "proj", type: conv, kernel: 1, channels: 64, act: ReLU}]
    merge: concat

  - name: Inception_4c
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc4c_1x1, name: "1x1", type: conv, kernel: 1, channels: 128, act: ReLU}
      - [{id: inc4c_3x3r, name: "reduce", type: conv, kernel: 1, channels: 128, act: ReLU},
         {id: inc4c_3x3, name: "3x3", type: conv, kernel: 3, channels: 256, act: ReLU}]
      - [{id: inc4c_5x5r, name: "reduce", type: conv, kernel: 1, channels: 24, act: ReLU},
         {id: inc4c_5x5, name: "5x5", type: conv, kernel: 5, channels: 64, act: ReLU}]
      - [{id: inc4c_pool, name: "pool", type: pool, kernel: 3},
         {id: inc4c_proj, name: "proj", type: conv, kernel: 1, channels: 64, act: ReLU}]
    merge: concat

  - name: Inception_4d
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc4d_1x1, name: "1x1", type: conv, kernel: 1, channels: 112, act: ReLU}
      - [{id: inc4d_3x3r, name: "reduce", type: conv, kernel: 1, channels: 144, act: ReLU},
         {id: inc4d_3x3, name: "3x3", type: conv, kernel: 3, channels: 288, act: ReLU}]
      - [{id: inc4d_5x5r, name: "reduce", type: conv, kernel: 1, channels: 32, act: ReLU},
         {id: inc4d_5x5, name: "5x5", type: conv, kernel: 5, channels: 64, act: ReLU}]
      - [{id: inc4d_pool, name: "pool", type: pool, kernel: 3},
         {id: inc4d_proj, name: "proj", type: conv, kernel: 1, channels: 64, act: ReLU}]
    merge: concat

  - name: Inception_4e
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc4e_1x1, name: "1x1", type: conv, kernel: 1, channels: 256, act: ReLU}
      - [{id: inc4e_3x3r, name: "reduce", type: conv, kernel: 1, channels: 160, act: ReLU},
         {id: inc4e_3x3, name: "3x3", type: conv, kernel: 3, channels: 320, act: ReLU}]
      - [{id: inc4e_5x5r, name: "reduce", type: conv, kernel: 1, channels: 32, act: ReLU},
         {id: inc4e_5x5, name: "5x5", type: conv, kernel: 5, channels: 128, act: ReLU}]
      - [{id: inc4e_pool, name: "pool", type: pool, kernel: 3},
         {id: inc4e_proj, name: "proj", type: conv, kernel: 1, channels: 128, act: ReLU}]
    merge: concat

  - {id: pool4, name: Pool4, type: pool, kernel: 3, stride: 2}

  - name: Inception_5a
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc5a_1x1, name: "1x1", type: conv, kernel: 1, channels: 256, act: ReLU}
      - [{id: inc5a_3x3r, name: "reduce", type: conv, kernel: 1, channels: 160, act: ReLU},
         {id: inc5a_3x3, name: "3x3", type: conv, kernel: 3, channels: 320, act: ReLU}]
      - [{id: inc5a_5x5r, name: "reduce", type: conv, kernel: 1, channels: 32, act: ReLU},
         {id: inc5a_5x5, name: "5x5", type: conv, kernel: 5, channels: 128, act: ReLU}]
      - [{id: inc5a_pool, name: "pool", type: pool, kernel: 3},
         {id: inc5a_proj, name: "proj", type: conv, kernel: 1, channels: 128, act: ReLU}]
    merge: concat

  - name: Inception_5b
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc5b_1x1, name: "1x1", type: conv, kernel: 1, channels: 384, act: ReLU}
      - [{id: inc5b_3x3r, name: "reduce", type: conv, kernel: 1, channels: 192, act: ReLU},
         {id: inc5b_3x3, name: "3x3", type: conv, kernel: 3, channels: 384, act: ReLU}]
      - [{id: inc5b_5x5r, name: "reduce", type: conv, kernel: 1, channels: 48, act: ReLU},
         {id: inc5b_5x5, name: "5x5", type: conv, kernel: 5, channels: 128, act: ReLU}]
      - [{id: inc5b_pool, name: "pool", type: pool, kernel: 3},
         {id: inc5b_proj, name: "proj", type: conv, kernel: 1, channels: 128, act: ReLU}]
    merge: concat

layers_after_blocks:
  - {id: pool5, name: Pool5, type: pool, kernel: 7, stride: 1}
  - {id: fc1, name: FC1, type: fc, size: 1024, act: ReLU, dropout: true}
  - {id: fc2, name: FC2, type: fc, size: 1000}
  - {id: output, name: Output, type: output, size: 1000, act: Softmax}
```

*图：GoogLeNet 网络架构图*

从这个架构图可以看到 GoogLeNet 的几个主要的设计特征：

- 第一 **全局平均池化替代全连接层**：这是 GoogLeNet 参数量大幅减少的最重要原因。VGG 用三个全连接层消耗了 89% 的参数，而 GoogLeNet 直接用全局平均池化将 $7 \times 7 \times 1024$ 的特征图压缩为 $1 \times 1 \times 1024$ 的向量，再接一个简单的分类层。这种设计不仅参数量极少，还避免了全连接层容易导致的过拟合问题。

- 第二 **辅助分类器**：在网络中间位置（Inception 4a 和 4d 后）各放置一个辅助分类器。训练时，两个辅助分类器的损失与主损失加权求和：

    $$L_{total} = L_{main} + 0.3 \times L_{aux1} + 0.3 \times L_{aux2}$$

    GoogLeNet 有 22 层（包含所有 Inception 子层），这个深度下梯度消失问题越来越难以抑制。与其让梯度从输出层传播太远逐渐消失，不如中途给它一个出口 —— 在网络中间位置插入分类器，直接对中间特征进行监督。这相当于把一段较深的网络切成几段较浅的网络，让每一段都能获得更强的梯度信号。    
    
    训练时，辅助分类器在反向传播中为中间层提供额外的梯度信号，缓解深层网络的梯度消失问题。推理时，辅助分类器不会起任何作用，只保留主分类输出。后续研究表明，随着 Batch Normalization 等技术的引入，梯度消失有了更有效的抑制手段，辅助分类器的作用变得不那么关键，但这一设计思想启发了后来的"残差连接"和多尺度监督技术。

- 第三 **多尺度特征的层次融合**：每个 Inception 模块同时提取 $1 \times 1$、$3 \times 3$、$5 \times 5$ 和池化四种尺度的特征。随着网络深入，不同层次的多尺度特征逐步融合，最终形成丰富的语义表示。

GoogLeNet 用约 VGG 5% 的参数量，达到了更低的错误率。这是架构设计的胜利，Inception 模块通过多尺度并行和 $1 \times 1$ 降维，实现了参数效率的革命性提升。GoogLeNet 与 AlexNet、VGG-16 的参数量对比如下表所示：

| 网络 | 总参数量 | 卷积层参数 | 全连接层参数 | Top-5 错误率 |
|:----|:--------|:----------|:------------|:------------|
| AlexNet | 62M | 3.75M | 58.63M | 15.3% |
| VGG-16 | 138M | 14.7M | 123.3M | 7.3% |
| GoogLeNet | ~7M | ~4.6M | ~2.4M | 6.7% |

## 网络深度与宽度的权衡

VGG 和 GoogLeNet 的对比引出了一个经典问题：神经网络架构设计应该优先增加深度还是宽度？这个问题至今没有定论，但理解两者的作用和权衡，是掌握现代网络设计的关键。**深度**（Depth）指网络的层数，从输入到输出经过多少个变换层；**宽度**（Width）指每层的通道数（特征数量），每层能同时提取多少种不同的特征。两者对网络性能各有不同影响：
- **深度增加**：层数更多，特征层次更丰富，每层都能在前一层基础上抽象出更高级的语义。但梯度传播路径更长，训练难度增加。
- **宽度增加**：每层通道数更多，特征种类更丰富，能同时捕捉更多样化的模式。但参数量和计算量增长更快，内存压力更大。

两者各有利弊，关键在于找到适合任务需求的平衡点。VGG 和 GoogLeNet 代表了两种截然不同的设计哲学，它们的选择和效果提供了宝贵的实践经验：

- **VGG 选择深度优先策略**：VGG 将 AlexNet 的 8 层扩展到 16-19 层，通道数从 64 逐步增加到 512。这种设计的核心假设是更深的网络能学到更抽象的特征层次，实验结果验证了其有效性。但深度优先的代价是巨大的参数量——VGG 的成功证明了深度的价值，同时也暴露了全连接层的效率问题。

- **GoogLeNet 选择宽度优先策略**：GoogLeNet 深度达到 22 层，与 VGG-19 相当，但通过每个 Inception 模块内部四路并行结构，显著提升了网络的有效宽度。GoogLeNet 同时提取四种尺度的特征，每层能学习更多样化的模式。宽度优先的收益是参数效率，仅 700 万参数达到比 VGG 更低的错误率，而代价是设计与计算复杂度更高，Inception 模块的四路并行需要同时处理多个分支，推理时的并行度要求自然也更高（并行度不是计算量，参数下来了计算量随之下降）。

深度优先和宽度优先的思想在后续都有持续发展。一方面，堆砌深度和参数量的策略让各大厂打起了算力军备竞赛，这种现象在 2017 年 Transformer 架构诞生后发展到顶峰。另一方面，现代网络设计确实也越来越重视效率优化，后续的 MobileNet 使用深度可分离卷积（Depthwise Separable Convolution）进一步降低计算量，EfficientNet 通过复合缩放（Compound Scaling）系统性地平衡深度、宽度、分辨率三个维度。这些创新都延续了 GoogLeNet 的核心思想，用更聪明的架构设计替代粗暴的资源堆砌。

## 本章小结

本章介绍了 2014 年 ImageNet 挑战赛上两项里程碑工作，它们从不同方向突破了 AlexNet 的设计瓶颈，现代 CNN 架构设计由此进入黄金时代。VGG 和 GoogLeNet 将网络深度推进到 20 层左右，但继续加深时遇到了新障碍。当层数超过一定阈值后，更深的网络反而表现更差，不是因为过拟合，而是因为优化愈发困难。这个问题在 2015 年被 ResNet 解决，残差连接让梯度能够直接跳过中间层传播，百层甚至千层网络由此成为现实。下一章将详细介绍 ResNet 的设计思想及其对深度学习领域的深远影响。

## 练习题

1. VGG 用两层 $3 \times 3$ 卷积替代一层 $5 \times 5$ 卷积，感受野相同但参数量更少。假设输入通道数为 $C_{in}$，输出通道数为 $C_{out}$，计算两种方案的参数量比值，并说明堆叠小卷积核的非线性表达优势。
    <details>
    <summary>参考答案</summary>

    **参数量计算**：

    - **单层 $5 \times 5$ 卷积**：参数量 = $5 \times 5 \times C_{in} \times C_{out} + C_{out} = 25 C_{in} C_{out} + C_{out}$
    - **两层 $3 \times 3$ 卷积**：
      - 第一层：$3 \times 3 \times C_{in} \times C_{out} + C_{out} = 9 C_{in} C_{out} + C_{out}$
      - 第二层：$3 \times 3 \times C_{out} \times C_{out} + C_{out} = 9 C_{out}^2 + C_{out}$
      - 总参数量 = $9 C_{in} C_{out} + 9 C_{out}^2 + 2 C_{out}$

    当 $C_{in} = C_{out}$ 时（如 VGG 中各 Block 内的卷积层），参数量比值：
    $$\frac{9 C^2 + 9 C^2 + 2C}{25 C^2 + C} = \frac{18 C^2 + 2C}{25 C^2 + C} \approx \frac{18}{25} = 72\%$$

    两层 $3 \times 3$ 卷积参数量仅为单层 $5 \times 5$ 的 **72%**，节省约 28% 参数。

    **非线性表达优势**：

    两层 $3 \times 3$ 卷积包含两个 ReLU 激活函数，意味着两次非线性变换。从函数空间角度看：
    - 单层 $5 \times 5$ 卷积：$f(x) = \text{ReLU}(W_5 x)$，一次线性变换 + 一次非线性
    - 两层 $3 \times 3$ 卷积：$f(x) = \text{ReLU}(W_3^{(2)} \cdot \text{ReLU}(W_3^{(1)} x))$，两次线性变换 + 两次非线性

    第二种方案的函数空间严格包含第一种，能拟合更复杂的特征关系。譬如，若第一层检测"水平边缘"，第二层可在其基础上检测"水平边缘的特定组合"，这种层次化抽象能力是深层网络的核心优势。
    </details>

