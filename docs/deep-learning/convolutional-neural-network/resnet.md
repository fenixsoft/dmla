# ResNet 残差网络

深度学习的章节进行到这里，让我们来思考一个问题，神经网络的深度是否可以无限增加？VGG 的实验结果证明网络深度确实可以提升精度，如果网络的深度可以无限增加，是否意味着只要硬件算力能持续发展，即使没有更好的算法和模型架构，我们理论上依然能得到任意精度的网络？这听起来很美好，但实际操作中研究者很快发现了问题。2014-2015 年期间，许多团队尝试训练比 VGG-19（19 层）、GoogLeNet（22 层）更深的网络，结果发现当网络深度超过一定阈值（约 20-30 层）后，精度不再提升反而下降。更令人困惑的是，这种现象并非由过拟合导致，过拟合的典型表现是训练集错误率降低而测试集错误率升高，但这里的情况是训练集和测试集的错误率都在升高。这说明网络根本没有学会训练数据中的规律，而是真正意义上的"学不动了"。

2015 年，时任微软研究院研究员的何恺明（Kaiming He）提出了 **ResNet**（Residual Network，残差网络），彻底解决了这一难题。ResNet 通过引入**残差连接**（Residual Connection，又称 Skip Connection）让网络能够训练到 150 层甚至 1000 层以上，同时将 ImageNet 的 Top-5 错误率降至 3.57%，这是机器在视觉上首次超越人类视觉水平（约 5.1%）。该成果以论文《Deep Residual Learning for Image Recognition》发表在 CVPR 2016 上，获得了当年最佳论文奖。这篇论文也成为深度学习历史上被引用最多的论文之一，截至目前引用数超过 15 万次。

## 残差学习思想

在深入理解 ResNet 之前，我们要具体了解前面提到的反直觉的现象：随着网络层数增加，精度不是提升变慢，而是在某个层数拐点出现下降。这不是理论预测的结果，是实际训练中反复出现的观察。

何恺明团队在论文中进行了对比实验：使用相同的数据集、相同的训练策略，分别训练一个 20 层网络和一个 56 层网络，结果发现 56 层网络在训练集和测试集上的错误率都高于 20 层版本。原本的预期是假如 20 层网络已经学得足够好，那么在其后面再添加 36 层（共 56 层），新增的层应该学会什么都不做，即恒等映射（Identity Mapping），这样 56 层网络的表现应该至少与 20 层一样好才对。但实际训练中，让这 36 层学会滥竽充数反而不容易。卷积层的初始化参数通常是随机的小数值，每次前向传播时，输入信号经过多层卷积后会发生显著变化。要让这些层学会保持输入不变，需要精确调整数百万个参数，使它们的组合效果恰好等于恒等映射，这在实际优化中非常困难。这种随着网络深度的增加，训练误差和测试误差同时增加的现象现在被称为神经网络的**退化问题**（Degradation Problem）。

ResNet 的重要创新是改变了模型的学习目标。标准 CNN 下，每个卷积层的目标都是从零开始学习从输入特征到输出特征的映射函数 $H(x)$，当最优解接近恒等映射 $H(x) = x$ 时，网络还是需要从零开始学习 $x$ 到 $x$ 的复杂参数组合。ResNet 更改了学习目标，让网络学习残差函数 $F(x) = H(x) - x$，让模型变为学习需要在恒等映射基础上添加多少修正能够得到最优的映射函数。

这个简单改动就让模型顺利摆脱了恒等映射学习困难的问题。当最优映射恰好是恒等映射时，ResNet 只需要学习 $F(x) = 0$，让所有权重趋近于零就能达到目的。由于权重通常初始化为接近零的小数值，网络一开始就很接近最优解，优化很容易收敛。当最优映射接近但不等于恒等映射时，残差函数只需要学习微小的调整量 $F(x) \approx 0$，而不是完整的映射函数，这也大幅降低了优化难度。

可以用一个类比来理解残差学习的思想，假设你要从一张白纸开始临摹一幅画作（标准 CNN 学习恒等映射），需要精确控制每一笔的位置、颜色、力度，难度极高。但如果让你在一幅几乎已经完成的画作上稍微添加几笔（ResNet 学习残差），难度将大为降低。这就是残差学习的本质，不是学习完整的目标，而是学习目标与基准之间的差异。残差块的数学表达可以清晰地展示这一思想，设 $x$ 是残差块的输入，也是当前模型的基准信号，定义一组权重参数 $W_i$ 和由若干卷积层构成的残差函数 $F$，$F$ 学习在 $x$ 的基础上需要多少修正量才能达到最优的结果 $y$，整个公式表示为输出 = 基准信号 + 修正量：

$$y = x + F(x, \{W_i\})$$

当残差函数学习到 $F(x) = 0$ 时（所有卷积权重趋近于零），输出恰好等于输入 $y = x$，实现了恒等映射。这个特性确保了深层网络的下限至少不会比浅层网络差，即使新增的层没有学到有用的信息，起码可以通过学习 $F(x) = 0$ 来保持输入不变。从梯度流动的角度也能看到残差连接的优势。反向传播时，梯度通过残差连接可以直接从深层传递到浅层，将公式 {{res_y}} 带入到梯度公式可得：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + 1\right)$$

这个公式的意思很明确：输入梯度 = 输出梯度 ×（残差路径梯度 + 直通路径梯度），即使 $\frac{\partial F}{\partial x}$ 趋近于零（残差函数出现梯度消失问题），$\frac{\partial L}{\partial x}$ 仍然等于 $\frac{\partial L}{\partial y}$，梯度通过残差连接的 $+1$ 项无衰减地传递到浅层。这从根本上解决了深层网络的梯度流动问题。

## 残差块

残差连接的实现方式有两种，取决于输入和输出的维度是否相同。ResNet 论文中给出的设计简洁而巧妙，用最小的改动实现了最大的效果。当输入 $x$ 和残差输出 $F(x)$ 的维度完全相同时（通道数、空间尺寸都一致），残差连接只需将输入直接逐元素相加即可。这是最简单的实现方式，也是 ResNet 中大部分残差块采用的方案，其结构如下图所示：

```nn-arch width=620
name: 基本残差块（维度相同）
layout: horizontal

layers:
  - {id: input, name: Input, type: input, size: "HxWxC"}

blocks:
  - name: Identity ResBlock
    type: residual
    style: arc
    main:
      - {id: conv1, name: Conv1, type: conv, kernel: 3, pad: 1, act: ReLU}
      - {id: conv2, name: Conv2, type: conv, kernel: 3, pad: 1, act: ReLU}
    skip: identity
    merge: add
    act: ReLU

layers_after_blocks:
  - {id: output, name: Output, type: output, size: "HxWxC"}
```
*图：基本残差块（维度相同）*

当输入和输出的维度不同时（如通道数从 64 变为 128，或空间尺寸从 $56×56$ 变为 $28×28$），维度不匹配导致无法直接相加。此时需要对残差连接上的输入 $x$ ，进行线性变换以匹配 $F(x)$ 的维度。ResNet 使用 $1×1$ 卷积来实现这个变换。具体结构如下图所示：

```nn-arch width=620
name: 基本残差块（维度不同）
layout: horizontal

layers:
  - {id: input, name: Input, type: input, size: "HxWxC"}

blocks:
  - name: Projection ResBlock
    type: residual
    style: parallel
    main:
      - {id: conv1, name: Conv1, type: conv, kernel: 3, stride: 2, pad: 1, act: ReLU}
      - {id: conv2, name: Conv2, type: conv, kernel: 3, pad: 1, act: ReLU}
    skip:
      - {id: skip_conv, name: "Conv_Skip", type: conv, kernel: 1, stride: 2}
    merge: add
    act: ReLU

layers_after_blocks:
  - {id: output, name: Output, type: output, size: "H'xW'xC'"}
```

其中 Conv_Skip 是一个 $1×1$ 卷积层，用于调整通道数。如果空间尺寸需要变化（如 Stride=2 下采样），则在残差连接上也添加 Stride=2 的池化或卷积操作。根据 ResNet 论文中的实验数据，使用 $1×1$ 卷积的投影方式略优于零填充维度匹配，但两者差距不大（约 0.2% 的精度差异）。考虑到计算效率，ResNet 仅在需要改变维度时才使用投影，其他情况下都采用直接相加。

以上两种残差块都统称为**基本块**（Basic Block），是 ResNet 论文中最基础的残差块结构，结构简单、感受野明确，都由两个 $3×3$ 卷积层构成，卷积叠加后的感受野为 $5×5$（第一个卷积感受野 $3×3$，第二个在此基础上扩展，总感受野 $5×5$）。这种设计借鉴了 VGG 的"小卷积堆叠"思想，用多个小卷积代替一个大卷积，既能减少参数量又能增加非线性激活，提升网络的表达能力，适用于 ResNet-18 和 ResNet-34 等相对浅层的网络。

对于 ResNet-50、ResNet-101、ResNet-152 等相对深层的网络，则使用另外一种被称为**瓶颈块**（Bottleneck Block）的结构，这是 ResNet 为深层网络设计的高效结构，由三个卷积层构成：$1×1$ 降维卷积、$3×3$ 主卷积、$1×1$ 升维卷积。瓶颈块的核心思想是通过 $1×1$ 卷积降低通道数，减少 $3×3$ 卷积的计算量，然后再用 $1×1$ 卷积恢复通道数。这种"降维 - 卷积 - 升维"的设计大幅减少了参数量和计算量，同时保持了 $3×3$ 卷积的感受野。
瓶颈块的具体结构如下图所示：

```nn-arch width=760
name: 瓶颈残差块
layout: horizontal

layers:
  - {id: input, name: Input, type: input, size: "HxWxC"}

blocks:
  - name: Bottleneck ResBlock
    type: residual
    style: arc
    main:
      - {id: conv1, name: Conv1, type: conv, kernel: 1, pad: 0, act: ReLU}
      - {id: conv2, name: Conv2, type: conv, kernel: 3, pad: 1, act: ReLU}
      - {id: conv3, name: Conv3, type: conv, kernel: 1, pad: 1, act: ReLU}
    skip: identity
    merge: add
    act: ReLU

layers_after_blocks:
  - {id: output, name: Output, type: output, size: "HxWxC"}
```
*图：残差块（维度相同）*

这些残差块有两个稍微反直觉的特点值得说明：一是虽说瓶颈块多用于深层网络，但网络深度很大程度是由瓶颈块本身带来的。网络深度的定义是有权重参数的层数，即卷积层和全连接层的总数。基本块有 2 个卷积层，瓶颈块却有 3 个，使用瓶颈块的网络在 Block 数量相同的前提下层数要比基本块多出 50%。如果 ResNet-34 全部改用瓶颈块，网络实际有约 50 层，即为 ResNet-50 结构。二是相较于基本块，瓶颈块虽然层数更多，但参数量却更少（所以才叫高级结构）。以输入、输出均为 256 通道为例计算两种残差块的参数量差异，基本块（两个 $3×3$ 卷积）的参数量为 $1,180,160$，瓶颈块（$1×1$ → $3×3$ → $1×1$，中间通道数 64）的参数量为 $69,632$，仅为基本块的 5.9% 而已，这是一个惊人的效率提升。同时，瓶颈块的中心感受野仍然来自 $3×3$ 卷积，与基本块的有效感受野相同，这使得深层网络（如 ResNet-152）能够在保持计算可行性的同时，堆叠更多的残差块。

## 残差网络架构

ResNet 论文中提出了多种深度的网络配置，如下表所示，从 18 层到 152 层，覆盖了从轻量级到重量级的多种应用场景。

| 网络 | 层数 | 残差块数量配置 | 残差块类型 | 参数量 | Top-1 错误率（单裁） |
|:----|:----|:-------------|:--------|:------|:------------------|
| ResNet-18 | 18 | [2, 2, 2, 2] | Basic | ~11.7M | 30.3% |
| ResNet-34 | 34 | [3, 4, 6, 3] | Basic | ~21.8M | 26.2% |
| ResNet-50 | 50 | [3, 4, 6, 3] | Bottleneck | ~25.6M | 23.9% |
| ResNet-101 | 101 | [3, 4, 23, 3] | Bottleneck | ~44.5M | 22.6% |
| ResNet-152 | 152 | [3, 8, 36, 3] | Bottleneck | ~60.2M | 21.6% |

以其中最浅的 ResNet-18 为例，它的完整网络架构可以用下图清晰展示。这个架构体现了 ResNet 几个关键设计决策。首先，初始层使用 $7×7$ 大卷积核配合 Stride=2 一次性将空间尺寸从 $224×224$ 降到 $112×112$，然后通过池化进一步降到 $56×56$。这与 VGG 的"堆叠 3×3 卷积"策略不同，ResNet 选择了更激进的下采样策略。其次，每个残差块进行一次下采样（Stride=2），将空间尺寸减半，通道数翻倍。这种设计确保了特征图在空间维度上逐渐缩小，在通道维度上逐渐增大，平衡了信息量和计算量。最后，[全局平均池化](cnn-basics.md#cnn-架构设计原则)（Global Average Pooling）替代了 VGG 和 AlexNet 中的多个全连接层，大幅减少了参数量（从 VGG-16 的约 100M 参数降到 ResNet-34 的约 11.7M），同时避免了全连接层容易过拟合的问题。

```nn-arch width=1400
name: ResNet-18 网络架构（基本残差块，2, 2, 2, 2 配置）
layout: horizontal

layers:
  - {id: input, name: Input, type: input, size: "224x224x3"}
  - {id: conv1, name: Conv1, type: conv, kernel: 7, stride: 2, channels: 64, out: "112×112x64", act: ReLU}
  - {id: pool1, name: Pool1, type: pool, kernel: 3, stride: 2, out: "56x56x64"}

blocks:
  - name: ResBlock1
    type: residual
    main:
      - {id: rb1_conv1, name: conv1, type: conv, kernel: 3, channels: 64, act: ReLU}
      - {id: rb1_conv2, name: conv2, type: conv, kernel: 3, channels: 64, act: ReLU}
    skip: identity
    merge: add
    act: ReLU

  - name: ResBlock2
    type: residual
    main:
      - {id: rb2_conv1, name: conv1, type: conv, kernel: 3, channels: 128, act: ReLU}
      - {id: rb2_conv2, name: conv2, type: conv, kernel: 3, channels: 128, act: ReLU}
    skip: identity
    merge: add
    act: ReLU

  - name: ResBlock3
    type: residual
    main:
      - {id: rb3_conv1, name: conv1, type: conv, kernel: 3, channels: 256, act: ReLU}
      - {id: rb3_conv2, name: conv2, type: conv, kernel: 3, channels: 256, act: ReLU}
    skip: identity
    merge: add
    act: ReLU

  - name: ResBlock4
    type: residual
    main:
      - {id: rb4_conv1, name: conv1, type: conv, kernel: 3, channels: 512, act: ReLU}
      - {id: rb4_conv2, name: conv2, type: conv, kernel: 3, channels: 512, act: ReLU}
    skip: identity
    merge: add
    act: ReLU


layers_after_blocks:
  - {id: globalpool, name: GlobalPool, type: pool, kernel: global, out: "1x1x512"}
  - {id: fc, name: FC, type: fc, size: 1000}
  - {id: output, name: Output, type: output, size: 1000, act: Softmax}
```
ResNet 论文发表后，何恺明团队继续深入研究残差网络的工作原理。2016 年，何恺明发表了另一篇重要论文《Identity Mappings in Deep Residual Networks》（ECCV 2016），提出了预激活（Pre-Activation）版本的残差块，进一步优化了梯度流动。原始 ResNet 的残差块采用后激活结构，卷积层后面紧跟 BN 和 ReLU，最后通过残差连接将输入加到输出上，然后再加一个 ReLU。这种结构的问题是残差连接后面有一个 ReLU，会强制输出为非负数，限制了恒等映射的表达能力。预激活版本的改进是将 BN 和 ReLU 移到卷积层之前。这样做的好处是残差连接直接加到输出上，不再经过 ReLU，保持了恒等映射的纯粹性。

预激活版本的优势可以通过梯度流动的角度来理解。原始版本中，残差连接后面有一个 ReLU，梯度在通过 ReLU 时会被截断（负梯度变成零）。预激活版本中，残差连接直接传递到输出，梯度可以无损地通过这条路径。何恺明的实验表明，预激活版本在深层网络（如 ResNet-1001）上的表现显著优于原始版本，训练更加稳定，收敛更快。

现代 ResNet 的实现（如 PyTorch 的 torchvision 库、Facebook 的 Detectron2 库）通常使用预激活版本。虽然对于 ResNet-50/101/152 等中等深度的网络，原始版本和预激活版本的表现差异不大，但预激活版本已经成为残差块在机器学习框架中预置的标准范式。

## 应用与影响

ResNet 的影响远超 ImageNet 分类比赛本身。残差学习提出的学习"相对于基准的改进"而非"完整的映射"的思想，已经成为深度神经网络设计的一种重要设计模式，被广泛应用于计算机视觉、自然语言处理、生成模型等几乎所有深度学习领域，如下表所示：

| 应用领域 | 代表模型 | 残差连接的具体作用 |
|:--------|:--------|:-----------------|
| 目标检测 | Faster R-CNN、Mask R-CNN | 使用 ResNet 作为 backbone，替代 VGG-16 |
| 语义分割 | DeepLab v3+、FCN-ResNet | 使用 ResNet 提取多尺度特征 |
| 自然语言处理 | Transformer、BERT | 残差连接贯穿整个架构，保证深层网络的梯度流动 |
| 生成模型 | StyleGAN、DDPM | 残差块用于生成器和去噪网络 |
| 视频理解 | 3D ResNet、I3D | 将 2D 残差块扩展为 3D，处理时空信息 |
| 自监督学习 | SimCLR、MoCo | ResNet 作为特征提取器，学习对比表示 |

其中最深远的影响是 Transformer 中的残差连接。Transformer 架构是下一部分大语言模型中的主角，它将残差连接作为核心设计，每个多头注意力层和每个前馈网络层都通过残差连接将输入加到输出上。这种设计让 Transformer 能够堆叠数十层（如 BERT-Base 有 12 层，BERT-Large 有 24 层，GPT-3 有 96 层），而不会出现优化困难。Transformer 的成功证明了残差学习的价值早已超越了图像识别的范畴，适用于各种神经网络架构。

## 本章小结

本章介绍了深度学习历史上最重要的网络架构之一 ResNet。它通过残差连接巧妙地解决了恒等映射难以学习的问题，将学习目标从学习完整映射 $H(x) = x$ 改为学习残差部分 $H(x) = F(x) + x$。残差连接的更重要贡献是改善梯度流动。残差连接为梯度提供了直通路径，反向传播时梯度可以直接从深层传递到浅层，不会因为中间层的导数连乘而衰减。这使得训练出 1000 层以上的深层网络成为可能，而不会出现梯度消失或退化问题。

ResNet-152 在 ImageNet 上取得了突破性成果：Top-5 错误率 3.57%（多裁），Top-1 错误率 21.6%（单裁），首次超越人类水平。更重要的是，残差学习思想被广泛应用于深度学习的各个领域。目标检测领域的 Faster R-CNN、Mask R-CNN 使用 ResNet 作为 Backbone；自然语言处理领域的 Transformer、BERT 将残差连接作为核心设计；生成模型领域的 StyleGAN、DDPM 在网络中大量使用残差块。可以说，残差连接已经成为现代神经网络不可或缺的组件。

## 练习题

1. 基本块和瓶颈块的参数量对比计算。设输入、输出通道数均为 256，计算两种残差块的参数量并分析瓶颈块为何参数量更少。
    <details>
    <summary>参考答案</summary>
    
    **基本块参数量计算**：
    
    基本块由两个 $3×3$ 卷积层构成：
    - 第一个 $3×3$ 卷积：$256 × 256 × 3 × 3 = 589,824$ 个权重参数 + $256$ 个偏置参数 = $590,080$
    - 第二个 $3×3$ 卷积：$256 × 256 × 3 × 3 = 589,824$ 个权重参数 + $256$ 个偏置参数 = $590,080$
    - **总参数量**：$590,080 × 2 = 1,180,160$
    
    **瓶颈块参数量计算**：
    
    瓶颈块采用"降维-卷积-升维"结构，中间通道数为 64：
    - $1×1$ 降维卷积：$256 × 64 × 1 × 1 = 16,384$ 个权重 + $64$ 个偏置 = $16,448$
    - $3×3$ 主卷积：$64 × 64 × 3 × 3 = 11,520$ 个权重 + $64$ 个偏置 = $11,584$
    - $1×1$ 升维卷积：$64 × 256 × 1 × 1 = 16,384$ 个权重 + $256$ 个偏置 = $16,640$
    - **总参数量**：$16,448 + 11,584 + 16,640 = 69,632$
    
    **效率对比**：
    
    瓶颈块参数量仅为基本块的 $\frac{69,632}{1,180,160} ≈ 5.9\%$。瓶颈块通过 $1×1$ 卷积将通道数从 256 降到 64，使得计算量最大的 $3×3$ 卷积只需处理 $64×64$ 的通道映射，而非 $256×256$。这个设计让深层网络（如 ResNet-152）能够堆叠更多残差块，同时保持计算可行性。
    </details>

2. 证明残差连接如何改善梯度流动。根据公式 $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (\frac{\partial F}{\partial x} + 1)$，解释为什么残差连接能避免梯度消失。
    <details>
    <summary>参考答案</summary>
    
    **梯度公式分析**：
    
    残差块的输出为 $y = x + F(x)$，反向传播时梯度传递公式为：
    
    $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + 1\right)$$
    
    这个公式分解为两部分：
    - $\frac{\partial L}{\partial y} \cdot \frac{\partial F}{\partial x}$：通过残差函数 $F$ 的梯度路径
    - $\frac{\partial L}{\partial y} \cdot 1$：通过残差连接的直通梯度路径
    
    **关键洞察**：
    
    即使残差函数出现梯度消失（$\frac{\partial F}{\partial x} \to 0$），直通路径的 $+1$ 项仍然存在，梯度变为：
    
    $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot 1 = \frac{\partial L}{\partial y}$$
    
    这意味着梯度可以无损地从深层直接传递到浅层，不经过任何衰减。
    
    **对比标准 CNN**：
    
    标准 CNN（无残差连接）中，梯度需要经过多层连续相乘：
    
    $$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_n} \cdot \prod_{i=1}^{n-1} \frac{\partial x_{i+1}}{\partial x_i}$$
    
    当每层导数 $\frac{\partial x_{i+1}}{\partial x_i} < 1$ 时（如 Sigmoid 的导数最大值仅 0.25），连乘后梯度会指数级衰减，深层梯度几乎无法传递到浅层。
    
    **ResNet 的优势**：
    
    残差连接为每层梯度提供了一个"保底"通道。即使 $F$ 的梯度消失，至少有 $+1$ 的直通梯度。这使得 ResNet 能够训练到 152 层甚至 1000 层，而不会出现严重的梯度消失问题。
    </details>

3. ResNet-34 使用基本块，配置为 [3, 4, 6, 3]。计算该网络的总层数（有权重参数的层数），并解释为何改用瓶颈块后层数会增加。
    <details>
    <summary>参考答案</summary>
    
    **ResNet-34 层数计算**：
    
    网络层数定义：有权重参数的层数（卷积层 + 全连接层）。
    
    - 初始层：$7×7$ 卷积（1 层） + 最大池化（无权重，不计入）
    - 残差块部分：
      - 第一阶段：3 个基本块 × 2 个卷积 = 6 层
      - 第二阶段：4 个基本块 × 2 个卷积 = 8 层
      - 第三阶段：6 个基本块 × 2 个卷积 = 12 层
      - 第四阶段：3 个基本块 × 2 个卷积 = 6 层
    - 结尾层：全局平均池化（无权重） + 全连接层（1 层）
    
    **总层数**：$1 + 6 + 8 + 12 + 6 + 1 = 34$ 层
    
    **改用瓶颈块的影响**：
    
    瓶颈块由 3 个卷积层构成（$1×1$ → $3×3$ → $1×1$），而基本块只有 2 个。如果 ResNet-34 的配置 [3, 4, 6, 3] 全部改用瓶颈块：
    
    - 残差块部分：$(3 + 4 + 6 + 3) × 3 = 48$ 层
    - **总层数**：$1 + 48 + 1 = 50$ 层
    
    这正是 ResNet-50 的层数配置。虽然层数增加了，但由于瓶颈块的参数效率更高，ResNet-50 的参数量（~25.6M）反而比 ResNet-34（~21.8M）增加不多，却能获得更好的性能（Top-1 错误率从 26.2% 降到 23.9%）。
    
    **设计权衡**：
    
    网络深度与参数量并非简单的正比关系。瓶颈块通过"降维"策略，用更多层数换取更强的表达能力，同时控制参数增长。这种设计让深层网络在计算资源和模型性能之间找到平衡点。
    </details>
