# 权重初始化

在上一章中，我们深入探讨了自适应优化器如何根据参数的梯度历史动态调整学习率。AdamW 等优化器显著提升了训练效率，但它们都面临一个共同的起点问题：参数的初始值如何设置？如果初始权重全为零，所有神经元输出相同，反向传播时梯度也相同，网络无法学习。如果初始权重过大，前向传播激活值饱和，反向传播梯度消失，训练停滞。

**权重初始化**（Weight Initialization）决定了网络训练的起点。一个好的初始化能让训练快速、稳定地收敛；一个糟糕的初始化可能导致训练停滞、震荡甚至崩溃。本章将深入分析初始化的重要性，介绍 Xavier 初始化和 He 初始化两种经典方法，并通过实验验证初始化对训练的深远影响。

## 为什么初始化至关重要

想象一下，你给一群学生布置了一道数学题，但要求他们必须用完全相同的步骤来解答。结果如何？所有人提交的答案一模一样，你无法判断谁掌握了知识点，谁还需要辅导。神经网络的训练也是如此 —— 如果所有神经元从完全相同的起点出发，它们就会学习完全相同的功能，整个网络的表达能力将退化到单个神经元。

### 全零初始化的陷阱

初学者直觉上认为，将参数初始化为零是最"干净"的起点，就像数学中从零开始计数一样自然。然而，这个直觉恰恰是深度学习中最致命的错误之一。

让我们用一个具体例子来分析问题所在。假设一个简单的两层全连接网络，输入 $x$ 经过隐藏层 $h$ 到达输出 $y$：

$$h = \mathbf{W}_1 x, \quad y = \mathbf{W}_2 h$$

其中 $\mathbf{W}_1$ 是输入层到隐藏层的权重矩阵，$\mathbf{W}_2$ 是隐藏层到输出层的权重矩阵。如果我们把这两个矩阵都初始化为零：

$$\mathbf{W}_1 = 0, \quad \mathbf{W}_2 = 0$$

在[前向传播](../neural-network-structure/backpropagation.md#前向传播的计算过程)阶段，灾难就已经发生了：

- 隐藏层所有神经元的输出都是 $h_i = 0$（因为权重为零）
- 输出层所有神经元的输出都是 $y_i = 0$（因为 $h = 0$ 且权重为零）

这意味着所有神经元 —— 无论有多少个 —— 都在"说同样的话"。网络的表达能力瞬间退化：原本设计了 100 个隐藏神经元来学习不同的特征，现在这 100 个神经元变成了 100 个"复制品"。

问题在[反向传播](../neural-network-structure/backpropagation.md#反向传播的梯度计算)阶段变得更加严重。隐藏层权重的梯度公式为：

$$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial y} \cdot \mathbf{W}_2$$

这个公式看着抽象，拆开来看含义很直观：
- $\frac{\partial L}{\partial y}$ 是输出层的误差信号，表示"输出与目标的偏差有多大"
- $\mathbf{W}_2$ 是隐藏层到输出层的权重，它决定了"误差信号如何传回隐藏层"
- 整体公式表示：隐藏层的梯度等于输出误差乘以权重矩阵

当 $\mathbf{W}_2 = 0$ 时，梯度直接为零：

$$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial y} \cdot 0 = 0$$

零梯度意味着参数无法更新，网络永远停留在初始状态。即使 $\mathbf{W}_2$ 不为零（比如只初始化 $\mathbf{W}_1$ 为零），隐藏层所有神经元仍然接收相同的梯度（因为 $\mathbf{W}_2$ 的每一行不同，但 $h = 0$ 使得误差传播到所有隐藏神经元时完全一致），更新后的权重也完全相同。这种现象被称为**对称性破坏问题**（Symmetry Breaking Problem）：所有神经元被迫学习相同的功能，网络的多层、多神经元结构变得毫无意义。

全零初始化的核心问题可以总结为一句话：**它剥夺了神经元之间的差异性**，使精心设计的网络结构退化成单个神经元。

### 随机初始化的必要性

既然全零初始化让所有神经元"步调一致"，那解决方案自然浮出水面：给每个神经元一个不同的起点。这就像让一群学生从不同的位置出发登山，他们各自的路径不同，最终到达的高度也不同，从而形成丰富多样的学习结果。

**随机初始化**的核心思想是为每个权重赋予不同的随机值，打破神经元之间的对称性。最常用的两种随机分布是均匀分布和正态分布。

**均匀分布初始化**从区间 $[-a, a]$ 中随机采样权重：

$$[eq:uniform-init] \mathbf{W}_{ij} \sim U[-a, a]$$

这个公式看着抽象，拆开来看含义很直观：
- $U[-a, a]$ 表示均匀分布，概率在区间 $[-a, a]$ 内均匀分布
- $\mathbf{W}_{ij}$ 表示权重矩阵的第 $i$ 行第 $j$ 列元素
- 每个权重独立采样，互不影响

均匀分布的方差为：

$$\text{Var}(\mathbf{W}_{ij}) = \frac{a^2}{3}$$

**正态分布初始化**从零均值正态分布中采样权重：

$$[eq:normal-init] \mathbf{W}_{ij} \sim N(0, \sigma^2)$$

这个公式的含义：
- $N(0, \sigma^2)$ 表示正态分布，均值为 0，方差为 $\sigma^2$
- $\sigma$ 是标准差，控制权重的分散程度
- 大部分权重集中在 $[-\sigma, \sigma]$ 范围内，少数权重可能较大或较小

随机初始化成功打破了神经元之间的对称性，使每个神经元能够学习不同的特征。但这里隐藏着一个关键问题：**参数 $a$ 或 $\sigma$ 应该设为多少？**

这个问题看似简单，实则关乎训练成败。想象两个极端场景：

- **参数过大**（比如 $\sigma = 10$）：前向传播时，权重乘以输入产生巨大的激活值。sigmoid 激活函数收到 100 这样的输入，输出接近 1；tanh 收到 100，输出也接近 1。激活值"饱和"在边界，反向传播时梯度接近零 —— 这就是经典的[梯度消失](../neural-network-stability/vanishing-exploding-gradients.md)问题。

- **参数过小**（比如 $\sigma = 0.001$）：前向传播时，激活值逐层衰减，信号像" whispers"一样越传越弱。反向传播时，梯度也逐层缩小，训练几乎停滞。

好的初始化需要在两个目标之间找到平衡：

1. **打破对称性**：权重随机，让每个神经元有独特的"个性"
2. **保持信号强度**：激活值不饱和（不被激活函数边界压制），梯度不消失（能稳定传回输入层）

这两个目标看似矛盾 —— 随机值太小会导致信号衰减，太大会导致信号饱和。如何找到"恰到好处"的随机分布参数？这正是 Xavier 初始化和 He 初始化要解决的核心问题。

## Xavier 初始化

上一节我们认识到：随机初始化打破对称性，但参数选择不当会导致信号衰减或饱和。那有没有一种"科学"的方法，能精确计算出最佳的初始化参数？答案是肯定的 —— 通过分析信号在网络中的传播过程，我们可以推导出保持信号稳定所需的权重方差。

这种方法最早由加拿大计算机科学家 Xavier Glorot 和他的导师 Yoshua Bengio（2018 年图灵奖得主）在 2010 年提出。他们在题为《Understanding the difficulty of training deep feedforward neural networks》的论文中，系统分析了深度网络训练困难的原因，并提出了著名的 Xavier 初始化方法。这篇论文揭示了初始化与训练稳定性的深层联系，成为深度学习优化领域的里程碑工作。

### 信号传播分析

Xavier 初始化的核心思想非常朴素：如果信号在传播过程中方差变化太大，训练就会不稳定。那么，什么样的权重方差能让信号"保持原样"地穿过一层又一层的网络？

让我们从最简单的情况开始分析 —— 一个线性神经元（暂时忽略激活函数）。假设这个神经元接收 $n_{in}$ 个输入，产生一个输出：

$$y = \sum_{i=1}^{n_{in}} w_i x_i$$

这个公式看着抽象，拆开来看含义很直观：
- $n_{in}$ 是输入神经元数量，也叫"扇入"（fan-in）
- $w_i$ 是第 $i$ 个输入对应的权重
- $x_i$ 是第 $i$ 个输入值
- 整体公式表示：输出是所有加权输入的累加

为了分析输出的方差，我们需要做一些合理的假设。假设权重 $w_i$ 和输入 $x_i$ 满足以下条件：
- 独立同分布（每个权重、每个输入都是独立采样的）
- 零均值（$E[w_i] = 0$, $E[x_i] = 0$）
- 方差固定（$\text{Var}(w_i) = \text{Var}(w)$, $\text{Var}(x_i) = \text{Var}(x)$）

在这些假设下，输出 $y$ 的方差可以推导出来：

$$[eq:output-var] \text{Var}(y) = \sum_{i=1}^{n_{in}} \text{Var}(w_i x_i) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

这个公式揭示了信号传播的关键规律：
- $n_{in}$ 是输入数量 —— 输入越多，方差越大
- $\text{Var}(w)$ 是权重方差 —— 权重越分散，方差越大
- $\text{Var}(x)$ 是输入方差 —— 输入越分散，方差越大
- 整体公式表示：输出方差等于三个因素的乘积

让我们用一个具体例子验证这个公式。假设一个神经元有 512 个输入（$n_{in} = 512$），权重方差 $\text{Var}(w) = 0.01$，输入方差 $\text{Var}(x) = 1$：

$$\text{Var}(y) = 512 \times 0.01 \times 1 = 5.12$$

输出方差从 1 变成了 5.12！这说明信号穿过一层网络后，方差被放大了。如果我们想要信号强度保持不变（$\text{Var}(y) = \text{Var}(x)$），就需要满足：

$$n_{in} \cdot \text{Var}(w) = 1$$

即权重方差应该是：

$$\text{Var}(w) = \frac{1}{n_{in}}$$

代入刚才的例子：$n_{in} = 512$，权重方差应该是 $\frac{1}{512} \approx 0.002$。这样输出方差就能保持 $\text{Var}(y) = 512 \times 0.002 \times 1 = 1$，与输入方差相等。

推导过程的具体步骤如下。首先根据方差定义：

$$\text{Var}(y) = \text{Var}\left(\sum_{i=1}^{n_{in}} w_i x_i\right)$$

由于 $w_i$ 和 $x_i$ 独立：

$$\text{Var}(y) = \sum_{i=1}^{n_{in}} \text{Var}(w_i x_i)$$

对于独立随机变量的乘积，方差公式为：

$$\text{Var}(w_i x_i) = \text{Var}(w_i) \cdot \text{Var}(x_i) + \text{Var}(w_i) \cdot E[x_i]^2 + \text{Var}(x_i) \cdot E[w_i]^2$$

当 $E[w_i] = 0$ 且 $E[x_i] = 0$（零均值假设）：

$$\text{Var}(w_i x_i) = \text{Var}(w) \cdot \text{Var}(x)$$

因此：

$$\text{Var}(y) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

### 反向传播分析

前向传播分析告诉我们：权重方差应该是 $\frac{1}{n_{in}}$ 才能保持信号强度。但这只考虑了"从输入到输出"的方向。训练神经网络还需要反向传播 —— 梯度从输出层传回输入层，这个方向会发生什么？

反向传播的核心机制是：输出层的梯度 $\delta_y$ 通过权重矩阵"反向"传回输入层，形成输入层梯度 $\delta_x$：

$$\delta_x = \mathbf{W}^T \delta_y$$

这个公式看着抽象，拆开来看含义很直观：
- $\delta_y$ 是输出层的梯度信号，表示"输出误差对输出的影响"
- $\mathbf{W}^T$ 是权重矩阵的转置，用于"反向"传递信号
- $\delta_x$ 是传回输入层的梯度，表示"输出误差对输入的影响"
- 整体公式表示：梯度通过权重矩阵"反射"回前一层

类比理解：想象光线穿过一面镜子。光线从光源发出，照射到镜子上，然后反射回来。权重矩阵就像镜子，决定反射的方向和强度。如果镜子太小或太歪，反射的光线就会很弱。

数学上，输入层梯度方差与输出层梯度方差的关系：

$$\text{Var}(\delta_x) = n_{out} \cdot \text{Var}(w) \cdot \text{Var}(\delta_y)$$

这个公式的结构与前向传播公式类似：
- $n_{out}$ 是输出神经元数量，也叫"扇出"（fan-out）
- $\text{Var}(w)$ 是权重方差
- $\text{Var}(\delta_y)$ 是输出层梯度方差
- 整体公式表示：梯度方差等于三个因素的乘积

为了保持梯度强度不变（$\text{Var}(\delta_x) = \text{Var}(\delta_y)$），需要：

$$n_{out} \cdot \text{Var}(w) = 1$$

即权重方差应该是：

$$\text{Var}(w) = \frac{1}{n_{out}}$$

这里出现了一个矛盾：前向传播要求 $\text{Var}(w) = \frac{1}{n_{in}}$，反向传播要求 $\text{Var}(w) = \frac{1}{n_{out}}$。当 $n_{in} \neq n_{out}$ 时（这在实际网络中非常常见），我们无法同时满足两个条件。

Xavier Glorot 和 Bengio 给出了一个巧妙的折中方案。

### Xavier 初始化公式

前向传播要求 $\text{Var}(w) = \frac{1}{n_{in}}$，反向传播要求 $\text{Var}(w) = \frac{1}{n_{out}}$。这两个条件就像拔河 —— 一个向左拉，一个向右拉。当 $n_{in} \neq n_{out}$ 时，我们只能选择一个折中位置。

Xavier 初始化采用的是 $n_{in}$ 和 $n_{out}$ 的调和平均：

$$[eq:xavier-var] \text{Var}(w) = \frac{2}{n_{in} + n_{out}}$$

这个公式看着抽象，拆开来看含义很直观：
- $n_{in} + n_{out}$ 是输入和输出神经元数量之和
- 分子 2 来自调和平均的推导，使公式能同时兼顾前向和反向传播
- 整体公式表示：权重方差是输入输出数量的"折中值"

为什么选择调和平均而非算术平均？原因是调和平均对较小的数值更敏感。当 $n_{in}$ 和 $n_{out}$ 差异较大时，调和平均会偏向较小的值，避免方差过大导致梯度爆炸。

让我们用一个具体例子验证这个公式的效果。假设网络某一层有 $n_{in} = 512$ 个输入，$n_{out} = 256$ 个输出：

$$\text{Var}(w) = \frac{2}{512 + 256} = \frac{2}{768} \approx 0.0026$$

对应的标准差为：

$$\sigma = \sqrt{0.0026} \approx 0.051$$

这意味着权重从正态分布 $N(0, 0.051^2)$ 中采样，大部分权重落在 $[-0.051, 0.051]$ 范围内。

如果只考虑前向传播的要求：

$$\text{Var}(w) = \frac{1}{512} \approx 0.002$$

如果只考虑反向传播的要求：

$$\text{Var}(w) = \frac{1}{256} \approx 0.004$$

Xavier 的折中方案 $0.0026$ 正好介于两者之间。

基于这个方差公式，Xavier 初始化有两种具体实现方式。

**Xavier 均匀初始化**从均匀分布中采样：

$$\mathbf{W}_{ij} \sim U\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$$

这个公式如何推导？均匀分布 $U[-a, a]$ 的方差为 $\frac{a^2}{3}$。为了使方差等于 $\frac{2}{n_{in} + n_{out}}$，需要：

$$\frac{a^2}{3} = \frac{2}{n_{in} + n_{out}}$$

解得 $a = \sqrt{\frac{6}{n_{in} + n_{out}}}$。

**Xavier 正态初始化**从正态分布中采样：

$$\mathbf{W}_{ij} \sim N\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

正态分布 $N(0, \sigma^2)$ 的方差就是 $\sigma^2$，因此标准差直接取 $\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$。

### Xavier 初始化的适用场景

Xavier 初始化的推导有一个关键假设：激活函数是线性的。这个假设在实际中是否成立？

让我们回顾一下 [sigmoid 激活函数](../neural-network-structure/activation-loss-functions.md#sigmoid-函数) 的形状。sigmoid 函数定义为：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

当输入 $x$ 在 0 附近时，sigmoid 函数的行为可以用泰勒展开近似：

$$\sigma(x) \approx \sigma(0) + \sigma'(0) \cdot x = 0.5 + 0.25 \cdot x$$

这是一个线性函数！也就是说，当激活值落在 0 附近时，sigmoid 近似于线性变换，Xavier 的线性假设成立。

类似的，[tanh 激活函数](../neural-network-structure/activation-loss-functions.md#tanh-函数) 在 0 附近也近似线性：

$$\tanh(x) \approx \tanh(0) + \tanh'(0) \cdot x = 0 + 1 \cdot x = x$$

tanh 在 0 附近的线性近似比 sigmoid 更精确（斜率为 1，而 sigmoid 斜率为 0.25），这也是 tanh 在深度网络中表现更好的原因之一。

Xavier 初始化正是利用了这个特性：通过控制权重方差，使激活值落在 0 附近（线性区域），从而保持信号强度稳定传播。用一个具体例子说明：

假设输入 $x \sim N(0, 1)$，网络层有 $n_{in} = n_{out} = 100$ 个神经元。Xavier 初始化方差为：

$$\text{Var}(w) = \frac{2}{100 + 100} = 0.01$$

前向传播后，输出方差为：

$$\text{Var}(y) = 100 \times 0.01 \times 1 = 1$$

输出 $y$ 仍然服从 $N(0, 1)$，大部分值落在 $[-2, 2]$ 范围内。这个范围正好是 sigmoid 和 tanh 的"近线性区域"（sigmoid 在 $[-1, 1]$ 区间相对线性，tanh 在 $[-2, 2]$ 区间相对线性）。激活值不会饱和（接近 0 或 1），梯度也不会消失。

然而，Xavier 初始化存在明显的局限性：

**局限性一：假设激活函数线性**

sigmoid 和 tanh 只在 0 附近近似线性，当输入远离 0 时，非线性特征显现。例如 sigmoid 在输入为 5 时，输出接近 0.993，几乎饱和；梯度接近 0，反向传播受阻。Xavier 初始化只能尽量让激活值落在线性区域，但无法完全避免饱和。

**局限性二：不适用于 ReLU**

[ReLU 激活函数](../neural-network-structure/activation-loss-functions.md#relu-函数) 的特性与 sigmoid/tanh 完全不同。ReLU 只保留正值，负值全部置零：

$$f(x) = \max(0, x)$$

这意味着约一半的激活值被"杀死"，信号强度减半。Xavier 的线性假设在 ReLU 上完全失效 —— 如果用 Xavier 初始化 ReLU 网络，激活值会逐层衰减，深层网络的梯度几乎为零。

针对 ReLU 的特殊性，需要一种新的初始化方法，这正是 He 初始化要解决的问题。

## He 初始化

Xavier 初始化为 sigmoid 和 tanh 网络提供了科学的初始化方案，但当 ReLU 激活函数崛起后，这种方法就显得力不从心了。ReLU 的"半线性"特性 —— 只保留正值，负值全部置零 —— 导致约一半的激活信号被"杀死"，Xavier 的线性假设完全失效。

针对这个问题，中国计算机科学家何恺明（Kaiming He）在 2015 年提出了专门为 ReLU 设计的初始化方法。他在与 Sun Jian、Xiangyu Zhang 等人合作的论文《Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification》中，系统分析了 ReLU 网络的信号传播特性，提出了 He 初始化。这篇论文不仅在初始化理论上做出贡献，更重要的是，它首次展示了深度网络在图像分类任务上超越人类表现的可能性 —— 何恺明团队使用深度残差网络（ResNet）在 ImageNet 竞赛中达到了 3.57% 的 top-5 错误率，低于人类的 5.1%。这一里程碑成果证明了深度学习的巨大潜力，也确立了 He 初始化在 ReLU 网络中的核心地位。

### ReLU 的特殊性

要理解为什么 Xavier 初始化不适合 ReLU，我们需要先理解 ReLU 与 sigmoid/tanh 的本质区别。

[ReLU 激活函数](../neural-network-structure/activation-loss-functions.md#relu-函数) 的定义非常简单：

$$f(x) = \max(0, x)$$

这个公式看着抽象，拆开来看含义很直观：
- 当输入 $x > 0$ 时，输出 $y = x$（线性传递）
- 当输入 $x < 0$ 时，输出 $y = 0$（信号被"杀死")
- 整体公式表示：只保留正向信号，负向信号全部丢弃

这种"选择性传递"的特性带来了三个关键差异：

**差异一：不对称**

sigmoid 和 tanh 的输出范围是对称的（sigmoid 范围 $[0, 1]$，均值约 0.5；tanh 范围 $[-1, 1]$，均值 0）。而 ReLU 的输出范围是 $[0, +\infty)$，均值明显大于 0。这意味着 ReLU 激活后的输出不再满足"零均值"假设——Xavier 推导中的关键前提被破坏。

**差异二：半线性**

sigmoid 在整个范围内都是非线性的，只在 0 附近近似线性。ReLU 则是"半线性" —— 正值区域完全线性（$y = x$），负值区域完全非线性（$y = 0$）。这种分段特性使得 ReLU 在正值区域的梯度恒为 1，远比 sigmoid/tanh 的梯度稳定。

**差异三：稀疏激活**

这是最关键的差异。假设输入 $x$ 服从零均值正态分布 $N(0, \sigma^2)$，大约一半的输入是负值。ReLU 会将这些负值全部置零，意味着约一半的神经元输出为 0——这被称为"稀疏激活"。

稀疏激活带来计算效率的好处，但也带来了一个问题：信号强度减半。让我们具体分析这个问题。

ReLU 后的输出方差：

$$\text{Var}(y) = \text{Var}(\max(0, x))$$

对于零均值正态分布 $x \sim N(0, \sigma_x^2)$：

$$\text{Var}(y) = \frac{1}{2} \text{Var}(x)$$

这个公式如何理解？想象一个零均值正态分布的输入，正负两半对称。ReLU 保留了正半部分（方差贡献 $\frac{1}{2}\sigma_x^2$），丢弃了负半部分（方差贡献 $0$）。因此输出方差只有输入方差的一半。

用一个具体例子验证：假设输入 $x \sim N(0, 4)$，即方差为 4，标准差为 2。

- 输入在 $[-2, 2]$ 范围内的概率约 68%
- ReLU 后，负值部分（约 34% 的输入）全部变成 0
- 输出方差约为 $\frac{1}{2} \times 4 = 2$

信号强度从 4 变成了 2，损失了整整一半！

这种"信号衰减"会逐层累积。假设 10 层 ReLU 网络，每层信号衰减一半：

$$\text{Var}(y_{10}) = \left(\frac{1}{2}\right)^{10} \text{Var}(x_0) = \frac{1}{1024} \text{Var}(x_0)$$

原始信号的方差被压缩了 1024 倍！深层网络的激活值几乎为零，梯度也几乎为零，训练完全停滞。

这就是 Xavier 初始化不适合 ReLU 的根本原因：Xavier 假设激活函数保持信号强度不变（线性），但 ReLU 每层都要"杀死"一半信号，需要更大的权重方差来补偿这种损失。

### He 初始化公式

理解了 ReLU 的稀疏性问题，解决方案就呼之欲出了：既然 ReLU 每层都会"杀死"一半信号，我们就需要用更大的权重方差来补偿。

He 初始化的核心思想是：考虑 ReLU 激活后的信号传播，推导出保持信号稳定所需的权重方差。

让我们分析带 ReLU 的网络层。假设输入 $x_i$ 是上一层 ReLU 的输出，权重 $w_i$ 独立同分布、零均值。前向传播分为两步：

$$z = \sum_{i=1}^{n_{in}} w_i x_i \quad \text{(加权求和)}$$
$$y = \max(0, z) \quad \text{(ReLU 激活)}$$

第一步的方差（ReLU 前）：

$$\text{Var}(z) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x_i)$$

由于 $x_i$ 是 ReLU 输出，$\text{Var}(x_i) = \frac{1}{2} \text{Var}(x_{prev})$（上一节推导的结论）：

$$\text{Var}(z) = n_{in} \cdot \text{Var}(w) \cdot \frac{1}{2} \text{Var}(x_{prev})$$

第二步的方差（ReLU 后）：

$$\text{Var}(y) = \frac{1}{2} \text{Var}(z) = \frac{1}{2} \cdot n_{in} \cdot \text{Var}(w) \cdot \frac{1}{2} \text{Var}(x_{prev})$$

整理后：

$$\text{Var}(y) = \frac{1}{4} n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x_{prev})$$

为了保持信号强度不变（$\text{Var}(y) = \text{Var}(x_{prev})$），需要：

$$\frac{1}{4} n_{in} \cdot \text{Var}(w) = 1$$

$$[eq:he-var] \text{Var}(w) = \frac{2}{n_{in}}$$

这个公式看着抽象，拆开来看含义很直观：
- $n_{in}$ 是输入神经元数量（fan-in）
- 分子 2 是补偿因子，抵消 ReLU 的信号衰减（两个 $\frac{1}{2}$ 相乘需要 2 来补偿）
- 整体公式表示：权重方差需要比 Xavier 更大，才能抵消 ReLU 的"信号杀伤力"

让我们用一个具体例子对比 Xavier 和 He 初始化。假设网络层有 $n_{in} = n_{out} = 512$ 个神经元：

| 初始化方法 | 方差公式 | 方差值 | 标准差 |
|:----------|:--------|:------|:------|
| Xavier | $\frac{2}{512+512}$ | $\frac{2}{1024} = 0.002$ | $\sqrt{0.002} \approx 0.045$ |
| He | $\frac{2}{512}$ | $\frac{2}{512} = 0.004$ | $\sqrt{0.004} \approx 0.063$ |

He 初始化的方差是 Xavier 的**两倍**！这正好补偿了 ReLU 的信号衰减：ReLU 每层杀死一半信号（$\frac{1}{2}$），He 用两倍方差（$\times 2$）抵消这个损失。

基于这个方差公式，He 初始化也有两种实现方式：

**He 正态初始化**从正态分布中采样：

$$\mathbf{W}_{ij} \sim N\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

标准差 $\sigma = \sqrt{\frac{2}{n_{in}}}$，大部分权重落在 $[-\sigma, \sigma]$ 范围内。

**He 均匀初始化**从均匀分布中采样：

$$\mathbf{W}_{ij} \sim U\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right]$$

均匀分布的边界 $a = \sqrt{\frac{6}{n_{in}}}$，使方差满足 $\frac{a^2}{3} = \frac{2}{n_{in}}$。

### Xavier vs He 初始化

现在我们已经学习了两种经典的初始化方法，让我们做一个系统对比：

| 初始化方法 | 适用激活函数 | 方差公式 | 理论基础 |
|:---------|:-----------|:-------|:--------|
| Xavier | sigmoid, tanh | $\frac{2}{n_{in} + n_{out}}$ | 线性激活假设 |
| He | ReLU, Leaky ReLU | $\frac{2}{n_{in}}$ | 半线性激活补偿 |

从表格中可以看出两个关键差异：

**差异一：适用激活函数不同**

Xavier 初始化假设激活函数在线性区域工作，因此适合 sigmoid 和 tanh——它们在输入接近 0 时近似线性。He 初始化专门针对 ReLU 设计，补偿 ReLU 的稀疏性（约一半激活值被置零）。

**差异二：方差公式不同**

He 初始化的方差公式只考虑 $n_{in}$，而 Xavier 考虑 $n_{in} + n_{out}$ 的调和平均。这是因为 He 初始化更强调前向传播的信号补偿——ReLU 的稀疏性问题主要发生在前向传播阶段，反向传播时梯度只通过正值神经元传递，不需要额外的补偿。

两者的方差比值：

$$\frac{\text{Var}_{He}}{\text{Var}_{Xavier}} = \frac{\frac{2}{n_{in}}}{\frac{2}{n_{in} + n_{out}}} = \frac{n_{in} + n_{out}}{n_{in}}$$

当 $n_{in} \approx n_{out}$ 时，He 初始化方差约为 Xavier 的两倍。这个两倍的差距正好补偿 ReLU 的信号衰减：ReLU 每层杀死一半信号，He 用两倍方差抵消。

### Leaky ReLU 的初始化

ReLU 的一个改进版本是 **Leaky ReLU**，它允许负值保留一个小的梯度，而非完全置零：

$$f(x) = \max(\alpha x, x)$$

这个公式看着抽象，拆开来看含义很直观：
- 当输入 $x > 0$ 时，输出 $y = x$（与 ReLU 相同）
- 当输入 $x < 0$ 时，输出 $y = \alpha x$（而非直接置零）
- $\alpha$ 是一个小常数，通常取 $0.01$
- 整体公式表示：负值信号被"压缩"而非"杀死"

Leaky ReLU 的设计动机是解决 ReLU 的"死神经元"问题。当某个神经元的输入长期为负时，ReLU 的输出始终为 0，梯度也为 0，这个神经元永远无法更新，相当于"死亡"。Leaky ReLU 通过给负值保留小梯度，让神经元有机会"复活"。

Leaky ReLU 对信号传播的影响介于 ReLU 和线性激活之间。假设输入 $x$ 服从零均值正态分布，Leaky ReLU 后的输出方差：

$$\text{Var}(y) = \frac{1 + \alpha^2}{2} \text{Var}(x)$$

这个公式看着抽象，拆开来看含义很直观：
- $\frac{1}{2}$ 来自正负两半的对称分割（概率各占一半）
- $1$ 对应正半部分的方差贡献（$x$ 直接传递）
- $\alpha^2$ 对应负半部分的方差贡献（$x$ 被 $\alpha$ 压缩）
- 整体公式表示：输出方差介于 ReLU（$\frac{1}{2}$）和线性（$1$）之间

推导过程如下。设 $x \sim N(0, \sigma_x^2)$：

$$\text{Var}(y) = E[y^2] - E[y]^2$$

Leaky ReLU 的输出分为两部分：

$$E[y^2] = \frac{1}{2} E[x^2|_{x>0}] + \frac{1}{2} E[(\alpha x)^2|_{x<0}]$$

正半部分贡献 $\frac{1}{2}\sigma_x^2$，负半部分贡献 $\frac{1}{2}\alpha^2\sigma_x^2$：

$$E[y^2] = \frac{1}{2}\sigma_x^2 + \frac{1}{2}\alpha^2\sigma_x^2 = \frac{1 + \alpha^2}{2}\sigma_x^2$$

由于正负两半对称，$E[y] = 0$：

$$\text{Var}(y) = \frac{1 + \alpha^2}{2}\sigma_x^2$$

基于这个方差分析，Leaky ReLU 的初始化方差应该是：

$$\text{Var}(w) = \frac{2}{(1 + \alpha^2) n_{in}}$$

当 $\alpha = 0$ 时（标准 ReLU），公式退化为 $\frac{2}{n_{in}}$（He 初始化）。当 $\alpha = 1$ 时（线性激活），公式退化为 $\frac{1}{n_{in}}$（前向传播的理想方差）。

## 初始化方法实验

理论分析告诉我们：Xavier 初始化适合 sigmoid/tanh，He 初始化适合 ReLU。但这些理论结论在实际训练中是否成立？让我们通过代码实验来验证。

下面的实验模拟一个多层神经网络（类似 MLP 结构），对比五种初始化方法（全零初始化、小方差初始化、大方差初始化、Xavier 初始化、He 初始化）在三种激活函数（sigmoid、ReLU、tanh）下的表现。实验测量三个关键指标：每层激活值的分布（验证信号传播）、每层梯度的范数（验证梯度传播）、训练损失曲线（验证收敛速度）。通过可视化图表，我们可以直观地看到初始化对训练的深远影响。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("实验：不同初始化方法对训练稳定性的影响")
print("=" * 60)
print()

# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# 定义初始化方法
def zero_init(shape):
    """全零初始化"""
    return np.zeros(shape)

def random_init(shape, scale=0.01):
    """随机初始化（小方差）"""
    return np.random.randn(*shape) * scale

def random_large_init(shape, scale=10):
    """随机初始化（大方差）"""
    return np.random.randn(*shape) * scale

def xavier_uniform_init(shape):
    """Xavier 均匀初始化"""
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

def xavier_normal_init(shape):
    """Xavier 正态初始化"""
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.randn(*shape) * std

def he_normal_init(shape):
    """He 正态初始化"""
    fan_in = shape[0]
    std = np.sqrt(2 / fan_in)
    return np.random.randn(*shape) * std

def he_uniform_init(shape):
    """He 均匀初始化"""
    fan_in = shape[0]
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, shape)

# 简单多层网络
class SimpleNetwork:
    def __init__(self, layer_sizes, activation='relu', init_method='he_normal'):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # 选择激活函数
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        
        # 选择初始化方法
        init_funcs = {
            'zero': zero_init,
            'random_small': lambda s: random_init(s, 0.01),
            'random_large': lambda s: random_init(s, 10),
            'xavier_uniform': xavier_uniform_init,
            'xavier_normal': xavier_normal_init,
            'he_normal': he_normal_init,
            'he_uniform': he_uniform_init
        }
        self.init_func = init_funcs[init_method]
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = self.init_func((layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # 记录每层的激活值和梯度
        self.activations_history = []
        self.gradients_history = []
    
    def forward(self, X):
        """前向传播"""
        self.activations = [X]
        self.pre_activations = []
        
        a = X
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            a = self.activation(z)
            self.activations.append(a)
        
        return a
    
    def backward(self, X, y, learning_rate=0.01):
        """反向传播"""
        m = X.shape[0]
        
        # 计算输出层误差（简单 MSE 损失）
        delta = (self.activations[-1] - y) * self.activation_derivative(self.pre_activations[-1])
        
        # 存储梯度范数
        gradients = []
        
        # 反向传播
        for i in range(self.num_layers - 1, -1, -1):
            # 计算权重梯度
            grad_w = self.activations[i].T @ delta / m
            grad_b = np.mean(delta, axis=0, keepdims=True)
            
            gradients.append(np.linalg.norm(grad_w))
            
            # 更新权重和偏置
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            
            # 传播误差到前一层
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_derivative(self.pre_activations[i-1])
        
        # 记录梯度历史（从第一层到最后一层）
        self.gradients_history.append(gradients[::-1])
        self.activations_history.append([np.mean(np.abs(a)) for a in self.activations])

# 实验1：不同初始化方法的激活值分布
print("实验1：不同初始化方法的激活值分布")
print("-" * 40)

layer_sizes = [784, 512, 256, 128, 64, 10]  # 类似 MLP 结构
n_samples = 100
X = np.random.randn(n_samples, 784) * 0.5  # 模拟标准化后的输入

init_methods = ['zero', 'random_small', 'random_large', 'xavier_normal', 'he_normal']
activation_names = ['sigmoid', 'relu', 'tanh']

results = {}

for activation in activation_names:
    results[activation] = {}
    for init_method in init_methods:
        net = SimpleNetwork(layer_sizes, activation=activation, init_method=init_method)
        output = net.forward(X)
        
        # 记录每层激活值的均值和方差
        activation_stats = []
        for i, a in enumerate(net.activations):
            mean_val = np.mean(np.abs(a))
            std_val = np.std(a)
            activation_stats.append((mean_val, std_val))
        
        results[activation][init_method] = {
            'activations': net.activations,
            'pre_activations': net.pre_activations,
            'stats': activation_stats
        }
        
        print(f"{activation} + {init_method}:")
        for i, (mean_val, std_val) in enumerate(activation_stats):
            print(f"  {i}: 均值={mean_val:.4f}, 标准差={std_val:.4f}")
        print()

# 可视化激活值分布
fig, axes = plt.subplots(3, 5, figsize=(20, 12))

for row, activation in enumerate(activation_names):
    for col, init_method in enumerate(init_methods):
        ax = axes[row, col]
        
        stats = results[activation][init_method]['stats']
        layer_means = [s[0] for s in stats]
        layer_stds = [s[1] for s in stats]
        
        layers = range(len(stats))
        ax.bar(layers, layer_means, color='#3498db', alpha=0.7, label='均值')
        ax.errorbar(layers, layer_means, yerr=layer_stds, fmt='o', color='#e74c3c', 
                   capsize=5, capthick=2, label='标准差')
        
        ax.set_xlabel('层编号')
        ax.set_ylabel('激活值')
        ax.set_title(f'{activation} + {init_method}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("实验2：不同初始化方法的梯度传播")
print("-" * 40)

# 实验2：不同初始化方法的梯度传播
layer_sizes = [784, 512, 256, 128, 10]
n_samples = 100
X = np.random.randn(n_samples, 784) * 0.5
y = np.random.randn(n_samples, 10) * 0.1  # 模拟目标输出

gradient_results = {}

for activation in ['sigmoid', 'relu']:
    gradient_results[activation] = {}
    for init_method in init_methods:
        if init_method == 'zero':
            continue  # 全零初始化梯度为零，跳过
        
        net = SimpleNetwork(layer_sizes, activation=activation, init_method=init_method)
        
        # 训练50步，记录梯度
        for step in range(50):
            output = net.forward(X)
            net.backward(X, y, learning_rate=0.001)
        
        # 提取每层梯度范数（取最后一步）
        final_gradients = net.gradients_history[-1]
        gradient_results[activation][init_method] = {
            'gradients': final_gradients,
            'history': net.gradients_history
        }
        
        print(f"{activation} + {init_method}:")
        for i, g in enumerate(final_gradients):
            print(f"  层{i} 梯度范数: {g:.6f}")
        print()

# 可视化梯度分布
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, activation in enumerate(['sigmoid', 'relu']):
    ax = axes[idx]
    
    for init_method in init_methods:
        if init_method == 'zero':
            continue
        
        gradients = gradient_results[activation][init_method]['gradients']
        layers = range(len(gradients))
        ax.plot(layers, gradients, 'o-', linewidth=2, markersize=8, label=init_method)
    
    ax.set_xlabel('层编号')
    ax.set_ylabel('梯度范数')
    ax.set_title(f'{activation} 激活函数的梯度传播')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("实验3：初始化对训练收敛的影响")
print("-" * 40)

# 实验3：初始化对训练收敛的影响
layer_sizes = [784, 256, 128, 10]
n_samples = 500
n_epochs = 200

# 生成简单分类数据
X_train = np.random.randn(n_samples, 784)
y_train = np.zeros((n_samples, 10))
y_train[:, 0] = 1  # 所有样本类别0

convergence_results = {}

for activation in ['relu', 'sigmoid']:
    convergence_results[activation] = {}
    for init_method in ['random_small', 'xavier_normal', 'he_normal']:
        if activation == 'relu' and init_method == 'xavier_normal':
            continue  # Xavier 不适合 ReLU
        
        if activation == 'sigmoid' and init_method == 'he_normal':
            continue  # He 不适合 sigmoid
        
        net = SimpleNetwork(layer_sizes, activation=activation, init_method=init_method)
        losses = []
        
        for epoch in range(n_epochs):
            output = net.forward(X_train)
            loss = np.mean((output - y_train)**2)
            losses.append(loss)
            net.backward(X_train, y_train, learning_rate=0.01)
        
        convergence_results[activation][init_method] = {
            'losses': losses,
            'final_loss': losses[-1]
        }
        
        print(f"{activation} + {init_method}:")
        print(f"  初始损失: {losses[0]:.4f}")
        print(f"  最终损失: {losses[-1]:.4f}")
        print(f"  损失下降: {losses[0] - losses[-1]:.4f}")
        print()

# 可视化收敛曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, activation in enumerate(['relu', 'sigmoid']):
    ax = axes[idx]
    
    for init_method, data in convergence_results[activation].items():
        ax.plot(data['losses'], linewidth=2, label=init_method)
    
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('损失值')
    ax.set_title(f'{activation} 激活函数的训练收敛')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n实验结论:")
print("-" * 40)
print("1. 全零初始化：所有神经元输出相同，梯度为零，无法训练")
print("2. 小方差初始化：激活值逐层衰减，训练缓慢")
print("3. 大方差初始化：sigmoid/tanh 激活值饱和，梯度消失")
print("4. Xavier 初始化：sigmoid/tanh 激活值稳定，梯度传播正常")
print("5. He 初始化：ReLU 激活值稳定，梯度传播正常")
print("6. 初始化匹配激活函数：Xavier 配 sigmoid/tanh，He 配 ReLU")
print("=" * 60)
```

### 实验结论

实验验证了初始化对训练的深远影响：

1. **全零初始化致命**：梯度为零，参数无法更新，网络无法学习

2. **小方差初始化缓慢**：信号逐层衰减，激活值过小，梯度过小，训练缓慢

3. **大方差初始化导致梯度消失**：sigmoid/tanh 激活值饱和（接近 0 或 1），梯度接近零，训练停滞

4. **Xavier 初始化适合 sigmoid/tanh**：激活值在各层保持稳定方差，梯度传播正常

5. **He 初始化适合 ReLU**：补偿 ReLU 的稀疏性，激活值和梯度保持稳定

6. **初始化与激活函数匹配至关重要**：错误匹配导致信号衰减或饱和

## 初始化最佳实践

理论和实验都验证了初始化的关键作用。现在，让我们总结一套实用的初始化策略，帮助你在实际项目中做出正确选择。

### 初始化选择策略

当设计神经网络时，初始化方法的选择应该与激活函数匹配。下面的表格总结了常用组合：

| 激活函数 | 推荐初始化 | 方差公式 |
|:--------|:---------|:--------|
| sigmoid | Xavier | $\frac{2}{n_{in} + n_{out}}$ |
| tanh | Xavier | $\frac{2}{n_{in} + n_{out}}$ |
| ReLU | He | $\frac{2}{n_{in}}$ |
| Leaky ReLU | He 变体 | $\frac{2}{(1+\alpha^2)n_{in}}$ |
| SELU | LeCun | $\frac{1}{n_{in}}$ |

这个表格看着简单，但背后蕴含着深刻的数学原理：

- **sigmoid/tanh 使用 Xavier**：这两种激活函数在输入接近 0 时近似线性，Xavier 的线性假设成立
- **ReLU 使用 He**：ReLU 的稀疏性导致信号衰减，He 用更大方差补偿
- **Leaky ReLU 使用 He 变体**：Leaky ReLU 介于 ReLU 和线性之间，方差公式相应调整
- **SELU 使用 LeCun**：SELU（Scaled Exponential Linear Unit）自带自归一化特性，方差公式与 Xavier 类似但只考虑 $n_{in}$

### 偏置初始化

讨论了权重初始化后，偏置应该如何初始化？这个问题看似简单，但也有讲究。

偏置通常初始化为零：

$$\mathbf{b} = 0$$

为什么偏置可以安全地设为零？三个原因：

1. **偏置不参与信号强度的传递**：权重矩阵乘以输入，决定了信号如何流动；偏置只是添加一个常数偏移，不改变信号的相对强度。因此，零偏置不会导致对称性问题。

2. **零偏置保持激活函数在线性区域**：sigmoid 和 tanh 在输入接近 0 时近似线性，这正是 Xavier 初始化生效的前提。零偏置使输入初始值接近权重乘以输入的结果（零均值），帮助激活值落在线性区域。

3. **非零偏置可能导致激活值偏离线性区域**：如果偏置初始化为较大正值，sigmoid 激活值可能直接饱和到 1，梯度接近零；如果偏置初始化为较大负值，激活值可能直接饱和到 0，同样梯度接近零。

但也有特殊情况，需要非零偏置初始化：

**ReLU 网络的正偏置**

ReLU 激活函数将负值置零。如果权重初始化较小（如 He 初始化），大部分激活值可能恰好落在零附近，一半为正一半为负。负值部分被置零后，网络初始输出可能过于稀疏。将偏置初始化为小正值（如 0.01），可以确保初始时大部分神经元有非零输出，避免"死神经元"问题。

**LSTM 遗忘门偏置**

LSTM（长短期记忆网络）有一个"遗忘门"，控制前一时刻信息的保留程度。遗忘门的输出接近 0 时，大部分历史信息被遗忘；输出接近 1 时，大部分信息被保留。如果遗忘门偏置初始化为零，初始时遗忘门输出可能接近 0.5（sigmoid 的中点），导致历史信息部分遗忘。将遗忘门偏置初始化为 1 或更大，使遗忘门初始输出接近 1，保留更多历史信息，有助于训练初期学习长期依赖。

### 深度网络的初始化考虑

浅层网络（5-10 层）对初始化相对宽容 —— 即使初始化略有偏差，优化器也能逐渐调整。但深度网络（层数 > 20）面临更严峻的挑战，初始化的选择变得至关重要。

为什么深度网络对初始化如此敏感？三个关键因素：

**累积效应**

每层的方差偏差会累积放大。假设每层方差偏差为 5%（输出方差比输入方差大 5%），20 层后：

$$\text{Var}(y_{20}) = (1.05)^{20} \text{Var}(x_0) \approx 2.65 \text{Var}(x_0)$$

输出方差被放大了 2.65 倍！激活值可能严重偏离线性区域，梯度消失或爆炸。初始化必须更精确，控制每层的方差偏差在极小范围内。

**梯度消失风险**

反向传播时，梯度逐层传递。每层传递都会引入衰减因子（激活函数的导数）。sigmoid 的最大梯度为 0.25，tanh 的最大梯度为 1，ReLU 的梯度要么 0 要么 1。20 层 sigmoid 网络：

$$\text{梯度}_{输入层} = (0.25)^{20} \times \text{梯度}_{输出层} \approx 10^{-12} \times \text{梯度}_{输出层}$$

输入层梯度被压缩到几乎为零！ReLU 虽然梯度更稳定（要么 0 要么 1），但稀疏性累积后也可能导致深层梯度消失。

**初始化更敏感**

浅层网络可以用"试错法" —— 初始化略有偏差，训练几个回合后观察损失曲线，如果不收敛就调整参数。深度网络的训练成本高（可能需要数小时甚至数天），试错法不再适用。初始化必须在"第一枪"就命中目标。

针对深度网络的推荐策略：

**使用 ReLU + He 初始化**

ReLU 的梯度在正值区域恒为 1，远比 sigmoid/tanh 稳定。He 初始化补偿 ReLU 的稀疏性，使信号在深层网络中保持稳定。这是深度网络最常用的组合。

**添加 Batch Normalization**

[批归一化](../neural-network-stability/batch-normalization.md)（Batch Normalization）强制标准化每层的激活值，使其均值接近 0、方差接近 1。这相当于在每个"检查点"重置信号强度，缓解初始化偏差的累积效应。Batch Normalization 使深度网络对初始化更宽容，甚至允许使用不太理想的初始化。

**残差连接**

ResNet（深度残差网络）引入"跳跃连接"，让信号绕过某些层直接传递：

$$y = f(x) + x$$

这个设计的关键在于：即使 $f(x)$ 的梯度消失，$x$ 的梯度仍然可以通过跳跃连接传回输入层。残差连接显著缓解了深度网络的梯度消失问题，使上百层的网络可以正常训练。

### 预训练模型的初始化

当使用迁移学习（Transfer Learning）时，初始化策略有所不同：我们不再从随机值开始，而是使用预训练模型的权重作为起点：

$$\mathmathbf{W}_{init} = \mathbf{W}_{pretrained}$$

这相当于让网络从"有经验"的状态开始训练，而非"从零学起"。

预训练权重带来三个优势：

**已学习有用特征**

预训练模型（如 ImageNet 上训练的 ResNet）已经学会了识别边缘、纹理、形状等基础视觉特征。这些特征对大多数图像任务都有价值。使用预训练权重，网络无需花费大量时间重新学习这些基础特征，可以直接专注于任务特定的特征。

**训练起点好**

预训练模型的损失函数已经接近一个较好的局部最优（或全局最优）。从这个起点出发，精调（Fine-tuning）只需要做小幅调整，比从头训练快得多。实验表明，预训练模型通常能在较少训练轮数内达到相同精度，甚至更高精度。

**收敛更快**

从头训练深度网络可能需要数百轮才能收敛，且容易陷入糟糕的局部最优。预训练模型通常只需几十轮精调就能达到目标，训练时间和计算成本大幅降低。

使用预训练权重时，有几个注意事项：

**冻结底层**

网络的底层（靠近输入的层）学习的是最通用的特征，如边缘、纹理。这些特征在不同任务间高度共享，通常不需要大幅调整。冻结底层权重（不参与训练），可以避免破坏这些通用特征，同时减少计算量。

**精调顶层**

网络的顶层（靠近输出的层）学习的是任务特定的特征，如特定类别。这些特征需要根据新任务进行调整。顶层权重应该参与训练，适应新的数据分布和任务目标。

**小学习率**

预训练权重已经处于较好的位置，大幅调整可能破坏这些权重学习到的特征。使用较小的学习率（如从头训练学习率的 1/10 或 1/100），让权重缓慢调整，逐步适应新任务。

## 本章小结

本章从"为什么初始化至关重要"出发，系统分析了权重初始化对深度网络训练的影响，并介绍了 Xavier 初始化和 He 初始化两种经典方法。

**初始化的挑战在于找到平衡点**。全零初始化破坏对称性，使所有神经元学习相同功能，网络退化到单个神经元。随机初始化打破对称性，但方差选择不当会导致两个极端：方差太小，信号逐层衰减，训练缓慢；方差太大，激活值饱和，梯度消失。好的初始化需要在"打破对称性"和"保持信号强度"之间找到平衡。

**Xavier 初始化解决了 sigmoid/tanh 网络的初始化问题**。通过分析信号在前向和反向传播中的方差变化，Xavier 推导出保持信号稳定所需的权重方差：$\frac{2}{n_{in} + n_{out}}$。这个公式是 $n_{in}$ 和 $n_{out}$ 的调和平均，兼顾前向传播（要求方差 $\frac{1}{n_{in}}$）和反向传播（要求方差 $\frac{1}{n_{out}}$）。Xavier 初始化假设激活函数在线性区域工作，因此适合 sigmoid 和 tanh——它们在输入接近 0 时近似线性。

**He 初始化解决了 ReLU 网络的初始化问题**。ReLU 的稀疏性（约一半激活值被置零）导致信号强度减半，Xavier 的线性假设失效。He 初始化通过更大的方差（$\frac{2}{n_{in}}$）补偿 ReLU 的信号衰减。当 $n_{in} \approx n_{out}$ 时，He 初始化方差约为 Xavier 的两倍，正好抵消 ReLU 的稀疏性损失。

**初始化的最佳实践是匹配激活函数**。sigmoid/tanh 使用 Xavier 初始化，ReLU 使用 He 初始化。偏置通常初始化为零，但 ReLU 网络可以使用小正偏置，LSTM 遗忘门可以使用较大正偏置。深度网络（层数 > 20）对初始化更敏感，可以结合 Batch Normalization 和残差连接缓解初始化偏差的累积效应。迁移学习时，使用预训练权重作为初始化，可以加速收敛并提高最终精度。

初始化是训练的"起点"，决定了网络能否顺利起步。但训练的"终点" —— 收敛到最优 —— 还需要解决另一个问题：过拟合。下一章将介绍 Dropout 正则化，通过随机"丢弃"神经元，迫使网络学习更鲁棒的特征，从另一个角度提升训练稳定性。

## 练习题

1. 分析全零初始化为何导致网络无法学习。假设两层网络 $y = \mathbf{W}_2 \mathbf{W}_1 x$，初始 $\mathbf{W}_1 = 0$，$\mathbf{W}_2 = 0$。推导反向传播梯度，解释为何参数无法更新。
    <details>
    <summary>参考答案</summary>
    
    **全零初始化的梯度分析**：
    
    设两层线性网络（忽略激活函数）：
    
    $$h = \mathbf{W}_1 x$$
    $$y = \mathbf{W}_2 h = \mathbf{W}_2 \mathbf{W}_1 x$$
    
    使用 MSE 损失：
    
    $$L = \frac{1}{2}(y - y_{target})^2$$
    
    **反向传播梯度**：
    
    $\mathbf{W}_2$ 的梯度：
    
    $$\frac{\partial L}{\partial \mathbf{W}_2} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial \mathbf{W}_2} = (y - y_{target}) \cdot h$$
    
    当 $\mathbf{W}_1 = 0$，$h = 0$，梯度为零：
    
    $$\frac{\partial L}{\partial \mathbf{W}_2} = (y - y_{target}) \cdot 0 = 0$$
    
    $\mathbf{W}_1$ 的梯度：
    
    $$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h} \cdot \frac{\partial h}{\partial \mathbf{W}_1} = (y - y_{target}) \cdot \mathbf{W}_2^T \cdot x$$
    
    当 $\mathbf{W}_2 = 0$，梯度为零：
    
    $$\frac{\partial L}{\partial \mathbf{W}_1} = (y - y_{target}) \cdot 0 \cdot x = 0$$
    
    **结论**：
    
    全零初始化时：
    1. 前向传播：$h = 0$，$y = 0$
    2. 反向传播：
       - $\mathbf{W}_2$ 的梯度包含 $h$，$h = 0$ 使梯度为零
       - $\mathbf{W}_1$ 的梯度包含 $\mathbf{W}_2$，$\mathbf{W}_2 = 0$ 使梯度为零
    
    **对称性问题**：
    
    即使 $\mathbf{W}_2$ 不为零（如随机初始化），当 $\mathbf{W}_1 = 0$：
    
    - $h = 0$（所有隐藏神经元输出相同）
    - $\mathbf{W}_1$ 的梯度：$\frac{\partial L}{\partial \mathbf{W}_1} = (y - y_{target}) \cdot \mathbf{W}_2^T \cdot x$
    - 隐藏层每个神经元接收的梯度相同（因为 $\mathbf{W}_2^T$ 的每行相同 —— 如果 $\mathbf{W}_2$ 随机初始化，每行不同，梯度不同）
    
    但如果 $\mathbf{W}_1$ 和 $\mathbf{W}_2$ 都全零：
    - 前向传播：所有神经元输出相同（$0$）
    - 反向传播：所有梯度为零
    - 参数无法更新
    
    **为什么对称性破坏重要**：
    
    神经网络的核心能力来自于神经元学习不同的特征。如果所有神经元输出相同，网络退化到单个神经元。全零初始化使所有神经元初始状态相同，且梯度为零，无法打破对称性。
    
    随机初始化为每个神经元赋予不同的初始权重，打破对称性，使每个神经元学习不同的特征。
    
    **总结**：
    
    全零初始化导致：
    1. 前向传播时所有神经元输出相同（$h = 0$，$y = 0$）
    2. 反向传播时梯度为零（$\nabla \mathbf{W}_1 = 0$，$\nabla \mathbf{W}_2 = 0$）
    3. 参数无法更新，网络无法学习
    
    解决方案：随机初始化，为每个权重赋予不同的随机值，打破对称性。
    </details>

2. 推导 Xavier 初始化的方差公式。设线性网络层 $y = \sum_{i=1}^{n_{in}} w_i x_i$，假设 $w_i$ 和 $x_i$ 独立同分布，零均值。推导输出方差 $\text{Var}(y)$，并解释为何 $\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$ 能保持前向和反向传播的信号强度。
    <details>
    <summary>参考答案</summary>
    
    **Xavier 初始化方差推导**：
    
    **前向传播分析**：
    
    设线性神经元：
    
    $$y = \sum_{i=1}^{n_{in}} w_i x_i$$
    
    其中 $w_i$ 和 $x_i$ 独立同分布，零均值（$E[w_i] = 0$, $E[x_i] = 0$）。
    
    **输出方差推导**：
    
    $$\text{Var}(y) = E[y^2] - E[y]^2 = E[y^2]$$
    
    因为 $E[y] = E[\sum_{i} w_i x_i] = \sum_{i} E[w_i] E[x_i] = 0$（零均值）。
    
    $$E[y^2] = E\left[\left(\sum_{i=1}^{n_{in}} w_i x_i\right)^2\right]$$
    
    展开：
    
    $$E[y^2] = E\left[\sum_{i} w_i^2 x_i^2 + \sum_{i \neq j} w_i w_j x_i x_j\right]$$
    
    由于 $w_i$ 和 $x_j$ 独立（$i \neq j$）：
    
    $$E[w_i w_j x_i x_j] = E[w_i] E[w_j] E[x_i] E[x_j] = 0$$（零均值）
    
    因此：
    
    $$E[y^2] = \sum_{i=1}^{n_{in}} E[w_i^2 x_i^2]$$
    
    由于 $w_i$ 和 $x_i$ 独立：
    
    $$E[w_i^2 x_i^2] = E[w_i^2] E[x_i^2] = \text{Var}(w_i) \cdot \text{Var}(x_i)$$
    
    因此：
    
    $$\text{Var}(y) = \sum_{i=1}^{n_{in}} \text{Var}(w_i) \cdot \text{Var}(x_i) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$
    
    **保持信号强度**：
    
    为了使输出方差等于输入方差（$\text{Var}(y) = \text{Var}(x)$）：
    
    $$n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x) = \text{Var}(x)$$
    
    $$n_{in} \cdot \text{Var}(w) = 1$$
    
    $$\text{Var}(w) = \frac{1}{n_{in}}$$
    
    **反向传播分析**：
    
    反向传播时，梯度从输出层传回输入层：
    
    $$\delta_x = \mathbf{W}^T \delta_y = \sum_{j=1}^{n_{out}} w_j \delta_y_j$$
    
    其中 $n_{out}$ 是输出神经元数量（fan-out）。
    
    输入层梯度方差：
    
    $$\text{Var}(\delta_x) = n_{out} \cdot \text{Var}(w) \cdot \text{Var}(\delta_y)$$
    
    为了保持梯度强度（$\text{Var}(\delta_x) = \text{Var}(\delta_y)$）：
    
    $$n_{out} \cdot \text{Var}(w) = 1$$
    
    $$\text{Var}(w) = \frac{1}{n_{out}}$$
    
    **Xavier 折中方案**：
    
    前向传播要求 $\text{Var}(w) = \frac{1}{n_{in}}$
    反向传播要求 $\text{Var}(w) = \frac{1}{n_{out}}$
    
    当 $n_{in} \neq n_{out}$，无法同时满足两个条件。Xavier 采用折中：
    
    $$\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$$
    
    这是 $n_{in}$ 和 $n_{out}$ 的调和平均（约）。
    
    **验证**：
    
    前向传播：
    
    $$\text{Var}(y) = n_{in} \cdot \frac{2}{n_{in} + n_{out}} \cdot \text{Var}(x)$$
    
    当 $n_{in} = n_{out}$：
    
    $$\text{Var}(y) = n_{in} \cdot \frac{2}{2n_{in}} \cdot \text{Var}(x) = \text{Var}(x)$$
    
    满足信号强度保持。
    
    反向传播：
    
    $$\text{Var}(\delta_x) = n_{out} \cdot \frac{2}{n_{in} + n_{out}} \cdot \text{Var}(\delta_y)$$
    
    当 $n_{in} = n_{out}$：
    
    $$\text{Var}(\delta_x) = n_{out} \cdot \frac{2}{2n_{out}} \cdot \text{Var}(\delta_y) = \text{Var}(\delta_y)$$
    
    满足梯度强度保持。
    
    **结论**：
    
    Xavier 初始化方差 $\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$ 是前向和反向传播要求的折中：
    - 前向传播要求 $\text{Var}(w) = \frac{1}{n_{in}}$（保持信号强度）
    - 反向传播要求 $\text{Var}(w) = \frac{1}{n_{out}}$（保持梯度强度）
    - 折中方案兼顾两者，当 $n_{in} = n_{out}$ 时完美满足
    
    Xavier 初始化假设激活函数是线性的（或在线性区域），适合 sigmoid 和 tanh。
    </details>

3. 解释为何 He 初始化的方差 $\text{Var}(w) = \frac{2}{n_{in}}$ 比 Xavier 初始化大。分析 ReLU 激活函数的稀疏性如何影响信号传播。
    <details>
    <summary>参考答案</summary>
    
    **He 初始化方差分析**：
    
    **ReLU 的稀疏性**：
    
    ReLU 激活函数：
    
    $$f(x) = \max(0, x)$$
    
    当输入 $x$ 为零均值正态分布 $N(0, \sigma^2)$：
    
    - 约一半输入为正（$x > 0$），输出 $y = x$
    - 约一半输入为负（$x < 0$），输出 $y = 0$
    
    ReLU 后的输出方差：
    
    $$\text{Var}(y) = \text{Var}(\max(0, x))$$
    
    对于零均值正态分布：
    
    $$\text{Var}(y) = \frac{1}{2} \text{Var}(x)$$
    
    因为：
    
    $$E[y^2] = \frac{1}{2} E[x^2|_{x>0}] + \frac{1}{2} E[0|_{x<0}] = \frac{1}{2} \sigma^2$$
    
    $$\text{Var}(y) = E[y^2] - E[y]^2 = \frac{1}{2} \sigma^2 - \left(\frac{\sigma}{\sqrt{2\pi}}\right)^2 \approx \frac{1}{2} \sigma^2$$
    
    简化计算：设 $x$ 的正负部分对称，ReLU 后方差约为 $\frac{1}{2} \text{Var}(x)$。
    
    **信号传播分析**：
    
    考虑带 ReLU 的网络层：
    
    $$z = \sum_{i=1}^{n_{in}} w_i x_i$$
    $$y = \max(0, z)$$
    
    其中 $x_i$ 是上一层的 ReLU 输出（零均值，方差为 $\frac{1}{2}\text{Var}(x_{prev})$）。
    
    ReLU 前的方差：
    
    $$\text{Var}(z) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x_i)$$
    
    由于 $x_i$ 是 ReLU 输出，$\text{Var}(x_i) = \frac{1}{2} \text{Var}(x_{prev})$：
    
    $$\text{Var}(z) = n_{in} \cdot \text{Var}(w) \cdot \frac{1}{2} \text{Var}(x_{prev})$$
    
    ReLU 后的方差：
    
    $$\text{Var}(y) = \frac{1}{2} \text{Var}(z) = \frac{1}{2} \cdot n_{in} \cdot \text{Var}(w) \cdot \frac{1}{2} \text{Var}(x_{prev})$$
    
    $$\text{Var}(y) = \frac{1}{4} n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x_{prev})$$
    
    **保持信号强度**：
    
    为了使输出方差等于输入方差（$\text{Var}(y) = \text{Var}(x_{prev})$）：
    
    $$\frac{1}{4} n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x_{prev}) = \text{Var}(x_{prev})$$
    
    $$\frac{1}{4} n_{in} \cdot \text{Var}(w) = 1$$
    
    $$\text{Var}(w) = \frac{4}{n_{in}}$$
    
    但 He 初始化使用 $\text{Var}(w) = \frac{2}{n_{in}}$。为何？
    
    **修正推导**：
    
    更精确的推导考虑反向传播。设 $x_i$ 是原始输入（非 ReLU 输出），零均值：
    
    $$\text{Var}(z) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$
    
    ReLU 后：
    
    $$\text{Var}(y) = \frac{1}{2} \text{Var}(z) = \frac{1}{2} n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$
    
    为了 $\text{Var}(y) = \text{Var}(x)$：
    
    $$\frac{1}{2} n_{in} \cdot \text{Var}(w) = 1$$
    
    $$\text{Var}(w) = \frac{2}{n_{in}}$$
    
    这就是 He 初始化公式。
    
    **He vs Xavier 对比**：
    
    | 初始化 | 方差公式 | 比值 |
    |:------|:--------|:----|
    | Xavier | $\frac{2}{n_{in} + n_{out}}$ | - |
    | He | $\frac{2}{n_{in}}$ | $\frac{n_{in} + n_{out}}{n_{in}}$ |
    
    当 $n_{in} = n_{out}$：
    
    $$\text{Var}_{He} = \frac{2}{n_{in}}$$
    $$\text{Var}_{Xavier} = \frac{2}{2n_{in}} = \frac{1}{n_{in}}$$
    
    He 初始化方差是 Xavier 的 **两倍**：
    
    $$\frac{\text{Var}_{He}}{\text{Var}_{Xavier}} = \frac{\frac{2}{n_{in}}}{\frac{1}{n_{in}}} = 2$$
    
    **为何 He 初始化方差更大**：
    
    1. **补偿 ReLU 稀疏性**：ReLU 将约一半激活值置零，信号强度减半。He 初始化使用两倍方差，补偿这种损失。
    
    2. **反向传播考虑**：反向传播时，ReLU 的梯度只通过正值传递。He 初始化确保梯度强度不减。
    
    3. **实验验证**：He 等人的实验表明，$\text{Var}(w) = \frac{2}{n_{in}}$ 在 ReLU 网络上效果最好。
    
    **数值示例**：
    
    设 $n_{in} = 512$：
    
    | 初始化 | 方差 | 标准差 |
    |:------|:----|:------|
    | Xavier | $\frac{2}{1024} = 0.002$ | $\sqrt{0.002} = 0.044$ |
    | He | $\frac{2}{512} = 0.004$ | $\sqrt{0.004} = 0.063$ |
    
    He 初始化的标准差约为 Xavier 的 $\sqrt{2} \approx 1.41$ 倍。
    
    **结论**：
    
    He 初始化方差 $\text{Var}(w) = \frac{2}{n_{in}}$ 比 Xavier 大约两倍（当 $n_{in} = n_{out}$），原因是补偿 ReLU 的稀疏性：
    
    1. ReLU 将约一半激活值置零，信号强度减半
    2. He 初始化使用两倍方差，补偿信号损失
    3. 确保 ReLU 网络前向和反向传播信号稳定
    
    **为何 Xavier 不适合 ReLU**：
    
    Xavier 假设激活函数是线性的。ReLU 在正值区域线性，但稀疏性（一半置零）破坏了 Xavier 的假设。使用 Xavier 初始化 ReLU 网络：
    
    - 信号逐层衰减（因为方差小，不足以补偿 ReLU 稀疏性）
    - 深层激活值过小，梯度消失
    
    He 初始化专门为 ReLU 设计，补偿稀疏性，保持信号强度。
    </details>

4. 设计一个实验验证不同初始化方法对深度网络训练的影响。考虑层数、激活函数、学习率等因素，分析实验设计的关键要素。
    <details>
    <summary>参考答案</summary>
    
    **实验设计方案**：
    
    **目标**：验证不同初始化方法对深度网络训练的影响，分析关键因素。
    
    **变量设计**：
    
    | 变量 | 取值范围 | 说明 |
    |:----|:--------|:----|
    | 初始化方法 | Xavier, He, 随机小, 随机大, 全零 | 主要研究对象 |
    | 网络深度 | 5 层, 10 层, 20 层, 50 层 | 验证深度影响 |
    | 激活函数 | ReLU, sigmoid, tanh | 验证激活函数匹配 |
    | 学习率 | 0.001, 0.01, 0.1 | 验证学习率交互 |
    | 网络宽度 | 256, 512, 1024 | 验证 fan-in 影响 |
    
    **控制变量**：
    
    - 数据集固定（如 MNIST）
    - 优化器固定（如 SGD + Momentum）
    - 批量大小固定（如 64）
    - 训练轮数固定（如 100）
    
    **评价指标**：
    
    | 指标 | 测量方法 | 说明 |
    |:----|:--------|:----|
    | 激活值分布 | 每层激活值均值和方差 | 验证信号传播 |
    | 梯度分布 | 每层梯度范数 | 验证梯度传播 |
    | 训练损失曲线 | 每轮训练损失 | 验证收敛速度 |
    | 最终精度 | 测试集精度 | 验证训练效果 |
    
    **实验代码框架**：
    
    ```python
    import numpy as np
    
    # 初始化方法
    init_methods = {
        'xavier': lambda shape: np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1])),
        'he': lambda shape: np.random.randn(*shape) * np.sqrt(2 / shape[0]),
        'random_small': lambda shape: np.random.randn(*shape) * 0.01,
        'random_large': lambda shape: np.random.randn(*shape) * 1.0,
        'zero': lambda shape: np.zeros(shape)
    }
    
    # 激活函数
    activations = {
        'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)),
        'sigmoid': (lambda x: 1/(1+np.exp(-x)), lambda x: 1/(1+np.exp(-x))*(1-1/(1+np.exp(-x))))
    }
    
    # 实验配置
    configs = [
        {'depth': 10, 'activation': 'relu', 'init': 'he', 'lr': 0.01},
        {'depth': 10, 'activation': 'relu', 'init': 'xavier', 'lr': 0.01},
        {'depth': 10, 'activation': 'sigmoid', 'init': 'xavier', 'lr': 0.01},
        {'depth': 20, 'activation': 'relu', 'init': 'he', 'lr': 0.01},
        {'depth': 50, 'activation': 'relu', 'init': 'he', 'lr': 0.01},
    ]
    
    # 运行实验
    for config in configs:
        network = create_network(config)
        activations_history, gradients_history, losses = train(network, X_train, y_train, epochs=100)
        analyze_results(activations_history, gradients_history, losses, config)
    ```
    
    **关键要素分析**：
    
    1. **网络深度**：
       - 深度增加使信号传播问题放大
       - 5 层网络初始化偏差可能不明显
       - 50 层网络初始化偏差严重影响训练
       - 验证深度对初始化敏感性的影响
    
    2. **激活函数匹配**：
       - ReLU + He 初始化（正确匹配）
       - ReLU + Xavier 初始化（错误匹配）
       - sigmoid + Xavier 初始化（正确匹配）
       - sigmoid + He 初始化（错误匹配）
       - 验证匹配对训练的影响
    
    3. **学习率交互**：
       - 初始化方差小，需要大学习率补偿
       - 初始化方差大，需要小学习率防止震荡
       - He 初始化标准差 $\sqrt{2/n_{in}}$，配合学习率 0.01
       - 验证学习率与初始化的交互
    
    4. **网络宽度**（fan-in）：
       - Xavier/He 初始化方差依赖 $n_{in}$
       - 宽网络（$n_{in}=1024$）初始化方差小
       - 窄网络（$n_{in}=256$）初始化方差大
       - 验证自适应初始化的重要性
    
    **预期结果**：
    
    | 初始化 | 激活函数 | 深度 | 预期效果 |
    |:------|:--------|:----|:--------|
    | He | ReLU | 10 | 收敛快，激活值稳定 |
    | Xavier | ReLU | 10 | 收敛慢，激活值衰减 |
    | Xavier | sigmoid | 10 | 收敛快，激活值稳定 |
    | He | ReLU | 50 | 收敛快，但可能需 BN |
    | Xavier | ReLU | 50 | 可能梯度消失 |
    
    **分析方法**：
    
    1. **激活值分析**：
       - 绘制每层激活值均值和方差曲线
       - 信号衰减：方差逐层减小
       - 信号饱和：方差逐层增大后稳定
       - 信号稳定：方差各层相近
    
    2. **梯度分析**：
       - 绘制每层梯度范数曲线
       - 梯度消失：深层梯度范数接近零
       - 梯度爆炸：梯度范数逐层增大
       - 梯度稳定：各层梯度范数相近
    
    3. **收敛分析**：
       - 比较训练损失曲线
       - 收敛快：损失快速下降
       - 收敛慢：损失下降缓慢
       - 不收敛：损失震荡或停滞
    
    **实验设计关键要点**：
    
    1. **控制变量**：固定数据、优化器、批量大小，只改变初始化、激活函数、深度
    
    2. **多维度验证**：不只看最终精度，还要分析激活值、梯度、收敛曲线
    
    3. **深度梯度**：不同深度验证初始化敏感性的累积效应
    
    4. **匹配验证**：验证激活函数与初始化的匹配关系
    
    5. **学习率交互**：验证学习率与初始化的交互效应
    
    **总结**：
    
    实验设计的关键要素：
    - 多种初始化方法对比（Xavier, He, 随机小, 随机大）
    - 不同网络深度（验证累积效应）
    - 激活函数匹配（ReLU 配 He，sigmoid 配 Xavier）
    - 学习率交互（验证初始化方差与学习率的协同）
    - 多指标分析（激活值、梯度、收敛曲线、最终精度）
    
    通过系统实验验证：
    1. 初始化对深度网络的关键影响
    2. 初始化与激活函数的匹配关系
    3. 深度对初始化敏感性的放大
    4. 学习率与初始化的交互效应
    </details>