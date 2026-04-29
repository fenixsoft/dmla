# 批归一化

深度神经网络训练过程中存在一个隐蔽但致命的问题：随着网络深度增加，各层输入分布不断变化，导致梯度传播不稳定。这种现象称为**内部协变量偏移**（Internal Covariate Shift），它使得深层网络难以训练、收敛缓慢、对初始化高度敏感。

2015 年，Google 的两位研究员谢尔盖·伊奥费（Sergey Ioffe）和克里斯蒂安·谢盖迪（Christian Szegedy） 在论文《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》中提出了**批归一化**（Batch Normalization，简称 BN）技术。这一方法的灵感来自传统机器学习中数据标准化预处理的思想，他们将其扩展到神经网络的每一层，实现了训练过程中实时、自适应的标准化。BN 使得深度网络训练更加稳定和快速，允许使用更大的学习率，也缓解了初始化敏感问题，甚至自带一定的正则化效果。从 ResNet 开始，BN 成为几乎所有现代深度网络的标准组成部件。

本章将深入分析内部协变量偏移问题的本质，介绍 BN 的算法原理与计算流程，探讨 BN 如何提升训练稳定性，讲解 BN 在卷积神经网络中的特殊应用方式，辨析训练与推理模式的差异，并通过实验验证 BN 的实际效果。最后，我们将讨论 BN 的局限性及其变体方案，帮助读者在不同场景下做出正确选择。

## 内部协变量偏移

前面我们解决过梯度消失、梯度爆炸、初始化不当等影响训练稳定性的问题，本节我们讨论的内部协变量偏移问题不如前面列举那些容易直接观察出来，却从根本上影响着网络的训练效率。以一个具体场景来解释什么是内部协变量偏移：

设想我们正在训练一个预测房价的神经网络，输入特征包括房屋面积、位置、年龄等，输出是预测价格。这个网络有三层隐藏层，使用 Sigmoid 激活函数。在训练的某一时刻，第二隐藏层的输出值分布在 0.3 到 0.6 之间，均值约为 0.45。这些值经过 Sigmoid 函数后，正好落在其舒适工作区间中 —— Sigmoid 在输入为 0 附近时最敏感，梯度最大；而 0.3 到 0.6 对应的 Sigmoid 输出区间梯度适中，训练正常进行。然而，当第一隐藏层的参数更新后，第二隐藏层的输入发生了变化。原本分布在 0.3 到 0.6 的值，现在变成了 -5.5 到 -5.0，均值变为 -5.3。Sigmoid 函数在这些输入值上的梯度变得很小。第三隐藏层突然发现自己的输入数据的分布完全变了，仿佛跟上一轮毫无关系，激活函数的行为也变了，之前学习到的参数不再适用。这就是内部协变量偏移现象，网络训练过程中，由于前层参数的更新，后续各层的输入分布随之持续变化，每层都需要不断适应这种变化，而非专注于学习稳定的特征表示。

从信息流动的视角看，深度网络像一个多级信号处理系统。传统系统中，我们会对输入信号进行标准化预处理，确保信号在各处理单元中保持稳定的统计特性。但在深度网络中，由于每个处理单元（隐藏层）的参数都在动态更新，信号在经过每一层后统计特性都会变化。如果不加以控制，这种变化会层层放大，最终导致信号在深层严重偏离正常范围，要么进入激活函数的饱和区导致梯度消失，要么进入极端区域导致梯度爆炸。这会导致以下四方面的问题：

1. **训练不稳定，收敛困难。** 输入分布的持续变化意味着梯度方向也在变化，参数更新因此变得震荡，一个方向上的优化可能被下一轮的分布变化打断，导致收敛缓慢甚至无法收敛。
2. **学习率被迫降低，训练加速受限。** 在传统优化理论中，较大的学习率意味着更大的步长，可以加速收敛。但在存在协变量偏移的网络中，大学习率会放大问题。参数的大幅度更新会导致输入分布的剧烈变化，可能使整个网络进入不稳定状态。为了安全，只能使用较小的学习率，以牺牲收敛速度换取训练稳定性。
3. **激活函数行为异常，梯度问题加剧。** 分布变化可能将激活值推入危险的区域。对于 Sigmoid 和 tanh 函数，如果输入值进入饱和区（譬如 Sigmoid 的输入绝对值大于 4 时，输出接近 0 或 1，导数接近 0），梯度会消失；对于 ReLU 函数，如果输入值持续为负（分布偏移使大部分值落入负区间），神经元会大量死亡，输出恒定为 0，梯度完全消失。相反，如果分布偏移使激活值变得极端大，梯度可能爆炸。协变量偏移与梯度问题相互加剧。
4. **初始化敏感度放大，容错性降低。** 深度网络的初始化本来就很关键，而协变量偏移进一步放大了初始化的影响。一个好的初始化使各层激活值分布在合理范围内；但协变量偏移可能很快破坏这种平衡。一个差的初始化（譬如权重方差设置不当）会使激活值从一开始就偏离正常范围，协变量偏移则使这个问题逐层放大。这意味着网络对初始化变得高度敏感，容错性大为降低。

## BN 算法原理

既然各层输入分布的变化是训练困难的根源，就要想办法在每一层引入一种机制，强制将输入分布标准化，使其保持稳定。这个想法实际上并不新颖。在传统机器学习中，**数据标准化**本来是最基本的预处理步骤。对于特征向量 $x$，我们通常将其标准化为：

$$\hat{x} = \frac{x - \mu}{\sigma}$$

其中 $\mu$ 是训练数据的均值，$\sigma$ 是标准差。标准化后的特征 $\hat{x}$ 具有零均值和单位方差的特点，这种分布更利于优化算法工作，梯度方向更稳定，学习率可以设置得更大。现在问题在于，在深度网络中，每一层的"训练数据"（即前一层的输出）的分布是动态变化的。我们无法预先计算出固定的 $\mu$ 和 $\sigma$，因为这些统计量随训练过程不断演变。此外，纯粹的数据标准化可能带来新的问题，如果强制将所有激活值标准化为零均值、单位方差，势必会限制网络的表示能力。Sigmoid 函数的非线性特性在输入远离零时更强，强制将输入拉向零附近，可能使网络变得过于线性。

BN 的解决思路正是围绕这两个问题展开的。首先，用 [Batch](../neural-network-structure/forward-propagation.md#批量计算与效率优化) 统计替代全局统计。既然无法预先计算固定的 $\mu$ 和 $\sigma$，BN 在每次参数更新时，用当前 Mini-Batch 的数据来估计统计量。每个 Mini-Batch 虽然只是全部数据的一个子集，但其均值和方差足以反映当前分布的大致特征。用这些估计值进行标准化，就能在训练过程中实现每层输入的实时稳定。这意味着无论前层参数如何变化，本层接收到的数据都会被强制拉到零均值、单位方差，从而切断了"参数更新 → 分布变化 → 训练不稳定"这一连锁反应。

其次，引入可学习的缩放和偏移参数。纯粹的标准化会将所有激活值限制在固定的分布中，削弱网络的非线性表达能力。BN 通过在标准化后增加两个可学习参数 $\gamma$（缩放）和 $\beta$（偏移），将"标准化"升级为"可学习的分布变换"。网络可以通过学习合适的 $\gamma$ 和 $\beta$，自主决定是否需要标准化、以及标准化到什么程度。如果某个层的最佳分布就是零均值、单位方差，$\gamma$ 和 $\beta$ 会学习为单位变换；如果某个层需要特定的分布来激活非线性特性，$\gamma$ 和 $\beta$ 会将标准化后的数据调整到合适的位置和尺度。这种设计保证了 BN 不会降低网络的表达能力 —— 它只是提供了一个更易于优化的起点，网络有权选择是否偏离这个起点。

将这两个思路结合起来，BN 在网络的每一层中嵌入了一个分布稳定器，训练时实时计算 Batch 统计量，对每层输入进行标准化，再用可学习参数微调分布。这使得各层不再需要不断适应前层带来的分布变化，可以专注于学习稳定的特征表示。BN 对每个特征维度独立操作，设 Mini-Batch 中某个特征的值为 $\{x_1, x_2, ..., x_m\}$，其中 $m$ 是 Batch Size，BN 的计算分为三个步骤：计算统计量、标准化、缩放偏移。每一步都有其特定的目的和设计考量。

- **计算 Batch 统计量**：计算当前 Mini-Batch 中该特征的均值和方差。设 $x_i$ 是 Batch 中第 $i$ 个样本在该特征上的值。以下公式求得均值 $\mu_B$ 是数据的中心位置，方差 $\sigma_B^2$ 是每个样本偏离均值的平方，衡量数据围绕中心的分散程度：

    $$[batch_mu]\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$

    $$[batch_sigma]\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$

- **标准化**：使用计算出的统计量对每个样本进行标准化，将数据归位到标准位置（零均值）和标准尺度（单位方差），就像将不同单位的测量结果统一到同一度量标准：

    $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

    其中 $x_i - \mu_B$ 将数据平移，使中心移到零点，消除位置偏差。$\sqrt{\sigma_B^2 + \epsilon}$ 是标准差的估计，$\epsilon$ 是一个小常数（通常取 $10^{-5}$），防止方差为零时出现除零错误。将平移后的数据除以标准差是对数据进行缩放，使分散程度变为单位方差。标准化后，$\hat{x}_i$ 具有零均值和单位方差的统计特性。无论原始 $x_i$ 的分布如何，标准化后的值都落在相似的数值范围内。

- **缩放和偏移**：如果所有激活值都被标准化到零附近，网络的非线性表达能力可能被削弱。为此，BN 在标准化后引入了 $\gamma$ 和 $\beta$ 两个可学习的参数，进行缩放和偏移变换：

    $$y_i = \gamma \hat{x}_i + \beta$$

    其中，$\gamma$ 是缩放参数，控制输出值的尺度， $\beta$ 是偏移参数，控制输出值的位置。标准化提供了稳定的训练基础，而 $\gamma$ 和 $\beta$ 给予网络选择权，让它可以保持标准化效果，也可以恢复或调整到更适合当前任务的分布。

    这两个参数的存在使 BN 不仅是标准化的操作，更是一个可学习的分布变换。当 $\gamma = \sigma_B$、$\beta = \mu_B$ 时，$y_i = \sigma_B \cdot \hat{x}_i + \mu_B = x_i$，BN 完全还原了原始分布；当 $\gamma$ 和 $\beta$ 学习到其他值时，BN 则将数据变换到一个新的、可能更适合当前任务的分布。这种设计在保证训练稳定性的同时，保留了网络的表示灵活性。

## BN 算法实现

BN 的三步都是纯粹的计算操作，可以用简单的 Python 代码实现，以下代码就与上面的理论公式一一对应：`np.mean` 和 `np.var` 计算 Batch 统计量，`(x - mu) / np.sqrt(var + eps)` 执行标准化，`gamma * x_hat + beta` 进行缩放偏移。在实际深度学习框架中的 BN 实现会更复杂，包括反向传播、滑动平均统计量维护等，但核心逻辑与此代码是一致的。

```python
def batch_norm(x, gamma, beta, eps=1e-5):
    """
    Batch Normalization 前向传播
    
    参数：
        x: 输入数据，形状为 [batch_size, num_features]
        gamma: 缩放参数，形状为 [num_features]，可学习
        beta: 偏移参数，形状为 [num_features]，可学习
        eps: 防止除零的小常数，默认 1e-5
    
    返回：
        y: BN 输出，形状与 x 相同
        mu: batch 均值，用于后续可能的操作
        var: batch 方差，用于后续可能的操作
    """
    # 步骤一：计算 batch 统计量（对应公式 μ_B 和 σ_B²）
    mu = np.mean(x, axis=0)      # 每个特征的均值，沿 batch 维度计算
    var = np.var(x, axis=0)      # 每个特征的方差，沿 batch 维度计算
    
    # 步骤二：标准化（对应公式 x̂_i）
    x_hat = (x - mu) / np.sqrt(var + eps)  # 减去均值，除以标准差
    
    # 步骤三：缩放和偏移（对应公式 y_i）
    y = gamma * x_hat + beta     # 应用可学习参数
    
    return y, mu, var
```


## BN 与反向传播

BN 作为神经网络中的一层，同样需要依靠反向传播来实现更新参数。设损失函数为 $l$，输入 Batch 为 $\{x_1, ..., x_m\}$，输出为 $\{y_1, ..., y_m\}$。已知从上层传来的梯度 $\frac{\partial l}{\partial y_i}$，我们需要计算可学习参数 $\gamma$ 和 $\beta$ 的梯度以便优化更新，同时计算输入 $x$ 的梯度以传递给前一层。注意，BN 层用到的 $\mu_B$ 和 $\sigma_B$ 虽然是反向传播中的中间节点，但它们并非可学习参数，不需要单独求梯度更新，它们对 $x$ 梯度的影响已通过链式法则计入 $\frac{\partial l}{\partial x_i}$ 中。

- **对 $\gamma$ 和 $\beta$ 的梯度**：由于 $y_i = \gamma \hat{x}_i + \beta$，$\gamma$ 和 $\beta$ 直接作用于输出，$\gamma$ 的梯度是输出梯度与标准化值的乘积之和（因为 $\gamma$ 以乘法方式作用于 $\hat{x}_i$），$\beta$ 的梯度直接是输出梯度的总和（因为 $\beta$ 以加法方式作用于输出），即：

    $$\frac{\partial l}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial l}{\partial y_i} \cdot \hat{x}_i$$

    $$\frac{\partial l}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \beta} =  \sum_{i=1}^{m} \frac{\partial l}{\partial y_i}$$

- **对输入 $x$ 的梯度**：$x_i$ 对输出 $y_j$ 的影响是多路径的，$x_i$ 不仅直接影响 $\hat{x}_i$ 和 $y_i$，还通过影响 $\mu_B$ 和 $\sigma_B$ 间接影响所有 $\hat{x}_j$ 和 $y_j$。完整的梯度推导需要同时考虑这三条路径：

    $$\frac{\partial l}{\partial x_i} = \frac{\partial l}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} + \sum_{j=1}^{m} \frac{\partial l}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \mu_B} \cdot \frac{\partial \mu_B}{\partial x_i} + \sum_{j=1}^{m} \frac{\partial l}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \sigma_B} \cdot \frac{\partial \sigma_B}{\partial x_i}$$

    经过有些繁琐但并不困难的推导（略），最终的梯度公式可以简化为：

    $$\frac{\partial l}{\partial x_i} = \frac{\gamma}{m\sigma_B}\left(m \frac{\partial l}{\partial y_i} - \sum_{j=1}^{m} \frac{\partial l}{\partial y_j} - \hat{x}_i \sum_{j=1}^{m} \frac{\partial l}{\partial y_j} \hat{x}_j\right)$$

    这个结果虽然看着有些复杂，但保证了梯度能够正确回传。BN 层不会阻断梯度流，梯度可以正常传递到前层这本身就是一个关键结论，是 BN 能够支持深度网络训练的重要原因之一，它不仅稳定了前向传播的数值分布，也保证了反向传播的梯度畅通。

## 训练稳定性

本节我们深入探讨 BN 具体如何改善深度网络的训练过程。BN 对训练的影响是多方面的：它直接解决了协变量偏移问题，间接带来了正则化效果，并改变了网络架构的结构设计选择。首先是 BN 对训练稳定性的提升，可以从四个维度理解，每个维度都直接对应我们在前文分析的协变量偏移问题：

- 首先，BN 实现了**分布稳定**。每层的输入被强制标准化为零均值、单位方差，无论前层参数如何更新，本层接收到的数据始终落在相似的数值范围内。这就像给每层装了一个稳压器，不管输入如何波动，经过 BN 处理后都变得平稳。协变量偏移的根本问题（输入分布随训练变化）被直接消除。
- 其次，BN 带来了**梯度稳定**。激活函数的梯度与其输入值密切相关：Sigmoid 在输入远离零时梯度很小（饱和区），ReLU 在输入为负时梯度为零（死亡区）。标准化将激活值拉到零附近，恰好落在 Sigmoid 和 tanh 的舒适工作区间，梯度适中且稳定；对于 ReLU，标准化也一定程度减少了持续负值的风险，降低了神经元死亡的概率，这些都对梯度稳定有正面作用，梯度稳定意味着参数更新方向稳定，收敛更可靠。
- 再次，BN 允许**更大的学习率**。协变量偏移的一个重要影响是限制了学习率的选择，大学习率导致分布剧烈变化，可能使训练崩溃。BN 消除了这种限制，即使参数大幅度更新，下一层的输入分布仍然被标准化到稳定范围。因此，我们可以放心使用更大的学习率，加速收敛。在 ResNet 等深度网络的训练中，学习率通常从 0.1 开始，这在没有 BN 层的网络中几乎是不可想象的。
- 最后，BN 减轻了**初始化敏感度**。初始化不当的问题主要在于权重方差设置错误会导致激活值分布偏离正常范围，逐层放大后使训练困难。BN 在每一层都进行标准化，相当于在初始化后立即修正了激活值的分布。即使初始化不够理想，BN 也能将其拉回合理范围。这大大提高了网络的容错性，减少了调试初始化参数的精力投入。

除了直接的稳定性提升，BN 还带来一个有趣的"副作用"，它间接带来了一定的正则化效果。这个效果源于 BN 对 Mini-Batch 统计量的依赖。BN 在训练时使用当前 Mini-Batch 的均值 $\mu_B$ 和方差 $\sigma_B^2$ 进行标准化。这两个统计量是 Batch 中数据的样本估计，而非真实的总体统计量。不同的 Batch 包含不同的样本，因此 $\mu_B$ 和 $\sigma_B^2$ 随 Batch 变化而波动。这种波动意味着相同的输入 $x_i$，在不同的 Batch 中可能得到不同的标准化结果 $\hat{x}_i$。

统计量波动本质上是一种噪声。从正则化的视角看，噪声迫使网络学习更加鲁棒的特征，因为输入的标准化结果有随机性，网络不能过度依赖任何特定数值，必须学习能够容忍这种变化的表示。这与 Dropout 思想异曲同工，Dropout 通过随机丢弃神经元引入噪声，BN 则通过随机标准化引入噪声。

需要注意的是，这种噪声仅在训练时存在。推理阶段使用全局统计量（所有训练 Batch 的滑动平均），标准化结果是确定的。这意味着训练时网络学习的是鲁棒特征，推理时这些特征被稳定地激活。这种训练有噪声、推理无噪声的设计，恰好符合正则化的目的。

BN 的正则化效果在实践中表现为使用 BN 的网络往往泛化能力更好，训练损失和测试损失的差距更小。在一些情况下，BN 自带的正则化效果已经足够，甚至可以在一定程度上替代 Dropout。它们两者都是改善训练的技术，但机制和效果有显著差异：

- 从**正则化机制**看，BN 的噪声来自 Batch 统计的随机性，噪声幅度取决于 Batch 内样本的分布差异；Dropout 的噪声来自随机丢弃神经元，噪声幅度由丢弃概率 $p$ 直接控制。BN 的噪声是被动的，取决于数据；Dropout 的噪声是主动的，可以人为调节。
- 从**训练稳定性**看，BN 通过标准化提升稳定性，使梯度传播更顺畅；Dropout 通过丢弃神经元降低稳定性，每次只使用部分网络，相当于降低了模型容量。BN 加速训练，Dropout 则可能减慢训练（因为每次更新的网络更小）。
- 从**适用场景**看，BN 是 CNN 等深度网络的标准配置，几乎所有现代视觉网络都使用 BN；Dropout 在全连接网络中更加常用，尤其是参数量大的分类器层。

现代深度网络（如 ResNet 及后续架构）的典型设计是卷积层后接 BN 和 ReLU，全连接层可以选择加或不加 Dropout。这种设计的原因有三：
- 其一，BN 已经提供了足够的正则化效果，额外 Dropout 收益有限；
- 其二，Dropout 在卷积层的正则化效果较弱（相邻像素高度相关，丢弃部分神经元信息损失有限）；
- 其三，Dropout 可能破坏 BN 的标准化效果，Dropout 改变激活值的分布，使 BN 计算的统计量不稳定；

在实践中，如果网络已经使用 BN，是否需要额外 Dropout 取决于具体场景。数据集小、过拟合严重时，可以在全连接层添加 Dropout；数据集大、BN 正则化效果足够时，则可以省略 Dropout。

## 推理模式

BN 的一个独特之处在于它在训练和推理阶段的行为不同，这种差异源于实际情况的限制，在推理时 Batch Size 可能很小甚至为 1，无法可靠计算 Batch 统计量，因此必须为推理和训练两种模式设计不同的操作流程。

训练时，BN 使用当前 Mini-Batch 的统计量进行标准化（见 {{batch_mu}} 和 {{batch_sigma}}），$\mu_B$ 和 $\sigma_B^2$ 是当前 Batch 中数据的均值和方差。除了标准化操作，BN 在训练时还维护一份全局统计量的估计，全局统计量是历史统计量的加权累积，每个 Batch 的贡献逐渐融入，通过滑动平均更新：

$$[global_mu] \mu_{global} = \alpha \mu_{global} + (1 - \alpha) \mu_B$$
$$[global_sigma] \sigma_{global}^2 = \alpha \sigma_{global}^2 + (1 - \alpha) \sigma_B^2$$

其中超参数 $\alpha$ 是衰减系数，通常取 0.9 或 0.99，控制新信息融入的比例。当 $\alpha$ 接近 1（如 0.99）意味着全局统计量更新缓慢，更依赖长期历史；$\alpha$ 较小（如 0.9）则更新更快，更依赖近期 Batch。训练充分后，全局统计量会收敛到一个稳定值，反映训练数据的整体分布特征。

全局统计量本身并不会影响训练，它们都是为推理阶段准备的。由于推理阶段的以下三个特点，需要有全局统计量支持才能顺利进行：

- **单样本推理问题**：推理最常见的场景是处理单个样本（Batch Size = 1）。当 Batch 只有一个样本时，$\mu_B = x_1$，$\sigma_B^2 = 0$，标准化公式 $\hat{x}_1 = \frac{x_1 - x_1}{\sqrt{0 + \epsilon}} = 0$，所有输入都被标准化为零，完全失去信息。
- **输出稳定性要求**：推理时我们需要确定性输出，就是相同输入应该得到相同输出。如果使用 Batch 统计，输出会随 Batch 组成变化，样本 A 单独推理的输出，与样本 A 和样本 B 组成 Batch 时的输出可能不同。这对于部署和调试是不可接受的。
- **部署一致性**：生产环境中，模型可能部署在各种场景，如实时推理、批量处理、分布式部署等。使用固定的全局统计量，确保了所有场景的输出一致，便于测试和验证。

基于以上原因，推理阶段改为使用全局统计量来计算 $\hat{x}$ 和 $y$，其中 $\mu_{global}$ 和 $\sigma_{global}^2$ 由公式 {{global_mu}} 和 {{global_sigma}} 得到：

$$\hat{x} = \frac{x - \mu_{global}}{\sqrt{\sigma_{global}^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

深度学习框架（如 PyTorch）是通过模式切换来控制 BN 的行为：训练时调用 `model.train()` 设置使用 Batch 统计并更新全局统计量；推理时调用 `model.eval()` 来使用全局统计量。模式切换看似一行命令的事情，却是实践中很容易出错的地方，推理时忘记切换到 `eval()` 模式会导致输出不稳定；训练时意外切换到 `eval()` 模式会导致全局统计量不更新。这些错误可能导致训练和推理结果不一致，性能下降。尤其要特别值得注意的场景是训练过程中的验证阶段。每个 epoch 结束时验证模型性能，需要切换到 `eval()` 模式；验证结束后继续训练，需要切回 `train()` 模式。如果验证后忘记切回 `train()`，否则训练效果可能受损。

## BN 训练实践

理论分析揭示了 BN 的设计原理和预期效果。现在，我们通过具体的代码实验验证这些理论。实验将从三个角度考察 BN 的实际影响：收敛速度、学习率容忍度、以及对深度网络的训练支持。下面的代码实现了一个完整的 BN 层，包括前向传播、反向传播、全局统计量维护等核心功能。我们将使用这个实现构建两个对比网络（一个使用 BN，一个不使用 BN）在相同的训练任务上进行对比实验。

- 实验一 **BN 对收敛速度的影响**：从损失曲线可以清晰看到，使用 BN 的网络训练损失下降更快、更稳定。无 BN 网络的损失曲线呈现明显的震荡，这是协变量偏移导致的梯度不稳定；有 BN 网络的损失曲线平滑下降，BN 的标准化稳定了每层的输入分布，使梯度方向更加一致。最终测试损失的比较也显示，BN 网络不仅收敛更快，泛化性能也更好（训练损失和测试损失差距更小）。

- 实验二 **BN 对学习率的容忍度**：当学习率从 0.001 增大到 0.1 时，无 BN 网络在高学习率下表现明显恶化，损失曲线剧烈震荡甚至发散；有 BN 网络则在高学习率下仍能稳定训练，只是收敛速度略有变化。这验证了 BN 的核心价值：标准化使网络对参数更新的幅度变得不敏感，大学习率不再导致分布剧烈变化，训练可以安全加速。

- 实验三 **BN 对深度网络的支持**：当网络深度从 5 层增加到 15 层时，无 BN 网络的训练难度显著上升：15 层网络的测试损失明显高于 5 层网络，甚至可能出现训练崩溃（梯度消失导致参数无法有效更新）。有 BN 网络则展现出对深度的良好适应性：从 5 层到 15 层，测试损失保持稳定，没有明显的性能下降。这正是 ResNet 等深度架构依赖 BN 的原因，没有 BN，深度网络的训练几乎不可能。


```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Batch Normalization 实现
class BatchNorm:
    def __init__(self, num_features, momentum=0.99, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.cache = None
    
    def forward(self, x, training=True):
        if training:
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            self.cache = (x, x_hat, mu, var)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta
    
    def backward(self, dout, learning_rate=0.01):
        x, x_hat, mu, var = self.cache
        m = x.shape[0]
        dgamma = np.sum(dout * x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (x - mu) * -0.5 * (var + self.eps)**(-1.5), axis=0)
        dmu = np.sum(dx_hat * -1 / np.sqrt(var + self.eps), axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)
        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mu) / m + dmu / m
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        return dx

# 简单网络（支持 BN）
class SimpleNetwork:
    def __init__(self, layer_sizes, use_bn=True, grad_clip=5.0):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.use_bn = use_bn
        self.grad_clip = grad_clip
        self.weights = []
        self.biases = []
        self.bn_layers = []

        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            if use_bn and i < self.num_layers - 1:
                self.bn_layers.append(BatchNorm(layer_sizes[i+1]))
            else:
                self.bn_layers.append(None)

    def forward(self, X, training=True):
        self.activations = [X]
        self.pre_activations = []
        self.bn_outputs = []
        a = X
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            if self.bn_layers[i] is not None:
                z_bn = self.bn_layers[i].forward(z, training=training)
                self.bn_outputs.append(z_bn)
                a = relu(z_bn)
            else:
                self.bn_outputs.append(None)
                a = relu(z)
            if i < self.num_layers - 1:
                a = np.clip(a, -10, 10)
            self.activations.append(a)
        return a

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        delta = self.activations[-1] - y
        delta = np.clip(delta, -5, 5)
        for i in range(self.num_layers - 1, -1, -1):
            if self.bn_layers[i] is not None:
                delta = self.bn_layers[i].backward(delta, learning_rate)
            grad_w = self.activations[i].T @ delta / m
            grad_b = np.mean(delta, axis=0, keepdims=True)
            if self.grad_clip is not None:
                grad_w = np.clip(grad_w, -self.grad_clip, self.grad_clip)
                grad_b = np.clip(grad_b, -self.grad_clip, self.grad_clip)
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            if i > 0:
                if self.bn_layers[i-1] is not None:
                    delta = (delta @ self.weights[i].T) * relu_derivative(self.bn_outputs[i-1])
                else:
                    delta = (delta @ self.weights[i].T) * relu_derivative(self.pre_activations[i-1])
                if np.isnan(delta).any():
                    delta = np.nan_to_num(delta, nan=0.0)

    def compute_loss(self, X, y, training=False):
        output = self.forward(X, training=training)
        if np.isnan(output).any() or np.isinf(output).any():
            return float('inf')
        return np.mean((output - y)**2)

print("实验1：BN 对训练收敛的影响")
print("-" * 40)

# 生成数据
n_train = 200
n_test = 100
n_features = 50

X_train = np.random.randn(n_train, n_features)
y_train = np.sin(X_train[:, 0] * 2) + np.cos(X_train[:, 1]) + np.random.randn(n_train) * 0.1
y_train = y_train.reshape(-1, 1)

X_test = np.random.randn(n_test, n_features)
y_test = np.sin(X_test[:, 0] * 2) + np.cos(X_test[:, 1]) + np.random.randn(n_test) * 0.1
y_test = y_test.reshape(-1, 1)

# 网络配置
layer_sizes = [n_features, 128, 64, 32, 1]

# Mini-Batch 配置（使用 Mini-Batch SGD 以体现 BN 稳定训练的效果）
batch_size = 32
n_batches = max(1, n_train // batch_size)

# 无 BN
net_no_bn = SimpleNetwork(layer_sizes, use_bn=False, grad_clip=5.0)

# 有 BN
net_bn = SimpleNetwork(layer_sizes, use_bn=True, grad_clip=5.0)

# 训练参数
n_epochs = 200
learning_rate = 0.01

# 记录训练过程
train_losses_no_bn = []
test_losses_no_bn = []
train_losses_bn = []
test_losses_bn = []

print("开始训练...")

for epoch in range(n_epochs):
    # Mini-Batch SGD 训练
    indices = np.random.permutation(n_train)
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_train)
        bi = indices[start:end]
        xb, yb = X_train[bi], y_train[bi]

        # 无 BN
        net_no_bn.forward(xb, training=True)
        net_no_bn.backward(xb, yb, learning_rate)

        # 有 BN
        net_bn.forward(xb, training=True)
        net_bn.backward(xb, yb, learning_rate)

    # 损失计算（training=True 保持与训练一致的 BN 行为）
    train_loss_no = net_no_bn.compute_loss(X_train, y_train, training=True)
    test_loss_no = net_no_bn.compute_loss(X_test, y_test, training=True)
    train_loss_bn = net_bn.compute_loss(X_train, y_train, training=True)
    test_loss_bn = net_bn.compute_loss(X_test, y_test, training=True)

    train_losses_no_bn.append(train_loss_no)
    test_losses_no_bn.append(test_loss_no)
    train_losses_bn.append(train_loss_bn)
    test_losses_bn.append(test_loss_bn)

print(f"\n无 BN:")
print(f"  最终训练损失: {train_losses_no_bn[-1]:.4f}")
print(f"  最终测试损失: {test_losses_no_bn[-1]:.4f}")

print(f"\n有 BN:")
print(f"  最终训练损失: {train_losses_bn[-1]:.4f}")
print(f"  最终测试损失: {test_losses_bn[-1]:.4f}")

# 可视化损失曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 无 BN
ax1 = axes[0]
ax1.plot(train_losses_no_bn, label='训练损失', linewidth=2, color='#3498db')
ax1.plot(test_losses_no_bn, label='测试损失', linewidth=2, color='#e74c3c')
ax1.set_xlabel('训练轮数', fontsize=11)
ax1.set_ylabel('损失值', fontsize=11)
ax1.set_title('无 Batch Normalization', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 有 BN
ax2 = axes[1]
ax2.plot(train_losses_bn, label='训练损失', linewidth=2, color='#3498db')
ax2.plot(test_losses_bn, label='测试损失', linewidth=2, color='#e74c3c')
ax2.set_xlabel('训练轮数', fontsize=11)
ax2.set_ylabel('损失值', fontsize=11)
ax2.set_title('有 Batch Normalization', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("实验2：BN 对不同学习率的影响")
print("-" * 40)

learning_rates = [0.001, 0.01, 0.05]
lr_results = {}

for lr in learning_rates:
    print(f"学习率 = {lr}")
    
    net_no = SimpleNetwork(layer_sizes, use_bn=False, grad_clip=5.0)
    net_bn_lr = SimpleNetwork(layer_sizes, use_bn=True, grad_clip=5.0)
    
    no_bn_losses = []
    bn_losses = []
    
    for epoch in range(n_epochs):
        with np.errstate(over='ignore', invalid='ignore'):
            indices = np.random.permutation(n_train)
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_train)
                bi = indices[start:end]
                xb, yb = X_train[bi], y_train[bi]
                
                net_no.forward(xb, training=True)
                net_no.backward(xb, yb, lr)
                
                net_bn_lr.forward(xb, training=True)
                net_bn_lr.backward(xb, yb, lr)
        
        no_bn_losses.append(net_no.compute_loss(X_test, y_test, training=True))
        bn_losses.append(net_bn_lr.compute_loss(X_test, y_test, training=True))
    
    lr_results[lr] = {
        'no_bn': no_bn_losses,
        'bn': bn_losses,
        'no_bn_final': no_bn_losses[-1],
        'bn_final': bn_losses[-1]
    }
    
    print(f"  无 BN 最终测试损失: {no_bn_losses[-1]:.4f}")
    print(f"  有 BN 最终测试损失: {bn_losses[-1]:.4f}")
    print()

# 可视化不同学习率
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = {'no_bn': '#e74c3c', 'bn': '#2ecc71'}

for idx, lr in enumerate(learning_rates):
    ax = axes[idx]
    ax.plot(lr_results[lr]['no_bn'], label='无 BN', linewidth=2, color=colors['no_bn'])
    ax.plot(lr_results[lr]['bn'], label='有 BN', linewidth=2, color=colors['bn'])
    ax.set_xlabel('训练轮数', fontsize=11)
    ax.set_ylabel('测试损失', fontsize=11)
    ax.set_title(f'学习率 = {lr}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("实验3：BN 对深度网络的影响")
print("-" * 40)

depth_configs = [
    {'depth': 5, 'sizes': [n_features, 128, 64, 32, 16, 1]},
    {'depth': 10, 'sizes': [n_features, 128, 128, 64, 64, 32, 32, 16, 16, 8, 1]},
    {'depth': 15, 'sizes': [n_features] + [64]*14 + [1]}
]

depth_results = {}

for config in depth_configs:
    depth = config['depth']
    sizes = config['sizes']

    print(f"网络深度 = {depth}")

    # 无 BN（不裁剪梯度，让梯度爆炸自然发生）
    try:
        net_no = SimpleNetwork(sizes, use_bn=False, grad_clip=None)
        no_bn_losses = []
        crashed = False

        with np.errstate(over='ignore', invalid='ignore'):
            for epoch in range(n_epochs):
                indices = np.random.permutation(n_train)
                for b in range(n_batches):
                    start = b * batch_size
                    end = min(start + batch_size, n_train)
                    bi = indices[start:end]
                    xb, yb = X_train[bi], y_train[bi]
                    net_no.forward(xb, training=True)
                    net_no.backward(xb, yb, 0.01)
                
                loss = net_no.compute_loss(X_test, y_test, training=True)
                if np.isnan(loss) or np.isinf(loss) or loss > 1e10:
                    crashed = True
                    break
                no_bn_losses.append(loss)

        if crashed:
            print(f"  无 BN 训练崩溃（梯度爆炸）")
            depth_results[depth] = {'no_bn': None, 'bn': None}
        else:
            depth_results[depth] = {'no_bn': no_bn_losses, 'bn': None}
            print(f"  无 BN 最终测试损失: {no_bn_losses[-1]:.4f}")
    except Exception as e:
        print(f"  无 BN 训练失败: {e}")
        depth_results[depth] = {'no_bn': None, 'bn': None}

    # 有 BN
    try:
        net_bn_depth = SimpleNetwork(sizes, use_bn=True, grad_clip=5.0)
        bn_losses = []
        bn_crashed = False

        for epoch in range(n_epochs):
            indices = np.random.permutation(n_train)
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_train)
                bi = indices[start:end]
                xb, yb = X_train[bi], y_train[bi]
                net_bn_depth.forward(xb, training=True)
                net_bn_depth.backward(xb, yb, 0.01)
            
            loss = net_bn_depth.compute_loss(X_test, y_test, training=True)
            if np.isnan(loss) or np.isinf(loss) or loss > 1e10:
                bn_crashed = True
                break
            bn_losses.append(loss)

        if bn_crashed:
            print(f"  有 BN 训练崩溃")
            depth_results[depth]['bn'] = None
        else:
            depth_results[depth]['bn'] = bn_losses
            print(f"  有 BN 最终测试损失: {bn_losses[-1]:.4f}")
    except Exception as e:
        print(f"  有 BN 训练失败: {e}")
        depth_results[depth]['bn'] = None

    print()

# 可视化深度影响
fig, ax = plt.subplots(figsize=(10, 6))

depths = list(depth_results.keys())
no_bn_finals = [depth_results[d]['no_bn'][-1] if depth_results[d]['no_bn'] else None for d in depths]
bn_finals = [depth_results[d]['bn'][-1] if depth_results[d]['bn'] else None for d in depths]

# 检测崩溃值
finite_no_bn = []
explode_flags = []
for v in no_bn_finals:
    if v is None or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
        explode_flags.append(True)
        finite_no_bn.append(0)
    else:
        explode_flags.append(False)
        finite_no_bn.append(v)

finite_bn = []
for v in bn_finals:
    if v is None or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
        finite_bn.append(0)
    else:
        finite_bn.append(v)

x = range(len(depths))
width = 0.4

bars1 = ax.bar([i - width/2 for i in x], finite_no_bn,
               width, label='无 BN', color='#e74c3c', alpha=0.7)
bars2 = ax.bar([i + width/2 for i in x], finite_bn,
               width, label='有 BN', color='#2ecc71', alpha=0.7)

# 在爆炸柱子上标注"崩溃"
for i, exploded in enumerate(explode_flags):
    if exploded:
        ax.text(i, ax.get_ylim()[1] * 0.95, '训练崩溃',
                ha='center', va='top', fontsize=11, color='#e74c3c', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([f'深度 {d}' for d in depths])
ax.set_xlabel('网络深度', fontsize=11)
ax.set_ylabel('最终测试损失 (对数尺度)', fontsize=11)
ax.set_title('BN 对不同深度网络的影响', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
plt.close()
```

## 局限与变体

BN 在大多数场景下表现优异，但它并非完美无缺。特定场景下，BN 的设计假设可能失效，需要使用替代方案或改进版本。理解这些变体的设计动机和应用场景，是灵活运用归一化技术的关键。BN 的设计依赖于一个关键假设：可以使用 Mini-Batch 的统计量来估计数据的整体分布。这个假设在多数情况下成立，但在以下场景中会遇到问题。

- **Batch Size 的依赖**。BN 的统计量估计质量直接取决于 Batch Size。当 Batch Size 很小（如 $m < 8$）时，统计量的方差很大，估计不稳定；极端情况下 Batch Size = 1，方差 $\sigma_B^2 = 0$，标准化完全失效。Batch Size 并未人为可随意调节，在显存受限、需要高分辨率图像训练等场景中，无法提高 Batch Size 是很现实的问题。
- **分布式训练的复杂性**。在多 GPU 或分布式训练中，每个设备处理不同的 Mini-Batch，计算各自的 Batch 统计。为了保持一致性，需要同步所有设备的统计量，增加了通信开销和实现复杂度。
- **训练与推理的不一致**。训练时使用 Batch 统计，推理时使用全局统计，两种模式的标准化结果可能不同。如果全局统计量在训练期间没有充分收敛，推理结果可能偏离预期。这种不一致在部署调试时可能造成困惑。
- **序列模型的不适用性**。在下一部分讲到大语言模型时会提到，RNN 和 Transformer 处理变长序列，每个时间步的隐藏状态需要独立的标准化。BN 跨 Batch 计算统计量，在序列模型中难以直接应用，同一 Batch 中不同样本的序列长度可能不同，不同时间步的隐藏状态分布也不同。

针对上述局限性，研究者提出了多种批归一化的变体方案，每种变体从不同角度解决 BN 的特定问题，适用于特定场景，主要包括有：

- **Batch Renormalization**（BrN）：在小 Batch 场景下的改进方案。BrN 的核心思想是当 Batch 统计与全局统计差异过大时，对 Batch 统计进行修正，而非完全依赖它。具体做法是引入两个修正因子 $r$ 和 $d$：

    $$\hat{x} = \frac{x - \mu_B}{\sigma_B} \cdot r + d$$

    其中 $r = clip(\frac{\sigma_B}{\sigma_{global}}, r_{min}, r_{max})$ 限制标准差的偏离，$d = clip(\frac{\mu_B - \mu_{global}}{\sigma_{global}}, d_{min}, d_{max})$ 限制均值的偏离。当 Batch 统计与全局统计接近时，BrN 行为与 BN 相同；当差异过大时，修正因子将其拉回合理范围，这种设计在小 Batch 场景下比 BN 更稳定。

- **Layer Normalization**（LN）：完全不依赖 Batch 的标准化方案。LN 对单个样本的所有特征计算统计量：

    $$\mu_L = \frac{1}{d}\sum_{j=1}^{d} x_j$$
    $$\sigma_L^2 = \frac{1}{d}\sum_{j=1}^{d} (x_j - \mu_L)^2$$

    LN 的统计量来自单个样本内部，与 Batch Size 完全无关。这使得 LN 天然适用于 RNN 和 Transformer，每个时间步的隐藏状态可以独立标准化，训练和推理行为一致。LN 是 Transformer 架构的默认归一化方案。

- **Group Normalization**（GN）：介于 LN 和 BN 之间的折衷方案。GN 将特征分成若干组，每组独立标准化：

    $$\mu_G = \frac{1}{G \cdot h \cdot w}\sum_{g=1}^{G}\sum_{p,q} x_{g,p,q}$$

    GN 的统计量来自单个样本的部分特征，不依赖 Batch Size。分组数量 $G$ 是可调参数：$G = 1$ 时 GN 等价于 LN（所有特征一组），$G = c$（通道数）时 GN 等价于 Instance Normalization（每个通道一组）。GN 在小 Batch CNN 场景下表现优于 BN，是目标检测、分割等显存密集型任务的推荐选择。

- **Instance Normalization**（IN）：每个样本每个通道独立标准化。IN 的统计量来自单个样本的单个通道：

    $$\mu_I = \frac{1}{h \cdot w}\sum_{p,q} x_{p,q}$$

    IN 的标准化粒度最细，保留了最多的样本间差异和通道间差异。这种特性在图像风格迁移任务中特别有用，风格特征主要体现在通道级的统计差异上，IN 可以有效分离内容和风格。IN 不常用于通用分类任务，但在生成式模型中有特殊价值。

不同归一化方案各有适用场景，选择归一化方案时，需要考虑以下三个因素：

1. **Batch Size 可用性**：Batch Size 充足（$\geq 16$）时优先 BN，受限时考虑 GN 或 LN。
2. **架构类型**：CNN 常用 BN 或 GN，RNN、Transformer 常用 LN。
3. **任务特性**：风格迁移等特殊任务可能需要 IN。

值得注意的是，这些方案并非互斥。在一些复杂架构中，不同部分可能使用不同的归一化，譬如 Transformer 中的卷积部分使用 GN，注意力部分使用 LN，灵活组合是高级设计的表现，下表总结了各方案的特点和推荐用途：

| 方法 | 适用场景 | Batch Size 依赖 | 统计量来源 |
|:----|:--------|:---------------|:----------|
| BN | CNN、深度网络 | 强（推荐 $m \geq 16$） | Batch + 特征维度 |
| LN | RNN、Transformer | 无 | 单样本 + 所有特征 |
| GN | 小 Batch CNN | 无 | 单样本 + 特征分组 |
| IN | 风格迁移、生成模型 | 无 | 单样本 + 单通道 |


## 本章小结

批归一化是深度学习发展史上的里程碑技术，它通过在每层对 Mini-Batch 数据进行标准化，解决了困扰深度网络训练多年的协变量偏移问题，深刻改变了深度学习的实践方式。批归一化的提出者 谢尔盖·伊奥费和克里斯蒂安·谢盖迪在论文中写道："我们希望批归一化能成为深度网络训练的标准组件。"十年后的今天，这个愿景已经成为现实，从 ResNet 到 Transformer，从计算机视觉到自然语言处理，BN 及其变体无处不在。掌握这项技术，是深度学习实践者的重要基础。

## 练习题

1. 给定 Mini-Batch 中某特征的 4 个样本值为 $\{2, 4, 6, 8\}$，设 $\epsilon = 0$，$\gamma = 2$，$\beta = 1$，按照 BN 的三个步骤（计算统计量、标准化、缩放偏移）手动计算每个样本的 BN 输出值。
    <details>
    <summary>参考答案</summary>

    **第一步：计算 Batch 统计量**

    $$\mu_B = \frac{2 + 4 + 6 + 8}{4} = \frac{20}{4} = 5$$

    $$\sigma_B^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = \frac{9 + 1 + 1 + 9}{4} = \frac{20}{4} = 5$$

    **第二步：标准化**（$\epsilon = 0$，故 $\sqrt{\sigma_B^2 + \epsilon} = \sqrt{5}$）

    $$\hat{x}_1 = \frac{2 - 5}{\sqrt{5}} = \frac{-3}{\sqrt{5}}, \quad \hat{x}_2 = \frac{4 - 5}{\sqrt{5}} = \frac{-1}{\sqrt{5}}, \quad \hat{x}_3 = \frac{6 - 5}{\sqrt{5}} = \frac{1}{\sqrt{5}}, \quad \hat{x}_4 = \frac{8 - 5}{\sqrt{5}} = \frac{3}{\sqrt{5}}$$

    **第三步：缩放偏移**（$y_i = \gamma \hat{x}_i + \beta = 2\hat{x}_i + 1$）

    $$y_1 = 2 \cdot \frac{-3}{\sqrt{5}} + 1 = 1 - \frac{6}{\sqrt{5}}, \quad y_2 = 2 \cdot \frac{-1}{\sqrt{5}} + 1 = 1 - \frac{2}{\sqrt{5}}$$

    $$y_3 = 2 \cdot \frac{1}{\sqrt{5}} + 1 = 1 + \frac{2}{\sqrt{5}}, \quad y_4 = 2 \cdot \frac{3}{\sqrt{5}} + 1 = 1 + \frac{6}{\sqrt{5}}$$

    验证：标准化后均值为 0、方差为 1；缩放偏移后均值变为 $\beta = 1$，方差变为 $\gamma^2 = 4$，与计算结果一致。
    </details>

1. 设 $\gamma = \sigma_B$、$\beta = \mu_B$，代入 BN 的缩放偏移公式 $y_i = \gamma \hat{x}_i + \beta$，证明此时 BN 层完全还原了原始输入 $x_i$，并说明这一性质在网络表达能力方面的意义。
    <details>
    <summary>参考答案</summary>
    将 $\gamma = \sigma_B$、$\beta = \mu_B$ 代入缩放偏移公式（忽略 $\epsilon$ 以简化推导）：

    $$y_i = \sigma_B \cdot \hat{x}_i + \mu_B = \sigma_B \cdot \frac{x_i - \mu_B}{\sigma_B} + \mu_B = x_i - \mu_B + \mu_B = x_i$$

    证毕。当可学习参数取上述值时，BN 层等价于恒等映射，原始信息完全保留。

    **意义**：这一性质保证了 BN 不会降低网络的表示能力。即使某层不需要归一化，网络也能通过将 $\gamma$ 和 $\beta$ 学习为合适的值来"绕过" BN 操作。BN 提供了一个更易优化的起点（零均值、单位方差），但网络有权选择偏离这个起点，它扩展了参数空间，而非缩小。这解决了"纯标准化会削弱非线性表达能力"的顾虑。
    </details>

1. BN 在训练阶段使用当前 Batch 的统计量，同时维护全局统计量用于推理。假设初始全局均值 $\mu_{global} = 0$，全局方差 $\sigma_{global}^2 = 1$，衰减系数 $\alpha = 0.9$。若前 3 个 Batch 的统计量依次为 $(\mu_{B1}=2,\; \sigma_{B1}^2=3)$、$(\mu_{B2}=1,\; \sigma_{B2}^2=2)$、$(\mu_{B3}=-1,\; \sigma_{B3}^2=4)$，请逐步计算第 3 个 Batch 处理完成后全局均值和全局方差的值。
    <details>
    <summary>参考答案</summary>
    全局均值更新公式：$\mu_{global} \leftarrow \alpha \cdot \mu_{global} + (1 - \alpha) \cdot \mu_B$
    全局方差更新公式：$\sigma_{global}^2 \leftarrow \alpha \cdot \sigma_{global}^2 + (1 - \alpha) \cdot \sigma_B^2$

    **第 1 个 Batch**：
    $$\mu_{global} = 0.9 \times 0 + 0.1 \times 2 = 0.2$$
    $$\sigma_{global}^2 = 0.9 \times 1 + 0.1 \times 3 = 0.9 + 0.3 = 1.2$$

    **第 2 个 Batch**：
    $$\mu_{global} = 0.9 \times 0.2 + 0.1 \times 1 = 0.18 + 0.1 = 0.28$$
    $$\sigma_{global}^2 = 0.9 \times 1.2 + 0.1 \times 2 = 1.08 + 0.2 = 1.28$$

    **第 3 个 Batch**：
    $$\mu_{global} = 0.9 \times 0.28 + 0.1 \times (-1) = 0.252 - 0.1 = 0.152$$
    $$\sigma_{global}^2 = 0.9 \times 1.28 + 0.1 \times 4 = 1.152 + 0.4 = 1.552$$

    因此第 3 个 Batch 完成后，全局均值为 $0.152$，全局方差为 $1.552$。可以看出，由于 $\alpha = 0.9$ 较大，全局统计量更依赖历史累积，更新较为缓慢。这正是滑动平均的特性，它通过大量 Batch 的逐步累积来逼近真实的数据分布。
    </details>

1. 在 Batch Size = 1 的极端情况下，BN 训练时 $\mu_B = x_1$、$\sigma_B^2 = 0$。从标准化公式出发分析此时 BN 的行为，并说明为什么推理阶段不允许使用 Batch 统计量。
    <details>
    <summary>参考答案</summary>
    当 Batch Size = 1 时，代入标准化公式：

    $$\hat{x}_1 = \frac{x_1 - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} = \frac{x_1 - x_1}{\sqrt{0 + \epsilon}} = \frac{0}{\sqrt{\epsilon}} = 0$$

    所有输入都被标准化为 0，信息完全丢失。后续的缩放偏移 $y_1 = \gamma \cdot 0 + \beta = \beta$ 只能输出一个常数偏移，原始输入特征 $x_1$ 对输出没有任何影响。

    **为什么推理阶段不能使用 Batch 统计量**：

    推理场景通常是单样本（Batch Size = 1），此时 Batch 统计量会导致：(1) 均值等于唯一样本自身，$\mu_B = x_1$；(2) 方差恒为零，$\sigma_B^2 = 0$；(3) 标准化后所有输入变为 0，网络输出退化。此外，如果使用多样本推理但 Batch 组成不固定，同一输入在不同 Batch 中的标准化结果会不同——违反了推理确定性原则（相同输入应得到相同输出）。因此推理必须使用训练时累积的全局统计量，保证输出稳定且保留信息。
    </details>

1. BN、Layer Normalization（LN）、Group Normalization（GN）和 Instance Normalization（IN）是四种常用的归一化方法。请从统计量计算范围的角度，用一张表格对比它们的差异，并说明为什么 Transformer 架构默认使用 LN 而非 BN。
    <details>
    <summary>参考答案</summary>
    **对比表格**：

    | 方法 | 统计量计算范围 | 依赖 Batch | 适用架构 |
    |:----|:-------------|:----------|:--------|
    | BN | 跨 Batch、同特征维度 | 是 | CNN |
    | LN | 单样本、所有特征 | 否 | RNN、Transformer |
    | GN | 单样本、特征分组 | 否 | CNN（小 Batch） |
    | IN | 单样本、单通道 | 否 | 风格迁移、生成模型 |

    **为什么 Transformer 默认使用 LN 而非 BN**：

    1. **序列长度可变**：同一个 Batch 中不同样本的序列长度可能不同，BN 跨 Batch 计算统计量时会遇到长度对齐问题；
    2. **Batch Size 通常较小**：由于 Transformer 参数量巨大（尤其是自注意力机制的 $O(n^2)$ 复杂度），实际训练中 Batch Size 往往受限，BN 的统计量估计质量差；
    3. **时间步独立性需求**：每个时间步的隐藏状态需要独立标准化，LN 对单个样本的所有特征计算统计量，天然支持变长序列且训练/推理行为一致。因此 LN 成为 Transformer 的标准选择。
    </details>
