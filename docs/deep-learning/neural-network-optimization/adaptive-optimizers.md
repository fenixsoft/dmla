# 自适应优化器

自适应学习率的思想源于一个朴素的观察，不同参数在训练过程中扮演着不同的角色，有些参数频繁更新，梯度大而稳定；有些参数更新稀疏，梯度小甚至偶尔为零。如果给所有参数配发同样的步长，频繁更新的参数可能步子太大、震荡难收敛，而稀疏更新的参数可能步子太小、前进缓慢。这就像给校长给全校学生布置同样的作业量，高年级的学生觉得无聊，低年级的学生觉得吃力。

**自适应优化器**（Adaptive Optimizers）正是为解决以上"一刀切"问题而生，它们根据参数的历史梯度自动调整学习率，为每个参数分配独立的更新步长。这种"因材施教"的策略最早由约翰·杜奇（John Duchi）于 2011 年提出，他的 AdaGrad 算法开创了自适应学习率的先河。随后，辛顿在 2012 年的 Coursera 课程中提出了 RMSprop，解决了 AdaGrad 学习率过早衰减的问题。2015 年，迪德里克·金玛（Diederik Kingma）和吉米·巴（Jimmy Ba）发表里程碑论文《Adam: A Method for Stochastic Optimization》，将动量法与自适应学习率完美结合，Adam 由此成为深度学习领域最流行的优化器。2019 年，伊利亚·洛希洛夫（Ilya Loshchilov）和弗兰克·赫特（Frank Hutter）发现 Adam 的权重衰减实现存在理论缺陷，提出了 AdamW，进一步提升了泛化能力。本章将会介绍以上四种重要的自适应优化器，分析它们的设计原理、优缺点和适用场景。

## AdaGrad

上一章介绍的动量法和 NAG 解决了梯度下降的震荡问题，但它们仍然给所有参数分配同样的学习率，这在许多场景中会受限，譬如在自然语言处理等任务中，词汇嵌入层的参数更新极不均匀，常用词频繁出现、梯度大，罕见词偶尔出现、梯度小甚至为零。如果用统一学习率，常用词的嵌入向量更新太快容易震荡，罕见词的嵌入向量更新太慢难以学到有效表示。这促使神经网络的研究者们开始需找让每个参数拥有自己的学习率的方法。

**AdaGrad**（Adaptive Gradient Algorithm）由约翰·杜奇在 2011 年提出，是最早的自适应优化器之一。AdaGrad 的设计思想直白明了，既然频繁更新的参数已经学到了很多信息，就应该放慢脚步，反之，稀疏更新的参数还有大量信息待学，就应该大步前进。这种"因材施教"的策略让每个参数都能在最适合自己的节奏下学习，避免统一学习率带来的效率浪费。

实现这一想法的关键问题是要度量出参数更新幅度的大小。AdaGrad 选择的度量项是**累积梯度平方**，选择梯度平方而非梯度本身有两个原因，一是平方运算将负梯度转为正数，避免正负抵消导致累积量失真（参数可能频繁在正负方向来回更新，但累积量应该如实反映更新活跃程度）；二是平方放大大的梯度、缩小小的梯度，使活跃参数的累积量更快增长，学习率衰减更显著，实现越活跃越谨慎的调节效果。选择累积而非滑动平均，是因为 AdaGrad 的设计初衷是为了处理稀疏梯度问题，稀疏参数偶尔才有梯度，累积量增长缓慢，能长期保持较大学习率；而频繁参数累积量快速增长，学习率快速衰减，防止震荡。这种累积增长、学习率递减的设计，本质上是用历史梯度作为"信用额度"去给未来做背书，梯度越活跃，累积量越大，学习率越小，表示该参数已经学得够多，后续更新应该更谨慎。

设 $\mathbf{G}_t$ 是历史梯度平方的累积量，记录了参数更新了多少次、每次更新幅度多大，$(\nabla L_t)^2$ 是当前梯度的平方，将梯度值转为非负数（负梯度同样要学习），以下公式表达了梯度累计的过程：

$$[eq:adagrad-update] \mathbf{G}_t = \mathbf{G}_{t-1} + (\nabla L_t)^2$$

AdaGrad 使用学习率超参数除以累积梯度的平方根来达到累积量越大、实际学习率越小的效果。为了防止开始时还没有累积梯度，导致出现除数为零，操作上会将累积梯度附加一个很小的基础值 $\epsilon$（通常为 $10^{-8}$）。AdaGrad 的有效学习率表示为 $\frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}}$，这里表示的是逐元素运算，即梯度中每个分量的学习率为 $\frac{\eta}{\sqrt{G_{t,i} + \epsilon}}$，这意味着同一批参数中，有的参数学习率可能很大，有的可能很小，完全取决于各自的梯度历史。综上，得到权重参数的更新过程：

$$[eq:adagrad-sum]\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \cdot \nabla L_t$$

AdaGrad 种自适应调整特别适合处理稀疏梯度问题（如自然语言处理中的词嵌入），稀疏更新的参数获得更大的学习率，加速学习。不过，AdaGrad 看似完美的累积历史梯度策略，其实隐藏着一个严重缺陷。既然 $\mathbf{G}_t$ 是历史梯度平方的累积，训练过程中它只会不断增大（梯度平方非负），$\sqrt{\mathbf{G}_t}$ 也只增不减。这意味着有效学习率 $\frac{\eta}{\sqrt{\mathbf{G}_t}}$ 只会不断下降，AdaGrad 调节的只是学习率变小的快慢而已。这样的话，参数每更新一步，学习率就变小一点。训练后期，学习率可能变得极小（如 $10^{-6}$），参数几乎不再更新，训练停滞。这个缺陷使 AdaGrad 陷入两难，它适合短期训练或稀疏梯度问题（频繁更新参数获得小学习率，稀疏参数获得大学习率），但长期训练会导致学习率过早衰减，后期训练停滞。为了解决这个缺陷，就引出了 AdaGrad 的改进版本 —— RMSprop。

## RMSprop

**RMSprop**（Root Mean Square Propagation）由杰弗里·辛顿在 2012 年的 Coursera 课程中提出。有趣的是，辛顿从未正式发表过关于 RMSprop 的论文，它仅作为课程笔记被流传，但因其简单有效迅速被社区采纳。RMSprop 对 AdaGrad 的改进是用指数滑动平均代替 AdaGrad 的累积。滑动平均是一种加权平均，新数据权重大、旧数据权重逐渐衰减，就像一个窗口在数据链中滑动，让历史信息有进有出，因此 RMSprop 能做到只保留最近若干步（约 $\frac{1}{1-\gamma}$ 步）的梯度信息。

设 $\mathbf{E}_t$ 是梯度平方的指数滑动平均，$\gamma$ 是衰减系数（通常取 $0.9$），用作历史累积梯度的权重，控制历史信息的保留程度，剩下的 $(1-\gamma)$ 用作新梯度的权重，这样保证了权重之和为 1。RMSprop 的梯度累计公式为（可与公式 {{eq:adagrad-update}} 对比）：

$$[eq:rmsprop-update] \mathbf{E}_t = \gamma \mathbf{E}_{t-1} + (1 - \gamma)(\nabla L_t)^2$$

如果 $\gamma$ 取 $0.9$ 的话，就是保留 90% 的历史梯度信息，加入 10% 的当前梯度信息，旧历史逐渐淡出，避免数值膨胀。有效窗口约为 $\frac{1}{1-0.9} = 10$ 步，大约只保留最近 10 步的梯度信息。除此以外，RMSprop 的权重更新与 AdaGrad 完全一致了（见 {{eq:adagrad-sum}} ）：

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\mathbf{E}_t + \epsilon}} \cdot \nabla L_t$$

RMSprop 使用指数滑动平均的特性带来了两点收益，首先是窗口效应，只保留最近约 $\frac{1}{1-\gamma}$ 步的梯度信息，避免了梯度累计只进不出，让学习率不会单调递减，而是呈现出平缓起伏的状态。此外，RMSprop 能够更快适应梯度变化，当梯度变化剧烈时，$\mathbf{E}_t$ 也随之快速调整，学习率响应更加灵敏。但 RMSprop 也有一个缺陷（当然，这个缺陷 AdaGrad 也有，并不是 RMSprop 的改进带来的），它只使用梯度平方调整学习率，没有累积历史梯度方向，相当于放弃了[动量法](gradient-descent.md#动量法)的平滑效果。能否同时拥有动量的平稳性和自适应学习率的灵活性？这引出了当前正在被广泛使用的自适应优化器 Adam。

## Adam

**Adam**（Adaptive Moment Estimation）由迪德里克·金玛（Diederik Kingma）和吉米·巴（Jimmy Ba） 2015 年在国际机器学习会议（ICLR）上提出。Adam 的名称源于其核心机制：同时估计梯度的[一阶矩](https://en.wikipedia.org/wiki/Moment_(mathematics))（First Moment，即均值）和二阶矩（Second Moment，即未中心化的方差），将动量法与自适应学习率融于一身。这种双管齐下的策略让参数更新既有方向上的平滑稳定，又有步长上的因材施教，成为当前深度学习领域最流行的优化器。

实现这一融合的关键是 Adam 同时维护两个状态变量。**一阶矩** $\mathbf{m}_t$ 累积历史梯度的方向信息，相当于动量法中的速度变量，用于平滑更新路径、抑制震荡。**二阶矩** $\mathbf{v}_t$ 累积历史梯度的平方，相当于 RMSprop 中的滑动平均，用于调整各参数的学习率。两者独立运作、互不干扰，一阶矩负责"往哪走"，二阶矩负责"走多远"。这种分工让 Adam 在各类任务上都表现稳健，对超参数选择不敏感，默认参数（$\beta_1=0.9, \beta_2=0.999, \eta=0.001$）就能适配绝大多数场景，这也是它能广泛流行的重要原因之一。

设 $\mathbf{m}_t$ 是梯度的一阶矩估计（动量），$\beta_1$ 是一阶矩的衰减系数（默认取 $0.9$），$\nabla L_t$ 是当前梯度方向，以下公式表达了 Adam 的动量累积的过程（与[动量法](gradient-descent.md#动量法)的原理一致）：

$$[eq:adam-m] \mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla L_t$$

设 $\mathbf{v}_t$ 是梯度的二阶矩估计（累积梯度），$\beta_2$ 是二阶矩的衰减系数（默认取 $0.999$），$(\nabla L_t)^2$ 是当前梯度的平方，以下公式表达了 Adam 的梯度平方的滑动平均过程（与 RMSprop 梯度累计公式 {{eq:rmsprop-update}} 也完全一致）：

$$ \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla L_t)^2$$

Adam 最开始初始化 $\mathbf{m}_0 = 0, \mathbf{v}_0 = 0$，这个看似自然的设定却带来了冷启动问题：展开一阶矩公式 {{eq:adam-m}}，$\mathbf{m}_t$ 实际上是历史梯度的加权求和，即 $\mathbf{m}_t = (1-\beta_1)[\nabla L_t + \beta_1 \nabla L_{t-1} + \beta_1^2 \nabla L_{t-2} + ...]$。权重之和本应为1，这样才能正确反映梯度的加权平均，但由于初始化 $\mathbf{m}_0 = 0$，实际权重之和是等比数列求和的结果 $1 - \beta_1^t$。训练初期 $t$ 的值小，$\beta_1^t$ 接近1（譬如 $\beta_1=0.9$ 时 $t=1$ 则 $\beta_1^t=0.9$），权重之和 $1-\beta_1^t$ 远小于1，缺失的权重被零初始化值占据，导致估计值偏向零，同理，二阶距公式也存在相同的缺陷，这就是 Adam 的冷启动问题。因此，需要定义中间变量 $\hat{\mathbf{m}}_t$ 和 $\hat{\mathbf{v}}_t$ ，让 $ \mathbf{m}_t$ 和 $ \mathbf{v}_t$ 除以权重之和来修正偏差，补足缺失部分：

$$ \hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

观察 Adam 的迭代过程，训练初期 $t$ 小，$\beta^t$ 接近 1，修正因子 $\frac{1}{1 - \beta^t}$ 就大，估计值被放大以抵消零初始化偏差；训练后期 $t$ 大，$\beta^t$ 接近 0，修正因子接近 1，修正影响消失。用具体数值来演示直观说明，设 $\beta_1 = 0.9$，$t = 1$，梯度 $\nabla L_1 = 10$，未修正的 $\mathbf{m}_1 = 0.9 \times 0 + 0.1 \times 10 = 1$，修正后的 $\hat{\mathbf{m}}_1 = \frac{1}{1 - 0.9^1} = 10$，恰好等于实际梯度。综合以上四个变量，Adam 的权重更新公式为：

$$[adam-w-update] \mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot \hat{\mathbf{m}}_t$$

这个公式可以理解为用偏差修正后的动量 $\hat{\mathbf{m}}_t$ 指明前进方向（平滑后的梯度），用自适应学习率 $\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t}}$ 控制步长（各参数独立调整）。用 $\epsilon$（通常 $10^{-8}$）防止除零，$\eta$ 是全局学习率（通常 $0.001$，比 SGD 的默认值小，因为自适应已放大有效步长）。Adam 结合了动量法和自适应学习率的优点，在计算机视觉、自然语言处理、推荐系统等领域广泛使用，成为深度学习研究的默认优化器。

Adam 提供了四个超参数的设置，但除了全局学习率外，其中三个通常使用默认值即可，实际训练时，损失震荡则降低学习率，收敛慢则增大学习率：

| 超参数 | 默认值 | 作用 |
|:------|:------|:-----|
| $\eta$ | $0.001$ | 全局学习率，范围通常 $[10^{-4}, 10^{-2}]$ |
| $\beta_1$ | $0.9$ | 一阶矩衰减系数，控制动量平滑程度 |
| $\beta_2$ | $0.999$ | 二阶矩衰减系数，控制学习率自适应程度 |
| $\epsilon$ | $10^{-8}$ | 数值稳定常数，防止除零 |

## AdamW

Adam 似乎已经集齐了动量法和自适应学习率的所有优点，成为深度学习的默认优化器。然而，2019 年一篇论文揭示了一个隐藏的理论缺陷，Adam 中权重衰减（L2 正则化）的实现方式与自适应学习率冲突，正则化效果不稳定。这个问题引出了 Adam 的最新修正版本 AdamW。

权重衰减（Weight Decay）是防止过拟合的常用技术，实质上就是 [L2 正则化](../../statistical-learning/linear-models/regularization-glm.md#正则化原理)。在损失函数中添加参数惩罚项，迫使参数保持较小值 $L_{total} = L_{data} + \lambda \|\mathbf{W}\|^2$，相应地，梯度变为 $\nabla L_{total} = \nabla L_{data} + 2\lambda \mathbf{W}$，在 SGD 中，权重衰减的实现很简单直观：

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla L_{data} - \eta \lambda \mathbf{W}_t = \mathbf{W}_t(1 - \eta \lambda) - \eta \nabla L_{data}$$

每步权重乘以 $(1 - \eta \lambda)$，逐渐衰减，所有参数同等对待。但 Adam 的实现方式不同。梯度 $\nabla L_{total} = \nabla L_{data} + 2\lambda \mathbf{W}$ 被累积到一阶矩 $\mathbf{m}_t$ 和二阶矩 $\mathbf{v}_t$ 中，权重衰减项 $2\lambda \mathbf{W}$ 也被自适应学习率缩放：

$$\Delta \mathbf{W}_{reg} = -\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot 2\lambda \mathbf{W}$$

这导致一个问题，当 $\hat{\mathbf{v}}_t$ 很大（参数梯度历史活跃），自适应学习率 $\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t}}$ 很小，权重衰减项被缩小，正则化效果减弱。当 $\hat{\mathbf{v}}_t$ 很小（参数梯度历史稀疏），自适应学习率很大，权重衰减项被放大，这与 L2 正则化均匀衰减所有权重的设计初衷明显不符。回顾 Adam 的机制，梯度更新被自适应学习率缩放，这是为了让频繁更新的参数放慢脚步、稀疏更新的参数加速前进。但权重衰减的目的是均匀约束所有参数，防止任何一个参数过大导致过拟合，与自适应调整的方向完全不同。将两者混在一起，相当于让本应"一视同仁"的正则化也被"因材施教"，逻辑上自相矛盾。

**AdamW**（Adam with Decoupled Weight Decay）由伊利亚·洛希洛夫（Ilya Loshchilov）在 2019 年的论文《Decoupled Weight Decay Regularization》中提出，论文揭示了 Adam 的权重衰减问题，并给出简洁的解决方案，将权重衰减从梯度更新中分离，让梯度更新负责学习数据规律，权重衰减负责控制模型复杂度，两者应该独立运作、互不干扰。AdamW 的更新规则与 Adam 相比（见 {{adam-w-update}}），区别在于多了权重衰减项：

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot \hat{\mathbf{m}}_t - \eta \lambda \mathbf{W}_t$$

其中加入的 $-\eta \lambda \mathbf{W}_t$ 就是权重衰减项，直接作用于参数，不被自适应学习率缩放， $-\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot \hat{\mathbf{m}}_t$ 是梯度更新项，被自适应学习率缩放。整体公式可以理解为权重衰减独立执行，梯度更新自适应执行，两者解耦不干扰。实践中，现在一般在所有场景中都可以使用 AdamW 替代 Adam，尤其在需要强正则化的任务上，譬如大规模语言模型训练，AdamW 如今已成为许多 Transformer 架构模型，如 BERT、GPT 的默认优化器。

## 优化器选择指南

至此，我们已经学习了 SGD、动量法、NAG、AdaGrad、RMSprop、Adam 和 AdamW 七种优化器。每种优化器各有特点，本节从优化器和任务类型两个角度出发，提供一份"两步走"的简单明了的选择指南。

- 步骤一，首先了解各个优化器的特点：

    | 优化器 | 核心机制 | 优势 | 缺陷 | 适用场景 |
    |:------|:--------|:-----|:-----|:--------|
    | SGD | 基础梯度下降 | 简单、稳定 | 震荡、慢 | 简单任务、精调 |
    | Momentum | 动量平滑 | 加速、抑制震荡 | 需调学习率 | 通用 |
    | NAG | 预测位置梯度 | 提前响应拐点 | 稍复杂 | 追求精度 |
    | AdaGrad | 累积梯度平方 | 稀疏梯度友好 | 学习率衰减 | 稀疏数据、短期训练 |
    | RMSprop | 滑动平均梯度平方 | 学习率稳定 | 无动量 | 长期训练、RNN |
    | Adam | 动量 + 自适应 | 鲁棒、快速 | 权重衰减问题 | 通用、默认选择 |
    | AdamW | Adam+解耦权重衰减 | 正则化稳定 | 无 | 默认选择 |

- 步骤二，根据任务特点选择优化器：

    | 任务类型 | 推荐优化器 | 理由 |
    |:--------|:----------|:-----|
    | 通用深度学习 | AdamW | 鲁棒性强，默认选择 |
    | 计算机视觉 | SGD + Momentum | 实验表明泛化能力更好 |
    | 自然语言处理 | AdamW | 稀疏梯度，自适应优势明显 |
    | RNN/LSTM | RMSprop / AdamW | 处理梯度消失，学习率稳定 |
    | 精调预训练模型 | SGD + Momentum | 小学习率精调，防止破坏预训练特征 |
    | 稀疏数据（推荐系统） | AdamW | 稀疏参数获得大学习率 |

## 自适应优化器实践

理论分析揭示了各优化器的设计原理，接下来通过代码实验对比 SGD、动量法、NAG、AdaGrad、RMSprop、Adam 和 AdamW 七种优化器在二次损失函数上的收敛效果。实验使用长椭圆形损失函数（不同方向梯度差异大），从同一起点出发，观察各优化器的参数路径、损失曲线和有效学习率变化。代码实现了七种优化器的完整更新逻辑，包括动量累积、梯度平方累积、滑动平均、偏差修正、预测位置梯度计算和权重衰减解耦，可视化对比各优化器的收敛行为。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 定义损失函数和梯度 - 使用10倍梯度差异以展示SGD震荡特征
def loss_function(W):
    """二次损失函数 L = 0.5 * W^T A W"""
    A = np.array([[1, 0], [0, 10]])  # 梯度差异10倍
    return 0.5 * np.dot(W, A @ W)

def gradient(W):
    """梯度 ∇L = A W"""
    A = np.array([[1, 0], [0, 10]])
    return A @ W

# 各优化器实现
class SGD:
    def __init__(self, lr=0.15):  # 学习率 > 0.1 产生震荡（lr > 1/梯度差异）
        self.lr = lr
        self.path = []

    def step(self, W, grad):
        W_new = W - self.lr * grad
        self.path.append(W_new.copy())
        return W_new

class Momentum:
    def __init__(self, lr=0.05, momentum=0.9):  # 学习率调小，避免过度震荡
        self.lr = lr
        self.momentum = momentum
        self.v = np.zeros(2)
        self.path = []

    def step(self, W, grad):
        self.v = self.momentum * self.v + self.lr * grad
        W_new = W - self.v
        self.path.append(W_new.copy())
        return W_new

class NAG:
    """Nesterov Accelerated Gradient - 在预测位置计算梯度"""
    def __init__(self, lr=0.05, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = np.zeros(2)
        self.path = []

    def step(self, W, grad_func):
        # NAG核心：在预测位置计算梯度，提前响应拐点
        W_lookahead = W - self.momentum * self.v
        grad_lookahead = grad_func(W_lookahead)
        self.v = self.momentum * self.v + self.lr * grad_lookahead
        W_new = W - self.v
        self.path.append(W_new.copy())
        return W_new

class AdaGrad:
    def __init__(self, lr=1.0, eps=1e-8):  # 学习率大，初期收敛快
        self.lr = lr
        self.eps = eps
        self.G = np.zeros(2)
        self.path = []

    def step(self, W, grad):
        self.G += grad ** 2  # 累积梯度平方
        lr_adaptive = self.lr / np.sqrt(self.G + self.eps)  # 学习率递减
        W_new = W - lr_adaptive * grad
        self.path.append(W_new.copy())
        return W_new

class RMSprop:
    def __init__(self, lr=0.3, gamma=0.9, eps=1e-8):  # 学习率适中
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.E = np.zeros(2)
        self.path = []

    def step(self, W, grad):
        self.E = self.gamma * self.E + (1 - self.gamma) * (grad ** 2)  # 滑动平均
        lr_adaptive = self.lr / np.sqrt(self.E + self.eps)  # 学习率稳定
        W_new = W - lr_adaptive * grad
        self.path.append(W_new.copy())
        return W_new

class Adam:
    def __init__(self, lr=0.3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(2)  # 一阶矩（动量）
        self.v = np.zeros(2)  # 二阶矩（梯度平方）
        self.t = 0
        self.path = []

    def step(self, W, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad  # 动量累积
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)  # 梯度平方累积

        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        W_new = W - self.lr / (np.sqrt(v_hat) + self.eps) * m_hat
        self.path.append(W_new.copy())
        return W_new

class AdamW:
    def __init__(self, lr=0.3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0
        self.path = []

    def step(self, W, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 解耦权重衰减：直接作用于参数，不被自适应学习率缩放
        W_new = W - self.lr * self.weight_decay * W
        W_new = W_new - self.lr / (np.sqrt(v_hat) + self.eps) * m_hat
        self.path.append(W_new.copy())
        return W_new

# 运行实验
W_init = np.array([5.0, 5.0])  # 起点
n_iterations = 50

optimizers = {
    'SGD': SGD(lr=0.15),
    'Momentum': Momentum(lr=0.05, momentum=0.9),
    'NAG': NAG(lr=0.05, momentum=0.9),
    'AdaGrad': AdaGrad(lr=1.0),
    'RMSprop': RMSprop(lr=0.3, gamma=0.9),
    'Adam': Adam(lr=0.3),
    'AdamW': AdamW(lr=0.3, weight_decay=0.01)
}

results = {}
for name, opt in optimizers.items():
    W = W_init.copy()
    losses = []

    for t in range(n_iterations):
        loss = loss_function(W)
        losses.append(loss)
        grad = gradient(W)

        # NAG需要传入梯度函数，其他优化器传入梯度值
        if name == 'NAG':
            W = opt.step(W, gradient)
        else:
            W = opt.step(W, grad)

    results[name] = {
        'path': np.array(opt.path),
        'losses': losses,
        'final_W': W,
        'final_loss': losses[-1]
    }

    print(f"{name:10s}: 最终位置 ({W[0]:.4f}, {W[1]:.4f}), 最终损失 {losses[-1]:.6f}")

print()

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

colors = {'SGD': '#e74c3c', 'Momentum': '#3498db', 'NAG': '#e67e22',
          'AdaGrad': '#f39c12', 'RMSprop': '#9b59b6', 'Adam': '#2ecc71', 'AdamW': '#1abc9c'}

# 图1：参数路径
ax1 = axes[0, 0]
W1_range = np.linspace(-6, 6, 100)
W2_range = np.linspace(-6, 6, 100)
W1_grid, W2_grid = np.meshgrid(W1_range, W2_range)
L_grid = 0.5 * (W1_grid**2 + 10 * W2_grid**2)

ax1.contour(W1_grid, W2_grid, L_grid, levels=[1, 5, 10, 25, 50, 100],
           colors='gray', alpha=0.5, linewidths=0.5)
ax1.contourf(W1_grid, W2_grid, L_grid, levels=[0, 1, 5, 10, 25, 50, 100, 200],
             cmap='Blues', alpha=0.3)

for name, result in results.items():
    path = result['path']
    ax1.plot(path[:, 0], path[:, 1], 'o-', color=colors[name],
             linewidth=2, markersize=3, alpha=0.7, label=name)

ax1.plot(W_init[0], W_init[1], 'ko', markersize=10, label='起点')
ax1.plot(0, 0, 'k*', markersize=15, label='最小值')
ax1.set_xlabel('W1', fontsize=11)
ax1.set_ylabel('W2', fontsize=11)
ax1.set_title('参数路径对比', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)

# 图2：损失曲线
ax2 = axes[0, 1]
for name, result in results.items():
    ax2.plot(result['losses'], color=colors[name], linewidth=2, label=name)

ax2.set_xlabel('迭代次数', fontsize=11)
ax2.set_ylabel('损失值', fontsize=11)
ax2.set_title('损失变化曲线', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# 图3：有效学习率变化（W1方向）- 仅自适应优化器
ax3 = axes[1, 0]
adaptive_optimizers = ['AdaGrad', 'RMSprop', 'Adam', 'AdamW']

for name, result in results.items():
    if name in adaptive_optimizers:
        path = result['path']
        lr_eff = []
        for i in range(len(path) - 1):
            W1_change = path[i+1, 0] - path[i, 0]
            W1_grad = path[i, 0]  # grad_W1 ≈ W1
            lr_eff.append(np.abs(W1_change) / np.abs(W1_grad + 1e-8))
        ax3.plot(lr_eff[:min(30, len(lr_eff))], color=colors[name], linewidth=2, label=name, alpha=0.7)

ax3.set_xlabel('迭代次数', fontsize=11)
ax3.set_ylabel('有效学习率（W1方向）', fontsize=11)
ax3.set_title('自适应学习率变化', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4：收敛速度对比（损失下降速度）
ax4 = axes[1, 1]

for name, result in results.items():
    losses = result['losses']
    loss_decrease = [losses[i] - losses[i+1] for i in range(len(losses)-1)]
    ax4.plot(loss_decrease[:min(30, len(loss_decrease))], color=colors[name], linewidth=2, label=name, alpha=0.7)

ax4.set_xlabel('迭代次数', fontsize=11)
ax4.set_ylabel('每步损失下降量', fontsize=11)
ax4.set_title('损失下降速度对比', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()
```

## 本章小结

本章介绍了自适应优化器的原理与应用，揭示了"为每个参数分配不同学习率"的核心思想如何提升优化效率。至此，第三章"神经网络优化"的内容已全部完成。我们掌握了梯度下降、动量法、NAG 和自适应优化器的原理与应用，理解了如何选择和调优优化算法。下一章将进入第四章"神经网络稳定性"，介绍权重初始化、Dropout、批归一化和梯度问题诊断，解决训练稳定性问题。

## 练习题

1. 分析 AdaGrad 学习率单调递减的原因。设梯度恒为 $g$，推导经过 $t$ 步后有效学习率 $\eta_{eff} = \frac{\eta}{\sqrt{t \cdot g^2}}$。这个性质有什么影响？
    <details>
    <summary>参考答案</summary>
    
    **AdaGrad 学习率递减推导**：
    
    AdaGrad 的累积梯度平方：
    
    $$G_t = \sum_{i=1}^{t} g_i^2$$
    
    设梯度恒为 $g$（$g_i = g$），则：
    
    $$G_t = \sum_{i=1}^{t} g^2 = t \cdot g^2$$
    
    有效学习率：
    
    $$\eta_{eff} = \frac{\eta}{\sqrt{G_t + \epsilon}} \approx \frac{\eta}{\sqrt{t \cdot g^2}} = \frac{\eta}{g \sqrt{t}}$$
    
    有效学习率与 $\sqrt{t}$ 成反比，随迭代次数增加单调递减。
    
    **影响分析**：
    
    1. **初期收敛快**：训练初期 $t$ 小，$\eta_{eff}$ 大，参数快速更新
    
    2. **后期收敛慢**：训练后期 $t$ 大，$\eta_{eff}$ 小，参数更新极慢
    
    3. **长期训练停滞**：当 $t$ 很大（如 $t = 10^6$），$\eta_{eff} \approx \frac{\eta}{g \cdot 1000}$，学习率极小，训练几乎停滞
    
    **数值示例**：设 $\eta = 0.1$, $g = 1$：
    
    | $t$ | $\eta_{eff}$ | 参数更新幅度 |
    |:---:|:-----------|:-----------|
    | 1 | 0.1 | 大 |
    | 100 | 0.01 | 中 |
    | 10000 | 0.001 | 小 |
    | $10^6$ | 0.0001 | 极小 |
    
    **结论**：
    
    AdaGrad 的学习率单调递减导致：
    - 适合短期训练或稀疏梯度问题（频繁更新参数获得小学习率，稀疏参数获得大学习率）
    - 不适合长期训练：学习率过早衰减，后期训练停滞
    - RMSprop 通过滑动平均解决了这个问题
    
    **改进方向**：
    
    RMSprop 使用指数滑动平均：
    
    $$E_t = \gamma E_{t-1} + (1-\gamma) g^2$$
    
    当梯度恒为 $g$：
    
    $$E_t = (1-\gamma) g^2 \sum_{i=0}^{t-1} \gamma^i = (1-\gamma) g^2 \frac{1-\gamma^t}{1-\gamma} \approx g^2$$
    
    $E_t$ 收敛到 $g^2$（稳定值），而非累积增长。有效学习率 $\eta_{eff} = \frac{\eta}{\sqrt{g^2}} = \frac{\eta}{g}$ 保持稳定。
    
    **总结**：AdaGrad 学习率单调递减源于累积梯度平方的累积增长。这个特性使 AdaGrad 适合短期训练和稀疏梯度，但长期训练会停滞。RMSprop 使用滑动平均避免累积，学习率稳定，适合长期训练。
    </details>

2. 解释 Adam 偏差修正的必要性。设初始化 $\mathbf{m}_0 = 0$, $\beta_1 = 0.9$，梯度 $\nabla L_1 = 10$。计算未修正的 $\mathbf{m}_1$ 和修正后的 $\hat{\mathbf{m}}_1$，分析差异。
    <details>
    <summary>参考答案</summary>
    
    **偏差修正计算**：
    
    Adam 一阶矩估计：
    
    $$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L_t$$
    
    设 $\beta_1 = 0.9$, $m_0 = 0$, $\nabla L_1 = 10$：
    
    **未修正**：
    
    $$m_1 = 0.9 \cdot 0 + 0.1 \cdot 10 = 1$$
    
    **修正后**：
    
    $$\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{1}{1 - 0.9} = \frac{1}{0.1} = 10$$
    
    **差异分析**：
    
    - 未修正 $m_1 = 1$（偏向零）
    - 修正后 $\hat{m}_1 = 10$（等于实际梯度）
    - 修正因子 $\frac{1}{1-\beta_1^t} = 10$ 放大了 $m_1$
    
    **偏差原因**：
    
    Adam 初始化 $m_0 = 0$，一阶矩估计是历史梯度的加权平均：
    
    $$m_t = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} \nabla L_i$$
    
    权重之和：
    
    $$\sum_{i=1}^{t} (1-\beta_1) \beta_1^{t-i} = (1-\beta_1) \frac{1-\beta_1^t}{1-\beta_1} = 1 - \beta_1^t$$
    
    当 $t$ 小（训练初期），权重之和 $1 - \beta_1^t < 1$：
    
    - $t=1$: 权重之和 $= 0.1$
    - $t=2$: 权重之和 $= 0.19$
    - $t=10$: 权重之和 $= 0.65$
    
    权重之和小于 1，估计偏向零（因为初始化 $m_0 = 0$ 占据了缺失的权重）。
    
    **修正原理**：
    
    偏差修正抵消初始化偏差：
    
    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
    
    将估计乘以 $\frac{1}{1-\beta_1^t}$，补足缺失的权重。
    
    当 $t$ 大（训练后期），$\beta_1^t \to 0$：
    
    - $t=100$: $\beta_1^{100} \approx 0$
    - 修正因子 $\frac{1}{1-\beta_1^{100}} \approx 1$
    - $\hat{m}_{100} \approx m_{100}$
    
    修正影响消失。
    
    **数值示例**：设梯度恒为 10：
    
    | $t$ | $m_t$（未修正） | $\hat{m}_t$（修正） | 比值 |
    |:---:|:-------------|:-----------------|:---:|
    | 1 | 1 | 10 | 10x |
    | 5 | 4.1 | 6.9 | 1.7x |
    | 10 | 6.5 | 10 | 1.5x |
    | 100 | 9.99 | 10 | 1x |
    
    **结论**：
    
    Adam 偏差修正的必要性：
    1. 训练初期（$t$ 小）：初始化 $m_0=0$ 导致估计偏向零，修正放大估计，抵消偏差
    2. 训练后期（$t$ 大）：权重之和接近 1，偏差消失，修正影响减弱
    3. 偏差修正使训练初期梯度估计准确，避免冷启动问题
    
    **实际意义**：
    
    无偏差修正的 Adam 在训练初期学习率可能过小（因为 $m_t$ 偏向零），参数更新慢。偏差修正使初期学习率正常，加速收敛。
    
    这就是为什么 Adam 的偏差修正是关键设计 —— 它解决了初始化偏差导致的冷启动问题。
    </details>

3. 解释 AdamW 为何比 Adam 的权重衰减效果更稳定。分析 Adam 中 L2 正则化梯度被自适应学习率缩放的问题。
    <details>
    <summary>参考答案</summary>
    
    **Adam 中权重衰减的实现问题**：
    
    L2 正则化在损失函数中添加惩罚项：
    
    $$L_{total} = L_{data} + \lambda ||\mathbf{W}||^2$$
    
    梯度变为：
    
    $$\nabla L_{total} = \nabla L_{data} + 2\lambda \mathbf{W}$$
    
    Adam 累积梯度到一阶矩 $m_t$ 和二阶矩 $v_t$：
    
    $$m_t = \beta_1 m_{t-1} + (1-\beta_1)(\nabla L_{data} + 2\lambda \mathbf{W})$$
    $$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L_{data} + 2\lambda \mathbf{W})^2$$
    
    参数更新：
    
    $$\Delta \mathbf{W} = -\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$
    
    权重衰减项 $2\lambda \mathbf{W}$ 被包含在 $\hat{m}_t$ 中，并被自适应学习率 $\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$ 缩放。
    
    **问题分析**：
    
    当某参数梯度历史活跃（$\hat{v}_t$ 大）：
    
    - 自适应学习率 $\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$ 小
    - 权重衰减项 $2\lambda \mathbf{W}$ 被缩小，正则化效果减弱
    
    当某参数梯度历史稀疏（$\hat{v}_t$ 小）：
    
    - 自适应学习率 $\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$ 大
    - 权重衰减项 $2\lambda \mathbf{W}$ 被放大，正则化效果增强
    
    这与 L2 正则化"均匀衰减所有权重"的设计初衷不符。L2 正则化应使所有权重同等衰减（每步乘以 $1 - \eta \lambda$），但 Adam 使权重衰减效果与参数的梯度历史相关。
    
    **数值示例**：
    
    设 $\eta = 0.001$, $\lambda = 0.01$, $\epsilon = 10^{-8}$:
    
    | 参数 | $\hat{v}_t$ | 自适应学习率 | 权重衰减幅度 |
    |:----|:----------|:-----------|:-----------|
    | 活跃参数 | 100 | $\frac{0.001}{10} = 10^{-4}$ | $10^{-4} \cdot 0.02 \approx 0$ |
    | 稀疏参数 | 1 | $\frac{0.001}{1} = 10^{-3}$ | $10^{-3} \cdot 0.02 = 2 \times 10^{-5}$ |
    
    活跃参数权重衰减几乎为 0，稀疏参数权重衰减较大。正则化效果不均匀。
    
    **AdamW 的解耦设计**：
    
    AdamW 将权重衰减从梯度更新中分离：
    
    $$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \lambda \mathbf{W}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$
    
    权重衰减项 $-\eta \lambda \mathbf{W}_t$ 直接作用于参数，不被自适应学习率缩放。
    
    每步权重乘以 $(1 - \eta \lambda)$：
    
    $$\mathbf{W}_{t+1} = \mathbf{W}_t(1 - \eta \lambda) - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$
    
    所有权重同等衰减，与 SGD 的行为一致。
    
    **AdamW vs Adam 权重衰减对比**：
    
    | 特性 | Adam | AdamW |
    |:-----|:-----|:------|
    | 权重衰减实现 | 梯度中添加 L2 项 | 直接衰减权重 |
    | 衰减效果 | 受 $\hat{v}_t$ 影响（不均匀） | 稳定均匀 |
    | 超参数耦合 | $\eta$ 和 $\lambda$ 耦合 | 解耦独立 |
    
    **结论**：
    
    Adam 的权重衰减问题源于 L2 正则化梯度被自适应学习率缩放。活跃参数衰减效果减弱，稀疏参数衰减效果增强，正则化不均匀。
    
    AdamW 将权重衰减解耦，直接作用于参数，衰减效果稳定均匀。实验表明 AdamW 泛化能力优于 Adam，尤其在需要强正则化的任务上。
    
    **建议**：优先使用 AdamW 替代 Adam，权重衰减效果更稳定可靠。
    </details>
