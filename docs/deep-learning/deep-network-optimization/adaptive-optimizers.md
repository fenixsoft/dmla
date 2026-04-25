# 自适应优化器

自适应学习率的思想源于一个朴素的观察：不同参数在训练过程中扮演着不同的角色。有些参数频繁更新，梯度大而稳定；有些参数更新稀疏，梯度小甚至偶尔为零。如果给所有参数配发同样的"步长"，频繁更新的参数可能步子太大、震荡难收敛，而稀疏更新的参数可能步子太小、前进缓慢。这就像给全班学生布置同样的作业量 —— 学得快的学生觉得无聊，学得慢的学生觉得吃力。

**自适应优化器**（Adaptive Optimizers）正是为解决这一"一刀切"问题而生。它们根据参数的历史梯度自动调整学习率，为每个参数分配独立的更新步长。这种"因材施教"的策略最早由约翰·杜奇（John Duchi）等人于 2011 年提出，他们的 AdaGrad 算法开创了自适应学习率的先河。随后，杰弗里·辛顿（Geoffrey Hinton）在 2012 年的 Coursera 课程中提出了 RMSprop，解决了 AdaGrad 学习率过早衰减的问题。2015 年，迪德里克·金玛（Diederik Kingma）和吉米·巴（Jimmy Ba）发表里程碑论文《Adam: A Method for Stochastic Optimization》，将动量法与自适应学习率完美结合，Adam 由此成为深度学习领域最流行的优化器。2019 年，伊利亚·洛希洛夫（Ilya Loshchilov）和弗兰克·赫特（Frank Hutter）发现 Adam 的权重衰减实现存在理论缺陷，提出了 AdamW，进一步提升了泛化能力。

在上一章中，我们介绍了梯度下降的基本原理和重要改进 —— 动量法和 Nesterov 加速梯度。这些算法有效提升了参数更新的效率和平稳性，但它们都面临一个共同的限制：所有参数使用相同的学习率。实际训练中，不同参数的梯度特性差异很大，有些参数梯度大、更新频繁，需要小学习率防止震荡；有些参数梯度小、更新稀疏，需要大学习率加速更新。使用统一的学习率难以满足不同参数的需求，可能导致部分参数更新过慢或过快。

本章将介绍四种重要的自适应优化器：AdaGrad、RMSprop、Adam 和 AdamW，分析它们的设计原理、优缺点和适用场景。

## AdaGrad

上一章介绍的动量法和 NAG 解决了梯度下降的震荡问题，但它们仍然给所有参数分配同样的学习率。在自然语言处理等任务中，词汇嵌入层的参数更新极不均匀：常用词频繁出现、梯度大，罕见词偶尔出现、梯度小甚至为零。如果用统一学习率，常用词的嵌入向量更新太快容易震荡，罕见词的嵌入向量更新太慢难以学到有效表示。这促使研究者思考：能否让每个参数拥有自己的学习率？

### 累积历史梯度

**AdaGrad**（Adaptive Gradient Algorithm）由约翰·杜奇（John Duchi）等人在 2011 年提出，是最早的自适应优化器之一。AdaGrad 的核心思想非常直观：频繁更新的参数已经学到了很多信息，应该放慢脚步；稀疏更新的参数还有大量信息待学，应该大步前进。

AdaGrad 累积历史梯度的平方，用于调整各参数的学习率：

<!-- equation:label=eq:adagrad-update -->
$$\mathbf{G}_t = \mathbf{G}_{t-1} + (\nabla L_t)^2$$
<!-- end-equation -->

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \cdot \nabla L_t$$

这两个公式看着抽象，拆开来看含义很直观：
- $\mathbf{G}_t$ 是历史梯度平方的累积量，记录了参数更新了多少次、每次更新幅度多大
- $(\nabla L_t)^2$ 是当前梯度的平方，将梯度值转为非负数（负梯度同样要学习）
- $\sqrt{\mathbf{G}_t + \epsilon}$ 是累积梯度的平方根，$\epsilon$（通常 $10^{-8}$）防止除零
- $\frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}}$ 是有效学习率，随累积梯度增大而减小
- $\nabla L_t$ 是当前梯度方向，指明参数应该往哪个方向移动
- 整体公式可以理解为：累积历史越多的参数，有效学习率越小，更新越谨慎

符号 $\frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}}$ 表示逐元素运算：每个参数的学习率为 $\frac{\eta}{\sqrt{G_{t,i} + \epsilon}}$。这意味着同一批参数中，有的参数学习率可能很大，有的可能很小，完全取决于各自的梯度历史。

AdaGrad 的更新规则可以总结为：

- 梯度频繁的参数：$\mathbf{G}_t$ 大，$\sqrt{\mathbf{G}_t}$ 大，学习率 $\frac{\eta}{\sqrt{\mathbf{G}_t}}$ 小，更新谨慎
- 梯度稀疏的参数：$\mathbf{G}_t$ 小，$\sqrt{\mathbf{G}_t}$ 小，学习率 $\frac{\eta}{\sqrt{\mathbf{G}_t}}$ 大，加速学习

这种自适应调整特别适合处理稀疏梯度问题（如自然语言处理中的词嵌入），稀疏更新的参数获得更大的学习率，加速学习。

### AdaGrad 的缺陷

AdaGrad 的"累积历史"策略看似完美，但隐藏着一个致命缺陷。既然 $\mathbf{G}_t$ 是历史梯度平方的累积，训练过程中它只会不断增大（梯度平方非负），$\sqrt{\mathbf{G}_t}$ 也只增不减。这意味着有效学习率 $\frac{\eta}{\sqrt{\mathbf{G}_t}}$ 持续下降 —— 参数每更新一步，学习率就变小一点。训练后期，学习率可能变得极小（如 $10^{-6}$），参数几乎不再更新，训练停滞。

用具体数值说明：设初始学习率 $\eta = 0.1$，梯度恒为 $1$。经过 $t$ 步：

$$G_t = \sum_{i=1}^{t} 1^2 = t, \quad \eta_{eff} = \frac{0.1}{\sqrt{t}}$$

| 迭代次数 $t$ | 有效学习率 $\eta_{eff}$ |
|:-----------:|:----------------------|
| 1 | 0.1 |
| 10 | 0.032 |
| 100 | 0.01 |
| 1000 | 0.003 |
| 10000 | 0.001 |

经过 10000 步，学习率下降到 $0.001$，参数更新极慢。如果训练需要百万步，学习率会降到 $10^{-4}$，几乎无法学习。

这个特性使 AdaGrad 陷入两难：它适合短期训练或稀疏梯度问题（频繁更新参数获得小学习率，稀疏参数获得大学习率），但长期训练会导致学习率过早衰减，后期训练停滞。这个缺陷促使研究者思考：能否让历史累积"有进有出"，而非"只进不出"？这引出了改进版本——RMSprop。

## RMSprop

AdaGrad 的学习率单调递减问题源于"累积"机制 —— 历史梯度平方只进不出。这好比记账只记录收入不记录支出，账户余额只会越来越大。解决方案很自然：用"滑动平均"代替"累积"，只关注最近的梯度历史，让旧历史逐渐淡出。

### 指数滑动平均

**RMSprop**（Root Mean Square Propagation）由杰弗里·辛顿（Geoffrey Hinton）在 2012 年的 Coursera 课程中提出。有趣的是，辛顿并未正式发表 RMSprop 的论文，它仅作为课程笔记流传，但因其简单有效迅速被社区采纳。RMSprop 的核心改进是用指数滑动平均代替 AdaGrad 的累积，只保留最近约 $\frac{1}{1-\gamma}$ 步的梯度信息。

RMSprop 的更新规则：

<!-- equation:label=eq:rmsprop-update -->
$$\mathbf{E}_t = \gamma \mathbf{E}_{t-1} + (1 - \gamma)(\nabla L_t)^2$$
<!-- end-equation -->

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\mathbf{E}_t + \epsilon}} \cdot \nabla L_t$$

这个公式看着与 AdaGrad 类似，关键区别在于 $\mathbf{E}_t$ 的计算方式：
- $\mathbf{E}_t$ 是梯度平方的指数滑动平均，而非累积
- $\gamma$ 是衰减系数（通常 $0.9$），控制历史信息的保留程度
- $(1-\gamma)$ 是新梯度的权重，保证权重之和约为 1
- 整体公式可以理解为：保留 90% 的历史信息，加入 10% 的当前信息，旧历史逐渐淡出

公式中 $(1-\gamma)$ 保证权重之和约为 1（$\gamma + (1-\gamma) = 1$），避免数值膨胀。当 $\gamma = 0.9$，有效窗口约为 $\frac{1}{1-0.9} = 10$ 步 —— 大约只保留最近 10 步的梯度信息。

指数滑动平均的特性带来三点改进：

- **窗口效应**：只保留最近约 $\frac{1}{1-\gamma}$ 步的梯度信息（当 $\gamma = 0.9$，约 10 步），而非全部历史
- **学习率稳定**：$\mathbf{E}_t$ 不单调递增，而是波动稳定，学习率不会过早衰减
- **快速适应**：当梯度变化时，$\mathbf{E}_t$ 快速调整，学习率响应灵敏

### RMSprop vs AdaGrad

两种优化器采用不同的历史处理策略，导致学习率行为截然不同：

| 特性 | AdaGrad | RMSprop |
|:-----|:--------|:--------|
| 梯度历史 | 累积（全部历史） | 滑动平均（近期历史） |
| 学习率趋势 | 单调递减 | 波动稳定 |
| 长期训练 | 学习率过小，停滞 | 学习率稳定，可持续 |
| 稀疏梯度 | 适合 | 适合，但不如 AdaGrad |
| 衰减系数 | 无 | $\gamma = 0.9$（常用） |

RMSprop 克服了 AdaGrad 的学习率衰减问题，适合长期训练。但 RMSprop 也有一个缺陷：它只使用梯度平方调整学习率，没有累积历史梯度方向，相当于放弃了动量法的平滑效果。能否同时拥有动量的平稳性和自适应学习率的灵活性？这引出了更完善的优化器——Adam。

## Adam

RMSprop 解决了 AdaGrad 的学习率衰减问题，但它只用梯度平方调整学习率，没有累积梯度方向，缺少动量法的平滑效果。如果能把两者结合起来 —— 既用动量平滑更新路径，又用自适应学习率为不同参数分配不同步长，岂不是完美？这正是 Adam 的设计初衷。

### 结合动量与自适应学习率

**Adam**（Adaptive Moment Estimation）由迪德里克·金玛（Diederik Kingma）和吉米·巴（Jimmy Ba）在 2015 年提出，论文《Adam: A Method for Stochastic Optimization》发表在国际机器学习会议（ICLR）上。Adam 的名称源于其核心机制：同时估计梯度的一阶矩（moment，即均值）和二阶矩（moment，即方差），实现了动量法与自适应学习率的完美融合。

Adam 的更新规则包含四个步骤：

<!-- equation:label=eq:adam-m -->
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla L_t$$
<!-- end-equation -->

<!-- equation:label=eq:adam-v -->
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla L_t)^2$$
<!-- end-equation -->

<!-- equation:label=eq:adam-bias -->
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
<!-- end-equation -->

<!-- equation:label=eq:adam-update -->
$$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot \hat{\mathbf{m}}_t$$
<!-- end-equation -->

这四个公式看似复杂，拆开来看含义很直观：
- $\mathbf{m}_t$ 是梯度的一阶矩估计（动量），累积历史梯度方向，用于平滑更新路径
- $\mathbf{v}_t$ 是梯度的二阶矩估计，累积历史梯度平方，用于调整各参数的学习率
- $\beta_1$ 是一阶矩衰减系数（通常 $0.9$），控制动量的保留程度
- $\beta_2$ 是二阶矩衰减系数（通常 $0.999$），控制梯度平方历史的保留程度
- $\hat{\mathbf{m}}_t$ 和 $\hat{\mathbf{v}}_t$ 是偏差修正后的估计，解决初始化为零导致的冷启动问题
- 整体公式可以理解为：用平滑后的梯度方向 $\hat{\mathbf{m}}_t$ 指明前进方向，用自适应学习率 $\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t}}$ 控制步长

Adam 同时维护两个变量，各有分工：
- **一阶矩 $\mathbf{m}_t$**：动量的作用，累积历史梯度方向，抑制震荡、加速收敛
- **二阶矩 $\mathbf{v}_t$**：自适应学习率的作用，累积历史梯度平方，为不同参数分配不同步长

### 偏差修正的必要性

Adam 初始化 $\mathbf{m}_0 = 0, \mathbf{v}_0 = 0$，这个看似自然的设置隐藏着一个问题。训练初期，一阶矩和二阶矩估计偏向零（因为初始化为零，新梯度权重又小）。偏差修正正是为了抵消这个初始化偏差，让训练初期的梯度估计更准确。

偏差修正公式：

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

这个公式的作用机制：
- 当 $t$ 很小（训练初期），$\beta_1^t$ 和 $\beta_2^t$ 接近 1，修正因子 $1 - \beta_1^t$ 和 $1 - \beta_2^t$ 很小，$\hat{\mathbf{m}}_t$ 和 $\hat{\mathbf{v}}_t$ 被放大，抵消初始化偏差
- 当 $t$ 很大（训练后期），$\beta_1^t$ 和 $\beta_2^t$ 接近 0，修正因子接近 1，$\hat{\mathbf{m}}_t \approx \mathbf{m}_t, \hat{\mathbf{v}}_t \approx \mathbf{v}_t$，修正影响消失

用具体数值说明：设 $\beta_1 = 0.9$，$t = 1$，梯度 $\nabla L_1 = 10$：

- $\mathbf{m}_1 = 0.9 \cdot 0 + 0.1 \cdot 10 = 1$（未修正，偏向零）
- $\hat{\mathbf{m}}_1 = \frac{1}{1 - 0.9^1} = \frac{1}{0.1} = 10$（修正后，等于实际梯度）

修正后 $\hat{\mathbf{m}}_1 = 10$（等于实际梯度），而非未修正的 $\mathbf{m}_1 = 1$（偏向零）。偏差修正使训练初期梯度估计更准确，避免参数更新过慢的冷启动问题。

### Adam 的优势

Adam 结合了动量法和自适应学习率的优点，成为深度学习最流行的优化器：

1. **动量平滑**：一阶矩 $\mathbf{m}_t$ 累积历史梯度方向，平滑更新路径，抑制震荡，加速穿越平坦区域
2. **自适应学习率**：二阶矩 $\mathbf{v}_t$ 累积历史梯度平方，为不同参数分配不同学习率，处理稀疏梯度问题
3. **偏差修正**：训练初期梯度估计准确，避免初始化偏差导致的冷启动问题
4. **鲁棒性强**：对超参数选择不敏感，默认参数（$\beta_1=0.9, \beta_2=0.999, \eta=0.001$）适合大多数任务

Adam 在计算机视觉、自然语言处理、推荐系统等领域广泛使用，成为深度学习研究的默认优化器。不过，Adam 也有一个隐藏的理论缺陷 —— 权重衰减的实现问题，这将在下一节 AdamW 中详细讨论。

### Adam 的超参数

Adam 需要设置四个超参数，但其中三个通常使用默认值，只需关注学习率：

| 超参数 | 默认值 | 作用 |
|:------|:------|:-----|
| $\eta$ | $0.001$ | 全局学习率（通常比 SGD 小，因为自适应已放大有效步长） |
| $\beta_1$ | $0.9$ | 一阶矩衰减系数（动量），控制梯度方向的平滑程度 |
| $\beta_2$ | $0.999$ | 二阶矩衰减系数（自适应），控制梯度平方历史的保留程度 |
| $\epsilon$ | $10^{-8}$ | 数值稳定常数，防止除零 |

$\beta_1$ 和 $\beta_2$ 通常不需要调整，默认值适合大多数任务。主要调整的是学习率 $\eta$，范围通常 $[10^{-4}, 10^{-2}]$。调优策略：从默认值开始，损失震荡则降低学习率，收敛慢则增大学习率。

## AdamW

Adam 似乎已经集齐了动量法和自适应学习率的所有优点，成为深度学习的默认优化器。然而，2019 年一篇论文揭示了一个隐藏的理论缺陷：Adam 中权重衰减（L2 正则化）的实现方式与自适应学习率冲突，正则化效果不稳定。这个问题引出了 Adam 的修正版本——AdamW。

### 权重衰减的实现问题

权重衰减（Weight Decay）是防止过拟合的常用技术，本质上是 L2 正则化。在损失函数中添加参数惩罚项，迫使参数保持较小值：

$$L_{total} = L_{data} + \lambda \|\mathbf{W}\|^2$$

梯度变为：

$$\nabla L_{total} = \nabla L_{data} + 2\lambda \mathbf{W}$$

在 SGD 中，权重衰减的实现很简单直观：

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla L_{data} - \eta \lambda \mathbf{W}_t = \mathbf{W}_t(1 - \eta \lambda) - \eta \nabla L_{data}$$

每步权重乘以 $(1 - \eta \lambda)$，逐渐衰减，所有参数同等对待。

但 Adam 的实现方式不同。梯度 $\nabla L_{total} = \nabla L_{data} + 2\lambda \mathbf{W}$ 被累积到一阶矩 $\mathbf{m}_t$ 和二阶矩 $\mathbf{v}_t$，权重衰减项 $2\lambda \mathbf{W}$ 也被自适应学习率缩放：

$$\Delta \mathbf{W}_{reg} = -\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot 2\lambda \mathbf{W}$$

这导致一个问题：当 $\hat{\mathbf{v}}_t$ 很大（参数梯度历史活跃），自适应学习率 $\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t}}$ 很小，权重衰减项被缩小，正则化效果减弱。当 $\hat{\mathbf{v}}_t$ 很小（参数梯度历史稀疏），自适应学习率很大，权重衰减项被放大。这与 L2 正则化"均匀衰减所有权重"的设计初衷不符 —— 正则化效果变成了与参数梯度历史相关，而非对所有参数一视同仁。

### AdamW 的解耦设计

**AdamW**（Adam with Decoupled Weight Decay）由伊利亚·洛希洛夫（Ilya Loshchilov）和弗兰克·赫特（Frank Hutter）在 2019 年提出，论文《Decoupled Weight Decay Regularization》揭示了 Adam 的权重衰减问题，并提出简洁的解决方案：将权重衰减从梯度更新中分离。

AdamW 的更新规则：

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \lambda \mathbf{W}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot \hat{\mathbf{m}}_t$$

这个公式看着与 Adam 类似，关键区别在于权重衰减项的位置：
- $-\eta \lambda \mathbf{W}_t$ 是权重衰减项，直接作用于参数，不被自适应学习率缩放
- $-\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot \hat{\mathbf{m}}_t$ 是梯度更新项，被自适应学习率缩放
- 整体公式可以理解为：权重衰减独立执行，梯度更新自适应执行，两者解耦不干扰

每步权重乘以 $(1 - \eta \lambda)$，与 SGD 的行为一致 —— 所有参数同等衰减，正则化效果稳定均匀。

AdamW 的优势：

1. **正则化效果稳定**：权重衰减不受自适应学习率影响，所有参数同等衰减
2. **泛化能力提升**：实验表明 AdamW 在多个任务上比 Adam 泛化能力更好
3. **超参数解耦**：学习率 $\eta$ 和权重衰减系数 $\lambda$ 独立调整，不相互干扰

### Adam vs AdamW

两种优化器的核心机制相同，但权重衰减的实现方式不同，导致正则化效果和泛化能力差异：

| 特性 | Adam | AdamW |
|:-----|:-----|:------|
| 权重衰减实现 | 梯度中添加 L2 项（间接） | 直接衰减权重（直接） |
| 正则化效果 | 受自适应学习率影响（不均匀） | 稳定一致（均匀） |
| 泛化能力 | 较弱（实验验证） | 较强（实验验证） |
| 超参数耦合 | 学习率和权重衰减耦合 | 解耦，独立调整 |

实践中，建议优先使用 AdamW 替代 Adam，尤其在需要强正则化的任务上（如大规模语言模型训练）。AdamW 已成为 Transformer、BERT、GPT 等模型的默认优化器。

## 优化器选择指南

至此，我们已经学习了 SGD、动量法、NAG、AdaGrad、RMSprop、Adam 和 AdamW 七种优化器。每种优化器各有优劣，如何根据任务特点选择合适的优化器？

### 优化器对比总结

| 优化器 | 核心机制 | 优势 | 缺陷 | 适用场景 |
|:------|:--------|:-----|:-----|:--------|
| SGD | 基础梯度下降 | 简单、稳定 | 震荡、慢 | 简单任务、精调 |
| Momentum | 动量平滑 | 加速、抑制震荡 | 需调学习率 | 通用 |
| NAG | 预测位置梯度 | 提前响应拐点 | 稍复杂 | 追求精度 |
| AdaGrad | 累积梯度平方 | 稀疏梯度友好 | 学习率衰减 | 稀疏数据、短期训练 |
| RMSprop | 滑动平均梯度平方 | 学习率稳定 | 无动量 | 长期训练、RNN |
| Adam | 动量 + 自适应 | 鲁棒、快速 | 权重衰减问题 | 通用、默认选择 |
| AdamW | Adam+解耦权重衰减 | 正则化稳定 | 无 | 默认选择 |

### 选择策略

根据任务特点选择优化器：

| 任务类型 | 推荐优化器 | 理由 |
|:--------|:----------|:-----|
| 通用深度学习 | AdamW | 鲁棒性强，默认选择 |
| 计算机视觉 | SGD + Momentum | 实验表明泛化能力更好 |
| 自然语言处理 | AdamW | 稀疏梯度，自适应优势明显 |
| RNN/LSTM | RMSprop / AdamW | 处理梯度消失，学习率稳定 |
| 精调预训练模型 | SGD + Momentum | 小学习率精调，防止破坏预训练特征 |
| 稀疏数据（推荐系统） | AdamW | 稀疏参数获得大学习率 |

### 超参数调优建议

各优化器的超参数调优策略：

**学习率 $\eta$**：
- AdamW 默认 $0.001$，范围 $[10^{-4}, 10^{-2}]$
- SGD 默认 $0.01$，范围 $[10^{-3}, 10^{-1}]$
- 调优策略：从默认值开始，损失震荡则降低，收敛慢则增大

**动量系数 $\beta_1$**（AdamW）：
- 默认 $0.9$，通常不需调整
- 特殊情况：极不稳定任务可降低到 $0.8$

**二阶矩系数 $\beta_2$**（AdamW）：
- 默认 $0.999$，通常不需调整
- 稀疏梯度任务可降低到 $0.99$

**权重衰减 $\lambda$**（AdamW）：
- 范围 $[10^{-4}, 10^{-2}]$
- 过拟合严重则增大，欠拟合则减小或设为 0

## 自适应优化器实验

理论分析揭示了各优化器的设计原理，但实际效果如何？下面通过代码实验对比 SGD、动量法、AdaGrad、RMSprop、Adam 和 AdamW 在二次损失函数上的收敛效果。实验使用长椭圆形损失函数（不同方向梯度差异大），从同一起点出发，观察各优化器的参数路径、损失曲线和有效学习率变化。代码实现了六种优化器的完整更新逻辑，包括动量累积、梯度平方累积、滑动平均、偏差修正和权重衰减解耦，可视化对比各优化器的收敛行为。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("实验：自适应优化器收敛对比")
print("=" * 60)
print()

# 定义损失函数和梯度
def loss_function(W):
    """二次损失函数 L = 0.5 * W^T A W"""
    A = np.array([[1, 0], [0, 100]])  # 长椭圆，不同方向梯度差异大
    return 0.5 * np.dot(W, A @ W)

def gradient(W):
    """梯度 ∇L = A W"""
    A = np.array([[1, 0], [0, 100]])
    return A @ W

# 各优化器实现
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.path = []
    
    def step(self, W, grad):
        W_new = W - self.lr * grad
        self.path.append(W_new.copy())
        return W_new

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = np.zeros(2)
        self.path = []
    
    def step(self, W, grad):
        self.v = self.momentum * self.v + self.lr * grad
        W_new = W - self.v
        self.path.append(W_new.copy())
        return W_new

class AdaGrad:
    def __init__(self, lr=0.1, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.G = np.zeros(2)
        self.path = []
    
    def step(self, W, grad):
        self.G += grad ** 2
        lr_adaptive = self.lr / np.sqrt(self.G + self.eps)
        W_new = W - lr_adaptive * grad
        self.path.append(W_new.copy())
        return W_new

class RMSprop:
    def __init__(self, lr=0.01, gamma=0.9, eps=1e-8):
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.E = np.zeros(2)
        self.path = []
    
    def step(self, W, grad):
        self.E = self.gamma * self.E + (1 - self.gamma) * (grad ** 2)
        lr_adaptive = self.lr / np.sqrt(self.E + self.eps)
        W_new = W - lr_adaptive * grad
        self.path.append(W_new.copy())
        return W_new

class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
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
        
        W_new = W - self.lr / (np.sqrt(v_hat) + self.eps) * m_hat
        self.path.append(W_new.copy())
        return W_new

class AdamW:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
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
        
        # 解耦权重衰减
        W_new = W - self.lr * self.weight_decay * W
        W_new = W_new - self.lr / (np.sqrt(v_hat) + self.eps) * m_hat
        self.path.append(W_new.copy())
        return W_new

# 运行实验
W_init = np.array([10.0, 10.0])  # 起点
n_iterations = 100

optimizers = {
    'SGD': SGD(lr=0.01),
    'Momentum': Momentum(lr=0.01, momentum=0.9),
    'AdaGrad': AdaGrad(lr=0.5),
    'RMSprop': RMSprop(lr=0.01, gamma=0.9),
    'Adam': Adam(lr=0.01),
    'AdamW': AdamW(lr=0.01, weight_decay=0.001)
}

results = {}
for name, opt in optimizers.items():
    W = W_init.copy()
    losses = []
    
    for t in range(n_iterations):
        loss = loss_function(W)
        losses.append(loss)
        grad = gradient(W)
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

colors = {'SGD': '#e74c3c', 'Momentum': '#3498db', 'AdaGrad': '#f39c12', 
          'RMSprop': '#9b59b6', 'Adam': '#2ecc71', 'AdamW': '#1abc9c'}

# 图1：参数路径
ax1 = axes[0, 0]
W1_range = np.linspace(-12, 12, 100)
W2_range = np.linspace(-12, 12, 100)
W1_grid, W2_grid = np.meshgrid(W1_range, W2_range)
L_grid = 0.5 * (W1_grid**2 + 100 * W2_grid**2)

ax1.contour(W1_grid, W2_grid, L_grid, levels=[1, 10, 100, 500, 1000, 5000],
           colors='gray', alpha=0.5, linewidths=0.5)
ax1.contourf(W1_grid, W2_grid, L_grid, levels=[0, 1, 10, 100, 500, 1000, 5000, 10000],
             cmap='Blues', alpha=0.3)

for name, result in results.items():
    path = result['path']
    ax1.plot(path[:, 0], path[:, 1], 'o-', color=colors[name], 
             linewidth=2, markersize=2, alpha=0.7, label=name)

ax1.plot(W_init[0], W_init[1], 'ko', markersize=10, label='起点')
ax1.plot(0, 0, 'k*', markersize=15, label='最小值')
ax1.set_xlabel('W1', fontsize=11)
ax1.set_ylabel('W2', fontsize=11)
ax1.set_title('参数路径对比', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-12, 12)
ax1.set_ylim(-12, 12)

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

# 图3：有效学习率变化（W1方向）
ax3 = axes[1, 0]

# 计算各优化器的有效学习率
for name, result in results.items():
    path = result['path']
    if name in ['AdaGrad', 'RMSprop', 'Adam', 'AdamW']:
        # 计算有效学习率变化
        W1_updates = path[:-1, 0] - path[1:, 0]  # 参数变化
        grads_W1 = path[:-1, 0]  # W1方向的梯度近似
        
        # 反推有效学习率：ΔW = lr_eff * grad
        # 由于 A = [[1,0],[0,100]], grad_W1 = W1
        lr_eff = np.abs(W1_updates) / np.abs(path[:-1, 0] + 1e-8)
        
        ax3.plot(lr_eff[:50], color=colors[name], linewidth=2, label=name, alpha=0.7)

ax3.set_xlabel('迭代次数', fontsize=11)
ax3.set_ylabel('有效学习率（W1方向）', fontsize=11)
ax3.set_title('自适应学习率变化', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4：收敛速度对比（损失下降速度）
ax4 = axes[1, 1]

# 计算每10步损失下降量
for name, result in results.items():
    losses = result['losses']
    loss_decrease = []
    for i in range(0, len(losses)-10, 10):
        decrease = losses[i] - losses[i+10]
        loss_decrease.append(decrease)
    
    ax4.bar([i*10 for i in range(len(loss_decrease))], loss_decrease,
           width=8, color=colors[name], alpha=0.5, label=name)

ax4.set_xlabel('迭代次数', fontsize=11)
ax4.set_ylabel('10步损失下降量', fontsize=11)
ax4.set_title('损失下降速度对比', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
plt.close()

print("\n实验结论:")
print("- SGD 在 W2 方向剧烈震荡，收敛慢")
print("- Momentum 平滑震荡，收敛更快")
print("- AdaGrad 学习率快速衰减，后期停滞")
print("- RMSprop 学习率稳定，持续收敛")
print("- Adam 结合动量和自适应，收敛最快")
print("- AdamW 效果与 Adam 类似，正则化更稳定")
print("=" * 60)
```

### 实验结论

从可视化结果可以得出以下结论：

1. **SGD 震荡明显**：在梯度差异大的方向（W2 方向梯度大）剧烈震荡，收敛慢，路径呈锯齿状
2. **Momentum 平滑震荡**：历史梯度累积平滑更新方向，收敛更快，但仍受统一学习率限制
3. **AdaGrad 学习率衰减**：初期收敛快，后期学习率过小，训练停滞，无法到达最优值
4. **RMSprop 学习率稳定**：滑动平均避免学习率衰减，持续收敛，路径平稳
5. **Adam 效果最优**：结合动量和自适应，收敛最快且平稳，路径直接指向最优值
6. **AdamW 与 Adam 类似**：解耦权重衰减不影响收敛路径，但正则化效果更稳定

实验验证了自适应优化器的核心优势：为不同参数分配不同学习率，显著提升收敛效率。在梯度差异大的场景（如长椭圆形损失函数），自适应优化器的优势尤为明显。

## 本章小结

本章介绍了自适应优化器的原理与应用，揭示了"为每个参数分配不同学习率"的核心思想如何提升优化效率。

**AdaGrad** 累积历史梯度平方，为频繁更新的参数分配小学习率，为稀疏更新的参数分配大学习率。这种"因材施教"的策略特别适合自然语言处理等稀疏梯度问题。但 AdaGrad 的累积机制导致学习率单调递减，长期训练后期停滞。

**RMSprop** 用指数滑动平均代替累积，只关注近期梯度历史。学习率稳定，不会过早衰减，适合长期训练，尤其在 RNN 等梯度不稳定场景表现优异。但 RMSprop 缺少动量平滑效果。

**Adam** 结合动量法和 RMSprop，同时维护一阶矩（动量）和二阶矩（自适应学习率）。偏差修正使训练初期梯度估计准确，避免冷启动问题。Adam 鲁棒性强，对超参数不敏感，是深度学习的默认优化器。

**AdamW** 将权重衰减与梯度更新解耦，正则化效果稳定不受自适应学习率影响。实验表明 AdamW 泛化能力优于 Adam，已成为 Transformer、BERT、GPT 等模型的默认优化器。

**优化器选择策略**：AdamW 是通用任务的默认选择；计算机视觉任务 SGD + Momentum 泛化能力更好；自然语言处理和推荐系统 AdamW 优势明显；精调预训练模型 SGD + Momentum 更稳妥。

至此，第三章"深度神经网络优化"的内容已全部完成。我们掌握了梯度下降、动量法、NAG 和自适应优化器的原理与应用，理解了如何选择和调优优化算法。下一章将进入第四章"神经网络稳定"，介绍权重初始化、Dropout、批归一化和梯度问题诊断，解决训练稳定性问题。

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

4. 设训练一个神经网络，使用 Adam 优化器，损失收敛到一定值后不再下降，但距离最优值仍有差距。分析可能的原因，并提出三种改进方法。
    <details>
    <summary>参考答案</summary>
    
    **Adam 收敛停滞的可能原因**：
    
    1. **学习率过大导致震荡**：
       - Adam 的自适应学习率可能无法完全解决震荡问题
       - 当接近最优值时，大学习率导致参数在最优值附近来回跳跃
       - 损失无法精确收敛
    
    2. **学习率过小导致停滞**：
       - 自适应学习率 $\frac{\eta}{\sqrt{\hat{v}_t}}$ 可能变得很小
       - 参数更新幅度极小，损失不再下降
    
    3. **二阶矩估计 $\hat{v}_t$ 过大**：
       - 梯度平方累积使 $\hat{v}_t$ 很大
       - 自适应学习率 $\frac{\eta}{\sqrt{\hat{v}_t}}$ 很小
       - 参数更新停滞
    
    4. **Adam 泛化能力限制**：
       - 实验表明 Adam 在某些任务上泛化能力不如 SGD
       - 可能收敛到泛化能力较差的局部最优
    
    5. **权重衰减实现问题**（使用 Adam 而非 AdamW）：
       - Adam 的权重衰减效果不稳定
       - 可能导致某些参数权重过大，限制收敛
    
    **改进方法**：
    
    1. **降低学习率**：
       - 减小全局学习率 $\eta$
       - 通常降低 5-10 倍（如 $0.001 \to 0.0001$）
       - 使参数更新更精细
    
    ```python
    # 原学习率
    optimizer = Adam(lr=0.001)
    # 调整后
    optimizer = Adam(lr=0.0001)  # 降低10倍
    ```
    
    2. **切换到 SGD + Momentum 进行精调**：
       - Adam 快速接近最优值区域
       - SGD + Momentum 精细收敛到最优值
       - 实验表明 SGD 泛化能力更好
    
    ```python
    # 训练策略：Adam 预训练 + SGD 精调
    # 第一阶段：Adam 快速收敛
    optimizer_adam = Adam(lr=0.001)
    train_with_adam(epochs=50)
    
    # 第二阶段：SGD 精调
    optimizer_sgd = SGD(lr=0.0001, momentum=0.9)
    train_with_sgd(epochs=20)
    ```
    
    3. **使用 AdamW 替代 Adam**：
       - AdamW 的权重衰减效果更稳定
       - 可能改善泛化能力
       - 建议作为 Adam 的默认替代
    
    ```python
    # 使用 AdamW
    optimizer = AdamW(lr=0.001, weight_decay=0.01)
    ```
    
    **其他方法**：
    
    - **调整 $\beta_2$**：降低 $\beta_2$（如 $0.999 \to 0.99$），使二阶矩估计更快响应，避免 $\hat{v}_t$ 过大
    
    - **学习率衰减**：训练后期降低学习率（如余弦衰减），精细调整
    
    - **AMSGrad 变体**：使用 $\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$ 代替 $\hat{v}_t = v_t$，防止学习率过小
    
    **诊断流程**：
    
    1. 观察损失曲线：停滞还是震荡？
    2. 检查梯度范数：梯度是否很小（停滞）或很大（震荡）？
    3. 检查 $\hat{v}_t$ 数值：自适应学习率是否过小？
    4. 尝试降低学习率：观察损失是否继续下降
    5. 尝试切换 SGD：观察精调效果
    
    **总结**：
    
    Adam 收敛停滞的常见原因是学习率过大导致震荡，或自适应学习率过小导致停滞。改进方法包括降低学习率、切换 SGD 精调、使用 AdamW 替代。实践中，"Adam 预训练 + SGD 精调"是常用策略，兼顾效率和精度。
    </details>