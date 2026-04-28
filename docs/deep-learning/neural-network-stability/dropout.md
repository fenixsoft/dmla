# Dropout 正则化

我们曾在线性模型的[正则化](../../statistical-learning/linear-models/regularization-glm.md)章节中讨论过拟合的成因与基本应对策略，学习了使用 L1、L2 正则化通过约束参数大小来限制模型复杂度。然而，深度神经网络有其特殊性，如参数数量巨大（百万级甚至亿级）、训练时间长、层级结构复杂。针对网络参数的正则化方法虽有帮助，但效果已经不足以抑制过拟合现象了，深度网络需要一种更为有力的正则化手段 —— 直接干预网络结构本身。

2014 年，杰弗里·辛顿在论文《Dropout: A Simple Way to Prevent Neural Networks from Overfitting》中提出了一种极其简单却异常有效的正则化方法 **Dropout**。辛顿当时面临一个实际问题，他的团队在 ImageNet 图像分类任务上训练深度网络，无论怎么调参，模型总是过拟合，在训练集上几乎完美，测试集上却差强人意。一天晚上，辛顿突然看到一本书中写到：银行工作人员为了避免贪污，不会让某一个人单独处理一笔大额交易，而是让多人分工协作，每人只负责一部分，这样任何一个人的缺失都不会导致系统崩溃。受此启发，他设计了一个方案，训练时随机丢弃一部分神经元，让每个神经元学会独立工作，不能依赖其他神经元的存在。这个想法看着简单粗暴，却取得了惊人的效果，Dropout 使得 ImageNet 分类准确率大幅提升，迅速成为深度学习的标配技术。

## 深度神经网络的过拟合挑战

深度网络的参数数量往往远超训练样本数，且暂不提下一阶段我们要学习的大语言模型，在 2015 年前后，一个典型的图像分类网络可能有 5000 万参数，而训练数据仅 10 万张图片。这意味着理论上，网络有足够宽容度可以记住每张图片的特征，而非学习图像分类的通用规律。

传统正则化方法（L1、L2 权重衰减）对深度网络有一定帮助，但效果有限。L2 正则化通过惩罚参数平方值来约束权重大小，本质上限制了参数的敏感度，参数值越小，对输入变化的响应越温和。然而，深度网络的过拟合不仅源于参数过大，更源于神经元之间的**共适应**（Co-adaptation），某些神经元只有在其他特定神经元存在时才能发挥作用，由此形成了复杂的依赖关系。L2 正则化无法打破这种依赖，因为它只约束单个参数，不干预神经元之间的协作方式。Dropout 的设计初衷正是解决这个深层问题，通过随机丢弃神经元，迫使每个神经元学会独立工作，不再依赖特定的同伴。这是一种更结构化的正则化手段，不调整参数数值，而是直接干预网络的拓扑结构。

## Dropout 原理

Dropout 的核心思想极其简单，训练时随机关闭一部分神经元，让剩下的神经元独立完成任务。具体来说，Dropout 在训练时对每个神经元进行随机判决，以概率 $p$ 保留该神经元（让其正常工作），以概率 $1-p$ 将其丢弃（输出强制置零）。设神经元原本的输出为 $y$，$r$ 随机开关，服从伯努利分布，决定该神经元是否被丢弃。当 $r=0$ 时，$y_{drop}=0$，神经元被关闭；当 $r=1$ 时，$y_{drop}=y$，神经元正常输出，经过 Dropout 后的网络输出为：

$$y_{drop} = r \cdot y$$

对于隐藏层的一组神经元 $\{h_1, h_2, ..., h_n\}$，每个神经元都有独立的随机变量 $r_i$：

$$h_i^{drop} = r_i \cdot h_i, \quad r_i \sim \text{Bernoulli}(p)$$

被丢弃的神经元（$r_i=0$）不参与前向传播和反向传播，其梯度为零，相当于从网络中暂时消失。这种暂时消失是关键，神经元只是本轮训练被丢弃，下一轮可能又被保留，所有神经元在整个训练过程中都会被反复使用。

### 训练与推理

Dropout 在训练和推理阶段的行为完全不同，差别的根源是要如何保证训练和推理时网络的输出期望一致。假设一个神经元输出为 $y$，训练时 Dropout 后的[期望](../../maths/probability/probability-basics.md#期望)输出为：

$$\mathbb{E}[y_{drop}] = \mathbb{E}[r \cdot y] = \mathbb{E}[r] \cdot y = p \cdot y$$

因为 $r$ 是伯努利变量，其期望 $\mathbb{E}[r]=p$。这意味着训练时该神经元的平均输出只有 $p \cdot y$，比原始输出 $y$ 小了 $p$ 倍。而推理时就不能继续丢弃神经元，那会导致预测结果不稳定（每次推理结果都不同），所以推理时所有神经元都保留（$r=1$），输出为 $y$。但如果直接用 $y$ 作为推理输出，又会比训练期望大 $p$ 倍，导致预测偏差。辛顿给出了两种方案解决这个问题：

- 方案一 **推理时缩放**：训练时输出 $y_{drop} = r \cdot y$，推理时输出缩小为 $y_{test} = p \cdot y$，这样训练期望 $\mathbb{E}[y_{drop}] = p \cdot y$ 与推理输出 $y_{test} = p \cdot y$ 相等。
- 方案二 **训练时缩放**（Inverted Dropout，更常用）：训练时输出放大为 $y_{drop} = \frac{r}{p} \cdot y$，这样期望输出为：

$$\mathbb{E}[y_{drop}] = \mathbb{E}\left[\frac{r}{p} \cdot y\right] = \frac{\mathbb{E}[r]}{p} \cdot y = \frac{p}{p} \cdot y = y$$

推理时直接输出 $y$（无需任何调整），训练期望与推理输出相等。方案二的优势是训练往往是一次性的，推理则要反复进行，推理时不需要额外操作，整体更简洁。PyTorch、TensorFlow 等框架都采用这种 Inverted Dropout 方式。

### Dropout 率选择

Dropout 率 $1-p$（丢弃概率）是关键的超参数，直接影响正则化强度。丢弃率越高，正则化越强，但训练收敛越困难；丢弃率过低则正则化不足。这个参数不应当一概而论，网络中不同类型的层对 Dropout 的敏感度不同，需要针对性设置：

| 层类型 | Dropout 率 $1-p$ | 保留概率 $p$ | 原因 |
|:------|:---------------:|:----------:|:----|
| 全连接层 | 0.5 | 0.5 | 参数多，易过拟合，需要强正则化 |
| 卷积层 | 0.1-0.25 | 0.75-0.9 | 参数相对少，卷积操作自带正则化（空间共享） |
| 输入层 | 0.2 | 0.8 | 丢弃输入特征可能损失信息，不宜过高 |
| 输出层 | 0 | 1.0 | 输出层需要准确预测，丢弃会影响稳定性 |

全连接层参数数量远大于卷积层。譬如，一个 $1000$ → $500$ 的全连接层有 50 万参数，而一个 $3 \times 3$ 卷积层（输入 64 通道，输出 128 通道）仅有约 7 万参数。全连接层的参数冗余度更高，更容易过拟合，因此需要更高的 Dropout 率。

卷积层的卷积核在空间位置上共享参数（同一个卷积核扫过整张图），这种参数共享本身就是一种正则化手段，它限制了模型的自由度。因此卷积层通常只需要较低的 Dropout 率（0.1-0.25），甚至完全不用 Dropout。

输入层丢弃过多会导致信息损失。譬如图像分类中，如果输入层 Dropout 率为 0.5，相当于每张图片随机丢失一半像素，模型很难学到完整特征。输出层则完全不应该用 Dropout，因为输出层负责最终预测，神经元丢弃会导致预测不稳定，影响模型准确性。

## 集成学习解释

接触 Dropout 的随机丢弃机制后，一个更深层次的问题浮现，为什么随机丢弃神经元能提升泛化能力？直觉上，丢弃神经元应该削弱网络的表达能力才对。集成学习（Ensemble Learning）理论尝试给予这个问题的答案，Dropout 实际上在训练时隐式地构建了海量的子网络，推理时相当于对这些子网络的预测进行平均。这种训练一个模型，获得多个模型效果的特性，才是 Dropout 的精髓所在。

想象一个团队项目，原本 10 个人协同工作，但因为每个人可能随时缺席，团队实际上演化出了无数种临时小组配置，今天张三和李四在，明天变成王五和赵六。每次缺席人员不同，团队协作方式也不同。久而久之，每种配置都积累了经验。Dropout 对神经网络做的正是同样的事情，每次训练样本经过网络时随机丢弃一部分神经元，形成不同的"子网络"结构。

设网络有 $n$ 个神经元，每个神经元独立以概率 $p$ 保留（$1-p$ 丢弃）。理论上，所有可能的子网络配置数量为 $2^n$，这是因为每个神经元有"保留"或"丢弃"两种状态，一个包含 100 个神经元的隐藏层，理论上可以产生 $2^{100}$ 种不同的子网络，这是一个天文数字，远超宇宙中所有原子数量。实际训练中，我们肯定不可能遍历所有 $2^n$ 种配置。但 Dropout 的随机性保证了每次训练样本都会经过一个随机采样的子网络，不同样本、不同训练轮次、不同层的 mask 都不同。一个包含 10 万训练样本、训练 100 轮的模型，实际采样了约 1000 万次不同的子网络配置。虽然远小于 $2^n$，但覆盖的多样性足以让每个神经元学会在各种"同伴缺席"的情况下独立工作。

在统计学习部分，讲解[随机森林](../../statistical-learning/decision-tree-ensemble/random-forest.md)时我们接触过集成学习算法，传统集成学习要训练多个独立模型来提升泛化能力。譬如训练 5 个不同的神经网络，推理时对 5 个模型的预测结果取平均。这种方法的成本消耗是显而易见的，训练成本 5 倍、推理成本 5 倍、存储成本 5 倍。相比之下，Dropout 只训练一个模型，却获得类似多个模型的集成效果。这里的关键优势是权重共享，所有子网络使用同一套参数 $\mathbf{W}$。子网络之间的差异仅在于哪些神经元被激活，而非参数本身不同。这带来三个巨大好处：

1. **存储效率**：不需要存储多个模型的参数，一个网络的参数即可。
2. **训练效率**：不需要训练多个独立网络，单次训练即可。
3. **推理效率**：推理时不丢弃，单次前向传播即可获得"集成平均"的效果。

从数学角度看，推理时的输出相当于对所有可能子网络的预测期望。设网络输出为 $f(\mathbf{x}; \mathbf{W})$，Dropout 后输出为 $f_{drop}(\mathbf{x}; \mathbf{W}, \mathbf{r}) = f(\mathbf{x}; \mathbf{W} \odot \mathbf{r})$，其中 $\mathbf{r}$ 是 Dropout Mask，决定每个神经元是否保留，$\odot$ 表示逐元素乘法。推理时的理想输出应该是：

$$f_{test}(\mathbf{x}; \mathbf{W}) = \mathbb{E}_{\mathbf{r}}[f_{drop}(\mathbf{x}; \mathbf{W}, \mathbf{r})] = \frac{1}{2^n}\sum_{\mathbf{r}} f(\mathbf{x}; \mathbf{W} \odot \mathbf{r})$$

这个公式看起来很美，对所有 $2^n$ 种 mask 的预测结果取平均。但现实中 $2^n$ 是天文数字，无法遍历计算。幸运的是，对于线性操作（如矩阵乘法），期望可以直接计算：

$$\mathbb{E}[\mathbf{W} \odot \mathbf{r} \cdot \mathbf{x}] = \mathbb{E}[\mathbf{W} \odot \mathbf{r}] \cdot \mathbf{x} = (p \cdot \mathbf{W}) \cdot \mathbf{x}$$

这正是我们在"训练与推理的差异"中讨论的缩放方案：训练时保留的神经元被放大 $1/p$，推理时直接输出原值。对于非线性网络（包含 ReLU、Sigmoid 等激活函数），这种近似存在误差，但实践证明效果良好，因为深度网络中的大部分操作接近线性（ReLU 在正值区域是线性函数），且随机采样的多样性弥补了近似误差。

## 防止过拟合机制

集成学习提供了 Dropout 为什么有效的理论基础，在实践中 Dropout 的工作机制更加直观。Dropout 通过打破神经元之间的依赖关系、降低有效网络复杂度、注入噪声提升鲁棒性三种途径防止过拟合。三者协同作用，形成有效的正则化效果。

- **破坏神经元共适应**：神经网络训练过程中，神经元之间会自发形成复杂的协作关系。某些神经元只有在其他特定神经元存在时才能发挥作用，它们依赖彼此，形成一种隐性的团队。这种依赖关系被称为神经元**共适应**（Co-adaptation）。

    举个例子：假设网络中有一组神经元 $\{A, B, C\}$，它们分工协作识别"猫"这个概念。神经元 $A$ 检测耳朵形状，神经元 $B$ 检测眼睛位置，神经元 $C$ 综合前两者的信息做出判断。如果训练过程中 $C$ 总是能获得 $A$ 和 $B$ 的可靠输入，$C$ 就会"依赖"这种协作模式，它的权重会专门针对 $A$ 和 $B$ 的输出特征进行优化。问题在于，这种依赖关系是脆弱的：如果 $A$ 或 $B$ 的特征在测试数据中略有变化（比如猫的耳朵形状稍有不同），$C$ 可能完全失效，导致整体预测失败。

    Dropout 通过随机丢弃神经元，强制打破这种依赖关系。训练时 $A$、$B$、$C$ 都可能被随机丢弃，$C$ 不能稳定依赖 $A$ 和 $B$ 的输入。久而久之，每个神经元都学会"独立生存"，即使同伴缺席，也能通过其他途径获取信息。这正是辛顿设计 Dropout 的初衷，模拟银行工作人员的"冗余协作"机制。

- **降低网络有效复杂度**：Dropout 在训练时动态降低网络的"有效容量"。假设一个隐藏层有 $N$ 个参数（权重 + 偏置），训练时 Dropout 率为 0.5，那么每个训练轮次只有约一半神经元参与计算和参数更新，"有效参数数"约为 $p \cdot N$。

这带来两个好处：

- **限制模型容量**：有效参数减少，模型拟合能力受限，无法"记住"训练数据的噪声细节
- **训练多个小网络**：每次样本经过不同的子网络，相当于训练多个参数较少的小模型

这与 L2 正则化的思路类似，但方式不同。L2 通过惩罚参数值来限制复杂度（让参数变小），Dropout 通过减少参与计算的参数数量来限制复杂度（让部分参数暂时"消失"）。两者可以互补使用 —— 实践中常同时采用 Dropout 和权重衰减。

### 注入噪声提升鲁棒性

Dropout 在网络内部注入随机噪声 —— 神经元输出随机置零。这与数据增强的思路一致，但作用位置不同：数据增强在输入端注入噪声（如图片旋转、裁剪），Dropout 在网络内部注入噪声（隐藏层激活值）。

噪声的作用是迫使网络学习"鲁棒"的特征表达。所谓鲁棒，就是即使输入或内部状态有扰动，输出仍然稳定。举个例子：网络学到"猫有三角形耳朵"这个特征，如果神经元经常被丢弃，网络就必须学会即使部分"耳朵检测"神经元失效，仍能通过其他线索（如眼睛、鼻子）识别猫。这种"多线索、冗余备份"的学习方式，正是鲁棒性的来源。

## Dropout 验证实践

理论分析揭示了 Dropout 的工作机制，但最终需要实验验证其效果。下面的代码构建一个模拟回归任务：使用一个小型训练集（100 样本）训练深度网络（64-32-1 结构），对比有无 Dropout 的训练过程。我们记录每轮训练的训练损失和测试损失，绘制损失曲线图，直观展示 Dropout 如何缩小过拟合差距。实验还对比不同 Dropout 率（0.0、0.2、0.5、0.7）的效果，以及不同训练集大小（50、100、200、500）对 Dropout 效果的影响。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

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

# Dropout 实现
def dropout(x, p, training=True):
    """
    Dropout 函数
    x: 神经元输出
    p: 保留概率
    training: 是否训练模式
    """
    if not training or p == 1.0:
        return x
    mask = (np.random.rand(*x.shape) < p).astype(float)
    return x * mask / p

# 多层网络（支持 Dropout）
class NeuralNetwork:
    def __init__(self, layer_sizes, dropout_rates=None, activation='relu'):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # 默认 Dropout 率
        if dropout_rates is None:
            dropout_rates = [0.0] * self.num_layers
        self.dropout_rates = dropout_rates
        
        # 激活函数
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        
        # He 初始化
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X, training=True):
        """前向传播"""
        self.activations = [X]
        self.pre_activations = []
        
        a = X
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            a = self.activation(z)
            
            # 应用 Dropout（除最后一层）
            if i < self.num_layers - 1:
                a = dropout(a, self.dropout_rates[i], training)
            
            self.activations.append(a)
        
        return a
    
    def backward(self, X, y, learning_rate=0.01):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层误差（MSE 损失）
        delta = (self.activations[-1] - y) * self.activation_derivative(self.pre_activations[-1])
        
        # 反向传播
        for i in range(self.num_layers - 1, -1, -1):
            # 计算梯度
            grad_w = self.activations[i].T @ delta / m
            grad_b = np.mean(delta, axis=0, keepdims=True)
            
            # 更新参数
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            
            # 传播误差
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_derivative(self.pre_activations[i-1])
                # Dropout mask 反向传播（梯度乘 mask）
                if self.dropout_rates[i-1] > 0:
                    # 重新生成 mask（训练时）
                    mask = (np.random.rand(*self.activations[i].shape) < self.dropout_rates[i-1]).astype(float)
                    delta = delta * mask / self.dropout_rates[i-1]
    
    def compute_loss(self, X, y, training=False):
        """计算损失"""
        output = self.forward(X, training=training)
        return np.mean((output - y)**2)

print("实验1：过拟合与 Dropout 对比")
print("-" * 40)

# 生成数据（小训练集，大测试集，模拟过拟合场景）
n_train = 100
n_test = 500
n_features = 20

# 训练数据
X_train = np.random.randn(n_train, n_features)
y_train = np.sin(X_train[:, 0] * 2) + np.cos(X_train[:, 1]) + np.random.randn(n_train) * 0.1
y_train = y_train.reshape(-1, 1)

# 测试数据
X_test = np.random.randn(n_test, n_features)
y_test = np.sin(X_test[:, 0] * 2) + np.cos(X_test[:, 1]) + np.random.randn(n_test) * 0.1
y_test = y_test.reshape(-1, 1)

# 网络配置
layer_sizes = [n_features, 64, 32, 1]

# 无 Dropout
net_no_dropout = NeuralNetwork(layer_sizes, dropout_rates=[0.0, 0.0], activation='relu')

# Dropout (p=0.5)
net_dropout = NeuralNetwork(layer_sizes, dropout_rates=[0.5, 0.5], activation='relu')

# 训练参数
n_epochs = 200
learning_rate = 0.01

# 记录训练过程
train_losses_no_drop = []
test_losses_no_drop = []
train_losses_drop = []
test_losses_drop = []

print("开始训练...")
for epoch in range(n_epochs):
    # 无 Dropout 训练
    net_no_dropout.forward(X_train, training=True)
    net_no_dropout.backward(X_train, y_train, learning_rate)
    
    train_loss_no = net_no_dropout.compute_loss(X_train, y_train, training=False)
    test_loss_no = net_no_dropout.compute_loss(X_test, y_test, training=False)
    
    train_losses_no_drop.append(train_loss_no)
    test_losses_no_drop.append(test_loss_no)
    
    # Dropout 训练
    net_dropout.forward(X_train, training=True)
    net_dropout.backward(X_train, y_train, learning_rate)
    
    train_loss_drop = net_dropout.compute_loss(X_train, y_train, training=False)
    test_loss_drop = net_dropout.compute_loss(X_test, y_test, training=False)
    
    train_losses_drop.append(train_loss_drop)
    test_losses_drop.append(test_loss_drop)

print(f"\n无 Dropout:")
print(f"  最终训练损失: {train_losses_no_drop[-1]:.4f}")
print(f"  最终测试损失: {test_losses_no_drop[-1]:.4f}")
print(f"  差异: {test_losses_no_drop[-1] - train_losses_no_drop[-1]:.4f}")

print(f"\nDropout (p=0.5):")
print(f"  最终训练损失: {train_losses_drop[-1]:.4f}")
print(f"  最终测试损失: {test_losses_drop[-1]:.4f}")
print(f"  差异: {test_losses_drop[-1] - train_losses_drop[-1]:.4f}")

# 可视化损失曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 无 Dropout
ax1 = axes[0]
ax1.plot(train_losses_no_drop, label='训练损失', linewidth=2, color='#3498db')
ax1.plot(test_losses_no_drop, label='测试损失', linewidth=2, color='#e74c3c')
ax1.fill_between(range(len(train_losses_no_drop)), train_losses_no_drop, test_losses_no_drop,
                 alpha=0.3, color='#f39c12', label='过拟合差距')
ax1.set_xlabel('训练轮数', fontsize=11)
ax1.set_ylabel('损失值', fontsize=11)
ax1.set_title('无 Dropout - 过拟合明显', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Dropout
ax2 = axes[1]
ax2.plot(train_losses_drop, label='训练损失', linewidth=2, color='#3498db')
ax2.plot(test_losses_drop, label='测试损失', linewidth=2, color='#e74c3c')
ax2.fill_between(range(len(train_losses_drop)), train_losses_drop, test_losses_drop,
                 alpha=0.3, color='#2ecc71', label='差距缩小')
ax2.set_xlabel('训练轮数', fontsize=11)
ax2.set_ylabel('损失值', fontsize=11)
ax2.set_title('Dropout (p=0.5) - 过拟合缓解', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("实验2：不同 Dropout 率的效果对比")
print("-" * 40)

dropout_rates_list = [0.0, 0.2, 0.5, 0.7]
results = {}

for rate in dropout_rates_list:
    dropout_config = [rate, rate] if rate > 0 else [0.0, 0.0]
    net = NeuralNetwork(layer_sizes, dropout_rates=dropout_config, activation='relu')
    
    train_losses = []
    test_losses = []
    
    for epoch in range(n_epochs):
        net.forward(X_train, training=True)
        net.backward(X_train, y_train, learning_rate)
        
        train_loss = net.compute_loss(X_train, y_train, training=False)
        test_loss = net.compute_loss(X_test, y_test, training=False)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    results[rate] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_gap': test_losses[-1] - train_losses[-1]
    }
    
    print(f"Dropout 率 {rate:.1f}:")
    print(f"  训练损失: {train_losses[-1]:.4f}")
    print(f"  测试损失: {test_losses[-1]:.4f}")
    print(f"  过拟合差距: {results[rate]['final_gap']:.4f}")
    print()

# 可视化不同 Dropout 率
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for idx, rate in enumerate(dropout_rates_list):
    ax = axes[idx // 2, idx % 2]
    ax.plot(results[rate]['train_losses'], label='训练损失', 
            linewidth=2, color=colors[idx])
    ax.plot(results[rate]['test_losses'], label='测试损失', 
            linewidth=2, color=colors[idx], linestyle='--')
    
    gap = results[rate]['final_gap']
    ax.set_xlabel('训练轮数', fontsize=11)
    ax.set_ylabel('损失值', fontsize=11)
    ax.set_title(f'Dropout 率 = {rate:.1f}\n过拟合差距 = {gap:.4f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("实验3：训练集大小对过拟合和 Dropout 效果的影响")
print("-" * 40)

train_sizes = [50, 100, 200, 500]
n_epochs = 150

size_results = {}

for n_train in train_sizes:
    # 生成数据
    X_train_small = np.random.randn(n_train, n_features)
    y_train_small = np.sin(X_train_small[:, 0] * 2) + np.cos(X_train_small[:, 1]) + \
                    np.random.randn(n_train) * 0.1
    y_train_small = y_train_small.reshape(-1, 1)
    
    # 无 Dropout
    net_no = NeuralNetwork(layer_sizes, dropout_rates=[0.0, 0.0], activation='relu')
    
    # Dropout
    net_drop = NeuralNetwork(layer_sizes, dropout_rates=[0.5, 0.5], activation='relu')
    
    no_drop_gaps = []
    drop_gaps = []
    
    for epoch in range(n_epochs):
        # 无 Dropout
        net_no.forward(X_train_small, training=True)
        net_no.backward(X_train_small, y_train_small, learning_rate)
        
        train_loss_no = net_no.compute_loss(X_train_small, y_train_small, training=False)
        test_loss_no = net_no.compute_loss(X_test, y_test, training=False)
        no_drop_gaps.append(test_loss_no - train_loss_no)
        
        # Dropout
        net_drop.forward(X_train_small, training=True)
        net_drop.backward(X_train_small, y_train_small, learning_rate)
        
        train_loss_drop = net_drop.compute_loss(X_train_small, y_train_small, training=False)
        test_loss_drop = net_drop.compute_loss(X_test, y_test, training=False)
        drop_gaps.append(test_loss_drop - train_loss_drop)
    
    size_results[n_train] = {
        'no_drop_gap': no_drop_gaps[-1],
        'drop_gap': drop_gaps[-1],
        'improvement': no_drop_gaps[-1] - drop_gaps[-1]
    }
    
    print(f"训练集大小 {n_train}:")
    print(f"  无 Dropout 过拟合差距: {no_drop_gaps[-1]:.4f}")
    print(f"  Dropout 过拟合差距: {drop_gaps[-1]:.4f}")
    print(f"  Dropout 改善: {size_results[n_train]['improvement']:.4f}")
    print()

# 可视化训练集大小影响
fig, ax = plt.subplots(figsize=(10, 6))

sizes = list(size_results.keys())
no_drop_gaps = [size_results[s]['no_drop_gap'] for s in sizes]
drop_gaps = [size_results[s]['drop_gap'] for s in sizes]
improvements = [size_results[s]['improvement'] for s in sizes]

x = range(len(sizes))
ax.bar(x, no_drop_gaps, width=0.4, label='无 Dropout', color='#e74c3c', alpha=0.7)
ax.bar([i + 0.4 for i in x], drop_gaps, width=0.4, label='Dropout', color='#2ecc71', alpha=0.7)

ax.set_xticks([i + 0.2 for i in x])
ax.set_xticklabels(sizes)
ax.set_xlabel('训练集大小', fontsize=11)
ax.set_ylabel('过拟合差距（测试-训练损失）', fontsize=11)
ax.set_title('训练集大小对过拟合的影响', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 添加改善标注
for i, imp in enumerate(improvements):
    ax.annotate(f'改善 {imp:.2f}', 
                xy=(i + 0.2, max(no_drop_gaps[i], drop_gaps[i]) + 0.02),
                ha='center', fontsize=10, color='#3498db')

plt.tight_layout()
plt.show()
plt.close()
```

## 实践经验

Dropout 的效果高度依赖于是否能正确使用，在错误的位置使用、错误的超参数配置、与[批归一化](./batch-normalization.md)不当组合，都可能削弱甚至得到反效果。本节总结了一些实践经验，帮助读者在真实项目中有效应用 Dropout。

- **Dropout 层位置**：Dropout 层在网络中的位置直接影响其效果。位置选择的原则是在参数密集、易过拟合的层后使用 Dropout，避免在信息关键、影响稳定性的层使用 Dropout。举几个例子：

    | 位置 | 建议 | 原因 |
    |:--------|:---------|:-----|
    | 全连接层后 | 推荐 | 参数多，易过拟合，需要强正则化 |
    | 卷积层后 | 可选 | 参数相对少，卷积自带正则化 |
    | LSTM 层后 | 谨慎 | 时序信息可能被破坏 |
    | 批归一化层前 | 避免 | BN 统计不稳定 |
    | 输入层 | 避免 | 丢弃输入特征可能损失信息 |
    | 输出层 | 避免 | 影响预测稳定性 |

- **Dropout 与权重衰减**：Dropout 与 L2 权重衰减可以同时使用，一个约束模型有效结构复杂度，一个约束模型参数大小，两者互补。在经验上，Dropout 对权重衰减的超参数效果有所影响，使用 Dropout 时，可适当降低权重衰减系数 $\lambda$，Dropout 率 $1-p$ 越高，权重衰减系数 $\lambda$ 应该越低。

- **训练技巧**：实际训练过程中，可考虑如下经验技巧

    1. **Dropout 率调优**：从 $p=0.5$ 开始，观察训练和测试差距调整
    2. **训练时间延长**：Dropout 降低有效网络复杂度，训练收敛可能需要更多轮数
    3. **学习率调整**：Dropout 增加噪声，可能需要稍高学习率
    4. **早停监控**：使用 Dropout 时，早停策略防止欠拟合

## 本章小结

Dropout 是深度学习中最简单却最有效的正则化技术之一。它的思想朴实，训练时随机关闭部分神经元，迫使网络学会独立生存。这种看似粗暴的操作，背后有三种协同作用的机制：打破神经元共适应（让每个神经元不再依赖特定同伴）、降低网络有效复杂度（每次训练只更新部分参数）、注入噪声提升鲁棒性（训练时扰动内部表示）。集成学习理论进一步揭示了 Dropout 相当于训练海量共享权重的子网络，推理时隐式平均所有子网络的预测，以单模型的成本获得多模型的集成效果。

本章遗留的问题是 Dropout 只解决了过拟合问题，但深度网络训练还有另一个挑战 —— 内部协变量偏移（Internal Covariate Shift），深层网络的每层输入分布会随前层参数更新而变化，导致梯度传播不稳定、收敛缓慢。下一章介绍的[批归一化](batch-normalization.md)（Batch Normalization）正是为解决这一问题而设计的，它标准化每层输入的分布，使训练更稳定、更快速，同时附带正则化效果。

## 练习题

1. 设某神经元输出为 $y = 2.5$，Dropout 保留概率 $p = 0.6$。计算：(a) 采用训练时缩放方案时，该神经元在训练时的期望输出；(b) 采用推理时缩放方案时，该神经元在推理时的输出值。
    <details>
    <summary>参考答案</summary>

    (a) **训练时缩放方案**：训练时输出为 $y_{drop} = \frac{r}{p} \cdot y$，其中 $r \sim \text{Bernoulli}(p)$。

    期望输出为：
    $$\mathbb{E}[y_{drop}] = \mathbb{E}\left[\frac{r}{p} \cdot y\right] = \frac{\mathbb{E}[r]}{p} \cdot y = \frac{p}{p} \cdot y = y = 2.5$$

    这意味着训练时的期望输出与原始输出相等。

    (b) **推理时缩放方案**：训练时输出为 $y_{drop} = r \cdot y$，推理时需要缩放。

    训练期望：
    $$\mathbb{E}[y_{drop}] = \mathbb{E}[r \cdot y] = \mathbb{E}[r] \cdot y = p \cdot y = 0.6 \times 2.5 = 1.5$$

    推理时输出（需缩小以匹配训练期望）：
    $$y_{test} = p \cdot y = 0.6 \times 2.5 = 1.5$$

    **总结**：两种方案的最终输出期望一致（都是 1.5），区别在于缩放时机。训练时缩放方案在训练时放大（除以 $p$），推理时无需操作；推理时缩放方案在训练时不放大，推理时缩小（乘以 $p$）。实践中训练时缩放更常用，因为推理次数远多于训练次数，推理时无额外计算更高效。
    </details>

1. 解释 Dropout 在训练阶段和推理阶段的行为差异，以及为什么需要这种差异。从期望输出一致性的角度说明两种缩放方案的优劣。
    <details>
    <summary>参考答案</summary>

    **训练与推理的行为差异**：

    | 阶段 | Dropout 行为 | 原因 |
    |:----|:------------|:----|
    | 训练 | 随机丢弃神经元 | 打破共适应，增强鲁棒性 |
    | 探理 | 所有神经元保留 | 需要稳定、确定的预测 |

    训练时丢弃是必要的正则化手段，但推理时不能继续丢弃——否则每次预测结果都不同，且预测质量不稳定（部分神经元被丢弃可能导致错误预测）。

    **期望输出一致性原理**：

    设神经元原始输出为 $y$，保留概率为 $p$。训练时随机丢弃后，期望输出为 $\mathbb{E}[y_{drop}] = p \cdot y$（仅为原始输出的 $p$ 倍）。推理时所有神经元保留，输出为 $y$。两者不一致会导致预测偏差。

    **两种缩放方案对比**：

    | 方案 | 训练时操作 | 推理时操作 | 优势 | 劣势 |
    |:----|:---------|:---------|:----|:----|
    | 推理时缩放 | $y_{drop} = r \cdot y$ | $y_{test} = p \cdot y$ | 训练计算简单 | 推理每次都要缩放 |
    | 训练时缩放 | $y_{drop} = \frac{r}{p} \cdot y$ | $y_{test} = y$ | 推理无额外操作 | 训练时需放大 |

    **优劣分析**：

    训练往往是一次性的，推理则要反复进行（生产环境中模型可能被调用数百万次）。训练时缩放方案将计算负担放在训练阶段（一次性），推理阶段无需任何调整（每次都省略缩放计算），整体更高效。这也是 PyTorch、TensorFlow 等框架采用训练时缩放的原因。
    </details>

1. 推导训练时缩放（Inverted Dropout）方案下，对于一个包含 $n$ 个神经元的隐藏层，其输出的期望值等于原始输出的条件。
    <details>
    <summary>参考答案</summary>

    设隐藏层有 $n$ 个神经元，原始输出向量为 $\mathbf{h} = [h_1, h_2, \ldots, h_n]$。每个神经元有独立的 Dropout mask $r_i \sim \text{Bernoulli}(p)$。

    **Inverted Dropout 公式**：
    $$h_i^{drop} = \frac{r_i}{p} \cdot h_i$$

    **期望推导**：
    $$\mathbb{E}[h_i^{drop}] = \mathbb{E}\left[\frac{r_i}{p} \cdot h_i\right] = \frac{\mathbb{E}[r_i]}{p} \cdot h_i$$

    由于 $r_i$ 服从伯努利分布，其期望为：
    $$\mathbb{E}[r_i] = p$$

    代入得：
    $$\mathbb{E}[h_i^{drop}] = \frac{p}{p} \cdot h_i = h_i$$

    对整个隐藏层向量：
    $$\mathbb{E}[\mathbf{h}^{drop}] = \mathbb{E}\left[\frac{\mathbf{r}}{p} \odot \mathbf{h}\right] = \frac{\mathbb{E}[\mathbf{r}]}{p} \odot \mathbf{h} = \frac{p}{p} \odot \mathbf{h} = \mathbf{h}$$

    **结论**：Inverted Dropout 通过训练时放大 $1/p$ 倍，使得期望输出 $\mathbb{E}[\mathbf{h}^{drop}] = \mathbf{h}$，与原始输出相等。因此推理时直接输出 $\mathbf{h}$ 即可，无需缩放调整。

    **关键假设**：上述推导假设 $r_i$ 与 $h_i$ 独立（随机 mask 不依赖神经元输出值），这在标准 Dropout 中成立。
    </details>
