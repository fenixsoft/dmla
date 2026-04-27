# 权重初始化

**权重初始化**（Weight Initialization）是神经网络训练的起点，一个好的初始化能让训练快速、稳定地收敛；一个糟糕的初始化可能导致训练停滞、震荡甚至崩溃。本章将深入分析初始化的重要性，介绍 Xavier 初始化和 He 初始化两种经典方法，并通过实验验证初始化对训练的实际影响。

## 对称性破坏问题

在介绍权重初始化方法前，我们先考虑如果不进行权重初始化，让神经元全部从相同的参数值（譬如零值）开始，训练会有什么结果？考虑一个简单的两层全连接网络，输入 $x$ 经过隐藏层 $h$ 到达输出 $y$（即 $h = \mathbf{W}_1 x$，$y = \mathbf{W}_2 h$），其中 $\mathbf{W}_1$ 是输入层到隐藏层的权重矩阵，$\mathbf{W}_2$ 是隐藏层到输出层的权重矩阵。如果我们把这两个矩阵都初始化为零（即 $\mathbf{W}_1 = 0, \quad \mathbf{W}_2 = 0$），那么在[前向传播](../neural-network-structure/backpropagation.md#前向传播的计算过程)阶段，就会发现隐藏层所有神经元的输出都是 $h_i = 0$（因为权重为零），输出层所有神经元的输出也都是 $y_i = 0$（因为 $h = 0$ 且权重为零）。这说明网络的表达能力完全退化，原本希望 100 个隐藏神经元来学习不同的特征，现在这 100 个神经元变成了 100 个复制品。

网络也无法通过训练来摆脱困境，因为在[反向传播](../neural-network-structure/backpropagation.md#反向传播的梯度计算)阶段问题同样严重。隐藏层权重的梯度为 $\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial y} \cdot \mathbf{W}_2$，当 $\mathbf{W}_2 = 0$ 时，梯度直接为零，意味着参数无法更新，网络永远停留在初始状态。哪怕只初始化 $\mathbf{W}_1$ 为零，$\mathbf{W}_2$ 不为零，由于 $h = 0$，即使 $\mathbf{W}_2$ 的每个分量有不同值，也会使得误差传播到所有隐藏神经元时完全一致，隐藏层所有神经元仍然接收相同的梯度，更新后的权重也完全相同。这种所有神经元被迫学习相同的功能，网络的多层、多神经元结构变得毫无意义的现象被称为**对称性破坏问题**（Symmetry Breaking Problem）。全零初始化剥夺了神经元之间的差异性，使精心设计的网络结构退化成单个神经元。

## 权重初始化

既然全零初始化有对称性破坏问题，那一个很符合直觉的方案自然浮出水面，给每个神经元的权重赋予不同的随机初始值。打破神经元之间的对称性。我们使用均匀分布和[正态分布](../../maths/probability/probability-basics.md#正态分布)两种最常见随机分布，尝试进行随机初始化。

- **均匀分布初始化**：从区间 $[-a, a]$ 中随机采样权重：$\mathbf{W}_{ij} \sim U[-a, a]$，每个权重独立采样，互不影响。均匀分布的方差为 $\text{Var}(\mathbf{W}_{ij}) = \frac{a^2}{3}$
- **正态分布初始化**：从零均值正态分布中采样权重：$\mathbf{W}_{ij} \sim N(0, \sigma^2)$，大部分权重集中在 $[-\sigma, \sigma]$ 范围内，少数权重可能较大或较小，均值为 0，方差为 $\sigma^2$

随机初始化成功打破了神经元之间的对称性，使每个神经元能够学习不同的特征。但这里要回答一个问题，分布参数（$a$ 或 $\sigma$）应该如何设置？这个问题关乎训练成败，以正态分布为例，考虑两个极端场景：

- **参数过大**（比如 $\sigma = 10$）：前向传播时，权重要乘以输入产生巨大的激活值。Sigmoid、tanh 这些激活函数收到 100 这样的输入，输出就很接近 1 了，激活值一直处于在饱和边界上，反向传播时梯度接近零，没有传递什么参数更新信息。
- **参数过小**（比如 $\sigma = 0.001$）：前向传播时，激活值逐层衰减，信号像" whispers"一样越传越弱。反向传播时，梯度也逐层缩小，到前面几层时，也没有了参数更新信息，训练几乎停滞。

以上这些现象就是经典的[梯度消失](../neural-network-structure/activation-loss-functions.md#梯度消失与梯度爆炸)问题，由此可见，好的初始化需要在"打破对称性，让每个神经元有独特的个性"和"保持信号强度，避免梯度消失，让误差信号能稳定传回输入层"两个目标之间找到平衡。这两个目标看似矛盾，随机值太小会导致信号衰减，太大会导致信号饱和。如何找到恰到好处的随机分布参数，这正是 Xavier 初始化和 He 初始化要解决的问题。

### Xavier 初始化

通过分析信号在网络中的传播过程，推导出保持信号稳定所需的权重方差，精确计算出最佳的初始化参数。这种方法最早由加拿大计算机科学家泽维尔·格洛罗（Xavier Glorot）和他的导师约书亚·本吉奥（Yoshua Bengio，2018 年图灵奖得主）在 2010 年提出。他们在题为《Understanding the difficulty of training deep feedforward neural networks》的论文中，系统分析了深度网络训练困难的原因，并提出了著名的 Xavier 初始化方法。这篇论文揭示了初始化与训练稳定性的深层联系，成为深度学习优化领域的里程碑工作。

让我们从最简单的情况开始分析什么样的权重方差能让信号穿过一层又一层的网络，如果有一个线性神经元（暂时忽略激活函数，或者假设激活函数是线性的），假设这个神经元接收 $n_{in}$ 个输入，产生一个输出 $y = \sum_{i=1}^{n_{in}} w_i x_i$，为了分析输出的方差，我们需要做一些合理的假设。设权重 $w_i$ 和输入 $x_i$ 满足以下条件：

- 独立同分布（每个权重、每个输入都是独立采样的）
- 零均值（$E[w_i] = 0$, $E[x_i] = 0$）
- 方差固定（$\text{Var}(w_i) = \text{Var}(w)$，$\text{Var}(x_i) = \text{Var}(x)$）

在这些假设下，输出 $y$ 的方差可以推导出来，首先根据方差定义 $\text{Var}(y) = \text{Var}\left(\sum_{i=1}^{n_{in}} w_i x_i\right)$，由于 $w_i$ 和 $x_i$ 独立，根据[独立变量求和方差](../../maths/probability/probability-basics.md)的性质（$\text{Var}[X + Y] = \text{Var}[X] + \text{Var}[Y]$）可以得到：

$$[eq:var-mul]\text{Var}(y) = \sum_{i=1}^{n_{in}} \text{Var}(w_i x_i)$$

又根据独立变量乘积方差的性质（$\text{Var}(XY) = \text{Var}(X) \cdot \text{Var}(Y) + \text{Var}(X) \cdot E[Y]^2 + \text{Var}(Y) \cdot E[X]^2$），结合零均值假设（$E[w_i] = 0$ 且 $E[x_i] = 0$），可得到：

$$[eq:var-sum]\text{Var}(w_i x_i) = \text{Var}(w) \cdot \text{Var}(x)$$

结合 {{eq:var-mul}} 与 {{eq:var-sum}} 得到：

$$ \text{Var}(y) = \sum_{i=1}^{n_{in}} \text{Var}(w_i x_i) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

这个公式揭示了信号前向传播的关键规律，输出方差等于三个因素的乘积：$n_{in}$ 是输入数量，输入越多，方差越大；$\text{Var}(w)$ 是权重方差，权重越分散，方差越大；$\text{Var}(x)$ 是输入方差，输入越分散，方差越大。要保证输入方差的信号强度，前两项应该相乘后抵消，即权重方差应该是 $\frac{1}{n_{in}}$ 才能保持信号强度。

到这里分析只进行了一半，只考虑了前向传播从输入到输出的方向，训练神经网络还需要反向传播，输出层的梯度 $\delta_y$ 通过权重矩阵反向传回输入层，形成输入层梯度，[传递公式](../neural-network-structure/backpropagation.md#隐藏层梯度传递)是 $ \delta^k = (\mathbf{W}^{k+1})^T \delta^{k+1} \cdot \sigma'(\mathbf{z}^k)$，由于暂时不考虑激活函数，激活函数的梯度为 $1$，由此得到：

$$\delta_x = \mathbf{W}^T \delta_y$$

将矩阵形式改写一下，输入层第 $i$ 个神经元的梯度是输出层所有神经元梯度的加权和：

$$\delta_{x_i} = \sum_{j=1}^{n_{out}} w_{ji} \delta_{y_j}$$

其中 $n_{out}$ 是输出神经元数量（称为 Fan-Out），$w_{ji}$ 是从输入 $i$ 到输出 $j$ 的权重。这个结构与公式 {{eq:var-mul}} 完全一样，我们同样假设各权重独立同分布、零均值和方差固定，在这些条件下，输入层梯度 $\delta_{x_i}$ 的方差可以与前面过程一样推导出来，得到：

$$\text{Var}(\delta_x) = n_{out} \cdot \text{Var}(w) \cdot \text{Var}(\delta_y)$$

为了保持梯度强度不变（$\text{Var}(\delta_x) = \text{Var}(\delta_y)$），需要 $n_{out} \cdot \text{Var}(w) = 1$，因此，理想的权重方差应该是 $\text{Var}(w) = \frac{1}{n_{out}}$。推导进行到这里出现了一个矛盾，前向传播要求 $\text{Var}(w) = \frac{1}{n_{in}}$，反向传播要求 $\text{Var}(w) = \frac{1}{n_{out}}$。实际网络中一般 $n_{in} \neq n_{out}$ ，现实局限了无法同时满足两个条件。

为此，泽维尔·格洛罗给出了一个巧妙的折中方案。前向传播要求 $\text{Var}(w) = \frac{1}{n_{in}}$，反向传播要求 $\text{Var}(w) = \frac{1}{n_{out}}$。这两个条件就像拔河，一个向左拉，一个向右拉。当 $n_{in} \neq n_{out}$ 时，我们只能选择一个折中位置。Xavier 初始化采用的是 $n_{in}$ 和 $n_{out}$ 的调和平均：

$$[eq:xavier-var] \text{Var}(w) = \frac{2}{n_{in} + n_{out}}$$

为什么选择调和平均而非算术平均？格洛罗给出的理由是调和平均对较小的数值更敏感，当 $n_{in}$ 和 $n_{out}$ 差异较大时，调和平均会偏向较小的值，避免方差过大导致梯度爆炸。基于这个方差公式，Xavier 初始化有两种具体实现方式。

- **Xavier 均匀初始化**：从均匀分布中采样，均匀分布 $U[-a, a]$ 的方差为 $\frac{a^2}{3}$。为了使方差等于 $\frac{2}{n_{in} + n_{out}}$，需要 $\frac{a^2}{3} = \frac{2}{n_{in} + n_{out}}$，解得 $a = \sqrt{\frac{6}{n_{in} + n_{out}}}$。

    $$\mathbf{W}_{ij} \sim U\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$$

- **Xavier 正态初始化**：从正态分布中采样，正态分布 $N(0, \sigma^2)$ 的方差就是 $\sigma^2$，因此标准差直接取 $\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$。

    $$\mathbf{W}_{ij} \sim N\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

Xavier 初始化的推导有一个关键假设 —— 激活函数是线性的。这个假设在实际中肯定不成立（激活函数就是为了非线性表达），但是如果考虑 Xavier 初始化控制了权重方差，绝大部分激活值不会落到整个函数定义域，而是落到零值附近呢？我们回顾一下 [Sigmoid 激活函数](../../statistical-learning/linear-models/logistic-regression.md#sigmoid-函数) 的形状，当输入 $x$ 在 0 附近时，Sigmoid 函数的行为可以用泰勒展开近似：

$$\sigma(x) \approx \sigma(0) + \sigma'(0) \cdot x = 0.5 + 0.25 \cdot x$$

这是一个线性函数，当激活值落在 0 附近时 Sigmoid 近似于线性变换，Xavier 的线性假设在这个区间中是基本成立的。类似的，[tanh 激活函数](../neural-network-structure/activation-loss-functions.md#双曲正切函数) 在 0 附近也近似线性：

$$\tanh(x) \approx \tanh(0) + \tanh'(0) \cdot x = 0 + 1 \cdot x = x$$

tanh 在 0 附近的线性近似比 sigmoid 更精确（斜率为 1，而 Sigmoid 斜率为 0.25），Xavier 初始化正是利用了这个特性，通过控制权重方差，使激活值落在 0 附近（线性区域），从而保持信号强度稳定传播。

当然，基于假设的约束，Xavier 初始化是存在明显的局限性的，一方面当输入远离 0 时，激活函数非线性特征显现，Xavier 初始化只能尽量让激活值落在线性区域，但无法完全避免饱和。另一方面更为严重，Xavier 初始化不适配 [ReLU 激活函数](../neural-network-structure/activation-loss-functions.md#relu-函数) ，ReLU 的特性与 Sigmoid 和 tanh 完全不同。ReLU 只保留正值，负值全部置零，这意味着约一半的激活值被"杀死"，信号强度减半。Xavier 的线性假设在 ReLU 上完全失效，如果用 Xavier 初始化 ReLU 网络，激活值会逐层衰减，深层网络的梯度几乎为零。针对 ReLU 的特殊性，需要一种新的初始化方法，这正是 He 初始化要解决的问题。

### He 初始化

针对 Xavier 初始化不适配 ReLU 的问题，中国计算机科学家何恺明（Kaiming He）在 2015 年提出了专门为 ReLU 设计的初始化方法。他在论文《Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification》中系统分析了 ReLU 网络的信号传播特性，提出了 He 初始化。这篇论文不仅在初始化理论上做出贡献，更重要的是，它首次展示了深度网络在图像分类任务上超越人类表现的可能性，何恺明团队使用深度残差网络（ResNet）在 ImageNet 竞赛中达到了 3.57% 的 Top-5 错误率，低于人类的 5.1%。这一里程碑成果证明了深度学习的巨大潜力，也确立了 He 初始化在 ReLU 网络中的核心地位。

Xavier 初始化的问题来源于 ReLU 每层都会杀死一半信号，我们需要用更大的权重方差来补偿。同前面 Xavier 初始化的推导类似，我们考虑 ReLU 激活后的信号传播，推导出保持信号稳定所需的权重方差。设输入 $x_i$ 是上一层 ReLU 的输出，权重 $w_i$ 独立同分布、零均值，前向传播分为加权求和（$z = \sum_{i=1}^{n_{in}} w_i x_i$）和ReLU 激活（$y = \max(0, z)$）两步，第一步的方差推理过程不变（见 {{eq:var-sum}}），仍为：

$$\text{Var}(z) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

经过 ReLU 激活 $y = \max(0, z)$ 后，对于零均值的 $z$，ReLU 输出的方差为原方差的一半：

$$\text{Var}(y) = \frac{1}{2} \text{Var}(z) = \frac{1}{2} \cdot n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

为了保持信号强度不变（$\text{Var}(y) = \text{Var}(x)$），需要前两项乘积为 1（$\frac{1}{2} n_{in} \cdot \text{Var}(w) = 1$），得到：

$$\text{Var}(w) = \frac{2}{n_{in}}$$

分子中的 2 是补偿因子，抵消 ReLU 的信号衰减（$\frac{1}{2}$ 相乘需要 $2$ 来补偿），He 初始化的权重方差需要比 Xavier 更大，才能抵消 ReLU 的"杀死一半信号"的威力。基于这个方差公式，He 初始化也有两种实现方式：

- **He 均匀初始化**：从均匀分布中采样，均匀分布的边界 $a = \sqrt{\frac{6}{n_{in}}}$，使方差满足 $\frac{a^2}{3} = \frac{2}{n_{in}}$。

$$\mathbf{W}_{ij} \sim U\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right]$$

- **He 正态初始化**：从正态分布中采样，标准差 $\sigma = \sqrt{\frac{2}{n_{in}}}$，大部分权重落在 $[-\sigma, \sigma]$ 范围内。

$$\mathbf{W}_{ij} \sim N\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

He 初始化的方差公式只考虑 $n_{in}$，而 Xavier 考虑 $n_{in} + n_{out}$ 的调和平均，这是因为 He 初始化更强调前向传播的信号补偿，ReLU 的稀疏性问题主要发生在前向传播阶段，反向传播时梯度只通过正值神经元传递，不需要额外的补偿。当 $n_{in} \approx n_{out}$ 时，He 初始化方差约为 Xavier 的两倍。

## 初始化方法实验

理论分析告诉我们，Xavier 初始化适合 Sigmoid/tanh，He 初始化适合 ReLU。本节让我们通过代码实验来验证该结论。

下面的实验模拟一个多层神经网络（类似 MLP 结构），对比五种初始化方法（全零初始化、小方差初始化、大方差初始化、Xavier 初始化、He 初始化）在三种激活函数（Sigmoid、ReLU、tanh）下的表现。实验测量三个关键指标：每层激活值的分布（验证信号传播）、每层梯度的范数（验证梯度传播）、训练损失曲线（验证收敛速度）。

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
```

## 偏置初始化

行文至此，我们讨论的都是权重初始化，但神经元的参数包括权重和偏置，偏置应该如何初始化？这个问题的答案很简单，偏置通常初始化为零。偏置可以安全地设为零，这是因为：

1. **偏置不参与信号强度的传递**：权重矩阵乘以输入，决定了信号如何流动；偏置只是添加一个常数偏移，不改变信号的相对强度。因此，零偏置不会导致对称性问题。
2. **零偏置保持激活函数在线性区域**：Sigmoid 和 tanh 在输入接近 0 时近似线性，这正是 Xavier 初始化生效的前提。零偏置使输入初始值接近权重乘以输入的结果（零均值），帮助激活值落在线性区域。
3. **非零偏置可能导致激活值偏离线性区域**：如果偏置初始化为较大正值，Sigmoid 激活值可能直接饱和到 1，梯度接近零；如果偏置初始化为较大负值，激活值可能直接饱和到 0，同样梯度接近零。

但也有一些特殊情况，需要非零偏置初始化，譬如：

- **ReLU 网络的正偏置**：ReLU 激活函数将负值置零。如果权重初始化较小（如 He 初始化），大部分激活值可能恰好落在零附近，一半为正一半为负。负值部分被置零后，网络初始输出可能过于稀疏。将偏置初始化为小正值（如 0.01），可以确保初始时大部分神经元有非零输出，有助于避免神经元死亡问题。

- **LSTM 遗忘门偏置**：LSTM（长短期记忆网络，后续章节会介绍）有一个遗忘门，控制前一时刻信息的保留程度。遗忘门的输出接近 0 时，大部分历史信息被遗忘；输出接近 1 时，大部分信息被保留。如果遗忘门偏置初始化为零，初始时遗忘门输出可能接近 0.5（Sigmoid 的中点），导致历史信息部分遗忘。将遗忘门偏置初始化为 1 或更大，使遗忘门初始输出接近 1，保留更多历史信息，有助于训练初期学习长期依赖。

## 本章小结

本章从"为什么初始化至关重要"出发，系统分析了权重初始化对深度网络训练的影响，并介绍了 Xavier 初始化和 He 初始化两种经典方法。初始化的挑战在于找到平衡点，全零初始化破坏对称性，使所有神经元学习相同功能，网络退化到单个神经元。随机初始化打破对称性，但方差选择不当会导致两个极端，方差太小，信号逐层衰减，训练缓慢；方差太大，激活值饱和，梯度消失。好的初始化需要在"打破对称性"和"保持信号强度"之间找到平衡。

初始化是训练的起点，决定了网络能否顺利起步。但训练的终点是能否收敛到最优，这里遇到的最大困难是过拟合，我们曾在线性模型部分讲解过通过 L1、L2 范数正则化来处理过拟合问题。下一章将还将介绍 Dropout 正则化，通过随机丢弃神经元，迫使网络学习更鲁棒的特征，从另一个角度提升训练稳定性。

## 练习题

1. 给定一个全连接层，输入维度 $n_{in} = 512$，输出维度 $n_{out} = 256$。分别计算 Xavier 均匀初始化和 Xavier 正态初始化的参数范围。
    <details>
    <summary>参考答案</summary>

    **Xavier 均匀初始化**：

    根据公式 $a = \sqrt{\frac{6}{n_{in} + n_{out}}}$，代入数值：
    $$a = \sqrt{\frac{6}{512 + 256}} = \sqrt{\frac{6}{768}} = \sqrt{\frac{1}{128}} \approx 0.088$$

    因此权重从 $U[-0.088, 0.088]$ 中采样。

    **Xavier 正态初始化**：

    根据公式 $\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$，代入数值：
    $$\sigma = \sqrt{\frac{2}{512 + 256}} = \sqrt{\frac{2}{768}} = \sqrt{\frac{1}{384}} \approx 0.051$$

    因此权重从 $N(0, 0.051)$ 中采样，大部分权重落在 $[-0.051, 0.051]$ 范围内。

    **对比分析**：均匀初始化的范围约为正态初始化标准差的 1.73 倍（$0.088 / 0.051 \approx 1.73$），这与均匀分布方差公式 $\frac{a^2}{3}$ 中因子 3 与正态分布方差 $\sigma^2$ 的关系一致。
    </details>

1. 解释为什么 Xavier 初始化采用 $n_{in}$ 和 $n_{out}$ 的调和平均 $\frac{2}{n_{in} + n_{out}}$，而非算术平均 $\frac{1}{2}\left(\frac{1}{n_{in}} + \frac{1}{n_{out}}\right)$。从梯度稳定性角度说明调和平均的优势。
    <details>
    <summary>参考答案</summary>

    前向传播要求 $\text{Var}(w) = \frac{1}{n_{in}}$，反向传播要求 $\text{Var}(w) = \frac{1}{n_{out}}$。当 $n_{in} \neq n_{out}$ 时，需要折中。

    **调和平均**：$\frac{2}{n_{in} + n_{out}}$

    **算术平均**：$\frac{1}{2}\left(\frac{1}{n_{in}} + \frac{1}{n_{out}}\right) = \frac{n_{in} + n_{out}}{2 n_{in} n_{out}}$

    两者的区别：调和平均对较小的数值更敏感。设 $n_{in} = 100$，$n_{out} = 1000$：

    - 调和平均：$\frac{2}{100 + 1000} = \frac{2}{1100} \approx 0.0018$
    - 算术平均：$\frac{100 + 1000}{2 \times 100 \times 1000} = \frac{1100}{200000} = 0.0055$

    调和平均结果约为算术平均的 $\frac{1}{3}$，偏向较小的 $\frac{1}{n_{out}} = 0.001$（反向传播需求），而非较大的 $\frac{1}{n_{in}} = 0.01$（前向传播需求）。

    **梯度稳定性角度**：反向传播的梯度范数 $\text{Var}(\delta_x) = n_{out} \cdot \text{Var}(w) \cdot \text{Var}(\delta_y)$ 中，$n_{out}$ 较大时，如果 $\text{Var}(w)$ 也较大，梯度范数会指数级增长，导致梯度爆炸。调和平均偏向较小的方差，能有效抑制反向传播中的梯度爆炸风险，优先保障梯度稳定性。
    </details>

1. 一个三层全连接网络，各层维度为 $[784, 512, 128, 10]$，使用 ReLU 激活函数。如果错误地使用 Xavier 初始化而非 He 初始化，分析信号在前向传播过程中的变化，并计算经过三层后信号方差的变化倍数。
    <details>
    <summary>参考答案</summary>

    **Xavier 初始化用于 ReLU 的问题**：

    Xavier 初始化假设激活函数近似线性，推导出的方差公式 $\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$ 没有考虑 ReLU 的信号衰减。但 ReLU 会将约一半的激活值置零，导致信号方差减半。

    **信号方差变化分析**：

    设第一层权重使用 Xavier 初始化，$n_{in} = 784$，$n_{out} = 512$：
    $$\text{Var}(w_1) = \frac{2}{784 + 512} = \frac{2}{1296} \approx 0.00154$$

    前向传播后（加权求和）：$\text{Var}(z_1) = n_{in} \cdot \text{Var}(w_1) \cdot \text{Var}(x) = 784 \times 0.00154 \times \text{Var}(x) = 1.21 \cdot \text{Var}(x)$

    经过 ReLU 激活：$\text{Var}(a_1) = \frac{1}{2} \text{Var}(z_1) = 0.61 \cdot \text{Var}(x)$（信号衰减）

    第二层、第三层类似衰减，每层信号方差约为前一层的 $0.5 \sim 0.6$ 倍。

    **三层后的累积效果**：

    粗略估算：$\text{Var}(a_3) \approx 0.61^3 \cdot \text{Var}(x) \approx 0.23 \cdot \text{Var}(x)$

    信号方差衰减到初始值的约 23%，深层网络的激活值趋于零，梯度传播受阻。

    **正确做法**：使用 He 初始化，方差公式 $\text{Var}(w) = \frac{2}{n_{in}}$ 中的因子 2 补偿了 ReLU 的信号减半，保持信号强度稳定。
    </details>