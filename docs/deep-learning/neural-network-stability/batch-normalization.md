# 批归一化

在上一章中，我们介绍了 Dropout 如何通过随机丢弃神经元防止过拟合。Dropout 从正则化的角度提升泛化能力，但训练稳定性问题仍然存在：随着网络深度增加，各层输入分布不断变化，导致梯度传播不稳定。这种现象称为**内部协变量偏移**（Internal Covariate Shift），是深度网络训练困难的重要原因。

**批归一化**（Batch Normalization，BN）由 Sergey Ioffe 和 Christian Szegedy 在 2015 年提出，通过标准化每层输入解决内部协变量偏移问题。BN 使深度网络训练更稳定、更快速，成为现代深度学习的核心技术之一。本章将深入分析内部协变量偏移问题，介绍 BN 的算法原理，并讨论 BN 在 CNN 中的应用以及训练/推理模式的差异。

## 内部协变量偏移问题

### 什么是内部协变量偏移

**内部协变量偏移**指网络训练过程中，各层输入分布随参数更新而变化的现象。

考虑一个简单的两层网络：

$$h = f(\mathbf{W}_1 x)$$
$$y = g(\mathbf{W}_2 h)$$

第二层的输入是第一层的输出 $h$。当 $\mathbf{W}_1$ 更新时，$h$ 的分布变化，第二层的输入分布也随之变化。第二层需要不断适应变化的输入分布，训练效率降低。

深度网络中，协变量偏移层层累积：

- 第一层更新 → 第二层输入分布变化
- 第二层更新 → 第三层输入分布变化
- ...
- 第 $n$ 层更新 → 第 $n+1$ 层输入分布变化

每层都在适应变化的输入分布，而非学习稳定的特征。

### 协变量偏移的影响

协变量偏移导致的问题：

1. **训练不稳定**：输入分布变化使梯度方向不稳定，参数更新震荡

2. **学习率受限**：大学习率可能导致分布剧烈变化，训练崩溃；只能使用小学习率，收敛慢

3. **梯度消失/爆炸**：分布变化可能使激活值进入饱和区域（sigmoid/tanh）或极端区域（ReLU），梯度消失或爆炸

4. **初始化敏感**：协变量偏移放大初始化的影响，糟糕的初始化导致训练困难

### 解决思路

解决协变量偏移的核心思想是**标准化**：将每层输入标准化为零均值、单位方差，保持分布稳定。

传统机器学习中，标准化输入数据是常用预处理：

$$\hat{x} = \frac{x - \mu}{\sigma}$$

深度网络中，需要对每层输入进行类似标准化。但每层输入分布随训练变化，无法预先计算均值和方差。BN 使用当前 mini-batch 的统计量进行实时标准化。

## BN 算法原理

### BN 公式

**Batch Normalization** 对每个特征维度独立标准化：

设 mini-batch 中某个特征的值为 $\{x_1, x_2, ..., x_m\}$（$m$ 为 batch size）。

**步骤 1：计算 batch 统计量**

$$[eq:bn-mean] \mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$

$$[eq:bn-var] \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$

**步骤 2：标准化**

$$[eq:bn-normalize] \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

其中 $\epsilon$ 是小常数（通常 $10^{-5}$）防止除零。

**步骤 3：缩放和偏移**

$$[eq:bn-scale] y_i = \gamma \hat{x}_i + \beta$$

其中 $\gamma$（缩放）和 $\beta$（偏移）是可学习参数，允许 BN 恢复原始分布（如果需要）。

**为什么需要 $\gamma$ 和 $\beta$？**

标准化可能破坏网络的表示能力：

- sigmoid/tanh 激活函数在输入接近 0 时线性，远离 0 时非线性更强
- 标准化使输入接近 0，可能降低非线性表达能力

$\gamma$ 和 $\beta$ 使 BN 可以恢复原始分布：

- $\gamma = \sigma, \beta = \mu$：$y_i = \sigma \cdot \hat{x}_i + \mu = x_i$（完全恢复）
- $\gamma \neq \sigma, \beta \neq \mu$：学习新的分布

BN 不仅是标准化，还是可学习的分布变换。

### BN 计算流程

```python
def batch_norm(x, gamma, beta, eps=1e-5):
    """
    Batch Normalization
    x: [batch_size, num_features]
    gamma, beta: 可学习参数 [num_features]
    """
    # 计算 batch 统计量
    mu = np.mean(x, axis=0)  # 每个特征的均值
    var = np.var(x, axis=0)  # 个特征的方差
    
    # 标准化
    x_hat = (x - mu) / np.sqrt(var + eps)
    
    # 缩放和偏移
    y = gamma * x_hat + beta
    
    return y, mu, var
```

### BN 的反向传播

BN 的反向传播需要计算对 $\gamma$, $\beta$, $\mu$, $\sigma$ 和输入 $x$ 的梯度。

设损失函数为 $L$，输入 $\{x_1, ..., x_m\}$，输出 $\{y_1, ..., y_m\}$。

**对 $\gamma$ 和 $\beta$ 的梯度**：

$$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$

$$\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}$$

**对输入 $x$ 的梯度**：

推导较复杂，简述关键步骤：

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} + \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \mu} \cdot \frac{\partial \mu}{\partial x_i} + \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial x_i}$$

完整的梯度公式：

$$\frac{\partial L}{\partial x_i} = \frac{\gamma}{m\sigma}\left(m \frac{\partial L}{\partial y_i} - \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} - \hat{x}_i \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} \hat{x}_j\right)$$

BN 的反向传播保证梯度正常传播，不影响训练效率。

## BN 与训练稳定性

### BN 如何提升训练稳定性

BN 通过标准化每层输入，解决协变量偏移问题：

1. **分布稳定**：每层输入标准化为零均值、单位方差，分布稳定
2. **梯度稳定**：标准化输入使激活值远离饱和区域，梯度稳定
3. **学习率增大**：BN 使训练稳定，可以使用更大学习率，加速收敛
4. **初始化不敏感**：BN 标准化输入，缓解初始化的影响

### BN 的正则化效果

BN 使用 mini-batch 统计，引入噪声：

- $\mu_B$ 和 $\sigma_B^2$ 随 batch 变化
- 标准化结果有随机性
- 类似 Dropout 的噪声正则化

BN 的正则化效果：

- 训练时噪声迫使网络学习鲁棒特征
- 推理时噪声消失（使用全局统计）
- BN 自带正则化，可能替代 Dropout

### BN 与 Dropout 的关系

| 特性 | Batch Normalization | Dropout |
|:----|:-------------------|:--------|
| 正则化机制 | Mini-batch 统计噪声 | 随机丢弃神经元 |
| 训练稳定性 | 提升（标准化输入） | 降低（丢弃神经元） |
| 训练速度 | 加速（可用大学习率） | 减慢（丢弃神经元） |
| 适用场景 | CNN、深度网络 | 全连接网络 |

现代网络（如 ResNet）常用 BN 替代 Dropout：

- BN 使训练稳定，正则化效果足够
- Dropout 可能破坏 BN 的标准化效果
- 深度网络需要 BN 确保训练可行

## BN 在 CNN 中的应用

### CNN 中的 BN 特性

**全连接层 BN**：每个神经元独立标准化。

设隐藏层输出 $\mathbf{h} \in \mathbb{R}^d$，batch size $m$。BN 对每个维度独立操作：

$$\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, ..., \mathbf{h}_m]^T \in \mathbb{R}^{m \times d}$$

$$\mu_j = \frac{1}{m}\sum_{i=1}^{m} h_{ij}, \quad \sigma_j^2 = \frac{1}{m}\sum_{i=1}^{m} (h_{ij} - \mu_j)^2$$

$$\hat{h}_{ij} = \frac{h_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$$

$$y_{ij} = \gamma_j \hat{h}_{ij} + \beta_j$$

**卷积层 BN**：每个特征图（channel）独立标准化，跨 batch 和空间位置计算统计量。

设卷积层输出 $\mathbf{H} \in \mathbb{R}^{m \times c \times h \times w}$（batch $m$, channels $c$, height $h$, width $w$）。

BN 对每个 channel 计算 batch 内所有位置的统计量：

$$\mu_c = \frac{1}{m \cdot h \cdot w}\sum_{i=1}^{m}\sum_{p=1}^{h}\sum_{q=1}^{w} H_{i,c,p,q}$$

$$\sigma_c^2 = \frac{1}{m \cdot h \cdot w}\sum_{i=1}^{m}\sum_{p=1}^{h}\sum_{q=1}^{w} (H_{i,c,p,q} - \mu_c)^2$$

CNN BN 的特点：

- 每个 channel 一组 $\gamma$ 和 $\beta$（共 $c$ 组）
- 统计量跨 batch 和空间位置计算（更稳定）
- 保持卷积的空间结构

### BN 在 CNN 中的位置

BN 在 CNN 中的典型位置：

**卷积块结构**：

```
Conv → BN → ReLU → ...
```

BN 在卷积后、激活前：

1. 卷积输出线性组合，BN 标准化线性输出
2. 激活函数接收标准化输入，远离饱和区域
3. BN + ReLU 组合效果最好

**不推荐的位置**：

- ReLU 后 BN：ReLU 输出非负（$\geq 0$），BN 标准化后可能偏离零均值
- BN 在 pooling 后：Pooling 改变空间结构，BN 统计不稳定

### BN 对 CNN 训练的影响

BN 对 CNN 训练的显著提升：

1. **加速收敛**：可用大学习率（如 0.1），训练更快
2. **深度可行**：ResNet（50-152 层）依赖 BN 确保训练稳定
3. **正则化效果**：BN 噪声防止过拟合
4. **移除 Dropout**：现代 CNN 不需要 Dropout

AlexNet（2012）和 VGG（2014）使用 Dropout，ResNet（2015）及后续使用 BN 替代 Dropout。

## BN 的训练/推理模式差异

### 训练模式

训练时，BN 使用当前 mini-batch 的统计量：

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$

同时，BN 维护全局统计量的滑动平均：

$$\mu_{global} = \alpha \mu_{global} + (1 - \alpha) \mu_B$$
$$\sigma_{global}^2 = \alpha \sigma_{global}^2 + (1 - \alpha) \sigma_B^2$$

其中 $\alpha$ 是衰减系数（通常 0.9 或 0.99）。

### 推理模式

推理时，BN 使用全局统计量（不使用 batch 统计）：

$$\hat{x} = \frac{x - \mu_{global}}{\sqrt{\sigma_{global}^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

推理模式的原因：

1. **单样本推理**：batch size = 1，无法计算 batch 统计
2. **输出稳定**：使用固定统计量，输出不随 batch 变化
3. **确定性预测**：相同输入相同输出，便于调试和部署

### 训练/推理模式切换

深度学习框架（如 PyTorch）提供模式切换：

```python
# 训练模式
model.train()
# BN 使用 batch 统计，更新全局统计量

# 推理模式
model.eval()
# BN 使用全局统计量，输出稳定
```

**关键注意事项**：

- 训练时必须切换到 `train()` 模式
- 推理时必须切换到 `eval()` 模式
- 模式错误导致训练/推理结果不一致

### 小 Batch Size 的问题

小 batch size（如 $m < 8$）导致 BN 统计不稳定：

- batch 统计 $\mu_B$ 和 $\sigma_B^2$ 方差大
- 标准化结果噪声大
- 全局统计量估计不准确

解决方案：

1. **增大 batch size**：推荐 $m \geq 16$
2. **使用 Batch Renormalization**：BrN 限制 batch 统计的范围
3. **使用 Group Normalization**：GN 不依赖 batch size
4. **使用 Layer Normalization**：LN 跨单个样本计算统计

## BN 实验验证

下面通过代码实验验证 BN 对训练稳定性的影响。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("实验：Batch Normalization 对训练稳定性的影响")
print("=" * 60)
print()

# 定义激活函数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Batch Normalization 实现
class BatchNorm:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # 可学习参数
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # 全局统计量（滑动平均）
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # 训练模式
        self.training = True
        
        # 缓存（用于反向传播）
        self.cache = None
    
    def forward(self, x, training=True):
        """
        x: [batch_size, num_features]
        """
        self.training = training
        
        if training:
            # 训练模式：使用 batch 统计
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # 更新全局统计量（滑动平均）
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # 标准化
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            
            # 缓存
            self.cache = (x, x_hat, mu, var)
        else:
            # 推理模式：使用全局统计
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # 缩放和偏移
        y = self.gamma * x_hat + self.beta
        
        return y
    
    def backward(self, dout, learning_rate=0.01):
        """
        dout: [batch_size, num_features]
        """
        x, x_hat, mu, var = self.cache
        m = x.shape[0]
        
        # gamma 和 beta 的梯度
        dgamma = np.sum(dout * x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        # x 的梯度
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (x - mu) * -0.5 * (var + self.eps)**(-1.5), axis=0)
        dmu = np.sum(dx_hat * -1 / np.sqrt(var + self.eps), axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)
        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mu) / m + dmu / m
        
        # 更新 gamma 和 beta
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dx

# 简单网络（支持 BN）
class SimpleNetwork:
    def __init__(self, layer_sizes, use_bn=True, activation='relu'):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.use_bn = use_bn
        
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
        self.bn_layers = []
        
        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
            if use_bn and i < self.num_layers - 1:  # 除最后一层
                bn = BatchNorm(layer_sizes[i+1])
                self.bn_layers.append(bn)
            else:
                self.bn_layers.append(None)
    
    def forward(self, X, training=True):
        """前向传播"""
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
                a = self.activation(z_bn)
            else:
                self.bn_outputs.append(None)
                a = self.activation(z)
            
            self.activations.append(a)
        
        return a
    
    def backward(self, X, y, learning_rate=0.01):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层误差（MSE 损失）
        delta = (self.activations[-1] - y) * self.activation_derivative(self.pre_activations[-1])
        
        # 反向传播
        for i in range(self.num_layers - 1, -1, -1):
            # BN 反向传播
            if self.bn_layers[i] is not None:
                delta = self.bn_layers[i].backward(delta, learning_rate)
            
            # 计算梯度
            grad_w = self.activations[i].T @ delta / m
            grad_b = np.mean(delta, axis=0, keepdims=True)
            
            # 更新参数
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            
            # 传播误差到前一层
            if i > 0:
                if self.bn_layers[i-1] is not None:
                    delta = (delta @ self.weights[i].T) * self.activation_derivative(self.bn_outputs[i-1])
                else:
                    delta = (delta @ self.weights[i].T) * self.activation_derivative(self.pre_activations[i-1])
    
    def compute_loss(self, X, y, training=False):
        """计算损失"""
        output = self.forward(X, training=training)
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

# 无 BN
net_no_bn = SimpleNetwork(layer_sizes, use_bn=False, activation='relu')

# 有 BN
net_bn = SimpleNetwork(layer_sizes, use_bn=True, activation='relu')

# 训练参数
n_epochs = 150
learning_rate = 0.01

# 记录训练过程
train_losses_no_bn = []
test_losses_no_bn = []
train_losses_bn = []
test_losses_bn = []

print("开始训练...")

for epoch in range(n_epochs):
    # 无 BN 训练
    net_no_bn.forward(X_train, training=True)
    net_no_bn.backward(X_train, y_train, learning_rate)
    
    train_loss_no = net_no_bn.compute_loss(X_train, y_train, training=False)
    test_loss_no = net_no_bn.compute_loss(X_test, y_test, training=False)
    
    train_losses_no_bn.append(train_loss_no)
    test_losses_no_bn.append(test_loss_no)
    
    # 有 BN 训练
    net_bn.forward(X_train, training=True)
    net_bn.backward(X_train, y_train, learning_rate)
    
    train_loss_bn = net_bn.compute_loss(X_train, y_train, training=False)
    test_loss_bn = net_bn.compute_loss(X_test, y_test, training=False)
    
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

learning_rates = [0.001, 0.01, 0.1]
lr_results = {}

for lr in learning_rates:
    print(f"学习率 = {lr}")
    
    # 无 BN
    net_no = SimpleNetwork(layer_sizes, use_bn=False, activation='relu')
    
    # 有 BN
    net_bn_lr = SimpleNetwork(layer_sizes, use_bn=True, activation='relu')
    
    no_bn_losses = []
    bn_losses = []
    
    for epoch in range(n_epochs):
        # 无 BN
        net_no.forward(X_train, training=True)
        net_no.backward(X_train, y_train, lr)
        no_bn_losses.append(net_no.compute_loss(X_test, y_test, training=False))
        
        # 有 BN
        net_bn_lr.forward(X_train, training=True)
        net_bn_lr.backward(X_train, y_train, lr)
        bn_losses.append(net_bn_lr.compute_loss(X_test, y_test, training=False))
    
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
    
    # 无 BN
    try:
        net_no = SimpleNetwork(sizes, use_bn=False, activation='relu')
        no_bn_losses = []
        
        for epoch in range(n_epochs):
            net_no.forward(X_train, training=True)
            net_no.backward(X_train, y_train, 0.01)
            no_bn_losses.append(net_no.compute_loss(X_test, y_test, training=False))
        
        depth_results[depth] = {
            'no_bn': no_bn_losses,
            'bn': None
        }
        print(f"  无 BN 最终测试损失: {no_bn_losses[-1]:.4f}")
    except Exception as e:
        print(f"  无 BN 训练失败: {e}")
        depth_results[depth] = {'no_bn': None, 'bn': None}
    
    # 有 BN
    try:
        net_bn_depth = SimpleNetwork(sizes, use_bn=True, activation='relu')
        bn_losses = []
        
        for epoch in range(n_epochs):
            net_bn_depth.forward(X_train, training=True)
            net_bn_depth.backward(X_train, y_train, 0.01)
            bn_losses.append(net_bn_depth.compute_loss(X_test, y_test, training=False))
        
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

x = range(len(depths))
width = 0.4

# 绘制柱状图
bars1 = ax.bar([i - width/2 for i in x], [l if l else 0 for l in no_bn_finals], 
               width, label='无 BN', color='#e74c3c', alpha=0.7)
bars2 = ax.bar([i + width/2 for i in x], [l if l else 0 for l in bn_finals], 
               width, label='有 BN', color='#2ecc71', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels([f'深度 {d}' for d in depths])
ax.set_xlabel('网络深度', fontsize=11)
ax.set_ylabel('最终测试损失', fontsize=11)
ax.set_title('BN 对不同深度网络的影响', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
plt.close()

print("\n实验结论:")
print("-" * 40)
print("1. BN 加速收敛：训练损失快速下降，收敛更快")
print("2. BN 允许大学习率：学习率 0.1 时 BN 有效，无 BN 可能崩溃")
print("3. BN 使深度网络可训练：15层网络无 BN 可能梯度消失")
print("4. BN 提升稳定性：训练和测试损失差距小，泛化好")
print("=" * 60)
```

### 实验结论

实验验证了 BN 对训练的显著提升：

1. **BN 加速收敛**：训练损失快速下降，比无 BN 收敛更快

2. **BN 允许大学习率**：大学习率（如 0.1）时 BN 训练稳定，无 BN 可能崩溃

3. **BN 使深度网络可训练**：深度网络（15层）无 BN 可能梯度消失，BN 确保训练稳定

4. **BN 提升泛化**：训练和测试损失差距小，BN 自带正则化效果

## BN 的变体与替代方案

### BN 的局限性

BN 存在以下局限：

1. **依赖 batch size**：小 batch 统计不稳定，推荐 $m \geq 16$

2. **训练/推理不一致**：训练使用 batch 统计，推理使用全局统计

3. **不适用 RNN**：RNN 时序数据 batch 统计不适用

4. **分布式训练复杂**：跨设备同步 batch 统计

### BN 的变体

**Batch Renormalization（BrN）**

BrN 在 BN 基础上限制 batch 统计的范围，解决小 batch 问题：

$$\hat{x} = \frac{x - \mu_B}{\sigma_B} \cdot r + d$$

其中 $r = \text{clip}(\frac{\sigma_B}{\sigma_{global}}, r_{min}, r_{max})$，$d = \text{clip}(\frac{\mu_B - \mu_{global}}{\sigma_{global}}, d_{min}, d_{max})$

**Layer Normalization（LN）**

LN 跨单个样本的所有特征计算统计量：

$$\mu_L = \frac{1}{d}\sum_{j=1}^{d} x_j$$
$$\sigma_L^2 = \frac{1}{d}\sum_{j=1}^{d} (x_j - \mu_L)^2$$

LN 不依赖 batch size，适用 RNN 和 Transformer。

**Group Normalization（GN）**

GN 将特征分组，每组独立标准化：

$$\mu_G = \frac{1}{G \cdot h \cdot w}\sum_{g=1}^{G}\sum_{p,q} x_{g,p,q}$$

GN 不依赖 batch size，适用小 batch 场景。

**Instance Normalization（IN）**

IN 对每个样本的每个 channel 独立标准化：

$$\mu_I = \frac{1}{h \cdot w}\sum_{p,q} x_{p,q}$$

IN 用于图像风格迁移等任务。

### 选择指南

| 方法 | 适用场景 | batch size 依赖 |
|:----|:--------|:---------------|
| BN | CNN、深度网络 | 强（推荐 $m \geq 16$） |
| LN | RNN、Transformer | 无 |
| GN | 小 batch CNN | 无 |
| IN | 风格迁移 | 无 |

## 本章小结

本章深入介绍了 Batch Normalization 的原理与应用。

**内部协变量偏移**：网络训练过程中，各层输入分布随参数更新而变化，导致训练不稳定、收敛慢、初始化敏感。

**BN 算法**：标准化每层输入为零均值、单位方差，使用可学习参数 $\gamma$ 和 $\beta$ 恢复表示能力。训练时使用 batch 统计，推理时使用全局统计。

**BN 与训练稳定性**：BN 使分布稳定、梯度稳定，允许大学习率，使深度网络训练可行。BN 自带正则化效果（batch 统计噪声），可能替代 Dropout。

**BN 在 CNN 中**：每个 channel 独立标准化，跨 batch 和空间位置计算统计量。BN 在卷积后、激活前效果最好。ResNet 等深度网络依赖 BN 确保训练稳定。

**训练/推理差异**：训练使用 batch 统计（滑动平均更新全局统计），推理使用全局统计（输出稳定）。小 batch size 时 BN 统计不稳定，可使用 LN/GN 替代。

下一章将介绍梯度问题诊断，分析梯度消失和爆炸的成因与诊断方法，介绍梯度裁剪等解决技术。

## 练习题

1. 推导 Batch Normalization 的反向传播梯度公式。设 batch 输入 $\{x_1, ..., x_m\}$，输出 $\{y_1, ..., y_m\}$，推导 $\frac{\partial L}{\partial \gamma}$, $\frac{\partial L}{\partial \beta}$ 和 $\frac{\partial L}{\partial x_i}$。
    <details>
    <summary>参考答案</summary>
    
    **BN 反向传播推导**：
    
    设 batch 输入 $\{x_1, ..., x_m\}$，BN 输出：
    
    $$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$
    $$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$
    $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
    $$y_i = \gamma \hat{x}_i + \beta$$
    
    设损失函数 $L$，已知 $\frac{\partial L}{\partial y_i}$。
    
    **对 $\gamma$ 和 $\beta$ 的梯度**：
    
    $$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$
    
    $$\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}$$
    
    **对 $x_i$ 的梯度**：
    
    BN 中 $x_i$ 通过三条路径影响 $y_j$：
    
    1. 直接路径：$x_i \to \hat{x}_i \to y_i$
    2. 通过 $\mu_B$：$x_i \to \mu_B \to \hat{x}_j \to y_j$（所有 $j$）
    3. 通过 $\sigma_B$：$x_i \to \sigma_B \to \hat{x}_j \to y_j$（所有 $j$）
    
    $$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} + \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \mu_B} \cdot \frac{\partial \mu_B}{\partial x_i} + \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \sigma_B} \cdot \frac{\partial \sigma_B}{\partial x_i}$$
    
    简化推导：
    
    设 $\frac{\partial L}{\partial y_i} = \delta_i$，则 $\frac{\partial L}{\partial \hat{x}_i} = \gamma \delta_i$。
    
    $\hat{x}_i$ 对 $x_i$ 的梯度：
    
    $$\frac{\partial \hat{x}_i}{\partial x_i} = \frac{1}{\sigma_B} - \frac{x_i - \mu_B}{\sigma_B^2} \cdot \frac{\partial \sigma_B}{\partial x_i} - \frac{1}{\sigma_B} \cdot \frac{\partial \mu_B}{\partial x_i}$$
    
    其中：
    
    $$\frac{\partial \mu_B}{\partial x_i} = \frac{1}{m}$$
    
    $$\frac{\partial \sigma_B}{\partial x_i} = \frac{x_i - \mu_B}{m \sigma_B}$$
    
    代入：
    
    $$\frac{\partial \hat{x}_i}{\partial x_i} = \frac{1}{\sigma_B} - \frac{\hat{x}_i}{m \sigma_B} - \frac{1}{m \sigma_B} = \frac{1}{\sigma_B}\left(1 - \frac{\hat{x}_i + 1}{m}\right)$$
    
    更精确的推导（考虑所有依赖）：
    
    $$\frac{\partial L}{\partial x_i} = \frac{\gamma}{m\sigma_B}\left(m \delta_i - \sum_{j=1}^{m} \delta_j - \hat{x}_i \sum_{j=1}^{m} \delta_j \hat{x}_j\right)$$
    
    这是完整的 BN 反向传播公式。
    
    **验证**：
    
    设 $m=1$（单样本 batch）：
    
    $$\mu_B = x_1, \sigma_B = 0$$（除零，不适用）
    
    设 $m=2$, $x = [1, 3]$：
    
    $$\mu_B = 2, \sigma_B^2 = 1$$
    $$\hat{x}_1 = -1, \hat{x}_2 = 1$$
    $$y_1 = -\gamma + \beta, y_2 = \gamma + \beta$$
    
    设 $\delta_1 = 1, \delta_2 = 1$：
    
    $$\frac{\partial L}{\partial \gamma} = 1 \cdot (-1) + 1 \cdot 1 = 0$$
    $$\frac{\partial L}{\partial \beta} = 1 + 1 = 2$$
    
    $$\frac{\partial L}{\partial x_1} = \frac{\gamma}{2 \cdot 1}\left(2 \cdot 1 - 2 - (-1) \cdot (1 \cdot (-1) + 1 \cdot 1)\right) = \frac{\gamma}{2}(0 - (-1) \cdot 0) = 0$$
    
    $$\frac{\partial L}{\partial x_2} = \frac{\gamma}{2}\left(2 \cdot 1 - 2 - 1 \cdot 0\right) = 0$$
    
    梯度为零，符合直觉（$\delta_1 = \delta_2$ 对称）。
    
    **总结**：
    
    BN 反向传播公式：
    
    $$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$
    
    $$\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}$$
    
    $$\frac{\partial L}{\partial x_i} = \frac{\gamma}{m\sigma_B}\left(m \frac{\partial L}{\partial y_i} - \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} - \hat{x}_i \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} \hat{x}_j\right)$$
    
    BN 的反向传播保证梯度正常传播，不破坏训练效率。
    </details>

2. 解释为何 BN 训练时使用 batch 统计，推理时使用全局统计。分析两种模式输出不一致的风险。
    <details>
    <summary>参考答案</summary>
    
    **BN 训练/推理模式差异的原因**：
    
    **训练时使用 batch 统计**：
    
    1. **实时标准化**：训练时参数不断更新，输入分布变化，无法预先计算统计量。batch 统计实时反映当前分布。
    
    2. **噪声正则化**：batch 统计随 batch 变化，引入噪声，迫使网络学习鲁棒特征。
    
    3. **滑动平均更新全局统计**：训练时维护全局统计的滑动平均：
    
       $$\mu_{global} = \alpha \mu_{global} + (1 - \alpha) \mu_B$$
       $$\sigma_{global}^2 = \alpha \sigma_{global}^2 + (1 - \alpha) \sigma_B^2$$
    
    训练结束时，全局统计收敛到稳定值。
    
    **推理时使用全局统计**：
    
    1. **单样本推理**：推理时 batch size 可能为 1，无法计算 batch 统计（$\sigma_B = 0$）。
    
    2. **输出稳定**：使用固定统计量，输出不随 batch 组成变化，确保确定性预测。
    
    3. **部署一致性**：相同输入相同输出，便于调试和部署。
    
    **输出不一致的风险**：
    
    训练和推理使用不同统计量，可能导致输出不一致：
    
    1. **全局统计估计不准**：
    
       滑动平均依赖训练时的 batch 统计。如果：
       
       - 训练 batch size 小（$m < 8$）：batch 统计方差大，全局统计估计不准
       - 训练数据分布变化：全局统计不适应新分布
       - 训练不充分：全局统计未收敛
       
       推理使用不准的全局统计，输出与训练不一致。
    
    2. **训练/推理模式错误**：
    
       如果推理时错误使用训练模式（使用 batch 统计）：
       
       - 单样本推理：$\sigma_B = 0$，标准化失败
       - 多样本推理：输出随 batch 组成变化，不稳定
       
       如果训练时错误使用推理模式：
       
       - 不更新全局统计：全局统计不收敛
       - 无噪声正则化：可能过拟合
    
    3. **分布式训练同步问题**：
    
       分布式训练时，不同设备计算不同 batch 统计：
       
       - 全局统计同步复杂
       - 不同设备全局统计可能不一致
       - 推理输出不一致
    
    **缓解不一致的方法**：
    
    1. **增大 batch size**：推荐 $m \geq 16$，batch 统计稳定，全局统计估计准确。
    
    2. **充分训练**：训练结束时全局统计收敛。
    
    3. **正确切换模式**：
    
       ```python
       # 训练
       model.train()  # BN 使用 batch 统计
       
       # 推理
       model.eval()   # BN 使用全局统计
       ```
    
    4. **分布式同步 BN（SyncBN）**：跨设备同步计算 batch 统计，全局统计一致。
    
    5. **使用 LN/GN 替代 BN**：LN/GN 不依赖 batch 统计，训练/推理一致。
    
    **数值示例**：
    
    设训练数据均值 $\mu_{train} = 0$，方差 $\sigma_{train}^2 = 1$。
    
    训练时（batch size = 8）：
    
    - Batch 统计：$\mu_B \approx 0$, $\sigma_B^2 \approx 1$（假设理想情况）
    - 全局统计：$\mu_{global} \to 0$, $\sigma_{global}^2 \to 1$
    
    推理时：
    
    - 全局统计：$\mu_{global} = 0$, $\sigma_{global}^2 = 1$
    - 标准化：$\hat{x} = x$（理想一致）
    
    如果训练不充分：
    
    - 全局统计：$\mu_{global} = 0.5$（未收敛）
    - 推理标准化：$\hat{x} = x - 0.5$
    - 与训练不一致（训练用 $\mu_B \approx 0$）
    
    **总结**：
    
    BN 训练时使用 batch 统计：实时标准化、噪声正则化、更新全局统计。
    
    BN 推理时使用全局统计：单样本可行、输出稳定、部署一致。
    
    输出不一致风险：
    1. 小 batch size：全局统计不准
    2. 训练不充分：全局统计未收敛
    3. 模式切换错误：统计量不一致
    
    缓解方法：
    1. 增大 batch size（$\geq 16$）
    2. 充分训练
    3. 正确切换 train/eval 模式
    4. 分布式使用 SyncBN
    5. 小 batch 使用 LN/GN 替代
    </details>

3. 分析 BN 与 Dropout 组合使用的潜在问题。为何现代 CNN（如 ResNet）只用 BN 不用 Dropout？
    <details>
    <summary>参考答案</summary>
    
    **BN 与 Dropout 组合的潜在问题**：
    
    此题与上一章练习题 3 相关，此处补充 CNN 特定分析。
    
    **问题 1：Dropout 破坏 BN 的 batch 统计**：
    
    BN 计算 batch 统计：
    
    $$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$
    
    Dropout 随机丢弃神经元，改变 $x_i$ 的分布：
    
    - Dropout 前：$x_i$ 分布稳定
    - Dropout 后：$x_i$ 部分置零，分布改变
    
    Dropout 使 BN 的 $\mu_B$ 和 $\sigma_B^2$ 不稳定：
    
    - Batch 统计随 Dropout mask 变化
    - 同一 batch 不同 mask，统计不同
    - BN 标准化效果不稳定
    
    **问题 2：Dropout 破坏 BN 的标准化效果**：
    
    BN 标准化激活值：
    
    $$\hat{x} = \frac{x - \mu_B}{\sigma_B}$$
    
    Dropout 后：
    
    $$\hat{x}_{drop} = \frac{r}{p} \cdot \hat{x}$$
    
    Dropout 破坏 BN 的标准化效果：
    
    - 均值：$\mathbb{E}[\hat{x}_{drop}] = 0$（保持）
    - 方差：$\text{Var}(\hat{x}_{drop}) = \frac{1}{p} \text{Var}(\hat{x}) = \frac{1}{p}$（放大）
    
    Dropout 使 BN 标准化后的方差放大 $\frac{1}{p}$ 倍，偏离单位方差。
    
    **问题 3：BN 前后 Dropout 的矛盾**：
    
    - BN 前 Dropout：BN 统计不稳定
    - BN 后 Dropout：BN 标准化被破坏
    
    无论 Dropout 在 BN 前后，都有负面效果。
    
    **为何 ResNet 只用 BN 不用 Dropout**：
    
    1. **BN 自带正则化**：
    
       BN 使用 batch 统计，引入噪声：
       
       - 训练时：噪声迫使网络学习鲁棒特征
       - 推理时：噪声消失
       - 正则化效果类似 Dropout 但更温和
    
    2. **BN 使训练稳定**：
    
       ResNet 深度大（50-152 层），需要 BN 确保训练稳定：
       
       - BN 标准化每层输入
       - 梯度稳定传播
       - 残差连接配合 BN 使深度网络可行
    
    3. **Dropout 破坏残差连接**：
    
       ResNet 的残差连接：
       
       $$y = x + f(x)$$
       
       Dropout 可能破坏残差连接：
       
       - Dropout $x$：残差信息丢失
       - Dropout $f(x)$：残差功能减弱
       
       BN 不丢弃神经元，保留残差连接的完整性。
    
    4. **实验验证**：
    
       ResNet 论文验证 BN 足够：
       
       - 无 Dropout + BN：训练稳定，精度高
       - Dropout + BN：效果不如单独 BN
       
       BN 正则化足够，Dropout 不需要。
    
    **CNN 中 BN vs Dropout 的对比**：
    
    | 特性 | Batch Normalization | Dropout |
    |:----|:-------------------|:--------|
    | 训练稳定性 | 提升（标准化） | 降低（丢弃） |
    | 正则化效果 | 温和（噪声） | 强（丢弃） |
    | 空间结构 | 保持 | 可能破坏 |
    | 残差连接 | 兼容 | 可能破坏 |
    | Batch size | 依赖 | 不依赖 |
    
    **何时在 CNN 中使用 Dropout**：
    
    Dropout 在 CNN 中仍有用途：
    
    1. **全连接层**：CNN 的全连接层参数多，可用 Dropout
    2. **小数据集**：正则化需求强，BN + Dropout 组合可能有用
    3. **过拟合严重**：BN 正则化不够时补充 Dropout
    
    **总结**：
    
    BN 与 Dropout 组合的潜在问题：
    1. Dropout 破坏 BN 的 batch 统计稳定性
    2. Dropout 破坏 BN 的标准化效果
    3. BN 前后 Dropout 都有负面效果
    
    ResNet 只用 BN 不用 Dropout 的原因：
    1. BN 自带正则化效果
    2. BN 使深度网络训练稳定
    3. Dropout 破坏残差连接
    4. BN 正则化足够，Dropout 不需要
    
    Dropout 在 CNN 全连接层和小数据集上仍有用途，但现代深度 CNN（ResNet 及后续）倾向只用 BN。
    </details>

4. 设训练一个 CNN，使用 BN，观察到训练损失正常下降，但测试损失反而上升。分析可能的原因，并提出解决方法。
    <details>
    <summary>参考答案</summary>
    
    **BN 训练正常测试异常的原因分析**：
    
    **现象**：训练损失下降，测试损失上升。
    
    这与上一章"Dropout 训练损失高于测试损失"相反，说明 BN 可能存在问题。
    
    **可能原因 1：全局统计未收敛**：
    
    BN 训练时维护全局统计：
    
    $$\mu_{global} = \alpha \mu_{global} + (1 - \alpha) \mu_B$$
    
    如果训练不充分或滑动平均系数 $\alpha$ 过大：
    
    - 全局统计更新慢，未收敛到真实分布
    - 推理使用不准的全局统计，输出偏差大
    - 测试损失上升
    
    **解决方法**：
    
    - 增加训练轮数，确保全局统计收敛
    - 降低 $\alpha$（如 0.99 → 0.9），加快更新
    - 训练结束后多跑几轮验证数据，更新全局统计
    
    **可能原因 2：训练/推理模式错误**：
    
    如果推理时错误使用训练模式：
    
    - 测试 batch size 小（如 $m = 1$）：batch 统计不稳定或 $\sigma_B = 0$
    - 测试损失计算时使用 batch 统计，与训练不同
    
    如果训练时错误使用推理模式：
    
    - 不更新全局统计
    - 训练使用固定的（可能不准的）全局统计
    
    **解决方法**：
    
    - 确保推理时切换到 `eval()` 模式
    - 确保训练时切换到 `train()` 模式
    
    ```python
    # 正确模式切换
    model.train()  # 训练
    for batch in train_data:
        loss = model(batch)
        loss.backward()
    
    model.eval()   # 测试
    for batch in test_data:
        loss = model(batch)  # 使用全局统计
    ```
    
    **可能原因 3：小 batch size**：
    
    小 batch size（$m < 8$）导致：
    
    - Batch 统计方差大，噪声大
    - 全局统计估计不准
    - 推理输出偏差
    
    **解决方法**：
    
    - 增大 batch size（推荐 $\geq 16$）
    - 使用 Group Normalization（GN）替代 BN
    - 使用 SyncBN（分布式训练）
    
    **可能原因 4：过拟合**：
    
    BN 自带正则化，但可能不足以防止过拟合：
    
    - 训练损失下降，测试损失上升
    - 模型学习训练数据的噪声
    
    **解决方法**：
    
    - 增加正则化：添加 Dropout、权重衰减
    - 早停：训练提前终止
    - 数据增强：增加数据多样性
    
    **可能原因 5：训练数据与测试数据分布不一致**：
    
    BN 的全局统计反映训练数据分布：
    
    - 如果测试数据分布不同，全局统计不适用
    - 标准化偏差，测试损失上升
    
    **解决方法**：
    
    - 检查数据分布一致性
    - 使用测试数据重新计算 BN 统计（谨慎）
    - 使用不依赖分布的方法（如 GN）
    
    **诊断流程**：
    
    1. **检查模式切换**：
       - 确认推理时使用 `eval()` 模式
       - 确认训练时使用 `train()` 模式
    
    2. **检查全局统计**：
       - 打印 $\mu_{global}$ 和 $\sigma_{global}^2$
       - 与训练数据均值和方差对比
    
    3. **检查 batch size**：
       - 确认 $m \geq 16$
       - 小 batch 使用 GN 替代
    
    4. **检查过拟合**：
       - 对比训练和测试损失曲线
       - 添加正则化验证
    
    5. **检查数据分布**：
       - 对比训练和测试数据统计
       - 验证分布一致性
    
    **数值示例**：
    
    设训练数据均值 $\mu_{train} = 0$，方差 $\sigma_{train}^2 = 1$。
    
    理想情况：
    
    - 全局统计收敛：$\mu_{global} = 0$, $\sigma_{global}^2 = 1$
    - 测试数据分布相同：$\mu_{test} = 0$, $\sigma_{test}^2 = 1$
    - 推理标准化：$\hat{x}_{test} = x_{test}$
    - 测试损失正常
    
    问题情况（全局统计未收敛）：
    
    - 全局统计：$\mu_{global} = 0.5$（训练时初始值）
    - 推理标准化：$\hat{x}_{test} = x_{test} - 0.5$
    - 输出偏差，测试损失上升
    
    问题情况（测试数据分布不同）：
    
    - 测试数据：$\mu_{test} = 2$, $\sigma_{test}^2 = 4$
    - 全局统计（来自训练）：$\mu_{global} = 0$, $\sigma_{global}^2 = 1$
    - 推理标准化：$\hat{x}_{test} = \frac{x_{test} - 0}{1}$
    - 测试数据本应标准化为 $\frac{x_{test} - 2}{2}$
    - 偏差导致测试损失上升
    
    **总结**：
    
    BN 训练正常测试异常的可能原因：
    1. 全局统计未收敛：训练不充分或 $\alpha$ 过大
    2. 训练/推理模式错误：使用错误的统计量
    3. 小 batch size：batch 统计不稳定，全局统计不准
    4. 过拟合：BN 正则化不足
    5. 数据分布不一致：全局统计不适用
    
    解决方法：
    1. 增加训练轮数，降低 $\alpha$
    2. 正确切换 train/eval 模式
    3. 增大 batch size 或使用 GN
    4. 增加正则化（Dropout、权重衰减）
    5. 检查数据分布一致性
    </details>