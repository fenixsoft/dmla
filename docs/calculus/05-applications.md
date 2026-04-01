---
title: "应用场景：微积分在机器学习中的实践"
---

# 应用场景：微积分在机器学习中的实践

前四章建立了微积分的理论基础，并通过 NumPy 和 PyTorch 实现了数值计算。本章将这些知识综合应用，深入剖析机器学习中最核心的算法——梯度下降和反向传播，并介绍优化算法的演进、损失函数的设计考量以及学习率调优策略。这些内容是理解深度学习训练过程的关键。

## 梯度下降算法

梯度下降是机器学习最基础、最重要的优化算法。它的核心思想简单直观：沿着函数下降最快的方向（负梯度方向）移动，逐步逼近最小值。

### 一元函数的梯度下降

首先看一个简单的一元函数例子。设 $f(x) = (x - 3)^2$，这是一个简单的凸函数，最小值在 $x = 3$ 处。

梯度下降的更新规则为：

$$x_{t+1} = x_t - \eta \cdot f'(x_t)$$

其中 $\eta$ 是学习率，$f'(x) = 2(x - 3)$ 是函数的导数。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """目标函数: f(x) = (x-3)²"""
    return (x - 3) ** 2

def df(x):
    """导数: f'(x) = 2(x-3)"""
    return 2 * (x - 3)

def gradient_descent_1d(start, learning_rate, num_steps):
    """一元函数梯度下降"""
    x = start
    history = [x]

    for _ in range(num_steps):
        grad = df(x)
        x = x - learning_rate * grad
        history.append(x)

    return np.array(history)

# 运行梯度下降
start = 0.0
learning_rate = 0.1
num_steps = 20

history = gradient_descent_1d(start, learning_rate, num_steps)

# 可视化
x = np.linspace(-1, 5, 100)
y = f(x)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：函数和梯度下降轨迹
axes[0].plot(x, y, 'b-', label='f(x) = (x-3)²')
axes[0].plot(history, f(history), 'ro-', markersize=4, label='梯度下降轨迹')
axes[0].axvline(x=3, color='g', linestyle='--', label='最优点 x=3')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title(f'梯度下降（学习率 η={learning_rate}）')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：损失下降曲线
axes[1].plot(range(len(history)), f(history), 'r-o', markersize=4)
axes[1].set_xlabel('迭代步数')
axes[1].set_ylabel('f(x)')
axes[1].set_title('目标函数值随迭代下降')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print(f"初始值: x = {start}")
print(f"最终值: x = {history[-1]:.6f}")
print(f"最优解: x = 3.0")
print(f"误差: {abs(history[-1] - 3):.6f}")
```

### 多元函数的梯度下降

在机器学习中，我们通常处理多元函数——损失函数是多个参数的函数。设参数为向量 $\boldsymbol{\theta}$，梯度下降的更新规则为：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla L(\boldsymbol{\theta}_t)$$

其中 $\nabla L$ 是损失函数的梯度。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 目标函数: f(x,y) = x² + y²
def f_2d(x, y):
    return x ** 2 + y ** 2

def grad_f_2d(x, y):
    """梯度: ∇f = (2x, 2y)"""
    return np.array([2 * x, 2 * y])

def gradient_descent_2d(start, learning_rate, num_steps):
    """二元函数梯度下降"""
    theta = np.array(start, dtype=float)
    history = [theta.copy()]

    for _ in range(num_steps):
        grad = grad_f_2d(theta[0], theta[1])
        theta = theta - learning_rate * grad
        history.append(theta.copy())

    return np.array(history)

# 运行梯度下降
start = [2.0, 2.0]
learning_rate = 0.1
num_steps = 20

history = gradient_descent_2d(start, learning_rate, num_steps)

# 可视化
x = np.linspace(-2.5, 2.5, 50)
y = np.linspace(-2.5, 2.5, 50)
X, Y = np.meshgrid(x, y)
Z = f_2d(X, Y)

fig, ax = plt.subplots(figsize=(8, 8))

# 等高线
contour = ax.contour(X, Y, Z, levels=20, cmap='coolwarm')
ax.clabel(contour, inline=True, fontsize=8)

# 梯度下降轨迹
ax.plot(history[:, 0], history[:, 1], 'ko-', markersize=4, label='梯度下降轨迹')
ax.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='起点')
ax.plot(history[-1, 0], history[-1, 1], 'r*', markersize=15, label='终点')
ax.plot(0, 0, 'b+', markersize=15, label='最优点 (0,0)')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('二元函数梯度下降')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print(f"起点: ({start[0]}, {start[1]})")
print(f"终点: ({history[-1, 0]:.6f}, {history[-1, 1]:.6f})")
print(f"最优解: (0, 0)")
```

### 收敛条件与停止条件

梯度下降的收敛取决于多个因素：

1. **学习率 $\eta$**：
   - 太大：可能震荡甚至发散
   - 太小：收敛太慢
   - 理想：足够大以快速收敛，又足够小以保证稳定

2. **停止条件**：
   - 梯度接近零：$\|\nabla L\| < \epsilon$
   - 损失变化小：$|L_{t+1} - L_t| < \epsilon$
   - 达到最大迭代次数

```python runnable
import numpy as np

def gradient_descent_with_stopping(f, grad_f, start, learning_rate=0.1,
                                   max_steps=1000, tol=1e-6):
    """带停止条件的梯度下降"""
    theta = np.array(start, dtype=float)
    history = [theta.copy()]

    for step in range(max_steps):
        grad = grad_f(theta)

        # 检查梯度是否接近零
        if np.linalg.norm(grad) < tol:
            print(f"收敛于第 {step} 步（梯度接近零）")
            break

        theta_new = theta - learning_rate * grad

        # 检查损失变化
        if abs(f(theta_new) - f(theta)) < tol:
            print(f"收敛于第 {step} 步（损失变化小）")
            break

        theta = theta_new
        history.append(theta.copy())
    else:
        print(f"达到最大迭代次数 {max_steps}")

    return np.array(history)

# 测试不同学习率的影响
def f(xy):
    return xy[0] ** 2 + xy[1] ** 2

def grad_f(xy):
    return np.array([2 * xy[0], 2 * xy[1]])

print("=== 学习率 0.01（较小）===")
history_small = gradient_descent_with_stopping(f, grad_f, [2.0, 2.0], learning_rate=0.01)

print("\n=== 学习率 0.1（适中）===")
history_medium = gradient_descent_with_stopping(f, grad_f, [2.0, 2.0], learning_rate=0.1)

print("\n=== 学习率 0.5（较大）===")
history_large = gradient_descent_with_stopping(f, grad_f, [2.0, 2.0], learning_rate=0.5)

print("\n=== 学习率 1.0（过大，可能发散）===")
history_too_large = gradient_descent_with_stopping(f, grad_f, [2.0, 2.0], learning_rate=1.0)
```

## 反向传播机制

反向传播（Backpropagation）是神经网络训练的核心算法，它利用链式法则高效计算损失函数关于每个参数的梯度。

### 神经网络的结构

一个简单的全连接神经网络包含：
- **输入层**：接收特征向量
- **隐藏层**：进行非线性变换
- **输出层**：产生预测结果

每一层的计算可以分解为：
1. 线性变换：$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$
2. 激活函数：$\mathbf{a} = \sigma(\mathbf{z})$

### 前向传播与反向传播

**前向传播**：从输入到输出，逐层计算。

**反向传播**：从输出到输入，利用链式法则计算梯度。

```python runnable
import numpy as np

# 激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 简单两层神经网络
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        """初始化网络参数"""
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

        # 缓存中间结果（用于反向传播）
        self.cache = {}

    def forward(self, X):
        """前向传播"""
        # 第一层
        self.cache['X'] = X
        self.cache['Z1'] = np.dot(X, self.W1) + self.b1
        self.cache['A1'] = sigmoid(self.cache['Z1'])

        # 第二层
        self.cache['Z2'] = np.dot(self.cache['A1'], self.W2) + self.b2
        self.cache['A2'] = sigmoid(self.cache['Z2'])

        return self.cache['A2']

    def backward(self, y_true, learning_rate=0.1):
        """反向传播"""
        m = y_true.shape[0]  # 样本数

        # 输出层误差
        dZ2 = self.cache['A2'] - y_true  # 交叉熵损失 + sigmoid 的简化形式
        dW2 = np.dot(self.cache['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 隐藏层误差（链式法则）
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.cache['Z1'])
        dW1 = np.dot(self.cache['X'].T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 更新参数（梯度下降）
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return dW1, db1, dW2, db2

    def compute_loss(self, y_pred, y_true):
        """计算损失（交叉熵）"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 创建网络并训练
np.random.seed(42)
nn = SimpleNN(input_size=2, hidden_size=4, output_size=1)

# 简单的 XOR 问题
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("训练简单的神经网络解决 XOR 问题")
print("-" * 50)

# 训练
for epoch in range(1000):
    # 前向传播
    y_pred = nn.forward(X)
    loss = nn.compute_loss(y_pred, y)

    # 反向传播
    nn.backward(y, learning_rate=0.5)

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}: Loss = {loss:.6f}")

# 测试
print("\n训练结果:")
y_pred = nn.forward(X)
for i, (x_i, y_i, pred) in enumerate(zip(X, y, y_pred)):
    print(f"  输入: {x_i}, 真实值: {y_i[0]}, 预测值: {pred[0]:.4f}")
```

### 链式法则的工程实现

反向传播的本质是**动态规划**的应用：每个节点只需要计算局部梯度，然后传递给上游节点。这种方式避免了重复计算，使得梯度计算的时间复杂度与前向传播相同。

关键点：
1. **缓存中间结果**：前向传播时保存每层的输入和输出
2. **逐层反向传递**：从输出层开始，逐层向前计算梯度
3. **局部梯度计算**：每层只需要知道输入和上游梯度

## 优化算法演进

虽然基础梯度下降算法有效，但在实际应用中存在一些问题：收敛慢、容易陷入局部最小值、对学习率敏感等。研究者提出了多种改进算法。

### SGD（随机梯度下降）

**随机梯度下降**（Stochastic Gradient Descent, SGD）每次只用一个样本（或一小批样本）计算梯度，而不是全部样本。

优点：
- 计算效率高
- 引入随机性，有助于跳出局部最小值
- 可以在线学习（逐样本更新）

```python
def sgd_update(params, grads, learning_rate):
    """SGD 参数更新"""
    for param, grad in zip(params, grads):
        param -= learning_rate * grad
```

### Momentum（动量法）

**动量法**（Momentum）引入"速度"概念，累积历史梯度信息：

$$\mathbf{v}_t = \gamma \mathbf{v}_{t-1} + \eta \nabla L$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \mathbf{v}_t$$

其中 $\gamma$ 是动量系数（通常取 0.9）。

动量法能够：
- 加速收敛（沿一致方向）
- 抑制震荡（平滑梯度变化）

```python runnable
import numpy as np
import matplotlib.pyplot as plt

def f(xy):
    """Rosenbrock 函数的简化版本"""
    x, y = xy
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def grad_f(xy):
    x, y = xy
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])

def sgd_trajectory(start, learning_rate=0.001, num_steps=500):
    """SGD 轨迹"""
    theta = np.array(start, dtype=float)
    history = [theta.copy()]

    for _ in range(num_steps):
        grad = grad_f(theta)
        theta = theta - learning_rate * grad
        history.append(theta.copy())

    return np.array(history)

def momentum_trajectory(start, learning_rate=0.001, gamma=0.9, num_steps=500):
    """Momentum 轨迹"""
    theta = np.array(start, dtype=float)
    v = np.zeros_like(theta)
    history = [theta.copy()]

    for _ in range(num_steps):
        grad = grad_f(theta)
        v = gamma * v + learning_rate * grad
        theta = theta - v
        history.append(theta.copy())

    return np.array(history)

# 比较两种方法
start = [-1.0, 1.0]

sgd_hist = sgd_trajectory(start)
mom_hist = momentum_trajectory(start)

# 绘制轨迹
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([[f([xi, yi]) for xi in x] for yi in y])

fig, ax = plt.subplots(figsize=(10, 8))

# 等高线
contour = ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='coolwarm', alpha=0.6)

# 轨迹
ax.plot(sgd_hist[:, 0], sgd_hist[:, 1], 'b-', label='SGD', linewidth=2)
ax.plot(mom_hist[:, 0], mom_hist[:, 1], 'r-', label='Momentum', linewidth=2)
ax.plot(1, 1, 'g*', markersize=15, label='最优点 (1,1)')
ax.plot(start[0], start[1], 'ko', markersize=10, label='起点')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('SGD vs Momentum 在 Rosenbrock 函数上的比较')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print(f"SGD 最终位置: ({sgd_hist[-1, 0]:.4f}, {sgd_hist[-1, 1]:.4f})")
print(f"Momentum 最终位置: ({mom_hist[-1, 0]:.4f}, {mom_hist[-1, 1]:.4f})")
print(f"最优解: (1, 1)")
```

### Adam（自适应学习率）

**Adam**（Adaptive Moment Estimation）结合了 Momentum 和 RMSprop 的优点：

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla L$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla L)^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

Adam 为每个参数维护独立的学习率，自适应调整。

```python runnable
import numpy as np

def adam_optimizer(grad_f, start, learning_rate=0.01, beta1=0.9, beta2=0.999,
                   epsilon=1e-8, num_steps=500):
    """Adam 优化器"""
    theta = np.array(start, dtype=float)
    m = np.zeros_like(theta)  # 一阶矩估计
    v = np.zeros_like(theta)  # 二阶矩估计
    history = [theta.copy()]

    for t in range(1, num_steps + 1):
        grad = grad_f(theta)

        # 更新矩估计
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # 偏差校正
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # 更新参数
        theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(theta.copy())

    return np.array(history)

# 比较 SGD、Momentum、Adam
start = [-1.0, 1.0]
adam_hist = adam_optimizer(grad_f, start, learning_rate=0.01)

print("优化算法比较（Rosenbrock 函数）:")
print("-" * 50)
print(f"起点: {start}")
print(f"最优解: (1, 1)")
print()
print(f"SGD 最终位置: ({sgd_hist[-1, 0]:.4f}, {sgd_hist[-1, 1]:.4f})")
print(f"Momentum 最终位置: ({mom_hist[-1, 0]:.4f}, {mom_hist[-1, 1]:.4f})")
print(f"Adam 最终位置: ({adam_hist[-1, 0]:.4f}, {adam_hist[-1, 1]:.4f})")
```

## 损失函数设计

损失函数的选择影响梯度的特性和优化难度。

### 常见损失函数

| 损失函数 | 公式 | 梯度 | 适用场景 |
|---------|------|------|---------|
| 均方误差 (MSE) | $L = \frac{1}{n}\sum(y - \hat{y})^2$ | $\nabla L = \frac{2}{n}(\hat{y} - y)$ | 回归 |
| 交叉熵 | $L = -\sum y \log(\hat{y})$ | $\nabla L = \hat{y} - y$（配合 softmax） | 分类 |
| Hinge Loss | $L = \max(0, 1 - y \cdot \hat{y})$ | 分段线性 | SVM |

### 梯度特性考量

选择损失函数时需要考虑：

1. **梯度消失**：当预测接近真实值时，梯度是否趋近于零？
2. **梯度爆炸**：是否存在梯度变得过大的情况？
3. **凸性**：损失函数是否凸？凸函数保证全局最优。

## 学习率调优

学习率是最重要的超参数之一。

### 学习率对收敛的影响

```python runnable
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 2

def df(x):
    return 2 * x

def gradient_descent(start, learning_rate, num_steps):
    x = start
    history = [x]
    for _ in range(num_steps):
        x = x - learning_rate * df(x)
        history.append(x)
    return np.array(history)

# 测试不同学习率
learning_rates = [0.01, 0.1, 0.5, 0.9, 1.1]
start = 2.0
num_steps = 20

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

x_range = np.linspace(-3, 3, 100)

for i, lr in enumerate(learning_rates):
    history = gradient_descent(start, lr, num_steps)

    axes[i].plot(x_range, f(x_range), 'b-', alpha=0.5)
    axes[i].plot(history, f(history), 'ro-', markersize=3)

    if lr < 1.0:
        status = "收敛"
    elif lr == 1.0:
        status = "震荡"
    else:
        status = "发散"

    axes[i].set_title(f'η = {lr} ({status})')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('f(x)')
    axes[i].grid(True, alpha=0.3)

# 隐藏最后一个子图
axes[5].axis('off')

plt.suptitle('学习率对梯度下降的影响', fontsize=14)
plt.tight_layout()
plt.show()
plt.close()
```

### 学习率衰减策略

训练过程中逐步减小学习率，可以兼顾收敛速度和精度。

常见策略：
- **步衰减**：每 $k$ 步学习率乘以 $\gamma$
- **指数衰减**：$\eta_t = \eta_0 \cdot \gamma^t$
- **余弦退火**：$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$

```python runnable
import numpy as np
import matplotlib.pyplot as plt

def step_decay(epoch, initial_lr=0.1, drop_rate=0.5, epochs_drop=10):
    """步衰减"""
    return initial_lr * (drop_rate ** (epoch // epochs_drop))

def exponential_decay(epoch, initial_lr=0.1, decay_rate=0.95):
    """指数衰减"""
    return initial_lr * (decay_rate ** epoch)

def cosine_annealing(epoch, initial_lr=0.1, min_lr=0.001, total_epochs=50):
    """余弦退火"""
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(epoch / total_epochs * np.pi))

epochs = np.arange(50)

plt.figure(figsize=(10, 5))
plt.plot(epochs, [step_decay(e) for e in epochs], label='步衰减', linewidth=2)
plt.plot(epochs, [exponential_decay(e) for e in epochs], label='指数衰减', linewidth=2)
plt.plot(epochs, [cosine_annealing(e) for e in epochs], label='余弦退火', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('学习率')
plt.title('学习率衰减策略比较')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()
```

## 本章小结

本章将微积分理论应用于机器学习实践，核心要点如下：

- **梯度下降**是最基础的优化算法，沿负梯度方向更新参数。学习率的选择至关重要：太大导致震荡或发散，太小导致收敛慢。

- **反向传播**利用链式法则高效计算梯度，是神经网络训练的核心算法。它通过缓存中间结果、逐层反向传递，避免了重复计算。

- **优化算法演进**从 SGD 到 Momentum 再到 Adam，不断改进收敛速度和稳定性。Adam 结合了动量和自适应学习率，是目前最流行的优化器。

- **损失函数设计**需要考虑梯度特性。交叉熵损失配合 softmax/sigmoid 的梯度形式简洁，是分类任务的标准选择。

- **学习率调优**是训练的关键。学习率衰减策略可以在训练初期快速收敛，后期精细调整。

微积分不仅是数学理论，更是理解机器学习、诊断训练问题、设计更好模型的工具。掌握这些核心概念，将为深入学习深度学习奠定坚实基础。

## 练习题

1. 实现一个完整的线性回归训练过程，使用梯度下降优化，并可视化损失下降曲线。
    <details>
    <summary>参考答案</summary>

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    # 生成数据
    np.random.seed(42)
    n_samples = 100
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1) * 0.5

    # 初始化参数
    w = np.random.randn(1, 1)
    b = np.random.randn(1, 1)

    # 训练参数
    learning_rate = 0.1
    n_epochs = 100
    losses = []

    # 训练
    for epoch in range(n_epochs):
        # 前向传播
        y_pred = X @ w + b
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)

        # 计算梯度
        dw = 2 * X.T @ (y_pred - y) / n_samples
        db = 2 * np.mean(y_pred - y)

        # 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('损失下降曲线')
    axes[0].grid(True, alpha=0.3)

    # 回归结果
    axes[1].scatter(X, y, alpha=0.5, label='数据点')
    x_line = np.linspace(0, 2, 100)
    y_line = w.item() * x_line + b.item()
    axes[1].plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {w.item():.2f}x + {b.item():.2f}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('y')
    axes[1].set_title('线性回归结果')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"学习到的参数: w = {w.item():.4f}, b = {b.item():.4f}")
    print(f"真实参数: w = 3, b = 4")
    ```
    </details>

2. 解释为什么 Momentum 能够加速收敛并减少震荡。
    <details>
    <summary>参考答案</summary>

    Momentum 通过引入"速度"概念，累积历史梯度信息：

    1. **加速收敛**：
       - 当梯度方向一致时，速度不断积累，相当于加大了有效学习率
       - 沿着下降方向"加速"，更快地到达最小值

    2. **减少震荡**：
       - 当梯度方向变化（如震荡时），正负梯度相互抵消
       - 速度的累积平滑了这些震荡，相当于对梯度进行了"平均"

    3. **物理类比**：
       - 想象一个小球从山上滚下
       - 动量使得小球能够"冲过"小的凹坑，不陷入局部最小值
       - 同时，小球在平坦区域会加速，在陡峭区域会减速

    数学解释：
    - 在一致梯度方向：$v_t \approx \gamma v_{t-1} + \eta g$，累积后有效学习率约为 $\frac{\eta}{1-\gamma}$
    - 在震荡方向：正负梯度相互抵消，速度较小，减少了震荡幅度
    </details>

3. 比较 SGD、Momentum 和 Adam 三种优化器在凸优化问题和非凸优化问题上的优缺点。
    <details>
    <summary>参考答案</summary>

    **SGD（随机梯度下降）**
    - 优点：简单、内存占用小、理论保证强
    - 缺点：收敛慢、对学习率敏感、容易陷入鞍点
    - 凸问题：收敛性好，但可能较慢
    - 非凸问题：容易陷入局部最小值和鞍点

    **Momentum**
    - 优点：加速收敛、减少震荡、有助于跳出浅层局部最小值
    - 缺点：增加了一个超参数（动量系数）、内存占用增加
    - 凸问题：比 SGD 更快收敛
    - 非凸问题：比 SGD 更容易跳出局部最小值

    **Adam**
    - 优点：自适应学习率、对超参数不敏感、收敛快
    - 缺点：内存占用更大、可能不收敛到最优解（有研究表明在某些凸问题上 Adam 的泛化性不如 SGD）
    - 凸问题：收敛快，但最终精度可能不如 SGD
    - 非凸问题：通常是最好的选择，能自适应不同参数的尺度

    **实践建议**：
    - 简单问题或需要精细调优：SGD + 学习率衰减
    - 深度网络、复杂任务：Adam 或 AdamW
    - 追求最佳泛化性：SGD + Momentum + 学习率衰减
    </details>