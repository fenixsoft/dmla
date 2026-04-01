---
title: "NumPy 实践：微积分计算"
---

# NumPy 实践：微积分计算

前两章建立了微积分的理论基础，理解了导数、梯度、链式法则等核心概念。本章将这些理论转化为可执行的代码，通过 NumPy 实现数值微分、梯度计算，并介绍 PyTorch 的自动微分机制。动手实践不仅能加深对概念的理解，更能培养解决实际问题的能力。

## 数值微分

数值微分是用数值方法近似计算导数的技术。虽然解析求导给出了精确的导数公式，但在很多情况下，我们只能获得函数值，无法获得解析表达式——这时就需要数值微分。

### 前向差分

最简单的数值微分方法是**前向差分**（Forward Difference）。根据导数的定义：

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

如果我们取一个很小的 $h$（但不能为零），就可以用差商近似导数：

$$f'(x) \approx \frac{f(x + h) - f(x)}{h}$$

这就是前向差分公式。

```python runnable
import numpy as np

def forward_difference(f, x, h=1e-5):
    """
    前向差分法计算数值导数

    参数:
        f: 待求导函数
        x: 求导点
        h: 步长（默认 1e-5）

    返回:
        导数的近似值
    """
    return (f(x + h) - f(x)) / h

# 测试：计算 f(x) = x^2 在 x=2 处的导数
f = lambda x: x ** 2
x = 2

# 解析导数：f'(x) = 2x，在 x=2 处为 4
analytical = 2 * x

# 数值导数
numerical = forward_difference(f, x)

print(f"函数: f(x) = x²")
print(f"求导点: x = {x}")
print(f"解析导数: {analytical}")
print(f"数值导数（前向差分）: {numerical:.6f}")
print(f"绝对误差: {abs(numerical - analytical):.2e}")
```

### 中心差分

前向差分虽然简单，但精度有限。更精确的方法是**中心差分**（Central Difference）：

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

中心差分的精度比前向差分高一阶，误差更小。

```python runnable
import numpy as np

def central_difference(f, x, h=1e-5):
    """
    中心差分法计算数值导数

    参数:
        f: 待求导函数
        x: 求导点
        h: 步长（默认 1e-5）

    返回:
        导数的近似值
    """
    return (f(x + h) - f(x - h)) / (2 * h)

# 比较前向差分和中心差分的精度
f = lambda x: np.sin(x)
x = np.pi / 4  # 45度

# 解析导数：f'(x) = cos(x)
analytical = np.cos(x)

# 数值导数
forward = forward_difference(f, x)
central = central_difference(f, x)

print(f"函数: f(x) = sin(x)")
print(f"求导点: x = π/4 ≈ {x:.4f}")
print(f"解析导数: {analytical:.6f}")
print(f"前向差分: {forward:.6f}, 误差: {abs(forward - analytical):.2e}")
print(f"中心差分: {central:.6f}, 误差: {abs(central - analytical):.2e}")
print(f"\n中心差分误差约为前向差分的 {abs(forward - analytical) / abs(central - analytical):.1f} 分之一")
```

### 数值微分的误差分析

数值微分的误差有两个来源：

1. **截断误差**：用差商代替极限带来的误差。前向差分的截断误差为 $O(h)$，中心差分为 $O(h^2)$。

2. **舍入误差**：计算机浮点数运算带来的误差。当 $h$ 太小时，$f(x+h)$ 和 $f(x)$ 的差可能被舍入误差淹没。

因此，$h$ 不能太大（截断误差大），也不能太小（舍入误差大）。通常取 $h \approx 10^{-5}$ 到 $10^{-8}$。

```python runnable
import numpy as np

# 分析步长对误差的影响
f = lambda x: np.exp(x)
x = 1.0
analytical = np.exp(x)  # 解析导数

h_values = np.logspace(-16, -1, 16)  # 从 10^-16 到 10^-1
forward_errors = []
central_errors = []

for h in h_values:
    forward = (f(x + h) - f(x)) / h
    central = (f(x + h) - f(x - h)) / (2 * h)
    forward_errors.append(abs(forward - analytical))
    central_errors.append(abs(central - analytical))

# 找到最优步长
best_h_forward = h_values[np.argmin(forward_errors)]
best_h_central = h_values[np.argmin(central_errors)]

print(f"前向差分最优步长: h = {best_h_forward:.2e}, 最小误差: {min(forward_errors):.2e}")
print(f"中心差分最优步长: h = {best_h_central:.2e}, 最小误差: {min(central_errors):.2e}")
```

## 梯度的手动计算

对于多元函数，我们需要计算**梯度**——各个偏导数组成的向量。数值方法同样可以用来计算梯度。

### 梯度的数值计算

对于 $n$ 元函数 $f(x_1, x_2, \ldots, x_n)$，梯度的第 $i$ 个分量为：

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h \mathbf{e}_i) - f(x - h \mathbf{e}_i)}{2h}$$

其中 $\mathbf{e}_i$ 是第 $i$ 个标准基向量（第 $i$ 个分量为 1，其余为 0）。

```python runnable
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    """
    计算多元函数的梯度（中心差分法）

    参数:
        f: 多元函数，接受 numpy 数组作为输入
        x: 求导点（numpy 数组）
        h: 步长

    返回:
        梯度向量（numpy 数组）
    """
    grad = np.zeros_like(x, dtype=float)
    n = len(x)

    for i in range(n):
        # 创建单位向量 e_i
        e_i = np.zeros(n)
        e_i[i] = 1

        # 中心差分计算偏导数
        grad[i] = (f(x + h * e_i) - f(x - h * e_i)) / (2 * h)

    return grad

# 测试：计算 f(x,y) = x² + y² 在 (3, 4) 处的梯度
def f(xy):
    x, y = xy
    return x ** 2 + y ** 2

x = np.array([3.0, 4.0])

# 解析梯度：∇f = (2x, 2y) = (6, 8)
analytical_grad = np.array([2 * x[0], 2 * x[1]])

# 数值梯度
numerical_grad = numerical_gradient(f, x)

print(f"函数: f(x, y) = x² + y²")
print(f"求导点: ({x[0]}, {x[1]})")
print(f"解析梯度: {analytical_grad}")
print(f"数值梯度: {numerical_grad}")
print(f"误差: {np.linalg.norm(numerical_grad - analytical_grad):.2e}")
```

### 验证梯度计算的正确性

在机器学习中，我们经常需要验证梯度计算是否正确。一个简单的方法是：对于任意方向 $\mathbf{v}$，应该有：

$$\nabla f(\mathbf{x}) \cdot \mathbf{v} \approx \frac{f(\mathbf{x} + h\mathbf{v}) - f(\mathbf{x} - h\mathbf{v})}{2h}$$

这个等式可以用来验证梯度计算的正确性。

```python runnable
import numpy as np

def check_gradient(f, grad_f, x, h=1e-5, num_tests=5):
    """
    验证梯度计算的正确性

    参数:
        f: 原函数
        grad_f: 计算梯度的函数
        x: 测试点
        h: 步长
        num_tests: 测试次数

    返回:
        是否通过验证
    """
    n = len(x)
    np.random.seed(42)

    print("梯度验证测试:")
    print("-" * 60)

    for i in range(num_tests):
        # 随机生成一个方向向量
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)  # 归一化

        # 计算方向导数（两种方法）
        numerical_directional = (f(x + h * v) - f(x - h * v)) / (2 * h)
        analytical_directional = np.dot(grad_f(x), v)

        error = abs(numerical_directional - analytical_directional)
        passed = error < 1e-6

        print(f"测试 {i+1}: 数值 = {numerical_directional:.6f}, 解析 = {analytical_directional:.6f}, 误差 = {error:.2e}, {'✓' if passed else '✗'}")

# 测试函数：f(x,y) = x²y + xy²
def test_f(xy):
    x, y = xy
    return x ** 2 * y + x * y ** 2

# 解析梯度：∂f/∂x = 2xy + y², ∂f/∂y = x² + 2xy
def test_grad(xy):
    x, y = xy
    return np.array([2 * x * y + y ** 2, x ** 2 + 2 * x * y])

# 验证
x = np.array([1.0, 2.0])
check_gradient(test_f, test_grad, x)
```

## 链式法则的代码实现

### 一元复合函数

链式法则可以用代码实现。对于复合函数 $y = f(g(x))$，导数为：

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

```python runnable
import numpy as np

def chain_rule_1d(outer_f, outer_df, inner_g, inner_dg, x):
    """
    使用链式法则计算复合函数的导数

    参数:
        outer_f: 外层函数 f
        outer_df: 外层函数的导数 f'
        inner_g: 内层函数 g
        inner_dg: 内层函数的导数 g'
        x: 输入值

    返回:
        复合函数值和导数
    """
    # 前向传播
    u = inner_g(x)  # 内层函数值
    y = outer_f(u)  # 外层函数值

    # 反向传播（链式法则）
    dy_du = outer_df(u)  # 外层函数对中间变量的导数
    du_dx = inner_dg(x)  # 内层函数对输入的导数
    dy_dx = dy_du * du_dx  # 链式法则

    return y, dy_dx

# 示例：y = sin(x²)
outer_f = np.sin
outer_df = np.cos
inner_g = lambda x: x ** 2
inner_dg = lambda x: 2 * x

x = 1.5
y, dy_dx = chain_rule_1d(outer_f, outer_df, inner_g, inner_dg, x)

# 验证
numerical = central_difference(lambda x: np.sin(x ** 2), x)

print(f"复合函数: y = sin(x²)")
print(f"x = {x}")
print(f"函数值: {y:.6f}")
print(f"解析导数（链式法则）: {dy_dx:.6f}")
print(f"数值导数: {numerical:.6f}")
print(f"误差: {abs(dy_dx - numerical):.2e}")
```

### 多层复合函数

对于多层复合函数，链式法则需要逐层应用。这正是反向传播算法的思想。

```python runnable
import numpy as np

class FunctionNode:
    """函数节点，用于构建计算图"""

    def __init__(self, name, forward, backward):
        """
        参数:
            name: 节点名称
            forward: 前向传播函数
            backward: 反向传播函数（输入上游梯度，输出本节点梯度）
        """
        self.name = name
        self.forward = forward
        self.backward = backward
        self.input_cache = None
        self.output_cache = None

    def __call__(self, x):
        """前向传播"""
        self.input_cache = x
        self.output_cache = self.forward(x)
        return self.output_cache

    def backward_pass(self, upstream_grad):
        """反向传播"""
        return self.backward(self.input_cache, upstream_grad)

# 定义一些基本函数节点
def square_node():
    """f(x) = x²"""
    return FunctionNode(
        "square",
        forward=lambda x: x ** 2,
        backward=lambda x, grad: grad * 2 * x
    )

def exp_node():
    """f(x) = e^x"""
    return FunctionNode(
        "exp",
        forward=lambda x: np.exp(x),
        backward=lambda x, grad: grad * np.exp(x)
    )

def sin_node():
    """f(x) = sin(x)"""
    return FunctionNode(
        "sin",
        forward=lambda x: np.sin(x),
        backward=lambda x, grad: grad * np.cos(x)
    )

# 构建 y = sin(e^(x²)) 的计算图
x = 0.5

# 前向传播
node1 = square_node()
node2 = exp_node()
node3 = sin_node()

a = node1(x)    # a = x²
b = node2(a)    # b = e^(x²)
y = node3(b)    # y = sin(e^(x²))

print("=== 前向传播 ===")
print(f"x = {x}")
print(f"a = x² = {a:.6f}")
print(f"b = e^a = {b:.6f}")
print(f"y = sin(b) = {y:.6f}")

# 反向传播
print("\n=== 反向传播 ===")
grad_y = 1.0  # 初始梯度为 1
grad_b = node3.backward_pass(grad_y)
print(f"∂y/∂b = {grad_b:.6f}")

grad_a = node2.backward_pass(grad_b)
print(f"∂y/∂a = {grad_a:.6f}")

grad_x = node1.backward_pass(grad_a)
print(f"∂y/∂x = {grad_x:.6f}")

# 数值验证
numerical = central_difference(lambda x: np.sin(np.exp(x ** 2)), x)
print(f"\n数值导数: {numerical:.6f}")
print(f"误差: {abs(grad_x - numerical):.2e}")
```

## PyTorch 自动微分简介

手动实现链式法则虽然有助于理解原理，但在实际应用中非常繁琐。PyTorch 提供了**自动微分**（Automatic Differentiation）功能，可以自动计算梯度。

### 计算图的概念

PyTorch 使用**动态计算图**（Dynamic Computational Graph）来记录运算过程。当我们对张量进行运算时，PyTorch 会自动构建计算图。反向传播时，沿着计算图反向遍历，自动应用链式法则计算梯度。

### autograd 的基本使用

```python runnable
# 注意：此代码需要 PyTorch 环境
# 如果运行报错，请确保已安装 PyTorch

try:
    import torch

    # 创建需要跟踪梯度的张量
    x = torch.tensor([0.5], requires_grad=True)

    # 定义计算过程（自动构建计算图）
    a = x ** 2
    b = torch.exp(a)
    y = torch.sin(b)

    print("=== PyTorch 自动微分 ===")
    print(f"x = {x.item():.6f}")
    print(f"a = x² = {a.item():.6f}")
    print(f"b = e^a = {b.item():.6f}")
    print(f"y = sin(b) = {y.item():.6f}")

    # 反向传播
    y.backward()

    print(f"\n自动计算的梯度 dy/dx = {x.grad.item():.6f}")

    # 与 NumPy 数值导数对比
    import numpy as np
    numerical = central_difference(lambda x: np.sin(np.exp(x ** 2)), 0.5)
    print(f"数值导数 = {numerical:.6f}")
    print(f"误差 = {abs(x.grad.item() - numerical):.2e}")

except ImportError:
    print("PyTorch 未安装。请运行以下命令安装：")
    print("pip install torch")
```

### 反向传播的自动实现

PyTorch 的自动微分不仅支持单变量函数，还支持多元函数的梯度计算。

```python runnable
try:
    import torch

    # 定义一个二元函数：f(x,y) = x² + 2xy + y²
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)

    # 前向传播
    f = x ** 2 + 2 * x * y + y ** 2

    print("函数: f(x,y) = x² + 2xy + y²")
    print(f"求导点: (x={x.item()}, y={y.item()})")
    print(f"函数值: {f.item():.6f}")

    # 反向传播
    f.backward()

    # 获取梯度
    print(f"\n偏导数 ∂f/∂x = {x.grad.item():.6f} (解析值: {2*1 + 2*2:.6f})")
    print(f"偏导数 ∂f/∂y = {y.grad.item():.6f} (解析值: {2*1 + 2*2:.6f})")

    # 解析梯度：∂f/∂x = 2x + 2y, ∂f/∂y = 2x + 2y
    analytical_grad_x = 2 * 1 + 2 * 2
    analytical_grad_y = 2 * 1 + 2 * 2

    print(f"\n梯度验证:")
    print(f"  ∂f/∂x 误差: {abs(x.grad.item() - analytical_grad_x):.2e}")
    print(f"  ∂f/∂y 误差: {abs(y.grad.item() - analytical_grad_y):.2e}")

except ImportError:
    print("PyTorch 未安装，跳过此示例")
```

### 使用 PyTorch 进行简单优化

```python runnable
try:
    import torch
    import matplotlib.pyplot as plt

    # 用梯度下降求 f(x) = (x-3)² 的最小值
    # 解析解：x = 3

    x = torch.tensor([0.0], requires_grad=True)  # 初始值
    learning_rate = 0.1
    num_steps = 20

    x_history = [x.item()]

    print("梯度下降优化: f(x) = (x-3)²")
    print("-" * 40)

    for step in range(num_steps):
        # 计算函数值
        f = (x - 3) ** 2

        # 反向传播计算梯度
        f.backward()

        # 更新参数（梯度下降）
        with torch.no_grad():
            x -= learning_rate * x.grad

        # 清除梯度
        x.grad.zero_()

        x_history.append(x.item())

        if step % 5 == 0:
            print(f"Step {step:2d}: x = {x.item():.6f}, f = {f.item():.6f}")

    print(f"\n最终结果: x = {x.item():.6f}")
    print(f"理论最优解: x = 3")
    print(f"误差: {abs(x.item() - 3):.2e}")

except ImportError:
    print("PyTorch 未安装，跳过此示例")
```

## 数值积分

积分是微积分的另一个核心概念。虽然机器学习主要关注微分，但数值积分在概率计算、信号处理等领域有重要应用。

### 梯形法则

**梯形法则**（Trapezoidal Rule）是最简单的数值积分方法。将积分区间分成若干小区间，用梯形面积近似曲线下面积：

$$\int_a^b f(x) \, dx \approx \frac{h}{2} \left[ f(a) + 2\sum_{i=1}^{n-1} f(a + ih) + f(b) \right]$$

其中 $h = \frac{b-a}{n}$ 是步长。

```python runnable
import numpy as np

def trapezoidal(f, a, b, n=1000):
    """
    梯形法则计算定积分

    参数:
        f: 被积函数
        a, b: 积分区间
        n: 分割数

    返回:
        积分近似值
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

# 计算积分 ∫₀^π sin(x) dx = 2
result = trapezoidal(np.sin, 0, np.pi)
print(f"∫₀^π sin(x) dx")
print(f"梯形法则结果: {result:.6f}")
print(f"精确值: 2.0")
print(f"误差: {abs(result - 2.0):.2e}")
```

### 辛普森法则

**辛普森法则**（Simpson's Rule）比梯形法则更精确，用二次曲线近似被积函数：

$$\int_a^b f(x) \, dx \approx \frac{h}{3} \left[ f(a) + 4\sum_{\text{odd }i} f(x_i) + 2\sum_{\text{even }i} f(x_i) + f(b) \right]$$

```python runnable
import numpy as np

def simpson(f, a, b, n=1000):
    """
    辛普森法则计算定积分

    参数:
        f: 被积函数
        a, b: 积分区间
        n: 分割数（必须为偶数）

    返回:
        积分近似值
    """
    if n % 2 != 0:
        n += 1  # 确保分割数为偶数

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # 辛普森公式
    result = y[0] + y[-1]
    result += 4 * np.sum(y[1:-1:2])   # 奇数索引
    result += 2 * np.sum(y[2:-1:2])   # 偶数索引
    result *= h / 3

    return result

# 比较梯形法则和辛普森法则
print("计算 ∫₀¹ e^(-x²) dx（高斯积分的一部分）")
print("-" * 50)

# 由于 e^(-x²) 没有初等原函数，我们用高精度数值积分作为参考
# 使用 scipy 的 quad 函数（如果可用）
try:
    from scipy.integrate import quad
    reference, _ = quad(lambda x: np.exp(-x**2), 0, 1)
except ImportError:
    # 如果 scipy 不可用，使用大 n 的辛普森法则作为参考
    reference = simpson(lambda x: np.exp(-x**2), 0, 1, n=10000)

for n in [10, 100, 1000]:
    trap = trapezoidal(lambda x: np.exp(-x**2), 0, 1, n)
    simp = simpson(lambda x: np.exp(-x**2), 0, 1, n)

    print(f"n = {n:4d}:")
    print(f"  梯形法则: {trap:.6f}, 误差: {abs(trap - reference):.2e}")
    print(f"  辛普森法: {simp:.6f}, 误差: {abs(simp - reference):.2e}")

print(f"\n参考值: {reference:.6f}")
```

### 积分在概率计算中的应用

数值积分可以用来计算连续型随机变量的概率。

```python runnable
import numpy as np

# 标准正态分布的概率密度函数
def normal_pdf(x, mu=0, sigma=1):
    """正态分布概率密度函数"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# 计算 P(-1 < X < 1) 对于标准正态分布
# 理论值约为 0.6827（68.27%）
prob = simpson(normal_pdf, -1, 1, n=1000)

print("标准正态分布 N(0,1)：")
print(f"P(-1 < X < 1) ≈ {prob:.4f}")
print(f"理论值 ≈ 0.6827")
print(f"误差: {abs(prob - 0.6827):.2e}")
print(f"\n这意味着约 68% 的数据落在均值的一个标准差范围内。")
```

## 本章小结

本章将微积分理论转化为可执行的代码，核心要点如下：

- **数值微分**用差商近似导数。前向差分简单但精度有限，中心差分精度更高。步长的选择需要在截断误差和舍入误差之间平衡。

- **梯度计算**可以用数值方法实现，通过逐个方向计算偏导数。梯度验证是检验梯度计算正确性的重要技巧。

- **链式法则**可以通过代码实现，逐层计算梯度。这正是反向传播算法的数学基础。

- **PyTorch 自动微分**简化了梯度计算。定义前向传播后，PyTorch 自动构建计算图并计算梯度，极大地提高了开发效率。

- **数值积分**用离散求和近似连续积分。梯形法则简单直观，辛普森法则精度更高。数值积分在概率计算中有重要应用。

下一章将把这些技术应用于机器学习场景，详细介绍梯度下降算法、反向传播机制和优化算法的演进。

## 练习题

1. 实现一个函数，使用中心差分法计算函数 $f(x) = x^3$ 在区间 $[-2, 2]$ 上等间隔取 100 个点处的导数，并绘制函数和导数的图像。
    <details>
    <summary>参考答案</summary>

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def central_difference(f, x, h=1e-5):
        return (f(x + h) - f(x - h)) / (2 * h)

    # 定义函数
    f = lambda x: x ** 3
    df_analytical = lambda x: 3 * x ** 2  # 解析导数

    # 计算点
    x = np.linspace(-2, 2, 100)
    y = f(x)
    dy_numerical = central_difference(f, x)
    dy_analytical = df_analytical(x)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x, y, 'b-', label='f(x) = x³')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('原函数')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(x, dy_numerical, 'r--', label='数值导数')
    axes[1].plot(x, dy_analytical, 'g-', label='解析导数')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel("f'(x)")
    axes[1].set_title('导数对比')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # 计算误差
    error = np.max(np.abs(dy_numerical - dy_analytical))
    print(f"最大误差: {error:.2e}")
    ```
    </details>

2. 编写一个通用的梯度检查函数，可以验证任意多元函数的梯度计算是否正确。
    <details>
    <summary>参考答案</summary>

    ```python
    import numpy as np

    def gradient_check(f, grad_f, x, h=1e-5, tol=1e-6):
        """
        验证多元函数梯度计算的正确性

        参数:
            f: 原函数
            grad_f: 梯度函数
            x: 测试点（numpy 数组）
            h: 数值微分的步长
            tol: 容差

        返回:
            (是否通过, 最大相对误差)
        """
        n = len(x)
        numerical_grad = np.zeros(n)

        # 计算数值梯度
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1
            numerical_grad[i] = (f(x + h * e_i) - f(x - h * e_i)) / (2 * h)

        # 计算解析梯度
        analytical_grad = grad_f(x)

        # 计算相对误差
        diff = np.abs(numerical_grad - analytical_grad)
        norm = np.maximum(np.abs(numerical_grad), np.abs(analytical_grad))
        relative_error = diff / (norm + 1e-10)  # 避免除零

        max_error = np.max(relative_error)
        passed = max_error < tol

        return passed, max_error

    # 测试
    def f(xy):
        x, y = xy
        return x ** 2 * y + x * y ** 2

    def grad_f(xy):
        x, y = xy
        return np.array([2 * x * y + y ** 2, x ** 2 + 2 * x * y])

    x = np.array([1.5, 2.0])
    passed, error = gradient_check(f, grad_f, x)
    print(f"梯度检查: {'通过' if passed else '失败'}")
    print(f"最大相对误差: {error:.2e}")
    ```
    </details>

3. 使用 PyTorch 实现一个简单的线性回归模型，并用梯度下降训练。
    <details>
    <summary>参考答案</summary>

    ```python
    import torch
    import numpy as np

    # 生成数据
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X + 3 + np.random.randn(100, 1) * 0.5

    # 转换为 PyTorch 张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # 初始化参数
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    # 训练参数
    learning_rate = 0.01
    num_epochs = 1000

    # 训练循环
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = X_tensor * w + b
        loss = torch.mean((y_pred - y_tensor) ** 2)

        # 反向传播
        loss.backward()

        # 更新参数
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # 清除梯度
        w.grad.zero_()
        b.grad.zero_()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    print(f"\n训练结果:")
    print(f"w = {w.item():.4f} (真实值: 2)")
    print(f"b = {b.item():.4f} (真实值: 3)")
    ```
    </details>

4. 使用辛普森法则计算 $\int_0^1 \sqrt{1-x^2} \, dx$，这个积分的值是多少？有什么几何意义？
    <details>
    <summary>参考答案</summary>

    ```python
    import numpy as np

    def simpson(f, a, b, n=1000):
        if n % 2 != 0:
            n += 1
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        result = y[0] + y[-1]
        result += 4 * np.sum(y[1:-1:2])
        result += 2 * np.sum(y[2:-1:2])
        result *= h / 3
        return result

    # 计算 ∫₀¹ √(1-x²) dx
    f = lambda x: np.sqrt(1 - x ** 2)
    result = simpson(f, 0, 1, n=1000)

    print(f"∫₀¹ √(1-x²) dx ≈ {result:.6f}")
    print(f"π/4 ≈ {np.pi/4:.6f}")
    print(f"误差: {abs(result - np.pi/4):.2e}")

    print("\n几何意义：")
    print("y = √(1-x²) 是单位圆的上半部分（x ∈ [0,1]）")
    print("积分值等于四分之一圆的面积，即 π/4")
    ```
    </details>