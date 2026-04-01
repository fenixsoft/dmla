---
title: "进阶概念：多元函数与优化基础"
---

# 进阶概念：多元函数与优化基础

上一章我们学习了单变量函数的导数和微分，建立了变化率的基本概念。然而，机器学习中的大多数问题涉及多个变量——神经网络的参数可能有数百万甚至数十亿个，损失函数是这些参数的多元函数。本章将把导数概念推广到多元函数，介绍偏导数、梯度、链式法则、方向导数和海森矩阵等核心概念，为理解机器学习优化算法奠定理论基础。

## 偏导数

### 多元函数的概念

**多元函数**（Multivariate Function）是单变量函数的自然推广。一个 $n$ 元函数 $f$ 将 $n$ 个输入 $(x_1, x_2, \ldots, x_n)$ 映射到一个输出值 $y$：

$$y = f(x_1, x_2, \ldots, x_n)$$

在机器学习中，多元函数无处不在：
- 损失函数 $L(\theta_1, \theta_2, \ldots, \theta_n)$ 是模型参数的多元函数
- 神经网络中每一层的输出是多个输入的多元函数
- 特征向量 $(x_1, x_2, \ldots, x_n)$ 对应的预测值 $f(x_1, x_2, \ldots, x_n)$

最常见的是二元函数 $z = f(x, y)$，它可以在三维空间中表示为一张曲面。

### 偏导数的定义与计算

当我们面对多元函数时，一个自然的问题是：如果只让其中一个变量变化，而保持其他变量不变，函数值会如何变化？这正是**偏导数**（Partial Derivative）要回答的问题。

**定义**：设 $z = f(x, y)$ 是一个二元函数，$f$ 在点 $(x_0, y_0)$ 处关于 $x$ 的偏导数定义为：

$$\frac{\partial f}{\partial x}\bigg|_{(x_0, y_0)} = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x, y_0) - f(x_0, y_0)}{\Delta x}$$

类似地，关于 $y$ 的偏导数为：

$$\frac{\partial f}{\partial y}\bigg|_{(x_0, y_0)} = \lim_{\Delta y \to 0} \frac{f(x_0, y_0 + \Delta y) - f(x_0, y_0)}{\Delta y}$$

偏导数的记号有多种：$\frac{\partial f}{\partial x}$、$f_x$、$\partial_x f$ 都表示 $f$ 关于 $x$ 的偏导数。

**计算方法**：求 $f(x, y)$ 关于 $x$ 的偏导数时，将 $y$ 视为常数，对 $x$ 求普通导数。这与单变量导数的计算方法完全一致。

**示例**：设 $f(x, y) = x^2 y + 3xy^2$，求 $\frac{\partial f}{\partial x}$ 和 $\frac{\partial f}{\partial y}$。

- 求 $\frac{\partial f}{\partial x}$：视 $y$ 为常数，$\frac{\partial f}{\partial x} = 2xy + 3y^2$
- 求 $\frac{\partial f}{\partial y}$：视 $x$ 为常数，$\frac{\partial f}{\partial y} = x^2 + 6xy$

### 偏导数的几何意义

偏导数有直观的几何解释。对于二元函数 $z = f(x, y)$，其图像是三维空间中的一张曲面。$\frac{\partial f}{\partial x}$ 表示在曲面上沿 $x$ 方向的"切线斜率"，$\frac{\partial f}{\partial y}$ 表示沿 $y$ 方向的"切线斜率"。

具体来说，$\frac{\partial f}{\partial x}(x_0, y_0)$ 是曲面与平面 $y = y_0$ 的交线在点 $(x_0, y_0, f(x_0, y_0))$ 处的切线斜率。这相当于"固定 $y$，只让 $x$ 变化"的切线斜率。

## 梯度

### 梯度的定义

偏导数告诉我们函数沿每个坐标轴方向的变化率。如果把这些信息组合起来，就得到一个向量，称为**梯度**（Gradient）。

**定义**：设 $f(x_1, x_2, \ldots, x_n)$ 是一个多元函数，其梯度定义为：

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

其中 $\nabla$ 称为**梯度算子**（Gradient Operator），读作"nabla"或"del"。

对于二元函数 $f(x, y)$，梯度为：

$$\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)$$

梯度是一个向量，它的每个分量是对应变量的偏导数。

### 梯度的几何意义

梯度有一个极其重要的几何性质：**梯度指向函数值增长最快的方向**。

这一性质可以这样理解：偏导数 $\frac{\partial f}{\partial x_i}$ 告诉我们沿 $x_i$ 轴方向的变化率。如果把这些"方向贡献"组合起来，梯度方向就是所有坐标轴方向的"最优组合"——使函数值增长最快的方向。

相应地，**负梯度方向**就是函数值下降最快的方向。这正是梯度下降算法的核心：沿着负梯度方向移动，可以最快地找到函数的最小值。

### 梯度与优化的关系

在机器学习中，我们的目标通常是**最小化损失函数**。设损失函数为 $L(\theta)$，其中 $\theta = (\theta_1, \theta_2, \ldots, \theta_n)$ 是模型参数。

梯度下降算法的更新规则是：

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

这里：
- $\nabla L(\theta_t)$ 是损失函数在当前参数处的梯度
- $\eta$ 是学习率，控制每一步的步长
- 负号表示沿负梯度方向移动

理解梯度的几何意义，对于理解梯度下降的收敛行为、选择合适的学习率、诊断训练问题等都至关重要。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 定义二元函数 f(x, y) = x^2 + y^2
def f(x, y):
    return x ** 2 + y ** 2

# 计算梯度
def gradient(x, y):
    return np.array([2 * x, 2 * y])

# 创建网格
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 绘制等高线和梯度场
fig, ax = plt.subplots(figsize=(8, 8))

# 等高线
contour = ax.contour(X, Y, Z, levels=20, cmap='coolwarm')
ax.clabel(contour, inline=True, fontsize=8)

# 梯度场（稀疏采样）
x_sparse = np.linspace(-2, 2, 10)
y_sparse = np.linspace(-2, 2, 10)
X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)

# 计算梯度
U = 2 * X_sparse  # ∂f/∂x
V = 2 * Y_sparse  # ∂f/∂y

# 归一化箭头长度
magnitude = np.sqrt(U ** 2 + V ** 2)
U_norm = U / (magnitude + 1e-8)
V_norm = V / (magnitude + 1e-8)

ax.quiver(X_sparse, Y_sparse, U_norm, V_norm, color='black', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('函数 f(x,y) = x² + y² 的等高线与梯度场\n梯度箭头指向函数值增长最快的方向')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.show()
plt.close()

print("观察：梯度箭头从低值区域指向高值区域")
print("梯度下降就是沿着箭头的反方向移动")
```

## 链式法则

### 一元复合函数求导

在上一章中，我们提到了链式法则（Chain Rule），但没有详细展开。链式法则是求导的核心工具，它告诉我们如何对复合函数求导。

**链式法则**：设 $y = f(u)$，$u = g(x)$，则 $y = f(g(x))$ 是 $x$ 的复合函数，其导数为：

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

用函数符号表示：$(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$。

**直观理解**：$y$ 关于 $x$ 的变化率，等于 $y$ 关于中间变量 $u$ 的变化率，乘以 $u$ 关于 $x$ 的变化率。这就像"连锁反应"——$x$ 的变化先影响 $u$，再通过 $u$ 影响 $y$。

**示例**：设 $y = \sin(x^2)$，求 $\frac{dy}{dx}$。

设 $u = x^2$，则 $y = \sin u$。由链式法则：

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = \cos u \cdot 2x = 2x \cos(x^2)$$

### 多元复合函数求导

对于多元函数，链式法则更加丰富。设 $z = f(x, y)$，而 $x = x(t)$，$y = y(t)$，则 $z$ 通过 $x$ 和 $y$ 成为 $t$ 的函数。此时：

$$\frac{dz}{dt} = \frac{\partial f}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial f}{\partial y} \cdot \frac{dy}{dt}$$

这是多元链式法则的一种形式：总变化率等于各路径贡献之和。

更一般地，设 $z = f(x_1, x_2, \ldots, x_n)$，而每个 $x_i$ 都是 $t$ 的函数，则：

$$\frac{dz}{dt} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \cdot \frac{dx_i}{dt}$$

### 反向传播的数学基础

链式法则是神经网络**反向传播**（Backpropagation）算法的数学基础。在神经网络中，损失函数 $L$ 是多层复合函数：

$$L = L(y^{(L)}) = L(f^{(L)}(y^{(L-1)})) = L(f^{(L)}(f^{(L-1)}(\ldots f^{(1)}(x))))$$

其中 $y^{(l)}$ 是第 $l$ 层的输出，$f^{(l)}$ 是第 $l$ 层的变换函数。

计算损失函数关于某层参数的梯度，需要反复应用链式法则。反向传播算法巧妙地利用了这一点：从输出层开始，逐层向前计算梯度，每一层的梯度依赖于上一层的梯度，避免了重复计算。

例如，对于简单的两层网络：
- 前向传播：$y = f(W_2 \cdot f(W_1 x))$
- 计算 $\frac{\partial L}{\partial W_1}$：需要经过 $W_1 \to h \to W_2 \to y \to L$ 这条路径

链式法则告诉我们：

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W_2 \cdot h} \cdot \frac{\partial W_2 \cdot h}{\partial h} \cdot \frac{\partial h}{\partial W_1 x} \cdot \frac{\partial W_1 x}{\partial W_1}$$

这个公式看起来复杂，但反向传播算法通过逐层传递梯度，高效地完成了这个计算。

## 方向导数

### 方向导数的定义

偏导数告诉我们函数沿坐标轴方向的变化率，但如果我们想知道沿任意方向的变化率呢？这就需要**方向导数**（Directional Derivative）。

**定义**：设 $f(x, y)$ 是一个二元函数，$\mathbf{u} = (u_1, u_2)$ 是一个单位向量（$\|\mathbf{u}\| = 1$），则 $f$ 在点 $(x_0, y_0)$ 处沿方向 $\mathbf{u}$ 的方向导数定义为：

$$D_{\mathbf{u}} f(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0 + h u_1, y_0 + h u_2) - f(x_0, y_0)}{h}$$

直观上，方向导数表示从点 $(x_0, y_0)$ 出发，沿方向 $\mathbf{u}$ 移动一小步 $h$ 时，函数值的平均变化率的极限。

### 方向导数与梯度的关系

方向导数有一个重要性质：**方向导数等于梯度与方向向量的点积**。

$$D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \|\mathbf{u}\| \cos\theta = \|\nabla f\| \cos\theta$$

其中 $\theta$ 是梯度 $\nabla f$ 与方向 $\mathbf{u}$ 之间的夹角。

这个公式揭示了梯度的深层含义：
- 当 $\theta = 0$（$\mathbf{u}$ 与 $\nabla f$ 同向）时，$D_{\mathbf{u}} f = \|\nabla f\|$ 最大，即沿梯度方向函数值增长最快
- 当 $\theta = \pi$（$\mathbf{u}$ 与 $\nabla f$ 反向）时，$D_{\mathbf{u}} f = -\|\nabla f\|$ 最小，即沿负梯度方向函数值下降最快
- 当 $\theta = \frac{\pi}{2}$（$\mathbf{u}$ 与 $\nabla f$ 垂直）时，$D_{\mathbf{u}} f = 0$，即沿与梯度垂直的方向，函数值不变（等高线方向）

这一性质为梯度下降算法提供了理论依据：负梯度方向确实是函数值下降最快的方向。

## 海森矩阵

### 二阶偏导数

与单变量函数类似，多元函数也可以有高阶偏导数。对于二元函数 $f(x, y)$，一阶偏导数 $\frac{\partial f}{\partial x}$ 和 $\frac{\partial f}{\partial y}$ 仍然是 $x, y$ 的函数，可以对它们继续求偏导，得到二阶偏导数：

- $\frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial x}\right)$
- $\frac{\partial^2 f}{\partial y^2} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial y}\right)$
- $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial x}\right)$
- $\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial y}\right)$

后两个称为**混合偏导数**。在大多数情况下，混合偏导数与求导顺序无关，即 $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$。

### 海森矩阵的定义

**海森矩阵**（Hessian Matrix）是将二阶偏导数组织成的矩阵。对于 $n$ 元函数 $f(x_1, x_2, \ldots, x_n)$，其海森矩阵为：

$$\mathbf{H} = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

海森矩阵是对称矩阵（在混合偏导数连续的条件下），它包含了函数的二阶导数信息。

### 函数凹凸性判定

海森矩阵在优化中的一个重要应用是**判定函数的凹凸性**：

- 如果海森矩阵在区域内**正定**（所有特征值 > 0），则函数在该区域**严格凸**
- 如果海森矩阵在区域内**负定**（所有特征值 < 0），则函数在该区域**严格凹**
- 如果海森矩阵**不定**（特征值有正有负），则函数在该区域**既非凸也非凹**

**凸函数**在优化中特别重要，因为它保证任何局部最小值都是全局最小值，这使得优化问题更容易求解。

```python runnable
import numpy as np

def check_convexity(f, grad_f, hess_f, x):
    """检查函数在点 x 处的凸性"""
    H = hess_f(x)
    eigenvalues = np.linalg.eigvals(H)

    print(f"海森矩阵：\n{H}")
    print(f"特征值：{eigenvalues}")

    if np.all(eigenvalues > 0):
        print("结论：函数在此点严格凸（海森矩阵正定）")
    elif np.all(eigenvalues < 0):
        print("结论：函数在此点严格凹（海森矩阵负定）")
    else:
        print("结论：函数在此点既非凸也非凹（海森矩阵不定）")

# 示例1：凸函数 f(x,y) = x^2 + y^2
print("=== 示例1：f(x,y) = x² + y² ===")
def f1(x):
    return x[0]**2 + x[1]**2

def hess_f1(x):
    return np.array([[2, 0], [0, 2]])

check_convexity(f1, None, hess_f1, np.array([1, 1]))

print("\n=== 示例2：鞍点函数 f(x,y) = x² - y² ===")
def f2(x):
    return x[0]**2 - x[1]**2

def hess_f2(x):
    return np.array([[2, 0], [0, -2]])

check_convexity(f2, None, hess_f2, np.array([0, 0]))
```

## 积分简介

### 定积分的定义

**定积分**（Definite Integral）是微积分的另一个核心概念。虽然机器学习主要关注微分（优化），但积分在概率论、信息论等领域有重要应用。

**定义**（黎曼积分）：设 $f(x)$ 在区间 $[a, b]$ 上有界，将区间分成 $n$ 个小区间，在每个小区间 $[x_{i-1}, x_i]$ 上任取一点 $\xi_i$，作和式：

$$\sum_{i=1}^{n} f(\xi_i) \Delta x_i$$

当分割无限细密（所有 $\Delta x_i \to 0$）时，如果这个和式趋于一个确定的极限，则称此极限为 $f(x)$ 在 $[a, b]$ 上的定积分，记作：

$$\int_a^b f(x) \, dx$$

### 定积分的几何意义

定积分有直观的几何意义：**定积分表示函数曲线与 $x$ 轴之间的有向面积**。

具体来说：
- 当 $f(x) > 0$ 时，面积取正值
- 当 $f(x) < 0$ 时，面积取负值
- 定积分是这些有向面积的代数和

### 微积分基本定理

微分和积分通过**微积分基本定理**联系起来：

**定理**：设 $f(x)$ 在 $[a, b]$ 上连续，$F(x)$ 是 $f(x)$ 的一个原函数（即 $F'(x) = f(x)$），则：

$$\int_a^b f(x) \, dx = F(b) - F(a)$$

这个定理告诉我们：要计算定积分，只需要找到被积函数的原函数，然后代入端点求值。这大大简化了积分的计算。

### 积分与概率分布的关系

积分在概率论中有着核心地位。对于连续型随机变量 $X$，其**概率密度函数**（Probability Density Function, PDF）$p(x)$ 满足：

$$\int_{-\infty}^{\infty} p(x) \, dx = 1$$

随机变量落在区间 $[a, b]$ 内的概率为：

$$P(a \leq X \leq b) = \int_a^b p(x) \, dx$$

**期望**（Expected Value）定义为：

$$E[X] = \int_{-\infty}^{\infty} x \cdot p(x) \, dx$$

这些积分概念在机器学习中广泛应用，例如在变分推断、生成模型（VAE、扩散模型）等场景中。

## 本章小结

本章将单变量微分学推广到多元情形，核心要点如下：

- **偏导数**是多元函数沿各坐标轴方向的变化率。计算时将其他变量视为常数，方法与单变量求导相同。

- **梯度**是偏导数组成的向量，指向函数值增长最快的方向。负梯度方向是函数值下降最快的方向，这是梯度下降算法的理论基础。

- **链式法则**处理复合函数的求导。对于多元复合函数，总变化率等于各路径贡献之和。链式法则是反向传播算法的数学基础。

- **方向导数**是函数沿任意方向的变化率，等于梯度与方向向量的点积。这进一步验证了梯度方向是函数值变化最快的方向。

- **海森矩阵**组织了二阶偏导数信息，用于判定函数的凹凸性。凸函数的海森矩阵正定，保证局部最小值就是全局最小值。

- **积分**刻画了函数曲线下的面积，在概率论中有重要应用。微积分基本定理将微分和积分联系起来。

这些概念构成了机器学习优化算法的数学基础。下一章将通过 NumPy 和 PyTorch 实践，将这些理论概念转化为可执行的代码。

## 练习题

1. 设 $f(x, y) = x^2 y + y^3$，求 $\frac{\partial f}{\partial x}$、$\frac{\partial f}{\partial y}$ 和 $\nabla f$。
    <details>
    <summary>参考答案</summary>

    求 $\frac{\partial f}{\partial x}$：视 $y$ 为常数，$\frac{\partial f}{\partial x} = 2xy$

    求 $\frac{\partial f}{\partial y}$：视 $x$ 为常数，$\frac{\partial f}{\partial y} = x^2 + 3y^2$

    梯度：$\nabla f = (2xy, x^2 + 3y^2)$

    在点 $(1, 2)$ 处：$\nabla f(1, 2) = (4, 13)$
    </details>

2. 设 $z = x^2 + y^2$，$x = t + 1$，$y = t^2$，用链式法则求 $\frac{dz}{dt}$。
    <details>
    <summary>参考答案</summary>

    方法一（链式法则）：
    $$\frac{dz}{dt} = \frac{\partial z}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial z}{\partial y} \cdot \frac{dy}{dt}$$

    计算：
    - $\frac{\partial z}{\partial x} = 2x$
    - $\frac{\partial z}{\partial y} = 2y$
    - $\frac{dx}{dt} = 1$
    - $\frac{dy}{dt} = 2t$

    因此：$\frac{dz}{dt} = 2x \cdot 1 + 2y \cdot 2t = 2(t+1) + 2t^2 \cdot 2t = 2t + 2 + 4t^3$

    方法二（直接代入验证）：
    $z = (t+1)^2 + t^4 = t^2 + 2t + 1 + t^4$
    $\frac{dz}{dt} = 2t + 2 + 4t^3$

    两种方法结果一致。
    </details>

3. 设 $f(x, y) = x^2 - y^2$，计算 $f$ 在点 $(1, 1)$ 处沿方向 $\mathbf{u} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$ 的方向导数。
    <details>
    <summary>参考答案</summary>

    首先计算梯度：$\nabla f = (2x, -2y)$

    在点 $(1, 1)$ 处：$\nabla f(1, 1) = (2, -2)$

    方向导数：$D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = (2, -2) \cdot (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}) = \frac{2}{\sqrt{2}} - \frac{2}{\sqrt{2}} = 0$

    解释：方向 $\mathbf{u}$ 与梯度垂直，因此沿这个方向函数值不变。这正是等高线的切线方向。
    </details>

4. 判断函数 $f(x, y) = x^2 + 2y^2 + 2xy$ 的凸性。
    <details>
    <summary>参考答案</summary>

    计算一阶偏导数：
    - $\frac{\partial f}{\partial x} = 2x + 2y$
    - $\frac{\partial f}{\partial y} = 4y + 2x$

    计算二阶偏导数：
    - $\frac{\partial^2 f}{\partial x^2} = 2$
    - $\frac{\partial^2 f}{\partial y^2} = 4$
    - $\frac{\partial^2 f}{\partial x \partial y} = 2$
    - $\frac{\partial^2 f}{\partial y \partial x} = 2$

    海森矩阵：$\mathbf{H} = \begin{pmatrix} 2 & 2 \\ 2 & 4 \end{pmatrix}$

    计算特征值：
    $\det(\mathbf{H} - \lambda \mathbf{I}) = \begin{vmatrix} 2-\lambda & 2 \\ 2 & 4-\lambda \end{vmatrix} = (2-\lambda)(4-\lambda) - 4 = \lambda^2 - 6\lambda + 4 = 0$

    解得：$\lambda = 3 \pm \sqrt{5}$，均为正数。

    结论：海森矩阵正定，函数严格凸。
    </details>

5. 设 $f(x) = e^{-x^2}$，计算 $\int_{-\infty}^{\infty} f(x) \, dx$，并解释其在概率论中的意义。
    <details>
    <summary>参考答案</summary>

    这个积分是著名的高斯积分：$\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}$

    概率论意义：
    标准正态分布的概率密度函数为 $\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$

    由于 $\int_{-\infty}^{\infty} \phi(x) \, dx = 1$，概率密度函数下的总面积为 1，这保证了概率的归一化。

    高斯积分在机器学习中广泛出现，例如：
    - 高斯核函数（RBF 核）
    - 变分推断中的 KL 散度计算
    - 高斯分布的参数估计
    </details>

6. 证明：若 $\nabla f(\mathbf{x}^*) = \mathbf{0}$ 且海森矩阵 $\mathbf{H}$ 在 $\mathbf{x}^*$ 处正定，则 $\mathbf{x}^*$ 是 $f$ 的局部最小值点。
    <details>
    <summary>参考答案</summary>

    这是二阶充分条件的核心结论。

    证明思路：
    1. $\nabla f(\mathbf{x}^*) = \mathbf{0}$ 意味着 $\mathbf{x}^*$ 是临界点
    2. 海森矩阵正定意味着在 $\mathbf{x}^*$ 附近，函数可以用二次函数近似：$f(\mathbf{x}^* + \mathbf{h}) \approx f(\mathbf{x}^*) + \frac{1}{2}\mathbf{h}^T \mathbf{H} \mathbf{h}$
    3. 由于 $\mathbf{H}$ 正定，对于任意非零 $\mathbf{h}$，有 $\mathbf{h}^T \mathbf{H} \mathbf{h} > 0$
    4. 因此 $f(\mathbf{x}^* + \mathbf{h}) > f(\mathbf{x}^*)$ 对足够小的 $\mathbf{h}$ 成立
    5. 这说明 $\mathbf{x}^*$ 是局部最小值点

    这个结论在优化算法中有重要应用：当我们找到梯度为零的点后，检查海森矩阵的正定性可以判断这是最小值点还是最大值点或鞍点。
    </details>