---
title: "基础概念：极限、导数与微分"
---

# 基础概念：极限、导数与微分

导数是微积分最核心的概念，它刻画了函数在某一点的"变化率"。理解导数，不仅需要掌握其代数定义，更需要建立几何直观。本章将从极限概念出发，逐步引出导数的定义、几何意义和计算方法，为后续学习多元函数微分学奠定基础。

## 极限与连续

在介绍导数之前，我们需要先理解**极限（Limit）**的概念。极限是微积分的理论基石，它描述了当自变量无限接近某个值时，函数值的变化趋势。

### 极限的直观理解

考虑一个简单的函数 $f(x) = \frac{x^2 - 1}{x - 1}$。当 $x = 1$ 时，分母为零，函数值不存在。但是，如果我们观察 $x$ 接近 1 时的函数值：

| $x$ | $f(x)$ |
|-----|--------|
| 0.9 | 1.9 |
| 0.99 | 1.99 |
| 0.999 | 1.999 |
| 1.001 | 2.001 |
| 1.01 | 2.01 |
| 1.1 | 2.1 |

可以看到，当 $x$ 从两侧趋近于 1 时，$f(x)$ 趋近于 2。这就是极限的直观含义：**当 $x$ 无限接近某个值 $a$ 时，函数值 $f(x)$ 无限接近某个值 $L$**，记作：

$$\lim_{x \to a} f(x) = L$$

注意，极限关注的是 $x$ "趋近于" $a$ 的过程，而不是 $x$ 等于 $a$ 时的函数值。在上面的例子中，$f(1)$ 甚至不存在，但 $\lim_{x \to 1} f(x) = 2$ 是完全确定的。

### 极限的严格定义（$\varepsilon-\delta$ 语言）

直观理解虽然有助于建立概念，但数学需要严格的定义。19 世纪，德国数学家卡尔·魏尔斯特拉斯（Karl Weierstrass）给出了极限的严格定义：

**定义**：设函数 $f$ 在点 $a$ 的某个去心邻域内有定义。如果存在常数 $L$，对于任意给定的正数 $\varepsilon$（无论多小），都存在正数 $\delta$，使得当 $0 < |x - a| < \delta$ 时，有 $|f(x) - L| < \varepsilon$，则称 $L$ 是函数 $f(x)$ 当 $x \to a$ 时的极限。

这个定义用两个不等式精确地刻画了"趋近"：
- $|x - a| < \delta$ 表示 $x$ 与 $a$ 的距离小于 $\delta$（$x$ 足够接近 $a$）
- $|f(x) - L| < \varepsilon$ 表示 $f(x)$ 与 $L$ 的距离小于 $\varepsilon$（$f(x)$ 足够接近 $L$）

定义的核心意思是：你想让 $f(x)$ 多接近 $L$（给定 $\varepsilon$），我就能找到 $x$ 足够接近 $a$ 的范围（确定 $\delta$），使得在这个范围内 $f(x)$ 达到你要求的接近程度。

对于非数学专业的读者，理解 $\varepsilon-\delta$ 语言的逻辑结构比记忆具体证明更重要。在实际应用中，我们主要依赖极限的直观理解和运算法则。

### 极限的运算法则

设 $\lim_{x \to a} f(x) = A$，$\lim_{x \to a} g(x) = B$，则：

| 运算 | 公式 |
|------|------|
| 加法 | $\lim_{x \to a} [f(x) + g(x)] = A + B$ |
| 减法 | $\lim_{x \to a} [f(x) - g(x)] = A - B$ |
| 乘法 | $\lim_{x \to a} [f(x) \cdot g(x)] = A \cdot B$ |
| 除法 | $\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{A}{B}$（当 $B \neq 0$） |

这些法则告诉我们：极限运算可以"穿透"加减乘除，分别对各部分求极限后再进行相应运算。

### 连续性

**连续（Continuous）** 是描述函数"没有断裂"的数学概念。直观上，连续函数的图像可以一笔画出，不需要抬笔。严格定义如下：

**定义**：函数 $f$ 在点 $a$ 处连续，当且仅当满足以下三个条件：
1. $f(a)$ 有定义
2. $\lim_{x \to a} f(x)$ 存在
3. $\lim_{x \to a} f(x) = f(a)$

第三个条件将极限值与函数值统一起来——"极限等于函数值"正是连续的核心含义。

连续函数有许多良好的性质。例如，**介值定理**（Intermediate Value Theorem）告诉我们：如果连续函数 $f$ 在区间 $[a, b]$ 上取值 $f(a)$ 和 $f(b)$，那么对于 $f(a)$ 和 $f(b)$ 之间的任何值 $c$，存在 $x \in (a, b)$ 使得 $f(x) = c$。这个定理在数值计算中用于求方程的根（如二分法）。

## 导数的定义

有了极限的概念，我们就可以定义导数了。

### 从平均变化率到瞬时变化率

考虑一个物体沿直线运动，其位置 $s$ 是时间 $t$ 的函数 $s = s(t)$。在时间段 $[t_0, t_0 + \Delta t]$ 内，物体移动的距离为 $s(t_0 + \Delta t) - s(t_0)$，**平均速度**为：

$$\bar{v} = \frac{s(t_0 + \Delta t) - s(t_0)}{\Delta t}$$

这就是**平均变化率**的概念：函数值的变化量除以自变量的变化量。

但是，我们更关心的是物体在某一时刻 $t_0$ 的**瞬时速度**。如何定义瞬时速度？直觉告诉我们，让时间间隔 $\Delta t$ 越来越小，平均速度就会越来越接近瞬时速度。当 $\Delta t$ 趋近于零时，平均速度的极限就是瞬时速度：

$$v(t_0) = \lim_{\Delta t \to 0} \frac{s(t_0 + \Delta t) - s(t_0)}{\Delta t}$$

这正是导数的思想。

### 导数的严格定义

**定义**：设函数 $y = f(x)$ 在点 $x_0$ 的某个邻域内有定义，如果极限

$$\lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$$

存在，则称函数 $f$ 在点 $x_0$ 处**可导**（Differentiable），此极限值称为 $f$ 在 $x_0$ 处的**导数**（Derivative），记作 $f'(x_0)$ 或 $\frac{df}{dx}\bigg|_{x=x_0}$。

这个定义中的分式 $\frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$ 称为**差商**（Difference Quotient），它表示函数在区间 $[x_0, x_0 + \Delta x]$ 上的平均变化率。导数就是差商当 $\Delta x \to 0$ 时的极限，即**瞬时变化率**。

导数的另一种等价定义形式是：

$$f'(x_0) = \lim_{x \to x_0} \frac{f(x) - f(x_0)}{x - x_0}$$

这两种定义是等价的，只需令 $x = x_0 + \Delta x$ 即可相互转换。

### 导数的几何意义：切线斜率

导数有非常直观的几何意义。考虑函数 $y = f(x)$ 的图像，在点 $(x_0, f(x_0))$ 处画一条**切线**（Tangent Line）。这条切线的斜率就是 $f'(x_0)$。

为什么？考虑通过点 $(x_0, f(x_0))$ 和 $(x_0 + \Delta x, f(x_0 + \Delta x))$ 的**割线**（Secant Line）。割线的斜率为：

$$\text{割线斜率} = \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$$

这正是差商！当 $\Delta x \to 0$ 时，点 $(x_0 + \Delta x, f(x_0 + \Delta x))$ 沿曲线趋近于 $(x_0, f(x_0))$，割线逐渐逼近切线。因此，导数 $f'(x_0)$ 就是切线的斜率。

![函数的切线与割线](./assets/tangent_line.png)

*图：割线逐渐逼近切线的过程*

用代码可以直观地展示这个过程：

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x):
    return x ** 2

# 定义点
x0 = 1
y0 = f(x0)

# 创建图形
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# 不同的 delta_x 值
deltas = [1.0, 0.5, 0.1]

for i, dx in enumerate(deltas):
    x = np.linspace(-0.5, 2.5, 100)
    axes[i].plot(x, f(x), 'b-', label='y = x²')

    # 绘制割线
    x1 = x0 + dx
    y1 = f(x1)
    slope = (y1 - y0) / dx
    line_x = np.linspace(-0.5, 2.5, 100)
    line_y = y0 + slope * (line_x - x0)
    axes[i].plot(line_x, line_y, 'r--', label=f'割线 (斜率={slope:.2f})')

    # 标记点
    axes[i].plot([x0, x1], [y0, y1], 'go', markersize=8)
    axes[i].set_xlim(-0.5, 2.5)
    axes[i].set_ylim(-0.5, 5)
    axes[i].set_title(f'Δx = {dx}')
    axes[i].legend(loc='upper left')
    axes[i].grid(True)

plt.tight_layout()
plt.show()
plt.close()

# 计算精确导数
print(f"当 Δx 趋近于 0 时，割线斜率趋近于 {2 * x0}（精确导数值）")
```

### 可导与连续的关系

一个重要的问题是：可导函数一定是连续函数吗？答案是肯定的。

**定理**：如果函数 $f$ 在点 $x_0$ 处可导，则 $f$ 在 $x_0$ 处连续。

**证明思路**：可导意味着 $\lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$ 存在。要证明连续，需要证明 $\lim_{\Delta x \to 0} [f(x_0 + \Delta x) - f(x_0)] = 0$。

$$f(x_0 + \Delta x) - f(x_0) = \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x} \cdot \Delta x$$

当 $\Delta x \to 0$ 时，第一项趋近于 $f'(x_0)$（导数存在），第二项趋近于 0，所以乘积趋近于 0。

但是，**连续不一定可导**。经典的反例是 $f(x) = |x|$ 在 $x = 0$ 处：函数连续，但在该点有一个"尖角"，左右导数不相等，因此不可导。

## 常见函数的导数

掌握基本函数的导数公式是进行微分运算的基础。

### 幂函数

对于幂函数 $f(x) = x^n$（$n$ 为正整数），其导数为：

$$\frac{d}{dx} x^n = nx^{n-1}$$

这个公式可以推广到任意实数 $n$，例如：
- $f(x) = x^{1/2} = \sqrt{x}$，则 $f'(x) = \frac{1}{2}x^{-1/2} = \frac{1}{2\sqrt{x}}$
- $f(x) = x^{-1} = \frac{1}{x}$，则 $f'(x) = -x^{-2} = -\frac{1}{x^2}$

### 指数函数与对数函数

对于自然指数函数 $f(x) = e^x$：

$$\frac{d}{dx} e^x = e^x$$

这是一个非常特殊的性质：$e^x$ 的导数等于它自己。正是这一性质使得 $e^x$ 在微分方程、概率论等领域具有核心地位。

对于一般指数函数 $f(x) = a^x$（$a > 0, a \neq 1$）：

$$\frac{d}{dx} a^x = a^x \ln a$$

对于自然对数函数 $f(x) = \ln x$：

$$\frac{d}{dx} \ln x = \frac{1}{x}$$

### 三角函数

基本三角函数的导数：

| 函数 | 导数 |
|------|------|
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x = \frac{1}{\cos^2 x}$ |

注意正弦和余弦的导数形成循环：$(\sin x)' = \cos x$，$(\cos x)' = -\sin x$，再求导又回到 $\sin x$（只差一个负号）。这一性质在求解微分方程时非常有用。

### 导数运算法则

对于函数的加减乘除，有相应的导数法则：

**加法法则**：$(f + g)' = f' + g'$

**减法法则**：$(f - g)' = f' - g'$

**乘法法则**（Product Rule）：$(f \cdot g)' = f' \cdot g + f \cdot g'$

**除法法则**（Quotient Rule）：$\left(\frac{f}{g}\right)' = \frac{f' \cdot g - f \cdot g'}{g^2}$

乘法法则的记忆口诀是"前导后不导，后导前不导，两者相加"。

### 用代码验证导数

我们可以用数值方法验证这些导数公式：

```python runnable
import numpy as np

def numerical_derivative(f, x, h=1e-5):
    """数值导数（中心差分）"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 验证 x^2 在 x=2 处的导数
f1 = lambda x: x ** 2
x = 2
analytical = 2 * x  # 解析导数：2x
numerical = numerical_derivative(f1, x)
print(f"x² 在 x={x} 处的导数：")
print(f"  解析值: {analytical}")
print(f"  数值值: {numerical:.6f}")
print(f"  误差: {abs(analytical - numerical):.2e}")

# 验证 sin(x) 在 x=π/4 处的导数
f2 = np.sin
x = np.pi / 4
analytical = np.cos(x)  # 解析导数：cos(x)
numerical = numerical_derivative(f2, x)
print(f"\nsin(x) 在 x=π/4 处的导数：")
print(f"  解析值: {analytical:.6f}")
print(f"  数值值: {numerical:.6f}")
print(f"  误差: {abs(analytical - numerical):.2e}")

# 验证 ln(x) 在 x=2 处的导数
f3 = np.log
x = 2
analytical = 1 / x  # 解析导数：1/x
numerical = numerical_derivative(f3, x)
print(f"\nln(x) 在 x={x} 处的导数：")
print(f"  解析值: {analytical:.6f}")
print(f"  数值值: {numerical:.6f}")
print(f"  误差: {abs(analytical - numerical):.2e}")
```

## 微分与线性近似

### 微分的概念

**微分（Differential）** 是导数的另一种表达形式。设函数 $y = f(x)$ 在点 $x$ 处可导，则称：

$$dy = f'(x) dx$$

为函数 $y = f(x)$ 在点 $x$ 处的微分。这里 $dx$ 是自变量的增量（一个独立的量），$dy$ 是因变量的微分。

微分与导数的区别在于：导数是一个比值 $\frac{dy}{dx}$，而微分 $dy$ 和 $dx$ 是独立的量。在机器学习中，我们经常使用微分形式的符号（如梯度下降中的参数更新），理解微分的概念有助于掌握这些符号的含义。

### 线性近似

微分的一个重要应用是**线性近似**（Linear Approximation）。当 $|\Delta x|$ 很小时，函数增量 $\Delta y = f(x + \Delta x) - f(x)$ 可以用微分 $dy = f'(x) \Delta x$ 来近似：

$$f(x + \Delta x) \approx f(x) + f'(x) \Delta x$$

这个公式在几何上表示：在点 $(x, f(x))$ 附近，用切线（直线）来近似曲线。

线性近似在工程计算中非常有用。例如，计算 $\sqrt{4.01}$：

设 $f(x) = \sqrt{x}$，取 $x = 4$，$\Delta x = 0.01$：

$$\sqrt{4.01} \approx \sqrt{4} + \frac{1}{2\sqrt{4}} \times 0.01 = 2 + \frac{1}{4} \times 0.01 = 2.0025$$

精确值 $\sqrt{4.01} \approx 2.002498$，误差仅约 $2 \times 10^{-6}$。

```python runnable
import numpy as np

# 线性近似计算 sqrt(4.01)
x = 4
dx = 0.01

# 线性近似
f = np.sqrt
df = lambda x: 1 / (2 * np.sqrt(x))  # sqrt(x) 的导数

approximation = f(x) + df(x) * dx
exact = f(x + dx)

print(f"线性近似: {approximation:.6f}")
print(f"精确值: {exact:.6f}")
print(f"误差: {abs(approximation - exact):.2e}")
```

线性近似是理解**泰勒展开**（Taylor Expansion）的基础。泰勒展开将函数在某点附近表示为多项式，一阶泰勒展开就是线性近似。

## 高阶导数

### 二阶导数

如果函数 $f$ 的导数 $f'$ 仍然可导，我们可以对 $f'$ 再求导，得到**二阶导数**（Second Derivative）：

$$f''(x) = \frac{d}{dx}\left(\frac{df}{dx}\right) = \frac{d^2 f}{dx^2}$$

二阶导数有重要的物理和几何意义：

- **物理意义**：如果 $f(t)$ 表示位置关于时间的函数，则 $f'(t)$ 是速度，$f''(t)$ 是**加速度**。
- **几何意义**：二阶导数描述了函数的**凹凸性**（Concavity）。

### 函数的凹凸性

**凹凸性判断法则**：
- 若 $f''(x) > 0$，则函数在 $x$ 处**下凸**（Convex，形状像碗口向上）
- 若 $f''(x) < 0$，则函数在 $x$ 处**上凸**（Concave，形状像碗口向下）
- 若 $f''(x) = 0$，则 $x$ 可能是**拐点**（Inflection Point）

在机器学习中，我们希望最小化的损失函数通常是下凸函数（Convex Function），这意味着二阶导数非负，函数图像"碗口向上"，有唯一的全局最小值。理解凹凸性对于理解优化算法（如梯度下降）的收敛性至关重要。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 演示凹凸性
x = np.linspace(-2, 2, 100)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# 下凸函数 f(x) = x^2
f1 = x ** 2
axes[0].plot(x, f1, 'b-', linewidth=2)
axes[0].set_title('下凸函数: f(x) = x²\nf\'\'(x) = 2 > 0')
axes[0].fill_between(x, 0, f1, alpha=0.2)
axes[0].grid(True)

# 上凸函数 f(x) = -x^2
f2 = -x ** 2
axes[1].plot(x, f2, 'r-', linewidth=2)
axes[1].set_title('上凸函数: f(x) = -x²\nf\'\'(x) = -2 < 0')
axes[1].fill_between(x, 0, f2, alpha=0.2)
axes[1].grid(True)

# 有拐点的函数 f(x) = x^3
f3 = x ** 3
axes[2].plot(x, f3, 'g-', linewidth=2)
axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[2].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[2].set_title('有拐点: f(x) = x³\nf\'\'(x) = 6x, 在 x=0 处变号')
axes[2].grid(True)

plt.tight_layout()
plt.show()
plt.close()
```

### 高阶导数的计算

对于某些函数，我们可以计算三阶、四阶甚至更高阶的导数。例如：

- $f(x) = e^x$，则 $f^{(n)}(x) = e^x$（任意阶导数都是它自己）
- $f(x) = \sin x$，则 $f'(x) = \cos x$，$f''(x) = -\sin x$，$f'''(x) = -\cos x$，$f^{(4)}(x) = \sin x$（每四阶循环一次）

高阶导数在泰勒展开、微分方程求解中有重要应用。

## 本章小结

本章建立了单变量微积分的理论基础，核心要点如下：

- **极限**描述了函数值的变化趋势，是微积分的理论基石。$\varepsilon-\delta$ 定义提供了严格的数学描述，而直观理解足以应对大多数应用场景。

- **导数**是差商的极限，表示函数在某点的瞬时变化率。几何上，导数是切线的斜率；物理上，位置函数的导数是速度，速度的导数是加速度。

- **常见函数的导数**包括幂函数、指数函数、对数函数、三角函数等，以及加减乘除的运算法则。这些公式是进行微分运算的基本工具。

- **微分**是导数的另一种表达形式，线性近似是微分的重要应用。在机器学习中，线性近似是理解泰勒展开和优化算法的基础。

- **高阶导数**中，二阶导数描述了函数的凹凸性，对于理解优化问题（如梯度下降的收敛性）至关重要。

这些概念相互关联：极限定义了导数，导数定义了微分，高阶导数揭示了函数更深层的性质。掌握这些基础概念，将为下一章学习多元函数微分学（偏导数、梯度、链式法则）奠定坚实基础。

## 练习题

1. 用极限的定义证明 $\lim_{x \to 2} (3x + 1) = 7$。
    <details>
    <summary>参考答案</summary>

    要证明 $\lim_{x \to 2} (3x + 1) = 7$，需要对于任意 $\varepsilon > 0$，找到 $\delta > 0$，使得当 $0 < |x - 2| < \delta$ 时，$|(3x + 1) - 7| < \varepsilon$。

    计算：$|(3x + 1) - 7| = |3x - 6| = 3|x - 2|$

    要使 $3|x - 2| < \varepsilon$，只需 $|x - 2| < \frac{\varepsilon}{3}$。

    因此，取 $\delta = \frac{\varepsilon}{3}$，当 $0 < |x - 2| < \delta$ 时，有 $|(3x + 1) - 7| = 3|x - 2| < 3 \cdot \frac{\varepsilon}{3} = \varepsilon$。

    这就证明了 $\lim_{x \to 2} (3x + 1) = 7$。
    </details>

2. 用导数的定义求 $f(x) = x^3$ 在 $x = 1$ 处的导数。
    <details>
    <summary>参考答案</summary>

    根据导数定义：$f'(1) = \lim_{\Delta x \to 0} \frac{f(1 + \Delta x) - f(1)}{\Delta x}$

    计算：
    - $f(1) = 1^3 = 1$
    - $f(1 + \Delta x) = (1 + \Delta x)^3 = 1 + 3\Delta x + 3(\Delta x)^2 + (\Delta x)^3$

    因此：
    $$f'(1) = \lim_{\Delta x \to 0} \frac{(1 + 3\Delta x + 3(\Delta x)^2 + (\Delta x)^3) - 1}{\Delta x} = \lim_{\Delta x \to 0} \frac{3\Delta x + 3(\Delta x)^2 + (\Delta x)^3}{\Delta x}$$

    $$= \lim_{\Delta x \to 0} [3 + 3\Delta x + (\Delta x)^2] = 3$$

    也可以直接用幂函数导数公式验证：$f'(x) = 3x^2$，所以 $f'(1) = 3$。
    </details>

3. 求下列函数的导数：
   - (a) $f(x) = x^4 - 3x^2 + 2x - 5$
   - (b) $g(x) = e^x \sin x$
   - (c) $h(x) = \frac{\ln x}{x}$
    <details>
    <summary>参考答案</summary>

    (a) 应用幂函数导数公式和加减法则：
    $$f'(x) = 4x^3 - 6x + 2$$

    (b) 应用乘法法则 $(f \cdot g)' = f' \cdot g + f \cdot g'$：
    $$g'(x) = \frac{d}{dx}(e^x) \cdot \sin x + e^x \cdot \frac{d}{dx}(\sin x) = e^x \sin x + e^x \cos x = e^x(\sin x + \cos x)$$

    (c) 应用除法法则 $\left(\frac{f}{g}\right)' = \frac{f' \cdot g - f \cdot g'}{g^2}$：
    $$h'(x) = \frac{\frac{1}{x} \cdot x - \ln x \cdot 1}{x^2} = \frac{1 - \ln x}{x^2}$$
    </details>

4. 设 $f(x) = x^3 - 3x$，求：
   - (a) 函数的单调递增和递减区间
   - (b) 函数的凹凸区间和拐点
    <details>
    <summary>参考答案</summary>

    (a) 首先求一阶导数：$f'(x) = 3x^2 - 3 = 3(x^2 - 1)$

    令 $f'(x) = 0$，得 $x = \pm 1$。
    - 当 $x < -1$ 或 $x > 1$ 时，$f'(x) > 0$，函数递增
    - 当 $-1 < x < 1$ 时，$f'(x) < 0$，函数递减

    (b) 求二阶导数：$f''(x) = 6x$

    令 $f''(x) = 0$，得 $x = 0$。
    - 当 $x < 0$ 时，$f''(x) < 0$，函数上凸
    - 当 $x > 0$ 时，$f''(x) > 0$，函数下凸

    因此，$x = 0$ 是拐点，拐点坐标为 $(0, 0)$。
    </details>

5. 用线性近似估算 $\sin(0.1)$ 的值（弧度制），并与精确值比较误差。
    <details>
    <summary>参考答案</summary>

    设 $f(x) = \sin x$，取 $x_0 = 0$，$\Delta x = 0.1$。

    线性近似公式：$f(x_0 + \Delta x) \approx f(x_0) + f'(x_0) \cdot \Delta x$

    计算：
    - $f(0) = \sin 0 = 0$
    - $f'(x) = \cos x$，所以 $f'(0) = \cos 0 = 1$

    因此：$\sin(0.1) \approx 0 + 1 \times 0.1 = 0.1$

    精确值：$\sin(0.1) \approx 0.099833$

    误差：$|0.1 - 0.099833| \approx 0.000167 \approx 1.67 \times 10^{-4}$

    相对误差：$\frac{0.000167}{0.099833} \approx 0.17\%$

    可见对于小角度，$\sin x \approx x$ 是一个很好的近似。
    </details>

6. 证明：如果 $f$ 在 $x_0$ 处可导，则 $f$ 在 $x_0$ 处连续。（提示：利用极限的运算法则）
    <details>
    <summary>参考答案</summary>

    要证明 $f$ 在 $x_0$ 处连续，需要证明 $\lim_{x \to x_0} f(x) = f(x_0)$。

    设 $h = x - x_0$，则 $x \to x_0$ 等价于 $h \to 0$。

    $$f(x) = f(x_0 + h) = f(x_0) + \frac{f(x_0 + h) - f(x_0)}{h} \cdot h$$

    当 $h \to 0$ 时：
    - $\frac{f(x_0 + h) - f(x_0)}{h} \to f'(x_0)$（因为 $f$ 在 $x_0$ 处可导）
    - $h \to 0$

    因此：
    $$\lim_{h \to 0} f(x_0 + h) = f(x_0) + f'(x_0) \cdot 0 = f(x_0)$$

    这证明了 $f$ 在 $x_0$ 处连续。

    注意：反之不成立。例如 $f(x) = |x|$ 在 $x = 0$ 处连续但不可导。
    </details>