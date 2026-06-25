# 概率统计实践

[概率统计](probability-basics.md)章节建立了概率统计的理论基础。本章将这些知识付诸实践，使用 NumPy 实现概率统计的核心计算。通过代码实践，不仅能加深对概念的理解，还能掌握实际数据分析的技能。

## 分布采样

**均匀分布**（Uniform Distribution）是最简单的连续概率分布，其概率密度函数在定义区间内处处相等。对于 $[a, b]$ 区间的均匀分布，变量 $X$ 落在该区间内任意子区间 $[c, d]$ 的概率与子区间长度成正比：$P(c \leq X \leq d) = \frac{d-c}{b-a}$。均匀分布在实际中常用于模拟"无偏好"的随机选择，譬如随机抽取样本、随机分配任务等。

[**正态分布**](probability-basics.md#正态分布)（Normal Distribution，也称高斯分布）是统计学中最重要的分布，其概率密度函数呈对称的钟形曲线。自然界和人类社会中大量现象近似服从正态分布，如人的身高、考试成绩、测量误差等。标准正态分布 $N(0, 1)$ 的概率密度函数为：$f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$，其特点是均值（期望）为 0，标准差为 1，曲线在均值处最高，向两侧逐渐降低，约 68% 的数据落在 $\pm 1$ 个标准差范围内，约 95% 落在 $\pm 2$ 个标准差范围内。

从概率分布中**采样**（Sampling），就是按照该分布的概率密度函数生成随机数，某个值出现的概率越高，生成该值的可能性就越大。采样是蒙特卡洛模拟、随机算法、统计推断等技术的基础操作。NumPy 提供了简洁的函数来实现这两种最基本分布的采样：

- `np.random.rand(n)`：从 $[0, 1)$ 区间的均匀分布采样，返回 $n$ 个样本。
- `np.random.randn(n)`：从标准正态分布 $N(0, 1)$ 采样，返回 $n$ 个样本。
- `np.random.uniform(low, high, n)`：从任意区间 $[low, high]$ 的均匀分布采样。
- `np.random.normal(mean, std, n)`：从任意参数的正态分布 $N(\mu, \sigma^2)$ 采样。

下面的代码演示了均匀分布和正态分布的采样过程，并通过可视化对比采样数据的直方图与理论[概率密度函数](probability-basics.md#概率密度函数)（PDF），验证采样的正确性。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 均匀分布采样
n = 10000
uniform_samples = np.random.rand(n)  # [0, 1) 均匀分布

# 正态分布采样
normal_samples = np.random.randn(n)  # 标准正态分布

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 均匀分布
axes[0].hist(uniform_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axhline(1, color='r', linestyle='--', linewidth=2, label='理论 PDF')
axes[0].set_xlabel('值')
axes[0].set_ylabel('概率密度')
axes[0].set_title(f'均匀分布采样 (n={n})')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 正态分布
axes[1].hist(normal_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')

# 理论 PDF
x = np.linspace(-4, 4, 100)
pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)
axes[1].plot(x, pdf, 'r-', linewidth=2, label='理论 PDF')
axes[1].set_xlabel('值')
axes[1].set_ylabel('概率密度')
axes[1].set_title(f'标准正态分布采样 (n={n})')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()
```

## 随机种子与可复现性

**随机种子**（Random Seed）是控制随机数生成器初始状态的数值。计算机中的"随机"实际上是伪随机，由确定性算法生成，给定相同种子，算法将产生完全相同的随机数序列。在机器学习实验、科学计算和数据分析中，**可复现性**（Reproducibility）是基本要求，他人能够重复你的实验并获得相同结果，对结果验证、错误排查、模型对比都是不可或缺的。

NumPy 的随机数生成器维护一个内部状态，通过 `np.random.seed(n)` 设置随机种子为整数 $n$，同一种子在不同运行中产生相同序列，此后所有随机操作结果便可确定。常用种子值如 42、0 等已成为社区惯例（并无特殊含义，只是便于记忆）。下面的代码对比了不设置种子与设置种子时随机数生成的差异，直观展示随机种子如何控制可复现性。

```python runnable
import numpy as np

# 不设置种子：每次结果不同
print("不设置随机种子:")
for i in range(3):
    samples = np.random.rand(5)
    print(f"  第 {i+1} 次：{samples}")

print()

# 设置种子：每次结果相同
print("设置随机种子 (seed=42):")
for i in range(3):
    np.random.seed(42)
    samples = np.random.rand(5)
    print(f"  第 {i+1} 次：{samples}")
```

## 随机选择与打乱

**随机选择**（Random Choice）是从给定集合中按指定规则抽取元素的操作。根据是否允许重复抽取，分为有放回选择（同一元素可被多次选中）和无放回选择（每个元素最多选中一次）。加权选择则允许为每个元素设定被选中的概率权重，在实际中可用于模拟不同选项具有不同概率的决策场景，如推荐系统的多样性采样、问卷调查的分层抽样等。

**随机打乱**（Random Shuffle）是对序列元素顺序的随机重排，使得每个元素出现在任意位置的概率相等。打乱操作在数据预处理中极为常用：训练集与测试集的随机划分、数据增强中的随机顺序、实验设计的随机分组等场景都需要打破数据的原始顺序以避免偏差。

NumPy 提供了以下函数实现这些操作：

- `np.random.choice(array, size, replace)`：从数组中随机选择元素，`replace` 控制是否允许重复
- `np.random.choice(array, size, p)`：加权选择，`p` 为各元素的选中概率（需与数组长度相同且总和为 1）
- `np.random.shuffle(array)`：原地打乱数组顺序，直接修改原数组

下面的代码演示了随机选择的三种模式（有放回、无放回、加权）以及随机打乱的效果。



```python runnable
import numpy as np
# 随机选择
data = np.array(['苹果', '香蕉', '橙子', '葡萄', '西瓜'])

# 有放回选择
choices_with_replacement = np.random.choice(data, size=10, replace=True)
print("有放回选择:", choices_with_replacement)

# 无放回选择
choices_without_replacement = np.random.choice(data, size=3, replace=False)
print("无放回选择:", choices_without_replacement)

# 加权选择
weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # 各元素被选中的概率
weighted_choices = np.random.choice(data, size=10, p=weights)
print("加权选择:", weighted_choices)

# 随机打乱
arr = np.arange(10)
print(f"\n原始数组：{arr}")
np.random.shuffle(arr)
print(f"打乱后：{arr}")
```

## 分布的可视化

通过可视化分布的形状，可以直观理解其特性：离散分布（[二项分布](probability-basics.md#二项分布)、[泊松分布](probability-basics.md#指数分布与泊松分布)）的直方图呈现条状，每个条形高度反映该取值的概率；连续分布（[指数分布](probability-basics.md#指数分布与泊松分布)）的直方图需通过调整参数呈现平滑曲线，高度反映概率密度。NumPy 提供了丰富的分布采样函数：

- `np.random.binomial(n, p, size)`：从二项分布 $B(n, p)$ 采样
- `np.random.poisson(lam, size)`：从泊松分布 $Poisson(\lambda)$ 采样
- `np.random.exponential(scale, size)`：从指数分布 $Exp(scale)$ 采样，`scale` 为平均等待时间

下面的代码演示了三种分布的采样与可视化，并使用 `np.histogram` 计算直方图的原始数据。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

n = 10000 # 生成多种分布的数据
# 二项分布
binomial = np.random.binomial(n=20, p=0.3, size=n)
# 泊松分布
poisson = np.random.poisson(lam=5, size=n)
# 指数分布
exponential = np.random.exponential(scale=2, size=n)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 二项分布（离散）
axes[0].hist(binomial, bins=range(0, 21), density=True, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('值')
axes[0].set_ylabel('概率')
axes[0].set_title('二项分布 B(20, 0.3)')
axes[0].grid(alpha=0.3, axis='y')

# 泊松分布（离散）
axes[1].hist(poisson, bins=range(0, 20), density=True, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('值')
axes[1].set_ylabel('概率')
axes[1].set_title('泊松分布 Poisson(5)')
axes[1].grid(alpha=0.3, axis='y')

# 指数分布（连续）
axes[2].hist(exponential, bins=50, density=True, color='steelblue', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('值')
axes[2].set_ylabel('概率密度')
axes[2].set_title('指数分布 Exp(2)')
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# 使用 np.histogram 计算直方图数据
hist, bin_edges = np.histogram(binomial, bins=range(0, 22), density=True)
print("二项分布直方图数据:")
print(f"  区间边界： {bin_edges[:6]}...")  # 只显示前几个
print(f"  概率： {hist[:5]}...")
```

## 统计量计算

**描述性统计量**（Descriptive Statistics）是对数据集基本特征的数值概括，分为三类：集中趋势（均值、中位数）、离散程度（方差、标准差、范围）和分布形状（分位数、偏度、峰度）。这些统计量不依赖于数据的分布假设，提供对数据的"第一眼"认知，帮助快速判断数据特征、发现异常值、比较不同数据集。NumPy 提供了高效的统计量计算函数：

- `np.mean(data)` / `np.median(data)`：计算均值和中位数
- `np.var(data)` / `np.std(data)`：计算方差和标准差
- `np.min(data)` / `np.max(data)` / `np.ptp(data)`：最小值、最大值、范围
- `np.percentile(data, p)`：计算 $p$ 分位数
- `np.cov(x, y)` / `np.corrcoef(x, y)`：协方差和相关系数

下面的代码以模拟的考试成绩数据为例，演示各项描述性统计量的计算，并手动实现偏度和峰度的计算公式。

```python runnable
import numpy as np

data = np.random.normal(100, 15, 1000)  # 模拟考试成绩

print("=== 描述性统计 ===")
print(f"样本量：{len(data)}")
print(f"最小值：{np.min(data):.2f}")
print(f"最大值：{np.max(data):.2f}")
print(f"范围：{np.ptp(data):.2f}")  # peak-to-peak
print()
print(f"均值：{np.mean(data):.2f}")
print(f"中位数：{np.median(data):.2f}")
print()
print(f"方差：{np.var(data):.2f}")
print(f"标准差：{np.std(data):.2f}")
print()

# 分位数
percentiles = [25, 50, 75]
for p in percentiles:
    print(f"{p}% 分位数：{np.percentile(data, p):.2f}")

# 四分位距
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
print(f"\n四分位距 (IQR): {iqr:.2f}")

# 偏度和峰度（手动计算）
mean = np.mean(data)
std = np.std(data)
skewness = np.mean(((data - mean) / std) ** 3)
kurtosis = np.mean(((data - mean) / std) ** 4) - 3

print(f"\n偏度：{skewness:.4f} (正态分布为 0)")
print(f"峰度：{kurtosis:.4f} (正态分布为 0)")
```

## 协方差与相关系数

**协方差**（Covariance）衡量两个随机变量协同变化的程度。对于变量 $X$ 和 $Y$，协方差定义为 $Cov(X, Y) = E[(X - E[X])(Y - E[Y])]$。协方差为正表示两变量倾向于同向变化（一个增大时另一个也增大），为负表示反向变化，为零表示线性无关。协方差的大小受变量单位影响，难以直接比较不同数据集之间的关系强度。

**相关系数**（Correlation Coefficient）是协方差的标准化版本，消除了单位的影响。相关系数定义为 $r = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}$，取值范围 $[-1, 1]$。$r = 1$ 表示完美正相关，$r = -1$ 表示完美负相关，$r = 0$ 表示无线性相关性。相关系数是数据分析中最常用的关系度量指标，广泛应用于特征选择、因子分析、回归诊断等场景。

协方差矩阵和相关系数矩阵将这种度量推广到多变量场景。对于 $n$ 个变量，矩阵的第 $i$ 行第 $j$ 列元素表示第 $i$ 个和第 $j$ 个变量的协方差或相关系数。矩阵的对角线元素是方差（协方差矩阵）或 1（相关系数矩阵）。NumPy 提供了以下函数计算这些统计量：

- `np.cov(x, y)`：计算两个变量的协方差
- `np.cov([x1, x2, ...])`：计算多变量的协方差矩阵
- `np.corrcoef(x, y)`：计算两个变量的相关系数
- `np.corrcoef([x1, x2, ...])`：计算多变量的相关系数矩阵

下面的代码生成三组数据：$y$ 与 $x$ 正相关、$z$ 与 $x$ 独立，演示协方差矩阵和相关系数矩阵的计算，并通过散点图直观展示不同相关强度的视觉效果。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 生成相关数据
n = 100
x = np.random.randn(n)
y = 0.8 * x + 0.2 * np.random.randn(n)  # y 与 x 正相关
z = np.random.randn(n)  # z 与 x 不相关

# 计算协方差矩阵
cov_matrix = np.cov([x, y, z])
print("协方差矩阵:")
print(np.round(cov_matrix, 3))
print()

# 计算相关系数矩阵
corr_matrix = np.corrcoef([x, y, z])
print("相关系数矩阵:")
print(np.round(corr_matrix, 3))
print()

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].scatter(x, y, alpha=0.6)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(f'x vs y (r = {corr_matrix[0,1]:.2f})')
axes[0].grid(alpha=0.3)
axes[1].scatter(x, z, alpha=0.6)
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
axes[1].set_title(f'x vs z (r = {corr_matrix[0,2]:.2f})')
axes[1].grid(alpha=0.3)
axes[2].scatter(y, z, alpha=0.6)
axes[2].set_xlabel('y')
axes[2].set_ylabel('z')
axes[2].set_title(f'y vs z (r = {corr_matrix[1,2]:.2f})')
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()
```

## 蒙特卡洛方法

许多实际问题难以用解析方法精确求解，譬如积分计算中函数可能没有闭式表达，概率估计中事件组合可能过于复杂，优化问题中目标函数可能不可导，等等。**蒙特卡洛方法**（Monte Carlo Method）为这类问题提供了一条近似求解的路径。

蒙特卡洛方法得名于摩纳哥的蒙特卡洛赌场，随机性是该方法的核心。该方法通过大量随机采样，将复杂问题转化为大量简单样本的统计聚合。其工作原理建立在两个数学基础之上：[大数定律](https://en.wikipedia.org/wiki/Law_of_large_numbers)保证当样本量趋于无穷时，样本均值会收敛于期望值（即真值）；[中心极限定理](https://en.wikipedia.org/wiki/Central_limit_theorem)保证收敛的误差分布近似正态，使估计误差可量化、可预测。具体而言，用样本均值估计积分值、用成功频率估计概率、用随机试验统计近似解析结果，采样数量越大，估计精度越高。

- 以下代码演示蒙特卡洛方法计算积分 $\int_a^b f(x) dx \approx \frac{b-a}{N} \sum_{i=1}^N f(x_i)$，其中 $x_i$ 是 $[a, b]$ 上的均匀随机采样：

    ```python runnable
    import numpy as np
    import matplotlib.pyplot as plt

    # 计算 ∫_0^1 sin(x) dx
    # 真实值： 1 - cos(1) ≈ 0.4597
    def f(x):
        return np.sin(x)
    a, b = 0, 1
    true_value = 1 - np.cos(1)  # 解析解

    # 不同采样数量的估计
    sample_sizes = [100, 1000, 10000, 100000]
    estimates = []

    for n in sample_sizes:
        x_samples = np.random.uniform(a, b, n)
        estimate = (b - a) * np.mean(f(x_samples))
        estimates.append(estimate)
        error = abs(estimate - true_value)
        print(f"n = {n:6d}: 估计值 = {estimate:.6f}, 误差 = {error:.6f}")
    print(f"\n真实值：{true_value:.6f}")

    # 可视化收敛过程
    n_range = np.arange(100, 10001, 100)
    convergence = []
    for n in n_range:
        x_samples = np.random.uniform(a, b, n)
        estimate = (b - a) * np.mean(f(x_samples))
        convergence.append(estimate)

    plt.figure(figsize=(10, 5))
    plt.plot(n_range, convergence, 'b-', alpha=0.5, label='Monte Carlo 估计')
    plt.axhline(true_value, color='r', linestyle='--', linewidth=2, label=f'真实值 = {true_value:.4f}')
    plt.xlabel('采样数量 n')
    plt.ylabel('积分估计')
    plt.title('Monte Carlo 积分估计收敛过程')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    ```

- 蒙特卡洛方法的另一个经典示例是用随机投点估计 π。在边长为 2 的正方形内随机投点，统计落在单位圆内的比例。由于圆面积 $\pi$ 与正方形面积 4 的比值为 $\pi/4$, 故该比例乘以 4 即得 π 的估计值。

    ```python runnable
    import numpy as np
    import matplotlib.pyplot as plt

    # 用 Monte Carlo 方法估计 π
    # 在单位正方形内随机投点，落在单位圆内的比例 = π/4
    n = 10000
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    # 判断是否在单位圆内
    inside = x**2 + y**2 <= 1
    # 估计 π
    pi_estimate = 4 * np.sum(inside) / n
    error = abs(pi_estimate - np.pi)

    print(f"采样点数：{n}")
    print(f"落在圆内的点数：{np.sum(inside)}")
    print(f"π 估计值：{pi_estimate:.6f}")
    print(f"真实值：{np.pi:.6f}")
    print(f"误差：{error:.6f}")

    # 可视化
    plt.figure(figsize=(8, 8))
    # 绘制点
    plt.scatter(x[inside], y[inside], c='blue', s=1, alpha=0.5, label='圆内')
    plt.scatter(x[~inside], y[~inside], c='red', s=1, alpha=0.5, label='圆外')
    # 绘制圆
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    # 绘制正方形
    plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Monte Carlo 估计 π = {pi_estimate:.4f}')
    plt.axis('equal')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

    # 收敛过程
    sample_sizes = [100, 1000, 5000, 10000, 50000, 100000]
    estimates = []
    for n_samples in sample_sizes:
        x = np.random.uniform(-1, 1, n_samples)
        y = np.random.uniform(-1, 1, n_samples)
        inside = x**2 + y**2 <= 1
        estimates.append(4 * np.sum(inside) / n_samples)

    print("\n收敛过程:")
    for n_samples, est in zip(sample_sizes, estimates):
        print(f"  n = {n_samples:6d}: π ≈ {est:.6f}, 误差 = {abs(est - np.pi):.6f}")
    ```

- 蒙特卡洛方法还可以估计复杂事件的概率。通过大量随机试验模拟事件发生过程，统计事件发生的频率作为概率的估计值，当试验次数足够多时，频率趋于概率。以下代码估计三个标准正态变量之和大于 3 的概率：

    ```python runnable
    import numpy as np
    import matplotlib.pyplot as plt

    # 估计三个标准正态变量之和大于 3 的概率
    def estimate_probability(n_samples=100000):
        """估计 P(X + Y + Z > 3)，其中 X, Y, Z ~ N(0,1)"""
        X = np.random.randn(n_samples)
        Y = np.random.randn(n_samples)
        Z = np.random.randn(n_samples)
        S = X + Y + Z
        count = np.sum(S > 3)   
        return count / n_samples

    # Monte Carlo 估计
    n_samples = 100000
    prob_estimate = estimate_probability(n_samples)

    # 理论值（三个独立 N(0,1) 变量之和 S 服从 N(0, 3)，标准化后 S/√3 ~ N(0,1)）
    from math import erf, sqrt
    z = 3 / sqrt(3)
    prob_theory = 0.5 * (1 - erf(z / sqrt(2)))
    print(f"P(X + Y + Z > 3) 的估计")
    print(f"  Monte Carlo 估计：{prob_estimate:.6f}")
    print(f"  理论值：{prob_theory:.6f}")
    print(f"  误差：{abs(prob_estimate - prob_theory):.6f}")

    # 可视化 S 的分布
    n = 100000
    X = np.random.randn(n)
    Y = np.random.randn(n)
    Z = np.random.randn(n)
    S = X + Y + Z

    plt.figure(figsize=(10, 5))
    plt.hist(S, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')

    # 理论 PDF
    x = np.linspace(-6, 6, 100)
    pdf = 1 / sqrt(2 * np.pi * 3) * np.exp(-x**2 / (2 * 3))
    plt.plot(x, pdf, 'r-', linewidth=2, label='理论 PDF: N(0, 3)')
    plt.axvline(3, color='green', linestyle='--', linewidth=2, label='阈值 = 3')
    plt.xlabel('S = X + Y + Z')
    plt.ylabel('概率密度')
    plt.title('三个标准正态变量之和的分布')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    ```

## 贝叶斯后验采样

**贝叶斯推断**（Bayesian Inference）是一种统计推断方法，通过观测数据更新对未知参数的认知。贝叶斯定理将参数的先验分布与数据的似然函数结合，得到后验分布：$P(\theta | data) = \frac{P(data | \theta) P(\theta)}{P(data)}$。其中，$P(\theta)$ 称为**先验分布**（Prior），代表观测数据前对参数的认知；$P(data | \theta)$ 称为**似然函数**（Likelihood），描述参数取值下观测数据出现的概率；$P(data)$ 是归一化常数，确保后验分布积分为 1。后验分布完整刻画了参数的不确定性，包含参数的所有可能取值及其相对概率。

**后验采样**是从后验分布中抽取样本的过程。当后验分布没有解析形式或计算困难时，采样方法成为获取后验信息的实用途径。通过大量后验样本，可以估计参数的后验均值、方差、可信区间等统计量，实现对参数不确定性的完整描述。贝叶斯后验采样广泛应用于机器学习中的参数估计、模型选择、预测推断等场景。

NumPy 提供了以下方法实现贝叶斯后验采样：

- `np.random.gamma(shape, scale, size)`：采样 Gamma 分布，用于 Beta-Gamma 关系采样
- `np.random.binomial(n, p, size)`：生成二项分布观测数据
- `np.percentile(samples, p)`：从后验样本计算可信区间

下面的代码以硬币正面概率估计为例，演示贝叶斯后验采样的完整流程：生成观测数据、计算后验参数、从 Beta 后验分布采样、统计后验信息。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 贝叶斯后验采样：估计硬币正面概率
# 真实参数
true_p = 0.6
n_flips = 50

# 生成观测数据
flips = np.random.binomial(1, true_p, n_flips)
n_heads = flips.sum()

print(f"观测数据：{n_flips} 次抛掷, {n_heads} 次正面")
print(f"MLE 估计（最大似然估计）： p̂ = {n_heads/n_flips:.3f}")
print()

# 使用拒绝采样从后验分布采样
# 先验： Beta(2, 2)
# 后验： Beta(2 + n_heads, 2 + n_flips - n_heads)

alpha_post = 2 + n_heads
beta_post = 2 + n_flips - n_heads

# 使用逆 CDF 方法采样 Beta 分布（简化版）
def sample_beta(alpha, beta, n_samples=10000):
    """使用 Beta-Gamma 关系采样 Beta 分布"""
    x = np.random.gamma(alpha, 1, n_samples)
    y = np.random.gamma(beta, 1, n_samples)
    return x / (x + y)

# 从后验采样
posterior_samples = sample_beta(alpha_post, beta_post, 10000)

# 后验统计量
print("后验分布统计:")
print(f"  后验均值：{posterior_samples.mean():.4f}")
print(f"  后验标准差：{posterior_samples.std():.4f}")
print(f"  95% 可信区间：[{np.percentile(posterior_samples, 2.5):.4f}, {np.percentile(posterior_samples, 97.5):.4f}]")

# 可视化
plt.figure(figsize=(10, 5))

plt.hist(posterior_samples, bins=50, density=True, alpha=0.7, 
         color='steelblue', edgecolor='black')

plt.axvline(true_p, color='green', linestyle='--', linewidth=2, label=f'真实值 p = {true_p}')
plt.axvline(n_heads/n_flips, color='red', linestyle=':', linewidth=2, label=f'MLE = {n_heads/n_flips:.3f}')
plt.axvline(posterior_samples.mean(), color='purple', linestyle='-.', linewidth=2, label=f'后验均值 = {posterior_samples.mean():.3f}')

plt.xlabel('p')
plt.ylabel('后验密度')
plt.title(f'贝叶斯后验采样 (Beta({alpha_post}, {beta_post}))')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()
```

## 本章小结

本章对概率统计计算进行了编程实践，从最基本的分布采样（均匀分布、正态分布）出发，逐步深入到随机种子控制、随机选择与打乱操作、多种概率分布的可视化、描述性统计量计算、协方差与相关系数分析、蒙特卡洛模拟方法，以及贝叶斯后验采样技术。这些方法构成了数据分析与机器学习的计算基础。均匀分布和正态分布采样是随机模拟的起点；随机种子确保了实验的可复现性；描述性统计量提供了对数据特征的第一眼认知；协方差和相关系数量化了变量之间的关系强度；蒙特卡洛方法为复杂问题的近似求解提供了通用框架；贝叶斯后验采样则在参数估计中实现了对不确定性的完整描述。

## 练习题

1. 使用蒙特卡洛方法估计 $\int_0^1 x^2 dx$ 的值，并与真实值比较。
   <details>
   <summary>参考答案</summary>

   ```python runnable
   import numpy as np

   n = 100000
   x = np.random.uniform(0, 1, n)
   estimate = np.mean(x**2)
   
   true_value = 1/3
   
   print(f"Monte Carlo 估计：{estimate:.6f}")
   print(f"真实值：{true_value:.6f}")
   print(f"误差：{abs(estimate - true_value):.6f}")
   ```

   </details>

2. 使用蒙特卡洛方法验证中心极限定理（无论原始分布如何，当样本量足够大时，样本均值的分布近似服从正态分布）。
   <details>
   <summary>参考答案</summary>

   ```python runnable
   import numpy as np
   import matplotlib.pyplot as plt
   
   # 从均匀分布采样
   n_samples = 10000
   sample_size = 30
   sample_means = []
   for _ in range(n_samples):
       sample = np.random.uniform(0, 1, sample_size)
       sample_means.append(np.mean(sample))
   sample_means = np.array(sample_means)
   
   # 理论值
   # X ~ U(0,1): E[X] = 0.5, Var[X] = 1/12
   # X̄: E[X̄] = 0.5, Var[X̄] = 1/(12*30)
   theoretical_mean = 0.5
   theoretical_std = np.sqrt(1 / (12 * sample_size))
   
   print(f"样本均值的均值：{sample_means.mean():.4f} (理论：{theoretical_mean})")
   print(f"样本均值的标准差：{sample_means.std():.4f} (理论：{theoretical_std:.4f})")
   
   # 可视化
   plt.hist(sample_means, bins=50, density=True, alpha=0.7)
   x = np.linspace(0.3, 0.7, 100)
   pdf = 1 / (theoretical_std * np.sqrt(2*np.pi)) * np.exp(-(x - theoretical_mean)**2 / (2 * theoretical_std**2))
   plt.plot(x, pdf, 'r-', linewidth=2, label='理论正态分布')
   plt.xlabel('样本均值')
   plt.ylabel('密度')
   plt.title('中心极限定理验证')
   plt.legend()
   plt.show()
   ```

   </details>