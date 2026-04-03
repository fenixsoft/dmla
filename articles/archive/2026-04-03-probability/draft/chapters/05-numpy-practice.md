---
title: "NumPy 实践：概率统计的计算工具"
---

# NumPy 实践：概率统计的计算工具

前四章建立了概率统计的理论基础。本章将这些知识付诸实践，使用 NumPy 实现概率统计的核心计算。通过代码实践，不仅能加深对概念的理解，还能掌握实际数据分析的技能。

## 随机数生成与采样

### np.random 模块概述

NumPy 的 `np.random` 模块提供了丰富的随机数生成功能，是概率统计计算的基础。

```python runnable
import numpy as np

# 查看随机数生成器
print("NumPy 随机数生成模块概览")
print("=" * 50)

# 常用函数列表
functions = [
    ("np.random.seed()", "设置随机种子"),
    ("np.random.rand()", "均匀分布 [0, 1)"),
    ("np.random.randn()", "标准正态分布"),
    ("np.random.randint()", "随机整数"),
    ("np.random.choice()", "随机选择"),
    ("np.random.shuffle()", "随机打乱"),
    ("np.random.binomial()", "二项分布"),
    ("np.random.normal()", "正态分布"),
    ("np.random.poisson()", "泊松分布"),
    ("np.random.exponential()", "指数分布"),
]

for func, desc in functions:
    print(f"  {func:30s} - {desc}")
```

### 均匀分布与正态分布采样

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，确保可复现
np.random.seed(42)

# 均匀分布采样
n = 10000
uniform_samples = np.random.rand(n)  # [0, 1) 均匀分布

# 正态分布采样
normal_samples = np.random.randn(n)  # 标准正态分布

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 均匀分布
axes[0].hist(uniform_samples, bins=50, density=True, alpha=0.7, 
             color='steelblue', edgecolor='black')
axes[0].axhline(1, color='r', linestyle='--', linewidth=2, label='理论 PDF')
axes[0].set_xlabel('值')
axes[0].set_ylabel('概率密度')
axes[0].set_title(f'均匀分布采样 (n={n})')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 正态分布
axes[1].hist(normal_samples, bins=50, density=True, alpha=0.7, 
             color='steelblue', edgecolor='black')

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

# 检验采样质量
print("均匀分布:")
print(f"  样本均值: {uniform_samples.mean():.4f} (理论: 0.5)")
print(f"  样本方差: {uniform_samples.var():.4f} (理论: 1/12 ≈ 0.0833)")
print()
print("标准正态分布:")
print(f"  样本均值: {normal_samples.mean():.4f} (理论: 0)")
print(f"  样本标准差: {normal_samples.std():.4f} (理论: 1)")
```

### 随机种子与可复现性

在机器学习实验中，**可复现性**至关重要。设置随机种子确保每次运行得到相同结果。

```python runnable
import numpy as np

# 不设置种子：每次结果不同
print("不设置随机种子:")
for i in range(3):
    samples = np.random.rand(5)
    print(f"  第 {i+1} 次: {samples}")

print()

# 设置种子：每次结果相同
print("设置随机种子 (seed=42):")
for i in range(3):
    np.random.seed(42)
    samples = np.random.rand(5)
    print(f"  第 {i+1} 次: {samples}")
```

### 随机选择与打乱

```python runnable
import numpy as np

np.random.seed(42)

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
print(f"\n原始数组: {arr}")
np.random.shuffle(arr)
print(f"打乱后: {arr}")
```

## 分布的 NumPy 表示

### 直方图与分布可视化

```python runnable
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 生成多种分布的数据
n = 10000

# 二项分布
binomial = np.random.binomial(n=20, p=0.3, size=n)

# 泊松分布
poisson = np.random.poisson(lam=5, size=n)

# 指数分布
exponential = np.random.exponential(scale=2, size=n)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 二项分布（离散）
axes[0].hist(binomial, bins=range(0, 21), density=True, 
             color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('值')
axes[0].set_ylabel('概率')
axes[0].set_title('二项分布 B(20, 0.3)')
axes[0].grid(alpha=0.3, axis='y')

# 泊松分布（离散）
axes[1].hist(poisson, bins=range(0, 20), density=True, 
             color='steelblue', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('值')
axes[1].set_ylabel('概率')
axes[1].set_title('泊松分布 Poisson(5)')
axes[1].grid(alpha=0.3, axis='y')

# 指数分布（连续）
axes[2].hist(exponential, bins=50, density=True, 
             color='steelblue', edgecolor='black', alpha=0.7)
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
print(f"  区间边界: {bin_edges[:6]}...")  # 只显示前几个
print(f"  概率: {hist[:5]}...")
```

### 统计量计算

```python runnable
import numpy as np

np.random.seed(42)
data = np.random.normal(100, 15, 1000)  # 模拟考试成绩

print("=== 描述性统计 ===")
print(f"样本量: {len(data)}")
print(f"最小值: {np.min(data):.2f}")
print(f"最大值: {np.max(data):.2f}")
print(f"范围: {np.ptp(data):.2f}")  # peak-to-peak
print()
print(f"均值: {np.mean(data):.2f}")
print(f"中位数: {np.median(data):.2f}")
print()
print(f"方差: {np.var(data):.2f}")
print(f"标准差: {np.std(data):.2f}")
print()

# 分位数
percentiles = [25, 50, 75]
for p in percentiles:
    print(f"{p}% 分位数: {np.percentile(data, p):.2f}")

# 四分位距
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
print(f"\n四分位距 (IQR): {iqr:.2f}")

# 偏度和峰度（手动计算）
mean = np.mean(data)
std = np.std(data)
skewness = np.mean(((data - mean) / std) ** 3)
kurtosis = np.mean(((data - mean) / std) ** 4) - 3

print(f"\n偏度: {skewness:.4f} (正态分布为 0)")
print(f"峰度: {kurtosis:.4f} (正态分布为 0)")
```

### 协方差与相关系数

```python runnable
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 生成相关数据
n = 100
x = np.random.randn(n)
y = 0.8 * x + 0.2 * np.random.randn(n)  # y 与 x 正相关
z = np.random.randn(n)  # z 与 x 独立

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

## Monte Carlo 模拟

### Monte Carlo 方法思想

**Monte Carlo 方法（蒙特卡洛方法）**通过随机采样来近似求解复杂问题。核心思想：**用频率估计概率，用随机模拟逼近确定性问题**。

### 积分估计

计算积分 $\int_a^b f(x) dx$ 的 Monte Carlo 方法：

$$\int_a^b f(x) dx \approx \frac{b-a}{N} \sum_{i=1}^N f(x_i)$$

其中 $x_i$ 是 $[a, b]$ 上的均匀随机采样。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# Monte Carlo 积分估计
np.random.seed(42)

# 计算 ∫_0^1 sin(x) dx
# 真实值: 1 - cos(1) ≈ 0.4597

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

print(f"\n真实值: {true_value:.6f}")

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

### π 的估计

经典的 Monte Carlo 示例：用随机投点估计 π。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

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

print(f"采样点数: {n}")
print(f"落在圆内的点数: {np.sum(inside)}")
print(f"π 估计值: {pi_estimate:.6f}")
print(f"真实值: {np.pi:.6f}")
print(f"误差: {error:.6f}")

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

### 概率估计

Monte Carlo 方法可以估计复杂事件的概率。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 估计三个标准正态变量之和大于 3 的概率
np.random.seed(42)

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

# 理论值（S ~ N(0, 3)，所以 S/sqrt(3) ~ N(0,1)）
from math import erf, sqrt
z = 3 / sqrt(3)
prob_theory = 0.5 * (1 - erf(z / sqrt(2)))

print(f"P(X + Y + Z > 3) 的估计")
print(f"  Monte Carlo 估计: {prob_estimate:.6f}")
print(f"  理论值: {prob_theory:.6f}")
print(f"  误差: {abs(prob_estimate - prob_theory):.6f}")

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

### 贝叶斯后验采样

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 贝叶斯后验采样：估计硬币正面概率
np.random.seed(42)

# 真实参数
true_p = 0.6
n_flips = 50

# 生成观测数据
flips = np.random.binomial(1, true_p, n_flips)
n_heads = flips.sum()

print(f"观测数据: {n_flips} 次抛掷, {n_heads} 次正面")
print(f"MLE 估计: p̂ = {n_heads/n_flips:.3f}")
print()

# 使用拒绝采样从后验分布采样
# 先验: Beta(2, 2)
# 后验: Beta(2 + n_heads, 2 + n_flips - n_heads)

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
print(f"  后验均值: {posterior_samples.mean():.4f}")
print(f"  后验标准差: {posterior_samples.std():.4f}")
print(f"  95% 可信区间: [{np.percentile(posterior_samples, 2.5):.4f}, {np.percentile(posterior_samples, 97.5):.4f}]")

# 可视化
plt.figure(figsize=(10, 5))

plt.hist(posterior_samples, bins=50, density=True, alpha=0.7, 
         color='steelblue', edgecolor='black')

plt.axvline(true_p, color='green', linestyle='--', linewidth=2, label=f'真实值 p = {true_p}')
plt.axvline(n_heads/n_flips, color='red', linestyle=':', linewidth=2, label=f'MLE = {n_heads/n_flips:.3f}')
plt.axvline(posterior_samples.mean(), color='purple', linestyle='-.', linewidth=2, 
            label=f'后验均值 = {posterior_samples.mean():.3f}')

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

本章通过 NumPy 实践了概率统计的核心计算：

1. **随机数生成**：`np.random` 模块提供了丰富的分布采样功能。设置随机种子确保可复现性。

2. **统计量计算**：均值、方差、分位数、协方差、相关系数等描述性统计。

3. **Monte Carlo 方法**：通过随机采样近似求解积分、概率估计等复杂问题。

4. **贝叶斯后验采样**：从后验分布采样，得到参数的完整不确定性描述。

这些计算技能是数据分析和机器学习实践的基础。掌握 NumPy 的概率统计功能，能够高效地实现各种统计算法。

## 练习题

1. 使用 Monte Carlo 方法估计 $\int_0^1 x^2 dx$ 的值，并与真实值比较。
   <details>
   <summary>参考答案</summary>

   ```python
   import numpy as np
   np.random.seed(42)
   
   n = 100000
   x = np.random.uniform(0, 1, n)
   estimate = np.mean(x**2)
   
   true_value = 1/3
   
   print(f"Monte Carlo 估计: {estimate:.6f}")
   print(f"真实值: {true_value:.6f}")
   print(f"误差: {abs(estimate - true_value):.6f}")
   ```

   </details>

2. 编写函数计算数据集的偏度和峰度。
   <details>
   <summary>参考答案</summary>

   ```python
   import numpy as np
   
   def skewness(data):
       """计算偏度"""
       mean = np.mean(data)
       std = np.std(data)
       return np.mean(((data - mean) / std) ** 3)
   
   def kurtosis(data):
       """计算峰度（超额峰度）"""
       mean = np.mean(data)
       std = np.std(data)
       return np.mean(((data - mean) / std) ** 4) - 3
   
   # 测试
   np.random.seed(42)
   data = np.random.randn(10000)
   
   print(f"正态分布数据的偏度: {skewness(data):.4f}")
   print(f"正态分布数据的峰度: {kurtosis(data):.4f}")
   ```

   </details>

3. 使用 Monte Carlo 方法验证中心极限定理：从均匀分布采样，观察样本均值的分布。
   <details>
   <summary>参考答案</summary>

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   
   np.random.seed(42)
   
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
   
   print(f"样本均值的均值: {sample_means.mean():.4f} (理论: {theoretical_mean})")
   print(f"样本均值的标准差: {sample_means.std():.4f} (理论: {theoretical_std:.4f})")
   
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