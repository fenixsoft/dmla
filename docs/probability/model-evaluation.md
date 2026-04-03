---
title: "模型评估：偏差、方差与选择"
---

# 模型评估：偏差、方差与选择

在[第3章](statistical-inference.md)中，我们学习了如何从数据估计模型参数。但一个关键问题尚未回答：**如何评估模型的好坏？如何选择最合适的模型？**

本章介绍机器学习中最重要的评估框架——偏差-方差分解，以及交叉验证、模型选择准则等实用方法。这些内容是诊断过拟合/欠拟合、选择最优模型的基石。

## 偏差-方差分解

### 过拟合与欠拟合的困境

在机器学习中，我们经常遇到两种问题：

**欠拟合（Underfitting）**：模型太简单，无法捕捉数据中的规律。表现为训练误差和测试误差都很高。

**过拟合（Overfitting）**：模型太复杂，记住了训练数据中的噪声而非规律。表现为训练误差很低，但测试误差很高。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    模型复杂度与误差的关系                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   误差                                                                   │
│    ↑                                                                    │
│    │                        测试误差                                    │
│    │                      ↗                                            │
│    │                    /                                              │
│    │                  /     最优点                                     │
│    │                /       ↓                                         │
│    │   ────────────/────────────                                      │
│    │  训练误差   /                                                      │
│    │          ↘                                                        │
│    │            ↓                                                      │
│    └───────────────────────────────────────────→ 模型复杂度            │
│          欠拟合              适中              过拟合                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 偏差-方差分解公式

对于回归问题，期望预测误差可以分解为三部分：

$$E[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

其中：

- **偏差（Bias）**：模型预测的期望值与真实值之差
  $$\text{Bias}[\hat{f}(x)] = E[\hat{f}(x)] - f(x)$$

- **方差（Variance）**：模型预测的波动程度
  $$\text{Var}[\hat{f}(x)] = E[(\hat{f}(x) - E[\hat{f}(x)])^2]$$

- **不可约误差（Irreducible Error）**：数据本身的噪声 $\sigma^2$

### 直观理解

| 现象 | 偏差 | 方差 | 典型表现 |
|------|------|------|----------|
| 欠拟合 | 高 | 低 | 训练误差高，测试误差高 |
| 适中 | 低 | 低 | 训练误差低，测试误差低 |
| 过拟合 | 低 | 高 | 训练误差低，测试误差高 |

**偏差**衡量模型是否"正确"——预测的平均值是否接近真实值。

**方差**衡量模型是否"稳定"——不同训练集上的预测是否一致。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 偏差-方差分解演示
np.random.seed(42)

# 真实函数
def true_function(x):
    return np.sin(2 * np.pi * x)

# 生成数据
n_train = 20
n_datasets = 100  # 模拟多个训练集
x_test = np.linspace(0, 1, 100)
y_true = true_function(x_test)

# 不同复杂度的模型：多项式回归
degrees = [1, 3, 15]  # 简单、适中、复杂

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, degree in enumerate(degrees):
    predictions = []
    
    for _ in range(n_datasets):
        # 生成训练数据
        x_train = np.random.uniform(0, 1, n_train)
        y_train = true_function(x_train) + np.random.normal(0, 0.3, n_train)
        
        # 多项式拟合
        coeffs = np.polyfit(x_train, y_train, degree)
        y_pred = np.polyval(coeffs, x_test)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    
    # 计算偏差和方差
    mean_prediction = np.mean(predictions, axis=0)
    bias_squared = (mean_prediction - y_true) ** 2
    variance = np.var(predictions, axis=0)
    
    # 绘图
    ax = axes[idx]
    
    # 绘制多条预测曲线
    for i in range(20):
        ax.plot(x_test, predictions[i], 'b-', alpha=0.1, linewidth=0.5)
    
    # 平均预测
    ax.plot(x_test, mean_prediction, 'r-', linewidth=2, label='平均预测')
    
    # 真实函数
    ax.plot(x_test, y_true, 'g--', linewidth=2, label='真实函数')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'多项式阶数 = {degree}')
    ax.legend()
    ax.set_ylim(-2, 2)
    ax.grid(alpha=0.3)
    
    # 计算平均偏差²和方差
    avg_bias2 = np.mean(bias_squared)
    avg_var = np.mean(variance)
    
    ax.text(0.05, -1.7, f'偏差² = {avg_bias2:.3f}\n方差 = {avg_var:.3f}', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('偏差-方差分解：不同复杂度模型的预测', fontsize=14)
plt.tight_layout()
plt.show()
plt.close()

print("关键观察：")
print("1. 低复杂度模型（阶数=1）：高偏差、低方差 → 欠拟合")
print("2. 适中复杂度模型（阶数=3）：低偏差、低方差 → 最优")
print("3. 高复杂度模型（阶数=15）：低偏差、高方差 → 过拟合")
```

### 偏差-方差权衡

理想的模型应该同时具有低偏差和低方差。但实际上，这两者往往此消彼长：

- **简单模型**：偏差高（假设太强）、方差低（预测稳定）
- **复杂模型**：偏差低（假设弱）、方差高（预测不稳定）

找到最佳平衡点是机器学习模型选择的核心。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 偏差-方差权衡可视化
np.random.seed(42)

degrees = range(1, 16)
n_train = 50
n_datasets = 50

bias_squared_list = []
variance_list = []
total_error_list = []

x_test = np.linspace(0, 1, 100)
y_true = np.sin(2 * np.pi * x_test)
noise_var = 0.3 ** 2

for degree in degrees:
    predictions = []
    
    for _ in range(n_datasets):
        x_train = np.random.uniform(0, 1, n_train)
        y_train = np.sin(2 * np.pi * x_train) + np.random.normal(0, 0.3, n_train)
        
        try:
            coeffs = np.polyfit(x_train, y_train, degree)
            y_pred = np.polyval(coeffs, x_test)
            predictions.append(y_pred)
        except:
            predictions.append(np.zeros_like(x_test))
    
    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    
    bias2 = np.mean((mean_prediction - y_true) ** 2)
    var = np.mean(np.var(predictions, axis=0))
    
    bias_squared_list.append(bias2)
    variance_list.append(var)
    total_error_list.append(bias2 + var + noise_var)

# 可视化
plt.figure(figsize=(10, 6))

plt.plot(degrees, bias_squared_list, 'o-', label='偏差²', linewidth=2, markersize=6)
plt.plot(degrees, variance_list, 's-', label='方差', linewidth=2, markersize=6)
plt.plot(degrees, total_error_list, '^-', label='总误差', linewidth=2, markersize=6)
plt.axhline(noise_var, color='gray', linestyle='--', label=f'噪声方差 = {noise_var:.2f}')

# 找到最优点
optimal_degree = list(degrees)[np.argmin(total_error_list)]
plt.axvline(optimal_degree, color='red', linestyle=':', linewidth=2, 
            label=f'最优复杂度 = {optimal_degree}')

plt.xlabel('模型复杂度（多项式阶数）')
plt.ylabel('误差')
plt.title('偏差-方差权衡')
plt.legend()
plt.grid(alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.show()
plt.close()

print(f"最优模型复杂度: 多项式阶数 = {optimal_degree}")
```

### 与正则化的联系

在[线性代数](../linear/matrices.md)中我们学过正则化。从偏差-方差视角理解：

- **L1/L2 正则化**：限制模型复杂度 → 增加偏差、降低方差 → 防止过拟合
- **正则化强度**：控制偏差-方差权衡的位置

## 交叉验证方法

### 为什么需要交叉验证？

如果只用一次训练-测试划分，评估结果可能受划分方式影响。**交叉验证（Cross-Validation）**通过多次划分得到更稳定的评估。

### K 折交叉验证

将数据分成 K 份，每次用 K-1 份训练，1 份测试，重复 K 次。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# K 折交叉验证实现
def k_fold_split(n_samples, k=5, shuffle=True, random_state=None):
    """生成 K 折交叉验证的索引"""
    if random_state:
        np.random.seed(random_state)
    
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1  # 处理不整除的情况
    
    folds = []
    current = 0
    for fold_size in fold_sizes:
        test_indices = indices[current:current + fold_size]
        train_indices = np.concatenate([indices[:current], indices[current + fold_size:]])
        folds.append((train_indices, test_indices))
        current += fold_size
    
    return folds

# 演示
n_samples = 20
k = 5

folds = k_fold_split(n_samples, k, random_state=42)

print(f"K 折交叉验证 (n={n_samples}, K={k})")
print("-" * 50)

for i, (train_idx, test_idx) in enumerate(folds):
    print(f"第 {i+1} 折: 训练集大小={len(train_idx)}, 测试集大小={len(test_idx)}")
    print(f"       测试索引: {test_idx}")

# 可视化
fig, ax = plt.subplots(figsize=(12, 4))

colors = plt.cm.tab10(np.linspace(0, 1, k))

for i, (train_idx, test_idx) in enumerate(folds):
    for j in range(n_samples):
        if j in test_idx:
            ax.scatter(j, i, c=[colors[i]], s=100, marker='s', edgecolor='black')
        else:
            ax.scatter(j, i, c='lightgray', s=100, marker='s', edgecolor='gray')

ax.set_xlabel('样本索引')
ax.set_ylabel('折数')
ax.set_title(f'{k} 折交叉验证数据划分')
ax.set_yticks(range(k))
ax.set_yticklabels([f'第 {i+1} 折' for i in range(k)])
ax.set_xticks(range(n_samples))
ax.grid(alpha=0.3, axis='x')

# 图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='lightgray', edgecolor='gray', label='训练集'),
                   Patch(facecolor=colors[0], edgecolor='black', label='测试集')]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()
plt.close()
```

### 使用交叉验证进行模型选择

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 使用交叉验证选择多项式阶数
np.random.seed(42)

# 生成数据
n_samples = 100
X = np.random.uniform(0, 1, n_samples)
y_true = np.sin(2 * np.pi * X)
y = y_true + np.random.normal(0, 0.3, n_samples)

# 交叉验证评估不同复杂度
degrees = range(1, 16)
k = 5

folds = k_fold_split(n_samples, k, random_state=42)

cv_errors = []

for degree in degrees:
    fold_errors = []
    
    for train_idx, test_idx in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 多项式拟合
        coeffs = np.polyfit(X_train, y_train, degree)
        y_pred = np.polyval(coeffs, X_test)
        
        # 测试误差
        mse = np.mean((y_pred - y_test) ** 2)
        fold_errors.append(mse)
    
    cv_errors.append(np.mean(fold_errors))

# 最优阶数
optimal_degree = list(degrees)[np.argmin(cv_errors)]

# 可视化
plt.figure(figsize=(10, 5))

plt.plot(degrees, cv_errors, 'o-', linewidth=2, markersize=8)
plt.axvline(optimal_degree, color='r', linestyle='--', linewidth=2, 
            label=f'最优阶数 = {optimal_degree}')

plt.xlabel('多项式阶数')
plt.ylabel('交叉验证误差 (MSE)')
plt.title('K 折交叉验证选择模型复杂度')
plt.legend()
plt.grid(alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()
plt.close()

print(f"交叉验证选择的最佳多项式阶数: {optimal_degree}")
print(f"对应的交叉验证误差: {min(cv_errors):.4f}")
```

### 留一交叉验证（LOOCV）

当 K = n（样本数）时，每次只留一个样本做测试。计算量大但估计最准确。

```python
# LOOCV 是 K 折交叉验证的特例
# K = n_samples
# 适用于小样本情况
```

## 模型选择准则

### AIC 准则

**赤池信息准则（Akaike Information Criterion, AIC）**：

$$AIC = 2k - 2\ln(\hat{L})$$

其中 $k$ 是参数数量，$\hat{L}$ 是最大似然值。AIC 越小越好。

AIC 惩罚模型复杂度，在拟合质量和复杂度之间权衡。

### BIC 准则

**贝叶斯信息准则（Bayesian Information Criterion, BIC）**：

$$BIC = k\ln(n) - 2\ln(\hat{L})$$

BIC 对复杂度的惩罚更强（$\ln(n) > 2$ 当 $n > 7$）。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# AIC/BIC 模型选择示例
np.random.seed(42)

n = 100
X = np.random.uniform(0, 1, n)
y_true = np.sin(2 * np.pi * X)
y = y_true + np.random.normal(0, 0.3, n)

degrees = range(1, 16)
aic_values = []
bic_values = []

sigma2_estimate = 0.3 ** 2  # 噪声方差估计

for degree in degrees:
    # 多项式拟合
    coeffs = np.polyfit(X, y, degree)
    y_pred = np.polyval(coeffs, X)
    
    # 残差
    residuals = y - y_pred
    rss = np.sum(residuals ** 2)
    
    # 参数数量（系数 + 可能的方差参数）
    k = degree + 1  # 多项式系数数量
    
    # 对数似然（假设正态分布）
    log_likelihood = -n/2 * np.log(2*np.pi) - n/2 * np.log(sigma2_estimate) - rss / (2 * sigma2_estimate)
    
    # AIC 和 BIC
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    
    aic_values.append(aic)
    bic_values.append(bic)

# 最优模型
best_aic = list(degrees)[np.argmin(aic_values)]
best_bic = list(degrees)[np.argmin(bic_values)]

# 可视化
plt.figure(figsize=(10, 5))

plt.plot(degrees, aic_values, 'o-', label='AIC', linewidth=2, markersize=6)
plt.plot(degrees, bic_values, 's-', label='BIC', linewidth=2, markersize=6)

plt.axvline(best_aic, color='blue', linestyle='--', alpha=0.7, label=f'AIC 最优 = {best_aic}')
plt.axvline(best_bic, color='orange', linestyle='--', alpha=0.7, label=f'BIC 最优 = {best_bic}')

plt.xlabel('多项式阶数')
plt.ylabel('信息准则值')
plt.title('AIC vs BIC 模型选择')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print(f"AIC 选择的最优阶数: {best_aic}")
print(f"BIC 选择的最优阶数: {best_bic}")
print("\n注意: BIC 对复杂度惩罚更强，倾向于选择更简单的模型")
```

## 泛化误差的置信区间

### 泛化误差估计

当我们评估模型时，关心的是在所有未见数据上的表现（泛化误差）。但只能通过测试集估计。

### 测试误差的置信区间

假设测试集有 $n$ 个样本，准确率为 $p$。根据二项分布，准确率的 95% 置信区间约为：

$$p \pm 1.96\sqrt{\frac{p(1-p)}{n}}$$

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 模型性能的置信区间
def accuracy_confidence_interval(accuracy, n, confidence=0.95):
    """计算准确率的置信区间"""
    z = 1.96 if confidence == 0.95 else 2.576  # 95% 或 99%
    margin = z * np.sqrt(accuracy * (1 - accuracy) / n)
    return accuracy - margin, accuracy + margin

# 不同测试集大小的置信区间
test_sizes = [30, 100, 500, 1000, 5000]
accuracy = 0.85

plt.figure(figsize=(10, 5))

for i, n in enumerate(test_sizes):
    lower, upper = accuracy_confidence_interval(accuracy, n)
    
    plt.errorbar(i, accuracy, yerr=[[accuracy-lower], [upper-accuracy]], 
                 fmt='o', markersize=10, capsize=5, capthick=2, linewidth=2)
    
    print(f"测试集大小 n={n}: 准确率 {accuracy:.1%} 的 95% CI = [{lower:.1%}, {upper:.1%}]")

plt.axhline(accuracy, color='r', linestyle='--', label=f'准确率 = {accuracy:.0%}')
plt.xticks(range(len(test_sizes)), [f'n={n}' for n in test_sizes])
plt.xlabel('测试集大小')
plt.ylabel('准确率')
plt.title('模型准确率的 95% 置信区间')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.ylim(0.7, 1.0)

plt.tight_layout()
plt.show()
plt.close()

print("\n关键洞察：")
print("1. 测试集越小，置信区间越宽（不确定性越大）")
print("2. 要达到 ±1% 的精度，需要约 5000 个测试样本")
print("3. 报告模型性能时，应同时报告置信区间")
```

### 比较两个模型

当比较两个模型的性能时，需要判断差异是否具有统计显著性。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# McNemar 检验的思想（简化版）
# 比较两个模型是否有显著差异

np.random.seed(42)

n_test = 100

# 模型 A 和 B 的预测
y_true = np.random.randint(0, 2, n_test)
y_pred_a = np.random.randint(0, 2, n_test)
y_pred_b = np.random.randint(0, 2, n_test)

# 为了演示，让模型 A 稍好
y_pred_a[:60] = y_true[:60]  # 模型 A 正确 60%
y_pred_b[:50] = y_true[:50]  # 模型 B 正确 50%

# 计算准确率
acc_a = np.mean(y_pred_a == y_true)
acc_b = np.mean(y_pred_b == y_true)

# 置信区间
ci_a = accuracy_confidence_interval(acc_a, n_test)
ci_b = accuracy_confidence_interval(acc_b, n_test)

# 可视化
plt.figure(figsize=(8, 5))

x = [0, 1]
accuracies = [acc_a, acc_b]
cis = [ci_a, ci_b]
labels = ['模型 A', '模型 B']
colors = ['steelblue', 'orange']

for i, (acc, ci, label, color) in enumerate(zip(accuracies, cis, labels, colors)):
    plt.errorbar(i, acc, yerr=[[acc-ci[0]], [ci[1]-acc]], 
                 fmt='o', markersize=12, capsize=8, capthick=2, 
                 linewidth=2, color=color, label=label)

plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='随机猜测')
plt.xticks(x, labels)
plt.ylabel('准确率')
plt.title(f'模型比较 (n={n_test})')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.ylim(0.3, 0.8)

plt.tight_layout()
plt.show()
plt.close()

# 判断置信区间是否重叠
if ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]:
    print("结论: 置信区间不重叠，差异具有统计显著性")
else:
    print("结论: 置信区间重叠，差异不具有统计显著性")
    print("       需要更多测试数据或使用更敏感的统计检验")

print(f"\n模型 A: 准确率 = {acc_a:.1%}, 95% CI = [{ci_a[0]:.1%}, {ci_a[1]:.1%}]")
print(f"模型 B: 准确率 = {acc_b:.1%}, 95% CI = [{ci_b[0]:.1%}, {ci_b[1]:.1%}]")
```

## 本章小结

本章介绍了模型评估的核心方法：

1. **偏差-方差分解**揭示了过拟合/欠拟合的本质。偏差衡量"正确性"，方差衡量"稳定性"。理想模型应同时具有低偏差和低方差。

2. **交叉验证**通过多次评估得到更稳定的结果。K 折交叉验证是最常用的方法。

3. **模型选择准则**（AIC/BIC）在拟合质量和复杂度之间权衡。BIC 对复杂度惩罚更强。

4. **置信区间**量化模型性能评估的不确定性。报告准确率时应同时报告置信区间。

这些方法是机器学习实践中的必备工具，帮助我们在有限的评估数据上做出可靠的决策。

## 练习题

1. 为什么增加模型复杂度通常会降低偏差但增加方差？
   <details>
   <summary>参考答案</summary>

   复杂模型有更强的表达能力，能够更准确地拟合训练数据，因此偏差降低。但复杂模型也更"敏感"——不同训练集可能学到不同的模式，导致预测波动大，方差增加。

   类比：简单的线性模型（低复杂度）对数据变化不敏感，预测稳定但可能不准；高阶多项式（高复杂度）可以完美拟合任何数据，但不同数据得到的拟合曲线差异很大。

   </details>

2. K 折交叉验证中，K 的选择如何影响偏差和方差？
   <details>
   <summary>参考答案</summary>

   - K 越小（如 2-5）：训练集占总数据的比例较小，评估结果的偏差可能较高（低估模型性能）
   - K 越大（如 n，即 LOOCV）：训练集几乎用全部数据，偏差低但计算量大
   - 方面：K 越大，每次训练的训练集越相似，评估结果的方差可能更高

   实践中，K=5 或 K=10 是常用的折中方案。

   </details>

3. 为什么 BIC 比 AIC 更倾向于选择简单的模型？
   <details>
   <summary>参考答案</summary>

   比较 AIC 和 BIC 对参数数量的惩罚：
   - AIC 惩罚：$2k$
   - BIC 惩罚：$k\ln(n)$

   当样本量 $n > 7$ 时，$\ln(n) > 2$，BIC 对每个额外参数的惩罚更重。因此 BIC 更倾向于选择参数较少的简单模型。

   这反映了不同的目标：AIC 优化预测性能，BIC 倾向于选择"真实"模型。

   </details>