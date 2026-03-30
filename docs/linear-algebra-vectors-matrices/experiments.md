---
title: "实验与练习"
issue:
  number: 22
---

# 实验与练习

本文档包含向量与矩阵运算基础章节的实验代码和练习题。每个实验代码都可以在沙箱环境中运行验证。

---

## 实验1: 向量运算可视化

**目标**: 通过图形直观理解向量加法、数乘的几何意义

**验证知识点**:
- 向量加法的几何意义（平行四边形法则）
- 向量数乘的几何意义（伸缩变换）
- 向量在二维平面中的可视化表示

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义两个向量
v1 = np.array([2, 3])
v2 = np.array([1, -1])

# 向量加法
v_sum = v1 + v2

# 向量数乘
v_scaled = 2 * v1

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 子图1: 向量加法
ax1 = axes[0]
ax1.set_xlim(-1, 5)
ax1.set_ylim(-2, 6)
ax1.grid(True, alpha=0.3)
ax1.set_title('向量加法: v1 + v2', fontsize=14)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# 绘制向量（使用箭头）
ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.02, label='v1 = [2, 3]')
ax1.quiver(v1[0], v1[1], v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
           color='green', width=0.02, label='v2 = [1, -1]')
ax1.quiver(0, 0, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1,
           color='red', width=0.02, label='v1 + v2 = [3, 2]')

ax1.legend(loc='upper left')

# 子图2: 向量数乘
ax2 = axes[1]
ax2.set_xlim(-1, 6)
ax2.set_ylim(-1, 8)
ax2.grid(True, alpha=0.3)
ax2.set_title('向量数乘: 2 * v1', fontsize=14)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

ax2.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.02, label='v1 = [2, 3]')
ax2.quiver(0, 0, v_scaled[0], v_scaled[1], angles='xy', scale_units='xy', scale=1,
           color='purple', width=0.02, label='2 * v1 = [4, 6]')

ax2.legend(loc='upper left')

plt.tight_layout()
print("向量加法结果:", v_sum)
print("向量数乘结果:", v_scaled)
print("图形已生成，展示了向量运算的几何意义")
```

**说明**:
- 左图展示向量加法：`v1 + v2 = [3, 2]`，验证了平行四边形法则
- 右图展示向量数乘：`2 * v1 = [4, 6]`，验证了数乘使向量沿原方向伸缩
- 从几何视角理解，向量加法相当于从原点先走 `v1`，再走 `v2`

---

## 实验2: 矩阵变换可视化

**目标**: 观察线性变换如何改变空间，理解矩阵乘法的几何意义

**验证知识点**:
- 旋转矩阵的几何效果
- 缩放矩阵的几何效果
- 剪切矩阵的几何效果
- 线性变换保持网格线平行和等距分布

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 设置图形参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_transformation(ax, matrix, title):
    """绘制线性变换前后的图形"""
    # 创建单位正方形和网格
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T

    # 应用变换
    transformed = matrix @ square

    # 绘制原始图形（虚线）
    ax.plot(square[0], square[1], 'b--', alpha=0.5, label='原始')

    # 绘制变换后图形（实线）
    ax.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='变换后')

    # 绘制基向量变换
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    te1 = matrix @ e1
    te2 = matrix @ e2

    ax.quiver(0, 0, te1[0], te1[1], angles='xy', scale_units='xy', scale=1,
              color='green', width=0.03, label=f"i' = [{te1[0]:.2f}, {te1[1]:.2f}]")
    ax.quiver(0, 0, te2[0], te2[1], angles='xy', scale_units='xy', scale=1,
              color='orange', width=0.03, label=f"j' = [{te2[0]:.2f}, {te2[1]:.2f}]")

    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')

# 定义变换矩阵
theta = np.pi / 4  # 45度旋转
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

scale_matrix = np.array([
    [2, 0],
    [0, 0.5]
])

shear_matrix = np.array([
    [1, 0.5],
    [0, 1]
])

# 创建图形
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 绘制三种变换
plot_transformation(axes[0], rotation_matrix, '旋转变换 (45度)')
plot_transformation(axes[1], scale_matrix, '缩放变换 (x2, x0.5)')
plot_transformation(axes[2], shear_matrix, '剪切变换')

plt.tight_layout()

# 打印变换矩阵
print("旋转变换矩阵 (45度):")
print(rotation_matrix)
print(f"\n缩放变换矩阵:")
print(scale_matrix)
print(f"\n剪切变换矩阵:")
print(shear_matrix)
print("\n观察: 线性变换后，网格线仍然保持平行且等距分布")
```

**说明**:
- **旋转变换**: 使空间绕原点旋转，矩阵的列向量是变换后的基向量
- **缩放变换**: 沿坐标轴方向拉伸或压缩空间
- **剪切变换**: 使空间沿某一方向倾斜
- 核心洞察：矩阵的列向量就是变换后的基向量 `i'` 和 `j'`

---

## 实验3: NumPy 性能对比

**目标**: 理解为什么机器学习需要向量化运算

**验证知识点**:
- 向量化运算与循环的性能差异
- 大规模数据下向量化的优势
- NumPy 底层优化原理

```python runnable
import numpy as np
import time

def loop_based_operation(a, b):
    """使用循环实现的向量点积"""
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def vectorized_operation(a, b):
    """使用向量化实现的向量点积"""
    return np.dot(a, b)

# 测试不同规模的性能
sizes = [100, 10000, 100000, 1000000]
results = []

print("NumPy 向量化 vs 循环 性能对比")
print("=" * 60)
print(f"{'数据规模':<12} {'循环耗时(ms)':<15} {'向量化耗时(ms)':<15} {'加速比':<10}")
print("-" * 60)

for size in sizes:
    # 生成随机数据
    a = np.random.randn(size)
    b = np.random.randn(size)

    # 循环版本计时
    start = time.time()
    result_loop = loop_based_operation(a, b)
    loop_time = (time.time() - start) * 1000  # 转换为毫秒

    # 向量化版本计时
    start = time.time()
    result_vec = vectorized_operation(a, b)
    vec_time = (time.time() - start) * 1000

    # 计算加速比
    speedup = loop_time / vec_time if vec_time > 0 else float('inf')

    print(f"{size:<12} {loop_time:<15.3f} {vec_time:<15.3f} {speedup:<10.1f}x")

    results.append({
        'size': size,
        'loop_time': loop_time,
        'vec_time': vec_time,
        'speedup': speedup
    })

print("=" * 60)
print("\n结论: 数据规模越大，向量化的优势越明显")
print("原因: NumPy 使用 C 语言实现，避免了 Python 循环的开销")
```

**说明**:
- 循环版本每次迭代都有 Python 解释器开销
- 向量化版本利用 NumPy 的 C 语言底层实现，批量处理数据
- 在机器学习中，处理数百万甚至数亿数据点时，这种差异是决定性的
- 这也是为什么深度学习框架（如 TensorFlow、PyTorch）都基于张量运算

---

## 练习1: 计算两个向量的内积和夹角

**题目**: 给定两个向量 `a = [1, 2, 3]` 和 `b = [4, 5, 6]`，计算：
1. 它们的内积（点积）
2. 它们的夹角（用角度表示）

**提示**:
- 内积公式：`a · b = |a| |b| cos(θ)`
- 夹角公式：`θ = arccos(a · b / (|a| |b|))`
- 使用 `np.arccos()` 计算反余弦，然后用 `np.degrees()` 转换为角度

```python runnable
import numpy as np

# 定义两个向量
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 计算内积
dot_product = np.dot(a, b)

# 计算向量范数（模长）
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)

# 计算夹角（弧度）
cos_theta = dot_product / (norm_a * norm_b)
theta_rad = np.arccos(cos_theta)

# 转换为角度
theta_deg = np.degrees(theta_rad)

print(f"向量 a = {a}")
print(f"向量 b = {b}")
print(f"内积 a · b = {dot_product}")
print(f"|a| = {norm_a:.4f}")
print(f"|b| = {norm_b:.4f}")
print(f"夹角 θ = {theta_deg:.2f} 度")
```

---

## 练习2: 实现矩阵乘法（不使用 NumPy 的 matmul）

**题目**: 不使用 `np.matmul()` 或 `@` 运算符，实现两个矩阵的乘法。

**要求**:
- 实现函数 `matrix_multiply(A, B)` 返回结果矩阵
- 验证 `(m, n)` 矩阵和 `(n, p)` 矩阵相乘得到 `(m, p)` 矩阵
- 使用三重循环实现，理解矩阵乘法的本质

**提示**:
- 结果矩阵 C 的元素：`C[i][j] = sum(A[i][k] * B[k][j] for k in range(n))`

```python runnable
import numpy as np

def matrix_multiply(A, B):
    """
    实现矩阵乘法（不使用 NumPy 的 matmul）

    参数:
        A: m x n 矩阵
        B: n x p 矩阵
    返回:
        C: m x p 矩阵
    """
    m = len(A)      # A 的行数
    n = len(A[0])   # A 的列数（也是 B 的行数）
    p = len(B[0])   # B 的列数

    # 创建结果矩阵
    C = [[0 for _ in range(p)] for _ in range(m)]

    # 三重循环实现矩阵乘法
    for i in range(m):          # 遍历结果矩阵的行
        for j in range(p):      # 遍历结果矩阵的列
            for k in range(n):  # 计算点积
                C[i][j] += A[i][k] * B[k][j]

    return C

# 测试
A = [[1, 2, 3],
     [4, 5, 6]]

B = [[7, 8],
     [9, 10],
     [11, 12]]

# 自定义实现
result_custom = matrix_multiply(A, B)

# NumPy 验证
A_np = np.array(A)
B_np = np.array(B)
result_numpy = A_np @ B_np

print("矩阵 A (2x3):")
print(np.array(A))
print("\n矩阵 B (3x2):")
print(np.array(B))
print("\n自定义实现结果 (2x2):")
print(np.array(result_custom))
print("\nNumPy 验证结果 (2x2):")
print(result_numpy)
print("\n结果一致:", np.allclose(result_custom, result_numpy))
```

---

## 练习3: 使用 NumPy 实现数据标准化

**题目**: 实现两种常见的数据标准化方法：
1. **Z-score 标准化**：`x' = (x - mean) / std`
2. **Min-Max 归一化**：`x' = (x - min) / (max - min)`

**要求**:
- 对给定数据矩阵的每列进行标准化
- 处理多维数据时使用 `axis` 参数
- 返回标准化后的数据

**提示**:
- 使用 `np.mean()` 和 `np.std()` 计算均值和标准差
- 使用 `np.min()` 和 `np.max()` 计算最小值和最大值
- `axis=0` 表示沿列方向操作

```python runnable
import numpy as np

def zscore_normalize(X):
    """
    Z-score 标准化
    使数据均值为 0，标准差为 1
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # 避免除零
    std = np.where(std == 0, 1, std)
    return (X - mean) / std

def minmax_normalize(X):
    """
    Min-Max 归一化
    将数据缩放到 [0, 1] 区间
    """
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    # 避免除零
    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1, range_val)
    return (X - min_val) / range_val

# 测试数据：4 个样本，3 个特征
X = np.array([
    [100, 0.001, 50000],
    [200, 0.005, 30000],
    [150, 0.003, 40000],
    [300, 0.008, 20000]
])

print("原始数据:")
print(X)
print("\n各列统计:")
print(f"均值: {np.mean(X, axis=0)}")
print(f"标准差: {np.std(X, axis=0)}")
print(f"最小值: {np.min(X, axis=0)}")
print(f"最大值: {np.max(X, axis=0)}")

# Z-score 标准化
X_zscore = zscore_normalize(X)
print("\nZ-score 标准化后:")
print(X_zscore)
print(f"均值: {np.mean(X_zscore, axis=0)}")
print(f"标准差: {np.std(X_zscore, axis=0)}")

# Min-Max 归一化
X_minmax = minmax_normalize(X)
print("\nMin-Max 归一化后:")
print(X_minmax)
print(f"最小值: {np.min(X_minmax, axis=0)}")
print(f"最大值: {np.max(X_minmax, axis=0)}")
```

---

## 沙箱验证状态

| 实验 | 状态 | 备注 |
|------|------|------|
| 实验1: 向量运算可视化 | 待验证 | 需要 Docker 环境 |
| 实验2: 矩阵变换可视化 | 待验证 | 需要 Docker 环境 |
| 实验3: NumPy 性能对比 | 待验证 | 需要 Docker 环境 |
| 练习1: 内积和夹角 | 待验证 | 需要 Docker 环境 |
| 练习2: 矩阵乘法实现 | 待验证 | 需要 Docker 环境 |
| 练习3: 数据标准化 | 待验证 | 需要 Docker 环境 |

**验证命令**:
```bash
curl -X POST http://localhost:3001/api/sandbox/run \
  -H "Content-Type: application/json" \
  -d '{"code": "你的代码", "useGPU": false}'
```