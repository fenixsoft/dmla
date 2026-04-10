# 核技巧——非线性问题的线性解法

## 引言：线性模型的边界

上一章我们学习了SVM的最大间隔原理，但有一个隐含假设：**数据线性可分**（或近似线性可分）。当数据分布呈现复杂的非线性模式时，线性分类器就无能为力了。

```
    ★★★                  ★★★
   ★    ★              ★    ★
  ★  ☆☆☆  ★            ★  ★★★  ★
  ★  ☆   ☆  ★    →     ★  ★   ★  ★
   ★ ☆☆☆ ★              ★  ★★★  ★
     ★★★                  ★    ★
                           ★★★
                           
   同心圆数据              线性分类器无法分离
```

**核技巧（Kernel Trick）**提供了一种优雅的解决方案：**将数据映射到高维空间，在高维空间中线性可分**。更妙的是，我们可以"隐式"地完成这个映射，而不需要显式计算高维特征。

---

## 特征空间映射：从低维到高维

### 基本思想

假设有一个映射函数 $\phi: \mathbb{R}^d \rightarrow \mathbb{R}^D$，将原始特征映射到更高维的空间：

$$x \mapsto \phi(x)$$

在高维空间中，原本线性不可分的数据可能变得线性可分。

### 示例：二次映射

考虑一维数据 $x \in \mathbb{R}$，正负样本交替分布：

```
位置: -2  -1   0   1   2
类别:  ☆   ★   ☆   ★   ☆
```

在一维空间中无法线性分离。但如果映射到二维：$\phi(x) = (x, x^2)$

```
      |         ★
   x² |    ☆         ★
      |         ☆
      |    ☆         ★
      +----------------
           x
```

在二维空间中，我们可以用一条直线分离！

### 映射的代价

映射到高维空间虽然可行，但有两个问题：
1. **维度爆炸**：$D$ 可能非常大甚至无穷
2. **计算成本**：计算 $\phi(x)^T \phi(x')$ 需要先映射再内积

**核技巧的核心洞察**：我们不需要显式计算 $\phi(x)$，只需要计算内积 $\phi(x)^T \phi(x')$，而这可以通过**核函数**直接计算！

---

## 核技巧：隐式内积计算

### 核函数定义

**核函数** $k(x, x')$ 定义为：

$$k(x, x') = \phi(x)^T \phi(x')$$

核函数直接计算两个样本在特征空间的内积，无需显式表示 $\phi(x)$。

### Mercer定理

一个函数 $k$ 能成为有效核函数，当且仅当对于任意数据集 $\{x_1, \ldots, x_n\}$，核矩阵 $K$ 是半正定的：

$$K_{ij} = k(x_i, x_j)$$

这保证了核函数对应某个特征空间的内积。

### SVM中的核化

回顾SVM的对偶问题：

$$\max_\alpha \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \underbrace{x_i^T x_j}_{\text{替换为核函数}}$$

只需将 $x_i^T x_j$ 替换为 $k(x_i, x_j)$，SVM就能处理非线性问题！

决策函数也变成：

$$f(x) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i k(x_i, x) + b\right)$$

---

## 常见核函数

### 线性核

$$k(x, x') = x^T x'$$

- 不做任何映射，就是原始线性SVM
- 适用于线性可分数据
- 计算最快

### 多项式核

$$k(x, x') = (x^T x' + c)^d$$

- $d$：多项式次数
- $c$：常数项（通常为0或1）
- 对应特征空间包含所有 $d$ 次及以下的特征组合

**示例**：对于 $x = (x_1, x_2)$，$d=2$，核函数对应特征：

$$\phi(x) = (1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, \sqrt{2}x_1 x_2, x_2^2)$$

### RBF核（高斯核）

$$k(x, x') = \exp\left(-\frac{||x - x'||^2}{2\sigma^2}\right) = \exp(-\gamma ||x - x'||^2)$$

- $\gamma = \frac{1}{2\sigma^2}$：核参数
- 对应**无穷维**特征空间
- 是最常用的核函数
- $\gamma$ 大：每个点影响范围小，模型复杂
- $\gamma$ 小：每个点影响范围大，模型简单

### 核函数对比

| 核函数 | 参数 | 特征空间维度 | 适用场景 |
|--------|------|-------------|----------|
| 线性核 | 无 | 原始维度 | 线性可分、高维稀疏 |
| 多项式核 | $d, c$ | $\binom{n+d}{d}$ | 特征交互重要 |
| RBF核 | $\gamma$ | 无穷维 | 通用非线性 |

---

## NumPy实现：核SVM

```python
import numpy as np

class KernelSVM:
    """
    核SVM实现
    支持线性核、多项式核、RBF核
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma=1.0, degree=3, coef0=1):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0  # 多项式核的常数项
        
        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.alpha_sv = None
    
    def _kernel(self, X1, X2):
        """计算核矩阵"""
        if self.kernel == 'linear':
            return X1 @ X2.T
        
        elif self.kernel == 'poly':
            return (X1 @ X2.T + self.coef0) ** self.degree
        
        elif self.kernel == 'rbf':
            # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2*x^T*x'
            X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            distances = X1_norm + X2_norm - 2 * X1 @ X2.T
            return np.exp(-self.gamma * distances)
        
        else:
            raise ValueError(f"未知核函数: {self.kernel}")
    
    def fit(self, X, y, lr=0.01, n_iterations=500):
        """训练模型（简化版SMO思想）"""
        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        
        # 计算核矩阵
        K = self._kernel(X, X)
        
        # 初始化
        self.alpha = np.zeros(n_samples)
        
        # 梯度上升优化
        for _ in range(n_iterations):
            for i in range(n_samples):
                # 梯度
                gradient = 1 - y[i] * np.sum(self.alpha * y * K[:, i])
                self.alpha[i] += lr * gradient
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)
            
            # 约束修正
            self.alpha = self.alpha - np.mean(self.alpha * y) * y
            self.alpha = np.clip(self.alpha, 0, self.C)
        
        # 支持向量
        sv_mask = self.alpha > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.support_vector_labels_ = y[sv_mask]
        self.alpha_sv = self.alpha[sv_mask]
        
        # 计算b
        if len(self.support_vectors_) > 0:
            K_sv = self._kernel(self.support_vectors_, self.support_vectors_)
            margins = np.sum(self.alpha_sv * self.support_vector_labels_ * K_sv, axis=1)
            self.b = np.mean(self.support_vector_labels_ - margins)
        else:
            self.b = 0
        
        return self
    
    def decision_function(self, X):
        """决策函数"""
        K = self._kernel(X, self.support_vectors_)
        return K @ (self.alpha_sv * self.support_vector_labels_) + self.b
    
    def predict(self, X):
        """预测类别"""
        return np.sign(self.decision_function(X)).astype(int)
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# 测试：非线性可分数据（同心圆）
np.random.seed(42)

def make_circles(n_samples=200, noise=0.1, factor=0.5):
    """生成同心圆数据"""
    n = n_samples // 2
    
    # 内圆
    theta_inner = np.random.uniform(0, 2*np.pi, n)
    r_inner = factor * np.random.uniform(0.8, 1.2, n)
    X_inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    
    # 外圆
    theta_outer = np.random.uniform(0, 2*np.pi, n)
    r_outer = np.random.uniform(0.8, 1.2, n)
    X_outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])
    
    X = np.vstack([X_inner, X_outer])
    y = np.hstack([-np.ones(n), np.ones(n)])
    
    # 添加噪声
    X += np.random.randn(*X.shape) * noise
    
    return X, y

X, y = make_circles(n_samples=200, noise=0.1)

# 对比不同核函数
print("=== 不同核函数对比（同心圆数据）===\n")

kernels = [
    ('linear', {}),
    ('poly', {'degree': 2}),
    ('rbf', {'gamma': 1.0})
]

for kernel_name, params in kernels:
    svm = KernelSVM(kernel=kernel_name, C=1.0, **params)
    svm.fit(X, y, lr=0.01, n_iterations=300)
    acc = svm.score(X, y)
    print(f"{kernel_name:8}核: 准确率 = {acc:.3f}, 支持向量数 = {len(svm.support_vectors_)}")
```

**输出示例：**
```
=== 不同核函数对比（同心圆数据）===

linear  核: 准确率 = 0.500, 支持向量数 = 100
poly    核: 准确率 = 0.985, 支持向量数 = 25
rbf     核: 准确率 = 0.990, 支持向量数 = 18
```

---

## 应用场景：非线性分类可视化

```python
import matplotlib.pyplot as plt

# 生成数据
X, y = make_circles(n_samples=200, noise=0.15)

# 训练RBF核SVM
svm_rbf = KernelSVM(kernel='rbf', C=1.0, gamma=2.0)
svm_rbf.fit(X, y)

# 绘制决策边界
plt.figure(figsize=(10, 8))

# 创建网格
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = svm_rbf.decision_function(grid).reshape(xx.shape)

# 绘制等高线
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Blues', alpha=0.5)
plt.contourf(xx, yy, Z, levels=np.linspace(0, Z.max(), 7), cmap='Reds', alpha=0.5)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

# 绘制数据点
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', marker='o', label='类别 -1', alpha=0.7)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='^', label='类别 +1', alpha=0.7)

# 绘制支持向量
plt.scatter(svm_rbf.support_vectors_[:, 0], svm_rbf.support_vectors_[:, 1], 
            s=100, facecolors='none', edgecolors='green', linewidths=2, label='支持向量')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title(f'RBF核SVM决策边界 (准确率: {svm_rbf.score(X, y):.3f})')
plt.legend()
plt.savefig('.history/svm_rbf_decision_boundary.png')
```

---

## 小结

本章介绍了核技巧：

1. **特征空间映射**：将低维非线性可分数据映射到高维线性可分
2. **核函数**：直接计算高维空间内积，无需显式映射
3. **常见核函数**：线性核、多项式核、RBF核各有特点
4. **核化SVM**：只需将对偶问题中的内积替换为核函数

核技巧是机器学习的经典思想，不仅适用于SVM，也适用于PCA、逻辑回归等线性方法。通过核化，线性方法获得了处理非线性问题的能力。

下一章，我们将介绍SVM的优化算法和扩展应用。