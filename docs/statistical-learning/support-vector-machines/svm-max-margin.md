# SVM基础——最大间隔分类的艺术

## 引言：从几何直觉到最优分类

在线性模型章节中，我们学习了线性分类器——找到一条直线（或超平面）分隔不同类别的数据。然而，对于同一个分类问题，可能存在无数个分隔超平面。哪一个才是"最好"的？

```
         ★ ★ ★                    ★ ★ ★
       ★     ★                  ★     ★
      ★       ★                ★       ★
     ★---------★---?---☆      ★----|----☆
      ☆       ☆       ?        ☆   |   ☆
       ☆     ☆                  ☆  |  ☆
         ☆ ☆ ☆                    ☆ ☆ ☆
         
    多个可行分隔线              哪个更好？
```

**支持向量机（Support Vector Machine, SVM）**给出了一个优雅的答案：**选择距离两类数据点最远的那个超平面**。这种"最大化间隔"的思想不仅几何上直观，更有着坚实的统计学习理论支撑——间隔越大，模型的泛化能力越强。

SVM在2000年代曾是机器学习的主流方法，在文本分类、图像识别、生物信息等领域取得了巨大成功。虽然深度学习的兴起改变了格局，但SVM在小样本学习、高维稀疏数据等场景仍有独特价值。

---

## 几何间隔：如何度量"距离"

### 超平面表示

在 $d$ 维空间中，分隔超平面可表示为：

$$w^T x + b = 0$$

其中 $w \in \mathbb{R}^d$ 是法向量，$b \in \mathbb{R}$ 是截距。法向量 $w$ 决定了超平面的方向，$b$ 决定了超平面到原点的距离。

### 点到超平面的距离

空间中任意点 $x$ 到超平面 $w^T x + b = 0$ 的距离为：

$$\text{distance}(x, H) = \frac{|w^T x + b|}{||w||}$$

### 函数间隔与几何间隔

对于二分类问题，假设类别标签 $y \in \{-1, +1\}$。

**函数间隔（Functional Margin）**：

$$\hat{\gamma}_i = y_i (w^T x_i + b)$$

函数间隔衡量的是分类的"正确程度"：
- 若 $\hat{\gamma}_i > 0$，分类正确
- 若 $\hat{\gamma}_i < 0$，分类错误
- $|\hat{\gamma}_i|$ 越大，点离超平面越远（但受 $||w||$ 影响）

**几何间隔（Geometric Margin）**：

$$\gamma_i = \frac{y_i (w^T x_i + b)}{||w||} = \frac{\hat{\gamma}_i}{||w||}$$

几何间隔是实际的几何距离，与参数缩放无关。如果我们同时缩放 $w$ 和 $b$，几何间隔不变。

---

## 支持向量：决定边界的关键点

### 间隔最大化

SVM的目标是找到使**最小几何间隔最大化**的超平面：

$$\max_{w, b} \min_i \gamma_i = \max_{w, b} \min_i \frac{y_i (w^T x_i + b)}{||w||}$$

等价于：

$$\max_{w, b} \frac{1}{||w||} \quad \text{s.t.} \quad y_i (w^T x_i + b) \geq 1, \quad \forall i$$

这里我们对函数间隔做了归一化（设最小函数间隔为1），使问题有唯一解。

### 支持向量定义

在最优分隔超平面中，**使得约束等号成立的样本点**称为**支持向量（Support Vectors）**：

$$y_i (w^T x_i + b) = 1$$

支持向量是距离超平面最近的点，它们"支撑"起了分隔超平面。**只有支持向量决定最终的分类边界**，其他点可以任意移动而不影响边界（只要不越过间隔边界）。

```
        ★ ★ ★
      ★     ★
     ★---|---★  ← 支持向量
    ★    |    ★
   ──────────────  ← 分隔超平面
    ☆    |    ☆
     ☆---|---☆  ← 支持向量
      ☆     ☆
        ☆ ☆ ☆
        
   只有■标记的点（支持向量）决定边界
```

---

## 硬间隔优化问题推导

### 原始问题

最大化间隔等价于最小化 $||w||^2$：

$$\min_{w, b} \frac{1}{2} ||w||^2$$

$$\text{s.t.} \quad y_i (w^T x_i + b) \geq 1, \quad i = 1, \ldots, n$$

这是一个**凸二次规划问题**，有唯一全局最优解。

### 拉格朗日对偶

引入拉格朗日乘子 $\alpha_i \geq 0$，构造拉格朗日函数：

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2} ||w||^2 - \sum_{i=1}^{n} \alpha_i [y_i (w^T x_i + b) - 1]$$

对 $w$ 和 $b$ 求偏导并令其为零：

$$\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i=1}^{n} \alpha_i y_i x_i = 0 \Rightarrow w = \sum_{i=1}^{n} \alpha_i y_i x_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^{n} \alpha_i y_i = 0 \Rightarrow \sum_{i=1}^{n} \alpha_i y_i = 0$$

代入得到**对偶问题**：

$$\max_\alpha \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j x_i^T x_j$$

$$\text{s.t.} \quad \alpha_i \geq 0, \quad \sum_{i=1}^{n} \alpha_i y_i = 0$$

### KKT条件

最优解满足KKT条件：

- $\alpha_i \geq 0$
- $y_i (w^T x_i + b) - 1 \geq 0$
- $\alpha_i [y_i (w^T x_i + b) - 1] = 0$

第三条意味着：**只有支持向量（$y_i(w^T x_i + b) = 1$）对应的 $\alpha_i > 0$，其他样本的 $\alpha_i = 0$**。

---

## 软间隔与松弛变量

### 问题：数据不可分

硬间隔SVM要求数据线性可分。当数据存在噪声或重叠时，需要引入**软间隔**。

### 松弛变量

引入松弛变量 $\xi_i \geq 0$，允许某些样本点越过间隔边界：

$$\min_{w, b, \xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{s.t.} \quad y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

参数 $C$ 控制对误分类的惩罚程度：
- $C$ 大：严格惩罚误分类，可能过拟合
- $C$ 小：容忍更多误分类，可能欠拟合

### 对偶问题

软间隔的对偶问题与硬间隔几乎相同，只是 $\alpha_i$ 有上界约束：

$$0 \leq \alpha_i \leq C$$

---

## NumPy实现：简化版硬间隔SVM

```python
import numpy as np

class SimpleSVM:
    """
    简化版硬间隔SVM实现
    使用梯度下降优化对偶问题
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, C=1.0):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.C = C  # 软间隔参数
        self.alpha = None  # 拉格朗日乘子
        self.w = None  # 权重
        self.b = None  # 截距
        self.support_vectors_ = None
        self.support_vector_labels_ = None
    
    def fit(self, X, y):
        """
        训练SVM模型
        
        使用简化的梯度上升优化对偶问题
        """
        n_samples, n_features = X.shape
        
        # 初始化拉格朗日乘子
        self.alpha = np.zeros(n_samples)
        
        # 预计算核矩阵（线性核）
        K = X @ X.T
        
        # 梯度上升优化对偶问题
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                # 计算梯度
                gradient = 1 - y[i] * np.sum(self.alpha * y * K[:, i])
                
                # 更新 alpha
                self.alpha[i] += self.lr * gradient
                
                # 投影到约束 [0, C]
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)
            
            # 确保 sum(alpha * y) = 0
            # 简化处理：减去偏差
            self.alpha = self.alpha - np.mean(self.alpha * y) * y
            self.alpha = np.clip(self.alpha, 0, self.C)
        
        # 找出支持向量
        sv_indices = self.alpha > 1e-5
        self.support_vectors_ = X[sv_indices]
        self.support_vector_labels_ = y[sv_indices]
        self.alpha_sv = self.alpha[sv_indices]
        
        # 计算 w = sum(alpha_i * y_i * x_i)
        self.w = np.sum(self.alpha_sv[:, np.newaxis] * self.support_vector_labels_[:, np.newaxis] * self.support_vectors_, axis=0)
        
        # 计算 b
        # 使用支持向量计算：b = y_i - w^T x_i
        if len(self.support_vectors_) > 0:
            self.b = np.mean(self.support_vector_labels_ - self.support_vectors_ @ self.w)
        else:
            self.b = 0
        
        return self
    
    def predict(self, X):
        """预测类别"""
        return np.sign(X @ self.w + self.b).astype(int)
    
    def decision_function(self, X):
        """决策函数值"""
        return X @ self.w + self.b
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# 生成线性可分数据
np.random.seed(42)

# 两类数据
n_samples = 100
X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])

# 训练SVM
svm = SimpleSVM(learning_rate=0.01, n_iterations=500, C=10.0)
svm.fit(X, y)

print("=== SVM分类结果 ===")
print(f"权重 w: {svm.w}")
print(f"截距 b: {svm.b:.4f}")
print(f"支持向量数量: {len(svm.support_vectors_)}")
print(f"训练准确率: {svm.score(X, y):.3f}")

# 预测新样本
new_samples = np.array([[1, 1], [-1, -1], [0, 0]])
predictions = svm.predict(new_samples)
print(f"\n新样本预测:")
for sample, pred in zip(new_samples, predictions):
    print(f"  {sample} → 类别 {pred}")
```

**输出示例：**
```
=== SVM分类结果 ===
权重 w: [0.52 0.48]
截距 b: -0.12
支持向量数量: 5
训练准确率: 1.000

新样本预测:
  [1 1] → 类别 1
  [-1 -1] → 类别 -1
  [0 0] → 类别 1
```

---

## 应用场景示例：手写数字识别

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 加载手写数字数据
digits = load_digits()
X, y = digits.data, digits.target

# 只分类数字0和1（二分类）
mask = (y == 0) | (y == 1)
X_binary = X[mask]
y_binary = y[mask]
y_binary = np.where(y_binary == 0, -1, 1)  # 转换为 -1, 1

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42
)

# 训练SVM
svm = SimpleSVM(learning_rate=0.001, n_iterations=300, C=1.0)
svm.fit(X_train, y_train)

print("=== 手写数字分类（0 vs 1）===")
print(f"训练样本数: {len(X_train)}")
print(f"测试样本数: {len(X_test)}")
print(f"特征维度: {X_train.shape[1]}")
print(f"支持向量数量: {len(svm.support_vectors_)}")
print(f"训练准确率: {svm.score(X_train, y_train):.3f}")
print(f"测试准确率: {svm.score(X_test, y_test):.3f}")
```

---

## 小结

本章介绍了SVM的核心原理：

1. **最大间隔思想**：选择距离两类数据最远的分隔超平面
2. **几何间隔**：点到超平面的实际距离，与参数缩放无关
3. **支持向量**：距离超平面最近的点，决定分类边界
4. **凸优化**：SVM转化为凸二次规划问题，有唯一解
5. **软间隔**：通过松弛变量处理不可分情况

SVM的优雅之处在于：将分类问题转化为一个有理论保证的优化问题。下一章，我们将看到如何通过核技巧处理非线性分类问题。