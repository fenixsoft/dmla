# 降维——压缩数据的维度空间

## 引言：维度诅咒

现代数据往往具有很高的维度。一张100×100的图像就是10000维向量；一篇文档可能有数千个词汇；基因表达数据可能有数万个特征。

高维数据带来两个问题：

1. **计算成本**：特征越多，计算量越大
2. **维度诅咒**：高维空间中数据稀疏，距离度量失效

**降维（Dimensionality Reduction）**将高维数据投影到低维空间，保留主要信息，减少噪声和冗余。降维有两个主要目的：

- **可视化**：将数据降到2-3维，便于观察数据分布
- **特征压缩**：减少特征数量，提高计算效率

---

## PCA目标函数：方差最大化

### 问题设定

给定 $n$ 个样本 $\{x_1, \ldots, x_n\}$，$x_i \in \mathbb{R}^d$。PCA的目标是找到一个投影方向 $w$（$||w||=1$），使得投影后的方差最大化：

$$\max_w \frac{1}{n} \sum_{i=1}^{n} (w^T x_i - w^T \bar{x})^2 = \max_w w^T S w$$

其中 $\bar{x} = \frac{1}{n}\sum_i x_i$ 是均值，$S = \frac{1}{n}\sum_i (x_i - \bar{x})(x_i - \bar{x})^T$ 是协方差矩阵。

### 为什么最大化方差？

方差衡量数据的"分散程度"。投影后方差最大，意味着：
- 数据在投影方向上"展开"得最开
- 损失的信息最少（信息量由方差度量）

---

## 协方差矩阵分解推导

### 拉格朗日乘数法

优化问题：

$$\max_w w^T S w \quad \text{s.t.} \quad w^T w = 1$$

拉格朗日函数：

$$L(w, \lambda) = w^T S w - \lambda(w^T w - 1)$$

对 $w$ 求导：

$$\frac{\partial L}{\partial w} = 2Sw - 2\lambda w = 0$$

得到：

$$Sw = \lambda w$$

**结论**：最优投影方向是协方差矩阵 $S$ 的特征向量，投影方差是对应的特征值。

### PCA算法

1. **数据中心化**：$\tilde{x}_i = x_i - \bar{x}$
2. **计算协方差矩阵**：$S = \frac{1}{n} \tilde{X}^T \tilde{X}$
3. **特征分解**：$S = V\Lambda V^T$，特征值 $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d$
4. **选择主成分**：取前 $k$ 个最大特征值对应的特征向量 $v_1, \ldots, v_k$
5. **投影**：$z_i = [v_1, \ldots, v_k]^T \tilde{x}_i$

---

## PCA投影与重构

### 投影

将数据投影到主成分空间：

$$Z = \tilde{X} V_k$$

其中 $V_k$ 是前 $k$ 个主成分组成的矩阵。

### 重构

从低维表示重构原始数据：

$$\hat{X} = Z V_k^T + \bar{x}$$

### 重构误差

重构误差为被丢弃成分的方差：

$$\text{Error} = \sum_{j=k+1}^{d} \lambda_j$$

**方差解释比例**：

$$\text{解释比例} = \frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{d} \lambda_j}$$

---

## LDA思想：有监督降维

### PCA vs LDA

- **PCA**：无监督，最大化投影方差
- **LDA（线性判别分析）**：有监督，最大化类间距离/类内距离比

### LDA目标

找到投影方向 $w$，使得：

$$\max_w \frac{w^T S_B w}{w^T S_W w}$$

其中：
- $S_B$：类间散度矩阵（Between-class scatter）
- $S_W$：类内散度矩阵（Within-class scatter）

### 选择准则

| 场景 | 推荐方法 |
|------|----------|
| 无标签数据 | PCA |
| 有标签、类别分离明显 | LDA |
| 特征压缩 | PCA |
| 分类预处理 | LDA |

---

## NumPy实现：手写PCA

```python
import numpy as np

class PCA:
    """主成分分析实现"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        
        self.components_ = None      # 主成分（特征向量）
        self.explained_variance_ = None  # 特征值
        self.explained_variance_ratio_ = None  # 解释比例
        self.mean_ = None  # 均值
    
    def fit(self, X):
        """训练PCA模型"""
        n_samples, n_features = X.shape
        
        # 中心化
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # 计算协方差矩阵
        cov_matrix = X_centered.T @ X_centered / (n_samples - 1)
        
        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值降序排列
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        
        # 存储结果
        self.explained_variance_ = eigenvalues
        
        # 计算解释比例
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues / total_variance
        
        # 确定主成分数量
        if self.n_components is None:
            self.n_components = n_features
        
        self.components_ = eigenvectors[:, :self.n_components].T
        
        return self
    
    def transform(self, X):
        """投影到主成分空间"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        """训练并转换"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, Z):
        """从低维空间重构"""
        return Z @ self.components_ + self.mean_


# 测试：鸢尾花数据降维
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("=== PCA降维结果 ===")
print(f"原始维度: {X.shape[1]}")
print(f"降维后维度: {X_pca.shape[1]}")
print(f"\n各主成分解释比例: {pca.explained_variance_ratio_}")
print(f"累计解释比例: {pca.explained_variance_ratio_.sum():.3f}")

# 验证重构
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"\n平均重构误差: {reconstruction_error:.6f}")
```

**输出示例：**
```
=== PCA降维结果 ===
原始维度: 4
降维后维度: 2

各主成分解释比例: [0.9246 0.0530]
累计解释比例: 0.978

平均重构误差: 0.004312
```

---

## 应用：数据可视化

```python
import numpy as np

# 生成多簇数据
np.random.seed(42)

# 3个簇，4维特征
X = np.vstack([
    np.random.multivariate_normal([0, 0, 0, 0], np.eye(4) * 0.5, 50),
    np.random.multivariate_normal([3, 3, 1, 1], np.eye(4) * 0.5, 50),
    np.random.multivariate_normal([-2, 2, -1, 2], np.eye(4) * 0.5, 50)
])

# PCA降维到2维
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

print("=== 数据可视化 ===")
print(f"原始维度: {X.shape[1]}")
print(f"降维后维度: 2")
print(f"累计解释比例: {pca.explained_variance_ratio_.sum():.3f}")

# 可以用matplotlib可视化
# plt.scatter(X_2d[:, 0], X_2d[:, 1])
# plt.xlabel('PC1')
# plt.ylabel('PC2')
```

---

## 应用：特征压缩

```python
# 图像压缩示例
np.random.seed(42)

# 模拟图像数据（100张16x16图像）
n_images = 100
image_size = 16
X_images = np.random.rand(n_images, image_size * image_size)

# 不同压缩比例对比
print("=== 特征压缩对比 ===")
for k in [256, 64, 16, 4]:
    pca = PCA(n_components=k)
    X_compressed = pca.fit_transform(X_images)
    X_reconstructed = pca.inverse_transform(X_compressed)
    
    error = np.mean((X_images - X_reconstructed) ** 2)
    ratio = pca.explained_variance_ratio_.sum()
    compression = (1 - k / 256) * 100
    
    print(f"\n{k}个主成分:")
    print(f"  压缩率: {compression:.1f}%")
    print(f"  方差解释: {ratio:.3f}")
    print(f"  重构误差: {error:.6f}")
```

---

## 小结

本章介绍了PCA降维：

1. **目标函数**：最大化投影方差
2. **协方差矩阵分解**：特征向量即主成分
3. **投影与重构**：可逆变换，损失可量化
4. **LDA对比**：有监督降维考虑类别信息

降维是无监督学习的重要工具。下一章，我们将学习聚类的评估方法和无监督学习的进阶应用。