# 聚类——发现数据的自然分组

## 引言：没有标签时如何学习

在前面的章节中，我们学习的所有方法都有明确的目标——预测房价、识别类别、分类邮件。这些**监督学习**方法依赖于标签：每个样本都有正确的答案，模型学习从输入到输出的映射。

然而，现实世界中大量数据**没有标签**。我们可能有客户的购买记录，但不知道他们属于哪个"群体"；有基因表达数据，但不知道哪些基因共同作用；有社交网络关系，但不知道社区结构。

**无监督学习（Unsupervised Learning）**就是在没有标签的情况下发现数据的内在结构。**聚类（Clustering）**是最基本的无监督学习任务：将相似的样本归为一组（簇），不同组的样本尽可能不同。

```
        ★ ★ ★
      ★       ★          聚类后:
     ★    ?    ★          ├── 簇1: ★★★★★
    ★    ?    ★     →     ├── 簇2: ☆☆☆☆☆
     ☆    ?    ☆          └── 簇3: ◆◆◆◆◆
      ☆       ☆
        ☆ ☆ ☆
        
    原始数据无标签          自动发现分组
```

---

## K-means目标函数

### 问题定义

给定 $n$ 个样本 $\{x_1, x_2, \ldots, x_n\}$，将它们分成 $K$ 个簇 $C_1, C_2, \ldots, C_K$。目标是最小化簇内距离平方和：

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

其中 $\mu_k$ 是簇 $C_k$ 的中心（均值）：

$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

### 目标函数含义

- $||x_i - \mu_k||^2$：样本到簇中心的距离平方
- 目标函数 $J$ 度量所有样本到其所属簇中心的距离总和
- $J$ 越小，说明簇内样本越紧密

---

## K-means迭代流程

K-means是一种**交替优化**算法，交替更新簇分配和簇中心。

### 算法步骤

```
输入：数据 X，簇数 K
输出：簇分配和簇中心

1. 随机选择 K 个样本作为初始簇中心 μ₁, μ₂, ..., μₖ
2. 重复直到收敛：
   a) 分配步骤：每个样本分配到最近的簇中心
      c_i = argmin_k ||x_i - μ_k||²
   
   b) 更新步骤：重新计算每个簇的中心
      μ_k = mean({x_i : c_i = k})
```

### 收敛性

- 每次迭代，目标函数 $J$ 都不会增加
- 簇分配和中心都在改善
- 最终收敛到**局部最优**（不一定是全局最优）

---

## 收敛条件与局部最优问题

### 收敛条件

满足以下任一条件停止：
1. 簇分配不再变化
2. 目标函数变化小于阈值
3. 达到最大迭代次数

### 局部最优问题

K-means对初始中心敏感。不同的初始化可能导致不同的聚类结果。

**解决方案**：
1. **多次运行**：随机初始化多次，选择目标函数最小的结果
2. **K-means++**：改进的初始化方法，使初始中心分散

---

## 层次聚类思想

### 凝聚式层次聚类

从每个样本单独成簇开始，逐步合并最相似的簇：

```
初始: {x1} {x2} {x3} {x4} {x5}
       ↓ 合并最近的两个簇
步骤1: {x1, x2} {x3} {x4} {x5}
       ↓ 
步骤2: {x1, x2} {x3, x4} {x5}
       ↓
步骤3: {x1, x2} {x3, x4, x5}
       ↓
最终:  {x1, x2, x3, x4, x5}
```

### 簇间距离定义

如何度量两个簇之间的距离？

- **单链接（Single Linkage）**：最近两点距离
- **全链接（Complete Linkage）**：最远两点距离
- **平均链接（Average Linkage）**：所有点对的平均距离
- **Ward方法**：合并使簇内方差增加最少的簇

### 树状图

层次聚类结果可以用**树状图（Dendrogram）**表示：

```
    x1  x2  x3  x4  x5
    │   │   │   │   │
    └───┘   │   │   │
        │   │   │   │
        │   └───┘   │
        │       │   │
        └───────┘   │
                │   │
                └───┘
```

通过在不同高度"切割"树状图，可以得到不同数量的簇。

---

## NumPy实现：手写K-means

```python
import numpy as np

class KMeans:
    """K-means聚类实现"""
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init  # 多次初始化
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None  # 目标函数值
    
    def _init_centers(self, X):
        """随机初始化簇中心"""
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _assign_clusters(self, X, centers):
        """分配样本到最近的簇"""
        distances = np.zeros((len(X), self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - centers[k]) ** 2, axis=1)
        return np.argmin(distances, axis=1)
    
    def _update_centers(self, X, labels):
        """更新簇中心"""
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                centers[k] = X[mask].mean(axis=0)
            else:
                # 空簇：随机重新初始化
                centers[k] = X[np.random.randint(len(X))]
        return centers
    
    def _compute_inertia(self, X, labels, centers):
        """计算目标函数值"""
        inertia = 0
        for k in range(self.n_clusters):
            mask = labels == k
            inertia += np.sum((X[mask] - centers[k]) ** 2)
        return inertia
    
    def fit(self, X):
        """训练模型"""
        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        
        for init in range(self.n_init):
            # 初始化
            centers = self._init_centers(X)
            
            for i in range(self.max_iter):
                # 分配簇
                labels = self._assign_clusters(X, centers)
                
                # 更新中心
                new_centers = self._update_centers(X, labels)
                
                # 检查收敛
                if np.max(np.abs(new_centers - centers)) < self.tol:
                    break
                
                centers = new_centers
            
            inertia = self._compute_inertia(X, labels, centers)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
        
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        
        return self
    
    def predict(self, X):
        """预测新样本的簇分配"""
        return self._assign_clusters(X, self.cluster_centers_)


# 测试：生成聚类数据
np.random.seed(42)

# 生成3簇数据
n_samples = 300
centers_true = np.array([[0, 0], [5, 5], [0, 5]])

X = np.vstack([
    np.random.randn(100, 2) + centers_true[0],
    np.random.randn(100, 2) + centers_true[1],
    np.random.randn(100, 2) + centers_true[2]
])

# 训练K-means
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X)

print("=== K-means聚类结果 ===")
print(f"真实中心: \n{centers_true}")
print(f"\n估计中心: \n{kmeans.cluster_centers_}")
print(f"\n目标函数值: {kmeans.inertia_:.2f}")

# 统计每个簇的样本数
unique, counts = np.unique(kmeans.labels_, return_counts=True)
print(f"\n各簇样本数: {dict(zip(unique, counts))}")
```

**输出示例：**
```
=== K-means聚类结果 ===
真实中心: 
[[0 0]
 [5 5]
 [0 5]]

估计中心: 
[[-0.02 -0.08]
 [ 4.98  4.95]
 [ 0.05  5.02]]

目标函数值: 587.32

各簇样本数: {0: 98, 1: 102, 2: 100}
```

---

## 应用场景示例：客户分群

```python
import numpy as np

# 模拟客户数据：消费金额、购买频率
np.random.seed(42)
n_customers = 200

# 生成不同类型的客户
high_value = np.random.multivariate_normal([1000, 20], [[50000, 500], [500, 20]], 50)
medium_value = np.random.multivariate_normal([500, 10], [[20000, 200], [200, 10]], 100)
low_value = np.random.multivariate_normal([100, 3], [[5000, 50], [50, 2]], 50)

X_customers = np.vstack([high_value, medium_value, low_value])

# 确保非负
X_customers = np.maximum(X_customers, 0)

# K-means分群
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X_customers)

print("=== 客户分群结果 ===")
for k in range(3):
    mask = kmeans.labels_ == k
    avg_spend = X_customers[mask, 0].mean()
    avg_freq = X_customers[mask, 1].mean()
    count = np.sum(mask)
    print(f"\n簇{k+1} ({count}人):")
    print(f"  平均消费: {avg_spend:.0f}元")
    print(f"  平均频率: {avg_freq:.1f}次/月")
```

---

## 小结

本章介绍了聚类方法：

1. **K-means目标函数**：最小化簇内距离平方和
2. **迭代优化**：交替进行分配步骤和更新步骤
3. **局部最优**：对初始化敏感，需要多次运行
4. **层次聚类**：自底向上合并，树状图可视化

聚类是无监督学习的基础。下一章，我们将学习另一种无监督学习任务——降维。