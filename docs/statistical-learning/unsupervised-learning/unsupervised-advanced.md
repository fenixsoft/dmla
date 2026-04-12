# 无监督学习进阶

## 引言：如何评估无监督学习结果

监督学习有明确的评估标准 —— 预测准确率、均方误差等。但无监督学习没有标签，如何判断聚类结果好不好？降维保留了多少信息？

本章介绍聚类评估指标、改进算法，以及无监督学习与深度学习的关系。

---

## 聚类评估指标：轮廓系数

### 定义

对于样本 $i$，定义：

- $a_i$：样本 $i$ 到同簇其他样本的平均距离（簇内距离）
- $b_i$：样本 $i$ 到最近其他簇所有样本的平均距离（最近簇距离）

**轮廓系数**（Silhouette Coefficient）：

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

### 取值范围

- $s_i \in [-1, 1]$
- $s_i \approx 1$：样本远离其他簇，聚类效果好
- $s_i \approx 0$：样本在簇边界
- $s_i \approx -1$：样本可能分错簇

### 使用方法

计算所有样本的平均轮廓系数：

$$S = \frac{1}{n} \sum_{i=1}^{n} s_i$$

$S$ 越大，聚类效果越好。可用于选择最佳 $K$ 值。

```python
def silhouette_score(X, labels):
    """计算轮廓系数"""
    n = len(X)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    if k == 1 or k == n:
        return 0
    
    scores = np.zeros(n)
    
    for i in range(n):
        # 簇内距离 a_i
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a_i = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
        else:
            a_i = 0
        
        # 最近簇距离 b_i
        b_i = float('inf')
        for label in unique_labels:
            if label == labels[i]:
                continue
            other_cluster = X[labels == label]
            dist = np.mean(np.linalg.norm(other_cluster - X[i], axis=1))
            b_i = min(b_i, dist)
        
        # 轮廓系数
        scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
    
    return np.mean(scores)
```

---

## 聚类评估指标：Calinski-Harabasz 指数

### 定义

$$CH = \frac{\text{SSB} / (K-1)}{\text{SSW} / (n-K)}$$

其中：
- $\text{SSB}$：簇间离散度（Between-cluster dispersion）
- $\text{SSW}$：簇内离散度（Within-cluster dispersion）

### 解释

- 分子：簇间距离，越大越好
- 分母：簇内距离，越小越好
- CH 值越大，聚类效果越好

### 与轮廓系数对比

| 指标 | 优点 | 缺点 |
|------|------|------|
| 轮廓系数 | 直观，可检测异常点 | 计算复杂度 O(n²) |
| CH 指数 | 计算快 O(n) | 对凸形簇偏好 |

---

## K-means++初始化改进

### 问题：随机初始化的局限

随机初始化可能导致：
- 初始中心聚集在一起
- 收敛到差的局部最优
- 需要多次运行

### K-means++思想

让初始中心**彼此分散**：

1. 随机选择第一个中心 $\mu_1$
2. 对于每个后续中心 $k=2,\ldots,K$：
   - 计算每个样本到最近已选中心的距离 $D(x)$
   - 以概率 $\frac{D(x)^2}{\sum D(x)^2}$ 选择下一个中心
3. 继续标准 K-means 迭代

### 为什么有效？

- 第一个中心随机选择
- 后续中心倾向于选择远离已有中心的点
- 初始分散，更容易找到好的聚类

```python
def kmeans_plusplus_init(X, K):
    """K-means++初始化"""
    n = len(X)
    centers = []
    
    # 随机选择第一个中心
    idx = np.random.randint(n)
    centers.append(X[idx].copy())
    
    for _ in range(1, K):
        # 计算每个点到最近中心的距离
        distances = np.zeros(n)
        for i, x in enumerate(X):
            min_dist = float('inf')
            for c in centers:
                dist = np.sum((x - c) ** 2)
                min_dist = min(min_dist, dist)
            distances[i] = min_dist
        
        # 按距离平方的概率选择
        probs = distances / distances.sum()
        idx = np.random.choice(n, p=probs)
        centers.append(X[idx].copy())
    
    return np.array(centers)
```

---

## K-means++与随机初始化对比

```python
# 对比实验
np.random.seed(42)

# 生成明显分离的数据
X = np.vstack([
    np.random.randn(100, 2) * 0.5 + [0, 0],
    np.random.randn(100, 2) * 0.5 + [5, 5],
    np.random.randn(100, 2) * 0.5 + [0, 5]
])

# 随机初始化 vs K-means++
print("=== 初始化方法对比 ===")

# 随机初始化
random_inertias = []
for _ in range(10):
    kmeans = KMeans(n_clusters=3, n_init=1)
    kmeans.fit(X)
    random_inertias.append(kmeans.inertia_)

# K-means++（单次运行）
class KMeansPP(KMeans):
    def _init_centers(self, X):
        return kmeans_plusplus_init(X, self.n_clusters)

kmeans_pp = KMeansPP(n_clusters=3, n_init=1)
kmeans_pp.fit(X)

print(f"随机初始化（10次平均）惯性: {np.mean(random_inertias):.2f}")
print(f"随机初始化（最佳）惯性: {np.min(random_inertias):.2f}")
print(f"K-means++ 惯性: {kmeans_pp.inertia_:.2f}")
```

---

## PCA+K-means 综合应用

### 为什么先降维再聚类？

1. **去除噪声**：PCA 丢弃噪声成分
2. **加速计算**：维度降低，距离计算更快
3. **可视化**：可以直观看到聚类效果

### 应用流程

```
原始数据 → PCA降维 → K-means聚类 → 可视化/分析
```

```python
import numpy as np

# 高维数据示例
np.random.seed(42)
n_samples = 300
n_features = 50

# 生成3簇高维数据
centers = np.random.randn(3, n_features) * 5
X_high = np.vstack([
    np.random.randn(100, n_features) * 0.5 + centers[0],
    np.random.randn(100, n_features) * 0.5 + centers[1],
    np.random.randn(100, n_features) * 0.5 + centers[2]
])

print("=== PCA+K-means综合应用 ===")
print(f"原始数据维度: {X_high.shape}")

# PCA降维
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_high)
print(f"PCA降维后: {X_pca.shape}")
print(f"保留方差比例: {pca.explained_variance_ratio_[:5].sum():.3f}")

# K-means聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_pca)
print(f"\n聚类惯性: {kmeans.inertia_:.2f}")

# 轮廓系数
sil_score = silhouette_score(X_pca, kmeans.labels_)
print(f"轮廓系数: {sil_score:.3f}")
```

---

## 与深度学习对比

### 无监督学习的现代发展

**自监督学习**：从数据中自动构造标签

- 对比学习（Contrastive Learning）
- 掩码自编码器（Masked Autoencoder）
- 生成式预训练（GPT、BERT）

### PCA 与自编码器

| 特性 | PCA | 自编码器 |
|------|-----|----------|
| 非线性 | 线性 | 可非线性 |
| 训练 | 特征分解 | 梯度优化 |
| 可解释性 | 强 | 弱 |
| 数据量 | 小样本即可 | 需要大量数据 |

**关系**：单层线性自编码器等价于 PCA

### 无监督学习的价值

1. **数据探索**：理解数据分布和结构
2. **特征工程**：为监督学习准备特征
3. **异常检测**：发现偏离正常模式的样本
4. **数据压缩**：减少存储和传输成本
5. **生成模型**：VAE、GAN 等生成方法的基础

### 与监督学习的关系

```
无监督学习 → 发现结构 → 指导监督学习
     ↓
   聚类 → 发现类别 → 构造标签
     ↓
   降维 → 提取特征 → 改善分类
```

---

## 小结

本章介绍了无监督学习进阶内容：

1. **轮廓系数**：评估聚类质量，选择最佳 K 值
2. **CH 指数**：快速评估聚类效果
3. **K-means++**：改进初始化，避免差局部最优
4. **PCA+K-means**：降维去噪，加速聚类
5. **与深度学习关系**：自监督学习、自编码器

无监督学习是机器学习的重要组成部分。虽然没有标签，但它能发现数据中的隐藏结构，为后续分析和建模提供基础。"经典统计学习方法"系列至此完结，希望读者通过这五章的学习，建立了对传统机器学习方法的系统认知。