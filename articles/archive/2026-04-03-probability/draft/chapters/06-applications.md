---
title: "应用场景：概率在机器学习中的实践"
---

# 应用场景：概率在机器学习中的实践

前五章建立了概率统计的理论和实践基础。本章将这些知识综合应用于机器学习的核心场景，展示概率思维如何指导算法设计和模型理解。我们将深入剖析 Naive Bayes 分类器、逻辑回归的概率解释、EM 算法思想，以及生成模型的入门知识。

## Naive Bayes 分类器

### 贝叶斯分类原理

在[第2章](probability-basics.md)中，我们学习了贝叶斯定理：

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

在分类问题中，我们要计算给定特征 $\mathbf{x}$ 时，类别为 $y$ 的概率：

$$P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y) \cdot P(y)}{P(\mathbf{x})}$$

对于分类，我们只需要比较不同类别的后验概率大小，因此可以忽略分母：

$$\hat{y} = \arg\max_y P(y) \cdot P(\mathbf{x}|y)$$

### 朴素（Naive）假设

直接计算 $P(\mathbf{x}|y) = P(x_1, x_2, \ldots, x_n|y)$ 需要大量数据。**朴素贝叶斯**引入一个强假设：**特征之间条件独立**。

$$P(\mathbf{x}|y) = \prod_{i=1}^n P(x_i|y)$$

这个假设虽然通常不成立（特征往往相关），但在实践中效果出奇地好，尤其是在文本分类中。

### 文本分类示例

```python runnable
import numpy as np

# 朴素贝叶斯文本分类器
class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.priors = {}  # P(y)
        self.likelihoods = {}  # P(x_i|y)
        self.vocabulary = set()
    
    def fit(self, X, y):
        """训练分类器"""
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # 构建词汇表
        for doc in X:
            self.vocabulary.update(doc.split())
        self.vocabulary = list(self.vocabulary)
        vocab_size = len(self.vocabulary)
        
        for c in self.classes:
            # 先验概率 P(y)
            self.priors[c] = np.sum(y == c) / n_samples
            
            # 该类别的所有文档
            class_docs = [X[i] for i in range(len(y)) if y[i] == c]
            
            # 所有词（展开为一个列表）
            all_words = []
            for doc in class_docs:
                all_words.extend(doc.split())
            
            # 似然 P(word|class) 使用拉普拉斯平滑
            word_counts = {}
            total_words = len(all_words)
            
            for word in self.vocabulary:
                count = all_words.count(word)
                # 拉普拉斯平滑：加 1 避免零概率
                word_counts[word] = (count + 1) / (total_words + vocab_size)
            
            self.likelihoods[c] = word_counts
    
    def predict(self, X):
        """预测"""
        predictions = []
        
        for doc in X:
            words = doc.split()
            class_scores = {}
            
            for c in self.classes:
                # log 后验 = log 先验 + sum(log 似然)
                score = np.log(self.priors[c])
                for word in words:
                    if word in self.likelihoods[c]:
                        score += np.log(self.likelihoods[c][word])
                    else:
                        # 未知词：使用平滑后的最小概率
                        score += np.log(1 / (len(self.vocabulary) * 10))
                class_scores[c] = score
            
            predictions.append(max(class_scores, key=class_scores.get))
        
        return predictions

# 示例：垃圾邮件分类
emails = [
    "buy cheap viagra now",           # spam
    "cheap pills for sale",           # spam
    "viagra discount free shipping",  # spam
    "meeting tomorrow at noon",       # ham
    "project deadline next week",     # ham
    "please review the document",     # ham
]
labels = np.array(['spam', 'spam', 'spam', 'ham', 'ham', 'ham'])

# 训练
clf = NaiveBayesClassifier()
clf.fit(emails, labels)

# 测试
test_emails = [
    "cheap viagra pills",
    "meeting about project",
    "free discount viagra",
    "document review tomorrow"
]

predictions = clf.predict(test_emails)

print("=== 朴素贝叶斯垃圾邮件分类器 ===")
print()
for email, pred in zip(test_emails, predictions):
    print(f"邮件: '{email}'")
    print(f"  预测: {pred}")
    print()

# 打印先验概率
print("先验概率:")
for c, p in clf.priors.items():
    print(f"  P({c}) = {p:.4f}")
```

### 为什么朴素贝叶斯效果好的直觉

尽管特征独立性假设通常不成立，朴素贝叶斯在实践中往往表现良好。原因：

1. **分类只需要相对大小**：我们只需要比较后验概率谁大，不需要准确估计概率值。

2. **决策边界简单**：在许多情况下，正确的分类不需要精确的概率估计。

3. **高维稀疏数据**：在文本分类中，特征（词）数量很大，每个样本只包含少量词，独立性假设"部分成立"。

## 逻辑回归的概率解释

### 从线性回归到逻辑回归

线性回归直接预测数值：

$$y = \mathbf{w}^T \mathbf{x} + b$$

逻辑回归预测**概率**：

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

其中 $\sigma(z) = \frac{1}{1+e^{-z}}$ 是 sigmoid 函数，将实数映射到 $(0, 1)$ 区间。

### Sigmoid 函数的概率含义

Sigmoid 函数的输出可以解释为**伯努利分布的参数**：

$$P(y|\mathbf{x}, \mathbf{w}) = \hat{y}^y (1-\hat{y})^{1-y}$$

其中 $\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b)$。

### 最大似然估计推导损失函数

给定训练数据，逻辑回归的参数通过**最大化似然**估计：

$$L(\mathbf{w}) = \prod_{i=1}^n P(y_i|\mathbf{x}_i, \mathbf{w})$$

取对数：

$$\ell(\mathbf{w}) = \sum_{i=1}^n \left[ y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \right]$$

最大化似然等价于**最小化交叉熵损失**：

$$J(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \right]$$

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 逻辑回归的 NumPy 实现
class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def sigmoid(self, z):
        """Sigmoid 函数"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def cross_entropy_loss(self, y_true, y_pred):
        """交叉熵损失"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.n_iterations):
            # 前向传播
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear)
            
            # 记录损失
            loss = self.cross_entropy_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # 计算梯度
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.mean(y_pred - y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X):
        """预测概率"""
        linear = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        return (self.predict_proba(X) >= threshold).astype(int)

# 生成数据
np.random.seed(42)
n_samples = 200

# 两类数据
X_class0 = np.random.randn(n_samples//2, 2) + np.array([-2, -2])
X_class1 = np.random.randn(n_samples//2, 2) + np.array([2, 2])

X = np.vstack([X_class0, X_class1])
y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))

# 训练
model = LogisticRegression(learning_rate=0.1, n_iterations=500)
model.fit(X, y)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：数据点和决策边界
ax = axes[0]
ax.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', label='类别 0', alpha=0.6)
ax.scatter(X_class1[:, 0], X_class1[:, 1], c='red', label='类别 1', alpha=0.6)

# 决策边界
x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[X1.ravel(), X2.ravel()]
proba_grid = model.predict_proba(X_grid).reshape(X1.shape)

ax.contour(X1, X2, proba_grid, levels=[0.5], colors='green', linewidths=2)
ax.contourf(X1, X2, proba_grid, levels=20, cmap='RdBu', alpha=0.3)

ax.set_xlabel('特征 1')
ax.set_ylabel('特征 2')
ax.set_title('逻辑回归分类')
ax.legend()
ax.grid(alpha=0.3)

# 右图：损失下降
axes[1].plot(model.loss_history, 'b-', linewidth=2)
axes[1].set_xlabel('迭代次数')
axes[1].set_ylabel('交叉熵损失')
axes[1].set_title('训练过程')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

# 评估
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)
print(f"训练准确率: {accuracy:.2%}")
print(f"最终损失: {model.loss_history[-1]:.4f}")
print()
print("关键洞察:")
print("1. 逻辑回归输出概率，可以设置不同阈值调整精确率/召回率")
print("2. 交叉熵损失来自最大似然估计")
print("3. 决策边界是线性的")
```

### 从概率视角理解正则化

逻辑回归的正则化可以从**MAP 估计**理解：

- **L2 正则化**：假设权重服从高斯先验
- **L1 正则化**：假设权重服从拉普拉斯先验

这解释了为什么正则化能防止过拟合：它在参数空间施加了"偏好简单模型"的先验信念。

## EM 算法思想

### 隐变量问题

许多机器学习问题涉及**隐变量（Latent Variables）**——我们无法直接观测的变量。例如：

- **聚类**：每个样本属于哪个簇（隐变量）
- **混合模型**：每个样本来自哪个成分分布
- **主题模型**：文档的主题分布

直接最大化似然困难，因为涉及隐变量的求和/积分。

### EM 算法的核心思想

**期望最大化算法（Expectation-Maximization, EM）**通过迭代逼近：

1. **E 步（期望步）**：给定当前参数，估计隐变量的分布
2. **M 步（最大化步）**：给定隐变量分布，最大化似然更新参数

重复 E-M 直到收敛。

### K-means 与 EM 的关系

K-means 可以看作 EM 算法的特例：

- **E 步**：将每个点分配到最近的质心（"硬"分配）
- **M 步**：更新质心为该簇所有点的均值

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# K-means 作为 EM 特例的演示
np.random.seed(42)

# 生成数据
n_samples = 300
X1 = np.random.randn(n_samples//3, 2) + np.array([0, 0])
X2 = np.random.randn(n_samples//3, 2) + np.array([5, 5])
X3 = np.random.randn(n_samples//3, 2) + np.array([5, 0])
X = np.vstack([X1, X2, X3])

def kmeans(X, k, max_iters=100):
    """K-means 聚类"""
    n_samples = X.shape[0]
    
    # 随机初始化质心
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].copy()
    
    history = [centroids.copy()]
    
    for _ in range(max_iters):
        # E 步：分配点到最近的质心
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.sqrt(np.sum((X - centroids[j])**2, axis=1))
        labels = np.argmin(distances, axis=1)
        
        # M 步：更新质心
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        # 检查收敛
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
        history.append(centroids.copy())
    
    return centroids, labels, history

# 运行 K-means
k = 3
final_centroids, final_labels, history = kmeans(X, k)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：聚类结果
colors = ['red', 'blue', 'green']
for j in range(k):
    cluster_points = X[final_labels == j]
    axes[0].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[j], alpha=0.6, label=f'簇 {j+1}')

axes[0].scatter(final_centroids[:, 0], final_centroids[:, 1], 
               c='black', s=200, marker='X', edgecolor='white', linewidth=2,
               label='质心')

axes[0].set_xlabel('特征 1')
axes[0].set_ylabel('特征 2')
axes[0].set_title('K-means 聚类结果')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右图：EM 迭代过程（质心轨迹）
axes[1].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.3)

history = np.array(history)
for j in range(k):
    axes[1].plot(history[:, j, 0], history[:, j, 1], 
                'o-', c=colors[j], linewidth=2, markersize=8)

axes[1].set_xlabel('特征 1')
axes[1].set_ylabel('特征 2')
axes[1].set_title('质心迭代轨迹（EM 过程）')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("K-means 作为 EM 算法的特例:")
print("  E 步: 将每个点分配到最近的质心")
print("  M 步: 更新质心为簇内点的均值")
print()
print(f"迭代次数: {len(history)}")
print("最终质心:")
for j, c in enumerate(final_centroids):
    print(f"  簇 {j+1}: ({c[0]:.2f}, {c[1]:.2f})")
```

### EM 算法的直觉

EM 算法的核心思想是**坐标上升**：交替优化隐变量和参数，每次都使似然增加或保持不变。虽然可能收敛到局部最优，但在实践中非常有效。

## 生成模型入门

### 判别模型 vs 生成模型

**判别模型**：学习 $P(y|\mathbf{x})$，直接建模决策边界。例如：逻辑回归、SVM、神经网络。

**生成模型**：学习 $P(\mathbf{x}, y) = P(\mathbf{x}|y)P(y)$，建模数据生成过程。例如：朴素贝叶斯、GMM、HMM、VAE、GAN。

生成模型可以：
- 生成新样本
- 处理缺失数据
- 进行无监督学习

### GAN 的概率视角

**生成对抗网络（GAN）**包含两个模型：

- **生成器 G**：将随机噪声映射为假样本，试图欺骗判别器
- **判别器 D**：区分真实样本和假样本

从概率视角，GAN 在学习数据分布 $P_{data}$，生成器学习到的分布 $P_G$ 逐渐逼近 $P_{data}$。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GAN 的概率解释                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   随机噪声 z          生成器 G            假样本 x_fake                 │
│   P(z)           ─────────────→       P_G(x)                           │
│   (如正态分布)                                                              │
│                                                                         │
│                         ↓                                               │
│                                                                         │
│                    判别器 D                                              │
│                                                                         │
│   真实样本 x_real  ────────→  区分真假  ────────→  D(x) ∈ [0,1]        │
│   P_data(x)                                    (真=1, 假=0)              │
│                                                                         │
│   目标: P_G(x) 逼近 P_data(x)                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### VAE 的概率视角

**变分自编码器（VAE）**从变分推断角度理解：

- **编码器**：学习 $q(z|x)$，将输入映射到潜在空间的分布
- **解码器**：学习 $p(x|z)$，从潜在分布重建输入

VAE 最大化**证据下界（ELBO）**：

$$\log p(x) \geq E_{q(z|x)}[\log p(x|z)] - KL(q(z|x) || p(z))$$

第一项是重建损失，第二项是正则化项（让学到的潜在分布接近先验分布）。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# VAE 思想的简化演示：编码-解码过程
np.random.seed(42)

# 模拟 VAE 的编码-解码
n_samples = 1000

# 原始数据（二维）
theta = np.random.uniform(0, 2*np.pi, n_samples)
X = np.column_stack([
    np.cos(theta) + np.random.normal(0, 0.1, n_samples),
    np.sin(theta) + np.random.normal(0, 0.1, n_samples)
])

# 编码器：将 x 映射到潜在空间 (z_mean, z_log_var)
# 这里简化为线性映射
def encode(x):
    z_mean = 0.5 * (x[:, 0] + x[:, 1])
    z_log_var = 0.1 * np.ones_like(z_mean)
    return z_mean, z_log_var

# 重参数化采样
def sample_latent(z_mean, z_log_var):
    epsilon = np.random.randn(len(z_mean))
    z = z_mean + np.exp(0.5 * z_log_var) * epsilon
    return z

# 解码器：从潜在空间重建
def decode(z):
    x_recon = np.column_stack([z, z])  # 简化解码
    return x_recon

# VAE 前向过程
z_mean, z_log_var = encode(X)
z = sample_latent(z_mean, z_log_var)
X_recon = decode(z)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 原始数据
axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5, c='steelblue')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('原始数据')
axes[0].set_aspect('equal')
axes[0].grid(alpha=0.3)

# 潜在空间
axes[1].hist(z, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[1].set_xlabel('z')
axes[1].set_ylabel('密度')
axes[1].set_title('潜在空间分布')
axes[1].grid(alpha=0.3)

# 重建数据
axes[2].scatter(X_recon[:, 0], X_recon[:, 1], alpha=0.5, c='orange')
axes[2].set_xlabel('x1')
axes[2].set_ylabel('x2')
axes[2].set_title('重建数据')
axes[2].set_aspect('equal')
axes[2].grid(alpha=0.3)

plt.suptitle('VAE 编码-解码过程演示', fontsize=14)
plt.tight_layout()
plt.show()
plt.close()

print("VAE 的关键概念:")
print("1. 编码器: 学习后验分布 q(z|x)")
print("2. 解码器: 学习生成分布 p(x|z)")
print("3. 损失函数: 重建损失 + KL 散度")
print("4. 重参数化: 使采样过程可微分")
```

## 本章小结

本章将概率统计应用于机器学习的核心场景：

1. **Naive Bayes 分类器**：基于贝叶斯定理和特征独立性假设。虽然假设通常不成立，但在文本分类等高维稀疏数据上效果很好。

2. **逻辑回归的概率解释**：输出概率，损失函数来自最大似然估计。正则化可从 MAP 估计理解。

3. **EM 算法思想**：通过 E 步和 M 步迭代，解决含隐变量的参数估计问题。K-means 是 EM 的特例。

4. **生成模型**：建模数据生成过程，可以生成新样本、处理缺失数据。GAN 和 VAE 是两种代表性的深度生成模型。

概率思维贯穿机器学习的核心算法。理解概率视角，不仅有助于正确使用这些算法，更能帮助诊断问题、设计改进方案。

## 练习题

1. 为什么朴素贝叶斯在文本分类中效果好，即使特征独立性假设不成立？
   <details>
   <summary>参考答案</summary>

   1. **分类只需要相对大小**：我们只需要比较不同类别的后验概率谁大，不需要精确估计概率值。即使概率估计有偏差，只要排序正确，分类就正确。

   2. **高维稀疏性**：文本数据特征（词）很多，但每个文档只包含少量词。词之间的依赖关系在稀疏数据上影响有限。

   3. **简单模型的泛化性**：朴素贝叶斯的"朴素"假设起到正则化作用，防止过拟合。

   4. **对数空间计算**：使用 log 概率可以避免数值下溢，同时将乘法变为加法，计算稳定。

   </details>

2. 解释为什么逻辑回归使用交叉熵损失而不是均方误差。
   <details>
   <summary>参考答案</summary>

   1. **概率解释**：交叉熵损失来自最大似然估计。如果假设标签服从伯努利分布，最大化似然等价于最小化交叉熵。

   2. **优化性质**：均方误差在 sigmoid 函数上会导致非凸优化问题，可能有多个局部最优。交叉熵损失配合 sigmoid 是凸函数，保证找到全局最优。

   3. **梯度性质**：交叉熵损失的梯度在预测错误时较大，预测正确时较小，有利于快速收敛。均方误差的梯度在饱和区（预测接近 0 或 1）会消失。

   </details>

3. 比较判别模型和生成模型的优缺点。
   <details>
   <summary>参考答案</summary>

   **判别模型**：
   - 优点：直接学习决策边界，分类任务上通常效果更好；不需要建模数据分布。
   - 缺点：无法生成新样本；难以处理缺失数据；需要标签数据。

   **生成模型**：
   - 优点：可以生成新样本；可以处理缺失数据；可以用于无监督/半监督学习；提供数据分布的完整描述。
   - 缺点：需要建模数据分布，更困难；分类任务上可能不如判别模型；计算复杂度更高。

   实践中选择：如果只需要分类，判别模型通常更简单有效；如果需要生成样本或理解数据分布，生成模型是必要选择。

   </details>