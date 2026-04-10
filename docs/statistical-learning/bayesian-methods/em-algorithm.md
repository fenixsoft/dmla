# EM算法——隐变量模型的参数估计

## 引言：隐变量的困境

在前两章，我们假设所有变量都是可观测的。然而，现实问题中常常存在**隐变量（Latent Variable）**——我们无法直接观测，但影响观测数据的变量：

- **聚类问题**：数据属于哪个"簇"是隐变量，我们只观测到数据点的特征
- **混合模型**：数据来自哪个"成分"是隐变量
- **缺失数据**：某些变量的值缺失，也可视为隐变量

当存在隐变量时，直接使用最大似然估计（MLE）变得困难——似然函数包含对隐变量的求和/积分，难以直接优化。

**EM算法（Expectation-Maximization）**是处理隐变量问题的经典方法。它通过迭代交替进行"期望"和"最大化"两步，逐步逼近最大似然解。

---

## 隐变量问题与EM思想

### 问题设定

设观测变量为 $X$，隐变量为 $Z$，参数为 $\theta$。我们想最大化观测数据的对数似然：

$$\ell(\theta) = \log P(X|\theta) = \log \sum_Z P(X, Z|\theta)$$

由于存在对 $Z$ 的求和，对数无法进入求和内部，直接优化困难。

### EM核心思想

EM算法的关键洞察是：**与其直接优化 $\ell(\theta)$，不如优化它的下界**。

根据Jensen不等式，对任意分布 $q(Z)$：

$$\log P(X|\theta) = \log \sum_Z P(X, Z|\theta) = \log \sum_Z q(Z) \frac{P(X, Z|\theta)}{q(Z)} \geq \sum_Z q(Z) \log \frac{P(X, Z|\theta)}{q(Z)}$$

这个下界称为**证据下界（Evidence Lower Bound, ELBO）**：

$$\mathcal{L}(q, \theta) = \sum_Z q(Z) \log P(X, Z|\theta) - \sum_Z q(Z) \log q(Z)$$

EM算法交替优化 $q$ 和 $\theta$：

- **E步**：固定 $\theta$，选择 $q(Z) = P(Z|X, \theta)$ 使下界紧贴目标函数
- **M步**：固定 $q$，优化 $\theta$ 使下界最大化

---

## EM算法推导

### E步：期望计算

在E步，我们计算隐变量的后验分布：

$$q(Z) = P(Z|X, \theta^{(t)})$$

这等价于计算完全数据对数似然的期望：

$$Q(\theta|\theta^{(t)}) = \mathbb{E}_{Z|X, \theta^{(t)}}[\log P(X, Z|\theta)]$$

### M步：最大化

在M步，我们最大化 $Q$ 函数：

$$\theta^{(t+1)} = \arg\max_\theta Q(\theta|\theta^{(t)})$$

### 算法流程

```
初始化参数 θ⁰
重复直到收敛:
    E步: 计算期望 Q(θ|θᵗ) = E[log P(X,Z|θ) | X, θᵗ]
    M步: 更新参数 θᵗ⁺¹ = argmax Q(θ|θᵗ)
```

### 收敛性

可以证明：**每次EM迭代，似然函数单调不减**：

$$\log P(X|\theta^{(t+1)}) \geq \log P(X|\theta^{(t)})$$

因此，EM算法保证收敛到局部最优（或鞍点）。

---

## 高斯混合模型（GMM）

### 模型定义

**高斯混合模型（Gaussian Mixture Model, GMM）**是最经典的EM应用。假设数据来自 $K$ 个高斯分布的混合：

$$P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

其中：
- $\pi_k$：第 $k$ 个成分的混合系数，$\sum_k \pi_k = 1$
- $\mu_k, \Sigma_k$：第 $k$ 个高斯成分的均值和协方差

隐变量 $z_i$ 表示样本 $x_i$ 来自哪个成分。

### GMM的EM推导

**E步**：计算每个样本属于各成分的后验概率（责任度）：

$$\gamma_{ik} = P(z_i = k | x_i) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

**M步**：更新参数：

$$N_k = \sum_{i=1}^{n} \gamma_{ik}$$

$$\mu_k^{new} = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} x_i$$

$$\Sigma_k^{new} = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} (x_i - \mu_k^{new})(x_i - \mu_k^{new})^T$$

$$\pi_k^{new} = \frac{N_k}{n}$$

---

## NumPy实现：GMM完整EM流程

```python
import numpy as np

class GaussianMixtureModel:
    """
    高斯混合模型实现
    使用EM算法求解
    """
    
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol  # 收敛阈值
        
        self.weights_ = None   # 混合系数 (K,)
        self.means_ = None     # 均值 (K, n_features)
        self.covariances_ = None  # 协方差矩阵 (K, n_features, n_features)
        self.log_likelihood_history_ = []
    
    def _initialize(self, X):
        """初始化参数"""
        n_samples, n_features = X.shape
        K = self.n_components
        
        # 随机初始化均值（从数据中随机选择K个点）
        indices = np.random.choice(n_samples, K, replace=False)
        self.means_ = X[indices].copy()
        
        # 初始化协方差为数据协方差的对角线
        data_cov = np.cov(X.T)
        self.covariances_ = np.array([np.diag(np.diag(data_cov)) + 1e-6 * np.eye(n_features) 
                                       for _ in range(K)])
        
        # 初始化混合系数为均匀分布
        self.weights_ = np.ones(K) / K
    
    def _gaussian_pdf(self, X, mean, cov):
        """计算多元高斯概率密度"""
        n_features = X.shape[1]
        diff = X - mean
        
        # 加小值保证数值稳定
        cov_reg = cov + 1e-6 * np.eye(n_features)
        
        # 使用Cholesky分解计算行列式和逆
        try:
            L = np.linalg.cholesky(cov_reg)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            diff_L = np.linalg.solve(L, diff.T).T
            mahalanobis = np.sum(diff_L ** 2, axis=1)
        except np.linalg.LinAlgError:
            # 如果Cholesky失败，使用标准方法
            sign, log_det = np.linalg.slogdet(cov_reg)
            cov_inv = np.linalg.inv(cov_reg)
            mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
        
        log_prob = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)
        return log_prob
    
    def _e_step(self, X):
        """E步：计算责任度"""
        n_samples = X.shape[0]
        K = self.n_components
        
        # 计算每个成分的对数概率
        log_probs = np.zeros((n_samples, K))
        for k in range(K):
            log_probs[:, k] = (np.log(self.weights_[k] + 1e-10) + 
                               self._gaussian_pdf(X, self.means_[k], self.covariances_[k]))
        
        # 计算对数似然
        log_likelihood = np.sum(np.log(np.sum(np.exp(log_probs), axis=1)))
        
        # 计算责任度（使用log-sum-trick避免数值下溢）
        log_sum = np.log(np.sum(np.exp(log_probs - log_probs.max(axis=1, keepdims=True)), axis=1, keepdims=True)) + log_probs.max(axis=1, keepdims=True)
        responsibilities = np.exp(log_probs - log_sum)
        
        return responsibilities, log_likelihood
    
    def _m_step(self, X, responsibilities):
        """M步：更新参数"""
        n_samples, n_features = X.shape
        K = self.n_components
        
        # 计算每个成分的有效样本数
        N_k = responsibilities.sum(axis=0) + 1e-10
        
        # 更新混合系数
        self.weights_ = N_k / n_samples
        
        # 更新均值
        self.means_ = (responsibilities.T @ X) / N_k[:, np.newaxis]
        
        # 更新协方差
        for k in range(K):
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            self.covariances_[k] = (weighted_diff.T @ diff) / N_k[k]
            # 添加正则化
            self.covariances_[k] += 1e-6 * np.eye(n_features)
    
    def fit(self, X):
        """训练模型"""
        self._initialize(X)
        self.log_likelihood_history_ = []
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E步
            responsibilities, log_likelihood = self._e_step(X)
            self.log_likelihood_history_.append(log_likelihood)
            
            # 检查收敛
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"EM收敛于第{iteration}次迭代")
                break
            
            # M步
            self._m_step(X, responsibilities)
            
            prev_log_likelihood = log_likelihood
        
        return self
    
    def predict(self, X):
        """预测聚类标签"""
        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        """预测属于各成分的概率"""
        responsibilities, _ = self._e_step(X)
        return responsibilities
    
    def score(self, X):
        """计算对数似然"""
        _, log_likelihood = self._e_step(X)
        return log_likelihood


# 测试：生成混合高斯数据
np.random.seed(42)

# 生成3个高斯分布的数据
n_samples = 300
true_means = np.array([[0, 0], [3, 3], [0, 4]])
true_covs = np.array([
    [[1, 0.3], [0.3, 1]],
    [[0.5, 0], [0, 0.5]],
    [[1, -0.5], [-0.5, 1]]
])

X = []
for i in range(3):
    samples = np.random.multivariate_normal(true_means[i], true_covs[i], 100)
    X.append(samples)
X = np.vstack(X)

# 打乱数据
np.random.shuffle(X)

# 训练GMM
gmm = GaussianMixtureModel(n_components=3, max_iter=100)
gmm.fit(X)

print("=== GMM聚类结果 ===")
print(f"收敛对数似然: {gmm.log_likelihood_history_[-1]:.2f}")
print(f"\n估计均值:")
for k, mean in enumerate(gmm.means_):
    print(f"  成分{k}: {mean}")
print(f"\n估计混合系数: {gmm.weights_}")

# 预测
labels = gmm.predict(X)
print(f"\n各成分样本数: {[np.sum(labels == k) for k in range(3)]}")
```

**输出示例：**
```
EM收敛于第15次迭代
=== GMM聚类结果 ===
收敛对数似然: -1152.34

估计均值:
  成分0: [0.12, 0.05]
  成分1: [3.02, 3.08]
  成分2: [0.08, 4.12]

估计混合系数: [0.33 0.34 0.33]

各成分样本数: [102, 98, 100]
```

### 可视化

```python
import matplotlib.pyplot as plt

# 可视化聚类结果
plt.figure(figsize=(12, 5))

# 左图：聚类结果
plt.subplot(1, 2, 1)
colors = ['red', 'blue', 'green']
for k in range(3):
    mask = labels == k
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, label=f'成分{k}')
    # 绘制均值
    plt.scatter(gmm.means_[k, 0], gmm.means_[k, 1], c='black', marker='x', s=100)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('GMM聚类结果')
plt.legend()

# 右图：对数似然收敛曲线
plt.subplot(1, 2, 2)
plt.plot(gmm.log_likelihood_history_)
plt.xlabel('迭代次数')
plt.ylabel('对数似然')
plt.title('EM收敛过程')

plt.tight_layout()
plt.savefig('.history/gmm_clustering.png')
```

---

## 与深度学习对比

### EM算法的价值

在深度学习时代，EM算法及其思想依然有重要价值：

**1. 概率建模基础**

EM是训练概率生成模型的基础工具。许多深度生成模型（如变分自编码器VAE）的理论基础正是EM算法的变分推断思想。

**2. 不确定性量化**

EM算法训练的是概率模型，可以给出预测的不确定性。深度学习模型的预测往往缺乏概率解释，而GMM可以直接输出"这个样本有80%概率属于成分A，20%属于成分B"。

**3. 小样本友好**

GMM等概率模型在样本量较小时表现稳定，而深度学习需要大量数据才能训练。对于几十到几百个样本，GMM往往比深度学习方法更可靠。

**4. 可解释性**

GMM的每个成分可以解释为数据的"模式"或"簇"，均值表示簇中心，协方差表示形状和方向。这种可解释性在很多应用中非常重要。

### EM的局限

当然，EM算法也有局限：

1. **局部最优**：EM只保证收敛到局部最优，对初始化敏感
2. **需要指定成分数**：GMM需要预先指定混合成分数K
3. **高维困难**：高维数据的协方差估计困难，需要正则化或降维
4. **复杂模型受限**：对于高度非线性的数据分布，简单的混合模型表达能力有限

### 与深度学习的结合

现代深度学习与EM思想的结合：

- **VAE（变分自编码器）**：将EM的变分推断思想引入神经网络
- **深度生成模型**：用神经网络参数化生成模型的分布
- **半监督学习**：用EM处理标签缺失问题

---

## 小结

本章介绍了EM算法：

1. **隐变量问题**：存在不可观测变量时的参数估计困难
2. **EM思想**：交替优化隐变量分布（E步）和参数（M步）
3. **收敛性保证**：似然函数单调不减，收敛到局部最优
4. **GMM应用**：高斯混合模型是EM最经典的应用
5. **深度学习关联**：EM思想是现代概率深度学习的理论基础

EM算法是统计学习的重要组成部分。它展示了概率建模的优雅思想：通过迭代推理和优化，从含隐变量的数据中学习模型参数。这种"推断-优化"交替的思想，在现代机器学习中随处可见。