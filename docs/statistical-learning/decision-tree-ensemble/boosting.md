# Boosting——从弱到强的迭代提升

## 引言：序列学习 vs 并行学习

Bagging（随机森林）采用**并行**方式：每棵树独立训练，最后投票。Boosting 采用**序列**方式：每棵新树专注于纠正前一棵树的错误。

想象一个学习小组：
- **Bagging**：每个人独立学习，最后一起讨论综合观点
- **Boosting**：第一个人学习后，第二个人专攻第一个人答错的题，第三个人继续攻克难点...

Boosting 的思想是：**将多个"弱学习器"组合成一个"强学习器"**。每个弱学习器只要比随机猜测好一点即可，通过加权组合，整体性能大幅提升。

---

## Boosting 核心思想

### 加权训练

Boosting 通过调整样本权重来聚焦困难样本：
- 被错误分类的样本：权重增加
- 被正确分类的样本：权重减少

### 序列学习流程

```
初始：所有样本权重相等
for t = 1 to T:
    1. 用当前权重训练弱学习器 h_t
    2. 评估 h_t 的错误率 ε_t
    3. 计算学习器权重 α_t = f(ε_t)
    4. 更新样本权重：错误样本权重增加
```

---

## AdaBoost 原理推导

### 问题设定

给定训练数据 $\{(x_i, y_i)\}_{i=1}^{n}$，其中 $y_i \in \{-1, +1\}$。初始样本权重 $w_i^{(1)} = 1/n$。

### 算法步骤

**第 $t$ 轮迭代：**

1. **训练弱学习器**：用权重 $w^{(t)}$ 训练分类器 $h_t$

2. **计算加权错误率**：
$$\epsilon_t = \frac{\sum_{i=1}^{n} w_i^{(t)} \cdot \mathbb{I}[h_t(x_i) \neq y_i]}{\sum_{i=1}^{n} w_i^{(t)}}$$

3. **计算学习器权重**：
$$\alpha_t = \frac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}$$

错误率越低，$\alpha_t$ 越大，这个学习器在最终模型中话语权越大。

4. **更新样本权重**：
$$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))$$

- 若 $y_i h_t(x_i) = +1$（预测正确）：权重乘以 $e^{-\alpha_t} < 1$，权重减小
- 若 $y_i h_t(x_i) = -1$（预测错误）：权重乘以 $e^{\alpha_t} > 1$，权重增大

### 最终模型

$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$$

加权投票：每个弱学习器按其权重 $\alpha_t$ 贡献投票。

---

## NumPy 实现：手写 AdaBoost

```python
import numpy as np

class DecisionStump:
    """决策桩：单层决策树（AdaBoost常用的弱学习器）"""
    
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1  # 1: <=阈值预测-1; -1: <=阈值预测+1
    
    def fit(self, X, y, sample_weights):
        """训练决策桩"""
        n_samples, n_features = X.shape
        min_error = float('inf')
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X[:, feature] <= threshold] = -1
                    else:
                        predictions[X[:, feature] > threshold] = -1
                    
                    # 加权错误
                    error = np.sum(sample_weights[predictions != y])
                    
                    if error < min_error:
                        min_error = error
                        self.feature = feature
                        self.threshold = threshold
                        self.polarity = polarity
                        self.error = error
        
        return self
    
    def predict(self, X):
        predictions = np.ones(X.shape[0])
        if self.polarity == 1:
            predictions[X[:, self.feature] <= self.threshold] = -1
        else:
            predictions[X[:, self.feature] > self.threshold] = -1
        return predictions


class AdaBoost:
    """AdaBoost实现"""
    
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # 初始化权重
        weights = np.ones(n_samples) / n_samples
        
        self.stumps = []
        self.alphas = []
        
        for t in range(self.n_estimators):
            # 训练决策桩
            stump = DecisionStump()
            stump.fit(X, y, weights)
            
            # 计算错误率
            predictions = stump.predict(X)
            error = np.sum(weights[predictions != y])
            
            # 避免除零
            error = max(error, 1e-10)
            error = min(error, 1 - 1e-10)
            
            # 计算alpha
            alpha = 0.5 * np.log((1 - error) / error)
            
            # 更新权重
            weights = weights * np.exp(-alpha * y * predictions)
            weights = weights / np.sum(weights)  # 归一化
            
            self.stumps.append(stump)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X):
        """加权投票"""
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        for stump, alpha in zip(self.stumps, self.alphas):
            scores += alpha * stump.predict(X)
        
        return np.sign(scores).astype(int)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# 测试：二分类问题
np.random.seed(42)

# 生成数据
n_samples = 200
X = np.random.randn(n_samples, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# 添加噪声
noise_idx = np.random.choice(n_samples, 10, replace=False)
y[noise_idx] = -y[noise_idx]

# 训练AdaBoost
adaboost = AdaBoost(n_estimators=50)
adaboost.fit(X, y)

print("=== AdaBoost分类 ===")
print(f"弱学习器数量: {adaboost.n_estimators}")
print(f"训练准确率: {adaboost.score(X, y):.3f}")

# 观察弱学习器权重变化
print("\n前10个弱学习器的权重(alpha):")
for i in range(min(10, len(adaboost.alphas))):
    print(f"  学习器{i+1}: α = {adaboost.alphas[i]:.4f}")

# 对比单个决策桩
single_stump = DecisionStump()
single_stump.fit(X, y, np.ones(n_samples) / n_samples)
stump_acc = np.mean(single_stump.predict(X) == y)
print(f"\n单个决策桩准确率: {stump_acc:.3f}")
```

**输出示例：**
```
=== AdaBoost分类 ===
弱学习器数量: 50
训练准确率: 0.970

前10个弱学习器的权重(alpha):
  学习器1: α = 0.5493
  学习器2: α = 0.4812
  学习器3: α = 0.4521
  ...

单个决策桩准确率: 0.785
```

---

## GBDT 核心思想：残差拟合

**梯度提升决策树**（Gradient Boosting Decision Tree, GBDT）是 Boosting 的另一种实现。

### 与 AdaBoost 的区别

| 特性 | AdaBoost | GBDT |
|------|----------|------|
| 目标 | 最小化分类错误率 | 最小化损失函数 |
| 权重更新 | 增加错误样本权重 | 拟合残差（负梯度） |
| 输出 | 加权投票 | 累加预测值 |

### GBDT 原理

GBDT 通过**拟合残差**来改进模型：

$$F_t(x) = F_{t-1}(x) + h_t(x)$$

其中 $h_t(x)$ 是新树，拟合目标为**负梯度**（残差的近似）：

$$r_i^{(t)} = -\frac{\partial L(y_i, F_{t-1}(x_i))}{\partial F_{t-1}(x_i)}$$

对于平方损失 $L = \frac{1}{2}(y - F(x))^2$，负梯度就是残差：

$$r_i^{(t)} = y_i - F_{t-1}(x_i)$$

### GBDT 算法

```
初始化 F_0(x) = argmin_c Σ L(y_i, c)
for t = 1 to T:
    1. 计算残差 r_i = y_i - F_{t-1}(x_i)
    2. 用{X, r}训练回归树 h_t
    3. 更新模型 F_t = F_{t-1} + η·h_t (η是学习率)
```

学习率 $\eta$（通常 0.01-0.1）控制每棵树的贡献，防止过拟合。

---

## 与深度学习对比

### 集成方法的价值

在深度学习时代，集成方法（尤其是 GBDT 系列）依然有重要价值：

**1. 工业界主流**

XGBoost、LightGBM、CatBoost 在工业界仍是首选：
- 数据量中等（万到百万级）
- 特征为结构化表格数据
- 需要快速迭代

**2. 不需要 GPU**

GBDT 在 CPU 上训练效率高，部署成本低。

**3. 特征工程友好**

可以直接使用原始特征，不需要复杂的特征提取。

**4. 可解释性**

可以分析特征重要性，理解模型决策过程。

### 与深度学习的互补

| 场景 | 推荐方法 |
|------|----------|
| 表格数据 | GBDT 系列 |
| 图像 / 语音 | 深度学习 |
| 文本 | Transformer/深度学习 |
| 小样本 | GBDT/集成方法 |
| 大样本 + 复杂模式 | 深度学习 |

**现代趋势**：深度学习用于特征提取，GBDT 用于最终预测。

---

## 小结

本章介绍了 Boosting 方法：

1. **Boosting 思想**：序列训练弱学习器，加权组合
2. **AdaBoost**：调整样本权重，聚焦错误样本
3. **GBDT**：拟合残差，梯度下降优化
4. **与深度学习对比**：表格数据场景仍有优势

集成学习展示了"群体智慧"的力量。无论是 Bagging 的并行投票，还是 Boosting 的序列纠错，都证明了一个道理：**组合多个简单模型，往往比一个复杂模型更有效**。