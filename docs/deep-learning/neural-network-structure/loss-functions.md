# 损失函数

在前两章中，我们介绍了反向传播算法和激活函数。反向传播的核心是计算损失函数对参数的梯度，激活函数决定了梯度在传递过程中的衰减程度。但还有一个关键问题尚未解答：损失函数本身如何设计？

损失函数定义了神经网络优化的目标。它衡量预测值与真实值之间的差距，引导参数更新方向。选择合适的损失函数，是神经网络设计的重要决策。不同的任务类型（回归、分类）需要不同的损失函数，损失函数的设计直接影响训练效率和模型性能。

本章将介绍常用损失函数的定义、特性、适用场景，以及正则化项的作用。理解损失函数，是掌握神经网络训练原理的最后一块拼图。

## MSE 与 MAE

### 均方误差 MSE

**均方误差**（Mean Squared Error, MSE）是回归问题最常用的损失函数：

$$L_{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

其中 $y_i$ 是真实值，$\hat{y}_i$ 是预测值，$m$ 是样本数量。

MSE 的特性：
- **凸函数**：全局最优，梯度下降可以收敛
- **二次惩罚**：误差越大，惩罚越重（平方增长）
- **梯度 $L' = 2(\hat{y} - y)$**：梯度与误差成正比，大误差时梯度大，小误差时梯度小

MSE 的优点：
- 数学性质良好，易于优化
- 对大误差敏感，有助于快速纠正明显错误

MSE 的缺点：
- 对异常值敏感：少数极端异常值会显著影响损失和梯度
- 输出层激活函数受限：通常需要线性输出（无范围限制）

### 平均绝对误差 MAE

**平均绝对误差**（Mean Absolute Error, MAE）是 MSE 的替代选择：

$$L_{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|$$

MAE 的特性：
- **线性惩罚**：误差与惩罚成比例，不放大异常值的影响
- **梯度恒定**：$L' = \text{sign}(\hat{y} - y)$，梯度大小固定（$\pm 1$），不随误差变化

MAE 的优点：
- 对异常值更鲁棒：极端异常值影响有限
- 损失值更直观：直接表示平均误差

MAE 的缺点：
- 在 $y = \hat{y}$ 处不可导（梯度跳变）
- 梯度恒定，小误差时梯度仍然较大，可能导致收敛震荡

### MSE vs MAE 对比

| 特性 | MSE | MAE |
|:-----|:----|:----|
| 惩罚方式 | 二次（平方） | 线性（绝对值） |
| 异常值敏感度 | 高（敏感） | 低（鲁棒） |
| 梯度变化 | 与误差成正比 | 恒定（$\pm 1$） |
| 优化难度 | 易（凸函数） | 中（零点不可导） |
| 适用场景 | 数据干净、无异常值 | 数据有异常值 |

### Huber Loss

**Huber Loss** 结合了 MSE 和 MAE 的优点：

$$L_{Huber} = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & |y - \hat{y}| > \delta \end{cases}$$

其中 $\delta$ 是阈值参数。

Huber Loss 的特性：
- 小误差（$|y - \hat{y}| \leq \delta$）：使用 MSE，梯度随误差减小，收敛稳定
- 大误差（$|y - \hat{y}| > \delta$）：使用 MAE，线性惩罚，对异常值鲁棒
- 处处可导，优化友好

Huber Loss 平衡了 MSE 的敏感性和 MAE 的鲁棒性，适合有少量异常值的回归任务。

## 交叉熵损失

### 信息论背景

**交叉熵**（Cross-Entropy）源于信息论，衡量两个概率分布之间的差异。设真实分布为 $P$，预测分布为 $Q$，交叉熵定义为：

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

当 $P = Q$（预测完全正确），交叉熵等于熵 $H(P)$，这是最小值。预测越偏离真实分布，交叉熵越大。

在机器学习中，真实分布 $P$ 由训练数据给定（通常是 one-hot 编码），预测分布 $Q$ 由模型输出。训练目标就是最小化交叉熵，使预测分布逼近真实分布。

### 二分类交叉熵

**二分类交叉熵损失**（Binary Cross-Entropy Loss）用于二分类问题：

$$L_{BCE} = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$$

其中 $y_i \in \{0, 1\}$ 是真实标签，$\hat{y}_i \in (0, 1)$ 是预测概率（通常由 Sigmoid 输出）。

二分类交叉熵可以理解为：
- 当 $y_i = 1$（正类）：损失为 $-\log \hat{y}_i$，预测越接近 1，损失越小
- 当 $y_i = 0$（负类）：损失为 $-\log(1-\hat{y}_i)$，预测越接近 0，损失越小

**梯度分析**：

设输出层使用 Sigmoid，$\hat{y} = \sigma(z)$，则损失对 $z$ 的梯度：

$$\frac{\partial L}{\partial z} = \hat{y} - y$$

这个简化结果与 Softmax + Cross-Entropy 类似：梯度等于预测概率减去真实标签。梯度计算高效，无需显式计算 Sigmoid 导数。

### 多分类交叉熵

**多分类交叉熵损失**（Categorical Cross-Entropy Loss）用于多分类问题：

$$L_{CE} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{ik} \log \hat{y}_{ik}$$

其中 $y_{ik}$ 是真实标签的 one-hot 编码（$y_{ik} = 1$ 表示样本 $i$ 属于类别 $k$），$\hat{y}_{ik}$ 是预测概率（通常由 Softmax 输出），$K$ 是类别数量。

由于 one-hot 编码中只有一个 $y_{ik} = 1$（设为类别 $c$），公式简化为：

$$L_{CE} = -\frac{1}{m} \sum_{i=1}^{m} \log \hat{y}_{ic}$$

即损失等于真实类别预测概率的负对数。

**梯度分析**：

如前两章所述，Softmax + Cross-Entropy 的梯度简化为：

$$\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k$$

这个简化是深度学习训练效率的关键之一。

### 交叉熵的优势

1. **概率语义清晰**：输出是概率分布，符合分类问题的语义
2. **梯度计算高效**：Sigmoid/Softmax + Cross-Entropy 的梯度简化
3. **惩罚非线性**：预测偏离真实时，对数惩罚快速增长
4. **避免梯度消失**：即使预测概率接近 0 或 1，梯度仍有意义

### 交叉熵与 MSE 在分类中的对比

| 特性 | 交叉熵 | MSE（用于分类） |
|:-----|:-------|:---------------|
| 输出解释 | 概率分布 | 数值 |
| 梯度特性 | 梯度与预测误差成比例 | 梯度随预测接近真实而消失 |
| 概率边界 | 自然约束输出在 $(0,1)$ | 无约束，可能超出 $(0,1)$ |
| 适用性 | 分类问题的标准选择 | 分类中表现较差 |

为什么 MSE 不适合分类？设使用 Sigmoid 输出，MSE 损失：

$$L_{MSE} = (y - \sigma(z))^2$$

梯度：

$$\frac{\partial L}{\partial z} = 2(y - \sigma(z)) \cdot \sigma'(z)$$

当预测接近真实（$\sigma(z) \approx y$），梯度趋近于 0。但更重要的是，当 $y=1$ 且预测正确（$\sigma(z) \approx 1$），$\sigma'(z) \approx 0$，梯度消失。这是分类任务中 MSE 的致命缺陷。

## Hinge Loss

### Hinge Loss 的定义

**Hinge Loss**（合页损失）主要用于支持向量机（SVM）和最大间隔分类：

$$L_{Hinge} = \max(0, 1 - y \cdot \hat{y})$$

其中 $y \in \{-1, 1\}$ 是真实标签（注意不是 $\{0, 1\}$），$\hat{y}$ 是预测值（可以是任意实数，不一定是概率）。

Hinge Loss 的含义：
- 当 $y \cdot \hat{y} \geq 1$：预测正确且置信度高，损失为 0
- 当 $0 < y \cdot \hat{y} < 1$：预测正确但置信度不足，损失为 $1 - y \cdot \hat{y}$
- 当 $y \cdot \hat{y} < 0$：预测错误，损失为 $1 - y \cdot \hat{y} > 1$

### Hinge Loss 的特性

1. **最大间隔原则**：要求预测值 $y \cdot \hat{y} \geq 1$，不仅是分类正确，还要有足够置信度。这鼓励模型学习更大的分类间隔，提高泛化能力。

2. **稀疏梯度**：只有当 $y \cdot \hat{y} < 1$ 时才有梯度，正确分类且置信度足够时梯度为 0。这使 SVM 的支持向量（决定决策边界的样本）较少。

3. **分段线性**：损失函数分段线性，优化相对简单。

### Hinge Loss vs Cross-Entropy

| 特性 | Hinge Loss | Cross-Entropy |
|:-----|:-----------|:--------------|
| 输出范围 | 实数 | 概率 $(0,1)$ |
| 优化目标 | 最大间隔 | 概率逼近 |
| 梯度特性 | 稀疏（置信度足够时无梯度） | 持续（总是有梯度） |
| 适用模型 | SVM | 神经网络 |

神经网络中很少直接使用 Hinge Loss，因为 Cross-Entropy 的梯度特性更适合梯度下降优化。但 Hinge Loss 的最大间隔思想影响了神经网络的设计，如 Large Margin Softmax Loss 等。

## 损失函数选择原则

### 按任务类型选择

| 任务类型 | 推荐损失函数 | 输出层激活 |
|:---------|:-----------|:----------|
| 回归（无异常值） | MSE | Linear |
| 回归（有异常值） | MAE / Huber | Linear |
| 二分类 | Binary Cross-Entropy | Sigmoid |
| 多分类 | Categorical Cross-Entropy | Softmax |
| 多标签分类 | Binary Cross-Entropy（每类） | Sigmoid |

### 选择原则

1. **回归问题**：
   - 数据干净、无异常值 → MSE
   - 数据有异常值 → MAE 或 Huber Loss
   - 需要快速收敛 → MSE（梯度大）
   - 需要稳定收敛 → MAE 或 Huber

2. **分类问题**：
   - 几乎总是使用 Cross-Entropy
   - 输出层配合对应的激活函数（Sigmoid/Softmax）
   - 避免使用 MSE 做分类损失

3. **多标签分类**：
   - 每个标签独立使用 Binary Cross-Entropy
   - 输出层使用 Sigmoid（每个标签一个概率）

4. **特殊情况**：
   - 概率分布输出 → Cross-Entropy
   - 最大间隔需求 → Hinge Loss（但神经网络中较少使用）

## 正则化项

### 正则化的作用

损失函数的核心是衡量预测误差，但单纯的误差最小化可能导致**过拟合**——模型过度拟合训练数据中的噪声，在新数据上表现糟糕。

**正则化**（Regularization）通过在损失函数中添加参数惩罚项，约束模型复杂度，防止过拟合：

$$L_{total} = L_{data} + \lambda R(\mathbf{W})$$

其中 $L_{data}$ 是数据损失（MSE、Cross-Entropy 等），$R(\mathbf{W})$ 是正则化项，$\lambda$ 是正则化系数（控制正则化强度）。

### L1 正则化

**L1 正则化**（Lasso Regularization）使用参数的绝对值之和作为惩罚：

$$R_{L1} = \sum_{i} |W_i|$$

L1 正则化的特性：
- **稀疏性**：L1 正则化倾向于使部分权重精确为 0，实现特征选择
- **不可导点**：$W_i = 0$ 处不可导，优化稍复杂
- **适用场景**：需要稀疏解、特征选择

### L2 正则化

**L2 正则化**（Ridge Regularization / Weight Decay）使用参数的平方和作为惩罚：

$$R_{L2} = \sum_{i} W_i^2 = ||\mathbf{W}||^2$$

L2 正则化的特性：
- **平滑性**：处处可导，优化友好
- **权重衰减**：鼓励权重整体变小，但不精确为 0
- **防止过拟合**：限制模型复杂度，改善泛化

**L2 正则化的梯度**：

$$\frac{\partial R_{L2}}{\partial W_i} = 2W_i$$

参数更新时：

$$W_i \leftarrow W_i - \eta \frac{\partial L_{data}}{\partial W_i} - \eta \lambda \cdot 2W_i = W_i(1 - 2\eta\lambda) - \eta \frac{\partial L_{data}}{\partial W_i}$$

这解释了 L2 正则化为何称为"权重衰减"（Weight Decay）——每步更新权重乘以 $(1 - 2\eta\lambda) < 1$，权重逐渐衰减。

### Elastic Net

**Elastic Net** 结合 L1 和 L2 正则化：

$$R_{ElasticNet} = \alpha \sum_i |W_i| + (1-\alpha) \sum_i W_i^2$$

其中 $\alpha \in [0, 1]$ 控制 L1 和 L2 的比例。

Elastic Net 平衡稀疏性和平滑性：既能实现特征选择，又能保持优化友好。

### 正则化系数的选择

正则化系数 $\lambda$ 是重要超参数：
- $\lambda$ 太小：正则化不足，仍可能过拟合
- $\lambda$ 太大：正则化过度，模型表达能力受限，欠拟合

选择方法：
- **交叉验证**：尝试多个 $\lambda$ 值，选择验证集表现最好的
- **经验范围**：通常 $\lambda \in [10^{-4}, 10^{-2}]$（神经网络）
- **网格搜索**：系统搜索 $\lambda$ 的最优值

## 损失函数实验

下面通过代码实验对比不同损失函数在回归和分类任务中的表现。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# ===== 第一部分：回归损失函数对比 =====

print("=" * 60)
print("实验1：回归损失函数对比（MSE vs MAE vs Huber）")
print("=" * 60)

# 生成回归数据（包含异常值）
np.random.seed(42)
n_samples = 50

# 正常数据
X_normal = np.linspace(0, 10, n_samples)
y_normal = 2 * X_normal + 1 + np.random.randn(n_samples) * 0.5

# 添加几个异常值
n_outliers = 5
outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
y_normal[outlier_indices] += np.random.randn(n_outliers) * 15  # 大偏差

X = X_normal
y_true = y_normal

# 定义损失函数
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mae_loss(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def huber_loss(y_pred, y_true, delta=1.0):
    diff = np.abs(y_pred - y_true)
    return np.mean(np.where(diff <= delta, 
                            0.5 * diff ** 2, 
                            delta * diff - 0.5 * delta ** 2))

# 简单线性回归（使用梯度下降）
class LinearRegression:
    def __init__(self, loss_type='mse', learning_rate=0.01, n_iterations=1000, delta=1.0):
        self.loss_type = loss_type
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.delta = delta
        self.w = None
        self.b = None
        self.loss_history = []
    
    def fit(self, X, y):
        # 初始化参数
        self.w = 0.0
        self.b = 0.0
        
        for i in range(self.n_iter):
            # 预测
            y_pred = self.w * X + self.b
            
            # 计算损失
            if self.loss_type == 'mse':
                loss = mse_loss(y_pred, y)
            elif self.loss_type == 'mae':
                loss = mae_loss(y_pred, y)
            elif self.loss_type == 'huber':
                loss = huber_loss(y_pred, y, self.delta)
            
            self.loss_history.append(loss)
            
            # 计算梯度
            if self.loss_type == 'mse':
                dw = 2 * np.mean((y_pred - y) * X)
                db = 2 * np.mean(y_pred - y)
            elif self.loss_type == 'mae':
                dw = np.mean(np.sign(y_pred - y) * X)
                db = np.mean(np.sign(y_pred - y))
            elif self.loss_type == 'huber':
                diff = y_pred - y
                indicator = np.where(np.abs(diff) <= self.delta, diff, self.delta * np.sign(diff))
                dw = np.mean(indicator * X)
                db = np.mean(indicator)
            
            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db
        
        return self
    
    def predict(self, X):
        return self.w * X + self.b

# 训练三个模型
models = {
    'MSE': LinearRegression(loss_type='mse', learning_rate=0.01, n_iterations=500),
    'MAE': LinearRegression(loss_type='mae', learning_rate=0.01, n_iterations=500),
    'Huber': LinearRegression(loss_type='huber', learning_rate=0.01, n_iterations=500, delta=5.0)
}

for name, model in models.items():
    model.fit(X, y_true)
    print(f"{name}: w={model.w:.3f}, b={model.b:.3f}, 最终损失={model.loss_history[-1]:.3f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：拟合结果对比
ax1 = axes[0]
ax1.scatter(X, y_true, c=['red' if i in outlier_indices else 'blue' for i in range(n_samples)],
           alpha=0.6, label='数据点（红色为异常值）')

for name, model in models.items():
    y_pred = model.predict(X)
    ax1.plot(X, y_pred, linewidth=2, label=f'{name}: y={model.w:.2f}x+{model.b:.2f}')

# 理想直线（无异常值影响）
y_ideal = 2 * X + 1
ax1.plot(X, y_ideal, 'g--', linewidth=2, label='理想: y=2x+1')

ax1.set_xlabel('X', fontsize=11)
ax1.set_ylabel('y', fontsize=11)
ax1.set_title('回归损失函数对比：异常值影响', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 图2：损失变化
ax2 = axes[1]
for name, model in models.items():
    ax2.plot(model.loss_history, linewidth=2, label=name)

ax2.set_xlabel('迭代次数', fontsize=11)
ax2.set_ylabel('损失值', fontsize=11)
ax2.set_title('训练过程损失变化', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n回归实验结论:")
print("- MSE 受异常值影响最大，拟合线偏离理想线")
print("- MAE 对异常值最鲁棒，拟合线接近理想线")
print("- Huber Loss 平衡两者，效果适中")
print("-" * 60)


# ===== 第二部分：分类损失函数对比 =====

print("\n" + "=" * 60)
print("实验2：分类损失函数对比（Cross-Entropy vs MSE）")
print("=" * 60)

# 生成二分类数据
np.random.seed(123)
n_class_samples = 100

# 类别0
X0 = np.random.randn(n_class_samples, 2) + np.array([-2, -2])
y0 = np.zeros(n_class_samples)

# 类别1
X1 = np.random.randn(n_class_samples, 2) + np.array([2, 2])
y1 = np.ones(n_class_samples)

X_class = np.vstack([X0, X1])
y_class = np.hstack([y0, y1])

# 简单逻辑回归
class LogisticRegression:
    def __init__(self, loss_type='ce', learning_rate=0.1, n_iterations=1000):
        self.loss_type = loss_type
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.w = None
        self.b = None
        self.loss_history = []
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        for i in range(self.n_iter):
            # 预测
            z = X @ self.w + self.b
            y_pred = self.sigmoid(z)
            
            # 计算损失
            if self.loss_type == 'ce':
                eps = 1e-15
                y_pred = np.clip(y_pred, eps, 1 - eps)
                loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            elif self.loss_type == 'mse':
                loss = np.mean((y - y_pred) ** 2)
            
            self.loss_history.append(loss)
            
            # 计算梯度
            if self.loss_type == 'ce':
                # Cross-Entropy + Sigmoid 的简化梯度
                dz = y_pred - y
            elif self.loss_type == 'mse':
                # MSE + Sigmoid 的梯度
                dz = 2 * (y_pred - y) * y_pred * (1 - y_pred)
            
            dw = np.mean(dz.reshape(-1, 1) * X, axis=0)
            db = np.mean(dz)
            
            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db
        
        return self
    
    def predict_proba(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

# 训练两个模型
model_ce = LogisticRegression(loss_type='ce', learning_rate=0.1, n_iterations=500)
model_mse = LogisticRegression(loss_type='mse', learning_rate=0.1, n_iterations=500)

model_ce.fit(X_class, y_class)
model_mse.fit(X_class, y_class)

print(f"Cross-Entropy: 准确率 {np.mean(model_ce.predict(X_class) == y_class):.2%}")
print(f"MSE: 准确率 {np.mean(model_mse.predict(X_class) == y_class):.2%}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：决策边界
ax1 = axes[0]
ax1.scatter(X0[:, 0], X0[:, 1], c='blue', alpha=0.6, label='类别0')
ax1.scatter(X1[:, 0], X1[:, 1], c='red', alpha=0.6, label='类别1')

# 绘制决策边界
x_min, x_max = X_class[:, 0].min() - 1, X_class[:, 0].max() + 1
y_min, y_max = X_class[:, 1].min() - 1, X_class[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = np.column_stack([xx.ravel(), yy.ravel()])

Z_ce = model_ce.predict_proba(grid).reshape(xx.shape)
Z_mse = model_mse.predict_proba(grid).reshape(xx.shape)

ax1.contour(xx, yy, Z_ce, levels=[0.5], colors='green', linewidths=2, linestyles='-', label='CE边界')
ax1.contour(xx, yy, Z_mse, levels=[0.5], colors='orange', linewidths=2, linestyles='--', label='MSE边界')

ax1.set_xlabel('x1', fontsize=11)
ax1.set_ylabel('x2', fontsize=11)
ax1.set_title('分类损失函数对比：决策边界', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 图2：损失变化
ax2 = axes[1]
ax2.plot(model_ce.loss_history, linewidth=2, color='green', label='Cross-Entropy')
ax2.plot(model_mse.loss_history, linewidth=2, color='orange', label='MSE')

ax2.set_xlabel('迭代次数', fontsize=11)
ax2.set_ylabel('损失值', fontsize=11)
ax2.set_title('训练过程损失变化', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n分类实验结论:")
print("- Cross-Entropy 收敛更快，损失下降迅速")
print("- MSE 收敛较慢，后期损失下降缓慢")
print("- 两者的决策边界接近，但 Cross-Entropy 效率更高")
print("- 这解释了为何 Cross-Entropy 是分类问题的标准选择")
print("=" * 60)
```

### 实验结论

**回归实验**：
1. MSE 对异常值敏感，拟合线偏离理想线
2. MAE 对异常值鲁棒，拟合线接近理想线
3. Huber Loss 平衡两者，效果适中

**分类实验**：
1. Cross-Entropy 收敛更快，效率更高
2. MSE 在分类中梯度消失问题导致收敛缓慢
3. Cross-Entropy 是分类问题的标准选择

## 本章小结

本章详细介绍了神经网络损失函数的定义、特性、适用场景，以及正则化的作用。核心要点如下：

1. **回归损失函数**：MSE 二次惩罚大误差，对异常值敏感；MAE 线性惩罚，对异常值鲁棒；Huber Loss 结合两者优点。回归问题根据数据特征选择合适的损失函数。

2. **分类损失函数**：Cross-Entropy 是分类问题的标准选择。它衡量概率分布差异，配合 Sigmoid/Softmax 输出层，梯度计算高效简洁。避免使用 MSE 做分类损失——MSE 在分类中存在梯度消失问题。

3. **Hinge Loss**：最大间隔损失，用于 SVM。鼓励模型学习更大的分类间隔，提高泛化能力。神经网络中较少直接使用，但其思想影响了其他损失函数设计。

4. **损失函数选择原则**：
   - 回归无异常值 → MSE
   - 回归有异常值 → MAE 或 Huber
   - 二分类 → Binary Cross-Entropy + Sigmoid
   - 多分类 → Categorical Cross-Entropy + Softmax

5. **正则化**：在损失函数中添加参数惩罚项，防止过拟合。L1 正则化产生稀疏解，实现特征选择；L2 正则化使权重整体衰减，改善泛化。正则化系数 $\lambda$ 是重要超参数，需要通过交叉验证选择。

损失函数定义了神经网络优化的目标，是训练的核心组件。理解各损失函数的特性，根据任务类型选择合适的损失函数，并适当使用正则化防止过拟合，是深度学习实践者的必备技能。至此，第二章"深度神经网络建模"的内容已全部完成，我们掌握了反向传播、激活函数、损失函数三大核心组件。下一章将进入第三章"深度神经网络优化"，介绍梯度下降算法和自适应优化器。

## 练习题

1. 证明当使用 Sigmoid 输出层配合 Binary Cross-Entropy 损失时，梯度 $\frac{\partial L}{\partial z} = \hat{y} - y$。这个简化有什么意义？
    <details>
    <summary>参考答案</summary>
    
    **梯度推导**：
    
    设 Sigmoid 输出 $\hat{y} = \sigma(z) = \frac{1}{1+e^{-z}}$，Binary Cross-Entropy 损失 $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$。
    
    计算损失对 $z$ 的梯度：
    
    $$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$
    
    首先，$\frac{\partial L}{\partial \hat{y}}$：
    
    $$\frac{\partial L}{\partial \hat{y}} = -\left[\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}\right] = -\frac{y(1-\hat{y}) - (1-y)\hat{y}}{\hat{y}(1-\hat{y})}$$
    
    $$= -\frac{y - y\hat{y} - \hat{y} + y\hat{y}}{\hat{y}(1-\hat{y})} = -\frac{y - \hat{y}}{\hat{y}(1-\hat{y})}$$
    
    然后，$\frac{\partial \hat{y}}{\partial z}$（Sigmoid 导数）：
    
    $$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1-\hat{y})$$
    
    代入：
    
    $$\frac{\partial L}{\partial z} = -\frac{y - \hat{y}}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) = \hat{y} - y$$
    
    **简化意义**：
    
    1. **计算高效**：无需显式计算 Sigmoid 导数，直接取预测概率与真实标签的差值。省去复杂的导数计算，提高训练效率。
    
    2. **数值稳定**：避免了单独计算 $\frac{1}{\hat{y}}$ 和 $\frac{1}{1-\hat{y}}$ 可能导致的数值问题（当 $\hat{y}$ 接近 0 或 1）。
    
    3. **梯度直观**：误差信号 $\hat{y} - y$ 直观表示"预测误差"：
       - 预测正确（$\hat{y} = y$）：梯度为 0
       - 预测偏高（$\hat{y} > y$）：梯度为正，参数向减小预测方向更新
       - 预测偏低（$\hat{y} < y$）：梯度为负，参数向增大预测方向更新
    
    4. **避免梯度消失**：即使 $\hat{y}$ 接近 0 或 1，梯度仍与预测误差成比例，不会消失。这与 MSE + Sigmoid 不同（后者在预测接近真实时梯度消失）。
    
    5. **统一形式**：与 Softmax + Cross-Entropy 的梯度形式一致（都是 $\hat{y} - y$），便于理解和实现。
    
    **总结**：Sigmoid + Binary Cross-Entropy 的梯度简化是深度学习训练效率的关键之一。它使梯度计算简洁高效，同时避免了数值问题和梯度消失。这就是为什么 Cross-Entropy 是分类问题的标准损失函数。
    </details>

2. 分析 MSE 在分类任务中为何表现不佳。设使用 Sigmoid 输出层，推导 MSE 损失的梯度，并说明其缺陷。
    <details>
    <summary>参考答案</summary>
    
    **MSE 损失定义**：
    
    设 Sigmoid 输出 $\hat{y} = \sigma(z)$，MSE 损失 $L = \frac{1}{2}(y - \hat{y})^2$（二分类单样本）。
    
    **梯度推导**：
    
    $$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$
    
    $$\frac{\partial L}{\partial \hat{y}} = -(y - \hat{y})$$
    
    $$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})$$
    
    代入：
    
    $$\frac{\partial L}{\partial z} = -(y - \hat{y}) \cdot \hat{y}(1 - \hat{y}) = (y - \hat{y}) \cdot \hat{y}(1 - \hat{y})$$
    
    （注意符号：$\frac{\partial L}{\partial \hat{y}} = -(y-\hat{y})$，所以 $\frac{\partial L}{\partial z} = -(y-\hat{y})\cdot\hat{y}(1-\hat{y})$）
    
    或者写成：
    
    $$\frac{\partial L}{\partial z} = (\hat{y} - y) \cdot \hat{y}(1 - \hat{y})$$
    
    **MSE 在分类中的缺陷**：
    
    1. **梯度消失问题**：
    
    当 $y=1$ 且预测接近正确（$\hat{y} \approx 1$）：
    - $\hat{y}(1-\hat{y}) \approx 1 \cdot 0 = 0$
    - 梯度 $\frac{\partial L}{\partial z} \approx 0$
    
    同样，当 $y=0$ 且预测接近正确（$\hat{y} \approx 0$）：
    - $\hat{y}(1-\hat{y}) \approx 0 \cdot 1 = 0$
    - 梯度 $\frac{\partial L}{\partial z} \approx 0$
    
    这意味着当预测接近正确时，梯度消失，参数几乎不更新。但此时损失并未达到最小（预测仍有改进空间），模型无法继续优化。
    
    2. **惩罚不合理**：
    
    MSE 假设预测值和真实值都是连续数值，对差值平方惩罚。但分类问题的真实值是类别标签（0或1），MSE 的平方惩罚在语义上不合适。
    
    3. **输出范围不约束**：
    
    MSE 不约束输出范围，理论上优化结果可能使 $\hat{y}$ 超出 $(0,1)$（虽然 Sigmoid 自然约束，但 MSE 的目标函数不尊重概率语义）。
    
    4. **收敛缓慢**：
    
    由于梯度消失问题，MSE 在分类任务中收敛比 Cross-Entropy 慢。实验表明，Cross-Entropy 在相同迭代次数下通常达到更好的精度。
    
    **与 Cross-Entropy 对比**：
    
    Cross-Entropy 的梯度 $\frac{\partial L}{\partial z} = \hat{y} - y$：
    - 与预测误差成比例，不随预测接近真实而消失
    - 只要预测不完全正确，梯度就有意义
    - 收敛更快，效率更高
    
    **总结**：MSE 在分类任务中的缺陷源于其梯度包含 Sigmoid 导数因子 $\hat{y}(1-\hat{y})$，当预测接近真实时该因子趋近于 0，导致梯度消失。Cross-Entropy 通过巧妙设计，梯度恰好消去这个因子，避免梯度消失。这是 Cross-Entropy 成为分类问题标准损失函数的核心原因。
    </details>

3. 解释 L1 正则化为何能产生稀疏解（权重精确为 0），而 L2 正则化只能使权重衰减但不会精确为 0。
    <details>
    <summary>参考答案</summary>
    
    **L1 正则化的稀疏性**：
    
    L1 正则化损失：$L = L_{data} + \lambda \sum_i |W_i|$
    
    L1 的导数（在 $W_i \neq 0$ 处）：$\frac{\partial L}{\partial W_i} = \frac{\partial L_{data}}{\partial W_i} + \lambda \cdot \text{sign}(W_i)$
    
    在 $W_i = 0$ 处不可导，导数从 $-\lambda$ 蛇变到 $+\lambda$。
    
    当 $W_i$ 很小（接近 0）时：
    - 如果数据梯度 $\frac{\partial L_{data}}{\partial W_i}$ 小于 $\lambda$，L1 正则化会将 $W_i$ 推向精确的 0
    - 具体机制：参数更新 $W_i \leftarrow W_i - \eta(\frac{\partial L_{data}}{\partial W_i} + \lambda \cdot \text{sign}(W_i))$
    - 当 $W_i$ 为正且数据梯度小，更新使 $W_i$ 减小，可能穿过 0 变为负；继续更新又会推向 0
    
    数学解释：L1 的惩罚是分段线性的，在 $W_i = 0$ 处形成一个"尖点"。优化算法倾向于收敛到这些尖点，因为尖点处导数不连续，梯度方向的约束更宽松。
    
    **L2 正则化为何不产生稀疏解**：
    
    L2 正则化损失：$L = L_{data} + \lambda \sum_i W_i^2$
    
    L2 的导数：$\frac{\partial L}{\partial W_i} = \frac{\partial L_{data}}{\partial W_i} + 2\lambda W_i$
    
    参数更新：$W_i \leftarrow W_i - \eta(\frac{\partial L_{data}}{\partial W_i} + 2\lambda W_i) = W_i(1 - 2\eta\lambda) - \eta\frac{\partial L_{data}}{\partial W_i}$
    
    L2 的惩罚是平滑的二次函数：
    - 没有"尖点"，导数连续
    - $W_i$ 接近 0 时，正则化梯度 $2\lambda W_i$ 也接近 0
    - 只要数据梯度 $\frac{\partial L_{data}}{\partial W_i}$ 不为 0，参数就会继续更新，不会停在 0
    
    **几何解释**：
    
    设参数空间为二维 $(W_1, W_2)$，绘制正则化约束的等值线：
    
    - **L1 正则化**：约束 $|W_1| + |W_2| \leq t$，等值线是菱形（正方形旋转）
    - **L2 正则化**：约束 $W_1^2 + W_2^2 \leq t$，等值线是圆形
    
    数据损失的等值线（假设为椭圆）与正则化约束的交点：
    - L1：椭圆与菱形顶点相交，顶点恰好在轴上（$W_1=0$ 或 $W_2=0$），产生稀疏解
    - L2：椭圆与圆相切，切点一般不在轴上，不会产生精确的 0
    
    **总结**：L1 正则化产生稀疏解的原因是其惩罚函数在 $W=0$ 处形成尖点（导数不连续），优化倾向于收敛到这些尖点。L2 正则化的惩罚函数平滑（导数连续），$W$ 接近 0 时正则化梯度也接近 0，不会将 $W$ 推向精确的 0。这使得 L1 正则化适合特征选择（删除不重要特征），而 L2 正则化适合防止过拟合（整体减小权重）。
    </details>

4. 设训练一个神经网络，训练损失很低但验证损失很高。分析可能的原因，并提出三种缓解方法。
    <details>
    <summary>参考答案</summary>
    
    **过拟合的诊断**：
    
    训练损失低、验证损失高，这是典型的**过拟合**（Overfitting）症状。模型过度拟合训练数据中的噪声和特定模式，在新数据上表现糟糕。
    
    **可能的原因**：
    
    1. **模型复杂度过高**：网络参数过多、隐藏层神经元过多、层数过深。模型表达能力超过问题需求，可以拟合训练数据中的噪声。
    
    2. **训练数据不足**：数据规模太小，无法提供足够信息学习真正的规律。模型转向拟合有限的训练样本。
    
    3. **训练时间过长**：过度训练使模型从学习规律转向记忆数据。后期训练主要拟合噪声而非改进泛化。
    
    4. **缺乏正则化**：损失函数中没有参数惩罚，模型自由度过高。
    
    5. **数据分布问题**：训练数据和验证数据分布不一致，模型学到的模式不适用于验证数据。
    
    **缓解方法**：
    
    1. **增加正则化**：
    
    - **L2 正则化**（权重衰减）：在损失函数中添加 $\lambda||\mathbf{W}||^2$
    - 效果：约束权重大小，限制模型复杂度，防止过拟合
    - 实现：在优化器中设置 `weight_decay` 参数
    
    ```python
    # PyTorch示例
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    ```
    
    2. **Dropout**：
    
    - 训练时随机丢弃部分神经元（输出置0）
    - 效果：降低过拟合风险，类似集成学习效果
    - 原理：每个神经元不能过度依赖其他神经元，学习更鲁棒的特征
    
    ```python
    # PyTorch示例
    dropout = nn.Dropout(p=0.5)  # 丢弃概率50%
    ```
    
    3. **早停**（Early Stopping）：
    
    - 监控验证损失，当验证损失开始上升时停止训练
    - 效果：防止过度训练
    - 原理：训练后期主要拟合噪声，及时停止可以保留泛化能力
    
    ```python
    # 伪代码
    best_val_loss = float('inf')
    patience = 10  # 容忍轮数
    for epoch in range(max_epochs):
        train_loss = train_one_epoch()
        val_loss = validate()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break  # 早停
    ```
    
    **其他方法**：
    
    - **减少模型复杂度**：减少隐藏层神经元数量、减少层数
    - **增加训练数据**：收集更多数据，或使用数据增强（如图像旋转、裁剪）
    - **数据预处理**：标准化数据分布，确保训练和验证数据一致
    - **交叉验证**：使用 K-fold 交叉验证更准确地评估泛化能力
    
    **总结**：训练损失低、验证损失高表明过拟合。缓解方法包括增加正则化（L2、Dropout）、早停、减少模型复杂度、增加数据。正则化是防止过拟合的核心手段，应该作为默认配置使用。早停是实用技巧，可以有效防止过度训练。
    </details>