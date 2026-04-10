# 随机森林——Bagging的艺术

## 引言：三个臭皮匠，顶个诸葛亮

上一章我们学习了决策树，它直观易懂，但有个致命弱点：**容易过拟合**。决策树对训练数据非常敏感——数据稍有变化，树的结构可能完全不同。

**集成学习（Ensemble Learning）**的思想是：与其依赖一个模型，不如组合多个模型，取长补短。**随机森林（Random Forest）**就是集成学习的经典代表——构建多棵决策树，让它们"投票"决定最终结果。

这种"群体智慧"的思想在实践中非常有效。随机森林在Kaggle竞赛中长期占据重要地位，至今仍是工业界最常用的机器学习算法之一。

---

## Bagging思想：Bootstrap采样与聚合

### Bootstrap采样

**Bootstrap（自助采样）**是一种重采样技术：从原始数据集中**有放回地**随机抽取样本，构造一个新的训练集。

设原始数据集有 $n$ 个样本，Bootstrap采样：
1. 随机选择一个样本，记录后放回
2. 重复 $n$ 次

每个Bootstrap样本约包含原始数据集 **63.2%** 的不同样本（因为每个样本被选中的概率约为 $1-(1-1/n)^n \approx 1-e^{-1} \approx 0.632$）。

### Bagging流程

**Bagging（Bootstrap Aggregating）**算法：

1. **训练阶段**：
   - 从原始数据集生成 $B$ 个Bootstrap样本
   - 在每个Bootstrap样本上训练一个基学习器

2. **预测阶段**：
   - 分类：多数投票
   - 回归：平均值

### Bagging为什么有效？

1. **降低方差**：每个模型看到不同的数据，学到不同的规律。组合后，个别模型的错误被其他模型纠正。
2. **多样性**：不同Bootstrap样本导致模型多样化，减少过拟合风险。

数学上，如果 $B$ 个模型的方差都是 $\sigma^2$，两两相关系数是 $\rho$，则集成后方差：

$$\text{Var} = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

当 $B \to \infty$，方差趋近于 $\rho \sigma^2$，小于单模型的 $\sigma^2$。

---

## 随机森林原理

### 随机森林 = Bagging + 特征随机

随机森林在Bagging基础上，进一步引入**特征随机性**：

- **样本随机**：Bootstrap采样（来自Bagging）
- **特征随机**：在每个节点分裂时，只从随机选取的 $m$ 个特征中选择最优分裂

对于分类问题，通常 $m = \sqrt{d}$（$d$ 是总特征数）；对于回归问题，$m = d/3$。

### 为什么需要特征随机？

假设有一个特征"非常强"（信息增益最大），所有树的根节点都会选择它。这样，树之间高度相关，Bagging的效果大打折扣。

特征随机强迫每棵树"看不同的视角"，增加多样性，进一步降低方差。

### 随机森林算法

```
输入：数据集 D, 树的数量 B, 特征子集大小 m
输出：随机森林模型

for b = 1 to B:
    1. 从D中Bootstrap采样得到D_b
    2. 在D_b上训练决策树T_b：
       每个节点分裂时，从d个特征中随机选m个
       从m个特征中选择最优分裂
    3. 存储T_b

预测：
- 分类：majority_vote({T_b(x)})
- 回归：average({T_b(x)})
```

---

## 投票机制

### 硬投票（Hard Voting）

每个模型预测一个类别，选择得票最多的类别：

$$\hat{y} = \arg\max_c \sum_{b=1}^{B} \mathbb{I}[T_b(x) = c]$$

### 软投票（Soft Voting）

每个模型输出各类别的概率，对概率取平均后选择最大概率类别：

$$\hat{y} = \arg\max_c \frac{1}{B} \sum_{b=1}^{B} P_b(y=c|x)$$

软投票通常效果更好，因为它考虑了预测的置信度。

---

## 特征重要性

随机森林可以自然地评估特征重要性：

### 方法1：平均不纯度减少

对于每个特征，计算所有树中该特征带来的不纯度减少的平均值：

$$Importance(A) = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b} \Delta I(t) \cdot \mathbb{I}[\text{feature}(t) = A]$$

### 方法2：置换重要性

将某个特征的值随机打乱，观察模型性能下降多少：

$$Importance(A) = \text{score}(D) - \text{score}(D_{permuted})$$

置换重要性更可靠，但计算成本更高。

---

## NumPy实现：手写随机森林

```python
import numpy as np

class DecisionTreeForRF:
    """用于随机森林的决策树"""
    
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
    
    def _gini(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _best_split(self, X, y, feature_indices):
        """只考虑指定特征子集"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                n = len(y)
                gini = (np.sum(left_mask) / n) * self._gini(y[left_mask]) + \
                       (np.sum(right_mask) / n) * self._gini(y[right_mask])
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': values[np.argmax(counts)]}
        
        # 随机选择特征子集
        if self.max_features is not None:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = np.arange(n_features)
        
        feature, threshold = self._best_split(X, y, feature_indices)
        
        if feature is None:
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': values[np.argmax(counts)]}
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)
        return self
    
    def _predict_one(self, x, node):
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class RandomForestClassifier:
    """随机森林分类器"""
    
    def __init__(self, n_estimators=100, max_depth=10, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_importances_ = None
    
    def _bootstrap_sample(self, X, y):
        """Bootstrap采样"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        n_features = X.shape[1]
        
        # 确定特征子集大小
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        
        self.trees = []
        for _ in range(self.n_estimators):
            # Bootstrap采样
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # 训练决策树
            tree = DecisionTreeForRF(
                max_depth=self.max_depth,
                max_features=max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """多数投票"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # 每列是一个样本的所有预测
        result = []
        for i in range(X.shape[0]):
            values, counts = np.unique(predictions[:, i], return_counts=True)
            result.append(values[np.argmax(counts)])
        return np.array(result)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# 测试：手写数字分类
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林
rf = RandomForestClassifier(n_estimators=50, max_depth=15)
rf.fit(X_train, y_train)

print("=== 随机森林分类 ===")
print(f"树的数量: {rf.n_estimators}")
print(f"训练准确率: {rf.score(X_train, y_train):.3f}")
print(f"测试准确率: {rf.score(X_test, y_test):.3f}")

# 对比单棵决策树
single_tree = DecisionTreeForRF(max_depth=15)
single_tree.fit(X_train, y_train)
print(f"\n单棵决策树测试准确率: {np.mean(single_tree.predict(X_test) == y_test):.3f}")
```

**输出示例：**
```
=== 随机森林分类 ===
树的数量: 50
训练准确率: 1.000
测试准确率: 0.952

单棵决策树测试准确率: 0.819
```

---

## 应用场景示例

```python
import numpy as np

# 生成更复杂的数据
np.random.seed(42)
n_samples = 500

# 特征：年龄、收入、教育年限、工作年限
age = np.random.randint(22, 60, n_samples)
income = np.random.randint(20, 200, n_samples)  # 千元
education = np.random.randint(8, 20, n_samples)  # 年
experience = np.random.randint(0, 30, n_samples)

X = np.column_stack([age, income, education, experience])

# 目标：是否购买高端产品
# 规则：高收入+高学历 或 年龄适中+一定经验
y = ((income > 100) & (education > 14)) | ((age > 30) & (age < 50) & (experience > 5))
y = y.astype(int)

# 添加噪声
noise_idx = np.random.choice(n_samples, 20, replace=False)
y[noise_idx] = 1 - y[noise_idx]

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, max_depth=8)
rf.fit(X, y)

print("=== 客户购买预测 ===")
print(f"模型准确率: {rf.score(X, y):.3f}")

# 预测新客户
new_customers = np.array([
    [35, 150, 16, 8],   # 高收入、高学历
    [25, 50, 12, 2],    # 年轻、低收入
    [40, 80, 14, 10],   # 中等条件
])

predictions = rf.predict(new_customers)
print("\n新客户预测:")
for i, (customer, pred) in enumerate(zip(new_customers, predictions)):
    print(f"客户{i+1}: 年龄{customer[0]}、收入{customer[1]}万、学历{customer[2]}年、经验{customer[3]}年 → {'购买' if pred == 1 else '不购买'}")
```

---

## 小结

本章介绍了随机森林：

1. **Bagging思想**：Bootstrap采样+聚合，降低方差
2. **特征随机**：每个节点只考虑随机选取的特征子集，增加多样性
3. **投票机制**：硬投票和软投票
4. **特征重要性**：评估特征对预测的贡献

随机森林通过集成多棵树，显著提升了单棵决策树的性能。下一章，我们将学习另一种集成思想——Boosting。