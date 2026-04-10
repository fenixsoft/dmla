# 决策树——从数据学习规则

## 引言：规则学习的直观性

如果说线性模型是"数学家的方法"，那么**决策树（Decision Tree）**就是"普通人的方法"。

想象你在判断今天是否适合打网球。你会怎么想？

```
天气如何？
├─ 晴天
│  └─ 湿度高吗？
│     ├─ 是 → 不打网球
│     └─ 否 → 打网球
└─ 阴天/雨天
   └─ 风大吗？
      ├─ 是 → 不打网球
      └─ 否 → 打网球
```

这就是一棵决策树。它通过一系列"是/否"问题，最终给出决策。这种**从数据自动学习规则**的方式，直观且易于理解。

决策树是机器学习中最直观的算法之一：
- **可解释性强**：每个决策分支都能用自然语言描述
- **不需要特征缩放**：对数据的数值范围不敏感
- **能处理混合类型**：数值型和类别型特征都能处理
- **自动特征选择**：分裂过程自动选择最重要的特征

---

## 分裂准则：如何选择最佳分割

决策树的核心问题是：**在每个节点，应该选择哪个特征进行分裂？**

### 信息增益（Information Gain）

**熵（Entropy）**衡量数据的不确定性：

$$H(D) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

其中 $p_k$ 是类别 $k$ 在数据集 $D$ 中的比例。

- 熵 = 0：数据完全纯净（只有一类）
- 熵 = $\log_2 K$：数据最混乱（各类均匀分布）

**信息增益**是分裂前后的熵减少量：

$$IG(D, A) = H(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} H(D_v)$$

其中 $D_v$ 是特征 $A$ 取值为 $v$ 的子集。

**选择信息增益最大的特征进行分裂**。

### 增益率（Gain Ratio）

信息增益偏向于取值多的特征。**增益率**通过归一化解决此问题：

$$GainRatio(D, A) = \frac{IG(D, A)}{SplitInfo(A)}$$

其中：

$$SplitInfo(A) = -\sum_{v} \frac{|D_v|}{|D|} \log_2 \frac{|D_v|}{|D|}$$

### Gini指数

**Gini指数**衡量数据的"不纯度"：

$$Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$$

分裂后的Gini指数加权平均：

$$Gini(D, A) = \sum_{v} \frac{|D_v|}{|D|} Gini(D_v)$$

**选择使Gini指数最小的特征进行分裂**。

### 三种准则对比

| 准则 | 算法 | 特点 |
|------|------|------|
| 信息增益 | ID3 | 偏向取值多的特征 |
| 增益率 | C4.5 | 修正信息增益的偏差 |
| Gini指数 | CART | 计算简单，常用 |

---

## ID3算法

### 算法流程

1. 若所有样本属于同一类，返回叶节点
2. 若特征集为空，返回多数类叶节点
3. 选择信息增益最大的特征作为当前节点
4. 对特征的每个取值，递归构建子树

### 局限性

- 只能处理离散特征
- 偏向取值多的特征
- 容易过拟合

---

## C4.5算法

C4.5是ID3的改进版本：

### 改进点

1. **增益率替代信息增益**：避免偏向取值多的特征
2. **处理连续特征**：通过二分法寻找最佳分割点
3. **处理缺失值**：按比例分配样本到各分支
4. **剪枝**：通过错误率估计进行后剪枝

### 连续特征处理

对于连续特征 $A$，计算所有可能的分割点：

$$GainRatio(D, A, t) = \frac{IG(D, A, t)}{SplitInfo(A, t)}$$

选择增益率最大的分割点 $t$。

---

## CART算法

**CART（Classification and Regression Trees）**是最常用的决策树算法。

### 特点

1. **二叉树**：每个节点只有两个分支
2. **Gini指数**：使用Gini指数作为分裂准则
3. **支持回归**：可用于分类和回归

### 分类树

选择使Gini指数最小的特征和分割点：

$$(A^*, t^*) = \arg\min_{A, t} Gini(D, A, t)$$

### 回归树

对于回归问题，使用**方差**代替Gini指数：

$$\text{Var}(D) = \frac{1}{|D|} \sum_{i \in D} (y_i - \bar{y})^2$$

选择使分裂后方差最小的分割点。

---

## NumPy实现：手写CART决策树

```python
import numpy as np

class DecisionTreeClassifier:
    """
    CART决策树分类器实现
    """
    
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def _gini(self, y):
        """计算Gini指数"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _gini_split(self, y_left, y_right):
        """计算分裂后的加权Gini指数"""
        n = len(y_left) + len(y_right)
        return (len(y_left) / n) * self._gini(y_left) + (len(y_right) / n) * self._gini(y_right)
    
    def _best_split(self, X, y):
        """寻找最佳分割点"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            # 获取唯一值作为候选分割点
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # 分割数据
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                gini = self._gini_split(y_left, y_right)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gini
    
    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        n_samples = len(y)
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            # 返回叶节点（多数类）
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': values[np.argmax(counts)]}
        
        # 寻找最佳分割
        feature, threshold, gini = self._best_split(X, y)
        
        if feature is None:
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': values[np.argmax(counts)]}
        
        # 分割数据
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # 递归构建子树
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """训练模型"""
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _predict_one(self, x, node):
        """预测单个样本"""
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])
    
    def predict(self, X):
        """预测"""
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# 测试：鸢尾花分类
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# 打乱数据
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# 训练决策树
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)

print("=== CART决策树分类 ===")
print(f"训练准确率: {tree.score(X_train, y_train):.3f}")
print(f"测试准确率: {tree.score(X_test, y_test):.3f}")

# 打印树结构
def print_tree(node, indent=""):
    if node['leaf']:
        print(f"{indent}叶节点: 类别 {node['class']}")
    else:
        print(f"{indent}特征{node['feature']} <= {node['threshold']:.2f}?")
        print(f"{indent}├─是:")
        print_tree(node['left'], indent + "│  ")
        print(f"{indent}└─否:")
        print_tree(node['right'], indent + "   ")

print("\n决策树结构:")
print_tree(tree.tree)
```

**输出示例：**
```
=== CART决策树分类 ===
训练准确率: 1.000
测试准确率: 0.967

决策树结构:
特征3 <= 0.80?
├─是:
│  叶节点: 类别 0
└─否:
   特征3 <= 1.75?
   ├─是:
   │  特征2 <= 4.95?
   │  ├─是: 叶节点: 类别 1
   │  └─否: 叶节点: 类别 2
   └─否:
      叶节点: 类别 2
```

---

## 应用场景示例：贷款审批

```python
import numpy as np

# 模拟贷款审批数据
np.random.seed(42)
n_samples = 200

# 特征：收入(高/中/低)、负债(高/低)、信用(好/差)
income = np.random.choice([0, 1, 2], n_samples)  # 0=低, 1=中, 2=高
debt = np.random.choice([0, 1], n_samples)       # 0=低, 1=高
credit = np.random.choice([0, 1], n_samples)     # 0=差, 1=好

X = np.column_stack([income, debt, credit])

# 决策规则：高收入+好信用=批准，中等收入+低负债+好信用=批准
y = np.zeros(n_samples, dtype=int)
y[(income == 2) & (credit == 1)] = 1
y[(income == 1) & (debt == 0) & (credit == 1)] = 1

# 添加一些噪声
noise_idx = np.random.choice(n_samples, 10, replace=False)
y[noise_idx] = 1 - y[noise_idx]

# 训练决策树
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X, y)

print("=== 贷款审批决策树 ===")
print(f"训练准确率: {tree.score(X, y):.3f}")

print("\n决策树规则:")
print_tree(tree.tree)

# 预测新申请
new_applicants = np.array([
    [2, 0, 1],  # 高收入、低负债、好信用
    [1, 1, 0],  # 中等收入、高负债、差信用
    [0, 0, 1],  # 低收入、低负债、好信用
])
predictions = tree.predict(new_applicants)
print("\n新申请预测:")
for i, (applicant, pred) in enumerate(zip(new_applicants, predictions)):
    income_label = ['低', '中', '高'][applicant[0]]
    debt_label = ['低', '高'][applicant[1]]
    credit_label = ['差', '好'][applicant[2]]
    result = '批准' if pred == 1 else '拒绝'
    print(f"  申请人{i+1}: 收入{income_label}、负债{debt_label}、信用{credit_label} → {result}")
```

---

## 小结

本章介绍了决策树的核心原理：

1. **分裂准则**：信息增益、增益率、Gini指数
2. **ID3算法**：使用信息增益，只能处理离散特征
3. **C4.5算法**：改进ID3，处理连续特征和缺失值
4. **CART算法**：二叉树结构，使用Gini指数，支持分类和回归

决策树虽然直观易懂，但单棵树容易过拟合。下一章，我们将看到如何通过集成多棵树来提升性能——随机森林。