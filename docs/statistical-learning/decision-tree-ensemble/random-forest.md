# 随机森林

[决策树](decision-tree.md)有着可解释、直观易懂的优势，也有着极其容易过拟合的问题，决策树对训练数据非常敏感，只要数据稍有变化，树的结构就可能完全不同。虽然可以靠剪枝控制模型规模来缓解，但这个先天缺陷一直难以根除。直到 2001 年，李昂·布莱曼（Leo Breiman，就是发明 [CART 算法](decision-tree.md#cart-算法)的那位统计学家）在《Machine Learning》发表了开创性论文《Random Forests》，提出了随机森林算法后这个问题才被彻底解决。

**随机森林**（Random Forest）是集成学习的经典代表，它构建多棵决策树，共同投票决定最终结果。每棵树看到不同的数据样本（Bootstrap 采样）、关注不同的特征（特征随机），因此它们学到的规律各不相同，组合后既能保持决策树的直观性，又能大幅提升预测稳定性和准确率。随机森林展现集成学习优势，是以群体智慧提升个体判断的经典案例。

## Bagging 思想

随机森林的核心技术是**Bagging**，这是个"**B**ootstrap **Agg**regat**ing**"两个单词首尾聚合成的单词。Bagging 思想是指将训练集分成几份，让每个模型看到不同的数据，学到不同的规律。其中**自助采样**（Bootstrap）是一种重采样技术，从原始数据集中有放回（Repeated Sampling）地随机抽取样本，构造新的训练集。数学上，每个样本在一次抽取中被选中的概率是 $\frac{1}{n}$，不被选中的概率是 $1-\frac{1}{n}$。经过较多的 $n$ 次抽取后，某个样本从未被选中的概率趋近于 $e^{-1}$（约 0.368）：

$$P(\text{未被选中}) = \left(1-\frac{1}{n}\right)^n \approx e^{-1} \approx 0.368$$

这意味着每次 Bootstrap 采样约有 63.2% 的样本被选中至少一次（$1-0.368=0.632$）。那些未被选中的样本称为 **OOB 样本**（Out-of-Bag），可以用来验证模型性能。Bagging 算法的流程分为两个阶段：

- **训练阶段**：从原始数据集生成 $B$ 个 Bootstrap 样本，在每个 Bootstrap 样本上训练一个基学习器（如决策树）。
- **预测阶段**：对于新样本，让所有模型分别预测，然后聚合结果：

    - **分类任务**：多数投票，每个模型预测一个类别，选择得票最多的。
    - **回归任务**：平均值，每个模型预测一个数值，取算术平均。

```mermaid compact
graph TD
    A[原始数据集 D] --> B[Bootstrap 采样]
    B --> C[样本 D_1]
    B --> D[样本 D_2]
    B --> E[样本 D_B]
    C --> F[训练模型 T_1]
    D --> G[训练模型 T_2]
    E --> H[训练模型 T_B]
    F --> I[聚合预测]
    G --> I
    H --> I
    I --> J[最终结果]
```
*图：Bagging 思想的流程*

Bagging 的有效性源于**方差降低**。假设你要射击靶心，单次射击可能因为手抖偏离目标。如果你射击 100 次取平均位置，随机抖动会被相互抵消，平均位置更接近靶心。数学上，如果 $B$ 个模型的方差都是 $\sigma^2$，两两相关系数是 $\rho \in [0, 1]$（0 表示完全独立，1 表示完全相同），则集成后方差为 $\text{Var} = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$，其中：
- $\rho \sigma^2$ 是模型间相关性带来的方差，这部分无法通过集成消除，因为相关模型犯同样的错误。
- $\frac{1-\rho}{B} \sigma^2$ 是模型间差异带来的方差，这部分可以通过增加模型数量来降低。分母中的 $B$ 表示模型数量越多，这部分方差越小。

整体公式可以理解为集成方差由两部分组成，一部分无法消除（相关性），一部分可以降低（差异性）。当 $B \to \infty$（模型数量趋于无穷），集成方差趋近于 $\rho \sigma^2$，只要模型不是完全相关，就肯定小于单模型的 $\sigma^2$。

## 特征随机

集成方差公式揭示 Bagging 中模型越不相关，集成效果越好。如果所有模型完全相同（$\rho=1$），集成没有任何效果；如果模型完全独立（$\rho=0$），集成方差趋近于零。因此，问题的关键转化为如何降低集成模型之间的相关性。为此，随机森林在 Bagging 基础上，进一步引入**特征随机性**，具体做法在每个节点分裂时，不是从全部 $d$ 个特征中选择最优分裂，而是先随机选取 $m$ 个特征（一般选择 $m = \sqrt{d}$ 或者 $m = d/3$），再从这 $m$ 个特征中选择最优分裂。

这就好比一个班级投票选班长，如果所有学生都只看"成绩"这个指标，投票结果可能偏向成绩好的候选人。但如果每个学生只能看部分信息，有的看成绩，有的看品德，有的看体育，看不同的视角投票结果才会更全面，不过度依赖单一指标。数学上的解释更严谨，假设某个特征非常强（信息增益最大），如果没有特征随机，所有树的根节点都会选择它分裂。这样，树的结构高度相似，两两相关系数 $\rho$ 很大，方差公式中的 $\rho \sigma^2$ 无法有效降低。特征随机强迫每棵树看不同的视角，增加多样性。当树之间相关性 $\rho$ 降低时，集成方差公式中的 $\rho \sigma^2$ 部分（无法通过增加树的数量来消除）也随之降低。

## 聚合预测

多棵决策树训练完成后，下一步就是如何让它们投票决定最终结果。对于分类任务，一般有两种投票机制：

- **硬投票**（Hard Voting）是最直接的聚合方式，每个模型预测一个类别，选择得票最多的类别，就是少数服从多数。
- **软投票**（Soft Voting）是考虑了预测置信度的聚合方式，每个模型输出各类别的概率预测，对各类别概率取平均后选择最大概率类别。这种投票方式在现实中专家委员会的投票中有所使用。

举个具体例子，假设有 3 棵树预测某样本：

| 树 | P(A) | P(B) | P(C) | 硬投票预测 |
|:--:|:----:|:----:|:----:|:----------:|
| 1 | 0.8 | 0.1 | 0.1 | A |
| 2 | 0.5 | 0.4 | 0.1 | A |
| 3 | 0.3 | 0.6 | 0.1 | B |

- 硬投票：A 得 2 票，B 得 1 票 → 预测 A
- 软投票：平均概率 $[P(A)=0.53, P(B)=0.37, P(C)=0.10]$ → 预测 A

但注意到树 3 对 B 的置信度较高（0.6），而树 2 对 A 的置信度相对较低（0.5）。如果树 2 对 A 的置信度更低一些（比如 0.4），软投票就会选择预测 B，而硬投票仍然预测 A。这就是软投票的优势：置信度高的预测有更大影响力。因此，尽管软投票要复杂一些，但实践中一般仍然更多采用软投票的方式聚合预测结果。

## 随机森林实践

理解理论后，我们手写一个随机森林分类器，完整实现 Bootstrap 采样、特征随机选择、多棵决策树训练和多数投票预测功能。代码演示了随机森林在手写数字分类任务上的效果，并与单棵决策树对比，展示集成学习的优势。

随机森林的核心实现分为两部分，一是支持特征随机选择的决策树类 `DecisionTreeForRF`，二是管理多棵树并聚合预测的 `RandomForestClassifier`。前者在每次分裂时只考虑随机选取的特征子集，后者通过 Bootstrap 采样训练多棵树并用多数投票决定最终结果。

从输出结果可以看到，随机森林的测试准确率明显高于单棵决策树（约 95% vs 约 82%）。这说明集成多棵树确实有效降低了过拟合风险，提升了预测稳定性。这正是"群体智慧"的体现：多棵树的投票结果比单棵树的判断更可靠。

```python runnable extract-class="RandomForestClassifier"
import numpy as np

class DecisionTreeForRF:
    """
    用于随机森林的决策树
    
    与普通决策树的区别：每次分裂时只考虑随机选取的特征子集
    
    参数:
        max_depth : int, 默认值 10
            树的最大深度
        min_samples_split : int, 默认值 2
            分裂所需的最小样本数
        max_features : int or None
            每次分裂时考虑的特征数量
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
    
    def _gini(self, y):
        """计算Gini指数"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _best_split(self, X, y, feature_indices):
        """
        寻找最佳分裂（只考虑指定特征子集）
        
        对应理论：特征随机——每个节点只从m个随机特征中选择最优分裂
        """
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
        """递归构建决策树"""
        n_samples, n_features = X.shape
        
        # 检查终止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': values[np.argmax(counts)]}
        
        # 随机选择特征子集（对应理论：特征随机）
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
        """训练决策树"""
        self.tree = self._build_tree(X, y, 0)
        return self
    
    def _predict_one(self, x, node):
        """预测单个样本"""
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])
    
    def predict(self, X):
        """批量预测"""
        return np.array([self._predict_one(x, self.tree) for x in X])


class RandomForestClassifier:
    """
    随机森林分类器
    
    实现：
    1. Bootstrap采样（对应理论：样本随机）
    2. 多棵决策树训练（每棵树使用不同的Bootstrap样本和特征子集）
    3. 多数投票预测（对应理论：投票机制）
    
    参数:
        n_estimators : int, 默认值 100
            树的数量（对应理论中的B）
        max_depth : int, 默认值 10
            每棵树的最大深度
        max_features : str or int, 默认值 'sqrt'
            每次分裂时考虑的特征数量（对应理论中的m）
    """
    
    def __init__(self, n_estimators=100, max_depth=10, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
    
    def _bootstrap_sample(self, X, y):
        """
        Bootstrap采样（对应理论：有放回重采样）
        
        从原始数据集中有放回地抽取n个样本
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """
        训练随机森林
        
        核心步骤：
        1. 确定特征子集大小m
        2. 对每棵树：Bootstrap采样 → 训练决策树
        """
        n_features = X.shape[1]
        
        # 确定特征子集大小m（对应理论：分类用sqrt(d)，回归用d/3）
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
            
            # 训练决策树（带特征随机）
            tree = DecisionTreeForRF(
                max_depth=self.max_depth,
                max_features=max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """
        多数投票预测（对应理论：硬投票）
        
        每棵树预测一个类别，选择得票最多的类别
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        result = []
        for i in range(X.shape[0]):
            values, counts = np.unique(predictions[:, i], return_counts=True)
            result.append(values[np.argmax(counts)])
        return np.array(result)
    
    def score(self, X, y):
        """计算准确率"""
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

print("=== 随机森林分类（手写数字数据集）===")
print(f"树的数量: {rf.n_estimators}")
print(f"训练准确率: {rf.score(X_train, y_train):.3f}")
print(f"测试准确率: {rf.score(X_test, y_test):.3f}")

# 对比单棵决策树（展示集成学习的优势）
single_tree = DecisionTreeForRF(max_depth=15, max_features=None)  # 不限制特征，相当于普通决策树
single_tree.fit(X_train, y_train)
print(f"\n单棵决策树测试准确率: {np.mean(single_tree.predict(X_test) == y_test):.3f}")
```

## 应用场景：客户购物预测

随机森林因其直观性和可解释性，在许多领域有广泛应用。下面通过客户购物预测展示随机森林的实际应用。企业需要根据客户的年龄、收入、教育程度、工作年限等因素判断是否购买高端产品。随机森林的优势在于能自然地评估特征重要性，企业可以知道哪些因素对购买决策影响最大，从而优化营销策略。

从输出结果可以看到，随机森林成功学习到了客户购买规则。客户 1（高收入、高学历）被预测为购买，客户 2（年轻、低收入）被预测为不购买，客户 3（中等条件）根据具体特征组合做出判断。这展示了随机森林在实际业务场景中的应用价值：它能从历史数据中学习复杂的决策规则，并对新客户做出合理预测。

```python runnable
import numpy as np
from shared.tree.random_forest_classifier import RandomForestClassifier

# 模拟客户数据
np.random.seed(42)
n_samples = 500

# 特征：年龄、收入、教育年限、工作年限
age = np.random.randint(22, 60, n_samples)
income = np.random.randint(20, 200, n_samples)  # 千元
education = np.random.randint(8, 20, n_samples)  # 年
experience = np.random.randint(0, 30, n_samples)

X = np.column_stack([age, income, education, experience])

# 决策规则：高收入+高学历 或 年龄适中+一定经验
y = ((income > 100) & (education > 14)) | ((age > 30) & (age < 50) & (experience > 5))
y = y.astype(int)

# 添加噪声（模拟现实中的不确定性）
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

## 本章小结

随机森林算法是集成学习"群体智慧优于个体判断"的典型例子。通过组合多棵并行的决策树，既保持了决策树的可解释性，又大幅提升了预测稳定性和准确率。集成学习的另一个代表 —— [提升算法](boosting.md)则选择了另一条道路，让多棵树串行增强，逐步积弱成强，这就是我们下一章的话题了。

## 练习题

1. 某随机森林有 5 棵决策树，对某样本预测各类别的概率如下表。分别用硬投票和软投票计算最终预测结果，并解释两种方法在什么情况下会产生不同结果。

    | 树 | P(A) | P(B) | P(C) |
    |:--:|:----:|:----:|:----:|
    | 1 | 0.9 | 0.05 | 0.05 |
    | 2 | 0.4 | 0.55 | 0.05 |
    | 3 | 0.4 | 0.55 | 0.05 |
    | 4 | 0.2 | 0.7 | 0.1 |
    | 5 | 0.2 | 0.7 | 0.1 |

    <details>
    <summary>参考答案</summary>

    **硬投票**：

    每棵树选择概率最高的类别作为投票：
    - 树 1 预测 A（得票 1）
    - 树 2、3 预测 B（得票 2）
    - 树 4、5 预测 B（得票 2）

    最终：A 得 1 票，B 得 4 票，C 得 0 票 → **预测 B**

    **软投票**：

    计算各类别的平均概率：
    $$P_{avg}(A) = \frac{0.9 + 0.4 + 0.4 + 0.2 + 0.2}{5} = 0.42$$
    $$P_{avg}(B) = \frac{0.05 + 0.55 + 0.55 + 0.7 + 0.7}{5} = 0.51$$
    $$P_{avg}(C) = \frac{0.05 + 0.05 + 0.05 + 0.1 + 0.1}{5} = 0.07$$

    最终：**预测 B**（平均概率最高）

    **两种方法的差异情况**：

    本例中硬投票和软投票结果相同（都预测 B），但存在差异的可能：

    - **关键洞察**：树 1 对 A 的置信度极高（0.9），而其他树对 B 的置信度相对分散（0.55~0.7）。如果树 1 的置信度再高一些，或支持 A 的树更多但置信度低，可能出现硬投票选 A、软投票选 B 的情况。

    - **产生差异的条件**：当少数模型有**极高置信度**，而多数模型只有**中等置信度**时，软投票可能偏向高置信度的少数意见。这体现了软投票的优势：置信度高的预测有更大影响力，专家的强烈意见不会被简单地"少数服从多数"淹没。

    例如，如果树 1 的 $P(A) = 0.99$，其他树不变：
    - 硬投票仍预测 B（4 票 vs 1 票）
    - 软投票：$P_{avg}(A) = (0.99 + 0.4 + 0.4 + 0.2 + 0.2)/5 = 0.438$，仍低于 B 的 0.51，仍预测 B

    但如果树 2 改为预测 A（比如 $P(A)=0.45, P(B)=0.5$）：
    - 硬投票：A 得 2 票，B 得 3 票 → 预测 B
    - 软投票需要重新计算，可能因置信度分布不同而改变结果
    </details>

1. 假设一个数据集有 20 个特征，用于分类任务。根据随机森林的经验法则，每次分裂时应选择多少个特征（$m$）？如果用于回归任务，$m$ 应如何选择？解释这些选择背后的原理。
    <details>
    <summary>参考答案</summary>

    **经验法则**：

    - **分类任务**：$m = \sqrt{d} = \sqrt{20} \approx 4$ 个特征
    - **回归任务**：$m = d/3 = 20/3 \approx 7$ 个特征

    **原理解释**：

    特征随机选择 $m$ 个特征是随机森林降低模型间相关性 $\rho$ 的关键。选择 $m$ 的原则是：

    - **平衡多样性（降低 $\rho$）与单模型质量**：$m$ 太小则单模型质量差，$m$ 太大则模型高度相似，需取适中值
    - **分类任务选 $\sqrt{d}$**：分类任务特征重要性分布集中，较小的 $m$ 能打破集中性，增加多样性
    - **回归任务选 $d/3$**：回归任务特征重要性均匀分散，需要更多特征保证预测能力

    **极端情况说明**：

    - 当 $d$ 很小（如 $d < 5$）时，可能直接取 $m = d$
    - 当特征高度相关时，可适当增大 $m$（因为随机选择的特征可能高度相似，实际多样性不足）

    这些经验法则来自大量实践验证，实际应用中可通过交叉验证调整最优 $m$。
    </details>

1. 用代码实现 Bootstrap 采样过程，验证"约 36.8% 样本未被选中"的理论结论。统计 10 次 Bootstrap 采样中 OOB 样本的比例，并观察结果的稳定性。
    <details>
    <summary>参考答案</summary>

    ```python runnable
    import numpy as np

    def bootstrap_sample(n_samples):
        """
        Bootstrap采样：有放回地抽取n个样本
        返回被选中的样本索引集合和未被选中的样本索引集合
        """
        # 有放回随机抽取n个索引
        selected_indices = np.random.choice(n_samples, n_samples, replace=True)
        # 计算被选中样本的唯一索引
        unique_selected = np.unique(selected_indices)
        # 计算OOB样本（未被选中的样本）
        oob_indices = np.setdiff1d(np.arange(n_samples), unique_selected)
        
        return unique_selected, oob_indices

    # 设置样本数量
    n_samples = 100

    print("=== Bootstrap采样验证 ===")
    print(f"原始样本数量: {n_samples}")
    oob_prob_theory = (1 - 1/n_samples) ** n_samples
    print(f"理论预测OOB比例: {oob_prob_theory:.3f} ≈ {np.exp(-1):.3f}")
    print()

    # 进行10次Bootstrap采样
    oob_ratios = []
    for i in range(10):
        selected, oob = bootstrap_sample(n_samples)
        oob_ratio = len(oob) / n_samples
        oob_ratios.append(oob_ratio)
        print(f"第{i+1}次采样: 被选中 {len(selected)} 个样本, OOB {len(oob)} 个样本, OOB比例 {oob_ratio:.3f}")

    print()
    print(f"10次采样平均OOB比例: {np.mean(oob_ratios):.3f}")
    print(f"标准差: {np.std(oob_ratios):.3f}")
    print(f"最小值: {np.min(oob_ratios):.3f}, 最大值: {np.max(oob_ratios):.3f}")

    # 验证被选中样本比例
    print()
    print("=== 进一步验证 ===")
    print(f"理论预测被选中比例: {1 - np.exp(-1):.3f} ≈ 0.632")
    print(f"实际平均被选中比例: {1 - np.mean(oob_ratios):.3f}")
    ```

    **验证结论**：

    从输出可以看到，10 次 Bootstrap 采样的 OOB 比例稳定在 0.36~0.38 范围内，平均值接近理论值 $e^{-1} \approx 0.368$。这验证了 Bootstrap 采样的经典结论：

    1. 约 36.8% 的样本不会被选中（成为 OOB 样本）
    2. 约 63.2% 的样本至少被选中一次
    3. 多次采样的结果稳定，标准差很小

    这个稳定性源于大数定律：当 $n$ 足够大时，$(1-\frac{1}{n})^n$ 稳定趋近于 $e^{-1}$。
    </details>
