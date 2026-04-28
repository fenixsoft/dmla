# 提升方法

如果说[随机森林](random-forest.md)算法是集成学习中"群体智慧优于个体判断"的代表，那提升方法（Boosting）就是"积弱成强，聚沙成塔"的典型。这个 1996 年由以色列计算机科学家约阿夫·弗罗因德（Yoav Freund）和罗伯特·沙皮尔（Robert Schapire）提出的学习算法，主张将一个困难问题拆成若干个简单问题，每个简单问题只需要一个"比随机猜测好一点"的弱学习器来解决，把多个"弱学习器"组合成一个"强学习器"，逐步解决困难问题。

## Boosting 思想

在提升算法出现之前，集成学习的主流一直是 [Bagging 思想](random-forest.md#bagging-思想)，采用并行方式，让每棵决策树独立并行训练。Boosting 思想给出了另一条完全不同的道路，采用串行方式，让每棵新树专注于纠正之前所有树的累积错误，逐渐改进效果，最终通过精心设计的加权组合，令整体性能得以大幅提升。这种接力纠错的方法，使得每个学习器都只做自己擅长的事，最终积小胜终成大胜。

Boosting 要做到后一个学习器能改进之前所有学习器的累积效果，首先得让每个弱学习器有途径得知之前学习器的组合在哪些样本上犯了错误，Boosting 引入加权训练和序列学习流程两个机制来达成这个目的。

- **加权训练**：通过调整样本权重来聚焦困难样本，对于被前一个学习器错误分类的样本，增加其权重，让后续学习器重点关注；对于被正确分类的样本，减少权重，降低关注度；
- **序列学习流程**：学习器按顺序训练，每个新学习器都基于前一个学习器的性能表现进行优化，通过计算学习器权重来确定其在最终组合中的话语权，表现好的学习器获得更高的投票权重。

Boosting 的算法流程分为训练和预测两个阶段。训练阶段从初始均匀权重开始，先用当前样本权重训练弱学习器，然后统计该学习器的加权错误率，根据错误率重新分配学习器权重（错误率越低，权重越高）并更新样本权重（错误样本权重增加，正确样本权重减少）。进入预测阶段后，对新样本，让所有弱学习器分别预测，然后将各学习器的结果按加权组合，输出最终结果。整个训练的过程如下图所示：

```mermaid compact
graph LR
    A["初始化样本权重"] --> B["第 n 轮迭代"]
    B --> C["用当前权重训练学习器"]
    C --> D["计算加权错误率"]
    D --> E["计算学习器权重"]
    E --> F["更新样本权重"]
    F --> G{"达到最大轮数?"}
    G -->|"否"| B
    G -->|"是"| H["输出加权组合"]
```
*图：Boosting 算法的流程*

Bagging 通过并行集成降低方差，而 Boosting 的有效性源于**偏差降低**。假设单个弱学习器的偏差为 $b$，经过 $T$ 轮迭代，每个弱学习器纠正一部分错误，最终模型的偏差被逐步减小。数学上，如果每个弱学习器的错误率为 $\epsilon < 0.5$（比随机猜测好即可），Boosting 可以将最终错误率指数级降低至接近零。当然，工程上这就会伴有很大过拟合风险。

## AdaBoost 算法

Boosting 的思想只是定性描述，对于实际应用，还需要有一套可操作的流程去量化和执行，譬如样本权重应该如何调整？学习器权重应该如何计算？最终组合时各学习器的贡献如何分配？等等。**AdaBoost**（Adaptive Boosting，自适应提升）算法是第一个给出完整数学框架的 Boosting 算法。

考虑一个具体的二分类场景：假设我们有 5 个训练样本，标签为 $y \in \{-1, +1\}$（-1 表示负类，+1 表示正类）。初始时，所有样本权重相等，每个样本权重为 $w_i = 1/5 = 0.2$。

| 样本 | 特征 $x$ | 标签 $y$ | 初始权重 $w$ |
|:----:|:--------:|:--------:|:------------:|
| 1 | 0.1 | +1 | 0.2 |
| 2 | 0.3 | +1 | 0.2 |
| 3 | 0.5 | -1 | 0.2 |
| 4 | 0.7 | -1 | 0.2 |
| 5 | 0.9 | +1 | 0.2 |

从初始权重开始，每一轮迭代（用第 $t$ 轮表示）都包含以下四个步骤：

- 步骤一 **训练弱学习器**：用当前权重 $w^{(t)}$ 训练学习器 $h_t$。学习器可以很弱，只要求比随机猜测好一些，譬如**决策桩**（Decision Stump），这是只有一层的决策树，即只根据单个特征的单个阈值进行分裂。
- 步骤二 **计算加权错误率**：错误率 $\epsilon_t = \frac{\sum_{i=1}^{n} w_i^{(t)} \cdot \mathbb{I}[h_t(x_i) \neq y_i]}{\sum_{i=1}^{n} w_i^{(t)}}$。其中 $w_i^{(t)}$ 是第 $t$ 轮中样本 $i$ 的权重，权重越高，该样本在错误率计算中话语权越大；$\mathbb{I}[h_t(x_i) \neq y_i]$ 是指示函数，当预测错误时取值为 1，预测正确时取值为 0。分子是对所有错误样本的权重求和，得到加权后的错误总量，分母是所有样本权重的总和，用于归一化。整个公式实际是在计算错误样本的权重占比，如果错误样本都是高权重样本，错误率就高；如果错误样本权重低，错误率就低。

    以之前 5 个样本的例子，假设第一轮学习器对样本 3 和 5 分类错误 $\epsilon_1 = \frac{0.2 + 0.2}{0.2 + 0.2 + 0.2 + 0.2 + 0.2} = \frac{0.4}{1.0} = 0.4$，这表示当前学习器在加权意义下的错误率是 40%。

- 步骤三 **计算学习器权重**：学习器权重 $\alpha_t = \frac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}$，这个公式决定当前学习器在最终组合中的话语权。其中 $\epsilon_t$ 是当前学习器的错误率，错误率越低，分母越小，$\alpha_t$ 越大；$\frac{1 - \epsilon_t}{\epsilon_t}$ 是正确率与错误率的比值，比值越大，说明学习器越靠谱；$\ln$ 是自然对数，将比值转换为权重，作用让权重随错误率降低而增大，且在错误率很低时增长更显著；$\frac{1}{2}$ 是缩放因子，用于控制权重的范围。整体公式可以理解为给表现好的学习器更多投票权，错误率越低，权重越大。

    代入例子中的 $\epsilon_1 = 0.4$，可得 $\alpha_1 = \frac{1}{2} \ln \frac{0.6}{0.4} = \frac{1}{2} \ln 1.5 \approx 0.2027$，这个学习器权重约为 0.2，表示它在最终投票时会贡献 0.2 的票数。如果错误率是 0.5（随机猜测水平），$\alpha_t$ 恰好是 0，这种学习器不参与投票，因为它没有提供有用信息。

- 步骤四 **更新样本权重**：权重 $w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))$，这个公式起到一种"放大镜"的作用，通过权重调整让后续学习器聚焦错误样本。$y_i h_t(x_i)$ 是预测正确性的指标，当预测正确时，$y_i$ 和 $h_t(x_i)$ 同号，乘积为 +1；预测错误时，乘积为 -1；$\exp(-\alpha_t \cdot \text{乘积})$ 是权重调整因子，利用指数函数的增长特性，若预测正确（乘积为 +1）权重乘以 $\exp(-\alpha_t) < 1$，权重减小，若预测错误（乘积为 -1）权重乘以 $\exp(\alpha_t) > 1$，权重增大。整体公式让错误样本浮出水面，正确样本沉入水底。

    代入例子中的 $\alpha_1 = 0.2027$，样本 3 和 5 分类错误（$y_i h_t(x_i) = -1$），其他样本分类正确：

    - 样本 1、2、4（正确）：$w_i^{(2)} = 0.2 \times \exp(-0.2027) = 0.2 \times 0.817 \approx 0.163$，归一化后（确保权重总和为 1）样本 1、2、4 权重约为 $0.143$
    - 样本 3、5（错误）：$w_i^{(2)} = 0.2 \times \exp(0.2027) = 0.2 \times 1.225 \approx 0.245$，归一化后样本 3、5 权重约为 $0.214$

    可以看到，错误样本的权重从 0.2 增加到约 0.214，正确样本的权重从 0.2 减少到约 0.143。下一轮学习器训练时，样本 3 和 5 将获得更多注意力。

经过 $T$ 轮迭代后，AdaBoost 的最终预测使用加权投票输出结果，确保表现好的学习器话语权更高，投票权重更大。

## AdaBoost 算法实践

前面详细推导了 AdaBoost 的数学原理，本节开始动手实现一个 AdaBoost 分类器，观察样本权重和学习器权重如何随迭代变化，以及最终的决策边界是如何形成的。

下面的代码实现了完整的 AdaBoost 算法流程：首先定义决策桩作为弱学习器，然后在每轮迭代中训练决策桩、计算加权错误率、计算学习器权重、更新样本权重，最后通过加权投票组合所有弱学习器。代码还包括权重变化追踪，展示随着迭代次数增加，学习器权重如何逐渐趋于稳定。

```python runnable extract-class="AdaBoost"
import numpy as np

class DecisionStump:
    """
    决策桩：单层决策树，AdaBoost 常用的弱学习器
    
    核心思想：只根据单个特征的单个阈值进行分类
    """
    
    def __init__(self):
        self.feature = None    # 选择的特征索引
        self.threshold = None  # 分裂阈值
        self.polarity = 1      # 分裂方向：1 表示 <=阈值预测 -1，-1 表示 >阈值预测 -1
    
    def fit(self, X, y, sample_weights):
        """
        训练决策桩
        
        核心步骤：
        1. 遍历所有特征和所有可能的阈值
        2. 尝试两种分裂方向（polarity）
        3. 选择加权错误率最小的分裂方案（对应理论中的 ε_t 最小化）
        """
        n_samples, n_features = X.shape
        min_error = float('inf')
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                for polarity in [1, -1]:
                    # 根据阈值和方向生成预测
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X[:, feature] <= threshold] = -1
                    else:
                        predictions[X[:, feature] > threshold] = -1
                    
                    # 计算加权错误率（对应理论中的 ε_t）
                    error = np.sum(sample_weights[predictions != y])
                    
                    if error < min_error:
                        min_error = error
                        self.feature = feature
                        self.threshold = threshold
                        self.polarity = polarity
                        self.error = error
        
        return self
    
    def predict(self, X):
        """根据训练好的阈值和方向进行预测"""
        predictions = np.ones(X.shape[0])
        if self.polarity == 1:
            predictions[X[:, self.feature] <= self.threshold] = -1
        else:
            predictions[X[:, self.feature] > self.threshold] = -1
        return predictions


class AdaBoost:
    """
    AdaBoost 分类器
    
    核心思想：序列训练多个弱学习器，加权组合成强学习器
    """
    
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []   # 存储所有弱学习器
        self.alphas = []   # 存储所有学习器权重
    
    def fit(self, X, y):
        """
        训练 AdaBoost
        
        核心步骤（对应理论中的迭代流程）：
        1. 初始化样本权重
        2. 每轮迭代：训练弱学习器 → 计算错误率 → 计算学习器权重 → 更新样本权重
        3. 保存所有弱学习器及其权重
        """
        n_samples = X.shape[0]
        
        # 初始化权重：所有样本权重相等（对应理论中的 w_i^(1) = 1/n）
        weights = np.ones(n_samples) / n_samples
        
        self.stumps = []
        self.alphas = []
        
        for t in range(self.n_estimators):
            # 步骤一：训练弱学习器（决策桩）
            stump = DecisionStump()
            stump.fit(X, y, weights)
            
            # 步骤二：计算加权错误率 ε_t
            predictions = stump.predict(X)
            error = np.sum(weights[predictions != y])
            
            # 防止极端情况（错误率为 0 或 1）
            error = max(error, 1e-10)
            error = min(error, 1 - 1e-10)
            
            # 步骤三：计算学习器权重 α_t（对应理论中的公式）
            alpha = 0.5 * np.log((1 - error) / error)
            
            # 步骤四：更新样本权重（对应理论中的权重更新公式）
            # 预测正确的样本权重减小，预测错误的样本权重增大
            weights = weights * np.exp(-alpha * y * predictions)
            weights = weights / np.sum(weights)  # 归一化
            
            self.stumps.append(stump)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X):
        """
        加权投票预测
        
        对应理论中的 H(x) = sign(Σ α_t * h_t(x))
        """
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        for stump, alpha in zip(self.stumps, self.alphas):
            scores += alpha * stump.predict(X)
        
        return np.sign(scores).astype(int)
    
    def score(self, X, y):
        """计算准确率"""
        return np.mean(self.predict(X) == y)


# 固定随机种子，确保这里输出的结果与后面的文字描述一致
np.random.seed(42)

# 生成数据：线性可分但添加噪声
n_samples = 200
X = np.random.randn(n_samples, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# 添加 5% 噪声，模拟真实数据中的噪声样本
noise_idx = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
y[noise_idx] = -y[noise_idx]

# 训练 AdaBoost
adaboost = AdaBoost(n_estimators=50)
adaboost.fit(X, y)

print("=== AdaBoost 分类结果 ===")
print(f"弱学习器数量: {adaboost.n_estimators}")
print(f"训练准确率: {adaboost.score(X, y):.3f}")

# 观察弱学习器权重变化
print("\n前 10 个弱学习器的权重 (alpha):")
for i in range(min(10, len(adaboost.alphas))):
    print(f"  学习器 {i+1}: α = {adaboost.alphas[i]:.4f}")

# 对比单个决策桩（验证"弱到强"的提升效果）
single_stump = DecisionStump()
single_stump.fit(X, y, np.ones(n_samples) / n_samples)
stump_acc = np.mean(single_stump.predict(X) == y)
print(f"\n单个决策桩准确率: {stump_acc:.3f}")
print(f"AdaBoost 组合后准确率: {adaboost.score(X, y):.3f}")
print(f"准确率提升: +{(adaboost.score(X, y) - stump_acc)*100:.1f}%")
```

从输出可以看到（这段代码固定了随机种子，所以我们能看到一样的输出），单个决策桩的准确率约为 76.5%，而 AdaBoost 组合 50 个弱学习器后准确率达到 94%，提升近 17.5 个百分点。这正是 Boosting 的威力：将比随机猜测好一点的弱学习器组合成了高性能的强学习器。

同时可以观察到，早期学习器的权重 α 较高（如第一个约 0.59），这是因为前几轮迭代中样本权重相对均匀，学习器更容易找到好的分裂点。随着迭代进行，样本权重越来越集中到困难样本上，学习器面临更大的挑战，权重逐渐降低。

## 本章小结

本章介绍了 Boosting 方法的思想和它的经典实现 AdaBoost 算法。集成学习揭示了机器学习的一条重要经验：不要追求单一模型的完美，而要学会组合多个不完美的模型。Bagging 的并行投票和 Boosting 的序列纠错，从两个不同角度验证了这个道理，群体的智慧，往往超越个体的极限。

在实际工程应用中，即使进入深度学习时代，Boosting 方法在很多场景中依然无可替代。本章未提及的许多提升方法，如 XGBoost、LightGBM、CatBoost 仍是工业界首选，原因在于训练效率高、无需 GPU、特征工程友好、可解释性强。

## 练习题

1. 比较 Bagging（以随机森林为例）与 Boosting（以 AdaBoost 为例）在以下维度的差异：(a) 学习器训练方式（并行/串行）；(b) 样本处理方式；(c) 集成策略；(d) 主要降低的误差类型（偏差/方差）。并解释为什么 Boosting 更容易过拟合。
    <details>
    <summary>参考答案</summary>

    | 维度 | Bagging（随机森林） | Boosting（AdaBoost） |
    |:----:|:-------------------:|:--------------------:|
    | **训练方式** | 并行训练，各树独立 | 串行训练，后树依赖前树 |
    | **样本处理** | 有放回随机采样（Bootstrap） | 加权训练，错误样本权重增加 |
    | **集成策略** | 简单投票/平均，各树权重相等 | 加权投票，表现好的树权重更高 |
    | **降低误差** | 主要降低方差 | 主要降低偏差 |

    **Boosting 更容易过拟合的原因**：

    1. **对噪声敏感**：Boosting 会不断加权错误样本，如果错误是由噪声导致而非真实模式，Boosting 会试图"纠正"这些噪声，导致模型学到虚假模式。
    2. **迭代加深**：随着迭代进行，模型越来越复杂，对训练数据的拟合程度越来越高，如果迭代次数过多，容易过度拟合训练数据。
    3. **偏差 - 方差权衡**：Boosting 主要降低偏差，但降低偏差的代价通常是增加方差。当偏差降到很低时，方差可能变得很大，导致过拟合。

    相比之下，Bagging 通过平均多个独立模型的预测来降低方差，对单个模型的过拟合有一定抑制作用，因此更不容易过拟合。
    </details>

1. 用代码实现一个简单的决策桩（Decision Stump），在以下数据上训练并验证其分类效果：$X = [[1], [2], [3], [4], [5]]$，$y = [1, 1, -1, -1, -1]$。然后分析：(a) 该决策桩的分类准确率是多少？(b) 为什么单个决策桩在这个数据集上无法达到 100% 准确率？
    <details>
    <summary>参考答案</summary>

    ```python runnable
    import numpy as np

    class DecisionStump:
        def __init__(self):
            self.threshold = None
            self.polarity = 1

        def fit(self, X, y):
            n_samples = len(X)
            min_error = float('inf')

            # 遍历所有可能的阈值
            for threshold in np.unique(X):
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X.flatten() <= threshold] = -1
                    else:
                        predictions[X.flatten() > threshold] = -1

                    error = np.mean(predictions != y)
                    if error < min_error:
                        min_error = error
                        self.threshold = threshold
                        self.polarity = polarity

            return self

        def predict(self, X):
            predictions = np.ones(len(X))
            if self.polarity == 1:
                predictions[X.flatten() <= self.threshold] = -1
            else:
                predictions[X.flatten() > self.threshold] = -1
            return predictions

    # 数据准备
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 1, -1, -1, -1])

    # 训练决策桩
    stump = DecisionStump()
    stump.fit(X, y)
    predictions = stump.predict(X)

    print(f"决策桩阈值: {stump.threshold}")
    print(f"分裂方向: {'≤阈值预测-1' if stump.polarity == 1 else '>阈值预测-1'}")
    print(f"预测结果: {predictions}")
    print(f"真实标签: {y}")
    print(f"分类准确率: {np.mean(predictions == y) * 100:.1f}%")
    ```

    **输出分析**：

    该决策桩会选择阈值 2，polarity=-1，将样本 1、2 分为正类（预测+1），样本 3、4、5 分为负类（预测-1）。

    **(a) 准确率**：100%（5 个样本全部正确分类）。

    **(b) 说明**：这个数据集是线性可分的，单个决策桩完全可以达到 100% 准确率。这也说明，即使是弱学习器，在某些简单数据集上也能达到高准确率。Boosting 方法的价值在于处理更复杂的非线性可分数据，通过组合多个弱学习器逐步提升性能。
    </details>

1. 在 AdaBoost 的某轮迭代中，有 6 个样本，当前权重和学习器的预测结果如下表所示。请计算加权错误率 $\epsilon_t$，并判断是否满足弱学习器的要求（比随机猜测好，即 $\epsilon_t < 0.5$）。

    | 样本 | 权重 $w_i$ | 真实标签 $y_i$ | 预测值 $h_t(x_i)$ | 是否正确 |
    |:----:|:----------:|:--------------:|:-----------------:|:--------:|
    | 1 | 0.10 | +1 | +1 | ✓ |
    | 2 | 0.15 | -1 | +1 | ✗ |
    | 3 | 0.20 | +1 | -1 | ✗ |
    | 4 | 0.25 | -1 | -1 | ✓ |
    | 5 | 0.15 | +1 | +1 | ✓ |
    | 6 | 0.15 | -1 | -1 | ✓ |

    <details>
    <summary>参考答案</summary>

    **步骤一：识别错误样本**

    从表格中可以看出，分类错误的样本是：样本 2 和样本 3。

    **步骤二：计算加权错误率**

    根据公式：$\epsilon_t = \frac{\sum_{i=1}^{n} w_i^{(t)} \cdot \mathbb{I}[h_t(x_i) \neq y_i]}{\sum_{i=1}^{n} w_i^{(t)}}$

    分子（错误样本权重之和）：
    $$\sum_{\text{错误}} w_i = w_2 + w_3 = 0.15 + 0.20 = 0.35$$

    分母（所有样本权重之和）：
    $$\sum_{i=1}^{6} w_i = 0.10 + 0.15 + 0.20 + 0.25 + 0.15 + 0.15 = 1.00$$

    加权错误率：
    $$\epsilon_t = \frac{0.35}{1.00} = 0.35$$

    **步骤三：判断弱学习器要求**

    由于 $\epsilon_t = 0.35 < 0.5$，该学习器满足弱学习器的要求（比随机猜测好）。

    **补充分析**：
    - 虽然学习器在 6 个样本中有 2 个错误，表面错误率为 $2/6 \approx 33.3\%$
    - 但考虑权重后，加权错误率为 35%，略高于简单错误率
    - 这是因为错误样本 3 的权重（0.20）高于样本 1、5、6 等正确样本
    - 体现了 AdaBoost 加权错误率的本质：**高权重样本的错误对错误率贡献更大**
    </details>

