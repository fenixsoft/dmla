# 朴素贝叶斯

十八世纪，英国牧师托马斯·贝叶斯（Thomas Bayes）在研究"如何从观测结果推断未知原因"这一哲学问题时，写下了一个看似简单的公式：$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$。这个公式描述了一个哲学思想：当我们获得新证据时，应该如何更新对世界的认知。两百多年后，这个公式成了现代统计学、机器学习乃至人工智能的核心工具，影响深远以至于人们用他的名字命名了整个学派 —— 贝叶斯学派。

在[概率统计系列](../../maths/probability/introduction.md)中，我们已经学习了贝叶斯定理的数学形式。但如何将这个哲学思想转化为可用的计算机工具？**朴素贝叶斯**（Naive Bayes）给出了最简洁的答案。它用一个"朴素"的假设（各特征相互独立）将贝叶斯定理从理论殿堂带入工程实践，成为机器学习中最古老却也最实用的分类算法之一。

## 贝叶斯分类

站在贝叶斯的角度，分类任务就是**计算给定特征下属于各类别的概率，选择概率最大的类别**。这个定义其实没有它看上去那么理所当然，它指出分类不再是寻找一条硬性的边界线，而是计算概率、比较概率、做出决策。这种思维方式与人类决策过程惊人相似，当我们判断一封邮件是否是垃圾邮件时，并没有固定的阈值准则，人的大脑在不断权衡证据、计算可能性、做出判断。

让我们从一个具体的例子开始。假设我们收到一封邮件，内容是"点击链接领取免费红包"。我们如何判断它是否是垃圾邮件？凭直觉，我们会关注几个关键词："点击链接"、"免费"、"红包"，这些词汇在垃圾邮件中出现频率高，在正常邮件中出现频率低。贝叶斯定理告诉我们如何将这些直觉量化：

$$P(\text{垃圾邮件}|\text{点击}, \text{免费}, \text{红包}) = \frac{P(\text{点击}, \text{免费}, \text{红包}|\text{垃圾邮件}) \cdot P(\text{垃圾邮件})}{P(\text{点击}, \text{免费}, \text{红包})}$$

这个公式将对该邮件是否为垃圾邮件的人类经验判断，转化为一项通过样本统计，知道垃圾邮件中出现"点击、免费、红包"的概率 $P(\text{点击}, \text{免费}, \text{红包}|\text{垃圾邮件})$，以及邮件确实为垃圾邮件的先验概率 $P(\text{垃圾邮件})$ 就能完成的量化计算的工作。

推广到一般的情况，给定特征向量 $X = (x_1, x_2, \ldots, x_d)$，要预测类别 $y$。根据贝叶斯定理 $P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$，只需要计算出不同类别的似然和先验概率大小即可（因为分母 $P(X)$ 不包含 $y$，对所有类别相同，可以忽略）。

```mermaid compact
flowchart LR
    A["先验<br>&nbsp;　　　P(垃圾邮件)　　　　&nbsp;"] --> C{贝叶斯定理}
    B["似然<br>P(点击,免费,红包|垃圾邮件)"] --> C
    C --> D["　后验概率　<br>P(垃圾邮件|证据)"]
    D --> E["　决策　<br>选择最大概率"]
```

*图：贝叶斯分类的决策过程：用证据更新信念*

然而，要计算联合概率 $P(\text{点击}, \text{免费}, \text{红包}|\text{垃圾邮件})$ 是极其困难的，语言的词汇之间并非独立，"免费"往往和"领取"一起出现，"点击"往往和"链接"一起出现，所以这个联合概率不能简单相乘，必须直接从样本中统计"点击 + 免费 + 红包"这个组合。这就需要统计垃圾邮件中同时出现这三个词的次数，词汇表可能有上万个词，三词组合的概率表将包含上万亿种可能性，要直接从样本中统计出三个词的联合概率，所需要的样本数量将是天文数字。

### 朴素假设

由例子可见，如果每个特征有 $v$ 个可能取值，需要估计 $v^d$ 个概率值，特征维度稍高就不可行了。面对这个困境，朴素贝叶斯做出了一个看似不符合事实却极为实用的假设：**假设在给定类别 $y$ 的条件下，各特征相互独立**，联合概率可以直接取它们的乘积：

$$P(X|y=c) = \prod_{j=1}^{d} P(x_j | y=c)$$

这个假设将联合概率估计的复杂度从 $v^d$ 降到 $d \times v$，使得高维问题的概率估计成为可能。回到垃圾邮件分类的例子，我们不再需要统计"点击、免费、红包"三个词同时出现的次数，只需要分别统计垃圾邮件中"点击"出现多少次、"免费"出现多少次、"红包"出现多少次，然后相乘即可。

这个假设之所以被称为**朴素**（Naive），正是因为它与现实严重偏离，真实世界中特征之间的相关性无处不在。但朴素贝叶斯却硬生生地假设它们相互独立，这就像假设一个人的身高和体重毫无关系，听起来简直荒谬。然而，这个"荒谬"的假设却屡屡成功，其原因潜藏在分类任务的性质中：

1. **分类只关心相对大小**：朴素贝叶斯的概率估计可能不准确，但只要各类别的相对排序正确，分类结果就是正确的。假设垃圾邮件的后验概率被低估为 0.4，正常邮件被低估为 0.35，两者都不精确，但 $0.4 > 0.35$ 的排序正确，分类结果就是"垃圾邮件"，这恰恰是正确答案。
   
2. **决策边界的简化**：取对数后，朴素贝叶斯的决策规则变为线性形式，这揭示了朴素贝叶斯本质上是**线性分类器**。决策边界由对数概率比的线性组合决定。线性分类器虽然在表达能力上受限，但对于许多实际问题已经足够用了。

3. **偏差 - 方差权衡**：朴素假设引入了偏差：概率估计不准确；但同时大大减少了方差：概率估计更稳定。在小样本场景，高方差模型的估计波动剧烈，朴素贝叶斯的"有偏但稳定"反而成为一种优势。这就像用一根粗糙的尺子测量，虽然每次测量都有误差，但误差方向一致，多次测量后反而能得到相对可靠的结论。

![偏差 - 方差权衡示意图](assets/bias-variance.png)

*图：偏差 - 方差权衡：朴素假设带来的稳定性的代价*

这三个性质共同指向一个洞察：**模型的假设未必要绝对精确，只需足够好用**。朴素贝叶斯用最简单的假设换取最实用的效果，这正是工程思维的精髓，不追求完美，追求有效。

## 朴素贝叶斯分类器

从贝叶斯定理出发，代入朴素假设，可得朴素贝叶斯分类器的决策规则公式：

$$\hat{y} = \arg\max_c \left[ P(y=c) \prod_{j=1}^{d} P(x_j | y=c) \right]$$

实际计算中，多个小概率相乘容易导致数值下溢，当词汇表很大时，几十个 $0.001$ 相乘可能得到 $10^{-60}$ 这样的极小值，计算机无法精确存储。取对数可以将乘法转化为加法，避免数值问题，最终得到：

$$\hat{y} = \arg\max_c \left[ \log P(y=c) + \sum_{j=1}^{d} \log P(x_j | y=c) \right]$$

这个公式展示了朴素贝叶斯的计算流程，包括如下三个步骤：

1. **先验概率 $\log P(y=c)$**：类别 $c$ 在训练集中的占比。如果训练集有 1000 封邮件，其中 300 封是垃圾邮件，则 $P(\text{垃圾邮件}) = 0.3$。
2. **条件概率 $\log P(x_j | y=c)$**：特征 $x_j$ 在类别 $c$ 中出现的概率。如果垃圾邮件中有 200 封包含"免费"，则 $P(\text{免费}|\text{垃圾邮件}) = 200/300 \approx 0.67$。
3. **加法求和**：将各特征的对数概率相加，得到每个类别的得分，选择得分最高的类别。

观察朴素贝叶斯的决策公式，不难发现一个概率连乘带来的致命问题：如果某个特征值在类别 $c$ 的训练样本中从未出现过，则 $P(x_j|y=c)=0$，导致整个概率乘积为零。譬如训练集中所有正常邮件都不包含"贷款"这个词，当一封新邮件出现"贷款"时 $P(\text{贷款}|\text{正常邮件}) = 0$。即使这封邮件其他词汇都指向正常邮件（如"会议"、"报告"），整个概率乘积也为零，朴素贝叶斯会错误地将其判定为垃圾邮件。

这个问题的解决方案是给每个计数加一个小常数 $\alpha$，确保所有概率值都大于零。这就像在投票系统中给每个候选人预设 1 张选票，即使某个候选人从未获得选民投票，也不会出现零票的极端情况，这种操作被称为**拉普拉斯平滑**（Laplace Smoothing），数学表示为 $P(x_j | y=c) = \frac{N_{jc} + \alpha}{N_c + \alpha \cdot V}$，其中 $N_{jc}$ 是类别 $c$ 中特征 $j$ 的词频总和（计数总和），$N_c$ 是类别 $c$ 的样本总数，$\alpha$ 是平滑参数（通常为 1），$V$ 是特征总数。


## 离散型朴素贝叶斯实践

离散型朴素贝叶斯适用于特征为离散值（如词频、类别标签）的场景。最典型的应用是文本分类，如垃圾邮件过滤、新闻分类、情感分析，等等。下面的代码实现了多项式朴素贝叶斯分类器，演示了从训练数据中学习词频特征与类别的关系，计算先验概率和条件概率，最后对新文档进行分类预测的完整过程。示例中使用了 5 个词汇特征对 6 个文档进行训练和 3 个测试文档的预测，并可视化了各类别下的词汇条件概率分布和测试文档的分类得分。

从运行后的可视化图表中可以清晰看到：正面类中"好"、"喜欢"等词汇条件概率高，负面类中"坏"、"讨厌"等词汇条件概率高。这正是朴素贝叶斯"学习"到的规律：通过简单的计数统计，它掌握了区分正面和负面的关键词汇。

```python runnable extract-class="MultinomialNaiveBayes"
import numpy as np

class MultinomialNaiveBayes:
    """
    多项式朴素贝叶斯实现
    适用于离散特征（如文本词频）
    """
    
    def __init__(self, alpha=1.0):
        """
        Parameters:
        alpha : float, 拉普拉斯平滑参数
        """
        self.alpha = alpha  # 拉普拉斯平滑
        self.class_prior_ = None  # P(y)
        self.feature_prob_ = None  # P(x|y)
        self.classes_ = None
    
    def fit(self, X, y):
        """
        训练模型
        
        Parameters:
        X : ndarray, shape (n_samples, n_features)
            特征矩阵（词频/计数）
        y : ndarray, shape (n_samples,)
            类别标签
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # 计算先验概率 P(y)
        class_counts = np.array([np.sum(y == c) for c in self.classes_])
        self.class_prior_ = class_counts / n_samples
        
        # 计算条件概率 P(x|y)
        # 对于每个类别，计算每个特征在该类别文档中的总计数
        self.feature_prob_ = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes_):
            # 获取类别c的所有样本
            X_c = X[y == c]
            # 该类别每个特征的总计数 + 平滑
            feature_counts = X_c.sum(axis=0) + self.alpha
            # 归一化得到条件概率
            total_count = feature_counts.sum()
            self.feature_prob_[i] = feature_counts / total_count
        
        return self
    
    def predict_log_proba(self, X):
        """
        计算对数概率
        """
        # log P(y) + sum(log P(x|y))
        log_prior = np.log(self.class_prior_)
        log_likelihood = X @ np.log(self.feature_prob_.T)  # (n_samples, n_classes)
        return log_prior + log_likelihood
    
    def predict(self, X):
        """
        预测类别
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# 模拟词频数据（5个词，6个文档）
# 特征：["好", "坏", "喜欢", "讨厌", "一般"]
X_train = np.array([
    [3, 0, 2, 0, 1],  # 文档1：好词多 → 正面
    [2, 1, 1, 0, 1],  # 文档2：偏正面
    [4, 0, 3, 0, 0],  # 文档3：明显正面
    [0, 3, 0, 2, 1],  # 文档4：坏词多 → 负面
    [1, 2, 0, 1, 2],  # 文档5：偏负面
    [0, 4, 0, 3, 0],  # 文档6：明显负面
])
y_train = np.array(['正面', '正面', '正面', '负面', '负面', '负面'])

# 训练模型
model = MultinomialNaiveBayes(alpha=1.0)
model.fit(X_train, y_train)

print("=== 朴素贝叶斯文本分类 ===")
print(f"类别: {model.classes_}")
print(f"先验概率: {dict(zip(model.classes_, model.class_prior_))}")
print(f"\n各特征在各类别中的条件概率:")
vocab = ["好", "坏", "喜欢", "讨厌", "一般"]
for i, c in enumerate(model.classes_):
    print(f"  {c}类: {dict(zip(vocab, model.feature_prob_[i].round(3)))}")

# 预测新文档
X_test = np.array([
    [2, 0, 1, 0, 0],  # 明显正面
    [0, 2, 0, 2, 0],  # 明显负面
    [1, 1, 1, 1, 1],  # 中性
])
y_pred = model.predict(X_test)
print(f"\n测试文档预测: {y_pred}")

# 输出对数概率
log_proba = model.predict_log_proba(X_test)
print(f"对数概率:")
for i, pred in enumerate(y_pred):
    print(f"  文档{i+1}: 正面={log_proba[i,0]:.2f}, 负面={log_proba[i,1]:.2f} → {pred}")

# 可视化：各类别特征概率对比
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：各类别下各词汇的条件概率
x_pos = np.arange(len(vocab))
width = 0.35

probs_positive = model.feature_prob_[0]  # 正面类
probs_negative = model.feature_prob_[1]  # 负面类

axes[0].bar(x_pos - width/2, probs_positive, width, label='正面', color='green', alpha=0.7)
axes[0].bar(x_pos + width/2, probs_negative, width, label='负面', color='red', alpha=0.7)
axes[0].set_xlabel('词汇')
axes[0].set_ylabel('条件概率 P(词|类别)')
axes[0].set_title('各类别词汇条件概率分布')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(vocab)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：测试文档的对数概率得分
test_labels = ['明显正面', '明显负面', '中性']
x_pos_test = np.arange(len(test_labels))  # 测试文档的x位置（3个）
axes[1].bar(x_pos_test - width/2, log_proba[:, 0], width, label='正面得分', color='green', alpha=0.7)
axes[1].bar(x_pos_test + width/2, log_proba[:, 1], width, label='负面得分', color='red', alpha=0.7)
axes[1].set_xlabel('测试文档')
axes[1].set_ylabel('对数概率得分')
axes[1].set_title('测试文档分类得分对比')
axes[1].set_xticks(x_pos_test)
axes[1].set_xticklabels(test_labels)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 标记预测结果
for i, pred in enumerate(y_pred):
    axes[1].annotate(f'预测: {pred}', 
                     xy=(i, log_proba[i, 0 if pred == '正面' else 1]),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.show()
plt.close()
```

## 本章小结

朴素贝叶斯的价值不局限在直接使用它能解决哪些问题，而是展示了概率机器学习的一套范式：**先假设数据的概率分布结构，再从数据中估计分布参数，最后用概率规则做出决策**。这套范式贯穿了整个贝叶斯机器学习领域，如[贝叶斯网络](bayesian-network.md)、隐马尔可夫模型、概率图模型等。理解朴素贝叶斯，就是理解了概率机器学习的起点。

## 练习题

1. 给定训练数据：垃圾邮件 60 封，正常邮件 40 封。垃圾邮件中"免费"出现 30 次，正常邮件中"免费"出现 5 次。计算：(a) 垃圾邮件的先验概率；(b) "免费"在垃圾邮件和正常邮件中的条件概率；(c) 如果一封邮件包含"免费"，用朴素贝叶斯判断它是否为垃圾邮件。
    <details>
    <summary>参考答案</summary>
    
    **(a) 先验概率**：
    $$P(\text{垃圾邮件}) = \frac{60}{60+40} = 0.6$$
    $$P(\text{正常邮件}) = \frac{40}{60+40} = 0.4$$
    
    **(b) 条件概率**（假设词汇表大小为 100，使用拉普拉斯平滑 $\alpha=1$）：
    
    垃圾邮件总词频未知，假设平均每封邮件 50 词，则垃圾邮件总词频约为 $60 \times 50 = 3000$。
    
    $$P(\text{免费}|\text{垃圾邮件}) = \frac{30 + 1}{3000 + 100} \approx 0.0103$$
    $$P(\text{免费}|\text{正常邮件}) = \frac{5 + 1}{2000 + 100} \approx 0.0029$$
    
    （实际计算中，总词频应从数据统计，此处为假设值）
    
    **(c) 分类判断**：
    
    计算"免费"出现在垃圾邮件和正常邮件中的后验概率（忽略分母）：
    
    $$\text{垃圾邮件得分} = \log P(\text{垃圾邮件}) + \log P(\text{免费}|\text{垃圾邮件}) = \log 0.6 + \log 0.0103 \approx -0.51 - 4.57 = -5.08$$
    
    $$\text{正常邮件得分} = \log P(\text{正常邮件}) + \log P(\text{免费}|\text{正常邮件}) = \log 0.4 + \log 0.0029 \approx -0.92 - 5.84 = -6.76$$
    
    垃圾邮件得分 $(-5.08) >$ 正常邮件得分 $(-6.76)$，因此判断为**垃圾邮件**。
    
    这符合直觉：虽然"免费"在正常邮件中也会出现，但在垃圾邮件中出现的频率更高，因此看到"免费"后，邮件更可能是垃圾邮件。
    </details>

2. 用代码实现一个垃圾邮件分类器，训练数据如下：
    - 垃圾邮件（3 封）：["免费中奖 点击领取", "限时优惠 买一送一", "贷款审批 即刻放款"]
    - 正常邮件（3 封）：["明天开会 请准时", "项目报告 请查收", "周末聚餐 时间确认"]
    
    预测以下邮件的类别："免费开会 贷款审批"。分析为什么这个预测结果可能不准确。
    <details>
    <summary>参考答案</summary>
    
    ```python runnable
    import numpy as np
    
    # 使用已定义的 MultinomialNaiveBayes 类
    from shared.linear.naive_bayes import MultinomialNaiveBayes
    
    # 构建词汇表
    spam_emails = ["免费中奖 点击领取", "限时优惠 买一送一", "贷款审批 即刻放款"]
    normal_emails = ["明天开会 请准时", "项目报告 请查收", "周末聚餐 时间确认"]
    
    all_emails = spam_emails + normal_emails
    vocab = set()
    for email in all_emails:
        vocab.update(email.split())
    vocab = sorted(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    print(f"词汇表: {vocab}")
    
    # 构建词频矩阵
    def email_to_vector(email):
        vec = np.zeros(len(vocab))
        for word in email.split():
            if word in word_to_idx:
                vec[word_to_idx[word]] += 1
        return vec
    
    X_train = np.array([email_to_vector(email) for email in all_emails])
    y_train = np.array(['垃圾'] * 3 + ['正常'] * 3)
    
    # 训练模型
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)
    
    print(f"\n先验概率: {dict(zip(model.classes_, model.class_prior_))}")
    
    # 预测
    test_email = "免费开会 贷款审批"
    X_test = email_to_vector(test_email).reshape(1, -1)
    prediction = model.predict(X_test)[0]
    
    print(f"\n测试邮件: '{test_email}'")
    print(f"预测类别: {prediction}")
    
    # 分析预测
    log_proba = model.predict_log_proba(X_test)[0]
    print(f"垃圾得分: {log_proba[0]:.2f}")
    print(f"正常得分: {log_proba[1]:.2f}")
    
    # 查看各词汇的条件概率
    print("\n各词汇在两类中的条件概率:")
    test_words = test_email.split()
    for word in test_words:
        if word in word_to_idx:
            idx = word_to_idx[word]
            print(f"  '{word}': P(垃圾)={model.feature_prob_[0, idx]:.3f}, P(正常)={model.feature_prob_[1, idx]:.3f}")
    ```
    
    **预测结果分析**：
    
    "免费"和"贷款审批"在垃圾邮件中出现，倾向于垃圾邮件判断；"开会"在正常邮件中出现，倾向于正常邮件判断。朴素贝叶斯会将这些倾向综合计算，选择得分最高的类别。
    
    **不准确的可能原因**：
    
    1. **词汇歧义**："开会"在正常邮件中出现，但"免费开会"可能是垃圾邮件中"免费"和正常邮件中"开会"的巧合组合。朴素贝叶斯假设独立，无法捕捉"免费开会"这一组合的特殊含义（可能是合法的会议邀请，而非垃圾邮件）。
    
    2. **特征相关性被忽略**："免费"和"贷款"往往在垃圾邮件中同时出现，表示某种推销行为；但朴素贝叶斯将它们视为独立，无法利用这种协同信息。
    
    3. **训练数据不足**：仅 6 封邮件的训练集太小，词汇的条件概率估计不稳定，容易受个别样本影响。
    
    这体现了朴素贝叶斯的局限性：简单假设带来高效计算，但也牺牲了对复杂语境的理解能力。
    </details>
