# 词嵌入与表示学习

前面的章节介绍了神经网络的基本结构和工作原理，这些模型处理的输入都是数值向量。后面将要介绍的序列模型与自然语言处理（Natural Language Processing, NLP）密切相关，但 NLP 面临的第一个问题是：人类使用的是符号（文字），而非神经网络擅长处理的数值。如何将离散的符号转换为连续的数值，让神经网络能够理解和处理自然语言？这个问题的答案就是词嵌入。

2003 年，约书亚·本吉奥（Yoshua Bengio）在论文《A Neural Probabilistic Language Model》中提出的神经概率语言模型可以说是现代 NLP 技术的起源，该方法首次引入了词的分布式表示（即后来的词嵌入概念）。词嵌入是一种将离散符号映射到连续向量空间的技术，得到的向量称为**词向量**（Word Vector）。它不仅解决了符号到数值的转换问题，更重要的是，这种映射能够捕捉词与词之间的语义关系。语义相似的词在向量空间中距离较近，语义无关的词距离较远。这种几何性质使得神经网络能够"理解"语言的含义，而非仅仅记忆符号的排列组合。本文将从最直观的 One-Hot 编码出发，分析其局限性，引出词嵌入的核心思想，并通过实验展示词嵌入的几何性质和实践应用。

## One-Hot 编码与词袋模型

将文字转换为数值最早也是最直观的方案是 **One-Hot 编码**（独热编码）。假设词汇表有 $V$ 个词，每个词被表示为一个 $V$ 维向量，该词对应的位置为 1，其余位置全为 0。譬如，假设词汇表为 `["春", "夏", "秋", "冬"]`，则各词的 One-Hot 编码为：

| 词 | One-Hot 向量 |
|:--:|:-------------|
| 春 | $[1, 0, 0, 0]$ |
| 夏 | $[0, 1, 0, 0]$ |
| 秋 | $[0, 0, 1, 0]$ |
| 冬 | $[0, 0, 0, 1]$ |

**词袋模型**（Bag of Words）在 1954 年由泽里格·哈里斯（Zellig Harris）提出，这是一种 One-Hot 的扩展形式，词袋模型对词汇表中每个词进行 One-Hot 编码，然后将这些编码向量相加（或计数）。具体做法是忽略文档中词语的顺序和语法结构，将文档视为词汇的"袋子"，只关注哪些词出现、出现多少次。它先构建一个包含所有文本词汇的词汇表，然后将一段文本表示为一个固定长度的向量，向量的每个维度对应词汇表中的一个词，数值代表该词在文本中的出现频率。尽管只通过词频来理解文本主题肯定不够准确，但是这种将文档转化数值矩阵的方法足够简单，通过统计词频来捕捉文档的语义内容，使计算机能够"理解"文本的主题和关键词分布。

One-Hot 编码与词袋模型简单直观，易于实现，但它们存在三个严重缺陷，使其难以满足实际的自然语言处理任务需要。

- 第一 **维度爆炸**。实际应用中，词汇表大小 $V$ 动辄数万甚至数十万。中文常用汉字约 3000-5000 个，但常用词汇可达数十万；英文词汇量更是庞大。每个词都需要一个 $V$ 维向量表示，当 $V = 50000$ 时，一个词的 One-Hot 向量就需要 50000 个数值。这不仅消耗大量存储空间，更导致神经网络的输入维度极高，权重矩阵的参数量随之爆炸式增长。

- 第二 **稀疏性**。One-Hot 向量只有一个位置为 1，其余 $V-1$ 个位置全为 0。这种极端稀疏的表示导致计算效率低下。假设一个神经网络接收 One-Hot 向量作为输入，输入层到第一隐藏层的权重矩阵维度为 $V \times d$，其中 $d$ 为隐藏层维度。前向传播时，输入向量与权重矩阵相乘 $\mathbf{h} = \mathbf{W} \cdot \mathbf{x}$，由于 $\mathbf{x}$ 只有一个位置非零，实际上只需要取出权重矩阵的一列，其他 $V-1$ 列的乘法运算完全是浪费。

- 第三 **无法表达语义关系**。这是 One-Hot 编码与池袋模型最根本的缺陷。在 One-Hot 表示下，任意两个不同词之间的欧氏距离都是 $\sqrt{2}$，余弦相似度都是 0。也就是说，"春"和"夏"的距离等于"春"和"冬"的距离，"猫"和"狗"的相似度等于"猫"和"汽车"的相似度。这种表示完全忽略了词与词之间的语义关联，神经网络无法从这种表示中学习到任何语义信息，只能从上下文位置信息中学习，而无法利用词本身的语义特征。

## 词嵌入

One-Hot 编码将每个词表示为 $V$ 维稀疏向量，词嵌入则将每个词表示为 $d$ 维稠密向量，其中 $d \ll V$。譬如，词汇表大小 $V = 50000$，嵌入维度 $d = 300$，则每个词从 50000 维的稀疏向量压缩为 300 维的稠密向量。这种压缩不是简单的降维，而是学习一种有意义的表示，使得语义相似的词在向量空间中距离较近。

词嵌入的思想可以用一个嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{V \times d}$ 来表示。矩阵的每一行对应词汇表中的一个词，是一个 $d$ 维向量。给定一个词的索引 $i$，其嵌入向量就是嵌入矩阵的第 $i$ 行：$\mathbf{e}_i = \mathbf{E}[i, :]$。从数学角度看，词嵌入可以理解为 One-Hot 向量与嵌入矩阵的矩阵乘法。设词 $w$ 的 One-Hot 向量为 $\mathbf{x} \in \mathbb{R}^V$（只有第 $i$ 个位置为 1），则嵌入向量为：

$$\mathbf{e} = \mathbf{E}^T \mathbf{x} = \mathbf{E}[i, :]^T$$

由于 $\mathbf{x}$ 只有一个位置非零，矩阵乘法实际上就是取出嵌入矩阵的第 $i$ 行。因此嵌入层在实现时就直接使用索引查找，无需矩阵乘法，效率更高，结果相同。词嵌入的关键优势在于它是可学习的。嵌入矩阵 $\mathbf{E}$ 是神经网络的参数，通过反向传播算法随模型训练而优化。训练过程中，模型会自动学习到有意义的词向量表示，经常出现在相似上下文中的词，其嵌入向量会逐渐靠近，语义无关的词，其嵌入向量会逐渐远离。这种学习机制使得词嵌入能够捕捉语言的统计规律和语义信息。

### 词嵌入的几何性质

词嵌入最迷人的特性是其几何性质。在训练好的词嵌入空间中，语义相似的词距离较近，语义无关的词距离较远。更神奇的是，词向量之间的方向可以表示语义关系。最著名的例子是：

$$\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$$

::: info 说明
下文的代码示例使用笔者手动构造的 3 维向量进行演示，目的是直观展示词向量运算的原理。实际应用中，词向量通常有 100-300 维，需要在大规模语料上通过 Word2Vec 或 GloVe 等方法训练得到。
:::

这个等式的含义是从"国王"向量减去"男人"向量，加上"女人"向量，结果接近"女王"向量。这表明词嵌入捕捉到了性别这一语义维度：$\vec{king} - \vec{queen}$ 的方向与 $\vec{man} - \vec{woman}$ 的方向相似，都表示从男性到女性的语义变化。衡量两个词向量相似度的标准方法是[余弦相似度](../../maths/linear/vectors.md#内积与投影)：

$$similarity(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \sqrt{\sum_{i=1}^{d} b_i^2}}$$

余弦相似度衡量两个向量的方向相似程度，取值范围 $[-1, 1]$。值为 1 表示方向完全相同，值为 0 表示正交（无关），值为 -1 表示方向相反。相比于欧氏距离，余弦相似度更关注向量的方向而非长度，更适合衡量语义相似度。

```python runnable
import numpy as np

# 模拟训练好的词嵌入（简化示例）
# 实际应用中需要大量数据训练得到
word_vectors = {
    "国王": np.array([0.8, 0.2, 0.9]),
    "女王": np.array([0.7, 0.8, 0.85]),
    "男人": np.array([0.9, 0.1, 0.3]),
    "女人": np.array([0.8, 0.7, 0.25]),
    "王子": np.array([0.75, 0.15, 0.7]),
    "公主": np.array([0.65, 0.75, 0.65]),
}

def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 验证词向量运算：king - man + woman ≈ queen
result = word_vectors["国王"] - word_vectors["男人"] + word_vectors["女人"]
print("词向量运算: 国王 - 男人 + 女人")
print(f"结果向量: {result.round(3)}")
print(f"女王向量: {word_vectors['女王'].round(3)}")
print(f"与女王的余弦相似度: {cosine_similarity(result, word_vectors['女王']):.4f}")

# 计算各词与结果的相似度
print("\n与各词的余弦相似度:")
for word, vec in word_vectors.items():
    sim = cosine_similarity(result, vec)
    print(f"  {word}: {sim:.4f}")

# 验证语义相似性
print("\n语义相似词的余弦相似度:")
print(f"  国王 vs 女王: {cosine_similarity(word_vectors['国王'], word_vectors['女王']):.4f}")
print(f"  王子 vs 公主: {cosine_similarity(word_vectors['王子'], word_vectors['公主']):.4f}")
print(f"  男人 vs 女人: {cosine_similarity(word_vectors['男人'], word_vectors['女人']):.4f}")
```

这段代码用简化的词向量演示了词嵌入的几何性质。实际应用中，词向量通常有数百维，需在大规模语料上训练。GloVe 和 Word2Vec 是两种最著名的预训练词向量方法，它们在大规模文本上训练得到的词向量展现出了丰富的语义关系。

### 词嵌入实践

理解了词嵌入的原理后，让我们通过 PyTorch 实践如何使用嵌入层。`nn.Embedding` 封装了嵌入矩阵的存储和查找操作，有两个关键参数：`num_embeddings` 表示词汇表大小 $V$（嵌入矩阵的行数），`embedding_dim` 表示嵌入维度 $d$（嵌入矩阵的列数）。

嵌入层的输入是词索引（整数张量），输出是对应的嵌入向量。输入可以是任意形状的张量，输出形状会在末尾增加一维（嵌入维度）。后面介绍的 LSTM、GRU 等序列模型，其首层通常都是嵌入层。

```python runnable
import torch
import torch.nn as nn

# 创建嵌入层
vocab_size = 1000  # 词汇表大小
embedding_dim = 64  # 嵌入维度

embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

print(f"嵌入矩阵形状: {embedding_layer.weight.shape}")
print(f"参数量: {vocab_size * embedding_dim:,}")

# 单个词的嵌入
word_idx = torch.tensor([42])
embedding = embedding_layer(word_idx)
print(f"\n输入形状: {word_idx.shape}")
print(f"输出形状: {embedding.shape}")

# 批量词的嵌入
batch_indices = torch.tensor([[1, 42, 100], [200, 300, 999]])
batch_embeddings = embedding_layer(batch_indices)
print(f"\n批量输入形状: {batch_indices.shape}")
print(f"批量输出形状: {batch_embeddings.shape}")

# 与下游任务联合训练示例
class TextClassifier(nn.Module):
    """简单的文本分类模型"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        # 简单平均池化
        pooled = embedded.mean(dim=1)  # (batch_size, embedding_dim)
        hidden = self.relu(self.fc(pooled))
        return self.output(hidden)

# 创建模型
model = TextClassifier(vocab_size=1000, embedding_dim=64, hidden_dim=32, num_classes=3)
print(f"\n模型结构:\n{model}")

# 前向传播测试
sample_input = torch.randint(0, 1000, (4, 10))  # batch_size=4, seq_len=10
output = model(sample_input)
print(f"\n输入形状: {sample_input.shape}")
print(f"输出形状: {output.shape}")
```

上述代码展示了 `nn.Embedding` 的基本用法以及如何与下游任务联合训练。嵌入层作为模型的第一层，将词索引转换为稠密向量，后续层基于这些向量进行计算。训练时，嵌入矩阵的参数会随整个模型一起优化，学习到适合当前任务的词向量表示。

### 预训练词向量

虽然嵌入层可以与下游任务联合训练，但在数据量较小的情况下，随机初始化的嵌入矩阵难以学习到高质量的词向量表示。**预训练词向量**（Pretrained Word Embeddings）是解决这个问题的有效方法。预训练词向量是在大规模语料（如维基百科、Common Crawl）上训练得到的，蕴含了丰富的语义信息，可以直接加载到嵌入层使用。两种最著名的预训练词向量方法是 Word2Vec 和 GloVe：

- **Word2Vec**：由 Google 于 2013 年提出，主要思想是出现在相似上下文中的词具有相似的含义。通过预测上下文或中心词，模型学习到能够捕捉语义信息的词向量。Word2Vec 包括两种训练方式：
    - Skip-Gram：给定中心词，预测上下文词。训练目标是最大化上下文词的预测概率。
    - CBOW（Continuous Bag of Words）：给定上下文词，预测中心词。训练目标是最小化中心词的预测误差。

- **GloVe**（Global Vectors for Word Representation）：由斯坦福大学于 2014 年提出，基于全局词共现统计。GloVe 构建词共现矩阵，统计任意两个词在窗口内共同出现的次数，然后通过矩阵分解学习词向量。GloVe 结合了全局统计信息和局部上下文信息，在许多任务上表现优于 Word2Vec。

预训练词向量的使用方式也有两种：一种是固定嵌入层，加载预训练词向量后，冻结嵌入层参数，不参与训练，适用于下游任务数据量小、预训练词向量质量高的场景；另一种是微调嵌入层，加载预训练词向量作为初始值，训练时允许嵌入层参数更新，适用于下游任务数据量充足、需要学习任务特定语义的场景。

## 本章小结

词嵌入是神经网络处理自然语言的基础技术，解决了将离散符号转换为连续数值的核心问题。从 One-Hot 编码到词嵌入，不仅是维度的压缩，更是从符号表示到语义表示的质变。词嵌入为后续的序列模型奠定了基础。LSTM、Transformer 等模型都使用词嵌入作为输入表示，在词嵌入的基础上学习序列的上下文依赖关系。理解词嵌入，是理解现代自然语言处理技术的起点。

## 练习题

1. 假设词汇表大小为 10000，嵌入维度为 300。计算 One-Hot 编码和词嵌入两种表示方式下，存储所有词向量所需的参数量。如果使用 FP32（4 字节）存储，各需要多少内存？
   <details>
   <summary>参考答案</summary>

   **One-Hot 编码**：
   - 每个词向量维度：10000
   - 总参数量：$10000 \times 10000 = 100,000,000$（1 亿）
   - 内存占用：$100,000,000 \times 4 = 400,000,000$ 字节 ≈ 400 MB

   **词嵌入**：
   - 每个词向量维度：300
   - 总参数量：$10000 \times 300 = 3,000,000$（300 万）
   - 内存占用：$3,000,000 \times 4 = 12,000,000$ 字节 ≈ 12 MB

   词嵌入的参数量仅为 One-Hot 的 3%，内存占用从 400 MB 降至 12 MB。
   </details>

2. 给定两个词向量 $\mathbf{a} = [0.8, 0.6]$ 和 $\mathbf{b} = [0.6, 0.8]$，计算它们的余弦相似度和欧氏距离。如果 $\mathbf{a}$ 表示"猫"，$\mathbf{b}$ 表示"狗"，这两个向量表示它们语义相似还是不同？
   <details>
   <summary>参考答案</summary>

   **余弦相似度**：
   $$\cos(\mathbf{a}, \mathbf{b}) = \frac{0.8 \times 0.6 + 0.6 \times 0.8}{\sqrt{0.8^2 + 0.6^2} \times \sqrt{0.6^2 + 0.8^2}} = \frac{0.48 + 0.48}{1.0 \times 1.0} = 0.96$$

   **欧氏距离**：
   $$d(\mathbf{a}, \mathbf{b}) = \sqrt{(0.8-0.6)^2 + (0.6-0.8)^2} = \sqrt{0.04 + 0.04} = \sqrt{0.08} \approx 0.28$$

   余弦相似度 0.96 接近 1，说明两个向量方向几乎相同，语义高度相似。这符合"猫"和"狗"都是宠物、动物的特点。
   </details>
