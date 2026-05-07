---
title: "Seq2Seq序列映射"
date: 2026-05-07
tags: [deep-learning, seq2seq, attention, sequence-models]
series:
  name: "深度学习经典模型"
  chapter: 6
  order: 3
---

# Seq2Seq 序列映射

前两篇文章介绍了 RNN、LSTM 和 GRU——这些网络能够处理序列数据，利用历史信息做出当前决策。但它们有一个共同特点：**输入序列和输出序列长度相同**（或在"多对一"模式下输出是单个值）。

许多实际任务不符合这个假设：

- **机器翻译**：输入英文句子（5 个词），输出中文翻译（7 个字）——长度不同
- **文本摘要**：输入长文章（500 词），输出摘要（50 词）——长度不同
- **对话系统**：输入用户问题（10 词），输出回答（20 词）——长度不同
- **语音识别**：输入语音信号（100 帧），输出文本（30 词）——长度不同

这类任务的核心需求：**将一个序列映射到另一个长度不同的序列**。这正是 **Seq2Seq**（Sequence-to-Sequence，序列到序列）模型的设计目标。

Seq2Seq 的核心架构是**编码器-解码器**（Encoder-Decoder）：
- **编码器**：将输入序列压缩为固定长度的向量表示（编码向量）
- **解码器**：根据编码向量，逐步生成输出序列

这个架构将序列映射分为两个阶段：
1. **理解阶段**：编码器"理解"输入序列，提取关键信息
2. **生成阶段**：解码器根据理解的内容，"生成"输出序列

2014 年，Sutskever 等人提出的 Seq2Seq 模型在机器翻译任务上取得突破，成为后续神经机器翻译、对话系统、文本摘要等任务的基础架构。更重要的是，Seq2Seq 的局限性催生了**注意力机制**（Attention Mechanism）——这是深度学习历史上最重要的创新之一，直接催生了 Transformer 和 GPT 等现代架构。

本文将介绍 Seq2Seq 的原理、训练方法、局限性，以及注意力机制的雏形。

## 编码器-解码器架构

### 架构设计思想

Seq2Seq 的核心问题是：如何将变长输入序列映射到变长输出序列？

**朴素方案的困境**：

尝试直接用 RNN 处理：
- 输入序列长度 $T$，输出序列长度 $T'$，要求 $T = T'$ —— 但实际任务中 $T \neq T'$ 是常态
- 强行对齐（如填充到相同长度）会导致输出序列包含大量无意义的填充内容

**编码器-解码器方案的解决思路**：

将问题分解为两个阶段：

```
Seq2Seq 架构示意:

输入序列: [x_1, x_2, ..., x_T]  (长度 T)
         ↓
    ┌────────────────────┐
    │     编码器         │  ← 逐时刻处理输入，累积信息
    │   (LSTM/GRU)      │
    └────────────────────┘
         ↓
    编码向量: h_enc      ← 输入序列的"压缩表示"
         ↓
    ┌────────────────────┐
    │     解码器         │  ← 根据编码向量，逐时刻生成输出
    │   (LSTM/GRU)      │
    └────────────────────┘
         ↓
输出序列: [y_1, y_2, ..., y_T']  (长度 T'，可能与 T 不同)
```

**关键设计**：
- 编码器将输入序列的所有信息压缩到一个固定维度的向量 $h_{enc}$（编码向量）
- 解码器以 $h_{enc}$ 为起点，逐步生成输出序列，每个时刻生成一个输出
- 输入和输出长度解耦：编码器处理 $T$ 个时刻，解码器生成 $T'$ 个时刻，两者独立

### 编码器的工作原理

编码器的作用：**将输入序列压缩为向量表示**。

使用 LSTM 或 GRU 作为编码器，逐时刻处理输入序列：

```
编码器处理过程:

时刻 1: 输入 x_1 → LSTM → h_1
时刻 2: 输入 x_2 → LSTM → h_2 (融合 h_1)
时刻 3: 输入 x_3 → LSTM → h_3 (融合 h_2)
...
时刻 T: 输入 x_T → LSTM → h_T (融合 h_{T-1})

编码向量: h_enc = h_T  (最后时刻的隐藏状态)
```

**编码向量 $h_{enc}$ 的含义**：

$h_{enc}$ 是编码器最后时刻的隐藏状态，理论上包含了输入序列的所有信息。它是对输入序列的"压缩表示"或"语义编码"。

**类比理解**：

将编码器类比为阅读理解：
- 输入序列是一篇文章
- 编码器逐词阅读，理解文章内容
- 编码向量是读完后的"理解状态"——对文章内容的抽象总结
- 这个"理解状态"用于后续回答问题（解码器生成）

**数学表示**：

使用 LSTM 作为编码器：

$$h_t, c_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1})$$

编码向量：

$$h_{enc} = h_T$$

（也可以使用 $c_T$ 作为编码向量，或 $h_T$ 和 $c_T$ 的组合）

### 解码器的工作原理

解码器的作用：**根据编码向量生成输出序列**。

解码器是另一个 LSTM 或 GRU，其初始隐藏状态由编码向量提供：

```
解码器生成过程:

初始状态: h_0 = h_enc (编码向量)
         c_0 = c_enc 或 0

时刻 1: 输入 <START> → LSTM → h_1 → y_1 (第一个输出词)
时刻 2: 输入 y_1 → LSTM → h_2 → y_2 (第二个输出词)
时刻 3: 输入 y_2 → LSTM → h_3 → y_3 (第三个输出词)
...
时刻 T': 输入 y_{T'-1} → LSTM → h_{T'} → y_{T'} = <END> (结束标志)
```

**解码方式**：

解码器有两种生成方式：

1. **教师强制**（Teacher Forcing）：训练时使用真实目标作为输入

```
时刻 1: 输入 <START> → LSTM → y_1
时刻 2: 输入 真实目标_1 → LSTM → y_2  (使用真实的上一个词，而非模型预测)
时刻 3: 输入 真实目标_2 → LSTM → y_3
```

教师强制加速训练收敛，因为模型总是获得正确的上下文。

2. **自由生成**（Free Running）：推理时使用模型预测作为输入

```
时刻 1: 输入 <START> → LSTM → y_1 (预测)
时刻 2: 输入 y_1 → LSTM → y_2 (使用预测的上一个词)
时刻 3: 输入 y_2 → LSTM → y_3
```

自由生成是实际使用时的模式，但训练时单独使用会导致误差累积（早期的预测错误影响后续所有生成）。

**数学表示**：

解码器的初始状态：

$$h_0^{dec} = h_{enc}$$

$$c_0^{dec} = c_{enc}$$ 或 $c_0^{dec} = \vec{0}$

解码器每个时刻的更新：

$$h_t^{dec}, c_t^{dec} = \text{LSTM}(y_{t-1}, h_{t-1}^{dec}, c_{t-1}^{dec})$$

输出计算：

$$y_t = \text{softmax}(W \cdot h_t^{dec})$$

（$y_t$ 是词汇表上的概率分布，取概率最大的词作为输出）

### Seq2Seq 的整体流程

将编码器和解码器组合起来，Seq2Seq 的完整流程：

```
完整 Seq2Seq 流程:

阶段 1: 编码
输入序列 "I love you"
↓
编码器 LSTM:
  I → h_1
  love → h_2
  you → h_3 (= h_enc)
↓
编码向量 h_enc = "I love you 的语义表示"

阶段 2: 解码
解码器 LSTM 初始状态 = h_enc
↓
<START> → LSTM → "我" (y_1)
"我" → LSTM → "爱" (y_2)
"爱" → LSTM → "你" (y_3)
"你" → LSTM → <END> (结束)
↓
输出序列 "我爱你"
```

**信息流动的视角**：

- 编码阶段：信息从输入序列流向编码向量（压缩过程）
- 解码阶段：信息从编码向量流向输出序列（展开过程）
- 编码向量是连接编码器和解码器的"信息桥梁"

### PyTorch 实现

```python runnable
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size, hidden_size):
        super().__init__()
        
        # 编码器
        self.encoder_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # 解码器
        self.decoder_embedding = nn.Embedding(output_vocab_size, embed_size)
        self.decoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_size, output_vocab_size)
        
        self.hidden_size = hidden_size
    
    def encode(self, input_seq):
        """编码器处理输入序列"""
        # input_seq: (batch, seq_len) → 嵌入 → (batch, seq_len, embed_size)
        embedded = self.encoder_embedding(input_seq)
        
        # LSTM 处理
        # output: (batch, seq_len, hidden_size)
        # h_n, c_n: (1, batch, hidden_size) - 最后时刻的状态
        _, (h_n, c_n) = self.encoder_lstm(embedded)
        
        return h_n, c_n  # 编码向量
    
    def decode(self, target_seq, h_0, c_0):
        """解码器生成输出序列（教师强制模式）"""
        # target_seq: (batch, seq_len) → 嵌入 → (batch, seq_len, embed_size)
        embedded = self.decoder_embedding(target_seq)
        
        # LSTM 处理，初始状态为编码向量
        output, _ = self.decoder_lstm(embedded, (h_0, c_0))
        
        # 输出层：隐藏状态 → 词汇表概率分布
        logits = self.decoder_fc(output)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def forward(self, input_seq, target_seq):
        """完整 Seq2Seq 流程"""
        # 编码
        h_enc, c_enc = self.encode(input_seq)
        
        # 解码
        logits = self.decode(target_seq, h_enc, c_enc)
        
        return logits

# 创建模型
input_vocab_size = 100  # 输入词汇表大小
output_vocab_size = 100  # 输出词汇表大小
embed_size = 32
hidden_size = 64

model = Seq2Seq(input_vocab_size, output_vocab_size, embed_size, hidden_size)

# 模拟输入
batch_size = 2
input_seq_len = 5
target_seq_len = 6

input_seq = torch.randint(0, input_vocab_size, (batch_size, input_seq_len))
target_seq = torch.randint(0, output_vocab_size, (batch_size, target_seq_len))

# 前向传播
logits = model(input_seq, target_seq)

print(f"输入序列形状: {input_seq.shape}")
print(f"目标序列形状: {target_seq.shape}")
print(f"输出 logits 形状: {logits.shape}")
print(f"输出词汇表概率分布: logits 的最后一维大小为 {output_vocab_size}")
print("Seq2Seq 模型构建成功")
```

## 序列到序列映射

### 机器翻译示例

以机器翻译为例，详细说明 Seq2Seq 的工作过程：

**任务**：将英文 "I love you" 翻译为中文 "我爱你"

**词汇表**：
- 输入词汇表（英文）：["I", "love", "you", ...]，共 $V_{in}$ 个词
- 输出词汇表（中文）：["我", "爱", "你", "<START>", "<END>", ...]，共 $V_{out}$ 个词

**输入预处理**：
- 将英文句子转换为词索引序列：["I", "love", "you"] → [10, 25, 30]
- 添加结束标志：[10, 25, 30, <END>] → [10, 25, 30, 1]

**编码过程**：

```
时刻 1: 输入词索引 10 ("I") → 嵌入 → LSTM → h_1
时刻 2: 输入词索引 25 ("love") → 嵌入 → LSTM → h_2
时刻 3: 输入词索引 30 ("you") → 嵌入 → LSTM → h_3
时刻 4: 输入词索引 1 (<END>) → 嵌入 → LSTM → h_4 = h_enc

编码向量: h_enc = h_4
```

编码向量 $h_{enc}$ 包含了 "I love you" 的语义信息。

**解码过程**：

```
初始状态: h_0 = h_enc, c_0 = c_enc

时刻 1: 输入 <START> 索引 → 嵌入 → LSTM → h_1 → softmax → 
        概率分布: {"我": 0.8, "爱": 0.1, ...} → 输出 "我" (索引 50)

时刻 2: 输入 "我" 索引 50 → 嵌入 → LSTM → h_2 → softmax →
        概率分布: {"爱": 0.9, "你": 0.05, ...} → 输出 "爱" (索引 51)

时刻 3: 输入 "爱" 索引 51 → 嵌入 → LSTM → h_3 → softmax →
        概率分布: {"你": 0.85, <END>: 0.1, ...} → 输出 "你" (索引 52)

时刻 4: 输入 "你" 索引 52 → 嵌入 → LSTM → h_4 → softmax →
        概率分布: {<END>: 0.95, ...} → 输出 <END> → 结束
```

**输出结果**：["我", "爱", "你", <END>] → 翻译完成

### 损失函数计算

Seq2Seq 的损失函数是解码器每个时刻的输出与真实目标的交叉熵损失之和：

$$L = \sum_{t=1}^{T'} L_t(y_t, \text{target}_t)$$

其中 $L_t$ 是时刻 $t$ 的交叉熵损失：

$$L_t = -\log P(\text{target}_t | y_{t-1}, ..., y_1, h_{enc})$$

**训练示例**：

```
真实目标序列: ["我", "爱", "你", <END>] (索引 [50, 51, 52, 1])

解码器预测:
时刻 1: 概率分布 {"我": 0.8, "爱": 0.1, ...} → L_1 = -log(0.8) = 0.22
时刻 2: 概率分布 {"爱": 0.9, "你": 0.05, ...} → L_2 = -log(0.9) = 0.11
时刻 3: 概率分布 {"你": 0.85, <END>: 0.1, ...} → L_3 = -log(0.85) = 0.16
时刻 4: 概率分布 {<END>: 0.95, ...} → L_4 = -log(0.95) = 0.05

总损失: L = L_1 + L_2 + L_3 + L_4 = 0.54
```

训练目标：通过反向传播调整编码器和解码器的参数，使解码器的预测概率分布接近真实目标。

## 注意力机制雏形

### Seq2Seq 的核心问题

Seq2Seq 的编码器将整个输入序列压缩到一个固定维度的向量 $h_{enc}$。这带来一个严重问题：**信息瓶颈**。

**信息瓶颈问题**：

当输入序列较长时，编码向量难以无损地存储所有信息：

```
输入序列: "The cat, which already ate a fish, was hungry"

编码器需要压缩:
- "The cat" (主语)
- "which already ate a fish" (从句，描述 cat 的动作)
- "was hungry" (主句谓语)

编码向量维度有限（如 256），无法存储所有细节
```

**影响**：解码器生成时，只能依赖"有损压缩"的编码向量：
- 生成 "hungry" 时，需要知道主语是 "cat"
- 生成翻译时，需要知道 "ate a fish" 的上下文
- 但编码向量可能丢失了部分细节信息

**实证观察**：

研究表明，当输入序列长度超过 15-20 个词时，Seq2Seq 的翻译质量显著下降。这是因为编码器无法有效压缩长序列的所有信息。

### 注意力机制的动机

**核心问题**：解码器在生成每个词时，依赖的是同一个编码向量 $h_{enc}$。这忽略了不同时刻需要的信息不同：

- 生成 "我" 时，需要关注输入的 "I"
- 生成 "爱" 时，需要关注输入的 "love"
- 生成 "你" 时，需要关注输入的 "you"

**理想方案**：解码器在每个时刻，能够"动态"地从编码器获取不同部分的信息，而非依赖一个固定的编码向量。

这就是**注意力机制**（Attention Mechanism）的核心思想：让解码器在生成每个词时，"注意"到输入序列中与当前生成最相关的部分。

### 注意力机制的雏形设计

Bahdanau 等人在 2015 年提出的注意力机制，让解码器在生成每个时刻，动态获取编码器的隐藏状态：

```
Seq2Seq + Attention 架构:

编码器:
输入序列 → LSTM → 输出所有时刻的隐藏状态 [h_1, h_2, ..., h_T]
          (而非仅使用最后时刻的 h_T)

解码器:
时刻 t 的生成过程:
  1. 计算注意力权重 α_t,i (解码器状态与编码器各时刻状态的相关性)
  2. 计算上下文向量 c_t = Σ α_t,i · h_i (加权求和编码器状态)
  3. 解码器 LSTM 输入: c_t + 上一个输出词
  4. 生成当前输出词 y_t
```

**关键创新**：
- 编码器不再仅输出最后时刻的状态，而是输出所有时刻的状态序列
- 解码器在每个时刻计算一个"上下文向量" $c_t$，动态聚合编码器的信息
- $c_t$ 根据当前生成需求，关注输入序列的不同部分

### 注意力计算过程

**注意力权重计算**：

解码器时刻 $t$，计算注意力权重 $\alpha_{t,i}$（对编码器时刻 $i$ 的注意力）：

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}$$

其中 $e_{t,i}$ 是"能量值"，表示解码器状态 $h_t^{dec}$ 与编码器状态 $h_i^{enc}$ 的相关性：

$$e_{t,i} = \text{score}(h_t^{dec}, h_i^{enc})$$

常见的 score 函数：
- **点积**：$\text{score}(h_t, h_i) = h_t^T h_i$
- **加性**：$\text{score}(h_t, h_i) = v^T \tanh(W_1 h_t + W_2 h_i)$

**上下文向量计算**：

$$c_t = \sum_{i=1}^{T} \alpha_{t,i} h_i^{enc}$$

$c_t$ 是编码器所有时刻状态的加权求和，权重由注意力 $\alpha_{t,i}$ 决定。

**解码器更新**：

解码器 LSTM 的输入包括上下文向量：

$$h_t^{dec} = \text{LSTM}([y_{t-1}, c_t], h_{t-1}^{dec})$$

其中 $[y_{t-1}, c_t]$ 表示拼接上一个输出词和上下文向量。

### 注意力机制的可视化

注意力权重可以可视化，直观展示解码器在生成每个词时关注了输入序列的哪个部分：

```
翻译: "I love you" → "我爱你"

注意力权重矩阵:

        I    love   you
我     0.9   0.05   0.05   ← 生成"我"时，主要关注"I"
爱     0.1   0.85   0.05   ← 生成"爱"时，主要关注"love"
你     0.05  0.1    0.85   ← 生成"你"时，主要关注"you"
```

这种可视化验证了注意力的效果：解码器在生成每个词时，确实关注了输入序列中语义对应的部分。

### 注意力机制的效果

加入注意力机制后，Seq2Seq 解决了信息瓶颈问题：

| 特性 | 标准 Seq2Seq | Seq2Seq + Attention |
|:-----|:-------------|:---------------------|
| 编码器输出 | 仅最后时刻状态 $h_T$ | 所有时刻状态 $[h_1, ..., h_T]$ |
| 解码器输入 | 固定编码向量 $h_{enc}$ | 动态上下文向量 $c_t$ |
| 长序列处理 | 性能下降（信息瓶颈） | 性能稳定（动态关注） |
| 可解释性 | 无 | 注意力权重可可视化 |

**实证效果**：

Bahdanau 等人的实验表明，加入注意力机制后，Seq2Seq 在长句子翻译任务上的 BLEU 分数提升 5-7 分，显著改善了翻译质量。

更重要的是，注意力机制的思想成为后续 Transformer、BERT、GPT 等模型的基础——这些现代架构的核心都是注意力机制，而非 LSTM/GRU。

## Seq2Seq 训练技巧

### 教师强制与Scheduled Sampling

**教师强制**（Teacher Forcing）是训练 Seq2Seq 的常用技巧：

训练时，解码器的输入使用真实目标序列，而非模型预测：

```
教师强制训练:

时刻 1: 输入 <START> → LSTM → 预测 y_1
        真实目标: target_1 → 计算损失 L_1

时刻 2: 输入 target_1 (真实，而非 y_1) → LSTM → 预测 y_2
        真实目标: target_2 → 计算损失 L_2

时刻 3: 输入 target_2 (真实) → LSTM → 预测 y_3
        ...
```

**优点**：
- 训练收敛快（模型总是获得正确的上下文）
- 梯度稳定（真实目标提供稳定的监督信号）

**缺点**：
- 训练和推理不一致（训练用真实目标，推理用模型预测）
- 可能导致"误差累积"问题：推理时早期的预测错误影响后续生成

**Scheduled Sampling**（计划采样）缓解这个问题：

训练过程中，逐步从教师强制过渡到自由生成：

```
Scheduled Sampling 策略:

训练初期: 100% 使用教师强制（真实目标）
训练中期: 50% 使用教师强制，50% 使用模型预测
训练后期: 10% 使用教师强制，90% 使用模型预测

逐步增加模型预测的比例，使训练和推理更一致
```

### 束搜索（Beam Search）

推理时，解码器需要选择输出词。最简单的方法是**贪婪搜索**（Greedy Search）：每个时刻选择概率最高的词。

```
贪婪搜索:

时刻 1: 概率分布 → 最高概率词 "我"
时刻 2: 输入 "我" → 概率分布 → 最高概率词 "爱"
时刻 3: 输入 "爱" → 概率分布 → 最高概率词 "你"
```

贪婪搜索的问题：局部最优不等于全局最优。某个时刻选择最高概率词，可能导致后续生成质量下降。

**束搜索**（Beam Search）保留多个候选序列，而非仅保留一个：

```
束搜索（Beam Width = 2）:

时刻 1:
  输入 <START> → 概率分布
  选择概率最高的 2 个词: "我"(0.8), "吾"(0.1)
  保留 2 个候选: ["我", "吾"]

时刻 2:
  输入 "我" → 概率分布: {"爱": 0.9, "喜欢": 0.08}
  输入 "吾" → 概率分布: {"爱": 0.7, "喜欢": 0.2}
  
  计算 4 个候选序列的概率:
  "我爱": 0.8 * 0.9 = 0.72
  "我喜欢": 0.8 * 0.08 = 0.064
  "吾爱": 0.1 * 0.7 = 0.07
  "吾喜欢": 0.1 * 0.2 = 0.02
  
  选择概率最高的 2 个序列: ["我爱", "吾爱"]

时刻 3:
  继续扩展...
```

束搜索保留多个候选，最终选择概率最高的完整序列。这比贪婪搜索更可能找到全局最优。

**束宽度选择**：
- Beam Width = 1：贪婪搜索（最快，但质量可能差）
- Beam Width = 5-10：常用选择（质量与速度平衡）
- Beam Width > 20：质量提升有限，但计算开销大增

### 处理变长序列

**动态结束检测**：

解码器生成结束标志 `<END>` 时，停止生成：

```
动态结束:

时刻 t: 输出 <END> → 停止生成
序列长度 = t（而非固定长度）
```

**最大长度限制**：

设置最大生成长度，防止无限生成：

```
最大长度 = 50

如果时刻 50 仍未生成 <END>:
  强制停止，输出当前序列
```

**长度惩罚**：

束搜索时，较长的序列概率自然较低（更多词的概率乘积）。长度惩罚调整评分：

$$\text{score}_{LP} = \frac{\text{score}}{(5 + \text{length})^\alpha / (5 + 1)^\alpha}$$

其中 $\alpha$ 是惩罚系数（通常 0.6-1.0）。

### PyTorch 完整训练示例

```python runnable
import torch
import torch.nn as nn
import torch.optim as optim

# 定义带注意力的 Seq2Seq
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size, hidden_size):
        super().__init__()
        
        # 编码器
        self.encoder_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # 解码器
        self.decoder_embedding = nn.Embedding(output_vocab_size, embed_size)
        self.decoder_lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_size, output_vocab_size)
        
        # 注意力层
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_weight = nn.Linear(hidden_size, 1)
        
        self.hidden_size = hidden_size
    
    def encode(self, input_seq):
        embedded = self.encoder_embedding(input_seq)
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(embedded)
        return encoder_outputs, h_n, c_n
    
    def attention_step(self, decoder_hidden, encoder_outputs):
        """计算注意力权重和上下文向量"""
        # decoder_hidden: (1, batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)
        
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # 扩展 decoder_hidden 以匹配 encoder_outputs
        decoder_hidden = decoder_hidden.squeeze(0)  # (batch, hidden_size)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(batch_size, seq_len, self.hidden_size)
        
        # 计算注意力能量
        concat = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)
        energy = torch.tanh(self.attention(concat))
        attention_weights = self.attention_weight(energy).squeeze(2)  # (batch, seq_len)
        
        # Softmax 归一化
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 计算上下文向量
        context = torch.sum(attention_weights.unsqueeze(2) * encoder_outputs, dim=1)
        
        return context, attention_weights
    
    def decode_step(self, input_token, hidden, cell, encoder_outputs):
        """单步解码"""
        # 嵌入输入词
        embedded = self.decoder_embedding(input_token)
        
        # 计算注意力
        context, _ = self.attention_step(hidden, encoder_outputs)
        
        # 拼接嵌入和上下文向量
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        
        # LSTM 处理
        output, (hidden, cell) = self.decoder_lstm(lstm_input, (hidden, cell))
        
        # 输出层
        logits = self.decoder_fc(output.squeeze(1))
        
        return logits, hidden, cell
    
    def forward(self, input_seq, target_seq):
        """完整前向传播"""
        # 编码
        encoder_outputs, h_enc, c_enc = self.encode(input_seq)
        
        # 解码
        batch_size = input_seq.size(0)
        target_len = target_seq.size(1)
        
        outputs = []
        hidden, cell = h_enc, c_enc
        
        for t in range(target_len):
            input_token = target_seq[:, t].unsqueeze(1)
            logits, hidden, cell = self.decode_step(input_token, hidden, cell, encoder_outputs)
            outputs.append(logits)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs

# 创建模型
input_vocab_size = 100
output_vocab_size = 100
embed_size = 32
hidden_size = 64

model = Seq2SeqWithAttention(input_vocab_size, output_vocab_size, embed_size, hidden_size)

# 模拟数据
batch_size = 4
input_len = 8
output_len = 10

input_seq = torch.randint(0, input_vocab_size, (batch_size, input_len))
target_seq = torch.randint(0, output_vocab_size, (batch_size, output_len))

# 前向传播
outputs = model(input_seq, target_seq)

print(f"输入序列形状: {input_seq.shape}")
print(f"目标序列形状: {target_seq.shape}")
print(f"输出 logits 形状: {outputs.shape}")
print("带注意力的 Seq2Seq 模型构建成功")

# 训练示例（简化）
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 模拟训练一步
optimizer.zero_grad()
outputs = model(input_seq, target_seq)
loss = criterion(outputs.view(-1, output_vocab_size), target_seq.view(-1))
loss.backward()
optimizer.step()

print(f"训练损失: {loss.item():.4f}")
print("Seq2Seq + Attention 训练流程成功")
```

## 小结

本文介绍了 Seq2Seq 模型的原理和训练方法：

**编码器-解码器架构**：
- 编码器将输入序列压缩为编码向量
- 解码器根据编码向量生成输出序列
- 输入和输出长度解耦，实现序列到序列的映射

**注意力机制的雏形**：
- 解决了编码向量的信息瓶颈问题
- 解码器在每个时刻动态获取编码器信息
- 注意力权重可视化展示信息关注点

**训练技巧**：
- 教师强制加速收敛，Scheduled Sampling 缓解训练-推理差异
- 束搜索提升生成质量，长度惩罚调整评分
- 动态结束检测处理变长序列

**Seq2Seq 的局限性**：
- 编码向量难以存储长序列的所有信息（注意力机制解决）
- LSTM/GRU 的序列计算无法并行（Transformer 解决）
- 注意力机制虽然有效，但计算开销较大（Transformer 优化）

下一章将介绍生成模型——VAE 和 GAN，探索深度学习如何生成新数据（而非映射现有数据）。

---

## 练习题

**1. 理论推导**

推导 Seq2Seq 的损失函数：

$$L = \sum_{t=1}^{T'} L_t(y_t, \text{target}_t)$$

解释为什么使用交叉熵损失而非 MSE 损失。

**2. 架构对比**

对比标准 Seq2Seq 和 Seq2Seq + Attention：
- 编码器输出的差异
- 解码器输入的差异
- 长序列处理的效果差异

**3. 注意力计算**

计算 Bahdanau 注意力机制的能量值：

$$e_{t,i} = v^T \tanh(W_1 h_t^{dec} + W_2 h_i^{enc})$$

分析这个计算的含义：为什么使用 tanh？为什么拼接两个隐藏状态？

**4. 编程实现**

实现一个简化版的机器翻译模型：
- 输入：英文数字词（"one", "two", "three", ...）
- 输出：中文数字词（"一", "二", "三", ...）
- 训练并测试翻译效果

---

## 参考资料

1. **Seq2Seq 原始论文**: "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)
2. **注意力机制论文**: "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015)
3. **Scheduled Sampling**: "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" (Bengio et al., 2015)
4. **束搜索**: "Beam Search Strategies for Neural Machine Translation" (Freitag & Al-Onaizan, 2017)