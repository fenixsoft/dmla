---
title: "大语言模型训练：从 Transformer 到推理模型"
date: 2026-05-12
tags:
  - LLM
  - Transformer
  - NLP
  - Alignment
  - Reasoning
series:
  name: 大语言模型训练
  order: 1
---

# 大语言模型训练：从 Transformer 到推理模型

本系列承接"深度学习经典模型（2006-2017）"系列，从 Transformer 架构（2017）这一关键奇点出发，系统讲解大语言模型训练的完整流程。读者已具备 RNN/LSTM/GRU/Seq2Seq 和 Bahdanau 注意力机制的知识基础，系列将自然过渡到 Transformer 及后续所有 LLM 训练技术。

## 与前系列的知识衔接

```
已有基础 (deep-learning-series)
──────────────────
RNN → LSTM → GRU → Seq2Seq → Bahdanau Attention
                                     ↓
═══════════════════════════════════════════════
本系列起点：Transformer — 语言模型的奇点
═══════════════════════════════════════════════
                                     ↓
架构 → 预训练 → 对齐 → 推理 → 前沿
```

## 目录

### 第一部分：架构基础

- [第 1 章：Transformer 架构——语言模型的奇点](./architecture-basics/transformer-architecture.md)
- [第 2 章：架构演进与变体——从固定窗口到无限上下文](./architecture-basics/architecture-evolution.md)

### 第二部分：预训练

- [第 3 章：语言模型与分词——从 N-gram 到子词分割](./pretraining/language-model-tokenization.md)
- [第 4 章：预训练数据工程——海量文本的炼金术](./pretraining/pretraining-data.md)
- [第 5 章：缩放定律——越大越好的数学原理](./pretraining/scaling-laws.md)
- [第 6 章：分布式训练基础设施——万卡并行的工程挑战](./pretraining/distributed-training.md)

### 第三部分：对齐训练

- [第 7 章：监督微调——从基础模型到可用模型](./alignment/supervised-finetuning.md)
- [第 8 章：RLHF——人类反馈强化学习](./alignment/rlhf.md)
- [第 9 章：对齐新范式——DPO、KTO 与 GRPO](./alignment/alignment-new-paradigms.md)

### 第四部分：推理能力

- [第 10 章：思维链与推理模型——让模型学会思考](./reasoning/chain-of-thought.md)
- [第 11 章：Test-Time Compute Scaling——推理算力扩展](./reasoning/test-time-compute.md)

### 第五部分：前沿与融合

- [第 12 章：多模态大模型——超越纯文本](./frontier/multimodal-llm.md)
- [第 13 章：模型评估与安全——如何衡量和守护大模型](./frontier/evaluation-safety.md)

## 目标读者

- 具备深度学习基础（神经网络结构、反向传播、优化器）
- 了解序列模型（RNN、LSTM/GRU、Seq2Seq、Bahdanau 注意力）
- 了解生成模型（VAE、GAN）
- 具备数学基础（线性代数、微积分、概率统计）

## 里程碑时间线

| 年份 | 里程碑 | 对应章节 |
|------|---------|---------|
| 2015 | Bahdanau 注意力机制 | 第1章（衔接点） |
| 2017 | Transformer 架构 | 第1章 |
| 2018 | GPT-1、BERT | 第1章 |
| 2019 | GPT-2（1.5B） | 第3章 |
| 2020 | GPT-3（175B）、Kaplan Scaling Laws | 第5章 |
| 2021 | Switch Transformer（MoE） | 第2章 |
| 2022 | InstructGPT/ChatGPT、Chinchilla | 第8章、第5章 |
| 2023 | GPT-4、LLaMA、多模态融合 | 第12章、第4章 |
| 2024 | Mixtral、DeepSeek-V2、o1 | 第2章、第10章 |
| 2025 | DeepSeek-R1、o3、GRPO | 第9章、第10章 |
| 2026 | DeepSeek-V4（CSA/HCA） | 第2章 |

## 系列统计

- **总章节**: 13 章
- **总字数**: 约 11 万字
- **实验代码**: 40+ 可运行代码块
- **架构图**: 30+ nn-arch/mermaid 图表