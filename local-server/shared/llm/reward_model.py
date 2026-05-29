# RewardModel 定义
# 从文档自动提取生成

import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    简化的奖励模型实现

    核心结构：Transformer 编码器提取语义特征 → 奖励头映射为标量评分

    参数:
        vocab_size : 词汇表大小
        d_model : 嵌入维度
        nhead : 注意力头数
        num_layers : Transformer 层数
    """
    def __init__(self, vocab_size=1000, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 奖励头：将语义特征映射为标量奖励值
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        """
        输入: input_ids (batch, seq_len) — 指令+回答的 token 序列
        输出: reward (batch,) — 标量奖励分数

        核心步骤：
        1. 嵌入 + 位置编码（对应理论中的输入表示）
        2. Transformer 编码（对应理论中的语义特征提取）
        3. 取最后 token 隐藏状态 → 线性层映射（对应理论中的奖励评分）
        """
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)

        # 取最后一个 token 的隐藏状态
        last_hidden = x[:, -1, :]  # (batch, d_model)
        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return reward
