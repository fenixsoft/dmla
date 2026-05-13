# PoetryLSTM 类定义
# 从文档自动提取生成

import torch
import torch.nn as nn

class PoetryLSTM(nn.Module):
    """LSTM 语言模型（用于古诗词生成）

    架构: Embedding -> LSTM -> Linear -> Softmax
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=2, dropout=0.3):
        super(PoetryLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：字符索引 -> 稠密向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层：隐藏状态 -> 词汇表概率分布
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        参数:
            x: 输入序列 (batch_size, seq_len)
            hidden: 初始隐藏状态 (可选)

        返回:
            output: 输出 logits (batch_size, seq_len, vocab_size)
            hidden: 最终隐藏状态
        """
        # 嵌入: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # LSTM: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim)
        lstm_out, hidden = self.lstm(embedded, hidden)

        # 输出: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, vocab_size)
        output = self.fc(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)
