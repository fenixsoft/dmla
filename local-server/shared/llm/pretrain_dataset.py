# PretrainDataset 定义
# 从文档自动提取生成

import json
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    """
    预训练数据集：从 JSONL 文件加载文本，tokenize 为 next-token prediction 格式
    
    每条样本格式：{"text": "一段文本"}
    输出：(input_ids, labels)，其中 labels 是 input_ids 的右移一位版本，
    用于计算 next-token prediction 的交叉熵损失
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 预读取所有样本的文本
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'text' in data and data['text'].strip():
                        self.samples.append(data['text'])
                except json.JSONDecodeError:
                    continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        text = self.samples[index]
        # tokenize：截断到 max_length - 2（预留 BOS 和 EOS 的位置）
        tokens = self.tokenizer(
            str(text), 
            add_special_tokens=False, 
            max_length=self.max_length - 2, 
            truncation=True
        ).input_ids
        
        # 添加 BOS 和 EOS 标记
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        
        # 填充到固定长度
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # 标签与输入相同，填充位置标记为 -100（交叉熵损失忽略）
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return input_ids, labels
