# SFTDataset, pre_processing_chat 定义
# 从文档自动提取生成

import json
import os
import random
import torch
from datasets import load_dataset, Features, Value
from torch.utils.data import Dataset

class SFTDataset(Dataset):
    """
    SFT 数据集：将对话数据 tokenize 为 next-token prediction 格式

    与 PretrainDataset 的核心差异：
    - 数据格式从 {"text": "..."} 变为 {"conversations": [...]}
    - 标签掩码：仅 assistant 回答部分参与 loss，其余标记为 -100
    - 使用 apply_chat_template 将对话转为 ChatML 格式
    """
    def __init__(self, jsonl_path, tokenizer, max_length=768):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({
            'conversations': [{'role': Value('string'), 'content': Value('string'),
                              'reasoning_content': Value('string'), 'tools': Value('string'),
                              'tool_calls': Value('string')}]
        })
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # 预计算 assistant 回答的起止标记 ID
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """将对话列表应用 chat template 转为文本"""
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )

    def generate_labels(self, input_ids):
        """生成标签：assistant 回答部分保留原始 ID，其余设为 -100"""
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 检测 <|im_start|>assistant\n 的位置
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 查找对应的 <|im_end|>\n
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 标记回答区间（包含 eos）
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 填充到固定长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """预处理对话数据：概率性添加系统提示词"""
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations):
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加 system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
