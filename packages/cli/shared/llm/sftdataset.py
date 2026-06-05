# SFTDataset, pre_processing_chat 定义
# 从文档自动提取生成

import json
import os
import random
import torch
from datasets import load_dataset, Features, Value
from datasets import logging as datasets_logging
from torch.utils.data import Dataset

class SFTDataset(Dataset):
    """
    SFT 数据集：将对话数据 tokenize 为 ChatML 格式

    与 PretrainDataset 的主要差异：
    - 数据格式从 {"text": "..."} 变为 {"conversations": [...]}
    - 标签掩码：仅 assistant 回答部分参与 loss，其余标记为 -100（PyTorch CrossEntropyLoss 默认忽略 -100 对应的位置）
    - 使用 apply_chat_template 将对话转为 ChatML 格式
    - SFT 数据集仍然支持工具调用训练，只要将训练集从 sft_t2t_tiny.jsonl 换回带有工具调用样例的 sft_t2t_mini.jsonl 即可
    """
    # 使用 ChatML 格式：<|im_start|>role\ncontent<|im_end|>\n
    # tokenizer 本身未内置 chat_template，需手动设置
    CHATML_TEMPLATE = (
        "{% for message in messages %}<|im_start|>{{ message.role }}\n"
        "{{ message.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

    def __init__(self, jsonl_path, tokenizer, max_length=768):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = tokenizer
        # Tokenizer 未内置 chat_template，需手动设置 ChatML 格式
        if not tokenizer.chat_template:
            tokenizer.chat_template = self.CHATML_TEMPLATE
        self.max_length = max_length
        features = Features({
            'conversations': [{'role': Value('string'), 'content': Value('string'),
                              'reasoning_content': Value('string'), 'tools': Value('string'),
                              'tool_calls': Value('string')}]
        })
        # 抑制 load_dataset 的 "Generating train split" 进度输出
        datasets_logging.set_verbosity_error()
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        datasets_logging.set_verbosity_warning()
        # 预计算 assistant 回答的起止 token ID
        # 即 <|im_start|>assistant\n 对应的 token ID 序列，用于定位助手回答的起始位置
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids  
        # 即 <|im_end|>\n 对应的 token ID 序列，用于定位助手回答的结束位置
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
        # 右侧填充至固定长度，填充部分的标签已由 generate_labels 设为 -100，不参与 loss
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
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
    ]
    # 概率性添加 system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
