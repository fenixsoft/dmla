# DPODataset 定义
# 从文档自动提取生成

import json
import os
import torch
from datasets import load_dataset, Features, Value
from datasets import logging as datasets_logging
from torch.utils.data import Dataset

class DPODataset(Dataset):
    """
    DPO 数据集：将偏好对比数据 tokenize 为模型可训练的格式

    每条样本格式：{"chosen": [{role, content}, ...], "rejected": [{role, content}, ...]}
    输出 chosen 和 rejected 的 input_ids、目标 ids 和 loss_mask
    loss_mask 仅在 assistant 回答部分为 1，其余为 0
    """
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
        if not tokenizer.chat_template:
            tokenizer.chat_template = self.CHATML_TEMPLATE
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 定位 assistant 回答的起止 token ID
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        features = Features({
            'chosen': [{'role': Value('string'), 'content': Value('string')}],
            'rejected': [{'role': Value('string'), 'content': Value('string')}]
        })
        datasets_logging.set_verbosity_error()
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        datasets_logging.set_verbosity_warning()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']
        rejected = sample['rejected']

        # 将对话转为 ChatML 格式文本
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        # Tokenize 并填充到固定长度
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # DPO 采用 next-token prediction 的输入-目标对齐方式
        # x 为输入序列（去掉最后一个 token），y 为目标序列（去掉第一个 token）
        # mask 对齐 y 的位置，用于在 DPO loss 中只计算 assistant 回答部分
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen, 'y_chosen': y_chosen, 'mask_chosen': mask_chosen,
            'x_rejected': x_rejected, 'y_rejected': y_rejected, 'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """生成 loss 掩码：仅在 assistant 回答部分为 1"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
