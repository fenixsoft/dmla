# VLMDataset 定义
# 从文档自动提取生成

import io
import json
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from shared.vlm.vlmconfig import MiniMindVLM
from torch.utils.data import Dataset

class VLMDataset(Dataset):
    """视觉语言模型数据集：从 Parquet 文件加载图像-对话对"""
    def __init__(self, parquet_path, tokenizer, preprocess=None,
                 max_length=512, image_special_token='<|image_pad|>', image_token_len=64):
        super().__init__()
        self.table = pa.Table.from_batches(pq.ParquetFile(parquet_path).iter_batches())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    def create_chat_prompt(self, conversations):
        """将对话列表转换为模型的输入文本，<image> 标记替换为视觉特殊标记"""
        text = ""
        for turn in conversations:
            content = turn['content'].replace('<image>', self.image_special_token) \
                if turn.get('role') != 'system' else turn['content']
            text += f"{self.tokenizer.bos_token}{turn['role']}\n{content}{self.tokenizer.eos_token}\n"
        return text

    def generate_labels(self, input_ids):
        """生成训练标签：仅计算 assistant 回复部分的损失"""
        labels = [-100] * len(input_ids)
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
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        conversations = json.loads(self.table['conversations'][index].as_py())
        image_bytes = self.table['image_bytes'][index].as_py()
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]

        prompt = self.create_chat_prompt(conversations)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        image_inputs_list = [
            MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess)
            for img in image_bytes
        ]
        if hasattr(image_inputs_list[0], 'keys'):
            image_data = {k: torch.cat([inp[k] for inp in image_inputs_list], dim=0)
                         for k in image_inputs_list[0].keys()}
        else:
            image_data = torch.stack(image_inputs_list)

        return torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(labels, dtype=torch.long), image_data
