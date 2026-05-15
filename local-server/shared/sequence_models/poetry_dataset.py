# PoetryDataset 类定义
# 从文档自动提取生成

import json
import os
import re
from collections import Counter, defaultdict, deque
from torch.utils.data import Dataset, DataLoader

class PoetryDataset:
    """古诗词数据集（字符级语言模型）

    从 chinese-poetry 数据集加载诗词，构建字符级词汇表，
    将诗词文本转换为数值序列用于 LSTM 训练。
    """
    def __init__(self, data_dir, min_length=10, max_length=100, vocab_size=4000):
        self.min_length = min_length
        self.max_length = max_length
        self.vocab_size = vocab_size

        # 加载诗词文本
        self.poems = self._load_poems(data_dir)
        print(f"加载完成: {len(self.poems)} 首诗词")

        # 构建词汇表
        self.char2idx, self.idx2char = self._build_vocab()
        print(f"词汇表大小: {len(self.char2idx)}")

        # 将诗词转换为序列
        self.sequences = self._encode_poems()
        print(f"有效序列数: {len(self.sequences)}")

    def _load_poems(self, data_dir):
        """加载诗词数据"""
        poems = []

        # 定义要加载的数据集
        datasets = ['全唐诗', '宋词', '诗经', '楚辞']

        for dataset in datasets:
            dataset_path = os.path.join(data_dir, dataset)
            if not os.path.exists(dataset_path):
                continue

            json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]

            for jf in json_files:
                file_path = os.path.join(dataset_path, jf)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    for poem in data:
                        # 提取诗词正文
                        text = self._extract_text(poem)
                        if text and self._is_valid(text):
                            poems.append(text)
                except Exception as e:
                    print(f"加载 {jf} 失败: {e}")

        return poems

    def _extract_text(self, poem):
        """从诗词数据中提取正文"""
        # 尝试不同的字段名
        if 'text' in poem:
            text = poem['text']
        elif 'paragraphs' in poem:
            text = ''.join(poem['paragraphs'])
        elif 'content' in poem:
            # content 可能是字符串或列表
            content = poem['content']
            if isinstance(content, list):
                text = ''.join(content)
            else:
                text = content
        else:
            return None

        # 清理文本：去除标点符号，只保留汉字
        # 保留常用标点用于断句
        text = re.sub(r'[^一-龥，。！？、；：""''（）]', '', text)

        return text

    def _is_valid(self, text):
        """检查文本是否有效"""
        # 长度检查
        if len(text) < self.min_length or len(text) > self.max_length:
            return False

        # 过滤包含缺字标记的诗句
        if '□' in text or '■' in text:
            return False

        return True

    def _build_vocab(self):
        """构建字符级词汇表"""
        # 统计字符频率
        char_counter = Counter()
        for poem in self.poems:
            char_counter.update(poem)

        # 选择高频字符
        most_common = char_counter.most_common(self.vocab_size - 2)  # 预留两个位置给特殊标记

        # 构建映射
        char2idx = {'<PAD>': 0, '<UNK>': 1}
        for i, (char, _) in enumerate(most_common, start=2):
            char2idx[char] = i

        idx2char = {idx: char for char, idx in char2idx.items()}

        return char2idx, idx2char

    def _encode_poems(self):
        """将诗词转换为数值序列"""
        sequences = []
        for poem in self.poems:
            seq = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in poem]
            sequences.append(seq)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # 输入序列：去掉最后一个字符
        # 目标序列：去掉第一个字符
        return seq[:-1], seq[1:]
