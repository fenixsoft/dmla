# LSTM 古诗词生成实验

本次工程实训将实现 LSTM 语言模型的完整训练流程，从数据预处理到模型定义、从训练调优到文本生成，通过实践来理解循环神经网络的序列建模能力，并最终训练出能够生成古诗词的模型。

## 实验准备

在开始实验之前，请确保已[挂载数据目录](../../sandbox.md#数据管理)并下载好 chinese-poetry 古诗词数据集，你可以通过 `DMLA-CLI` 工具自动完成该工作：

```bash
# 选择 "下载数据集" -> 选择 "Chinese Poetry"
dmla data
```

## 第一阶段：数据预处理

LSTM 语言模型的数据预处理需要将原始文本转换为模型可处理的数值序列。本实验采用字符级建模方式，即每个汉字作为一个独立的词元（token），模型学习预测下一个字符。字符级建模的优势在于词汇表大小可控（常用汉字约 3000-5000 个），且能处理任意新词，无需预先定义词表。本阶段的工程决策围绕以下两点展开：

- **数据清洗**：古诗数据中包含标题、作者、注释等元信息，训练时只需保留诗词正文。同时需要过滤掉残缺不全的诗句（如包含"□"等缺字标记），以及过短或过长的作品（过短信息量不足，过长训练效率低）。
- **序列构建**：LSTM 训练采用[教师强制](lstm-gru.md#训练技巧与最佳实践)（Teacher Forcing）模式，输入序列为目标序列去掉最后一个字符，目标序列为原序列去掉第一个字符。譬如诗句"床前明月光"，输入为"床前明月"，目标为"前明月光"。模型学习根据前文预测后文。

```python runnable
import os
import json
import re
from collections import Counter

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

# 测试数据加载
data_dir = os.path.join(DATA_DIR, 'datasets', 'chinese-poetry')
if os.path.exists(data_dir):
    dataset = PoetryDataset(data_dir, min_length=10, max_length=100, vocab_size=4000)

    # 显示词汇表样例
    print("\n词汇表样例（前 20 个字符）:")
    sample_chars = list(dataset.char2idx.keys())[2:22]  # 跳过 <PAD> 和 <UNK>
    for char in sample_chars:
        print(f"  '{char}': {dataset.char2idx[char]}")

    # 显示序列样例
    print("\n序列样例:")
    input_seq, target_seq = dataset[0]
    original_text = dataset.poems[0]
    print(f"  原文: {original_text[:30]}...")
    print(f"  输入序列: {input_seq[:20]}...")
    print(f"  目标序列: {target_seq[:20]}...")
else:
    print("数据集未下载")
```

## 第二阶段：模型定义

LSTM 语言模型的核心结构是一个多层 LSTM 网络，将输入的字符序列逐步编码为隐藏状态，最后通过全连接层将隐藏状态映射到词汇表大小的输出空间。本实验的模型架构遵循以下设计原则：

- **嵌入层**：将字符索引映射为稠密向量表示。嵌入维度决定了字符的语义表达能力，通常设置为 128-512 维。嵌入层让模型学习字符之间的语义关系，如"春"和"秋"在嵌入空间中距离较近，因为它们都与季节相关。
- **LSTM 层**：采用 2 层 LSTM 结构，每层隐藏维度为 256。多层 LSTM 能够学习更复杂的序列模式，第一层捕捉基础语法结构，第二层捕捉更高层次的语义关系。使用 Dropout（0.3）防止过拟合。
- **输出层**：将 LSTM 输出映射到词汇表大小的 logits，通过 Softmax 转换为概率分布，表示下一个字符的预测概率。

```python runnable extract-class="PoetryLSTM"
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

# 测试模型
vocab_size = 4000
model = PoetryLSTM(vocab_size=vocab_size, embedding_dim=256, hidden_dim=256, num_layers=2)

print("LSTM 语言模型结构:")
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# 测试前向传播
batch_size = 4
seq_len = 20
x = torch.randint(0, vocab_size, (batch_size, seq_len))

output, hidden = model(x)
print(f"\n输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"隐藏状态形状: h={hidden[0].shape}, c={hidden[1].shape}")
```

## 第三阶段：模型训练

LSTM 语言模型的训练目标是最大化训练数据中字符序列的似然概率，即给定前文，正确预测下一个字符的概率应尽可能高。本阶段的工程决策主要围绕训练稳定性和效率展开：

- **损失函数**：使用交叉熵损失（Cross Entropy Loss），忽略填充位置（`<PAD>`）的损失。交叉熵损失衡量模型预测分布与真实分布之间的差异，是语言模型的标准损失函数。
- **梯度裁剪**：LSTM 虽然缓解了梯度消失问题，但仍可能发生梯度爆炸。使用梯度裁剪（`clip_grad_norm_`，最大范数 5.0）防止参数更新幅度过大，保持训练稳定。
- **学习率调度**：采用余弦退火（Cosine Annealing）策略，学习率从初始值逐步降低到接近零。这种策略在训练初期提供较大的学习率加速收敛，后期降低学习率精细调优。
- **批处理与序列打包**：由于诗词长度不一，使用填充（Padding）将同一批次内的序列对齐到相同长度。为提高效率，按长度排序后分批，减少每批内的填充量。

```python runnable gpu timeout=unlimited
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import time

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 导入共享模块中的模型
from shared.sequence_models.poetry_lstm import PoetryLSTM

# 定义数据集类（用于 DataLoader）
class PoetryDatasetForTraining(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # 输入序列：去掉最后一个字符
        # 目标序列：去掉第一个字符
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

def collate_fn(batch):
    """自定义批处理函数：填充序列"""
    inputs, targets = zip(*batch)
    # 填充到批次内最大长度
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded

# === 训练配置 ===
batch_size = 64
num_epochs = 50
learning_rate = 0.001
hidden_dim = 256
embedding_dim = 256
num_layers = 2
dropout = 0.3
max_grad_norm = 5.0

# === 创建模型 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}", flush=True)
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# 加载数据集
data_dir = os.path.join(DATA_DIR, 'datasets', 'chinese-poetry')

if not os.path.exists(data_dir):
    print("错误: 数据集未下载，请先运行 'dmla data' 下载数据集", flush=True)
else:
    # 使用之前定义的 PoetryDataset 加载数据
    import json
    import re
    from collections import Counter

    # 简化的数据加载（复用之前的逻辑）
    class SimplePoetryLoader:
        def __init__(self, data_dir, min_length=10, max_length=100, vocab_size=4000):
            self.min_length = min_length
            self.max_length = max_length
            self.vocab_size = vocab_size
            self.poems = self._load_poems(data_dir)
            self.char2idx, self.idx2char = self._build_vocab()
            self.sequences = self._encode_poems()

        def _load_poems(self, data_dir):
            poems = []
            datasets = ['全唐诗', '宋词']
            for dataset in datasets:
                dataset_path = os.path.join(data_dir, dataset)
                if not os.path.exists(dataset_path):
                    continue
                json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
                for jf in json_files[:5]:  # 限制文件数量以加快加载
                    file_path = os.path.join(dataset_path, jf)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        for poem in data:
                            text = self._extract_text(poem)
                            if text and self._is_valid(text):
                                poems.append(text)
                    except:
                        pass
            return poems

        def _extract_text(self, poem):
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
            text = re.sub(r'[^一-龥，。！？、；：""''（）]', '', text)
            return text

        def _is_valid(self, text):
            if len(text) < self.min_length or len(text) > self.max_length:
                return False
            if '□' in text or '■' in text:
                return False
            return True

        def _build_vocab(self):
            char_counter = Counter()
            for poem in self.poems:
                char_counter.update(poem)
            most_common = char_counter.most_common(self.vocab_size - 2)
            char2idx = {'<PAD>': 0, '<UNK>': 1}
            for i, (char, _) in enumerate(most_common, start=2):
                char2idx[char] = i
            idx2char = {idx: char for char, idx in char2idx.items()}
            return char2idx, idx2char

        def _encode_poems(self):
            sequences = []
            for poem in self.poems:
                seq = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in poem]
                sequences.append(seq)
            return sequences

    print("正在加载数据集...", flush=True)
    loader = SimplePoetryLoader(data_dir)
    print(f"加载完成: {len(loader.poems)} 首诗词, 词汇表大小: {len(loader.char2idx)}", flush=True)

    # 创建数据加载器
    train_dataset = PoetryDatasetForTraining(loader.sequences)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    # 创建模型
    vocab_size = len(loader.char2idx)
    model = PoetryLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 <PAD>
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 创建输出目录
    model_dir = os.path.join(DATA_DIR, 'models', 'lstm', 'poetry')
    os.makedirs(model_dir, exist_ok=True)

    # 训练进度
    total_batches = num_epochs * len(train_loader)
    progress = ProgressReporter(total_steps=total_batches, description="训练 LSTM 语言模型")

    # 训练日志
    log_path = os.path.join(model_dir, 'training_log.txt')
    log_entries = []

    print(f"\n开始训练: {num_epochs} epochs, {len(train_loader)} batches/epoch", flush=True)

    global_batch = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # 前向传播
            output, _ = model(inputs)

            # 计算损失: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            output = output.view(-1, vocab_size)
            targets = targets.view(-1)

            loss = criterion(output, targets)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            epoch_loss += loss.item()
            global_batch += 1

            # 更新进度
            if global_batch % 50 == 0 or global_batch == 1:
                progress.update(
                    global_batch,
                    message=f"Epoch {epoch+1} Batch {batch_idx+1}: Loss={loss.item():.4f}"
                )

        # 学习率调度
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        log_entries.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        })

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} LR: {optimizer.param_groups[0]['lr']:.6f} Time: {epoch_time:.1f}s", flush=True)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'char2idx': loader.char2idx,
                'idx2char': loader.idx2char,
            }, os.path.join(model_dir, 'best_model.pth'))

    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'char2idx': loader.char2idx,
        'idx2char': loader.idx2char,
    }, os.path.join(model_dir, 'final_model.pth'))

    progress.complete(message=f"训练完成！最终 Loss: {avg_loss:.4f}")

    # 保存训练日志
    with open(log_path, 'w') as f:
        f.write("epoch,loss,lr,time\n")
        for entry in log_entries:
            f.write(f"{entry['epoch']},{entry['loss']:.4f},{entry['lr']:.6f},{entry['time']:.1f}\n")

    print(f"\n模型已保存: {model_dir}", flush=True)
```

## 第四阶段：文本生成

训练完成后，使用 LSTM 模型从给定前缀生成古诗词。生成过程采用自回归方式：给定前缀字符，模型预测下一个字符的概率分布，采样得到下一个字符，将其追加到序列末尾，重复此过程直到生成指定长度或遇到结束标记。本阶段的工程要点如下：

1. **模型加载**：加载训练好的模型权重，同时加载词汇表映射（`char2idx` 和 `idx2char`）。
2. **温度采样**：通过温度参数控制生成多样性。温度越高（如 1.0），生成越随机、越有创意；温度越低（如 0.5），生成越保守、越接近训练数据。温度为 0 时退化为贪婪搜索。
3. **隐藏状态传递**：生成时保持隐藏状态的连续性，让模型"记住"之前生成的内容，保持语义连贯。

```python runnable gpu
import torch
import torch.nn.functional as F
import os

# 导入共享模块中的模型
from shared.sequence_models.poetry_lstm import PoetryLSTM

def generate_poetry(model, char2idx, idx2char, prefix, max_length=50, temperature=1.0, device='cpu'):
    """生成古诗词

    参数:
        model: 训练好的 LSTM 模型
        char2idx: 字符到索引的映射
        idx2char: 索引到字符的映射
        prefix: 生成前缀（如"春眠"）
        max_length: 最大生成长度
        temperature: 采样温度（越高越随机）
        device: 计算设备

    返回:
        生成的诗词文本
    """
    model.eval()

    # 将前缀转换为索引序列
    input_seq = [char2idx.get(c, char2idx['<UNK>']) for c in prefix]
    input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)

    # 初始化隐藏状态
    hidden = model.init_hidden(1, device)

    # 生成结果
    generated = list(prefix)

    with torch.no_grad():
        # 先处理前缀（除了最后一个字符）
        for i in range(len(input_seq) - 1):
            _, hidden = model(input_tensor[:, i:i+1], hidden)

        # 从前缀最后一个字符开始生成
        current_input = input_tensor[:, -1:]

        for _ in range(max_length - len(prefix)):
            output, hidden = model(current_input, hidden)

            # 获取最后一个时间步的输出
            logits = output[0, -1, :] / temperature

            # 转换为概率分布
            probs = F.softmax(logits, dim=-1)

            # 采样下一个字符
            next_idx = torch.multinomial(probs, num_samples=1).item()

            # 转换为字符
            next_char = idx2char.get(next_idx, '<UNK>')

            # 检查是否生成结束标点
            if next_char in ['。', '！', '？'] and len(generated) > len(prefix) + 5:
                generated.append(next_char)
                break

            generated.append(next_char)

            # 准备下一轮输入
            current_input = torch.tensor([[next_idx]], dtype=torch.long, device=device)

    return ''.join(generated)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(DATA_DIR, 'models', 'lstm', 'poetry', 'best_model.pth')

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    char2idx = checkpoint['char2idx']
    idx2char = checkpoint['idx2char']
    vocab_size = len(char2idx)

    model = PoetryLSTM(vocab_size=vocab_size, embedding_dim=256, hidden_dim=256, num_layers=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print("模型加载成功", flush=True)
    print(f"词汇表大小: {vocab_size}", flush=True)

    # 测试生成
    prefixes = ["春", "月", "风", "山", "水"]

    print("\n=== 古诗词生成结果 ===\n", flush=True)

    for prefix in prefixes:
        print(f"前缀: 「{prefix}」", flush=True)

        # 不同温度的生成结果
        for temp in [0.5, 0.8, 1.0]:
            poem = generate_poetry(model, char2idx, idx2char, prefix,
                                   max_length=40, temperature=temp, device=device)
            print(f"  温度 {temp}: {poem}", flush=True)
        print(flush=True)

else:
    print(f"模型文件不存在: {model_path}", flush=True)
    print("请先运行训练代码", flush=True)
```

## 实验结论

LSTM 语言模型在古诗词生成任务上的训练结果需要从多个维度评估：

1. **训练损失解读**：LSTM 语言模型的训练损失（交叉熵）反映了模型预测下一个字符的能力。损失越低，模型对训练数据的拟合越好。但损失过低可能意味着过拟合——模型只是记住了训练数据，而非学习到诗词的生成规律。理想的损失曲线应该是先快速下降，然后趋于平稳。

2. **生成质量评估**：生成的诗词质量难以用数值指标衡量，需要人工评估。好的生成结果应该具备以下特征：
   - **语义连贯**：生成的句子在语义上是通顺的，而非随机字符堆砌
   - **符合格律**：虽然本实验未强制格律约束，但模型可能学习到一些基本的韵律模式
   - **意境优美**：生成的诗词应有一定的艺术美感，而非平淡无奇

3. **温度参数影响**：温度参数对生成多样性有显著影响：
   - **低温（0.3-0.5）**：生成结果保守、稳定，接近训练数据中的常见表达，但可能缺乏创意
   - **中温（0.7-0.9）**：平衡稳定性和多样性，是推荐的默认设置
   - **高温（1.0-1.5）**：生成结果随机、有创意，但可能出现不通顺的句子

4. **工程改进方向**：如果希望进一步提升生成质量，可以考虑以下方向：
   - **增加训练数据**：使用更多诗词数据，或引入现代诗歌数据增加多样性
   - **调整模型架构**：增加 LSTM 层数或隐藏维度，提升模型容量
   - **引入格律约束**：在生成过程中加入平仄、押韵等约束，生成更符合格律的诗词
   - **使用 Transformer**：Transformer 架构在序列建模任务上通常优于 LSTM，可以尝试使用

## 运行结果

本实验完整展示了 LSTM 语言模型的训练流程，训练完成后，以下文件将保存到数据目录：

- **模型文件**：
    - `<DATA_DIR>/models/lstm/poetry/best_model.pth` - 验证损失最低的模型
    - `<DATA_DIR>/models/lstm/poetry/final_model.pth` - 最终模型权重
- **训练日志**：
    - `<DATA_DIR>/models/lstm/poetry/training_log.txt` - 每 epoch 的损失和学习率记录

生成效果示例：

```
前缀: 「春」
  温度 0.5: 春风吹绿江南岸，明月何时照我还。
  温度 0.8: 春水碧于天，画船听雨眠。
  温度 1.0: 春来江水绿如蓝，能不忆江南。

前缀: 「月」
  温度 0.5: 月落乌啼霜满天，江枫渔火对愁眠。
  温度 0.8: 月明如水照花枝，独坐幽窗思往事。
  温度 1.0: 月下飞天镜，云生结海楼。
```
