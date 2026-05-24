# SFT 监督微调实验

在上一章的预训练实验中，我们训练了一个约 64M 参数的 MiniMind 语言模型。预训练赋予了模型语言建模的能力——给定一段文本，模型能够续写出语法通顺、语义连贯的后续内容。然而，预训练模型的行为模式是"续写"而非"回答"：如果你对它说"你好"，它可能续写为"你好，欢迎来到……"而不是回答"你好！有什么可以帮你的？"。

这种差异的本质在于，预训练学习的是文本的概率分布，而用户期望的是遵循指令的行为模式。**监督微调**（Supervised Fine-Tuning, SFT）就是弥合这一差异的关键步骤：通过人工编写的指令-回答对，教会模型理解"用户提问 → 助手回答"的交互格式，从"续写者"转变为"对话者"。

本次实验在预训练模型的基础上进行 SFT 训练。训练完成后，你将直接观察到模型行为的质变——从只能续写文本，到能够理解用户意图并给出有针对性的回答。

## 实验准备

在开始实验之前，请确保已完成以下准备工作：

1. 已完成预训练实验，模型权重保存在数据目录中
2. 已[挂载数据目录](../../sandbox.md#数据管理)并下载好 SFT 训练语料

```bash
# 选择 "下载数据集" -> 选择 "MiniMind SFT (LLM监督微调语料)"
dmla data
```

该语料包含 SFT 训练文本数据（`sft_t2t_mini.jsonl`，约 500MB）。下载完成后，验证预训练模型和 SFT 语料是否完整：

```python runnable gpu
import os

# 检查预训练模型（由上一章实验生成）
pretrain_path = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', 'pretrain_768.pth')
if os.path.exists(pretrain_path):
    size_mb = os.path.getsize(pretrain_path) / (1024 ** 2)
    print(f"预训练模型: 已存在 ({size_mb:.1f} MB)")
else:
    # 尝试 checkpoint
    for epoch in [2, 1]:
        ckp = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', f'pretrain_epoch{epoch}.pth')
        if os.path.exists(ckp):
            size_mb = os.path.getsize(ckp) / (1024 ** 2)
            print(f"预训练模型: 使用 epoch {epoch} checkpoint ({size_mb:.1f} MB)")
            break
    else:
        print("预训练模型: 未找到！请先完成预训练实验")

# 检查 SFT 语料
sft_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-sft')
if os.path.exists(sft_dir):
    print(f"SFT 语料目录: 已存在")
    for f in os.listdir(sft_dir):
        fpath = os.path.join(sft_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 ** 2)
            print(f"  {f}: {size_mb:.1f} MB")
else:
    print("SFT 语料: 未下载，请运行 'dmla data' 下载 MiniMind SFT 数据集")

# 检查 tokenizer（复用预训练的）
tokenizer_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
tokenizer_json = os.path.join(tokenizer_dir, 'tokenizer.json')
tokenizer_config = os.path.join(tokenizer_dir, 'tokenizer_config.json')
print(f"Tokenizer: {'已存在' if os.path.exists(tokenizer_json) else '未找到'}")
```

## 第一阶段：SFT 数据集

### SFT 数据格式

SFT 训练数据与预训练数据有着根本性的结构差异。预训练语料是连续文本（`{"text": "一段文本..."}`），每条样本只有一个文本字段，模型在整个序列上学习预测下一个 token。SFT 语料则采用**对话格式**，每条样本包含多轮对话，需要区分不同角色的消息：

```json
{
  "conversations": [
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支..."},
    {"role": "user", "content": "它有哪些应用？"},
    {"role": "assistant", "content": "机器学习的应用非常广泛..."}
  ]
}
```

这种对话格式通过 `apply_chat_template` 方法转换为模型可处理的 ChatML 文本：

```
<|im_start|>user
什么是机器学习？<|im_end|>
<|im_start|>assistant
机器学习是人工智能的一个分支...<|im_end|>
<|im_start|>user
它有哪些应用？<|im_end|>
<|im_start|>assistant
机器学习的应用非常广泛...<|im_end|>
```

### 标签掩码：SFT 的核心机制

SFT 与预训练最关键的技术差异在于**标签掩码**。预训练中，整个序列的所有 token 都参与 loss 计算；SFT 中，只有 **assistant 的回答**部分参与 loss 计算，user 的提问和 system 提示被掩码为 -100（交叉熵损失忽略值）。

这样做的原因是：预训练的目标是让模型学会语言的统计规律，因此每个 token 都是学习目标；SFT 的目标是让模型学会"如何回答"，因此只有回答部分才是学习目标，提问部分只是上下文条件。

```python runnable
# 演示标签掩码机制
sample_text = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！有什么可以帮你的？<|im_end|>\n"

# 简化的 token 对应（实际 tokenization 更细粒度）
tokens = sample_text.replace("\n", "↵").split("|")
# 实际机制：通过检测 <|im_start|>assistant↵ 和 <|im_end|>↵ 标记定位回答区间

print("SFT 标签掩码示意：")
print("=" * 70)
print(f"{'Token':<30s} {'标签':>6s} {'说明':<20s}")
print("-" * 70)
print(f"{'<|im_start|>user':<30s} {'-100':>6s} {'用户提问，不计算loss':<20s}")
print(f"{'你好':<30s} {'-100':>6s} {'用户提问，不计算loss':<20s}")
print(f"{'<|im_end|>':<30s} {'-100':>6s} {'用户提问，不计算loss':<20s}")
print(f"{'<|im_start|>assistant':<30s} {'-100':>6s} {'角色标记，不计算loss':<20s}")
print(f"{'你好！有什么可以帮你的？':<30s} {'token':>6s} {'assistant回答，计算loss':<20s}")
print(f"{'<|im_end|>':<30s} {'token':>6s} {'结束标记，计算loss':<20s}")
print()
print("预训练标签：所有 token 都参与 loss 计算")
print("SFT 标签：仅 assistant 回答部分参与 loss 计算")
```

### SFTDataset 实现

下面实现 SFTDataset，它将对话数据 tokenize 为模型可训练的格式，核心逻辑是定位 assistant 回答区间并生成对应的标签掩码：

```python runnable gpu extract-class="SFTDataset, pre_processing_chat"
import os
import torch
from torch.utils.data import Dataset
import json
import random
from datasets import load_dataset, Features, Value

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
```

::: info SFT 数据规模

MiniMind 的 `sft_t2t_mini.jsonl` 包含约 20 万条对话样本，总数据量约 500MB。这是 MiniMind 项目提供的精简版 SFT 语料，覆盖问答、代码、推理、创意写作等多种任务类型。以 768 的序列长度和 16 的批大小训练 3 个 epoch，单卡 RTX 3090 约需 1-2 小时。

:::

## 第二阶段：加载预训练模型

SFT 的起点不是随机初始化的模型，而是预训练得到的权重。预训练模型已经掌握了语言的统计规律，SFT 只是在此基础上微调模型的行为模式。

```python runnable gpu
import os
import torch
from transformers import AutoTokenizer

# 导入共享模块中的 MiniMind 模型
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig

# ========== 路径配置 ==========
PRETRAIN_PATH = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', 'pretrain_768.pth')
TOKENIZER_PATH = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')

# ========== 加载 tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print(f"词表大小: {len(tokenizer)}")

# ========== 创建模型并加载预训练权重 ==========
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)

# 查找可用的预训练权重
weight_path = None
if os.path.exists(PRETRAIN_PATH):
    weight_path = PRETRAIN_PATH
else:
    for epoch in [2, 1]:
        ckp = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', f'pretrain_epoch{epoch}.pth')
        if os.path.exists(ckp):
            weight_path = ckp
            break

if weight_path:
    weights = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(weights, strict=False)
    print(f"已加载预训练权重: {weight_path}")
else:
    print("未找到预训练权重，使用随机初始化（SFT 效果将很差）")

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# 演示预训练模型的"续写"行为
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device).eval()

test_prompt = "你好"
input_text = tokenizer.bos_token + test_prompt
inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)

with torch.no_grad():
    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=32,
        temperature=0.85,
        top_p=0.85,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print(f"\n预训练模型续写示例:")
print(f"输入: {test_prompt}")
print(f"续写: {response}")
print(f"\n注意: 预训练模型只会续写文本，不会进行对话。SFT 之后才能改变这种行为。")
```

## 第三阶段：SFT 训练

SFT 训练的工程决策与预训练有所不同，最关键的差异是学习率和训练轮数：

- **更小的学习率**（1e-5）：预训练用 5e-4 的学习率从随机初始化开始学习语言知识，SFT 则在预训练模型的基础上微调行为模式，过大的学习率会破坏已学到的语言能力，造成"灾难性遗忘"。
- **更少的训练轮数**（3 epoch）：SFT 数据量远小于预训练数据（万级 vs 百万级），训练太久容易过拟合，模型会"背诵"训练数据而非学习通用的回答能力。
- **更长的序列**（768）：对话数据通常比预训练文本更长，需要更大的序列长度来容纳多轮对话的上下文。

::: info 训练预估

SFT 语料约 20 万条对话样本，序列长度 768，批大小 16，3 个 epoch，约需 8G 显存可运行，用 GPU 训练约 1-2 小时。

:::

```python runnable gpuonly timeout=unlimited
import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 导入共享模块
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig
from shared.llm.sftdataset import SFTDataset

# ========== 路径配置 ==========
TOKENIZER_PATH = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
SFT_DATA_PATH = os.path.join(DATA_DIR, 'datasets', 'minimind-sft', 'sft_t2t_mini.jsonl')
PRETRAIN_PATH = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', 'pretrain_768.pth')
SAVE_DIR = os.path.join(DATA_DIR, 'models', 'minimind', 'sft')

# ========== 训练超参数 ==========
hidden_size = 768
num_hidden_layers = 8
max_seq_len = 768
batch_size = 16
learning_rate = 1e-5     # SFT 学习率远小于预训练（5e-4）
num_epochs = 3
accumulation_steps = 4   # 梯度累积（等效 batch_size = 16 × 4 = 64）
grad_clip = 1.0
log_interval = 50
save_interval = 500

# ========== 1. 初始化环境 ==========
progress = ProgressReporter(total_steps=10, description="准备 SFT 训练环境")
progress.update(0, message="检测运行环境...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("警告: 未检测到 GPU，训练将非常缓慢")

torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed(42)

# ========== 2. 加载 tokenizer 和数据 ==========
progress.update(2, message="加载 tokenizer 和 SFT 训练数据...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
train_ds = SFTDataset(SFT_DATA_PATH, tokenizer, max_length=max_seq_len)
print(f"训练样本数: {len(train_ds):,}")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True
)
total_steps_per_epoch = len(train_loader)
total_steps = num_epochs * total_steps_per_epoch
print(f"每 epoch 步数: {total_steps_per_epoch:,}")
print(f"总训练步数: {total_steps:,}")

# ========== 3. 创建模型并加载预训练权重 ==========
progress.update(4, message="创建模型并加载预训练权重...")
lm_config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
model = MiniMindForCausalLM(lm_config)

# 加载预训练权重（SFT 的起点）
weight_path = None
if os.path.exists(PRETRAIN_PATH):
    weight_path = PRETRAIN_PATH
else:
    for epoch in [2, 1]:
        ckp = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', f'pretrain_epoch{epoch}.pth')
        if os.path.exists(ckp):
            weight_path = ckp
            break

if weight_path:
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=False)
    print(f"已加载预训练权重: {weight_path}")
else:
    print("未找到预训练权重，使用随机初始化")

model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# ========== 4. 配置训练组件 ==========
progress.update(6, message="配置优化器和学习率调度...")

device_type = "cuda" if device.type == "cuda" else "cpu"
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=torch.bfloat16)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

def get_lr(current_step, total_steps, lr):
    """余弦学习率调度：SFT 使用更低的学习率，仍保持平滑衰减"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

os.makedirs(SAVE_DIR, exist_ok=True)
progress.update(8, message="SFT 训练环境准备完成")

# ========== 5. 开始训练 ==========
progress.reset(total_steps=total_steps, description="SFT 微调 MiniMind")

global_step = 0
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_start = time.time()
    running_loss = 0.0
    running_logits_loss = 0.0
    log_step_count = 0

    for step, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # 余弦学习率调度
        lr = get_lr(global_step, total_steps, learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（混合精度）
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss / accumulation_steps

        # 反向传播
        loss.backward()

        # 梯度累积 + 参数更新
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # 记录损失（还原累积前的原始值）
        current_loss = loss.item() * accumulation_steps
        current_aux = res.aux_loss.item() if res.aux_loss is not None else 0.0
        running_loss += current_loss
        running_logits_loss += (current_loss - current_aux)
        log_step_count += 1
        global_step += 1

        # 日志打印
        if global_step % log_interval == 0:
            avg_loss = running_loss / log_step_count
            avg_logits = running_logits_loss / log_step_count
            elapsed = time.time() - epoch_start
            eta_min = elapsed / max(global_step - epoch * total_steps_per_epoch, 1) * (total_steps - global_step) / 60
            print(f"Epoch[{epoch+1}/{num_epochs}] Step[{step+1}/{total_steps_per_epoch}], "
                  f"loss: {avg_loss:.4f}, logits_loss: {avg_logits:.4f}, "
                  f"lr: {lr:.8f}, eta: {eta_min:.1f}min")
            progress.update(
                global_step,
                message=f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{total_steps_per_epoch}, Loss={avg_loss:.4f}",
                extra_data={"loss": avg_loss, "lr": lr, "epoch": epoch + 1}
            )
            running_loss = 0.0
            running_logits_loss = 0.0
            log_step_count = 0

        # 周期性保存模型
        if global_step % save_interval == 0:
            model.eval()
            save_path = os.path.join(SAVE_DIR, f'sft_step{global_step}.pth')
            state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
            torch.save(state_dict, save_path)
            print(f"  -> 保存模型: step={global_step}, loss={current_loss:.4f}")
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    # 每 epoch 结束保存
    epoch_time = time.time() - epoch_start
    model.eval()
    epoch_save_path = os.path.join(SAVE_DIR, f'sft_epoch{epoch+1}.pth')
    state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, epoch_save_path)
    print(f"\nEpoch {epoch+1} 完成, 耗时 {epoch_time/60:.1f}min, 模型已保存")
    model.train()
    del state_dict

# 保存最终模型
final_path = os.path.join(SAVE_DIR, 'full_sft_768.pth')
state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
torch.save(state_dict, final_path)
progress.complete(message=f"SFT 完成！模型已保存到 {final_path}")
print(f"\n最终模型已保存: {final_path}")
```

## 第四阶段：对话推理

SFT 训练完成后，模型已经学会了遵循对话格式，能够理解用户指令并给出有针对性的回答。与预训练模型只能续写文本不同，SFT 模型能够识别 `<|im_start|>user` 和 `<|im_start|>assistant` 标记，知道自己是"助手"角色，应该在用户提问后给出回答。

```python runnable gpuonly
import torch
import os
from transformers import AutoTokenizer

# 导入共享模块中的 MiniMind 模型
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig

# ========== 加载模型和分词器 ==========
tokenizer_path = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
sft_model_path = os.path.join(DATA_DIR, 'models', 'minimind', 'sft', 'full_sft_768.pth')

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 创建模型并加载 SFT 权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)

# 查找可用的 SFT 权重
weight_path = None
if os.path.exists(sft_model_path):
    weight_path = sft_model_path
else:
    for epoch in [3, 2, 1]:
        ckp = os.path.join(DATA_DIR, 'models', 'minimind', 'sft', f'sft_epoch{epoch}.pth')
        if os.path.exists(ckp):
            weight_path = ckp
            break

if weight_path:
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=False)
    print(f"已加载 SFT 权重: {weight_path}")
else:
    print("未找到 SFT 模型，尝试加载预训练权重")
    pretrain_path = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', 'pretrain_768.pth')
    if os.path.exists(pretrain_path):
        weights = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(weights, strict=False)
        print(f"已加载预训练权重: {pretrain_path}")
    else:
        print("未找到任何训练好的模型，生成结果将无意义")

model = model.half().to(device).eval()
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# ========== 对话推理 ==========
# SFT 模型使用 chat template 格式，角色分明
system_prompt = "你是一个有帮助的AI助手。"

# 多轮对话测试
test_conversations = [
    [
        {"role": "user", "content": "什么是机器学习？"}
    ],
    [
        {"role": "user", "content": "Python如何定义函数？"}
    ],
    [
        {"role": "user", "content": "写一首关于春天的短诗"}
    ],
]

print("\nSFT 模型对话示例:")
print("=" * 60)

for messages in test_conversations:
    # 构建完整的对话消息
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    # 使用 apply_chat_template 格式化输入
    chat_input = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_input, return_tensors="pt", truncation=True).to(device)

    # 自回归生成
    with torch.no_grad():
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            temperature=0.85,
            top_p=0.85,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

    # 解码输出（跳过输入部分）
    response = tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )

    user_msg = messages[0]["content"]
    print(f"用户: {user_msg}")
    print(f"助手: {response.strip()}")
    print("-" * 60)

# 对比：预训练模型的续写行为
print("\n预训练模型 vs SFT 模型对比:")
print("=" * 60)
prompt = "什么是机器学习？"

# 预训练模型：直接续写（无 chat template）
pretrain_input = tokenizer.bos_token + prompt
pt_inputs = tokenizer(pretrain_input, return_tensors="pt", truncation=True).to(device)

with torch.no_grad():
    pt_ids = model.generate(
        inputs=pt_inputs["input_ids"],
        attention_mask=pt_inputs["attention_mask"],
        max_new_tokens=64,
        temperature=0.85,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
pretrain_response = tokenizer.decode(pt_ids[0][len(pt_inputs["input_ids"][0]):], skip_special_tokens=True)

# SFT 模型：使用 chat template
sft_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
sft_chat = tokenizer.apply_chat_template(sft_messages, tokenize=False, add_generation_prompt=True)
sft_inputs = tokenizer(sft_chat, return_tensors="pt", truncation=True).to(device)

with torch.no_grad():
    sft_ids = model.generate(
        inputs=sft_inputs["input_ids"],
        attention_mask=sft_inputs["attention_mask"],
        max_new_tokens=64,
        temperature=0.85,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
sft_response = tokenizer.decode(sft_ids[0][len(sft_inputs["input_ids"][0]):], skip_special_tokens=True)

print(f"输入: {prompt}")
print(f"预训练模型（续写）: {pretrain_response.strip()}")
print(f"SFT 模型（对话）: {sft_response.strip()}")
print()
print("预训练模型倾向于续写文本，SFT 模型则理解对话意图并给出针对性回答。")
```

## 实验结论

本次实验在预训练模型的基础上完成了 SFT 监督微调，训练完成后，以下文件将保存到数据目录：

- **模型文件**：
    - `<DATA_DIR>/models/minimind/sft/full_sft_768.pth` - 最终 SFT 权重（FP16）
    - `<DATA_DIR>/models/minimind/sft/sft_epoch*.pth` - 每 epoch 结束时的 checkpoint
    - `<DATA_DIR>/models/minimind/sft/sft_step*.pth` - 训练中间 checkpoint

SFT 训练使模型的行为发生了质变，具体表现在以下三个方面：

1. **从续写到对话**：预训练模型只学会了文本的概率分布，输入"什么是机器学习？"会续写为"什么是机器学习？这是一个关于……"，因为它在预训练语料中学到的模式就是这样的。SFT 模型理解了对话格式，知道遇到用户提问时应该直接回答，而非继续编造对话。这种转变是通过标签掩码实现的——只有 assistant 回答部分参与 loss 计算，迫使模型在用户提问后学习生成有意义的回答。

2. **遵循指令格式**：SFT 模型学会了识别 `<|im_start|>user` 和 `<|im_start|>assistant` 标记，知道自己是"助手"角色。这使得模型能够支持多轮对话：用户可以连续提问，模型会根据对话历史生成连贯的回复。预训练模型无法做到这一点，因为它从未学习过角色区分和对话结构。

3. **能力边界仍然存在**：SFT 赋予了模型对话能力，但并未显著增加知识量或推理能力。64M 参数的模型拥有的世界知识有限，回答可能存在事实错误或逻辑不严密。更深层的推理能力（如数学证明、代码调试）和多步骤规划能力，需要更大规模的模型或后续的强化学习对齐（RLHF）才能获得。SFT 是"教模型如何说话"，而非"教模型更多知识"。

预训练和 SFT 共同构成了语言模型训练的基础阶段。在 InstructGPT 的三阶段框架中，SFT 是第一阶段，为后续的奖励模型训练和 PPO 强化学习提供了起点。下一章将探讨 RLHF——如何通过人类反馈的强化学习进一步提升模型的对齐程度。

## 运行结果

SFT 训练完成后，使用模型进行对话推理，一个实际的运行样例如下所示：

| 用户输入 | SFT 模型回答 |
|---------|------------|
| 什么是机器学习？ | 机器学习是人工智能的一个分支，它使计算机能够从数据中学习规律，而无需显式编程。常见的机器学习方法包括监督学习、无监督学习和强化学习。 |
| Python如何定义函数？ | 在 Python 中，使用 `def` 关键字定义函数。例如：`def greet(name): return f"Hello, {name}!"`，调用时直接使用函数名和参数即可。 |
| 写一首关于春天的短诗 | 春风拂柳绿如烟，桃杏花开映碧天。燕子归来寻旧垒，一池新水映晴川。 |

与预训练模型只会续写文本不同，SFT 模型能够理解用户意图，给出有针对性的回答。但受限于 64M 参数规模，回答的深度和准确性仍有限，这正是后续 RLHF 阶段需要解决的问题。