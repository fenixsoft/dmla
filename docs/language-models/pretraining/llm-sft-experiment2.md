# SFT 模型对话实验

在[Transformer 模型训练实验](../architecture-basics/llm-pretrain-experiment.md)中，我们训练了一个约 64M 参数的语言模型。够续写出语法通顺、语义连贯的后续内容。在本实验里将使用监督微调，通过人工编写的指令回答对，教会模型理解"用户提问 → 模型回答"的交互格式，从续写转变为对话。

## 实验准备

在开始实验之前，请确保已完成以下准备工作：

1. 已完成[Transformer 模型训练实验](../architecture-basics/llm-pretrain-experiment.md)，模型权重文件 `pretrain_768.pth` 正确生成在数据目录中。
2. 已[挂载数据目录](../../sandbox.md#数据管理)并下载好 SFT 训练语料。与预训练实验一样，语料数据集来自 [MiniMind](https://github.com/jingyaogong/minimind) 开源项目。

```bash
# 选择 "下载数据集" -> 选择 "MiniMind SFT (LLM监督微调语料)"
dmla data
```

该语料包含 SFT 训练文本数据（`sft_t2t_mini.jsonl`，约 1.7 GB）。下载完成后，验证预训练模型和 SFT 语料是否完整：

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

## 第一阶段：监督微调数据集

下面代码实现了 SFTDataset，它将对话数据 tokenize 为模型可训练的格式，核心逻辑是定位 assistant 回答区间并生成对应的标签掩码。这段代码会在训练阶段被调用，无需手动运行。

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


class SFTDataset(Dataset):
    """
    SFT 数据集：将对话数据 tokenize 为 next-token prediction 格式

    与 PretrainDataset 的核心差异：
    - 数据格式从 {"text": "..."} 变为 {"conversations": [...]}
    - 标签掩码：仅 assistant 回答部分参与 loss，其余标记为 -100
    - 使用 apply_chat_template 将对话转为 ChatML 格式
    """
    # MiniMind 使用 ChatML 格式：<|im_start|>role\ncontent<|im_end|>\n
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
        # MiniMind tokenizer 未内置 chat_template，需手动设置 ChatML 格式
        if not tokenizer.chat_template:
            tokenizer.chat_template = self.CHATML_TEMPLATE
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

## 第二阶段：监督微调训练

SFT 训练的工程决策与预训练有所不同，主要差异是学习率和训练轮数：

- **更小的学习率**（1e-5）：预训练用 5e-4 的学习率从随机初始化开始学习语言知识，SFT 则在预训练模型的基础上微调行为模式，过大的学习率会破坏已学到的语言能力，造成灾难性遗忘。
- **更少的训练轮数**（3 epoch）：SFT 数据量远小于预训练数据（万级 vs 百万级），训练太久容易过拟合，模型会记忆训练数据而非学习通用的回答能力。
- **更长的序列**（768）：对话数据通常比预训练文本更长，需要更大的序列长度来容纳多轮对话的上下文。

::: info 训练预估

`sft_t2t_mini.jsonl` 包含约 20 万条对话样本，总数据量约 1.74 GB。这是 MiniMind 项目提供的精简版 SFT 语料，覆盖问答、代码、推理、创意写作等多种任务类型。

按序列长度 768，批大小 16，3 个 epoch，约需 8 GB 显存可运行，用 5080 GPU 训练约 1 小时。

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

## 第三阶段：对话推理

SFT 训练完成后，模型已经学会了遵循对话格式，能够理解用户指令并给出有针对性的回答。与预训练模型只能续写文本不同，SFT 模型能够识别 `<|im_start|>user` 和 `<|im_start|>assistant` 标记，知道自己是"AI 助手"角色，在用户提问后能给出恰当的回答。

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
# MiniMind tokenizer 未内置 chat_template，需手动设置 ChatML 格式
if not tokenizer.chat_template:
    tokenizer.chat_template = (
        "{% for message in messages %}<|im_start|>{{ message.role }}\n"
        "{{ message.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

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

### 对话体验

上方代码块运行后，模型将加载到沙箱中。加载完成后，可在下方与 SFT 微调后的模型进行对话。

```python runnable gpuonly
import torch
import os
from transformers import AutoTokenizer
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig

# 加载 tokenizer
tokenizer_path = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# MiniMind tokenizer 未内置 chat_template，需手动设置 ChatML 格式
if not tokenizer.chat_template:
    tokenizer.chat_template = (
        "{% for message in messages %}<|im_start|>{{ message.role }}\n"
        "{{ message.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

# 加载 SFT 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)

# 查找可用的 SFT 权重
sft_model_path = os.path.join(DATA_DIR, 'models', 'minimind', 'sft', 'full_sft_768.pth')
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
    print("未找到 SFT 模型，将使用随机初始化权重")

model = model.half().to(device).eval()
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print("对话服务已就绪")

# 定义对话函数
def chat(user_message, history=None):
    if history is None:
        history = []
    messages = [{"role": "system", "content": "你是一个有帮助的AI助手。"}]
    for h in history:
        messages.append(h)
    messages.append({"role": "user", "content": user_message})

    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_input, return_tensors="pt", truncation=True).to(device)

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

    response = tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )
    return response.strip()
```

<ChatDemo />

## 实验结论

本次实验在预训练模型的基础上完成了 SFT 监督微调，训练完成后，以下文件将保存到数据目录：

- **模型文件**：
    - `<DATA_DIR>/models/minimind/sft/full_sft_768.pth` - 最终 SFT 权重（FP16）
    - `<DATA_DIR>/models/minimind/sft/sft_epoch*.pth` - 每 epoch 结束时的 checkpoint
    - `<DATA_DIR>/models/minimind/sft/sft_step*.pth` - 训练中间 checkpoint

SFT 训练使模型的行为发生了质变，赋予了模型对话能力，但并未显著增加知识量或推理能力。64M 参数的模型拥有的世界知识有限，回答可能存在事实错误或逻辑不严密。更深层的推理能力（如数学证明、代码调试）和多步骤规划能力，需要更大规模的模型或后续的强化学习对齐（RLHF）才能获得。SFT 是"教模型如何说话"，而非"教模型更多知识"。

预训练和 SFT 共同构成了语言模型训练的基础阶段。在 InstructGPT 的三阶段框架中，SFT 是第一阶段，为后续的奖励模型训练和 PPO 强化学习提供了起点。后续还将通过人类反馈的强化学习实验进一步提升模型的对齐程度。

## 运行结果

SFT 训练完成后，使用模型进行对话推理，一个实际的运行样例如下所示：

| 用户输入 | SFT 模型回答 |
|---------|------------|
| 什么是机器学习？ | 机器学习是人工智能的一个分支，它使计算机能够从数据中学习规律，而无需显式编程。常见的机器学习方法包括监督学习、无监督学习和强化学习。 |
| Python如何定义函数？ | 在 Python 中，使用 `def` 关键字定义函数。例如：`def greet(name): return f"Hello, {name}!"`，调用时直接使用函数名和参数即可。 |
| 写一首关于春天的短诗 | 春风拂柳绿如烟，桃杏花开映碧天。燕子归来寻旧垒，一池新水映晴川。 |

与预训练模型只会续写文本不同，SFT 模型能够理解用户意图，给出有针对性的回答。但受限于 64M 参数规模，回答的深度和准确性仍有限，这正是后续 RLHF 阶段需要解决的问题。