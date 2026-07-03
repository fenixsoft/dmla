# DPO 对齐训练实验

在 [SFT 模型对话实验](../pretraining/llm-sft-experiment.md)中，我们通过监督微调教会了模型遵循对话格式，实现了从续写到对话的转变。SFT 的训练数据是"用户问 → 模型答"的指令回答对，模型学到的是模仿训练数据中的回答模式。但同一个问题可以有多种合理的回答方式，SFT 无法告诉模型哪一种更符合人类的偏好，回答更准确、语气更友好、拒绝更礼貌，这些维度上的差异 SFT 都捕捉不到。

本实验使用直接偏好优化方法，在 SFT 模型的基础上进行对齐训练。DPO 的训练数据是同一个问题的两个回答，人类标注哪个更好，模型从中学会区分回答的优劣，在生成时更倾向于选择偏好的回答风格。

## 实验准备

在开始实验之前，请确保已完成以下准备工作：

1. 已完成 [SFT 模型对话实验](../pretraining/llm-sft-experiment.md)，模型权重文件 `full_sft_768.pth` 已在数据目录中正确生成。
2. 已[挂载数据目录](../../appendixes/sandbox.md#数据管理)并下载好 DPO 偏好语料。

```bash
# 选择 "下载数据集" -> 选择 "MiniMind Alignment (LLM对齐语料)"
dmla data
```

MiniMind 项目的 DPO 偏好语料（`dpo.jsonl`）包含约 20K 条偏好对比数据，数据抽样自 [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)，体积约 53 MB。数据集下载完成后，以下代码可验证 SFT 模型和 DPO 语料是否完整：

```python runnable gpu
import os

# 检查 SFT 模型（由上一章实验生成）
sft_dir = os.path.join(DATA_DIR, 'models', 'minimind', 'sft')
sft_path = os.path.join(sft_dir, 'full_sft_768.pth')
if os.path.exists(sft_path):
    size_mb = os.path.getsize(sft_path) / (1024 ** 2)
    print(f"SFT 模型: 已存在 ({size_mb:.1f} MB)")
else:
    # 尝试 epoch checkpoint
    for epoch in [2, 1]:
        ckp = os.path.join(sft_dir, f'sft_epoch{epoch}.pth')
        if os.path.exists(ckp):
            size_mb = os.path.getsize(ckp) / (1024 ** 2)
            print(f"SFT 模型: 使用 epoch {epoch} checkpoint ({size_mb:.1f} MB)")
            break
    else:
        print("SFT 模型: 未找到！请先完成 SFT 实验")

# 检查 DPO 语料
dpo_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-alignment')
if os.path.exists(dpo_dir):
    print(f"DPO 语料目录: 已存在")
    for f in os.listdir(dpo_dir):
        fpath = os.path.join(dpo_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 ** 2)
            print(f"  {f}: {size_mb:.1f} MB")
else:
    print("DPO 语料: 未下载，请运行 'dmla data' 下载 MiniMind Alignment 数据集")

# 检查 tokenizer（复用预训练的）
tokenizer_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
tokenizer_json = os.path.join(tokenizer_dir, 'tokenizer.json')
print(f"Tokenizer: {'已存在' if os.path.exists(tokenizer_json) else '未找到'}")
```

## 第一阶段：偏好对比数据集

DPO 的训练数据格式与 SFT 不同。SFT 的每条样本是一个指令回答对 $(x, y)$，DPO 的每条样本是一个偏好对比三元组 $(x, y_w, y_l)$，其中 $x$ 是用户指令，$y_w$ 是被选中的好回答（chosen），$y_l$ 是被拒绝的差回答（rejected）。chosen 和 rejected 对应同一个用户指令，只有 assistant 的回答不同。

数据以 JSONL 格式存储，每行一条偏好对：

```json
{
  "chosen": [
    {"role": "user", "content": "什么是机器学习"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习规律..."}
  ],
  "rejected": [
    {"role": "user", "content": "什么是机器学习"},
    {"role": "assistant", "content": "机器学习就是让电脑自己学东西"}
  ]
}
```

下面代码实现了 DPODataset，将偏好对比数据转换为模型可训练的格式，每条样本包含 chosen 和 rejected 两条对话，分别 tokenize 后生成对应的输入序列和掩码。掩码的作用是只在 assistant 回答部分计算对数概率，用户提问部分不参与 DPO 损失计算。这段代码会在训练阶段被调用，无需手动运行。

```python runnable gpu extract-class="DPODataset"
import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Features, Value
from datasets import logging as datasets_logging

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
```

## 第二阶段：DPO 损失函数

DPO 的损失函数是整个训练的核心，也是 DPO 与 PPO 的根本区别所在。在 [对齐方法的演进](./alignment-new-paradigms.md)中，我们推导了 DPO 的损失函数：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right) \right]$$

其中 $\pi_\theta$ 是策略模型（训练中更新参数），$\pi_{\text{ref}}$ 是参考模型（参数冻结），$y_w$ 是 chosen 回答，$y_l$ 是 rejected 回答，$\beta$ 控制模型偏离参考模型的程度。实现上，先计算模型在每个 token 位置的对数概率，再沿序列求和得到整条回答的对数概率，最后代入 DPO 损失公式。关键步骤如下：

1. **计算对数概率**：对模型输出的 logits 取 Softmax，再提取目标 token 对应位置的值，得到每个 token 的对数概率 $\log \pi(y_t | x, y_{<t})$。
2. **掩码求和**：用掩码遮蔽提问部分，只对回答部分求和，得到整条回答的对数概率 $\sum_{t \in \text{assistant}} \log \pi(y_t | x, y_{<t})$。
3. **计算隐式奖励**：$\beta(\log \pi_\theta - \log \pi_{\text{ref}})$，即策略模型相对于参考模型的对数概率比乘以 $\beta$。
4. **计算 DPO 损失**：$-\log \sigma(\text{chosen\_reward} - \text{rejected\_reward})$。

```python runnable gpu extract-class="logits_to_log_probs, dpo_loss"
import torch
import torch.nn.functional as F

def logits_to_log_probs(logits, labels):
    """
    从模型输出的 logits 计算每个 token 位置的对数概率

    Args:
        logits: 模型输出, shape [batch, seq_len, vocab_size]
        labels: 目标 token ids, shape [batch, seq_len]

    Returns:
        每个位置的对数概率, shape [batch, seq_len]
    """
    # 在 float32 下计算 log_softmax，避免 bfloat16 精度不足导致数值溢出
    log_probs = F.log_softmax(logits.float(), dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算 DPO 损失

    Args:
        ref_log_probs: 参考模型的对数概率, shape [batch, seq_len]
        policy_log_probs: 策略模型的对数概率, shape [batch, seq_len]
        mask: loss 掩码, shape [batch, seq_len]
        beta: DPO 温度参数

    Returns:
        标量损失值
    """
    # 沿序列求和（仅在 mask 为 1 的位置）
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开
    # batch 中前半部分是 chosen，后半部分是 rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 计算隐式奖励差值
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios

    # DPO 损失 = -log(sigmoid(beta * logits))
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

## 第三阶段：DPO 训练

DPO 训练的起点是 SFT 阶段的模型。与 SFT 只需一个模型不同，DPO 需要同时维护策略模型 $\pi_\theta$（可训练）和参考模型 $\pi_{\text{ref}}$（冻结）两个模型。策略模型从 SFT 权重初始化，训练中参数会更新。参考模型同样从 SFT 权重初始化，但参数冻结不动，充当行为基准。训练过程中，DPO 损失驱动策略模型在参考模型的基础上调整生成概率，让 chosen 回答的概率相对升高、rejected 回答的概率相对降低。下表列出了本实验的关键工程决策及原因：

| 训练决策 | MiniMind | 本实验 | 调整原因 |
|---------|----------|-------|---------|
| 学习率 | 4e-8 | 1e-5 | MiniMind 的学习率极小，因为它的 DPO 训练在全量 20K 数据上运行数千步，余弦调度有足够的步数缓慢衰减。本实验数据量相同但 batch_size 更小，总步数更少，4e-8 的学习率几乎无法产生有效的参数更新。但 DPO 对参数更新非常敏感（策略模型与参考模型的 log_prob 差值即使微小变化也会显著影响梯度），学习率过大会导致 loss 震荡。1e-5 在有效更新与训练稳定性之间取得平衡 |
| $\beta$ | 0.15 | 0.1 | $\beta$ 控制模型偏离参考模型的程度。MiniMind 用 0.15 略保守，本实验降低至 0.1，让偏好信号的影响更明显，便于观察训练效果 |
| 序列长度 | 1024 | 768 | 与 SFT 实验保持一致。DPO 的显存占用是 SFT 的约 2.5 倍（策略模型 + 参考模型 + chosen + rejected），序列越长显存压力越大 |
| 批大小 | 4 | 4 | 一致。DPO 的每个批次包含 chosen 和 rejected 两条序列，实际前向传播的 batch_size 等效为 8，显存占用较高 |
| 梯度累积 | 1 | 4 | 本实验 batch_size = 4，梯度累积 4 步，等效 batch_size = 16，与 MiniMind 的有效批大小接近 |

::: info 训练预估

`dpo.jsonl` 包含约 20K 条偏好对比样本，体积约 53 MB。按序列长度 768，批大小 4（梯度累积 × 4，等效批大小 16），1 个 epoch，约需 16 GB 显存可运行（策略模型 + 参考模型同时加载），用 RTX 5080 GPU 训练时间约为 20 分钟。

DPO 的显存占用显著高于 SFT，原因是训练过程中同时加载了策略模型和参考模型两个完整的模型副本，且每个批次需要分别对 chosen 和 rejected 两条序列做前向传播。如果显存不足，可以降低 `batch_size` 并等比例增大 `accumulation_steps`。

:::

```python runnable gpuonly timeout=unlimited
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer

# 导入进度报告模块
from dmla_progress import ProgressReporter

# 导入共享模块
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig
from shared.llm.dpodataset import DPODataset
from shared.llm.logits_to_log_probs import logits_to_log_probs, dpo_loss

# ========== 路径配置 ==========
TOKENIZER_PATH = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
DPO_DATA_PATH = os.path.join(DATA_DIR, 'datasets', 'minimind-alignment', 'dpo.jsonl')
SFT_MODEL_PATH = os.path.join(DATA_DIR, 'models', 'minimind', 'sft', 'full_sft_768.pth')
SAVE_DIR = os.path.join(DATA_DIR, 'models', 'minimind', 'dpo')

# ========== 训练超参数 ==========
hidden_size = 768
num_hidden_layers = 8
max_seq_len = 768
batch_size = 4             # DPO 显存占用高（双模型 + chosen/rejected），batch_size 不宜过大
learning_rate = 1e-5       # DPO 学习率（DPO 对参数敏感，学习率不宜过大）
beta = 0.1                 # DPO 温度参数，控制偏离参考模型的程度
num_epochs = 1
accumulation_steps = 4     # 梯度累积（等效 batch_size = 4 × 4 = 16）
grad_clip = 1.0
log_interval = 50
save_interval = 200

# ========== 1. 初始化环境 ==========
progress = ProgressReporter(total_steps=10, description="准备 DPO 训练环境")
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
progress.update(2, message="加载 tokenizer 和 DPO 训练数据...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
train_ds = DPODataset(DPO_DATA_PATH, tokenizer, max_length=max_seq_len)
print(f"训练样本数: {len(train_ds):,}")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True
)
total_steps_per_epoch = len(train_loader) // accumulation_steps
total_steps = num_epochs * total_steps_per_epoch
print(f"每 epoch 优化步数: {total_steps_per_epoch:,}（mini-steps: {len(train_loader):,} / 累积: {accumulation_steps}）")
print(f"总优化步数: {total_steps:,}")

# ========== 3. 创建策略模型和参考模型 ==========
progress.update(4, message="创建策略模型和参考模型...")
lm_config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)

# 策略模型（可训练）
model = MiniMindForCausalLM(lm_config)

# 参考模型（冻结）
ref_model = MiniMindForCausalLM(lm_config)

# 加载 SFT 权重作为两个模型的初始化
weight_path = None
if os.path.exists(SFT_MODEL_PATH):
    weight_path = SFT_MODEL_PATH
else:
    for epoch in [2, 1]:
        ckp = os.path.join(DATA_DIR, 'models', 'minimind', 'sft', f'sft_epoch{epoch}.pth')
        if os.path.exists(ckp):
            weight_path = ckp
            break

if weight_path:
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=False)
    ref_model.load_state_dict(weights, strict=False)
    print(f"已加载 SFT 权重: {weight_path}")
else:
    print("未找到 SFT 权重，使用随机初始化")

model = model.to(device)
ref_model = ref_model.to(device)
ref_model.eval()
ref_model.requires_grad_(False)

total_params = sum(p.numel() for p in model.parameters())
print(f"策略模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"参考模型参数量: {total_params:,} ({total_params/1e6:.2f}M，冻结)")

# ========== 4. 配置训练组件 ==========
progress.update(6, message="配置优化器和学习率调度...")

device_type = "cuda" if device.type == "cuda" else "cpu"
autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, dtype=torch.bfloat16)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

def get_lr(current_step, total_steps, lr):
    """线性 warmup（前 10%）+ 余弦衰减"""
    warmup_steps = int(0.1 * total_steps)
    if current_step < warmup_steps:
        return lr * current_step / warmup_steps
    progress_ratio = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress_ratio)))

os.makedirs(SAVE_DIR, exist_ok=True)
progress.update(8, message="DPO 训练环境准备完成")

# ========== 5. 开始训练 ==========
progress.reset(total_steps=total_steps, description="DPO 对齐训练")

global_step = 0

for epoch in range(num_epochs):
    model.train()
    epoch_start = time.time()
    running_dpo_loss = 0.0
    log_step_count = 0

    for step, batch in enumerate(train_loader):
        # 将 chosen 和 rejected 拼接为一个批次，一次前向传播同时计算
        x_chosen = batch['x_chosen'].to(device)
        x_rejected = batch['x_rejected'].to(device)
        y_chosen = batch['y_chosen'].to(device)
        y_rejected = batch['y_rejected'].to(device)
        mask_chosen = batch['mask_chosen'].to(device)
        mask_rejected = batch['mask_rejected'].to(device)

        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 前向传播（混合精度）
        with autocast_ctx:
            # 参考模型前向传播（不计算梯度）
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 策略模型前向传播
            outputs = model(x)
            policy_logits = outputs.logits
            policy_log_probs = logits_to_log_probs(policy_logits, y)

            # 计算 DPO 损失
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val / accumulation_steps

        # 反向传播
        loss.backward()

        # 记录损失（每个 mini-step 都记录，用于日志平均）
        current_dpo = dpo_loss_val.item()
        running_dpo_loss += current_dpo
        log_step_count += 1

        # 梯度累积 + 参数更新
        if (step + 1) % accumulation_steps == 0:
            # 学习率调度（基于实际优化步数）
            lr = get_lr(global_step, total_steps, learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # 日志打印
            if global_step % log_interval == 0:
                avg_dpo = running_dpo_loss / log_step_count
                elapsed = time.time() - epoch_start
                eta_min = elapsed / max(global_step, 1) * (total_steps - global_step) / 60
                print(f"Epoch[{epoch+1}/{num_epochs}] Step[{global_step}/{total_steps}], "
                      f"dpo_loss: {avg_dpo:.4f}, lr: {lr:.8f}, eta: {eta_min:.1f}min")
                progress.update(
                    global_step,
                    message=f"Epoch {epoch+1}/{num_epochs}, Step {global_step}/{total_steps}, DPO Loss={avg_dpo:.4f}",
                    extra_data={"dpo_loss": avg_dpo, "lr": lr, "epoch": epoch + 1}
                )
                running_dpo_loss = 0.0
                log_step_count = 0

            # 周期性保存模型
            if global_step % save_interval == 0:
                model.eval()
                save_path = os.path.join(SAVE_DIR, f'dpo_step{global_step}.pth')
                state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
                torch.save(state_dict, save_path)
                print(f"  -> 保存模型: step={global_step}, dpo_loss={avg_dpo:.4f}")
                model.train()
                del state_dict

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected
        del x, y, mask, ref_outputs, ref_logits, ref_log_probs
        del outputs, policy_logits, policy_log_probs, dpo_loss_val

    # 每 epoch 结束保存
    epoch_time = time.time() - epoch_start
    model.eval()
    epoch_save_path = os.path.join(SAVE_DIR, f'dpo_epoch{epoch+1}.pth')
    state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, epoch_save_path)
    print(f"\nEpoch {epoch+1} 完成, 耗时 {epoch_time/60:.1f}min, 模型已保存")
    model.train()
    del state_dict

# 保存最终模型
final_path = os.path.join(SAVE_DIR, 'full_dpo_768.pth')
state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
torch.save(state_dict, final_path)
progress.complete(message=f"DPO 训练完成！模型已保存到 {final_path}")
print(f"\n最终模型已保存: {final_path}")
```

## 第四阶段：对话推理

DPO 训练完成后，模型在 SFT 的基础上进一步学会了区分回答的优劣。与 SFT 模型相比，DPO 模型在回答风格上更符合偏好数据中的选择倾向，回答更有条理、语气更恰当、拒绝时也更礼貌。不过，64M 参数的模型能力有限，DPO 对齐的改善幅度不如 7B 级模型那么显著，但训练流程和原理是相同的。

运行下方代码块后，模型将加载到沙箱中。加载完成后，可在下方的对话框中与对齐后的模型进行对话。体验结束后，点击 Stop 按钮停止推理进程。

```python runnable gpuonly mode=chat
import torch
import os
from transformers import AutoTokenizer
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig

# 加载 tokenizer
tokenizer_path = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
if not tokenizer.chat_template:
    tokenizer.chat_template = (
        "{% for message in messages %}<|im_start|>{{ message.role }}\n"
        "{{ message.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

# 加载 DPO 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)

# 查找可用的 DPO 权重
dpo_model_path = os.path.join(DATA_DIR, 'models', 'minimind', 'dpo', 'full_dpo_768.pth')
weight_path = None
if os.path.exists(dpo_model_path):
    weight_path = dpo_model_path
else:
    for epoch in [1]:
        ckp = os.path.join(DATA_DIR, 'models', 'minimind', 'dpo', f'dpo_epoch{epoch}.pth')
        if os.path.exists(ckp):
            weight_path = ckp
            break

if not weight_path:
    # 回退到 SFT 模型
    sft_path = os.path.join(DATA_DIR, 'models', 'minimind', 'sft', 'full_sft_768.pth')
    if os.path.exists(sft_path):
        weight_path = sft_path
        print("未找到 DPO 模型，回退到 SFT 模型")

if weight_path:
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=False)
    print(f"已加载权重: {weight_path}")
else:
    print("未找到模型权重，将使用随机初始化")

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
            max_new_tokens=512,
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

::: details 运行上面代码后，点击这里进行对话
<ChatDemo />
:::

## 实验结论

本次实验在 SFT 模型的基础上完成了 DPO 对齐训练，训练完成后，以下文件将保存到数据目录：

- **模型文件**：
    - `<DATA_DIR>/models/minimind/dpo/full_dpo_768.pth` - 最终 DPO 权重（FP16 精度）
    - `<DATA_DIR>/models/minimind/dpo/dpo_epoch*.pth` - 每 epoch 结束时的 Checkpoint
    - `<DATA_DIR>/models/minimind/dpo/dpo_step*.pth` - 训练中间 Checkpoint

DPO 训练使模型从"学会回答"进阶到"学会区分回答优劣"。与 RLHF 的三模型架构（策略模型 + 奖励模型 + 参考模型）相比，DPO 只需两个模型（策略模型 + 参考模型），绕过了奖励模型的训练和 PPO 的不稳定性，将对齐训练的工程门槛大幅降低。DPO 的局限在于 $\beta$ 参数是固定的，不如 PPO 的自适应 KL 惩罚灵活，且对长序列的对数概率计算可能不稳定。在 [对齐方法的演进](./alignment-new-paradigms.md)中介绍的 KTO 和 GRPO 等方法，从不同角度进一步简化了对齐训练的流程。

至此，我们完成了语言模型训练的完整流程。预训练赋予模型语言能力，SFT 赋予模型对话能力，DPO 赋予模型偏好对齐能力。三个阶段层层递进，每一步都建立在前一步的基础之上。
