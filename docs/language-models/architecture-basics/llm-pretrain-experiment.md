# Minimind 语言模型实验

在前面几章中，我们了解了 Transformer 架构的原理、现代 LLM 的各种改进技术、以及分词器的工作方式。这些知识构成了理解大语言模型的理论基础，但仅凭理论很难真正体会训练一个语言模型时面临的种种工程抉择。本次实验中，我们将亲手训练一个约 64M 参数的大语言模型 MiniMind，从语料准备到预训练再到推理对话，完整走一遍语言模型从零到能说话的全流程。

MiniMind 是一个教学向的小型语言模型项目，其架构设计遵循现代 LLM 的主流选择：RMSNorm 替代 LayerNorm、RoPE 旋转位置编码、GQA 分组查询注意力、SwiGLU 激活函数，以及词表规模仅 6400 的紧凑 BPE 分词器。这些组件在前面的章节中都已经介绍过，本次实验的目的是将它们组装起来，观察训练过程中 loss 的下降曲线，理解预训练赋予模型的语言能力边界，以及预训练模型与经过 SFT 对齐后的模型在对话能力上的差距。

## 实验准备

在开始实验之前，请确保已[挂载数据目录](../../sandbox.md#数据管理)并下载好 MiniMind 预训练语料，你可以通过 `DMLA-CLI` 工具自动完成该工作：

```bash
# 选择 "下载数据集" -> 选择 "MiniMind Pretrain (LLM预训练语料)"
dmla data
```

该语料包包含预训练文本数据（`pretrain_t2t_mini.jsonl`，约 1.2GB）以及配套的 BPE 分词器文件（`tokenizer.json` 和 `tokenizer_config.json`）。下载完成后，验证语料和分词器文件是否完整：

```python runnable gpu
import os

# 语料和分词器目录（DATA_DIR 由 kernel 自动注入）
# Docker 模式: DATA_DIR='/data', Native 模式: DATA_DIR='~/dmla-data'
data_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')

if os.path.exists(data_dir):
    print("语料目录已存在")
    
    # 检查预训练语料
    pretrain_file = os.path.join(data_dir, 'pretrain_t2t_mini.jsonl')
    if os.path.exists(pretrain_file):
        file_size = os.path.getsize(pretrain_file) / (1024 ** 3)
        print(f"预训练语料: {file_size:.2f} GB")
        
        # 统计行数（采样估计）
        with open(pretrain_file, 'r', encoding='utf-8') as f:
            sample_lines = sum(1 for _ in f for _ in range(1) if f.tell() < 10 * 1024 * 1024)
        print(f"语料前 10MB 包含 {sample_lines} 条样本")
    else:
        print("预训练语料未找到")
    
    # 检查分词器文件
    tokenizer_json = os.path.join(data_dir, 'tokenizer.json')
    tokenizer_config = os.path.join(data_dir, 'tokenizer_config.json')
    print(f"tokenizer.json: {'已存在' if os.path.exists(tokenizer_json) else '未找到'}")
    print(f"tokenizer_config.json: {'已存在' if os.path.exists(tokenizer_config) else '未找到'}")
else:
    print("语料目录未下载，请运行 'dmla data' 下载 MiniMind 预训练语料")
```

## 第一阶段：Tokenizer 与数据加载

### 加载分词器

MiniMind 使用词表规模为 6400 的 BPE 分词器，采用 Qwen 风格的 chat template 格式。词表虽小，但通过 BPE 的子词合并机制，6400 个词表条目就足以覆盖中英文的常见字词组合，未登录词会被拆解为更小的子词单元而非直接标记为 `<unk>`。词表中还包含了 `<|im_start|>` 和 `<|im_end|>` 等对话控制标记，它们是后续 SFT 阶段实现多轮对话的基础，预训练阶段虽然不使用这些标记，但它们已经占据词表中的固定位置，确保预训练权重与 SFT 权重的词表对齐。

```python runnable gpu
from transformers import AutoTokenizer
import os

# 加载 MiniMind 分词器
tokenizer_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

# 词表基本信息
print(f"词表大小: {len(tokenizer)}")
print(f"BOS 标记: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
print(f"EOS 标记: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"PAD 标记: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

# 分词演示
text = "大语言模型是人工智能的重要方向"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens, skip_special_tokens=True)

print(f"\n分词演示:")
print(f"原文: {text}")
print(f"Token IDs: {tokens}")
print(f"Token 数量: {len(tokens)}")
print(f"解码还原: {decoded}")

# 压缩率（字符数 / token 数）
compression = len(text) / len(tokens)
print(f"压缩率: {compression:.2f} 字符/token")
```

### 预训练数据集

预训练语料的格式为 JSONL（每行一个 JSON 对象），每条样本包含一个 `text` 字段，存储一段连续文本。预训练的目标是让模型学会预测下一个 token，即给定序列 $w_1, w_2, ..., w_t$，模型需要学会输出 $P(w_{t+1} | w_1, ..., w_t)$。因此，数据集的加载逻辑相对简单：将每条文本 tokenize 为 token ID 序列，加上 BOS 和 EOS 标记，然后对齐到固定长度即可。

与 AlexNet 实验中图像数据的预处理不同，文本数据的预处理开销极小，tokenize 操作本身是 CPU 上的查表与字符串匹配，速度远快于 JPEG 解码和 Resize。因此预训练数据集不需要 LMDB 缓存等优化手段，直接从 JSONL 文件逐行读取并实时 tokenize 即可。

```python runnable gpu extract-class="PretrainDataset"
import torch
from torch.utils.data import Dataset
import json

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
```

::: info 预训练语料规模

MiniMind 的 `pretrain_t2t_mini.jsonl` 包含约 200 万条文本样本，总数据量约 1.2GB。这是 MiniMind 项目提供的精简版语料，适合在单卡 GPU 上快速复现预训练流程。完整版语料 `pretrain_t2t.jsonl` 约 10GB，训练效果更好但耗时更长。本实验使用精简版语料，以 512 的序列长度和 32 的批大小训练 2 个 epoch，单卡 RTX 3090 约需 2-3 小时。

:::

## 第二阶段：模型定义

MiniMind 的模型架构遵循现代小型 LLM 的主流设计，每个组件都可以在前面的章节中找到对应的介绍。这里我们将这些组件组装成一个完整的因果语言模型。

模型的核心配置如下：

| 配置项 | 值 | 说明 |
|--------|------|------|
| `hidden_size` | 768 | 隐藏层维度 |
| `num_hidden_layers` | 8 | Transformer 层数 |
| `num_attention_heads` | 8 | 查询头数 |
| `num_key_value_heads` | 4 | KV 头数（GQA，2 组共享） |
| `head_dim` | 96 | 每个头的维度（768 / 8） |
| `intermediate_size` | 2432 | FFN 中间层维度（SwiGLU） |
| `vocab_size` | 6400 | 词表大小 |
| `tie_word_embeddings` | True | 词嵌入与输出头共享权重 |

GQA（Grouped Query Attention）是 MiniMind 相对原始 Transformer 的一个重要改进。8 个查询头共享 4 组 KV 头，每组 2 个查询头共用同一组 Key 和 Value。这不会影响模型的表达能力（训练时与 MHA 等价），但推理时 KV Cache 的显存占用减半，对于小模型在资源受限环境下的部署尤其重要。

SwiGLU 激活函数替代了原始 Transformer 的 ReLU，是 LLaMA 系列模型开创的设计。SwiGLU 将 FFN 的计算从 $W_2 \cdot \text{ReLU}(W_1 x)$ 改为 $W_{down} \cdot (\text{SiLU}(W_{gate} x) \odot W_{up} x)$，引入门控机制让模型自适应地选择信息通道，代价是参数量增加约 50%（三个投影矩阵而非两个），但实验表明同等计算预算下 SwiGLU 的效果优于 ReLU 和 GELU。

词嵌入与输出头共享权重（`tie_word_embeddings=True`）是另一个常见的参数节省策略。词嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{V \times d}$ 将 token ID 映射为向量，输出头 $\mathbf{W}_{out} \in \mathbb{R}^{d \times V}$ 将隐藏状态映射回词表概率。两者的形状互为转置，共享权重意味着 $\mathbf{W}_{out} = \mathbf{E}^T$，节省了 $V \times d = 6400 \times 768 \approx 4.9\text{M}$ 个参数。这个设计在 GPT-2 和 LLaMA 等模型中都被采用。

```python runnable gpu extract-class="MiniMindConfig, RMSNorm, precompute_freqs_cis, apply_rotary_pos_emb, repeat_kv, Attention, FeedForward, MiniMindBlock, MiniMindModel, MiniMindForCausalLM"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

class MiniMindConfig(PretrainedConfig):
    """MiniMind 模型配置"""
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32, "beta_slow": 1, "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0, "type": "yarn"
        } if self.inference_rope_scaling else None

class RMSNorm(nn.Module):
    """RMS 归一化：比 LayerNorm 更高效，省去均值计算"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim, end=32768, rope_base=1e6, rope_scaling=None):
    """预计算 RoPE 旋转位置编码的 cos 和 sin 值"""
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """应用旋转位置编码到查询和键"""
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x, n_rep):
    """重复 KV 头以匹配查询头数（GQA 推理时使用）"""
    bs, slen, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_kv_heads, n_rep, head_dim).reshape(bs, slen, num_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
    """GQA 分组查询注意力"""
    def __init__(self, config):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # QK-Norm：对查询和键做 RMS 归一化，提升训练稳定性
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # KV Cache：推理时拼接历史 KV
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # 优先使用 Flash Attention（GPU 上更快更省显存）
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    """SwiGLU 前馈网络：gate 和 up 两条路径，门控选择信息通道"""
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MiniMindBlock(nn.Module):
    """单个 Transformer 层：Pre-Norm + Attention + FFN"""
    def __init__(self, layer_id, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    """MiniMind 主体：词嵌入 + 多层 Transformer + 最终归一化"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 预计算 RoPE 的 cos/sin 缓冲区
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, end=config.max_position_embeddings,
            rope_base=config.rope_theta, rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 重新计算可能因 meta device 丢失的 RoPE 缓冲区
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim, end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling
            )
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, position_embeddings,
                past_key_value=past_key_value, use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        return hidden_states, presents, hidden_states.new_zeros(1).squeeze()

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """MiniMind 因果语言模型：用于预训练和推理"""
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config=None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=512, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """自回归生成：逐 token 采样，支持 top-k、top-p、重复惩罚"""
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            # 重复惩罚：降低已出现 token 的概率
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            # Top-k 过滤
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            # Top-p（nucleus）过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            # 采样或贪心选择
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break
        if streamer:
            streamer.end()
        return input_ids

# 创建模型实例并统计参数量
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"MiniMind 模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"词表大小: {config.vocab_size}")
print(f"隐藏层维度: {config.hidden_size}")
print(f"Transformer 层数: {config.num_hidden_layers}")
print(f"注意力头数: {config.num_attention_heads} (Q) / {config.num_key_value_heads} (KV)")
print(f"FFN 中间维度: {config.intermediate_size}")
print(f"词嵌入与输出头共享: {config.tie_word_embeddings}")
```

## 第三阶段：预训练

预训练是语言模型最核心的训练阶段，目标是让模型通过大量文本数据学会预测下一个 token。训练流程包含三个关键步骤：前向传播计算预测值与损失、反向传播计算梯度、优化器更新参数。这三个步骤在数百万条文本样本上反复执行，模型的 loss 从初始的约 8-9（接近 $\ln(6400) \approx 8.76$，即均匀分布的交叉熵）逐步下降到 3-4 左右，意味着模型从对文本一无所知到能够给出相当合理的下一个词预测。

本次预训练的核心工程决策如下：

- **混合精度训练**：使用 BFloat16 精度进行前向和反向计算，减少显存占用并加速计算，同时保持足够的数值精度避免训练不稳定。BFloat16 相比 Float16 的优势在于指数位与 Float32 相同，不会出现上溢和下溢问题，因此不需要 GradScaler。
- **余弦学习率调度**：学习率从初始值出发，按余弦曲线平滑衰减到接近零。相比固定学习率或阶梯衰减，余弦调度在训练初期保持较高学习率加速收敛，后期缓慢降低学习率精细调整参数，是预训练中最常用的调度策略。
- **梯度裁剪**：将梯度的全局范数裁剪到 1.0 以内，防止梯度爆炸导致训练崩溃。预训练的 loss 曲线在初期波动较大，梯度裁剪是保证训练稳定性的重要安全阀。
- **周期性保存**：每 1000 步保存一次模型权重，训练结束后保存最终权重。保存的权重文件可以在推理阶段直接加载，也可以作为 SFT 阶段的初始化权重。

::: info 训练预估

训练语料约 200 万条样本，序列长度 512，批大小 32，2 个 epoch，约需 8G 显存可运行，用 GPU 训练约 2-3 小时。

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

# 导入共享模块中的 MiniMind 模型和数据集
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig
from shared.llm.pretrain_dataset import PretrainDataset

# ========== 路径配置（DATA_DIR 由 kernel 自动注入） ==========
DATA_PATH = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain', 'pretrain_t2t_mini.jsonl')
TOKENIZER_PATH = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
SAVE_DIR = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain')

# ========== 训练超参数 ==========
hidden_size = 768
num_hidden_layers = 8
max_seq_len = 512
batch_size = 32
learning_rate = 5e-4
num_epochs = 2
accumulation_steps = 8    # 梯度累积步数（等效 batch_size = 32 × 8 = 256）
grad_clip = 1.0           # 梯度裁剪阈值
log_interval = 100        # 日志打印间隔
save_interval = 1000      # 模型保存间隔

# ========== 1. 初始化环境 ==========
progress = ProgressReporter(total_steps=10, description="准备训练环境")
progress.update(0, message="检测运行环境...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("警告: 未检测到 GPU，训练将非常缓慢")

# 设置随机种子
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed(42)

# ========== 2. 加载分词器和数据 ==========
progress.update(2, message="加载分词器和训练数据...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
train_ds = PretrainDataset(DATA_PATH, tokenizer, max_length=max_seq_len)
print(f"训练样本数: {len(train_ds):,}")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True
)
total_steps_per_epoch = len(train_loader)
total_steps = num_epochs * total_steps_per_epoch
print(f"每 epoch 步数: {total_steps_per_epoch:,}")
print(f"总训练步数: {total_steps:,}")

# ========== 3. 创建模型 ==========
progress.update(4, message="创建 MiniMind 模型...")
lm_config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
model = MiniMindForCausalLM(lm_config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# ========== 4. 配置训练组件 ==========
progress.update(6, message="配置优化器和学习率调度...")

# 混合精度（BFloat16 不需要 GradScaler）
device_type = "cuda" if device.type == "cuda" else "cpu"
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=torch.bfloat16)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

def get_lr(current_step, total_steps, lr):
    """余弦学习率调度：平滑衰减，训练后期精细调整"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

os.makedirs(SAVE_DIR, exist_ok=True)
progress.update(8, message="训练环境准备完成")

# ========== 5. 开始训练 ==========
progress.reset(total_steps=total_steps, description="预训练 MiniMind")

global_step = 0
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_start = time.time()
    running_loss = 0.0
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
        
        # 记录损失
        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss
        log_step_count += 1
        global_step += 1
        
        # 日志打印
        if global_step % log_interval == 0:
            avg_loss = running_loss / log_step_count
            elapsed = time.time() - epoch_start
            eta_min = elapsed / max(global_step - epoch * total_steps_per_epoch, 1) * (total_steps - global_step) / 60
            print(f"Epoch[{epoch+1}/{num_epochs}] Step[{step+1}/{total_steps_per_epoch}], "
                  f"loss: {avg_loss:.4f}, lr: {lr:.8f}, eta: {eta_min:.1f}min")
            progress.update(
                global_step,
                message=f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{total_steps_per_epoch}, Loss={avg_loss:.4f}",
                extra_data={"loss": avg_loss, "lr": lr, "epoch": epoch + 1}
            )
            running_loss = 0.0
            log_step_count = 0
        
        # 周期性保存模型
        if global_step % save_interval == 0:
            model.eval()
            save_path = os.path.join(SAVE_DIR, f'pretrain_step{global_step}.pth')
            state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
            torch.save(state_dict, save_path)
            print(f"  -> 保存模型: step={global_step}, loss={current_loss:.4f}")
            model.train()
            del state_dict
        
        del input_ids, labels, res, loss
    
    # 每 epoch 结束保存
    epoch_time = time.time() - epoch_start
    model.eval()
    epoch_save_path = os.path.join(SAVE_DIR, f'pretrain_epoch{epoch+1}.pth')
    state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, epoch_save_path)
    print(f"\nEpoch {epoch+1} 完成, 耗时 {epoch_time/60:.1f}min, 模型已保存")
    model.train()
    del state_dict

# 保存最终模型
final_path = os.path.join(SAVE_DIR, 'pretrain_768.pth')
state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
torch.save(state_dict, final_path)
progress.complete(message=f"预训练完成！模型已保存到 {final_path}")
print(f"\n最终模型已保存: {final_path}")
```

## 第四阶段：推理与对话

预训练完成后，模型已经学会了语言的统计规律，能够根据前文预测下一个 token。但预训练模型的能力与经过 SFT 对齐的模型有本质区别：预训练模型只学会了文本续写，即给定一段文本开头，模型会按照训练语料中的统计规律继续生成文本；而 SFT 模型学会了遵循指令和对话格式，能够理解用户的意图并给出有针对性的回答。

用人类学习来类比，预训练相当于广泛阅读了大量书籍，积累了语言和知识的基础，但还不知道如何与人对话；SFT 相当于学习了对话的示范，知道了面对提问应该如何回答。本次实验只完成预训练阶段，SFT 将在下一章进行。

预训练模型的推理使用自回归生成：给定输入文本的 token 序列，模型逐 token 预测下一个 token 的概率分布，通过采样策略（top-k、top-p）从分布中选取一个 token，将其拼接到序列末尾，再以新序列为输入继续预测下一个 token，直到生成 EOS 标记或达到最大长度。MiniMind 的 `generate` 方法已经内置了这一逻辑，包括 KV Cache 加速和重复惩罚机制。

```python runnable gpuonly
import torch
import os
from transformers import AutoTokenizer

# 导入共享模块中的 MiniMind 模型
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig

# ========== 加载模型和分词器 ==========
tokenizer_path = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
model_path = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', 'pretrain_768.pth')

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 创建模型并加载预训练权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)

if os.path.exists(model_path):
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights, strict=False)
    print(f"已加载预训练权重: {model_path}")
else:
    # 尝试加载 epoch checkpoint
    for epoch in [2, 1]:
        ckp_path = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', f'pretrain_epoch{epoch}.pth')
        if os.path.exists(ckp_path):
            weights = torch.load(ckp_path, map_location=device)
            model.load_state_dict(weights, strict=False)
            print(f"已加载 epoch {epoch} 权重: {ckp_path}")
            break
    else:
        print("未找到训练好的模型，使用未训练的模型（生成结果将无意义）")

model = model.half().to(device).eval()
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# ========== 自回归生成对话 ==========
# 预训练模型使用 BOS 标记 + 文本格式（非 chat template）
# 预训练模型只能做文本续写，SFT 之后才能做指令跟随对话

test_prompts = [
    "人工智能是",
    "深度学习在自然语言处理中的应用包括",
    "The transformer architecture is",
    "机器学习模型训练的关键步骤是",
]

print("\n预训练模型文本续写示例:")
print("=" * 60)

for prompt in test_prompts:
    # 预训练模型：直接用 BOS + 文本作为输入
    input_text = tokenizer.bos_token + prompt
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    
    # 自回归生成
    with torch.no_grad():
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
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
    
    print(f"输入: {prompt}")
    print(f"续写: {response}")
    print("-" * 60)

print("\n提示: 预训练模型只能做文本续写，无法进行指令跟随对话。")
print("经过 SFT（监督微调）后，模型将学会遵循对话格式，理解用户意图并给出针对性回答。")
```

## 实验结论

本次实验完整展示了从零开始预训练一个 64M 参数大语言模型的全流程。训练完成后，以下生成的文件将保存到数据目录：

- **模型文件**：
    - `<DATA_DIR>/models/minimind/pretrain/pretrain_768.pth` - 最终预训练权重（FP16）
    - `<DATA_DIR>/models/minimind/pretrain/pretrain_epoch*.pth` - 每 epoch 结束时的 checkpoint
    - `<DATA_DIR>/models/minimind/pretrain/pretrain_step*.pth` - 训练中间 checkpoint

预训练模型的 loss 从初始的约 8.76（均匀分布的交叉熵 $\ln(6400)$）下降到 3-4 左右，说明模型已经学会了中文和英文文本的基本统计规律。但预训练模型的能力存在明确边界：

1. **只能做文本续写，不能做对话**：预训练的目标是 next-token prediction，模型学会了根据前文生成合理的后续文本，但不知道"问答"这种交互格式。输入"你好"，模型可能续写为"你好，欢迎来到..."而不是回答"你好！有什么可以帮你的？"。SFT 阶段通过对话格式的示范训练，教会模型遵循 `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` 的格式进行多轮对话。

2. **知识是隐式的，无法精确检索**：预训练模型的知识编码在参数权重中，无法像数据库那样精确查询。模型可能"知道"某个事实（在相关提示下能正确续写），但无法保证事实的准确性，这是 LLM 幻觉问题的根源之一。

3. **没有工具使用、推理等高级能力**：预训练只赋予了语言建模的基础能力，工具调用、思维链推理、指令遵循等能力都需要后续的对齐训练（SFT、RLHF）来获得。

MiniMind 的 64M 参数规模在 LLM 领域属于极小模型，作为对比，GPT-2 Small 有 117M 参数，LLaMA-2 7B 有 70 亿参数，GPT-4 估计有万亿级参数。参数量的差距直接决定了模型能力的上限——64M 参数的模型不可能拥有 7B 模型那样丰富的世界知识和推理能力。但 MiniMind 的价值在于完整性和可复现性：在单卡 GPU 上 2-3 小时即可完成预训练，让每个人都能亲手体验训练语言模型的全过程，这是千亿参数模型无法提供的学习体验。

## 运行结果

预训练完成后，使用模型进行文本续写，一个实际的运行样例如下所示：

| 输入提示 | 模型续写 |
|---------|---------|
| 人工智能是 | 人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似方式做出反应的智能机器... |
| 深度学习在自然语言处理中的应用包括 | 深度学习在自然语言处理中的应用包括机器翻译、文本分类、情感分析、命名实体识别、问答系统等多个方向... |
| The transformer architecture is | The transformer architecture is a neural network design that relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions... |
| 机器学习模型训练的关键步骤是 | 机器学习模型训练的关键步骤是数据准备、特征工程、模型选择、训练优化和评估验证... |

预训练模型的续写结果在语法和语义上基本通顺，说明模型已经学会了语言的统计规律。但续写内容可能存在事实不准确、逻辑不连贯等问题，这正是 SFT 阶段需要解决的。
