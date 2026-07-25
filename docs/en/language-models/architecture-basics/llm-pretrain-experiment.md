# Transformer Model Pretraining Experiment

Through the past few chapters, we have learned about the principles of the Transformer architecture, various improvement techniques for modern LLMs, and how tokenizers work. This knowledge forms the theoretical foundation for understanding large language models, but theory alone makes it hard to truly appreciate the engineering choices involved in training a language model. In this experiment, we will train a Transformer-based large language model with approximately 64M parameters from scratch, going through the complete pipeline from data preparation to pretraining to inference and conversation — the full journey of a language model from zero to being able to speak.

> The training data for this experiment comes entirely from the open-source project [MiniMind](https://github.com/jingyaogong/minimind), and the model configuration ($dim=768$, $layers=8$) is consistent with the MiniMind-v3 mainline. We extend our gratitude to the project author ([@jingyaogong](https://github.com/jingyaogong)) for their excellent work.
>
> For demonstration purposes and due to differences in the technical stack, the author has rewritten all the training code based on the MiniMind source code and reorganized the program structure so that training can be completed directly within DMLA pages.

## Experiment Preparation

Before starting the experiment, please ensure you have [mounted the data directory](../../appendixes/sandbox.md#data-management) and downloaded the MiniMind pretraining corpus. You can do this automatically using the `DMLA-CLI` tool:

```bash
# Select "Download Dataset" -> Select "MiniMind Pretrain (LLM pretraining corpus)"
dmla data
```

This corpus contains pretraining text data (`pretrain_t2t_mini.jsonl`, approximately 1.2 GB) and the accompanying BPE tokenizer files (`tokenizer.json` and `tokenizer_config.json`). After downloading, verify the corpus and tokenizer files are complete:

```python gpuonly runnable
import os

# Corpus and tokenizer directory (DATA_DIR is automatically injected by the kernel)
# Docker mode: DATA_DIR='/data', Native mode: DATA_DIR='~/dmla-data'
data_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')

if os.path.exists(data_dir):
    print("Corpus directory exists")
    
    # Check pretraining corpus
    pretrain_file = os.path.join(data_dir, 'pretrain_t2t_mini.jsonl')
    if os.path.exists(pretrain_file):
        file_size = os.path.getsize(pretrain_file) / (1024 ** 3)
        print(f"Pretraining corpus: {file_size:.2f} GB")
        
        # Count lines in the first 10MB
        with open(pretrain_file, 'r', encoding='utf-8') as f:
            sample_lines = 0
            while f.tell() < 10 * 1024 * 1024:
                if not f.readline():
                    break
                sample_lines += 1
        print(f"First 10MB of corpus contains {sample_lines} samples")
    else:
        print("Pretraining corpus not found")
    
    # Check tokenizer files
    tokenizer_json = os.path.join(data_dir, 'tokenizer.json')
    tokenizer_config = os.path.join(data_dir, 'tokenizer_config.json')
    print(f"tokenizer.json: {'exists' if os.path.exists(tokenizer_json) else 'not found'}")
    print(f"tokenizer_config.json: {'exists' if os.path.exists(tokenizer_config) else 'not found'}")
else:
    print("Corpus directory not downloaded. Please run 'dmla data' to download the MiniMind pretraining corpus")
```

## Phase 1: Tokenizer and Data Loading

The model uses a BPE tokenizer with a vocabulary of 6400 tokens. Although the vocabulary is small, the BPE subword merging mechanism allows 6400 entries to cover common Chinese and English word combinations. Unknown words are decomposed into smaller subword units rather than being directly marked as `<unk>`. The vocabulary also includes dialogue control tokens such as `<|im_start|>` and `<|im_end|>`, which are essential for multi-turn conversation in the subsequent SFT phase. Although these tokens are not used during pretraining, they already occupy fixed positions in the vocabulary, ensuring vocabulary alignment between pretrained and SFT weights.

```python runnable gpu
from transformers import AutoTokenizer
import os

# Load tokenizer
tokenizer_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

# Basic vocabulary information
print(f"Vocabulary size: {len(tokenizer)}")
print(f"BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

# Tokenization demo
text = "Large language models are an important direction in artificial intelligence"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens, skip_special_tokens=True)

print(f"\nTokenization demo:")
print(f"Original: {text}")
print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Decoded: {decoded}")

# Compression ratio (characters / tokens)
compression = len(text) / len(tokens)
print(f"Compression ratio: {compression:.2f} chars/token")
```

The pretraining corpus is in JSONL format (one JSON object per line), where each sample contains a `text` field storing a continuous text segment. The goal of pretraining is to teach the model to predict the next token — given a sequence $w_1, w_2, ..., w_t$, the model learns to output $P(w_{t+1} | w_1, ..., w_t)$. Therefore, the data loading logic is relatively simple: tokenize each text into a token ID sequence, add BOS and EOS tokens, then pad to a fixed length.

Unlike the image data preprocessing in the [AlexNet experiment](../../deep-learning/convolutional-neural-network/alexnet-experiment.md), text data preprocessing has minimal overhead. Tokenization itself is a CPU-bound lookup and string matching operation, far faster than JPEG decoding and resizing. The pretraining dataset does not require optimization techniques like LMDB caching — it reads directly from the JSONL file line by line and tokenizes on the fly. Therefore, the following dataset code will be called during training and does not need to be executed manually.

```python runnable gpuonly extract-class="PretrainDataset"
import torch
from torch.utils.data import Dataset
import json

class PretrainDataset(Dataset):
    """
    Pretraining dataset: loads text from JSONL file and tokenizes into next-token prediction format
    
    Each sample format: {"text": "a piece of text"}
    Output: (input_ids, labels), where labels is input_ids shifted right by one position,
    used for computing cross-entropy loss for next-token prediction
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Pre-read all sample texts
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
        # Tokenize: truncate to max_length - 2 (reserve space for BOS and EOS)
        tokens = self.tokenizer(
            str(text), 
            add_special_tokens=False, 
            max_length=self.max_length - 2, 
            truncation=True
        ).input_ids
        
        # Add BOS and EOS tokens
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        
        # Pad to fixed length
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Labels are the same as input, padding positions are set to -100 (ignored by cross-entropy loss)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return input_ids, labels
```

::: info Pretraining Corpus Scale

The `pretrain_t2t_mini.jsonl` pretraining dataset contains approximately 1.27 million text samples, totaling about 1.2 GB. This is a lightweight corpus provided by the MiniMind project, suitable for quickly reproducing the pretraining pipeline on a single GPU. The full corpus `pretrain_t2t.jsonl` is about 10 GB, offering better training results but requiring more time. This experiment uses the lightweight corpus.

:::

## Phase 2: Model Definition

The model architecture follows mainstream design for modern small LLMs, with each component corresponding to concepts introduced in previous chapters. Here, we assemble these components into a complete causal language model, observe the loss curve during training, understand the boundaries of language capabilities acquired through pretraining, and appreciate the gap between the pretrained model and the model after SFT alignment (covered in the next experiment).

The core components of the model are:

| Component | Choice |
|-----------|--------|
| Normalization | [RMSNorm](architecture-evolution.md#rmsnorm) replacing [LayerNorm](./transformer-architecture.md#layer-normalization) |
| Positional Encoding | [RoPE (Rotary Position Embedding)](./transformer-architecture.md#rope-rotary-position-embedding) (with [YaRN extension](./architecture-evolution.md#yarn-position-encoding)) |
| Attention KV Cache Strategy | [GQA (Grouped Query Attention)](./architecture-evolution.md#gqa-grouped-query-attention) replacing [MHA (Multi-Head Attention)](./architecture-evolution.md#mha-multi-head-attention) |
| Attention Efficiency Strategy | Prefer [Flash Attention](./architecture-evolution.md#flash-attention) (depending on hardware support) |
| Activation Function | [SwiGLU](./architecture-evolution.md#swiglu) replacing [ReLU](../../deep-learning/neural-network-structure/activation-loss-functions.md#relu-and-its-variants) |
| Tokenizer | [BPE Tokenizer](language-model-tokenization.md#bpe) |
| Optimizer | [AdamW Adaptive Optimizer](../../deep-learning/neural-network-optimization/adaptive-optimizers.md#adamw) |

The core model configuration is:

| Config | Value | Description |
|--------|-------|-------------|
| `hidden_size` | 768 | Hidden layer dimension |
| `num_hidden_layers` | 8 | Number of Transformer layers |
| `num_attention_heads` | 8 | Number of query heads |
| `num_key_value_heads` | 4 | Number of KV heads (GQA, 2 groups shared) |
| `head_dim` | 96 | Dimension per head (768 / 8) |
| `intermediate_size` | 2432 | FFN intermediate dimension (SwiGLU) |
| `vocab_size` | 6400 | Vocabulary size |
| `tie_word_embeddings` | True | Tie word embedding and output head weights |

```python runnable gpuonly extract-class="MiniMindConfig, RMSNorm, precompute_freqs_cis, apply_rotary_pos_emb, repeat_kv, Attention, FeedForward, MiniMindBlock, MiniMindModel, MiniMindForCausalLM"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

class MiniMindConfig(PretrainedConfig):
    """Model configuration"""
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
    """RMS Normalization: more efficient than LayerNorm, eliminates mean computation"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim, end=32768, rope_base=1e6, rope_scaling=None):
    """Precompute cos and sin values for RoPE rotary position embeddings"""
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
    """Apply rotary position embeddings to queries and keys"""
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x, n_rep):
    """Repeat KV heads to match query head count (used for GQA inference)"""
    bs, slen, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_kv_heads, n_rep, head_dim).reshape(bs, slen, num_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
    """GQA (Grouped Query Attention)"""
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
        # QK-Norm: apply RMS normalization to queries and keys for improved training stability
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # KV Cache: concatenate historical KV during inference
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # Prefer Flash Attention (faster and more memory-efficient on GPU)
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
    """SwiGLU feed-forward network: gate and up paths, gating selects information channel"""
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
    """Single Transformer layer: Pre-Norm + Attention + FFN"""
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
    """Model body: word embedding + multiple Transformer layers + final normalization"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Precompute RoPE cos/sin buffers
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
        # Recompute RoPE buffers if lost due to meta device
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
    """Causal language model: used for pretraining and inference"""
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
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=512,
                 temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2,
                 streamer=None, use_cache=True, num_return_sequences=1,
                 do_sample=True, repetition_penalty=1.0, **kwargs):
        """Autoregressive generation: token-by-token sampling with top-k, top-p, and repetition penalty"""
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
            # Repetition penalty: reduce probability of previously seen tokens
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            # Top-k filtering
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            # Sampling or greedy selection
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

# Create model instance and count parameters
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"Vocabulary size: {config.vocab_size}")
print(f"Hidden size: {config.hidden_size}")
print(f"Number of Transformer layers: {config.num_hidden_layers}")
print(f"Attention heads: {config.num_attention_heads} (Q) / {config.num_key_value_heads} (KV)")
print(f"FFN intermediate size: {config.intermediate_size}")
print(f"Tie word embeddings: {config.tie_word_embeddings}")
```

## Phase 3: Pretraining

Pretraining is the most critical training stage for language models. The goal is to teach the model to predict the next token by learning from large amounts of text data. The training pipeline consists of three key steps: forward pass to compute predictions and loss, backward pass to compute gradients, and optimizer update to adjust parameters. These three steps are repeated millions of times across text samples. The model's loss drops from an initial value of about 8-9 (close to $\ln(6400) \approx 8.76$, the cross-entropy of a uniform distribution) to around 1.85, meaning the model has progressed from knowing nothing about text to being able to make reasonably good next-word predictions.

The key engineering decisions for this pretraining run are:

- **Mixed precision training**: Uses BF16 precision for forward and backward computation, reducing memory usage and accelerating computation while maintaining sufficient numerical precision to avoid training instability. Compared to FP16, BF16 has the advantage of sharing the same exponent bits as FP32 so it does not suffer from overflow or underflow issues, eliminating the need for GradScaler.
- **Cosine learning rate schedule**: The learning rate starts from its initial value and smoothly decays along a cosine curve to 10% of the initial value (i.e., $5 \times 10^{-5}$). Compared to fixed learning rates or step decay, cosine scheduling maintains a higher learning rate at the beginning for faster convergence and slowly reduces it later for fine-grained parameter adjustments. It is the most commonly used scheduling strategy in pretraining.
- **Gradient clipping**: Clips the global norm of gradients to within 1.0 to prevent gradient explosion from crashing training. The pretraining loss curve fluctuates significantly in the early stages, and gradient clipping is an important safety valve for ensuring training stability.
- **Periodic saving**: Saves model weights every 1000 steps, with the final weights saved after training completes. The saved weight files can be loaded directly during inference or used as initialization weights for the SFT phase.

::: info Training Estimate

The training corpus contains approximately 1.27 million samples, with a sequence length of 512, batch size of 32, and 2 epochs. On a single RTX 5080 GPU, this takes approximately 2.5 hours.

- Peak memory usage is approximately 7.4 GB (with Flash Attention hardware support) or 12.0 GB (without Flash Attention hardware support). GPUs with 8 GB of VRAM have a significant risk of OOM; GPUs with 12 GB or more and Flash Attention support can train stably.

- NVIDIA natively supports Flash Attention starting from the Ampere architecture (Compute Capability >= 8.0), i.e., RTX 30 series / A100 and later.

:::

::: details Training Memory Estimation

1. **Static Memory** (approximately 0.95 GB)

    Memory that stays resident throughout training, including model parameters, optimizer states, and gradients.

    | Item | Calculation | Memory |
    |------|-------------|--------|
    | Model Parameters (FP32) | 63.9M x 4 bytes | 244 MB |
    | AdamW First Moment $m$ (FP32) | 63.9M x 4 bytes | 244 MB |
    | AdamW Second Moment $v$ (FP32) | 63.9M x 4 bytes | 244 MB |
    | Gradients (FP32) | 63.9M x 4 bytes | 244 MB |
    | **Total** | | **0.95 GB** |

    Although training uses BF16 mixed precision, PyTorch's autocast mechanism only temporarily converts parameters to BF16 during forward and backward passes. The parameters themselves, optimizer states, and gradients are all stored in FP32 to maintain numerical precision.

2. **Forward Pass Activations** (approximately 3.12 GB / 7.12 GB)

    Backpropagation requires intermediate results from the forward pass to compute gradients. PyTorch stores these intermediate results in the computation graph during forward propagation. This is the largest contributor to memory usage and scales linearly with `batch_size` and `seq_len`. The activations saved per Transformer layer (using BF16, i.e., 2 bytes) are:

    | Activation Item | Calculation | Flash Attention | Standard Attention |
    |-----------------|-------------|:---------------:|:------------------:|
    | LayerNorm output x 2 | $B \times S \times D \times 2 \times 2$ | 48.0 MB | 48.0 MB |
    | Q, K, V projection results | $B \times S \times (n_q + 2 n_{kv}) \times d_h \times 2$ | 48.0 MB | 48.0 MB |
    | Attention output | $B \times S \times D \times 2$ | 24.0 MB | 24.0 MB |
    | Attention score matrix | $B \times n_q \times S \times S \times \mathbf{4}$ | — | 256 MB |
    | Softmax output | $B \times n_q \times S \times S \times \mathbf{4}$ | — | 256 MB |
    | SwiGLU gate | $B \times S \times I \times 2$ | 76.0 MB | 76.0 MB |
    | SwiGLU up | $B \times S \times I \times 2$ | 76.0 MB | 76.0 MB |
    | SiLU(gate) | $B \times S \times I \times 2$ | 76.0 MB | 76.0 MB |
    | SiLU(gate) x up | $B \times S \times I \times 2$ | 76.0 MB | 76.0 MB |
    | **Per Layer Total** | | **424 MB** | **936 MB** |
    | **8 Layers Total** | | **3.31 GB** | **7.31 GB** |

    Where $B=32$, $S=512$, $D=768$, $I=2432$, $n_q=8$, $n_{kv}=4$, $d_h=96$. The attention score matrix and Softmax output are stored in FP32 because Softmax numerical precision is critical for gradient computation. Flash Attention avoids writing these two matrices to memory through operator fusion, which is the core reason for its memory savings.

3. **Logits and Loss Computation** (approximately 1.17 GB)

    The model's output layer `lm_head` maps hidden states to probability distributions over the vocabulary. This tensor is another major memory consumer during training.

    | Item | Calculation | Memory |
    |------|-------------|--------|
    | logits (BF16) | $32 \times 512 \times 6400 \times 2$ | 200 MB |
    | `logits[..., :-1, :].contiguous()` (BF16) | $32 \times 511 \times 6400 \times 2$ | 200 MB |
    | cross_entropy internal upcast (FP32) | $32 \times 511 \times 6400 \times 4$ | 399 MB |
    | Softmax intermediate (FP32) | $32 \times 511 \times 6400 \times 4$ | 399 MB |
    | labels (int64) | $32 \times 511 \times 8$ | ~ 0 MB |
    | **Total** | | **1.17 GB** |

    `cross_entropy` automatically upcasts BF16 logits to FP32 during computation. Combined with the slice copy created by `contiguous()`, the same logits data exists in three copies in memory. This is the direct cause of the memory peak occurring in the loss computation stage. At this point, the original BF16 logits tensor, the BF16 slice copy, and the FP32 upcast version all exist simultaneously.

- **Peak Memory Summary**

    | Item | Flash Attention | Standard Attention |
    |------|:---------------:|:------------------:|
    | Static Memory | 0.95 GB | 0.95 GB |
    | Forward Pass Activations | 3.31 GB | 7.31 GB |
    | Logits + Loss | 1.17 GB | 1.17 GB |
    | CUDA Runtime | ~1.0 GB | ~1.0 GB |
    | Fragmentation Overhead (~15%) | ~0.96 GB | ~1.56 GB |
    | **Estimated Peak** | **7.4 GB** | **12.0 GB** |

    The above estimates are theoretical minimums. In practice, PyTorch's caching allocator reservation strategy (allocating memory in blocks and retaining free blocks for reuse), cuDNN workspaces, and CUDA runtime add an extra 0.5-1 GB. Therefore, with Flash Attention and `batch_size=32`, more than 8 GB of VRAM is required. On GPUs like the RTX 4060 (8 GB with approximately 7.6 GB usable), there is a significant chance of OOM. It is recommended to reduce `batch_size` to 16 and correspondingly adjust `accumulation_steps` to 16.

- **Adjustment Plan for Insufficient Memory**

    If GPU memory is insufficient for the default `batch_size=32`, you can reduce `batch_size` and proportionally increase gradient accumulation steps to maintain the same effective batch size (training results are equivalent, but each step clears and accumulates gradients more frequently, slowing down training):

    | batch_size | accumulation_steps | Effective Batch Size | Estimated Peak Memory |
    |:----------:|:------------------:|:--------------------:|:--------------------:|
    | 32 | 8 | 256 | ~7.2 GB |
    | 16 | 16 | 256 | ~4.7 GB |
    | 8 | 32 | 256 | ~3.5 GB |
    | 4 | 64 | 256 | ~2.9 GB |

    If the GPU does not support Flash Attention (PyTorch version < 2.0 or CUDA Compute Capability < 8.0), it is recommended to reduce `batch_size` to 4 or 8 to avoid OOM.

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

# Import progress reporting module
from dmla_progress import ProgressReporter

# Import model and dataset from shared modules
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig
from shared.llm.pretrain_dataset import PretrainDataset

# ========== Path Configuration (DATA_DIR is automatically injected by the kernel) ==========
DATA_PATH = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain', 'pretrain_t2t_mini.jsonl')
TOKENIZER_PATH = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
SAVE_DIR = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain')

# ========== Training Hyperparameters ==========
hidden_size = 768
num_hidden_layers = 8
max_seq_len = 512
batch_size = 32
learning_rate = 5e-4
num_epochs = 2
accumulation_steps = 8    # Gradient accumulation steps (effective batch_size = 32 x 8 = 256)
grad_clip = 1.0           # Gradient clipping threshold
log_interval = 100        # Log printing interval
save_interval = 1000      # Model save interval

# ========== 1. Initialize Environment ==========
progress = ProgressReporter(total_steps=10, description="Preparing training environment")
progress.update(0, message="Checking runtime environment...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Warning: No GPU detected, training will be very slow")

# Set random seed
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed(42)

# ========== 2. Load Tokenizer and Data ==========
progress.update(2, message="Loading tokenizer and training data...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
train_ds = PretrainDataset(DATA_PATH, tokenizer, max_length=max_seq_len)
print(f"Training samples: {len(train_ds):,}")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True
)
total_steps_per_epoch = len(train_loader)
total_steps = num_epochs * total_steps_per_epoch
print(f"Steps per epoch: {total_steps_per_epoch:,}")
print(f"Total training steps: {total_steps:,}")

# ========== 3. Create Model ==========
progress.update(4, message="Creating Transformer model...")
lm_config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
model = MiniMindForCausalLM(lm_config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

# ========== 4. Configure Training Components ==========
progress.update(6, message="Configuring optimizer and learning rate scheduler...")

# Mixed precision (BFloat16 does not require GradScaler)
device_type = "cuda" if device.type == "cuda" else "cpu"
autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, dtype=torch.bfloat16)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

def get_lr(current_step, total_steps, lr):
    """Cosine learning rate schedule: smooth decay, fine-grained adjustment in later stages"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

os.makedirs(SAVE_DIR, exist_ok=True)
progress.update(8, message="Training environment ready")

# ========== 5. Start Training ==========
progress.reset(total_steps=total_steps, description="Pretraining Transformer model")

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
        
        # Cosine learning rate scheduling
        lr = get_lr(global_step, total_steps, learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass (mixed precision)
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation + parameter update
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Log loss
        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss
        log_step_count += 1
        global_step += 1
        
        # Log printing
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
        
        # Periodic model saving
        if global_step % save_interval == 0:
            model.eval()
            save_path = os.path.join(SAVE_DIR, f'pretrain_step{global_step}.pth')
            state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
            torch.save(state_dict, save_path)
            print(f"  -> Model saved: step={global_step}, loss={current_loss:.4f}")
            model.train()
            del state_dict
        
        del input_ids, labels, res, loss
    
    # Save at the end of each epoch
    epoch_time = time.time() - epoch_start
    model.eval()
    epoch_save_path = os.path.join(SAVE_DIR, f'pretrain_epoch{epoch+1}.pth')
    state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, epoch_save_path)
    print(f"\nEpoch {epoch+1} completed, time: {epoch_time/60:.1f}min, model saved")
    model.train()
    del state_dict

# Save final model
final_path = os.path.join(SAVE_DIR, 'pretrain_768.pth')
state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
torch.save(state_dict, final_path)
progress.complete(message=f"Pretraining complete! Model saved to {final_path}")
print(f"\nFinal model saved: {final_path}")
```

## Phase 4: Inference and Conversation

After pretraining, the model has learned the statistical patterns of language and can predict the next token given preceding context. However, there is a fundamental difference between the capabilities of a pretrained model and a model aligned via SFT: the pretrained model has only learned text continuation -- given a text prefix, it generates subsequent text according to the statistical patterns in the training corpus. In contrast, an SFT model has learned to follow instructions and dialogue formats, understanding user intent and providing targeted responses.

Using a human learning analogy: pretraining is equivalent to extensively reading a large number of books, building a foundation of language and knowledge, but not yet knowing how to converse with people. SFT is equivalent to learning from dialogue demonstrations, knowing how to respond when asked a question. This experiment only completes the pretraining phase; SFT will be covered in the next chapter.

Inference with the pretrained model uses autoregressive generation. Given an input text token sequence, the model predicts the probability distribution for the next token one at a time, selects a token from the distribution using sampling strategies (Top-k, Top-p), appends it to the sequence, and continues predicting the next token using the new sequence, until an EOS token is generated or the maximum length is reached.

```python runnable gpuonly
import torch
import os
from transformers import AutoTokenizer

# Import model from shared modules
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig

# ========== Load Model and Tokenizer ==========
tokenizer_path = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
model_path = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', 'pretrain_768.pth')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Create model and load pretrained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)

if os.path.exists(model_path):
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights, strict=False)
    print(f"Loaded pretrained weights: {model_path}")
else:
    # Try loading epoch checkpoint
    for epoch in [2, 1]:
        ckp_path = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', f'pretrain_epoch{epoch}.pth')
        if os.path.exists(ckp_path):
            weights = torch.load(ckp_path, map_location=device)
            model.load_state_dict(weights, strict=False)
            print(f"Loaded epoch {epoch} weights: {ckp_path}")
            break
    else:
        print("No trained model found. Using untrained model (generation results will be meaningless)")

model = model.half().to(device).eval()
print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# ========== Autoregressive Generation ==========
# The pretrained model uses BOS token + text format (not chat template)
# The pretrained model can only do text continuation; instruction-following requires SFT

test_prompts = [
    "Artificial intelligence is",
    "Applications of deep learning in natural language processing include",
    "The transformer architecture is",
    "Key steps in machine learning model training are",
]

print("\nPretrained model text continuation examples:")
print("=" * 60)

for prompt in test_prompts:
    # Pretrained model: use BOS + text directly as input
    input_text = tokenizer.bos_token + prompt
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    
    # Autoregressive generation
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
    
    # Decode output (skip input portion)
    response = tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )
    
    print(f"Input: {prompt}")
    print(f"Continuation: {response}")
    print("-" * 60)
```

## Experimental Conclusions

This experiment demonstrates the complete pipeline for pretraining a 64M parameter large language model from scratch. After training, the following files are saved to the data directory:

- **Model Files**:
    - `<DATA_DIR>/models/minimind/pretrain/pretrain_768.pth` - Final pretrained weights (FP16)
    - `<DATA_DIR>/models/minimind/pretrain/pretrain_epoch*.pth` - End-of-epoch checkpoints
    - `<DATA_DIR>/models/minimind/pretrain/pretrain_step*.pth` - Intermediate training checkpoints

The pretrained model's loss drops from an initial value of about 8.76 (uniform distribution cross-entropy $\ln(6400)$) to around 1.85, indicating that the model has learned the basic statistical patterns of Chinese and English text. However, the pretrained model's capabilities have clear boundaries:

1. **Text continuation only, no conversational ability**: The pretraining objective is to predict the next word given preceding context. The model learns to generate reasonable continuations, but it does not understand the "question-and-answer" interaction format. Given the input "Hello," the model might continue with "Hello, welcome to..." rather than answering "Hello! How can I help you?" The SFT phase, through conversational format demonstrations, teaches the model to follow the `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` format for multi-turn conversations.

2. **Implicit knowledge without precise retrieval**: The pretrained model's knowledge is encoded in its parameter weights and cannot be queried with database-like precision. The model might know a certain fact (producing the correct continuation under the right prompt), but factuality cannot be guaranteed -- this is one of the root causes of LLM hallucination.

3. **No advanced capabilities like tool use or reasoning**: Pretraining only provides the basic language modeling foundation. Capabilities such as tool calling, chain-of-thought reasoning, and instruction following require subsequent alignment training (SFT, RLHF) to acquire.

The 64M parameter scale is extremely small by LLM standards. For comparison, GPT-2 Small has 117M parameters, LLaMA-2 7B has 7 billion parameters, and GPT-4 is estimated to have trillions of parameters. The parameter count directly determines the upper bound of model capability. A 64M parameter model cannot possess the world knowledge and reasoning abilities of a 7B model. However, this experiment focuses on completeness and reproducibility -- it can be completed in 2-3 hours on a single GPU, allowing everyone to experience the full process of training a language model firsthand, which is a learning experience that billion-parameter models cannot provide.

## Sample Output

After pretraining, using the model for text continuation, a sample run looks like this:

| Input Prompt | Model Continuation |
|-------------|-------------------|
| Artificial intelligence is | Artificial intelligence is a branch of computer science that seeks to understand the essence of intelligence and produce new intelligent machines that can react in ways similar to human intelligence... |
| Applications of deep learning in natural language processing include | Applications of deep learning in natural language processing include machine translation, text classification, sentiment analysis, named entity recognition, question answering systems, and many other areas... |
| The transformer architecture is | The transformer architecture is a neural network design that relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions... |
| Key steps in machine learning model training are | Key steps in machine learning model training are data preparation, feature engineering, model selection, training optimization, and evaluation validation... |

The pretrained model's continuations are generally fluent in terms of grammar and semantics, indicating that the model has learned the statistical patterns of language. However, the continuations may contain factual inaccuracies, logical inconsistencies, and other issues -- precisely what the SFT phase aims to address.
