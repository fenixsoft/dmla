# 视觉语言模型训练实验

在[预训练实验](../architecture-basics/llm-pretrain-experiment.md)中，我们训练了一个能够理解和生成文本的纯语言模型。但人类感知世界的方式远不止语言，视觉是我们获取信息最直接的途径。本次实验将在预训练语言模型的基础上，为其安装一双眼睛 —— 视觉编码器，让模型能够同时理解图像和文本，完成图像描述、视觉问答等多模态任务。本实验的代码参考 [MiniMind-V](https://github.com/jingyaogong/minimind-v) （模型架构与 MiniMind-V 保持一致，为适配 DMLA 网页训练环境重写了训练代码），数据集来自 [ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V)。

## 实验准备

在开始实验之前，请确保已完成[预训练实验](../architecture-basics/llm-pretrain-experiment.md)，并下载了视觉训练数据。通过 `DMLA-CLI` 工具完成数据下载：

```bash
# 选择 "下载数据集" -> 选择 "MiniMind Vision (VLM视觉训练数据)"
dmla data
```

该数据集包含三个部分：

| 文件 | 说明 | 大小 |
|------|------|------|
| `pretrain_i2t.parquet` | 视觉预训练数据（图像-描述对, 约 25 万条） | ~1.4 GB |
| `sft_i2t.parquet` | 视觉指令微调数据（多轮对话, 约 58 万条） | ~2.4 GB |
| `siglip2-base-p32-256-ve` | SigLIP 视觉编码器预训练权重 | ~181 MB |

> **数据说明**：对于 64M 参数量的 VLM 实验而言，原有的数据规模偏大，本实验从原始 ALLaVA-4V 数据集中随机抽取了 20% 的样本。由于随机采样打破了原始文件中相邻图片的压缩局部性，文件体积不会严格按 20% 线性缩小。如需完整数据集（原始 417 万条样本），可从 [MiniMind-V 官方仓库](https://github.com/jingyaogong/minimind-v) 获取。

下载完成后，验证数据完整性：

```python runnable
import os

data_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-vision')

if os.path.exists(data_dir):
    print("数据目录已存在")

    # 检查预训练数据
    pretrain_path = os.path.join(data_dir, 'pretrain_i2t.parquet')
    if os.path.exists(pretrain_path):
        size_gb = os.path.getsize(pretrain_path) / (1024 ** 3)
        print(f"预训练数据: {size_gb:.2f} GB")
    else:
        print("预训练数据未找到")

    # 检查SFT数据
    sft_path = os.path.join(data_dir, 'sft_i2t.parquet')
    if os.path.exists(sft_path):
        size_gb = os.path.getsize(sft_path) / (1024 ** 3)
        print(f"SFT数据: {size_gb:.2f} GB")
    else:
        print("SFT数据未找到")

    # 检查视觉编码器
    clip_dir = os.path.join(data_dir, 'siglip2-base-p32-256-ve')
    if os.path.exists(clip_dir):
        model_file = os.path.join(clip_dir, 'model.safetensors')
        config_file = os.path.join(clip_dir, 'config.json')
        print(f"SigLIP模型: {'已存在' if os.path.exists(model_file) else '未找到'}")
        print(f"SigLIP配置: {'已存在' if os.path.exists(config_file) else '未找到'}")
    else:
        print("SigLIP视觉编码器未找到")

    # 检查预训练LLM权重（来自预训练实验）
    llm_path = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', 'pretrain_768.pth')
    print(f"\n预训练LLM权重: {'已存在' if os.path.exists(llm_path) else '未找到（请先完成预训练实验）'}")
else:
    print("数据目录不存在，请运行 'dmla data' 下载 MiniMind Vision 数据集")
```

## 第一阶段：视觉编码器与投影层

视觉语言模型将图像转化为语言模型能理解的视觉 token，然后让语言模型像处理文本 token 一样处理这些视觉 token。这需要引入两个新组件：**视觉编码器**（Vision Encoder）将输入图像编码为视觉特征向量序列，**投影层**（Vision Projector）将视觉编码器的输出映射到语言模型的嵌入空间。VLM 的整体架构如下图所示：

```nn-arch width=780
name: MiniMind-VLM 架构
layout: horizontal

sections:
  - name: 视觉编码分支
    layers: [img_input, vision_enc, projector]
  - name: 文本处理分支
    layers: [text_input, embedding, transformer, lm_head, output]

layers:
  - {id: img_input, name: "输入图像", type: input, size: "256×256"}
  - {id: vision_enc, name: "视觉编码器", type: rnn, size: "SigLIP"}
  - {id: projector, name: "投影层", type: fc, size: "两层MLP"}
  - {id: text_input, name: "文本输入", type: input, size: "tokens"}
  - {id: embedding, name: "词嵌入层", type: fc, size: "词表=6400\ndim=768"}
  - {id: transformer, name: "Transformer", type: rnn, size: "8层 GQA\ndim=768"}
  - {id: lm_head, name: "输出头", type: fc, size: "lm_head"}
  - {id: output, name: "输出", type: output, size: "token 概率"}
```
*图：MiniMind-VLM 架构*

> 本实验的第一、第二阶段代码部分是纯用于教学讲解，其余代码会在第三阶段预训练和第四阶段的监督微调中被调用，它们均无需手动运行。

### 视觉编码器

SigLIP（Sigmoid Loss for Language-Image Pre-training）是[多模态大模型](./multimodal-llm.md)一章中介绍的 CLIP 家族成员。与原始 CLIP 使用 Softmax 对比损失不同，SigLIP 使用 Sigmoid 损失对图像-文本对进行训练，在 ImageNet 零样本分类和跨模态检索任务上表现优于同等规模的 CLIP 模型。本实验使用 [SigLIP2-Base-P32/256](https://huggingface.co/google/siglip2-base-patch32-256) 作为视觉编码器，其配置如下：

| 配置项 | 值 | 说明 |
|--------|------|------|
| `image_size` | 256 | 输入图像分辨率 |
| `patch_size` | 32 | 块大小 |
| `num_hidden_layers` | 12 | Transformer 层数 |
| `hidden_size` | 768 | 隐藏层维度 |
| `num_attention_heads` | 12 | 注意力头数 |
| `intermediate_size` | 3072 | FFN 中间维度 |

SigLIP 编码器输入 $256 \times 256$ 的图像，块大小 $32 \times 32$，每张图片输出 $8 \times 8 = 64$ 个块 token，每个 token 是 768 维向量，与语言模型的隐藏维度恰好一致。这并不是巧合，维度匹配可以简化后续投影层的实现。

```python runnable gpu
import torch
import os
from transformers import SiglipVisionModel, SiglipImageProcessor
from PIL import Image

# 加载视觉编码器
vision_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-vision', 'siglip2-base-p32-256-ve')
vision_model = SiglipVisionModel.from_pretrained(vision_dir)
processor = SiglipImageProcessor.from_pretrained(vision_dir)

# 统计视觉编码器参数量
total_params = sum(p.numel() for p in vision_model.parameters())
print(f"SigLIP 视觉编码器参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# 用随机图像演示编码过程
dummy_image = Image.fromarray(torch.randint(0, 255, (256, 256, 3), dtype=torch.uint8).numpy())
inputs = processor(images=dummy_image, return_tensors="pt")

with torch.no_grad():
    outputs = vision_model(**inputs)

print(f"输入图像: {dummy_image.size}")
print(f"视觉特征形状: {outputs.last_hidden_state.shape}")
print(f"  batch_size={outputs.last_hidden_state.shape[0]}")
print(f"  patch_tokens={outputs.last_hidden_state.shape[1]}")
print(f"  hidden_dim={outputs.last_hidden_state.shape[2]}")
```

### 投影层

视觉编码器的输出是 64 个 768 维的块 token，它们处于 SigLIP 的特征空间。语言模型的词嵌入也产生 768 维向量，但处于语言模型的语义空间。两个空间虽然维度相同，向量的分布和含义却完全不一样。投影层充当两个空间之间的翻译器，采用两层 MLP 结构（LayerNorm → Linear → GELU → Linear）。LayerNorm 先将视觉特征归一化，两次线性变换逐步将视觉特征从 SigLIP 的特征空间映射到语言模型的语义空间，GELU 激活函数在中间引入非线性，使投影层能够学习两个空间之间复杂的映射关系。

```python runnable gpu extract-class="MMVisionProjector"
import torch.nn as nn

class MMVisionProjector(nn.Module):
    """视觉-语言投影层：将视觉编码器的输出映射到语言模型的嵌入空间"""
    def __init__(self, in_dim=768, out_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)

# 创建投影层并统计参数量
proj = MMVisionProjector(768, 768)
total_params = sum(p.numel() for p in proj.parameters())
print(f"投影层参数量: {total_params:,} ({total_params/1e6:.2f}M)")
```

### 注入机制

投影后的视觉 token 需要被插入到文本 token 序列中，才能被 Transformer 层处理。token 注入机制是在词表中预留一个特殊标记 `<|image_pad|>`，数据集构建时将图像对应的 `<image>` 标记替换为 64 个连续的 `<|image_pad|>` token。模型在词嵌入之后、送入 Transformer 层之前，找到序列中的 `<|image_pad|>` 位置，将这些位置上的词嵌入替换为投影后的视觉特征。

```python runnable gpu extract-class="VLMConfig,MiniMindVLM"
import os
import torch
import torch.nn as nn
import warnings
from transformers import SiglipVisionModel, SiglipImageProcessor
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig, precompute_freqs_cis, F
from shared.vlm.mmvision_projector import MMVisionProjector

warnings.filterwarnings('ignore')

class VLMConfig(MiniMindConfig):
    """视觉语言模型配置，继承自语言模型配置，新增视觉相关参数"""
    model_type = "minimind-v"
    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        super().__init__(**kwargs)

class MiniMindVLM(MiniMindForCausalLM):
    """视觉语言模型：在语言模型基础上添加视觉编码器和投影层"""
    config_class = VLMConfig

    def __init__(self, config=None, vision_model_path=None):
        self.config = config or VLMConfig()
        super().__init__(self.config)
        # 加载视觉编码器和预处理器
        self.vision_encoder = None
        self.processor = None
        if vision_model_path and os.path.exists(vision_model_path):
            self.vision_encoder = SiglipVisionModel.from_pretrained(vision_model_path)
            self.processor = SiglipImageProcessor.from_pretrained(vision_model_path)
            # 冻结视觉编码器参数
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder = self.vision_encoder.eval()
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size)

    @staticmethod
    def image2tensor(image, processor):
        """将 PIL 图像转换为视觉编码器的输入张量"""
        if image.mode in ['RGBA', 'LA']:
            image = image.convert('RGB')
        return processor(images=image, return_tensors="pt")

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        """通过视觉编码器获取图像特征"""
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v
                           for k, v in image_inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    def inject_vision_tokens(self, tokens, h, vision_tensors=None, seqlen=512):
        """将投影后的视觉特征注入到词嵌入序列的 <|image_pad|> 位置"""
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker = self.config.image_ids[0]
        vf = vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None, pixel_values=None, **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 文本词嵌入
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # 视觉特征注入（仅首步推理时）
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                sample_val = next(iter(pixel_values.values()))
                if sample_val.ndim == 5:
                    bs, num = sample_val.shape[:2]
                    vision_tensors = self.vision_proj(
                        self.get_image_embeddings(
                            {k: v.flatten(0, 1) for k, v in pixel_values.items()},
                            self.vision_encoder
                        )
                    ).view(bs, num, self.config.image_token_len, -1)
                else:
                    vision_tensors = self.vision_proj(
                        self.get_image_embeddings(pixel_values, self.vision_encoder)
                    )
            else:
                vision_tensors = self.vision_proj(
                    self.get_image_embeddings(pixel_values, self.vision_encoder)
                )
            hidden_states = self.inject_vision_tokens(
                tokens=input_ids, h=hidden_states,
                vision_tensors=vision_tensors, seqlen=input_ids.shape[1]
            )

        # 重新计算可能丢失的 RoPE 缓冲区
        if self.model.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim, end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling
            )
            self.model.freqs_cos = freqs_cos.to(hidden_states.device)
            self.model.freqs_sin = freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        # Transformer 层处理
        presents = []
        for layer, past_key_value in zip(self.model.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, position_embeddings,
                past_key_value=past_key_value, use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.model.norm(hidden_states)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100
            )

        return MoeCausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=presents,
            hidden_states=hidden_states
        )

# 创建模型实例并统计参数量
vision_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-vision', 'siglip2-base-p32-256-ve')
vlm_config = VLMConfig(hidden_size=768, num_hidden_layers=8)
vlm_model = MiniMindVLM(vlm_config, vision_model_path=vision_dir)

total_params = sum(p.numel() for p in vlm_model.parameters())
trainable_params = sum(p.numel() for p in vlm_model.parameters() if p.requires_grad)
vision_params = sum(p.numel() for p in vlm_model.vision_encoder.parameters())
proj_params = sum(p.numel() for p in vlm_model.vision_proj.parameters())
llm_params = total_params - vision_params - proj_params

print(f"VLM 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"  视觉编码器 (冻结): {vision_params:,} ({vision_params/1e6:.2f}M)")
print(f"  投影层: {proj_params:,} ({proj_params/1e6:.2f}M)")
print(f"  语言模型: {llm_params:,} ({llm_params/1e6:.2f}M)")
print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
```

## 第二阶段：数据加载

相比纯语言模型，视觉语言模型的训练样本除了有文本，还包含一张或多张图像。本实验使用 Parquet 格式存储数据，每条记录包含 `conversations`（对话内容）和 `image_bytes`（图像原始字节）两个字段。训练分为两个阶段，对应两种不同的数据格式：

- **视觉预训练**（`pretrain_i2t.parquet`）：约 25 万条图像-描述对，目标是让模型学会将视觉信息与语言描述对齐。对话格式很简单，通常是用户请求描述图像、模型给出描述。
- **视觉指令微调**（`sft_i2t.parquet`）：约 58 万条多轮对话，目标是让模型学会根据图像回答各种问题。对话格式多样，涵盖视觉问答、图像分析、推理判断等任务。

```python runnable gpu extract-class="VLMDataset"
import json
import io
import torch
from PIL import Image
from torch.utils.data import Dataset
import pyarrow as pa
import pyarrow.parquet as pq
from shared.vlm.vlmconfig import MiniMindVLM

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
```

在纯语言模型的预训练中，预测目标是输入的下一个 token，标签与输入只差一个位置。但在视觉指令微调中，我们只希望让模型学习生成 AI 的回复，而不是学习预测用户的提问。`generate_labels()` 方法通过定位 `bos_token` + `assistant\n` 到 `eos_token` + `\n` 之间的 token，只暴露出这些位置对应的 token ID，其余位遮掩起来，标记为交叉熵损失自动忽略的标记值（PyTorch 默认是 -100）。

```python runnable gpu
import os
import json
import io
from PIL import Image
from transformers import AutoTokenizer
import pyarrow.parquet as pq

# 加载分词器和视觉预处理器
tokenizer_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

# 查看预训练数据样本
pretrain_path = os.path.join(DATA_DIR, 'datasets', 'minimind-vision', 'pretrain_i2t.parquet')
pf = pq.ParquetFile(pretrain_path)
print(f"预训练数据: {pf.metadata.num_rows:,} 条样本")

table = pf.read_row_group(0).slice(0, 2)
for i in range(2):
    conv = json.loads(table['conversations'][i].as_py())
    print(f"\n--- 预训练样本 {i} ---")
    for turn in conv:
        content = turn['content'][:80].replace('\n', ' ')
        print(f"  {turn['role']}: {content}...")

# 查看SFT数据样本
sft_path = os.path.join(DATA_DIR, 'datasets', 'minimind-vision', 'sft_i2t.parquet')
pf2 = pq.ParquetFile(sft_path)
print(f"\nSFT数据: {pf2.metadata.num_rows:,} 条样本")

table2 = pf2.read_row_group(0).slice(0, 2)
for i in range(2):
    conv = json.loads(table2['conversations'][i].as_py())
    print(f"\n--- SFT样本 {i} ---")
    for turn in conv:
        content = turn['content'][:80].replace('\n', ' ')
        print(f"  {turn['role']}: {content}...")

# 查看图像信息
img_bytes = table['image_bytes'][0].as_py()
if isinstance(img_bytes, list):
    img_bytes = img_bytes[0]
img = Image.open(io.BytesIO(img_bytes))
print(f"\n图像分辨率: {img.size}, 模式: {img.mode}")
```

## 第三阶段：视觉预训练

视觉预训练的目标是让投影层学会将视觉特征翻译为语言模型能理解的向量。给定一张图像，模型需要学会用语言描述图像内容。

在[多模态模型训练](multimodal-llm.md#训练多模态模型)中提到过，视觉编码器已经在数十亿图像-文本对上完成了预训练，具备视觉特征提取能力。语言模型也已通过文本预训练学会了语言的基本规律，两者都无需重新训练。视觉预训练阶段的任务是训练投影层，让投影层学会在两个已固化的空间之间建立映射。因此本阶段训练中，视觉编码器和语言模型的参数都是冻结的，只有投影层参数会被更新。这样既大幅减少可训练参数量（从 64M 降到约 1.2M），训练速度更快且显存占用更低，又避免训练初期投影层输出不稳定时对语言模型已学到的知识造成破坏。预训练阶段结束后，投影层已经能够产生合理的视觉 token，后续 SFT 阶段再解冻语言模型的首尾层，让模型在稳定的基础上精细调整。

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

from dmla_progress import ProgressReporter
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig
from shared.vlm.vlmconfig import VLMConfig, MiniMindVLM
from shared.vlm.mmvision_projector import MMVisionProjector
from shared.vlm.vlmdataset import VLMDataset

# ========== 路径配置 ==========
DATA_DIR_DATASETS = os.path.join(DATA_DIR, 'datasets')
VISION_DATA_DIR = os.path.join(DATA_DIR_DATASETS, 'minimind-vision')
TOKENIZER_PATH = os.path.join(DATA_DIR_DATASETS, 'minimind-pretrain')
VISION_MODEL_PATH = os.path.join(VISION_DATA_DIR, 'siglip2-base-p32-256-ve')
PRETRAIN_DATA = os.path.join(VISION_DATA_DIR, 'pretrain_i2t.parquet')
LLM_WEIGHT = os.path.join(DATA_DIR, 'models', 'minimind', 'pretrain', 'pretrain_768.pth')
SAVE_DIR = os.path.join(DATA_DIR, 'models', 'minimind-vlm', 'pretrain')

# ========== 训练超参数 ==========
hidden_size = 768
num_hidden_layers = 8
max_seq_len = 450
batch_size = 16
learning_rate = 4e-4
num_epochs = 1
accumulation_steps = 1
grad_clip = 1.0
log_interval = 100
save_interval = 2000

# ========== 1. 初始化环境 ==========
progress = ProgressReporter(total_steps=10, description="准备视觉预训练环境")
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

# ========== 2. 加载分词器和数据 ==========
progress.update(2, message="加载分词器和训练数据...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# ========== 3. 创建模型 ==========
progress.update(4, message="创建视觉语言模型...")
vlm_config = VLMConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
model = MiniMindVLM(vlm_config, vision_model_path=VISION_MODEL_PATH)

# 加载预训练语言模型权重
if os.path.exists(LLM_WEIGHT):
    weights = torch.load(LLM_WEIGHT, map_location=device)
    model.load_state_dict(weights, strict=False)
    print(f"已加载预训练LLM权重: {LLM_WEIGHT}")
else:
    print("警告: 未找到预训练LLM权重，将使用随机初始化（训练效果将大幅下降）")

# 冻结策略：完全冻结视觉编码器和语言模型，仅训练投影层
for name, param in model.named_parameters():
    if 'vision_proj' not in name:
        param.requires_grad = False

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

# ========== 4. 加载数据集 ==========
progress.update(6, message="加载视觉训练数据...")
train_ds = VLMDataset(
    PRETRAIN_DATA, tokenizer, preprocess=model.processor,
    max_length=max_seq_len, image_token_len=vlm_config.image_token_len
)
print(f"训练样本数: {len(train_ds):,}")

def vlm_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True,
    collate_fn=vlm_collate_fn
)
total_steps = num_epochs * len(train_loader)
print(f"每 epoch 步数: {len(train_loader):,}")
print(f"总训练步数: {total_steps:,}")

# ========== 5. 配置训练组件 ==========
progress.update(8, message="配置优化器...")
device_type = "cuda" if device.type == "cuda" else "cpu"
autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, dtype=torch.bfloat16)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

os.makedirs(SAVE_DIR, exist_ok=True)
progress.update(10, message="训练环境准备完成")

# ========== 6. 开始训练 ==========
progress.reset(total_steps=total_steps, description="视觉预训练 VLM 模型")

global_step = 0
for epoch in range(num_epochs):
    model.train()
    # 确保视觉编码器保持评估模式
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
    epoch_start = time.time()
    running_loss = 0.0
    log_step_count = 0

    for step, (input_ids, labels, pixel_values) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if isinstance(pixel_values, dict):
            pixel_values = {k: v.to(device) for k, v in pixel_values.items()}
        else:
            pixel_values = pixel_values.to(device)

        lr = get_lr(global_step, total_steps, learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = res.loss / accumulation_steps

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss
        log_step_count += 1
        global_step += 1

        if global_step % log_interval == 0:
            avg_loss = running_loss / log_step_count
            elapsed = time.time() - epoch_start
            eta_min = elapsed / max(global_step - epoch * len(train_loader), 1) * (total_steps - global_step) / 60
            print(f"Epoch[{epoch+1}/{num_epochs}] Step[{step+1}/{len(train_loader)}], "
                  f"loss: {avg_loss:.4f}, lr: {lr:.8f}, eta: {eta_min:.1f}min")
            progress.update(
                global_step,
                message=f"Epoch {epoch+1}, Step {step+1}, Loss={avg_loss:.4f}",
                extra_data={"loss": avg_loss, "lr": lr, "epoch": epoch + 1}
            )
            running_loss = 0.0
            log_step_count = 0

        if global_step % save_interval == 0:
            model.eval()
            state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()
                         if not k.startswith('vision_encoder.')}
            save_path = os.path.join(SAVE_DIR, f'pretrain_vlm_step{global_step}.pth')
            torch.save(state_dict, save_path)
            print(f"  -> 保存模型: step={global_step}, loss={current_loss:.4f}")
            model.train()
            if model.vision_encoder is not None:
                model.vision_encoder.eval()
            del state_dict

        del input_ids, labels, pixel_values, res, loss

    epoch_time = time.time() - epoch_start
    model.eval()
    state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()
                 if not k.startswith('vision_encoder.')}
    epoch_save_path = os.path.join(SAVE_DIR, f'pretrain_vlm_epoch{epoch+1}.pth')
    torch.save(state_dict, epoch_save_path)
    print(f"\nEpoch {epoch+1} 完成, 耗时 {epoch_time/60:.1f}min, 模型已保存")
    model.train()
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
    del state_dict

# 保存最终模型
final_path = os.path.join(SAVE_DIR, 'pretrain_vlm_768.pth')
state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()
             if not k.startswith('vision_encoder.')}
torch.save(state_dict, final_path)
progress.complete(message=f"视觉预训练完成！模型已保存到 {final_path}")
print(f"\n最终模型已保存: {final_path}")
```

::: info 训练预估

视觉预训练阶段约 25 万条样本，序列长度 450，批大小 16。由于只训练投影层（约 1.2M 参数），每步训练的显存占用远低于纯语言模型预训练，峰值显存占用约 3-5 GB，8 GB 显存的 GPU 即可运行。1 个 epoch 的耗时在 RTX 5080 上约 20 分钟。

:::

## 第四阶段：视觉指令微调

视觉预训练让模型学会了看图说话的基本能力，但预训练数据的对话格式单一（主要是图像描述），模型还无法应对多样的视觉问答场景。视觉指令微调（SFT）阶段使用包含多种任务类型的多轮对话数据，让模型学会根据图像回答各种问题。SFT 阶段与预训练阶段有两个区别：

1. **冻结策略不同**：SFT 阶段采用分层策略，除了训练投影层外，还解冻了语言模型的首层、末层，冻结了嵌入层（`embed_tokens`）、最终归一化层（`norm`）和中间层。首层是第一个处理视觉与文本混合特征的层，需要适应视觉 token 注入后的表示分布变化，末层负责生成更准确的回复。只解冻首层、末层是考虑到模型的语言主干仅 64M 参数，若全参解冻，LLM 原有的通用语言能力极易被图文任务所稀释。
2. **学习率不同**：预训练阶段学习率较高（4e-4），是因为投影层从零开始训练需要较大的学习率来快速收敛。SFT 阶段学习率较低（1e-5），是因为模型已经具备基本能力，只需要小幅调整参数来适应新的对话格式。

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

from dmla_progress import ProgressReporter
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig
from shared.vlm.vlmconfig import VLMConfig, MiniMindVLM
from shared.vlm.mmvision_projector import MMVisionProjector
from shared.vlm.vlmdataset import VLMDataset

# ========== 路径配置 ==========
DATA_DIR_DATASETS = os.path.join(DATA_DIR, 'datasets')
VISION_DATA_DIR = os.path.join(DATA_DIR_DATASETS, 'minimind-vision')
TOKENIZER_PATH = os.path.join(DATA_DIR_DATASETS, 'minimind-pretrain')
VISION_MODEL_PATH = os.path.join(VISION_DATA_DIR, 'siglip2-base-p32-256-ve')
SFT_DATA = os.path.join(VISION_DATA_DIR, 'sft_i2t.parquet')
PRETRAIN_VLM_WEIGHT = os.path.join(DATA_DIR, 'models', 'minimind-vlm', 'pretrain', 'pretrain_vlm_768.pth')
SAVE_DIR = os.path.join(DATA_DIR, 'models', 'minimind-vlm', 'sft')

# ========== 训练超参数 ==========
hidden_size = 768
num_hidden_layers = 8
max_seq_len = 768
batch_size = 16
learning_rate = 1e-5
num_epochs = 1
accumulation_steps = 1
grad_clip = 1.0
log_interval = 100
save_interval = 2000

# ========== 1. 初始化环境 ==========
progress = ProgressReporter(total_steps=10, description="准备视觉SFT环境")
progress.update(0, message="检测运行环境...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed(42)

# ========== 2. 创建模型并加载预训练权重 ==========
progress.update(3, message="创建模型并加载预训练权重...")
vlm_config = VLMConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
model = MiniMindVLM(vlm_config, vision_model_path=VISION_MODEL_PATH)

# 加载视觉预训练权重
weight_found = False
for weight_path in [PRETRAIN_VLM_WEIGHT]:
    if os.path.exists(weight_path):
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
        print(f"已加载视觉预训练权重: {weight_path}")
        weight_found = True
        break

if not weight_found:
    # 尝试加载 epoch checkpoint
    for epoch in [1]:
        ckp_path = os.path.join(DATA_DIR, 'models', 'minimind-vlm', 'pretrain', f'pretrain_vlm_epoch{epoch}.pth')
        if os.path.exists(ckp_path):
            weights = torch.load(ckp_path, map_location=device)
            model.load_state_dict(weights, strict=False)
            print(f"已加载 epoch {epoch} 权重: {ckp_path}")
            weight_found = True
            break
    if not weight_found:
        print("警告: 未找到视觉预训练权重，将使用未训练的模型")

# SFT 冻结策略：冻结视觉编码器，解冻投影层 + 首尾层
for name, param in model.named_parameters():
    if 'vision_encoder' in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# 冻结语言模型中间层，仅保留首层和末层可训练
last_idx = vlm_config.num_hidden_layers - 1
for name, param in model.model.named_parameters():
    if 'layers.0.' not in name and f'layers.{last_idx}.' not in name:
        param.requires_grad = False

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

# ========== 3. 加载数据 ==========
progress.update(6, message="加载SFT训练数据...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
train_ds = VLMDataset(
    SFT_DATA, tokenizer, preprocess=model.processor,
    max_length=max_seq_len, image_token_len=vlm_config.image_token_len
)
print(f"训练样本数: {len(train_ds):,}")

def vlm_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True,
    collate_fn=vlm_collate_fn
)
total_steps = num_epochs * len(train_loader)
print(f"每 epoch 步数: {len(train_loader):,}")
print(f"总训练步数: {total_steps:,}")

# ========== 4. 配置训练组件 ==========
progress.update(8, message="配置优化器...")
device_type = "cuda" if device.type == "cuda" else "cpu"
autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, dtype=torch.bfloat16)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

os.makedirs(SAVE_DIR, exist_ok=True)
progress.update(10, message="SFT环境准备完成")

# ========== 5. 开始训练 ==========
progress.reset(total_steps=total_steps, description="视觉SFT微调 VLM 模型")

global_step = 0
for epoch in range(num_epochs):
    model.train()
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
    epoch_start = time.time()
    running_loss = 0.0
    log_step_count = 0

    for step, (input_ids, labels, pixel_values) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if isinstance(pixel_values, dict):
            pixel_values = {k: v.to(device) for k, v in pixel_values.items()}
        else:
            pixel_values = pixel_values.to(device)

        lr = get_lr(global_step, total_steps, learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = res.loss / accumulation_steps

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss
        log_step_count += 1
        global_step += 1

        if global_step % log_interval == 0:
            avg_loss = running_loss / log_step_count
            elapsed = time.time() - epoch_start
            eta_min = elapsed / max(global_step - epoch * len(train_loader), 1) * (total_steps - global_step) / 60
            print(f"Epoch[{epoch+1}/{num_epochs}] Step[{step+1}/{len(train_loader)}], "
                  f"loss: {avg_loss:.4f}, lr: {lr:.8f}, eta: {eta_min:.1f}min")
            progress.update(
                global_step,
                message=f"Epoch {epoch+1}, Step {step+1}, Loss={avg_loss:.4f}",
                extra_data={"loss": avg_loss, "lr": lr, "epoch": epoch + 1}
            )
            running_loss = 0.0
            log_step_count = 0

        if global_step % save_interval == 0:
            model.eval()
            state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()
                         if not k.startswith('vision_encoder.')}
            save_path = os.path.join(SAVE_DIR, f'sft_vlm_step{global_step}.pth')
            torch.save(state_dict, save_path)
            print(f"  -> 保存模型: step={global_step}, loss={current_loss:.4f}")
            model.train()
            if model.vision_encoder is not None:
                model.vision_encoder.eval()
            del state_dict

        del input_ids, labels, pixel_values, res, loss

    epoch_time = time.time() - epoch_start
    model.eval()
    state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()
                 if not k.startswith('vision_encoder.')}
    epoch_save_path = os.path.join(SAVE_DIR, f'sft_vlm_epoch{epoch+1}.pth')
    torch.save(state_dict, epoch_save_path)
    print(f"\nEpoch {epoch+1} 完成, 耗时 {epoch_time/60:.1f}min, 模型已保存")
    model.train()
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
    del state_dict

# 保存最终模型
final_path = os.path.join(SAVE_DIR, 'sft_vlm_768.pth')
state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()
             if not k.startswith('vision_encoder.')}
torch.save(state_dict, final_path)
progress.complete(message=f"视觉SFT完成！模型已保存到 {final_path}")
print(f"\n最终模型已保存: {final_path}")
```

::: info 训练预估

SFT 阶段约 58 万条样本，序列长度 768，批大小 16。RTX 5080 上 1 个 epoch 的耗时约 1 小时。

峰值显存占用约 4-6 GB（BF16 混合精度），8 GB 以上显存的 GPU 可稳定训练。如果显存不足，可减小 `batch_size` 并相应增大 `accumulation_steps`。

SFT 阶段使用较低的学习率（1e-5），训练过程中 loss 的下降幅度不如预训练明显，但模型在对话质量上的提升是显著的。

:::

## 第五阶段：推理与视觉对话

训练完成后，模型具备了同时理解图像和文本的能力。推理时，用户输入一张图像和一段文本提示，模型先通过视觉编码器提取图像特征，再经投影层转换为视觉 token，与文本 token 一起送入 Transformer 层处理，自回归生成回复。

```python runnable gpuonly
import os
import glob
import torch
from PIL import Image
from transformers import AutoTokenizer

from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig
from shared.vlm.vlmconfig import VLMConfig, MiniMindVLM
from shared.vlm.mmvision_projector import MMVisionProjector

# ========== 加载模型 ==========
tokenizer_path = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
vision_model_path = os.path.join(DATA_DIR, 'datasets', 'minimind-vision', 'siglip2-base-p32-256-ve')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vlm_config = VLMConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindVLM(vlm_config, vision_model_path=vision_model_path)

# 加载 SFT 权重（优先）或预训练权重
weight_loaded = False
for weight_name in ['sft_vlm_768', 'pretrain_vlm_768']:
    weight_path = os.path.join(DATA_DIR, 'models', 'minimind-vlm',
                               'sft' if 'sft' in weight_name else 'pretrain',
                               f'{weight_name}.pth')
    if os.path.exists(weight_path):
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
        print(f"已加载权重: {weight_path}")
        weight_loaded = True
        break

if not weight_loaded:
    # 尝试加载 epoch checkpoint
    for stage in ['sft', 'pretrain']:
        for epoch in [2, 1]:
            ckp_path = os.path.join(DATA_DIR, 'models', 'minimind-vlm', stage, f'{stage}_vlm_epoch{epoch}.pth')
            if os.path.exists(ckp_path):
                weights = torch.load(ckp_path, map_location=device)
                model.load_state_dict(weights, strict=False)
                print(f"已加载 epoch {epoch} 权重: {ckp_path}")
                weight_loaded = True
                break
        if weight_loaded:
            break
    if not weight_loaded:
        print("未找到训练好的模型，使用未训练的模型（生成结果将无意义）")

model = model.half().to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# MiniMind tokenizer 未预设 chat_template，使用 ChatML 格式手动设置
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{ bos_token }}{{ message['role'] }}\n{{ message['content'] }}{{ eos_token }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ bos_token }}assistant\n{% endif %}"
)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# ========== 视觉对话推理 ==========
eval_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-vision', 'eval_images')
image_files = sorted(glob.glob(os.path.join(eval_dir, '*.jpg')) + glob.glob(os.path.join(eval_dir, '*.png')))

if not image_files:
    print(f"未找到评估图片，请确认 {eval_dir} 目录存在且包含图片")
else:
    print(f"找到 {len(image_files)} 张评估图片\n")

    image_special_tokens = model.config.image_special_token * model.config.image_token_len
    question = "请描述这张图片中的内容。"

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')

        # 展示图片
        print(f"{'='*60}")
        print(f"📷 图片: {img_name}")
        display(image.resize((256, 256), Image.LANCZOS))
        print(f"❓ 问题: {question}")

        # 构建输入
        prompt_text = f"{image_special_tokens}\n{question}"
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)

        # 视觉编码
        pixel_values = {k: v.to(device) for k, v in MiniMindVLM.image2tensor(image, model.processor).items()}

        # 自回归生成
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=pixel_values,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.85,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )

        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        print(f"💬 模型: {response}\n")
```

## 实验结论

本次实验在预训练语言模型的基础上，通过添加视觉编码器和投影层，构建了一个能够同时理解图像和文本的视觉语言模型。训练完成后，以下文件将保存到数据目录：

- **视觉预训练模型**：
    - `<DATA_DIR>/models/minimind-vlm/pretrain/pretrain_vlm_768.pth` - 视觉预训练最终权重
    - `<DATA_DIR>/models/minimind-vlm/pretrain/pretrain_vlm_epoch*.pth` - 每 epoch Checkpoint
    - `<DATA_DIR>/models/minimind-vlm/pretrain/pretrain_vlm_step*.pth` - 训练中间 Checkpoint

- **视觉指令微调模型**：
    - `<DATA_DIR>/models/minimind-vlm/sft/sft_vlm_768.pth` - SFT 最终权重
    - `<DATA_DIR>/models/minimind-vlm/sft/sft_vlm_epoch*.pth` - 每 epoch Checkpoint

实验展示了构建视觉语言模型的几个设计决策：

1. **模块化架构**：VLM 的架构可以拆分为视觉编码器、投影层、语言模型三个独立模块，每个模块都有明确的职责边界。视觉编码器负责特征提取，投影层负责空间对齐，语言模型负责理解与生成。这种模块化设计使得我们可以独立选择和替换每个组件，譬如将 SigLIP 替换为其他视觉编码器，或将语言模型替换为更大的模型。

2. **分阶段训练**：视觉预训练先让投影层学会将视觉特征翻译到语言空间，SFT 再在稳定的基础上微调整体能力。这种先对齐再微调的思路避免了训练初期投影层输出不稳定时对语言模型造成干扰，是小规模 VLM 训练中经过验证的有效策略。

3. **选择性冻结**：视觉编码器在整个训练过程中始终冻结，因为它已在海量数据上完成了视觉特征提取的预训练，重新训练不仅浪费算力，还可能在有限数据上过拟合。语言模型的冻结策略则根据训练阶段动态调整，预训练阶段完全冻结，SFT 阶段解冻首尾层。这种策略在可训练参数量和模型能力之间取得了平衡。

64M 语言模型 + 94.6M 视觉编码器的组合在 VLM 领域属于极小规模。作为参考，LLaVA-1.5 使用 13B 的 Vicuna（基于 LLaMA 微调）和 304M 的 CLIP ViT-L/14，GPT-4V 的视觉编码器规模更大。参数量的差距决定了模型能力的上限。本实验的模型能够完成简单的图像描述和基础问答，但无法进行复杂的视觉推理、OCR 识别或细粒度的图像分析。与纯语言模型的预训练实验一样，本次实验关注的是架构完整性和流程可复现性，在单卡 GPU 上体验模型从只能处理文字到可以看图说话的跨越，这是千亿参数模型难以提供的学习体验。

## 运行结果

使用训练完成的模型对 `eval_images` 目录下的 8 张图片进行图像描述推理，问题统一为"请描述这张图片中的内容"，模型回答的结果如下：

| 图片 | 模型回答 |
|------|---------|
| ![Astronaut-Space](./assets/eval_images/Astronaut-Space.jpg) | 这幅图片描绘了一个充满机械和复杂机械的宇宙场景，可能是一架现代飞船正在进行一项任务或维修工作。它似乎被设计成能够在高速运行中提供稳定性和控制能力的装置，有六个大灯和一个带有红色标记的标志来指示其位置。这些符号可能是表示方向或安全机制的指令或提示，如"我正在执行"。背景是蓝天、云朵和星星构成了黑暗且反射光线的混合物。在前景中有一个大型火 |
| ![Bicycle-Flowers](./assets/eval_images/Bicycle-Flowers.jpg) | 这幅图片描绘了一个室外的场景，有两个车厢和一个浴缸区域。左边的人是一辆白色的汽车，里面摆放着各种颜色的花朵，包括红色、粉色、黄色和绿色等。这个人正在享受一部新上映的电影或电视节目。右边的人是一个木制的马桶，有一根圆形的油漆刷，可以看到前方的水龙头或镜子。它看起来是由金属制成的，可能是玻璃或 |
| ![Chair-Elderly-Reading](./assets/eval_images/Chair-Elderly-Reading.jpg) | 这幅图片描绘了一个男人坐在一块木板上，头发梳理得像一个笔记本电脑或文件夹。他穿着深色西装、白衬衣和黑色外套。背景是一片郁绿的草地，暗示这是一个公园或度假胜地。窗户是柔软的木质表面，可以看到部分水滴，可能是由于云层所导致的水蒸气。天空多云，表明天气晴朗。画面中没有人物。 |
| ![Dog-Woman-Sea](./assets/eval_images/Dog-Woman-Sea.jpg) | 这幅图片描绘了一个宁静的海滩场景，天空晴朗蓝天，海浪轻轻拍打着岸边。它是一个阳光明媚的日子，带有白色和棕色调的渐变，沙滩上散落着各种大小的白色贝壳。在前景中，有一个坐在一个木制结构上的人站立，头顶是一面浅色的墙，与周围的植被形成鲜明对比。这个人穿着一件宽松的短袖T恤，短 |
| ![Panda-Grassland](./assets/eval_images/Panda-Grassland.jpg) | 这幅图片展示了一只小型的、类似于狗和羊群的陆地生物，它们的毛皮呈现出黑色，带有灰褐色斑点图案。这只狗有着明显的棕色头部和圆形的脸部，直立着。它的眼睛是深棕色的，眼周处可见一条粉红色细线，可能是由树叶或其他植物制成，表明了一种自然而可能性的栖息地。背景简单朴素，没有任何阴影 |
| ![Rainbow-Falls](./assets/eval_images/Rainbow-Falls.jpg) | 这幅图片展示了一片宁静的自然风景，从岩石上俯瞰着一座冰山。雪地呈现出深蓝色调，形成了温暖的光晕，表明光线进入其中。天空多云，暗示着晨曦或傍晚的天气条件。在远处，可以看到一群人站在一个高耸的湖泊中，水面平静无声。湖边有几艘船只，为整个场景增添了戏剧性的氛围。背景是一个模糊而 |
| ![city-traffic](./assets/eval_images/city-traffic.jpg) | 这幅图片描绘了一座繁华的城市街道，天空晴朗，暗示着多云的天气条件。在前景中，一辆汽车停在一个大光环上，照亮了道路和车库区域。车辆两侧是红色的摩托车和蓝色的电动车，配备了一个带有闪烁效果的灯泡，为驾驶者提供了一个独特而引人注目的视图。建筑物周围散布着各种建筑元素，包括高楼大厦、行李架和其他物品。远处 |
| ![dance](./assets/eval_images/dance.jpg) | 这幅图片展示了一位女性舞者在表演中，她站立着，双手合十。她穿着一件花裙子，肩上挂满了各种色彩鲜艳的花朵和彩色蝴蝶结。她的手臂伸出身体向前倾斜，头顶朝下看去，并面带微笑。她的姿势轻盈而流畅，双腿交叉放在身体两侧。她用右膝盖支撑自己的左脚，另外一只 |

从运行结果可以看出，64M 参数的 VLM 模型已经能够识别图像中的部分元素（如海滩、城市街道、舞者等），并生成结构完整的中文描述。但受限于参数规模，模型对图像细节的理解存在明显偏差，譬如将熊猫识别为"狗和羊群"、将自行车与花朵的场景描述为"汽车和浴缸"。这些偏差源于模型有限的参数规模，在更大规模的 VLM（如 LLaVA-1.5、Qwen-VL）中会显著改善。
