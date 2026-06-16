# VLMConfig, MiniMindVLM 定义
# 从文档自动提取生成

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from PIL import Image
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig, precompute_freqs_cis, F
from shared.vlm.mmvision_projector import MMVisionProjector
from transformers import SiglipVisionModel, SiglipImageProcessor
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

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
