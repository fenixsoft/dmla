# EmbeddingIndexer 定义
# 从文档自动提取生成

from __future__ import annotations
import json as _json
import numpy as np
import os
import os; os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertConfig

class EmbeddingIndexer:
    """嵌入生成与向量索引

    使用 BGE-small-zh 模型将文本转换为语义向量，
    通过 scikit-learn NearestNeighbors 构建可检索的向量索引。
    """

    def __init__(self, model_name: str = None,
                 device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"加载嵌入模型 {model_name}（设备: {device}）...")
        import json as _json
        with open(os.path.join(model_name, 'config.json')) as _f: _cfg = _json.load(_f)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_config(BertConfig.from_dict(_cfg)).to(device)
        state_dict = torch.load(os.path.join(model_name, 'pytorch_model.bin'),
                               map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.embeddings = None    # numpy 数组 [N, dim]
        self.chunks = None        # 对应的文档块列表
        self.nn_index = None      # sklearn NearestNeighbors

    def _mean_pooling(self, hidden_states, attention_mask):
        """对 token 级别的隐藏状态做平均池化得到句子向量"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            hidden_states.size()).float()
        masked = hidden_states * mask_expanded
        summed = masked.sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _normalize(self, vecs):
        """L2 归一化"""
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)

    def encode(self, texts: list[str], is_query: bool = False,
               batch_size: int = 16) -> np.ndarray:
        """将文本列表编码为向量

        Args:
            texts: 文本列表
            is_query: 是否为查询（BGE 模型查询需特殊前缀）
            batch_size: 批次大小
        """
        if is_query:
            texts = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts]

        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                vecs = self._mean_pooling(outputs.last_hidden_state,
                                          inputs['attention_mask'])
            vecs = vecs.cpu().numpy()
            all_vecs.append(vecs)

        result = np.concatenate(all_vecs, axis=0)
        return self._normalize(result)

    def build_index(self, chunks: list[Chunk]):
        """为文档块生成嵌入并构建向量索引"""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        print(f"为 {len(texts)} 个文档块生成嵌入...")
        self.embeddings = self.encode(texts, is_query=False)

        self.nn_index = NearestNeighbors(
            n_neighbors=min(20, len(chunks)),
            metric='cosine'
        )
        self.nn_index.fit(self.embeddings)

        print(f"嵌入维度: {self.embeddings.shape[1]}")
        print(f"索引构建完成，共 {len(chunks)} 个向量")

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """向量检索：返回 top_k 个最相关的文档块"""
        if self.nn_index is None:
            raise RuntimeError("索引未构建，请先调用 build_index()")

        query_vec = self.encode([query], is_query=True)
        distances, indices = self.nn_index.kneighbors(query_vec)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            score = 1.0 - dist / 2.0   # 余弦距离转余弦相似度
            results.append((self.chunks[idx], float(score)))

        return results
