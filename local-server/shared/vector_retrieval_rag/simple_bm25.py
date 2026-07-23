# SimpleBM25, HybridRetriever 定义
# 从文档自动提取生成

import numpy as np
import re
from collections import Counter, defaultdict, deque
from shared.vector_retrieval_rag import (MarkdownChunker, EmbeddingIndexer,

class SimpleBM25:
    """BM25 稀疏检索的简化实现

    为每个文档统计词频和计算 IDF，支持中文 bigram 和英文空格分词。
    使用标准 BM25 公式：
    score(d, q) = sum(IDF(t) * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_stats = []   # [{term: tf}, ...]
        self.idf = {}         # {term: idf}
        self.avgdl = 0
        self.N = 0

    def _tokenize(self, text: str) -> list[str]:
        """分词：中文 bigram + 英文/数字词"""
        tokens = []
        parts = re.split(r'([a-zA-Z0-9_]+)', text)
        for part in parts:
            if re.match(r'^[a-zA-Z0-9_]+$', part):
                if len(part) > 0:
                    tokens.append(part.lower())
            else:
                clean = re.sub(r'\s+', '', part)
                for i in range(len(clean) - 1):
                    tokens.append(clean[i:i+2])
        return tokens

    def fit(self, documents: list[str]):
        """构建 BM25 索引"""
        self.corpus = documents
        self.N = len(documents)
        doc_lengths = []

        for doc in documents:
            tokens = self._tokenize(doc)
            doc_lengths.append(len(tokens))
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self.doc_stats.append(dict(tf))

        self.avgdl = np.mean(doc_lengths) if doc_lengths else 1

        for term in set().union(*[d.keys() for d in self.doc_stats]):
            df = sum(1 for d in self.doc_stats if term in d)
            self.idf[term] = np.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """检索并返回 (doc_index, score) 列表"""
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.N)

        for term in q_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, stats in enumerate(self.doc_stats):
                if term not in stats:
                    continue
                tf = stats[term]
                dl = sum(stats.values())
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * numerator / denominator

        top_indices = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


class HybridRetriever:
    """混合检索引擎：稠密向量 + 稀疏 BM25，RRF 融合"""

    def __init__(self, dense_indexer: EmbeddingIndexer):
        self.dense = dense_indexer
        self.bm25 = None
        self.chunks = None

    def build(self, chunks: list[Chunk]):
        """构建稠密索引和稀疏索引"""
        self.chunks = chunks

        # 稠密索引
        self.dense.build_index(chunks)

        # 稀疏索引
        self.bm25 = SimpleBM25()
        self.bm25.fit([c.text for c in chunks])
        print(f"BM25 索引构建完成，词表大小: {len(self.bm25.idf)}")

    def search(self, query: str, top_k: int = 5,
               strategy: str = "hybrid") -> list[dict]:
        """统一检索接口

        Args:
            query: 查询文本
            top_k: 返回结果数
            strategy: 'dense' | 'sparse' | 'hybrid'
        """
        if strategy == "dense":
            return self._dense_search(query, top_k)
        elif strategy == "sparse":
            return self._sparse_search(query, top_k)
        else:
            return self._hybrid_search(query, top_k)

    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        results = self.dense.search(query, top_k=top_k)
        return [{"chunk": c, "score": s, "source": "dense"}
                for c, s in results]

    def _sparse_search(self, query: str, top_k: int) -> list[dict]:
        results = self.bm25.search(query, top_k=top_k)
        output = []
        for idx, score in results:
            output.append({
                "chunk": self.chunks[idx],
                "score": score,
                "source": "sparse"
            })
        return output

    def _hybrid_search(self, query: str, top_k: int,
                       k_rrf: int = 60) -> list[dict]:
        """RRF 融合：稠密 + 稀疏"""
        pool_size = max(top_k * 3, 10)
        dense_results = self.dense.search(query, top_k=pool_size)
        sparse_results = self.bm25.search(query, top_k=pool_size)

        rrf_scores = {}
        chunk_map = {}

        for rank, (chunk, _) in enumerate(dense_results):
            rrf_scores[chunk.chunk_id] = 1.0 / (k_rrf + rank + 1)
            chunk_map[chunk.chunk_id] = chunk

        for rank, (idx, _) in enumerate(sparse_results):
            chunk = self.chunks[idx]
            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0)
                + 1.0 / (k_rrf + rank + 1))
            chunk_map[chunk.chunk_id] = chunk

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        return [{"chunk": chunk_map[cid], "score": s, "source": "hybrid"}
                for cid, s in ranked]
