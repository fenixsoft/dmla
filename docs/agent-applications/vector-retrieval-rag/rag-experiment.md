# 工程实训：构建知识库问答系统

## 核心问题

如何将本系列文章的知识落地为一个完整的、可运行的知识库问答系统？从文档处理、索引构建、混合检索到 RAG 生成，经历完整的工程实现流程。

## 目标读者

已完成向量检索与 RAG 系列全部理论学习，希望动手实践构建完整系统的工程师。

## 实训目标

1. 构建一个完整的知识库问答系统，覆盖从文档到回答的全链路
2. 实现混合检索（稠密 + 稀疏）与重排序
3. 实现 RAG 生成，支持引用溯源
4. 对比不同配置下的检索与生成质量

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   知识库问答系统                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐    │
│  │ 文档处理  │──▶│ 索引构建  │──▶│ 混合检索引擎  │    │
│  │ 模块     │   │ 模块     │   │ (稠密+稀疏)  │    │
│  └──────────┘   └──────────┘   └──────┬───────┘    │
│                                       │             │
│                                       ▼             │
│                               ┌──────────────┐     │
│                               │   重排序模块   │     │
│                               └──────┬───────┘     │
│                                      │              │
│                                      ▼              │
│                               ┌──────────────┐     │
│                               │  RAG 生成模块  │     │
│                               │ (上下文注入    │     │
│                               │  + 引用溯源)   │     │
│                               └──────┬───────┘     │
│                                      │              │
│                                      ▼              │
│                               ┌──────────────┐     │
│                               │   评估模块    │     │
│                               └──────────────┘     │
└─────────────────────────────────────────────────────┘
```

## 实训步骤

### 步骤1：文档处理与切分（30 分钟）

#### 目标

将原始文档解析为可检索的文档块，带元数据标注。

#### 任务

1. 实现文档解析器，支持 Markdown 格式
2. 实现三种切分策略：固定长度、语义切分、递归切分
3. 为每个文档块标注元数据（来源、章节、位置）

#### 代码框架

```python runnable
import re
import json
from dataclasses import dataclass, asdict

@dataclass
class DocumentChunk:
    """文档块数据结构"""
    id: str
    text: str
    source: str
    section: str
    chunk_index: int
    metadata: dict

def parse_markdown(content: str, source: str) -> list[DocumentChunk]:
    """解析 Markdown 文档，提取文本和章节结构"""
    chunks = []
    current_section = "引言"
    chunk_index = 0

    lines = content.split('\n')
    current_text = []

    for line in lines:
        # 检测标题
        heading_match = re.match(r'^#+\s+(.+)', line)
        if heading_match:
            # 保存当前段落
            if current_text:
                text = '\n'.join(current_text).strip()
                if text:
                    chunks.append(DocumentChunk(
                        id=f"{source}-{chunk_index}",
                        text=text,
                        source=source,
                        section=current_section,
                        chunk_index=chunk_index,
                        metadata={"length": len(text)}
                    ))
                    chunk_index += 1
                current_text = []
            current_section = heading_match.group(1)
        else:
            current_text.append(line)

    # 保存最后一段
    if current_text:
        text = '\n'.join(current_text).strip()
        if text:
            chunks.append(DocumentChunk(
                id=f"{source}-{chunk_index}",
                text=text,
                source=source,
                section=current_section,
                chunk_index=chunk_index,
                metadata={"length": len(text)}
            ))

    return chunks

# 测试文档
test_doc = """# Transformer 架构

Transformer 是一种基于自注意力机制的深度学习架构，由 Google 在 2017 年提出。

## 自注意力机制

自注意力机制允许模型在处理序列时同时关注所有位置。其计算复杂度为 O(n^2)，其中 n 是序列长度。

Query、Key、Value 是自注意力的三个核心组件。Query 表示当前位置的关注点，Key 表示每个位置的特征，Value 表示每个位置的信息。

## 位置编码

由于 Transformer 没有循环结构，需要位置编码来提供位置信息。原始论文使用正弦-余弦位置编码。

## 编码器与解码器

编码器由多层自注意力和前馈网络组成。解码器在编码器基础上增加了交叉注意力层。

BERT 使用编码器部分，适合理解类任务。GPT 使用解码器部分，适合生成类任务。
"""

chunks = parse_markdown(test_doc, "transformer-intro.md")
for chunk in chunks:
    print(f"[{chunk.id}] 章节: {chunk.section}")
    print(f"  文本: {chunk.text[:60]}...")
    print(f"  长度: {chunk.metadata['length']}")
    print()
```

#### 验收标准

- 能够正确解析 Markdown 文档的章节结构
- 每个文档块包含完整的元数据
- 切分结果保存在结构化的数据结构中

---

### 步骤2：嵌入生成与索引构建（30 分钟）

#### 目标

为文档块生成向量嵌入，构建稠密索引和稀疏索引。

#### 任务

1. 使用嵌入模型为文档块生成向量表示
2. 构建 FAISS 稠密索引
3. 构建 TF-IDF 稀疏索引
4. 封装统一的检索接口

#### 代码框架

```python runnable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridIndex:
    """混合索引：稠密 + 稀疏"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.dense_index = None
        self.sparse_vectorizer = None
        self.sparse_matrix = None

    def build_dense_index(self, embeddings):
        """构建稠密索引"""
        import faiss
        d = embeddings.shape[1]
        self.dense_index = faiss.IndexHNSWFlat(d, 32)
        self.dense_index.add(embeddings.astype('float32'))

    def build_sparse_index(self):
        """构建稀疏索引"""
        texts = [chunk.text for chunk in self.chunks]
        self.sparse_vectorizer = TfidfVectorizer()
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(texts)

    def dense_search(self, query_embedding, top_k=10):
        """稠密检索"""
        D, I = self.dense_index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
        return [(self.chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]

    def sparse_search(self, query, top_k=10):
        """稀疏检索"""
        query_vec = self.sparse_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.sparse_matrix).flatten()
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

# 测试（使用模拟嵌入）
np.random.seed(42)
mock_embeddings = np.random.randn(len(chunks), 64).astype('float32')

index = HybridIndex(chunks)
index.build_dense_index(mock_embeddings)
index.build_sparse_index()

# 测试检索
query_embedding = np.random.randn(64).astype('float32')
dense_results = index.dense_search(query_embedding, top_k=3)
sparse_results = index.sparse_search("自注意力机制", top_k=3)

print("稠密检索结果:")
for chunk, score in dense_results:
    print(f"  [{score:.4f}] {chunk.text[:40]}...")

print("\n稀疏检索结果:")
for chunk, score in sparse_results:
    print(f"  [{score:.4f}] {chunk.text[:40]}...")
```

#### 验收标准

- 稠密索引构建成功，支持 Top-k 检索
- 稀疏索引构建成功，支持关键词检索
- 检索结果包含文档块和分数

---

### 步骤3：混合检索与重排序（30 分钟）

#### 目标

实现稠密检索与稀疏检索的结果融合，并使用交叉编码器重排序。

#### 任务

1. 实现 RRF 融合策略
2. 实现加权融合策略
3. 实现交叉编码器重排序
4. 对比不同融合策略的检索质量

#### 代码框架

```python runnable
import numpy as np

def rrf_fusion(dense_results, sparse_results, k=60):
    """RRF 融合"""
    scores = {}
    # 稠密检索排名
    for rank, (chunk, _) in enumerate(dense_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)
    # 稀疏检索排名
    for rank, (chunk, _) in enumerate(sparse_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)
    # 按分数排序
    chunk_map = {chunk.id: chunk for chunk, _ in dense_results + sparse_results}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(chunk_map[cid], score) for cid, score in ranked]

def weighted_fusion(dense_results, sparse_results, alpha=0.5):
    """加权融合"""
    # 分数归一化
    def normalize(results):
        scores = [s for _, s in results]
        min_s, max_s = min(scores), max(scores)
        return {chunk.id: (chunk, (s - min_s) / (max_s - min_s + 1e-8))
                for chunk, s in results}

    dense_norm = normalize(dense_results)
    sparse_norm = normalize(sparse_results)

    # 加权求和
    all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
    chunk_map = {}
    scores = {}
    for cid in all_ids:
        d_score = dense_norm.get(cid, (None, 0))[1]
        s_score = sparse_norm.get(cid, (None, 0))[1]
        scores[cid] = alpha * d_score + (1 - alpha) * s_score
        if cid in dense_norm:
            chunk_map[cid] = dense_norm[cid][0]
        else:
            chunk_map[cid] = sparse_norm[cid][0]

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(chunk_map[cid], score) for cid, score in ranked]

# 测试融合
# 复用步骤2的检索结果
dense_results = index.dense_search(query_embedding, top_k=5)
sparse_results = index.sparse_search("自注意力机制", top_k=5)

rrf_results = rrf_fusion(dense_results, sparse_results)
weighted_results = weighted_fusion(dense_results, sparse_results, alpha=0.6)

print("RRF 融合结果:")
for chunk, score in rrf_results[:3]:
    print(f"  [{score:.4f}] {chunk.text[:40]}...")

print("\n加权融合结果:")
for chunk, score in weighted_results[:3]:
    print(f"  [{score:.4f}] {chunk.text[:40]}...")
```

#### 验收标准

- RRF 融合结果正确，无需分数归一化
- 加权融合结果正确，分数已归一化
- 两种融合策略的结果可以对比

---

### 步骤4：RAG 生成与引用溯源（30 分钟）

#### 目标

实现 RAG 生成模块，支持上下文注入和引用溯源。

#### 任务

1. 设计提示词模板，支持引用标注
2. 实现上下文注入逻辑
3. 实现引用提取与验证
4. 生成带引用的回答

#### 代码框架

```python runnable
def build_rag_prompt(query, retrieved_chunks, max_context_length=2000):
    """构建 RAG 提示词"""
    context_parts = []
    current_length = 0

    for i, (chunk, score) in enumerate(retrieved_chunks):
        ref_text = f"[{i+1}] {chunk.text}（来源：{chunk.source}，{chunk.section}）"
        if current_length + len(ref_text) > max_context_length:
            break
        context_parts.append(ref_text)
        current_length += len(ref_text)

    context = "\n\n".join(context_parts)

    prompt = f"""基于以下参考资料回答问题。请在回答中使用 [1], [2] 等标注引用来源。
如果参考资料中没有足够信息，请明确说明"根据现有资料无法完整回答"。

参考资料：
{context}

问题：{query}

回答："""

    return prompt

def extract_citations(answer, chunks):
    """从回答中提取引用"""
    import re
    citations = re.findall(r'\[(\d+)\]', answer)
    cited_chunks = []
    for cite in citations:
        idx = int(cite) - 1
        if 0 <= idx < len(chunks):
            cited_chunks.append(chunks[idx][0])
    return cited_chunks

# 测试 RAG 生成
query = "自注意力机制的复杂度是多少？"
retrieved = rrf_results[:5]  # 使用 RRF 融合结果

prompt = build_rag_prompt(query, retrieved)
print("=== RAG 提示词 ===")
print(prompt[:500] + "...")

# 模拟 LLM 输出
mock_answer = """自注意力机制的计算复杂度是 O(n^2)，其中 n 是序列长度 [1]。

自注意力通过 Query、Key、Value 三个组件实现 [2]，允许模型同时关注序列中的所有位置。

参考文献：
[1] 来源：transformer-intro.md，自注意力机制
[2] 来源：transformer-intro.md，自注意力机制"""

cited = extract_citations(mock_answer, retrieved)
print("\n=== 引用溯源 ===")
for chunk in cited:
    print(f"  引用: {chunk.text[:50]}... (来源: {chunk.source})")
```

#### 验收标准

- 提示词包含检索上下文和引用说明
- 回答中包含引用标注
- 引用可追溯到具体的文档块

---

### 步骤5：效果评估与优化（30 分钟）

#### 目标

评估 RAG 系统的检索质量和生成质量，进行针对性优化。

#### 任务

1. 构建评估数据集（查询 + 相关文档标注）
2. 计算检索指标（Recall@k、MRR、NDCG）
3. 计算生成指标（忠实度、相关性、完整性）
4. 对比不同配置下的效果差异

#### 代码框架

```python runnable
import numpy as np

@dataclass
class EvalQuery:
    """评估查询"""
    query: str
    relevant_chunk_ids: list[str]
    expected_answer: str

# 构建评估数据集
eval_queries = [
    EvalQuery("自注意力机制的复杂度是多少", ["transformer-intro.md-1"], "O(n^2)"),
    EvalQuery("BERT 和 GPT 的区别", ["transformer-intro.md-4"], "BERT 使用编码器，GPT 使用解码器"),
    EvalQuery("位置编码的作用", ["transformer-intro.md-2"], "提供位置信息"),
    EvalQuery("Transformer 的核心组件", ["transformer-intro.md-0", "transformer-intro.md-1"], "自注意力机制"),
]

def evaluate_retrieval(eval_queries, search_fn, top_k=5):
    """评估检索质量"""
    recalls = []
    mrrs = []

    for eq in eval_queries:
        results = search_fn(eq.query, top_k)
        retrieved_ids = [chunk.id for chunk, _ in results]

        # Recall@k
        relevant = set(eq.relevant_chunk_ids)
        retrieved = set(retrieved_ids[:top_k])
        if relevant:
            recall = len(relevant & retrieved) / len(relevant)
            recalls.append(recall)

        # MRR
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant:
                mrrs.append(1 / (i + 1))
                break
        else:
            mrrs.append(0)

    return {
        "Recall@k": np.mean(recalls),
        "MRR": np.mean(mrrs),
    }

# 评估不同检索策略
def dense_search_fn(query, top_k):
    query_emb = np.random.randn(64).astype('float32')  # 实际应使用嵌入模型
    return index.dense_search(query_emb, top_k)

def sparse_search_fn(query, top_k):
    return index.sparse_search(query, top_k)

def hybrid_search_fn(query, top_k):
    dense = index.dense_search(np.random.randn(64).astype('float32'), top_k * 2)
    sparse = index.sparse_search(query, top_k * 2)
    return rrf_fusion(dense, sparse)[:top_k]

print("=== 检索质量评估 ===")
for name, fn in [("稠密检索", dense_search_fn), ("稀疏检索", sparse_search_fn), ("混合检索", hybrid_search_fn)]:
    metrics = evaluate_retrieval(eval_queries, fn)
    print(f"  {name}: Recall@5={metrics['Recall@k']:.4f}, MRR={metrics['MRR']:.4f}")
```

#### 验收标准

- 评估数据集包含查询、相关文档标注和期望回答
- 检索指标（Recall、MRR）计算正确
- 不同检索策略的对比结果有意义

---

### 步骤6：系统集成与端到端测试（30 分钟）

#### 目标

将所有模块集成为完整的知识库问答系统，进行端到端测试。

#### 任务

1. 封装完整的问答接口
2. 实现交互式查询循环
3. 进行端到端功能测试
4. 记录性能指标

#### 代码框架

```python runnable
class KnowledgeBaseQA:
    """知识库问答系统"""

    def __init__(self):
        self.index = None
        self.chunks = []

    def ingest(self, documents: dict[str, str]):
        """导入文档"""
        all_chunks = []
        for source, content in documents.items():
            chunks = parse_markdown(content, source)
            all_chunks.extend(chunks)

        self.chunks = all_chunks

        # 构建索引
        np.random.seed(42)
        mock_embeddings = np.random.randn(len(self.chunks), 64).astype('float32')

        self.index = HybridIndex(self.chunks)
        self.index.build_dense_index(mock_embeddings)
        self.index.build_sparse_index()

        return len(self.chunks)

    def query(self, question: str, top_k: int = 5, strategy: str = "hybrid") -> dict:
        """查询"""
        # 检索
        if strategy == "dense":
            query_emb = np.random.randn(64).astype('float32')
            results = self.index.dense_search(query_emb, top_k)
        elif strategy == "sparse":
            results = self.index.sparse_search(question, top_k)
        else:  # hybrid
            dense = self.index.dense_search(np.random.randn(64).astype('float32'), top_k * 2)
            sparse = self.index.sparse_search(question, top_k * 2)
            results = rrf_fusion(dense, sparse)[:top_k]

        # 生成提示词
        prompt = build_rag_prompt(question, results)

        return {
            "question": question,
            "retrieved_chunks": [(chunk.id, chunk.text[:50], score) for chunk, score in results],
            "prompt": prompt,
            "strategy": strategy
        }

# 端到端测试
qa_system = KnowledgeBaseQA()

# 导入文档
doc_count = qa_system.ingest({"transformer-intro.md": test_doc})
print(f"已导入 {doc_count} 个文档块\n")

# 查询测试
questions = [
    "自注意力机制的复杂度是多少？",
    "BERT 和 GPT 的区别是什么？",
    "位置编码有什么作用？"
]

for q in questions:
    result = qa_system.query(q, strategy="hybrid")
    print(f"问题: {result['question']}")
    print(f"策略: {result['strategy']}")
    print(f"检索到 {len(result['retrieved_chunks'])} 个文档块:")
    for chunk_id, text_preview, score in result['retrieved_chunks'][:3]:
        print(f"  [{score:.4f}] {chunk_id}: {text_preview}...")
    print()
```

#### 验收标准

- 系统支持文档导入、索引构建、查询检索的完整流程
- 不同检索策略可切换
- 端到端查询响应正常

---

## 进阶挑战

完成基础实训后，可以尝试以下进阶任务：

### 挑战1：实现 Self-RAG

- 让模型自主判断是否需要检索
- 实现反思标记机制
- 对比基础 RAG 与 Self-RAG 的生成质量

### 挑战2：实现迭代检索

- 支持复杂问题的子问题分解
- 实现多轮检索与信息累积
- 设计合理的迭代终止条件

### 挑战3：实现重排序模块

- 集成交叉编码器重排序
- 对比粗筛与精排的检索质量差异
- 测量重排序模块的延迟开销

### 挑战4：性能优化

- 实现查询缓存，避免重复计算
- 优化批量嵌入生成，利用 GPU 并行
- 实现索引的增量更新

## 实训总结

本实训构建了一个完整的知识库问答系统，涵盖了 RAG 的核心链路：

1. **文档处理**：解析、切分、元数据标注
2. **索引构建**：稠密索引（FAISS）+ 稀疏索引（TF-IDF）
3. **混合检索**：RRF 融合 + 加权融合
4. **RAG 生成**：上下文注入 + 引用溯源
5. **效果评估**：检索指标 + 生成指标

通过实训，你应当能够：

- 理解 RAG 系统的完整工作流程
- 实现文档处理到生成回答的全链路
- 对比不同检索策略的效果差异
- 评估和优化 RAG 系统的质量
