# 工程实训：构建知识库问答系统

本次工程实训中，笔者将与你一同构建一个完整的知识库问答系统，覆盖从文档解析、向量索引构建、混合检索到 RAG 生成的完整链路。本次实训将使用真实的嵌入模型为文档生成语义向量，知识库直接使用 DMLA 项目自身的 Markdown 文档，在这些真实文档上验证检索与生成效果。

## 实验准备

在开始实验之前，请确保已完成以下准备工作：

1. 已下载 [BGE-small-zh-v1.5](https://modelscope.cn/models/BAAI/bge-small-zh-v1.5) 嵌入模型和 [Qwen3.5-0.8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B) 语言模型。
2. 已克隆 [DMLA 文档工程](https://github.com/fenixsoft/dmla)作为知识库。

```bash
# 选择 "下载模型" -> 选择 "BGE-small-zh-v1.5"
dmla model

# 如果使用 Docker 沙箱，由于沙箱中没有 GIT 工具，需要手动克隆
# 如果使用 Native 沙箱，且本机已部署 GIT 工具，验证代码会自动完成克隆
git clone --depth=1 https://github.com/fenixsoft/dmla.git
```

知识库的数据来源是本项目的文档目录。下面的代码从 GitHub 克隆 DMLA 仓库（浅克隆以节省时间），然后扫描 `docs/` 目录下的所有 Markdown 文件，过滤掉目录页、留言板等非内容页面后作为知识库文档集。注意，如果你使用的是 Docker 沙箱，由于镜像中没有部署 GIT 工具，需要使用上面命令，在 `DATA_DIR` 下手动完成项目克隆。

```python runnable gpuonly
import os
import subprocess

# 知识库存放路径（DATA_DIR 由 kernel 自动注入）
KB_DIR = os.path.join(DATA_DIR, 'datasets', 'rag-knowledge-base')
DOCS_DIR = os.path.join(KB_DIR, 'docs')

# 检查是否已克隆，避免重复下载
if not os.path.exists(os.path.join(KB_DIR, '.git')):
    print("正在克隆 DMLA 文档仓库（浅克隆）...")
    result = subprocess.run(
        ['git', 'clone', '--depth=1',
         'https://github.com/fenixsoft/dmla.git', KB_DIR],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"克隆失败: {result.stderr}")
        print("请检查网络连接后重试。")
        raise RuntimeError("仓库克隆失败")
    print("克隆完成。")
else:
    print("知识库已存在，跳过克隆。")

# 扫描 docs/ 下的 Markdown 文件
# 过滤掉非内容页面
EXCLUDE_FILES = {
    'README.md', 'boards.md', 'contents.md', 'todo.md', 'test.md',
    'settings-preview.md', 'rag-experiment.md'
}

doc_files = []
total_size = 0
for root, dirs, files in os.walk(DOCS_DIR):
    # 跳过 assets 目录
    dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'assets']
    for fname in files:
        if fname.endswith('.md') and fname not in EXCLUDE_FILES:
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)
            total_size += size
            relpath = os.path.relpath(fpath, DOCS_DIR)
            doc_files.append((relpath, size))

print(f"找到 {len(doc_files)} 篇文档")
print(f"总大小: {total_size / 1024:.0f} KB")

# 按目录分组统计
from collections import defaultdict
dir_counts = defaultdict(lambda: [0, 0])  # [count, total_size]
for path, size in doc_files:
    top_dir = path.split('/')[0]
    dir_counts[top_dir][0] += 1
    dir_counts[top_dir][1] += size

print("\n各目录文档分布:")
for d in sorted(dir_counts.keys()):
    count, size = dir_counts[d]
    bar = '█' * min(count, 30)
    print(f"  {d:<30s} {bar} ({count} 篇, {size/1024:.0f} KB)")

print(f"\n知识库路径: {DOCS_DIR}")
# 验证所需模型是否已下载
print("\n--- 模型检查 ---")
models_to_check = {
    "BGE 嵌入模型": os.path.join(DATA_DIR, "models", "pretrained", "bge-small-zh-v1.5"),
    "Qwen3.5-0.8B-Instruct": os.path.join(DATA_DIR, "models", "llm", "qwen3.5-0.8b-instruct"),
}
all_ready = True
for name, mpath in models_to_check.items():
    if os.path.isdir(mpath):
        has_config = os.path.exists(os.path.join(mpath, "config.json"))
        has_model = (os.path.exists(os.path.join(mpath, "model.safetensors")) or
                     os.path.exists(os.path.join(mpath, "pytorch_model.bin")))
        if has_config and has_model:
            size_mb = sum(
                os.path.getsize(os.path.join(mpath, f))
                for f in os.listdir(mpath)
                if os.path.isfile(os.path.join(mpath, f))
            ) / (1024 * 1024)
            print(f"  ✓ {name}: {mpath} ({size_mb:.0f} MB)")
        else:
            print(f"  ✗ {name}: 模型文件不完整")
            all_ready = False
    else:
        print(f"  ✗ {name}: 未找到，请运行 dmla data 下载")
        all_ready = False

if all_ready:
    print("\n所有模型就绪，可以进行后续实验。")
else:
    print("\n部分模型缺失，请先下载后再继续。")
```

## 第一阶段：文档解析与文本切分

文档解析是知识库构建的第一步。本实验的知识库文档使用 Markdown 格式编写，其中 `#` 开头的标题定义了文档的章节结构。解析器需要在提取文本的同时保留章节信息，为后续的检索结果溯源提供元数据支持。

文本切分策略直接影响检索质量。切分粒度过粗会导致检索结果中包含大量与查询无关的内容，增加 LLM 阅读负担和推理成本。切分粒度过细则破坏了段落内部的语义连贯性，可能使得完整的推理链条被拦腰截断。本阶段的工程决策围绕以下两点展开：

- **按章节边界切分而非按固定长度切分**。固定长度切分虽然简单，但经常在句子中间或公式中间截断，破坏语义完整性。章节边界是文档作者定义的自然断点，同一章节内的内容在语义上高度相关。本实验以 Markdown 标题为切分边界，每个标题下的文本段作为一个独立的文档块。

- **保留元数据用于引用溯源**。每个文档块都携带来源文件名、章节标题和块内位置索引。在 RAG 生成阶段，这些元数据将被注入提示词，使 LLM 能够在回答中标注信息来源地址。没有元数据支撑的引用溯源就只能给出模糊的"根据资料显示"，无法做到精确到具体章节。

```python runnable gpuonly extract-class="Chunk,MarkdownChunker"
import re
import os
from dataclasses import dataclass, field

@dataclass
class Chunk:
    """文档块数据结构"""
    chunk_id: str          # 唯一标识
    text: str              # 文档块文本
    source: str            # 来源文件的相对路径
    section: str           # 所属章节标题
    chunk_index: int       # 块在文档内的序号
    char_count: int = 0    # 字符数

    def __post_init__(self):
        self.char_count = len(self.text)

class MarkdownChunker:
    """Markdown 文档解析与切分器

    以 Markdown 标题（# 开头行）为边界将文档切分为多个块，
    每个块包含标题层级下全部文本。保留来源、章节等元数据。
    """

    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 3000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def parse_file(self, filepath: str) -> list[Chunk]:
        """解析单个 Markdown 文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        source = os.path.basename(filepath)
        return self._parse_content(content, source)

    def parse_directory(self, dirpath: str,
                        exclude_files: set = None) -> list[Chunk]:
        """解析目录下所有 Markdown 文件"""
        if exclude_files is None:
            exclude_files = set()

        all_chunks = []
        file_count = 0
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith('.md') or fname in exclude_files:
                continue
            fpath = os.path.join(dirpath, fname)
            if not os.path.isfile(fpath):
                continue
            chunks = self.parse_file(fpath)
            all_chunks.extend(chunks)
            file_count += 1

        # 递归处理子目录
        for sub in sorted(os.listdir(dirpath)):
            subpath = os.path.join(dirpath, sub)
            if os.path.isdir(subpath) and not sub.startswith('.') and sub != 'assets':
                sub_chunks = self.parse_directory(subpath, exclude_files)
                all_chunks.extend(sub_chunks)

        return all_chunks

    def _parse_content(self, content: str, source: str) -> list[Chunk]:
        """按标题边界切分文档内容"""
        chunks = []
        lines = content.split('\n')
        current_section = ''     # 当前章节标题
        current_lines = []       # 当前块内的文本行
        block_index = 0          # 文档内的绝对块序号
        seen_h1 = False          # 是否已经遇到文档一级标题
        in_code_block = False    # 是否在代码块内

        for line in lines:
            # 跟踪代码块边界（代码块内的 # 不是标题）
            if line.strip()[:3] == chr(96) * 3:
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            heading = re.match(r'^(#{1,4})\s+(.+)', line)

            if heading and not in_code_block and seen_h1:
                # 遇到新标题：保存当前块
                text = '\n'.join(current_lines).strip()
                if len(text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        chunk_id=f"{source}#{block_index}",
                        text=text,
                        source=source,
                        section=current_section or '引言',
                        chunk_index=block_index,
                    ))
                    block_index += 1

                current_section = heading.group(2)
                current_lines = []
            elif heading and not in_code_block and not seen_h1:
                # 文档一级标题：记录章节但不保存前面内容
                seen_h1 = True
                current_section = heading.group(2)
            else:
                current_lines.append(line)

        # 保存最后一个块
        if current_lines:
            text = '\n'.join(current_lines).strip()
            if len(text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    chunk_id=f"{source}#{block_index}",
                    text=text,
                    source=source,
                    section=current_section or '引言',
                    chunk_index=block_index,
                ))

        return chunks

# --- 运行解析 ---
KB_DIR = os.path.join(DATA_DIR, 'datasets', 'rag-knowledge-base')
DOCS_DIR = os.path.join(KB_DIR, 'docs')

EXCLUDE = {'README.md', 'boards.md', 'contents.md', 'todo.md',
           'test.md', 'settings-preview.md', 'rag-experiment.md'}

chunker = MarkdownChunker(min_chunk_size=100)
all_chunks = chunker.parse_directory(DOCS_DIR, exclude_files=EXCLUDE)

print(f"总计: {len(all_chunks)} 个文档块\n")

# 统计信息
source_counts = {}
total_chars = 0
for c in all_chunks:
    source_counts[c.source] = source_counts.get(c.source, 0) + 1
    total_chars += c.char_count

avg_chars = total_chars / len(all_chunks) if all_chunks else 0
print(f"总字符数: {total_chars:,}")
print(f"平均每块: {avg_chars:.0f} 字符")

# 按目录统计块数
dir_block_counts = {}
for c in all_chunks:
    top_dir = c.source.split('.')[0] if '.' in c.source else c.source
    # 尝试提取更具体的主题
    parts = c.source.replace('.md', '').split('-')
    tag = parts[0] if parts else 'other'
    dir_block_counts[tag] = dir_block_counts.get(tag, 0) + 1

print("\n按主题的块数分布（前 10）:")
for tag, count in sorted(dir_block_counts.items(),
                         key=lambda x: -x[1])[:10]:
    bar = '█' * min(count, 40)
    print(f"  {tag:<20s} {bar} ({count})")
```

## 第二阶段：嵌入生成与向量索引

有了文档块之后，需要将其转换为向量表示并建立索引以支持快速检索。这个阶段要选定嵌入模型及向量索引结构。嵌入模型的选择要在精度和效率之间做好权衡。参数量大的模型（如 BGE-large，约 1.3 GB）生成的向量质量更高，但编码速度慢、内存占用大。参数量小的模型（如 BGE-small，约 100MB）编码速度快一个数量级，在多数检索场景中的精度损失有限。本实验选用 `BAAI/bge-small-zh-v1.5` 作为嵌入模型，它在 MTEB 中文榜单上保持了不错的排名，体量小适合演示场景。向量索引结构方面，本实验使用 SciKit-Learn 的 NearestNeighbors 配合余弦距离实现暴力检索（Flat Index）。文档集产生的块数量在千个级别，暴力检索在这种规模下毫无压力。但当知识库规模扩大到万级以上文档时，就需要切换到 IVF 或 HNSW 索引来降低检索延迟。索引结构的升级路径在本系列[嵌入与向量检索](embedding-and-indexing.md)一章中有详细讨论。

另外，由于 BGE 模型的特殊要求是查询和文档使用不同的前缀提示。查询前添加 `"为这个句子生成表示以用于检索相关文章："`，文档前只需留空。这个设计来自 BGE 训练时对查询侧和文档侧做了显式区分，让模型学会两种不同的编码模式。如果给查询和文档使用相同的前缀，检索精度会有明显下降。

本阶段代码会在第四阶段中自动调用，无需手动执行。

```python runnable gpuonly extract-class="EmbeddingIndexer"
from __future__ import annotations
import os
import numpy as np
import os; os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertConfig
from sklearn.neighbors import NearestNeighbors

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
            score = 1.0 - dist   # 余弦距离转余弦相似度
            results.append((self.chunks[idx], float(score)))

        return results
```

## 第三阶段：稀疏检索与混合检索

向量检索擅长捕捉语义层面的相似性，但难以完成对术语的精确匹配。稀疏检索通过对关键词的精确匹配来填补这个缺口，混合检索则将两者的结果进行融合。本实验自行实现了一个简易的 BM25 检索器，分词采用字符级 [Bigram](../../language-models/architecture-basics/language-model-tokenization.md#n-gram-语言模型) 处理中文、空格切分处理英文的方式兼顾中英混合场景。用 RRF 融合将稠密和稀疏两路检索结果合并为单一排序。RRF的巧妙之处在于只依赖排名而非原始得分，天然规避了两种检索方式得分尺度不同的问题。公式为 $RRF(d) = \sum_{r \in R} 1/(k + rank_r(d))$，其中 $k=60$ 是经验常数，$R$ 是所有检索通路的集合。排名越靠前的文档获得越高的 RRF 分数。

```python runnable gpuonly extract-class="SimpleBM25,HybridRetriever"
from __future__ import annotations
import re
import numpy as np
from collections import defaultdict

# ============================================================
# 稀疏检索器（BM25 实现）
# ============================================================

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

# ============================================================
# 混合检索器
# ============================================================

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

# --- 构建索引 ---
from shared.vector_retrieval_rag import (MarkdownChunker, EmbeddingIndexer,
                                          HybridRetriever)

KB_DIR = os.path.join(DATA_DIR, 'datasets', 'rag-knowledge-base')
DOCS_DIR = os.path.join(KB_DIR, 'docs')
EXCLUDE = {'README.md', 'boards.md', 'contents.md', 'todo.md',
           'test.md', 'settings-preview.md', 'rag-experiment.md'}

BGE_MODEL_PATH = os.path.join(DATA_DIR, "models", "pretrained", "bge-small-zh-v1.5")

chunker = MarkdownChunker(min_chunk_size=100)
all_chunks = chunker.parse_directory(DOCS_DIR, exclude_files=EXCLUDE)
print(f"文档块: {len(all_chunks)}")

indexer = EmbeddingIndexer(model_name=BGE_MODEL_PATH)
retriever = HybridRetriever(indexer)
retriever.build(all_chunks)
```

## 第四阶段：RAG 对话推理

前面三个阶段完成了从文档解析到混合检索的全部准备工作。本阶段将加载 Qwen3.5-0.8B-Instruct 模型，将检索和生成串联起来，构建一个可以实际对话的 RAG 问答系统。一轮 RAG 对话的完整流程是用户输入问题后，系统先通过混合检索从文档块中召回最相关的内容，将这些内容连同引用标记拼接为增强提示词，然后交由 Qwen 模型生成带来源标注的回答。生成参数方面，`temperature=0.7` 和 `top_p=0.9` 在创造性与确定性之间取了一个适中的平衡。RAG 场景不同于创意写作，回答应忠实于检索到的文档，因此温度不宜过高。`repetition_penalty=1.15` 避免模型在引用文档内容时陷入重复。

运行下方代码块后，模型将加载到沙箱中。加载完成后，可在下方的对话框中输入问题，体验 RAG 问答系统。体验结束后，点击 Stop 按钮停止推理进程。

```python runnable gpuonly mode=chat
import os, re, json
import numpy as np
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertConfig, AutoModelForCausalLM

# ================================================================
# 如果前几个阶段的变量仍在作用域中则直接复用，否则重建
# ================================================================

KB_DIR = os.path.join(DATA_DIR, 'datasets', 'rag-knowledge-base')
DOCS_DIR = os.path.join(KB_DIR, 'docs')
EXCLUDE = {'README.md', 'boards.md', 'contents.md', 'todo.md',
           'test.md', 'settings-preview.md', 'rag-experiment.md'}

# 检查 retriever 是否已在作用域内
if 'retriever' not in globals():
    print("正在重建知识库索引...")

    # ---- 文档块数据结构 ----
    @dataclass
    class Chunk:
        chunk_id: str
        text: str
        source: str
        section: str
        chunk_index: int
        char_count: int = 0
        def __post_init__(self):
            self.char_count = len(self.text)

    # ---- 文档解析器 ----
    class MarkdownChunker:
        def __init__(self, min_chunk_size=100):
            self.min_chunk_size = min_chunk_size

        def parse_directory(self, dirpath, exclude_files=None):
            if exclude_files is None:
                exclude_files = set()
            all_chunks = []
            for fname in sorted(os.listdir(dirpath)):
                if not fname.endswith('.md') or fname in exclude_files:
                    continue
                fpath = os.path.join(dirpath, fname)
                if not os.path.isfile(fpath):
                    continue
                chunks = self._parse_file(fpath)
                all_chunks.extend(chunks)
            for sub in sorted(os.listdir(dirpath)):
                subpath = os.path.join(dirpath, sub)
                if os.path.isdir(subpath) and not sub.startswith('.') and sub != 'assets':
                    all_chunks.extend(self.parse_directory(subpath, exclude_files))
            return all_chunks

        def _parse_file(self, filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            source = os.path.basename(filepath)
            lines = content.split('\n')
            current_section, current_lines = '', []
            chunks, block_idx, seen_h1, in_code = [], 0, False, False
            for line in lines:
                if line.strip()[:3] == chr(96) * 3:
                    in_code = not in_code
                    current_lines.append(line)
                    continue
                heading = re.match(r'^(#{1,4})\s+(.+)', line)
                if heading and not in_code and seen_h1:
                    text = '\n'.join(current_lines).strip()
                    if len(text) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            chunk_id=f"{source}#{block_idx}", text=text,
                            source=source, section=current_section or '引言',
                            chunk_index=block_idx))
                        block_idx += 1
                    current_section = heading.group(2)
                    current_lines = []
                elif heading and not in_code and not seen_h1:
                    seen_h1 = True
                    current_section = heading.group(2)
                else:
                    current_lines.append(line)
            if current_lines:
                text = '\n'.join(current_lines).strip()
                if len(text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        chunk_id=f"{source}#{block_idx}", text=text,
                        source=source, section=current_section or '引言',
                        chunk_index=block_idx))
            return chunks

    # ---- 嵌入索引器 ----
    class EmbeddingIndexer:
        def __init__(self, model_name=None, device=None):
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
            self.embeddings = None
            self.chunks = None

        def _mean_pooling(self, hidden_states, attention_mask):
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        def _normalize(self, vecs):
            return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)

        def encode(self, texts, is_query=False, batch_size=16):
            if is_query:
                texts = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts]
            all_vecs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True,
                                        max_length=512, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    vecs = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                all_vecs.append(vecs.cpu().numpy())
            return self._normalize(np.concatenate(all_vecs, axis=0))

        def build_index(self, chunks):
            self.chunks = chunks
            texts = [c.text for c in chunks]
            print(f"为 {len(texts)} 个文档块生成嵌入...")
            self.embeddings = self.encode(texts, is_query=False)
            from sklearn.neighbors import NearestNeighbors
            self.nn_index = NearestNeighbors(
                n_neighbors=min(20, len(chunks)), metric='cosine')
            self.nn_index.fit(self.embeddings)
            print(f"嵌入维度: {self.embeddings.shape[1]}，索引就绪")

        def search(self, query, top_k=5):
            q_vec = self.encode([query], is_query=True)
            distances, indices = self.nn_index.kneighbors(q_vec)
            return [(self.chunks[idx], float(1.0 - d))
                    for d, idx in zip(distances[0], indices[0])]

    # ---- BM25 稀疏检索器 ----
    class SimpleBM25:
        def __init__(self, k1=1.5, b=0.75):
            self.k1, self.b = k1, b
            self.doc_stats, self.idf, self.avgdl, self.N = [], {}, 0, 0

        def _tokenize(self, text):
            tokens = []
            for part in re.split(r'([a-zA-Z0-9_]+)', text):
                if re.match(r'^[a-zA-Z0-9_]+$', part):
                    if part:
                        tokens.append(part.lower())
                else:
                    clean = re.sub(r'\s+', '', part)
                    for i in range(len(clean) - 1):
                        tokens.append(clean[i:i+2])
            return tokens

        def fit(self, documents):
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

        def search(self, query, top_k=5):
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
                    scores[i] += idf * tf * (self.k1+1) / (tf + self.k1*(1-self.b+self.b*dl/self.avgdl))
            top = np.argsort(-scores)[:top_k]
            return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    # ---- 混合检索器 ----
    class HybridRetriever:
        def __init__(self, dense_indexer):
            self.dense = dense_indexer
            self.bm25 = None
            self.chunks = None

        def build(self, chunks):
            self.chunks = chunks
            self.dense.build_index(chunks)
            self.bm25 = SimpleBM25()
            self.bm25.fit([c.text for c in chunks])
            print(f"BM25 索引构建完成，词表: {len(self.bm25.idf)}")

        def search(self, query, top_k=5, strategy="hybrid"):
            if strategy == "dense":
                return [{"chunk": c, "score": s, "source": "dense"}
                        for c, s in self.dense.search(query, top_k=top_k)]
            elif strategy == "sparse":
                return [{"chunk": self.chunks[idx], "score": s, "source": "sparse"}
                        for idx, s in self.bm25.search(query, top_k=top_k)]
            else:
                pool = max(top_k * 3, 10)
                dense_r = self.dense.search(query, top_k=pool)
                sparse_r = self.bm25.search(query, top_k=pool)
                rrf, cmap = {}, {}
                for rank, (chunk, _) in enumerate(dense_r):
                    rrf[chunk.chunk_id] = 1.0 / (60 + rank + 1)
                    cmap[chunk.chunk_id] = chunk
                for rank, (idx, _) in enumerate(sparse_r):
                    c = self.chunks[idx]
                    rrf[c.chunk_id] = rrf.get(c.chunk_id, 0) + 1.0 / (60 + rank + 1)
                    cmap[c.chunk_id] = c
                ranked = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
                return [{"chunk": cmap[cid], "score": s, "source": "hybrid"}
                        for cid, s in ranked]

    # ---- 执行重建 ----
    chunker = MarkdownChunker(min_chunk_size=100)
    all_chunks = chunker.parse_directory(DOCS_DIR, exclude_files=EXCLUDE)
    print(f"文档块总数: {len(all_chunks)}")

    BGE_MODEL_PATH = os.path.join(DATA_DIR, "models", "pretrained", "bge-small-zh-v1.5")
    indexer = EmbeddingIndexer(model_name=BGE_MODEL_PATH)
    retriever = HybridRetriever(indexer)
    retriever.build(all_chunks)

else:
    print(f"复用已有检索引擎（{len(retriever.chunks)} 个文档块）")

# ================================================================
# 加载 Qwen3.5-0.8B-Instruct 模型
# ================================================================

MODEL_PATH = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"加载 Qwen3.5-0.8B-Instruct（设备: {device}）...")
qwen_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
    device_map="auto" if device.type == 'cuda' else None,
    local_files_only=True,
)
if device.type == 'cpu':
    qwen_model = qwen_model.to(device)
qwen_model.eval()

param_count = sum(p.numel() for p in qwen_model.parameters()) / 1e6
print(f"Qwen 模型参数量: {param_count:.0f}M")
print("RAG 对话服务已就绪")

# ================================================================
# RAG 对话函数
# ================================================================

def chat(user_message, history=None):
    """RAG 对话：检索 → 构建提示词 → LLM 生成"""

    # 1. 混合检索
    hits = retriever.search(user_message, top_k=5, strategy="hybrid")

    # 2. 构建上下文
    n = len(hits)
    if n > 2:
        ordered = []
        front, back, toggle = 0, n - 1, True
        while front <= back:
            ordered.append(hits[front] if toggle else hits[back])
            if toggle:
                front += 1
            else:
                back -= 1
            toggle = not toggle
        hits = ordered

    context_parts = []
    total_chars = 0
    max_ctx = 3000

    for i, hit in enumerate(hits):
        c = hit["chunk"]
        ref = f"[来源{i+1}]（{c.source} / {c.section}）\n{c.text}"
        if total_chars + len(ref) > max_ctx:
            if total_chars == 0:
                ref_short = ref[:max_ctx]
                context_parts.append(ref_short)
            break
        context_parts.append(ref)
        total_chars += len(ref)

    context = "\n\n---\n\n".join(context_parts)

    # 3. 构建提示词
    prompt = (
        "基于以下参考资料回答问题。引用参考资料时使用 [来源N] 标记（N 为数字，无空格）。"
        "如果参考资料中没有足够信息，请直接说明'根据现有资料无法回答'。\n\n"
        f"---\n参考资料：\n{context}\n---\n\n"
        f"问题：{user_message}"
    )

    # 4. Qwen 生成
    messages = [{"role": "user", "content": prompt}]
    text = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=4096).to(qwen_model.device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=qwen_tokenizer.pad_token_id,
            eos_token_id=qwen_tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )

    response = qwen_tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )

    # 附上引用来源
    sources = []
    for i, hit in enumerate(hits):
        c = hit["chunk"]
        sources.append(f"[来源{i+1}] {c.source} / {c.section}")
    return response.strip() + "\n\n---\n" + "\n".join(sources)
```

::: details 运行上面代码后，点击这里进行对话
<ChatDemo />
:::


## 实训总结

本次实训构建了一个完整的知识库问答系统，以 DMLA 教程自身的约 90 篇 Markdown 文档为知识库，使用 Qwen3.5-0.8B-Instruct 作为生成模型，覆盖了从文档处理到真实 LLM 对话的全链路。RAG 服务启动后，与模型进行对话推理，实际运行样例如下：

![](./assets/result.gif)