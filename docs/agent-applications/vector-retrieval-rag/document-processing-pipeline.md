# 文档处理流水线

## 核心问题

如何将非结构化的原始文档（PDF、网页、代码等）转换为检索系统可高效利用的结构化数据？文档切分策略、元数据管理和增量更新机制如何影响最终的检索质量？

## 目标读者

已掌握向量检索和混合检索原理，需要了解 RAG 系统上游——文档处理环节的工程师。

## 核心知识点

1. 文档解析：从多种格式提取文本内容
2. 文档切分策略：固定长度、语义切分、递归切分
3. 元数据管理：为检索提供过滤与排序依据
4. 嵌入生成与索引构建：从文本到可检索的向量
5. 增量更新与版本管理：知识库的持续维护

## 章节结构

### 1. 引言：垃圾进，垃圾出

- RAG 系统的质量上限由文档处理质量决定
- 文档处理是 RAG 系统中最容易被忽视但影响最大的环节
- 典型问题：切分不当导致语义断裂、元数据缺失导致过滤失效、更新不及时导致信息过时
- 文档处理流水线的整体架构
- 预计字数: 500

### 2. 文档解析

#### 2.1 常见文档格式与解析挑战

- PDF：布局复杂，表格、图片、多栏排版
- HTML/网页：导航栏、广告、正文提取
- Markdown：结构化标记，解析相对简单
- Office 文档（Word、PPT、Excel）：格式多样
- 代码文件：语法结构、注释与代码的区分

#### 2.2 解析工具与策略

- 文本提取：PyPDF2、pdfplumber、unstructured
- 布局感知解析：识别标题、段落、表格、列表
- OCR 处理：扫描件 PDF 的文字识别
- 多模态解析：图表描述、表格结构化

#### 2.3 解析质量保障

- 文本完整性检查：是否遗漏内容
- 编码一致性：UTF-8 统一处理
- 噪声过滤：页眉页脚、水印、版权声明

- 预计字数: 800

### 3. 文档切分策略

#### 3.1 为什么需要切分

- 嵌入模型的输入长度限制（通常 512 token）
- 长文档的嵌入会稀释关键信息
- 检索粒度：太粗则不精确，太细则丢失上下文

#### 3.2 固定长度切分

- 按字符数或 token 数切分
- 重叠窗口（Overlap）：相邻块之间保留重叠区域
- 参数选择：chunk_size 和 chunk_overlap
- 优势：实现简单，长度可控
- 劣势：可能在句子或段落中间切断

#### 3.3 语义切分

- 按自然边界切分：段落、章节、标题
- 语义边界检测：嵌入相似度突变点作为切分位置
- 优势：保持语义完整性
- 劣势：块长度不均匀，可能超出嵌入模型限制

#### 3.4 递归切分

- 策略：先按最大粒度（章节）切分，超长块再按次级粒度（段落）切分，依此类推
- 切分层级：章节 → 段落 → 句子 → 字符
- 优势：兼顾语义完整性和长度限制
- 劣势：实现复杂度较高

#### 3.5 切分策略对比

| 策略 | 语义完整性 | 长度均匀性 | 实现复杂度 | 适用场景 |
|-----|-----------|-----------|-----------|---------|
| 固定长度 | 低 | 高 | 低 | 通用 |
| 语义切分 | 高 | 低 | 中 | 结构化文档 |
| 递归切分 | 中高 | 中 | 高 | 复杂文档 |

#### 3.6 切分对检索质量的影响

- 切分粒度与召回率的关系：粒度越细，召回率越高，但上下文越少
- 切分粒度与精确率的关系：粒度越粗，精确率越高，但噪声越多
- 实验数据：不同 chunk_size 下的检索质量对比

- 预计字数: 1500

### 4. 元数据管理

#### 4.1 元数据的类型

- 文档级元数据：标题、作者、日期、来源、领域
- 块级元数据：所属章节、页码、位置信息
- 自定义元数据：权限标签、版本号、分类标签

#### 4.2 元数据在检索中的作用

- **过滤**（Filtering）：只检索特定时间范围、特定作者的文档
- **加权**（Boosting）：新文档权重更高，权威来源权重更高
- **分组**（Grouping）：同一文档的多个块在结果中只出现一次
- **溯源**（Provenance）：生成回答时引用来源

#### 4.3 元数据索引

- 标量索引：对元数据字段建立 B-Tree 或倒排索引
- 向量 + 标量联合查询：先过滤再检索 vs 先检索再过滤
- Milvus 的标量过滤机制

- 预计字数: 1000

### 5. 嵌入生成与索引构建

#### 5.1 嵌入生成

- 批量编码：利用 GPU 并行处理大量文档
- 输入预处理：添加指令前缀（如 BGE 的 "为这个句子生成表示"）
- 长文本处理：截断、分段编码后聚合

#### 5.2 索引构建流程

1. 文档解析 → 文本提取
2. 文本切分 → 文档块
3. 元数据标注 → 带元数据的文档块
4. 批量嵌入 → 向量
5. 索引构建 → 可检索的向量索引

#### 5.3 索引选择

- 小规模（< 100 万）：HNSW，高召回率
- 中规模（100 万 - 1000 万）：IVF-HNSW
- 大规模（> 1000 万）：IVF-PQ + 分布式

- 预计字数: 800

### 6. 增量更新与版本管理

#### 6.1 知识库的动态性

- 文档新增：新文档需要及时入库
- 文档修改：已入库文档内容变更
- 文档删除：过期或错误的文档需要移除
- 更新频率：实时 vs 批量 vs 定时

#### 6.2 增量更新策略

- 追加式更新：新文档直接插入索引
- 替换式更新：删除旧版本，插入新版本
- 双缓冲切换：构建新索引，原子切换

#### 6.3 版本管理

- 文档版本追踪：记录每个文档的版本历史
- 索引快照：定期保存索引快照，支持回滚
- 变更日志：记录所有增删改操作

#### 6.4 一致性保障

- 写入后立即可读（Strong Consistency）vs 最终一致性
- 索引重建期间的查询路由
- 分布式场景下的一致性挑战

- 预计字数: 1000

### 7. 本章小结

- 文档处理是 RAG 系统的基础，"垃圾进，垃圾出"
- 切分策略需要在语义完整性和检索粒度之间权衡
- 元数据为检索提供过滤、加权和溯源能力
- 增量更新机制保障知识库的时效性
- 预计字数: 300

## 可视化建议

1. **文档处理流水线流程图**：从原始文档到可检索索引的完整流程
2. **切分策略对比图**：固定长度、语义切分、递归切分的可视化对比
3. **切分粒度与检索质量曲线**：chunk_size 对 Recall@k 的影响
4. **增量更新流程图**：双缓冲切换的原子更新过程

## 代码示例建议

### 示例1：文档切分策略对比

```python runnable
import re

def fixed_length_split(text, chunk_size=200, overlap=50):
    """固定长度切分"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def semantic_split(text):
    """按段落切分（语义切分的简化版）"""
    paragraphs = re.split(r'\n\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]

def recursive_split(text, max_length=300, separators=['\n\n', '\n', '。', '，', '']):
    """递归切分"""
    if len(text) <= max_length:
        return [text]

    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ''
            for part in parts:
                if len(current) + len(part) + len(sep) <= max_length:
                    current += part + sep
                else:
                    if current:
                        chunks.append(current.strip())
                    current = part + sep
            if current:
                chunks.append(current.strip())
            # 递归处理超长块
            result = []
            for chunk in chunks:
                if len(chunk) > max_length:
                    result.extend(recursive_split(chunk, max_length, separators[1:]))
                else:
                    result.append(chunk)
            return result
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# 测试文本
sample_text = """Transformer 架构是自然语言处理领域的里程碑式突破。

自注意力机制是 Transformer 的核心组件。它允许模型在处理序列时，同时关注序列中的所有位置，而不像 RNN 那样只能顺序处理。自注意力的计算复杂度是 O(n^2)，其中 n 是序列长度。

位置编码解决了 Transformer 无法感知位置信息的问题。原始 Transformer 使用正弦-余弦位置编码，后续研究提出了旋转位置编码（RoPE）等改进方案。

Transformer 由编码器和解码器两部分组成。BERT 使用编码器部分，适合理解类任务；GPT 使用解码器部分，适合生成类任务。"""

# 对比三种切分策略
print("=== 固定长度切分 ===")
for i, chunk in enumerate(fixed_length_split(sample_text, 100, 20)):
    print(f"块{i+1} ({len(chunk)}字): {chunk[:50]}...")

print("\n=== 语义切分 ===")
for i, chunk in enumerate(semantic_split(sample_text)):
    print(f"块{i+1} ({len(chunk)}字): {chunk[:50]}...")

print("\n=== 递归切分 ===")
for i, chunk in enumerate(recursive_split(sample_text, 100)):
    print(f"块{i+1} ({len(chunk)}字): {chunk[:50]}...")
```

### 示例2：带元数据的文档索引构建

```python runnable
import numpy as np
import faiss

# 模拟文档块与元数据
chunks = [
    {"text": "Transformer 使用自注意力机制", "source": "nlp-basics.pdf", "page": 12, "section": "架构概述"},
    {"text": "BERT 是双向编码器模型", "source": "nlp-basics.pdf", "page": 25, "section": "预训练模型"},
    {"text": "GPT 使用自回归生成", "source": "nlp-basics.pdf", "page": 30, "section": "预训练模型"},
    {"text": "FAISS 支持多种索引类型", "source": "vector-search.pdf", "page": 5, "section": "工具介绍"},
    {"text": "HNSW 是图索引方法", "source": "vector-search.pdf", "page": 15, "section": "索引结构"},
    {"text": "RAG 结合检索与生成", "source": "rag-overview.pdf", "page": 1, "section": "引言"},
]

# 模拟嵌入（实际应使用嵌入模型）
np.random.seed(42)
d = 64
embeddings = np.random.randn(len(chunks), d).astype('float32')

# 构建 FAISS 索引
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# 模拟查询
query_embedding = np.random.randn(1, d).astype('float32')
D, I = index.search(query_embedding, k=3)

print("检索结果（带元数据）:")
for i, idx in enumerate(I[0]):
    chunk = chunks[idx]
    print(f"  第{i+1}名 (距离={D[0][i]:.4f}):")
    print(f"    文本: {chunk['text']}")
    print(f"    来源: {chunk['source']}, 第{chunk['page']}页, {chunk['section']}")
```

### 示例3：增量更新模拟

```python runnable
import numpy as np
import faiss

# 初始索引
np.random.seed(42)
d = 64
initial_data = np.random.randn(1000, d).astype('float32')
index = faiss.IndexHNSWFlat(d, 32)
index.add(initial_data)
print(f"初始索引大小: {index.ntotal}")

# 新增文档
new_data = np.random.randn(200, d).astype('float32')
index.add(new_data)
print(f"增量更新后索引大小: {index.ntotal}")

# 查询验证
query = np.random.randn(1, d).astype('float32')
D, I = index.search(query, k=5)
print(f"查询结果: Top-5 ID = {I[0]}")
```

## 练习题

1. 对同一文档使用不同切分策略，对比检索质量差异
2. 设计一个支持元数据过滤的检索系统，实现"只检索特定来源文档"的功能
3. 实现一个简单的增量更新机制，支持文档的新增、修改和删除
