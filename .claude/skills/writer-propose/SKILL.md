---
name: writer-propose
description: 根据探索结果创建文章大纲和任务列表，生成文章工作目录。类似于 OpenSpec 的 propose 阶段。
license: MIT
compatibility: None
metadata:
  author: ideaspaces
  version: "1.0"
---

根据探索结果创建文章大纲和任务列表。

**输入**: 用户的探索结论（可能在对话中），或用户直接描述想要写的文章。

**输出**: 文章工作目录，包含 `.article.yaml`、`outline.md`、`tasks.md`

---

## 步骤

### 1. 确认文章信息

如果对话中没有明确的探索结果，使用 AskUserQuestion 工具询问：
- 文章主题是什么？
- 目标读者是谁？
- 核心知识点有哪些？
- **是否为系列文章？**（多篇文章共用一个工作目录）

从用户描述中生成一个 kebab-case 的文章名称。

例如：`深入理解 Transformer 的 Attention 机制` → `transformer-attention`

**系列文章命名**：如 `微积分入门系列` → `calculus-series`

### 2. 创建文章工作目录

**单篇文章模式**：
```bash
mkdir -p articles/{article-name}
mkdir -p articles/{article-name}/draft
mkdir -p articles/{article-name}/draft/chapters  # 章节模式需要
mkdir -p articles/{article-name}/reviews
```

**系列文章模式**（多篇文章共用一个工作目录）：
```bash
mkdir -p articles/{series-name}
mkdir -p articles/{series-name}/draft      # 每篇独立草稿文件
mkdir -p articles/{series-name}/code       # 代码示例（可选，统一目录）
mkdir -p articles/{series-name}/assets     # 图片资源（可选，统一目录）
mkdir -p articles/{series-name}/reviews    # 校审报告
```

### 3. 创建 .article.yaml

使用模板 `articles/templates/.article.yaml`，填充以下字段：

**单篇文章**：
```yaml
name: {article-name}
title: "{文章标题}"
status: proposed
created: {YYYY-MM-DD}
updated: {YYYY-MM-DD}

audience:
  level: {beginner/intermediate/advanced}
  background: [前置知识列表]

config:
  needExperiments: {true/false}
  targetLength: {字数}
  categories: [分类列表]

reviewHistory: []

flags:
  needsUserConfirmation: false
  confirmationReason: null
```

**系列文章**（额外字段）：
```yaml
name: {series-name}
title: "{系列标题}"
status: proposed
created: {YYYY-MM-DD}
updated: {YYYY-MM-DD}
type: series              # 标识为系列

series:
  articles:
    - name: {article-1-name}
      title: "{文章1标题}"
      status: proposed
    - name: {article-2-name}
      title: "{文章2标题}"
      status: proposed
    # ... 其他文章
  totalArticles: {N}

audience:
  level: {beginner/intermediate/advanced}
  background: [前置知识列表]

config:
  needExperiments: {true/false}
  categories: [分类列表]

reviewHistory: []

flags:
  needsUserConfirmation: false
  confirmationReason: null
```

### 4. 创建 outline.md

根据探索结果生成文章大纲，使用模板 `articles/templates/outline.md`：

**单篇文章**：
```markdown
# 文章大纲

## 主题
{一句话描述}

## 目标读者
{目标读者描述}

## 核心知识点
1. {知识点1}
2. {知识点2}
3. {知识点3}

## 章节结构

### 1. 引言
- {要点1}
- {要点2}
- 预计字数: XXX

### 2. {章节名称}
- {要点1}
- {要点2}
- 预计字数: XXX

...

## 实验设计
- 实验1: {实验名称} - {验证什么}
- 实验2: {实验名称} - {验证什么}

## 练习题
- 练习1: {练习名称}
- 练习2: {练习名称}
```

**系列文章**：
```markdown
# 系列大纲

## 系列主题
{一句话描述}

## 目标读者
{目标读者描述}

## 知识点递进关系
```
文章1: {主题}
  ↓
文章2: {主题}
  ↓
...
```

## 文章列表

### 文章1: {标题}
- 核心知识点: {列表}
- 预计字数: XXX

### 文章2: {标题}
- 核心知识点: {列表}
- 前置: 文章1
- 预计字数: XXX

...

## 实验设计
- 文章1 实验: {实验名称}
- 文章2 实验: {实验名称}
...

## 练习题分布
- 文章1: {练习列表}
...
```

### 5. 创建 tasks.md

**根据总字数决定写作模式**：

| 总字数 | 写作模式 | 说明 |
|--------|----------|------|
| ≤ 5000字 | 单文件模式 | 所有章节写入 `draft/content.md`，一次性校审 |
| > 5000字 | 章节模式 | 每章独立文件，串行写作校审 |

根据大纲生成任务列表，使用模板 `articles/templates/tasks.md`：

**单文件模式（≤5000字）**：

```markdown
# 任务列表

## 状态概览

- 总任务: X
- 已完成: 0
- 进行中: 0
- 待开始: X

---

## 写作任务

- [ ] 写作-完整文章
  - 文件: draft/content.md
  - 预计字数: XXX
  - 模式: 单文件
  #owner: author

---

## 实验任务

- [ ] 实验-{实验名称}                                 #owner: experimenter
  - 沙箱验证: pending

---

## 校审任务

- [ ] 校审-第1轮                                      #owner: reviewer
  - 依赖: 写作任务完成

---

## 整合任务

- [ ] 整合发布                                        #owner: lead
  - 依赖: 校审通过
```

**章节模式（>5000字）**：

```markdown
# 任务列表

## 状态概览

- 总章节: X
- 已完成: 0
- 进行中: 0
- 待开始: X

---

## 章节写作任务（串行执行）

### 第1章：{章节名称}

- [ ] 写作-{章节名称}
  - 文件: draft/chapters/01-{name}.md
  - 预计字数: XXX
  - 内容:
    - {要点1}
    - {要点2}
- [ ] 校审-{章节名称}
  - 校审文件: reviews/review-ch01.md

### 第2章：{章节名称}

- [ ] 写作-{章节名称}
  - 文件: draft/chapters/02-{name}.md
  - 预计字数: XXX
  - 前置: 第1章完成
  - 内容:
    - {要点1}
    - {要点2}
- [ ] 校审-{章节名称}

...（后续章节类似）

---

## 实验任务（可并行）

- [ ] 实验-{实验名称}                                 #owner: experimenter
  - 沙箱验证: pending

---

## 整合任务

- [ ] 整合发布                                        #owner: lead
  - 依赖: 所有章节校审通过
```

**章节模式的关键规则**：
1. 每个章节独立成一个 markdown 文件
2. 章节必须串行执行：写完一章 → 校审通过 → 开始下一章
3. 写作下一章时，必须阅读大纲和前面所有章节，确保内容连贯
4. 文件命名规范：`{序号}-{name}.md`，如 `01-introduction.md`

**系列文章模式**（多篇文章共用一个工作目录）：

```markdown
# 任务列表

## 状态概览

- 总文章: N
- 已完成: 0
- 进行中: 0
- 待开始: N

---

## 文章写作任务（串行执行）

### 文章1：{标题}

- [ ] 写作-文章1-{标题}
  - 文件: draft/01-{name}.md
  - 预计字数: XXX
  - 内容:
    - {要点1}
    - {要点2}
- [ ] 校审-文章1
  - 校审文件: reviews/review-01.md

### 文章2：{标题}

- [ ] 写作-文章2-{标题}
  - 文件: draft/02-{name}.md
  - 预计字数: XXX
  - 前置: 文章1完成
  - 内容:
    - {要点1}
    - {要点2}
- [ ] 校审-文章2

...（后续文章类似）

---

## 实验任务（可并行）

- [ ] 实验-{实验名称}                 #owner: experimenter
  - 沙箱验证: pending

---

## 整合任务

- [ ] 整合发布系列                        #owner: lead
  - 依赖: 所有文章校审通过
```

**系列文章的关键规则**：
1. 每篇文章独立成一个 markdown 文件，存放于 `draft/` 目录
2. 文章必须串行执行：写完一篇 → 校审通过 → 开始下一篇
3. 写作下一篇文章时，必须阅读大纲和前面所有文章，确保内容连贯
4. 文件命名规范：`{序号}-{name}.md`，如 `01-limit.md`、`02-derivative.md`
5. 代码和图片资源统一存放在 `code/` 和 `assets/` 目录

### 6. 更新索引

更新 `articles/index.json`：

```json
{
  "articles": [
    {
      "name": "{article-name}",
      "title": "{文章标题}",
      "status": "proposed",
      "categories": [...],
      "created": "{YYYY-MM-DD}"
    }
  ],
  "stats": {
    "total": X,
    "proposed": X,
    ...
  }
}
```

---

## 输出示例

```
## 提案创建完成

**文章**: transformer-attention
**标题**: 深入理解 Transformer 的 Attention 机制

**工作目录**: articles/transformer-attention/
├── .article.yaml    # 文章元数据
├── outline.md       # 文章大纲
├── tasks.md         # 任务列表
├── draft/           # 草稿目录
└── reviews/         # 校审目录

**章节结构**:
1. 引言 (500字)
2. Attention 机制直觉理解 (600字)
3. Self-Attention 数学原理 (800字)
4. Multi-Head Attention (600字)
5. 代码实现 (500字)

**实验**: 2 个
**练习题**: 2 个

**下一步**: 运行 `/writer:compose` 开始写作。
```

---

## Guardrails

- 工作目录必须创建在 `articles/` 下，而不是 `docs/_work/`
- 必须创建完整的工作目录结构
- 章节任务要细化到可执行的粒度
- 实验任务要与知识点对应
- 更新索引文件确保一致性
- 如果文章名称已存在，询问用户是否继续或创建新文章
- **字数规则**：
  - 总字数 ≤ 5000字：使用单文件模式，输出到 `draft/content.md`
  - 总字数 > 5000字：使用章节模式，每章独立文件，串行写作校审
- **章节模式要求**：
  - 创建 `draft/chapters/` 目录存放章节文件
  - 文件命名：`{序号}-{kebab-case名称}.md`
  - 每章写完后立即校审，校审通过后才能开始下一章
  - 写作时必须阅读大纲和前面章节，确保连贯性
- **系列文章要求**：
  - 创建 `draft/` 目录存放每篇文章（与单篇文章一致）
  - 文件命名：`{序号}-{kebab-case名称}.md`，如 `01-limit.md`、`02-derivative.md`
  - 代码目录：`code/`（统一目录）
  - 图片目录：`assets/`（统一目录）
  - 每篇文章写完后立即校审，校审通过后才能开始下一篇