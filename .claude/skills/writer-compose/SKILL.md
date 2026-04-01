---
name: writer-compose
description: 执行文章写作任务。启动 Agent Team 协调 Author、Reviewer、Experimenter 完成写作。这是整个系统的核心 skill。
license: MIT
compatibility: Requires Claude Code Agent Teams feature enabled. Set CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 in settings.json
metadata:
  author: ideaspaces
  version: "1.0"
---

执行文章写作任务，协调 Agent Team 完成写作和校审。

**输入**: 文章名称（可选，从上下文推断）

**输出**: 完成的文章草稿

**重要**: 此 skill 需要启用 Agent Teams 功能。在 settings.json 中设置：
```json
{ "env": { "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1" } }
```

---

## 步骤

### 1. 确定文章

如果未指定文章名称：
- 检查 `articles/` 目录下的文章
- 找到 `status: proposed` 或 `status: composing` 的文章
- 如果有多个，使用 AskUserQuestion 让用户选择
- 如果没有，提示用户先运行 `/writer:propose`

### 2. 读取文章状态

读取以下文件了解当前状态：
- `articles/{name}/.article.yaml` - 元数据（检查 `config.targetLength` 或 `type: series`）
- `articles/{name}/outline.md` - 大纲
- `articles/{name}/tasks.md` - 任务列表

**判断写作模式**：

| 类型 | 总字数 | 写作模式 | 输出目录 |
|------|--------|----------|----------|
| 单篇文章 | ≤ 5000字 | 单文件模式 | `draft/content.md` |
| 单篇文章 | > 5000字 | 章节模式 | `draft/chapters/` |
| 系列文章 | - | 系列模式 | `drafts/` |

**系列文章标识**：`.article.yaml` 中 `type: series`

### 3. 更新状态

将 `.article.yaml` 中的 `status` 更新为 `composing`

### 4. 执行写作

根据 tasks.md 中的写作模式执行：

#### 单文件模式（≤5000字）

直接写作完整文章到 `draft/content.md`，完成后进行校审。

```markdown
工作流程：
1. 阅读 outline.md 了解完整结构
2. 写作所有章节到 draft/content.md
3. 完成后进行校审
4. 根据反馈修改（最多3轮）
```

#### 章节模式（>5000字）

**串行执行**：每章独立文件，写完一章、校审通过后才开始下一章。

```markdown
工作流程（循环执行每个章节）：

对于每个章节 i (i = 1, 2, ..., N):

1. **写作前准备**：
   - 阅读 outline.md 了解本章要点
   - 阅读前面所有章节（如有）
   - 理解上下文，确保内容连贯

2. **写作章节 i**：
   - 输出文件: draft/chapters/{i:02d}-{name}.md
   - 遵循中文写作规范
   - 技术术语首次出现给出英文原文
   - 与前面章节保持连贯

3. **校审章节 i**：
   - 输出文件: reviews/review-ch{i:02d}.md
   - 检查技术准确性、逻辑清晰度、语言规范
   - 检查与前面章节的连贯性

4. **更新任务状态**：
   - 在 tasks.md 中标记本章完成
   - 更新状态概览

5. **继续下一章**：
   - 校审通过 → 开始下一章
   - 校审不通过 → 修改后重新校审
```

#### 系列模式（多篇文章）

**串行执行**：每篇文章独立文件，写完一篇、校审通过后才开始下一篇。

```markdown
工作流程（循环执行每篇文章）：

对于每篇文章 i (i = 1, 2, ..., N):

1. **写作前准备**：
   - 阅读 outline.md 了解本篇文章要点
   - 阅读前面所有文章（如有）
   - 理解上下文，确保内容连贯

2. **写作文章 i**：
   - 输出文件: draft/{i:02d}-{name}.md
   - 遵循中文写作规范
   - 技术术语首次出现给出英文原文
   - 与前面文章保持连贯

3. **校审文章 i**：
   - 输出文件: reviews/review-{i:02d}.md
   - 检查技术准确性、逻辑清晰度、语言规范
   - 检查与前面文章的连贯性

4. **更新任务状态**：
   - 在 tasks.md 中标记本篇文章完成
   - 更新状态概览

5. **继续下一篇**：
   - 校审通过 → 开始下一篇
   - 校审不通过 → 修改后重新校审
```

**关键规则**：
- 必须按文章顺序串行执行
- 写作下一篇前必须阅读大纲和前面文章
- 每篇文章校审通过后才能进入下一篇
- 文章之间保持内容连贯、术语一致
- 文件命名：`{序号}-{name}.md`，如 `01-limit.md`
- 代码和图片资源统一存放在 `code/` 和 `assets/` 目录

### 5. 监控执行

监控写作执行状态：

**单文件模式**：
- 检查 `draft/content.md` 的生成
- 检查 `reviews/review-{n}.md` 的校审报告

**章节模式**：
- 检查 tasks.md 中各章节的完成状态
- 检查 `draft/chapters/{nn}-{name}.md` 的生成
- 检查 `reviews/review-ch{nn}.md` 的校审报告
- 确保每章校审通过后才进入下一章

**系列模式**：
- 检查 tasks.md 中各文章的完成状态
- 检查 `draft/{nn}-{name}.md` 的生成
- 检查 `reviews/review-{nn}.md` 的校审报告
- 检查 `code/` 和 `assets/` 的资源
- 确保每篇文章校审通过后才进入下一篇

### 6. 处理迭代循环

如果校审未通过：
- Reviewer 会创建修改任务
- Author 需要根据反馈修改
- 进入下一轮校审
- 最多 3 轮

### 7. 完成或暂停

**如果成功**:
- 更新 `.article.yaml` 的 `status` 为 `verified`
- 输出完成信息

**如果 3 轮后仍未通过**:
- 标记 `flags.needsUserConfirmation: true`
- 输出提示信息，建议用户检查

---

## 输出示例

### 单文件模式成功完成

```
## 写作完成

**文章**: transformer-attention
**标题**: 深入理解 Transformer 的 Attention 机制
**模式**: 单文件

**执行情况**:
- ✅ 写作完成: draft/content.md (3,500字)
- ✅ 校审通过

**状态**: verified

**下一步**: 运行 `/writer:publish` 发布文章。
```

### 章节模式成功完成

```
## 写作完成

**文章**: linear-algebra-vectors-matrices
**标题**: 向量与矩阵运算基础
**模式**: 章节模式

**执行情况**:
- ✅ 第1章：引言 (~2400字) → 校审通过
- ✅ 第2章：向量基础 (~4200字) → 校审通过
- ✅ 第3章：矩阵基础 (~4600字) → 校审通过
- ✅ 第4章：Python实践 (~3800字) → 校审通过
- ✅ 第5章：应用场景 (~4100字) → 校审通过
- ✅ 第6章：总结 (~2500字) → 校审通过

**总字数**: 约 21000 字

**生成文件**:
- draft/chapters/01-introduction.md
- draft/chapters/02-vectors.md
- draft/chapters/03-matrices.md
- draft/chapters/04-numpy.md
- draft/chapters/05-applications.md
- draft/chapters/06-summary.md

**状态**: verified

**下一步**: 运行 `/writer:publish` 发布文章。
```

### 需要用户确认

```
## 写作暂停 - 需要用户确认

**文章**: transformer-attention
**标题**: 深入理解 Transformer 的 Attention 机制

**问题**: 3轮校审后仍有 2 个严重问题未解决

**未解决问题**:
1. Self-Attention 的 scale 因子解释不够清晰
2. 实验代码在 GPU 环境下运行失败

**建议**:
- 查看 articles/transformer-attention/reviews/review-3.md
- 决定是否接受当前版本或手动修改

**状态**: needs_user_confirmation
```

---

## Teammate 角色定义

### Author (作者)

```
职责：写作文章内容

单文件模式工作流程：
1. 阅读 outline.md 了解完整结构
2. 写作所有章节到 draft/content.md
3. 完成后等待校审反馈
4. 根据反馈修改内容

章节模式工作流程：
1. 阅读 outline.md 了解本章要点
2. 阅读前面所有章节（如有）
3. 写作本章到 draft/chapters/{nn}-{name}.md
4. 等待本章校审通过
5. 进入下一章
6. 重复步骤1-5

写作规范：
- 使用中文
- 技术术语首次出现时给出英文原文
- 代码示例要有注释
- 章节之间要有过渡
- 章节模式：必须阅读前面章节，确保内容连贯
```

### Experimenter (实验者)

```
职责：设计实验代码和练习题

工作流程：
1. 分析 outline.md 识别关键知识点
2. 为每个知识点设计验证性实验
3. 调用沙箱 API 验证代码
4. 设计配套练习题
5. 失败重试最多 3 次

实验规范：
- 代码要有清晰的注释
- 每个实验说明验证目标
- 练习题要有提示或参考答案
- 使用 ```python runnable 标记可运行代码
```

### Reviewer (校审)

```
职责：校审文章质量

校审维度：
1. 技术准确性：概念正确、代码可运行、术语准确
2. 逻辑清晰：结构符合大纲、论述有条理
3. 语言规范：无错别字、句式通顺、中文表达
4. 完整性：覆盖所有知识点、实验完整

决策逻辑：
- 严重问题 > 0 → 不通过
- 中等问题 > 3 → 不通过
- 否则 → 通过

迭代控制：
- 每轮校审输出详细报告
- 不通过时列出具体修改建议
- 最多 3 轮
```

---

## 容错机制

| 场景 | 处理方式 |
|------|----------|
| Agent Team 启动失败 | 报告错误，提示用户检查设置 |
| Author 写作卡住 | 检查 tasks.md，提示用户可能需要简化大纲 |
| 实验代码 3 次失败 | 标记代码需确认，继续其他工作 |
| 校审 3 轮不通过 | 标记文章需确认，更新状态 |
| 沙箱不可用 | 跳过实验验证，标记待确认 |

---

## Guardrails

- 工作目录必须在 `articles/` 下，而不是 `docs/_work/`
- 根据文章类型选择正确的写作模式：
  - 单篇 ≤5000字：单文件模式
  - 单篇 >5000字：章节模式
  - 系列文章：系列模式
- 章节模式和系列模式必须串行执行，不能并行
- 章节模式写作时必须阅读前面章节，确保连贯
- 系列模式写作时必须阅读前面文章，确保连贯
- 每章/每篇文章校审通过后才能进入下一章/下一篇
- 严格执行迭代限制
- 保持文章状态与实际进度一致
- 遇到错误不要猜测，报告给用户