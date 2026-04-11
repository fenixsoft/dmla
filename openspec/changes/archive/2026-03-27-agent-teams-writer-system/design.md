# 技术设计：Agent Teams 文章写作系统

## 1. 系统架构

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Agent Teams 文章写作系统                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   用户工作流                                                                 │
│   ─────────────────────────────────────────────────────────────────────    │
│                                                                             │
│   /writer:explore  →  /writer:propose  →  /writer:compose  →  /writer:publish
│        │                    │                     │                 │       │
│        ▼                    ▼                     ▼                 ▼       │
│     探索主题            创建大纲/任务         Agent Team 写作       发布归档  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        Skills Layer                                  │   │
│   │                                                                      │   │
│   │   .claude/skills/                                                    │   │
│   │   ├── writer-explore/SKILL.md      # 探索主题                       │   │
│   │   ├── writer-propose/SKILL.md      # 创建提案                       │   │
│   │   ├── writer-compose/SKILL.md      # 执行写作（核心）               │   │
│   │   └── writer-publish/SKILL.md      # 发布归档                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     Agent Teams Layer                                │   │
│   │                                                                      │   │
│   │   /writer:compose 启动 Agent Team:                                  │   │
│   │                                                                      │   │
│   │   ┌─────────────┐                                                    │   │
│   │   │  Team Lead  │  创建团队、分配任务、协调工作、整合结果            │   │
│   │   │   (经理)    │                                                    │   │
│   │   └──────┬──────┘                                                    │   │
│   │          │                                                           │   │
│   │          │ 生成 Teammates                                             │   │
│   │          │                                                           │   │
│   │          ├──────────────────┬──────────────────┐                    │   │
│   │          ▼                  ▼                  ▼                    │   │
│   │   ┌────────────┐    ┌────────────┐    ┌────────────┐               │   │
│   │   │   Author   │    │ Reviewer   │    │Experimenter│               │   │
│   │   │  (作者)    │    │  (校审)    │    │  (实验者)  │               │   │
│   │   └────────────┘    └────────────┘    └────────────┘               │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Storage Layer                                   │   │
│   │                                                                      │   │
│   │   articles/                    # 文章工作目录                        │   │
│   │   └── {article-name}/                                                │   │
│   │       ├── .article.yaml        # 元数据 + 状态                      │   │
│   │       ├── outline.md           # 文章大纲                            │   │
│   │       ├── tasks.md             # 任务列表                            │   │
│   │       ├── draft/                                                     │   │
│   │       │   ├── content.md       # 文章内容                            │   │
│   │       │   └── experiments.md   # 实验代码                            │   │
│   │       ├── reviews/                                                   │   │
│   │       │   └── review-{n}.md    # 校审报告                            │   │
│   │       └── archive/             # 归档目录                            │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     Integration Layer                                │   │
│   │                                                                      │   │
│   │   Sandbox API (现有)                                                 │   │
│   │   POST http://localhost:3001/api/sandbox/run                        │   │
│   │   • 执行实验代码验证                                                 │   │
│   │   • 返回运行结果                                                     │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 组件职责

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `/writer:explore` | 探索文章主题 | 主题描述 | 探索结论 |
| `/writer:propose` | 创建文章大纲 | 探索结果 | `.article.yaml`, `outline.md`, `tasks.md` |
| `/writer:compose` | 执行写作任务 | 文章名称 | 完成的文章草稿 |
| `/writer:publish` | 验收发布 | 文章名称 | 发布到 `docs/` |
| Team Lead | 协调 Agent Team | 任务列表 | 整合结果 |
| Author | 写作内容 | 章节/大纲 | 文章内容 |
| Reviewer | 校审文章 | 文章内容 | 校审报告 |
| Experimenter | 设计实验 | 知识点 | 实验代码 + 练习题 |

## 2. 文件系统设计

### 2.1 目录结构

```
articles/
├── index.json                        # 文章索引
├── {article-name}/                   # 一篇文章的工作目录
│   ├── .article.yaml                 # 文章元数据
│   ├── outline.md                    # 文章大纲
│   ├── tasks.md                      # 任务列表
│   ├── draft/
│   │   ├── content.md                # 文章内容
│   │   └── experiments.md            # 实验代码
│   └── reviews/
│       ├── review-1.md               # 校审报告
│       └── review-2.md
└── archive/                          # 已发布文章归档
    └── {YYYY-MM-DD}-{article-name}/
        └── article.md
```

### 2.2 文件格式定义

#### .article.yaml

```yaml
name: transformer-attention
title: "深入理解 Transformer 的 Attention 机制"
status: composing  # exploring | proposed | composing | reviewing | verified | published
created: 2026-03-25
updated: 2026-03-25

audience:
  level: intermediate  # beginner | intermediate | advanced
  background:
    - Python 基础
    - 神经网络基础概念

config:
  needExperiments: true
  targetLength: 3000
  categories: [ai, deep-learning]

reviewHistory:
  - round: 1
    status: rejected
    date: 2026-03-25
    issues: 3
  - round: 2
    status: approved
    date: 2026-03-25
    issues: 0

flags:
  needsUserConfirmation: false  # 3轮后标记
  confirmationReason: null
```

#### index.json

```json
{
  "articles": [
    {
      "name": "transformer-attention",
      "title": "深入理解 Transformer 的 Attention 机制",
      "status": "published",
      "categories": ["ai", "deep-learning"],
      "publishedDate": "2026-03-25",
      "path": "docs/ai/transformer-attention.md"
    }
  ],
  "stats": {
    "total": 100,
    "published": 80,
    "inProgress": 15,
    "proposed": 5
  }
}
```

## 3. Agent Team 配置

### 3.1 Team Lead 角色

Team Lead 是主 Claude Code 会话，负责：

1. **创建团队**：根据任务类型生成合适的 teammates
2. **分配任务**：将 tasks.md 中的任务分配给对应 teammate
3. **协调工作**：监控进度，处理依赖关系
4. **整合结果**：合并各 teammate 的输出
5. **迭代控制**：管理校审循环，执行容错策略

### 3.2 Teammates 配置

#### Author (作者)

```
职责：写作文章所有章节
工作目录：draft/content.md
任务来源：tasks.md 中的 "写作-*" 任务

工作流程：
1. 读取 outline.md 了解章节结构
2. 按顺序写作各章节
3. 每完成一章，更新 tasks.md 标记完成
4. 等待 Reviewer 反馈
5. 根据反馈修改内容
```

#### Experimenter (实验者)

```
职责：设计实验代码和练习题
工作目录：draft/experiments.md
任务来源：tasks.md 中的 "实验-*" 任务

工作流程：
1. 分析 outline.md 识别关键知识点
2. 设计验证性实验代码
3. 调用沙箱 API 验证代码
4. 设计配套练习题
5. 失败重试最多 3 次
```

#### Reviewer (校审)

```
职责：校审文章质量
工作目录：reviews/review-{n}.md
任务来源：tasks.md 中的 "校审-*" 任务

校审维度：
1. 技术准确性：概念正确、代码可运行、术语准确
2. 逻辑清晰：结构符合大纲、论述有条理、过渡自然
3. 语言规范：无错别字、句式通顺、中文表达
4. 完整性：覆盖所有知识点、实验完整

工作流程：
1. 等待 Author + Experimenter 完成
2. 执行校审清单
3. 输出校审报告
4. 通过/不通过决策
```

### 3.3 并行与串行策略

```
┌─────────────────────────────────────────────────────────────────┐
│                     执行时序                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Time →                                                        │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Phase 1: 并行写作                                        │  │
│   │                                                          │  │
│   │ Author:         ████████████████████████                │  │
│   │                 写章节1  写章节2  写章节3  写章节4        │  │
│   │                                                          │  │
│   │ Experimenter:   ████████████████                        │  │
│   │                 设计实验1  验证  设计实验2  验证          │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                    │                            │
│                                    ▼                            │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Phase 2: 校审                                            │  │
│   │                                                          │  │
│   │ Reviewer:      ████████████                              │  │
│   │                 校审  输出报告                            │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                    │                            │
│                          ┌────────┴────────┐                    │
│                          ▼                 ▼                    │
│                       通过              不通过                   │
│                          │                 │                    │
│                          ▼                 ▼                    │
│                       整合发布      ┌─────────────────────┐     │
│                                    │ Phase 3: 修改        │     │
│                                    │ Author: 修改         │     │
│                                    │ Reviewer: 再次校审   │     │
│                                    └─────────────────────┘     │
│                                           │                     │
│                                           └──→ 最多 3 轮        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4. 校审机制设计

### 4.1 校审清单

```markdown
## 技术准确性 (Technical Accuracy)
- [ ] 概念定义准确，无误导性描述
- [ ] 代码示例可运行（已在沙箱验证）
- [ ] 技术术语使用正确
- [ ] 公式/算法描述正确
- [ ] 引用来源可靠

## 逻辑清晰 (Logical Clarity)
- [ ] 文章结构符合大纲
- [ ] 论述有条理，论据充分
- [ ] 章节之间过渡自然
- [ ] 没有逻辑矛盾或循环论证
- [ ] 结论与论述一致

## 语言规范 (Language Quality)
- [ ] 无错别字
- [ ] 句式通顺，易于理解
- [ ] 术语翻译一致（如需要）
- [ ] 代码注释清晰
- [ ] 符合 CLAUDE.md 中的语言要求（中文）

## 完整性 (Completeness)
- [ ] 覆盖大纲所有知识点
- [ ] 实验代码完整且有注释
- [ ] 练习题有参考答案或提示
- [ ] 必要的图表/流程图已包含
- [ ] 前置知识说明清楚

## 格式规范 (Format)
- [ ] Markdown 语法正确
- [ ] 代码块指定语言类型
- [ ] 可运行代码使用 `python runnable` 标记
- [ ] frontmatter 完整
```

### 4.2 校审报告格式

```markdown
# 校审报告 - 第 {N} 轮

文章: {文章标题}
审核时间: {ISO datetime}
审核结果: ✅ 通过 / ❌ 需修改

---

## 问题汇总

### 🔴 严重问题 (必须修改)

#### 问题 1: {问题标题}
- **位置**: {章节/行号}
- **原文**: {引用原文}
- **问题**: {问题描述}
- **建议**: {修改建议}

### 🟡 中等问题 (建议修改)
...

### 🟢 小问题 (可选修改)
...

---

## 审核统计

| 类别 | 数量 |
|------|------|
| 严重问题 | X |
| 中等问题 | X |
| 小问题 | X |
| **总计** | **X** |

---

## 结论

{通过/需修改，理由...}
```

### 4.3 迭代控制逻辑

```
校审迭代流程：

1. Reviewer 完成校审
2. 判断结果：
   - 通过 → 标记文章为 "verified"，准备发布
   - 不通过 → 进入迭代流程

迭代流程：
1. 创建修改任务（基于校审报告）
2. Lead 发送消息给 Author
3. Author 修改内容
4. Reviewer 再次校审
5. 计数 +1

终止条件：
- 通过 → 继续发布流程
- 3 轮未通过 → 标记 .article.yaml:
  flags:
    needsUserConfirmation: true
    confirmationReason: "3轮校审未通过"
  → 继续下一篇文章
```

## 5. 实验验证设计

### 5.1 沙箱集成

```javascript
// Experimenter 调用沙箱 API
const response = await fetch('http://localhost:3001/api/sandbox/run', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    code: experimentCode,
    useGPU: needsGPU
  })
});

const result = await response.json();
// result: { success, output, error, executionTime, gpuUsed }
```

### 5.2 实验代码格式

````markdown
## 实验：{实验标题}

### 目标
{实验目标描述}

### 代码
```python runnable
# 实验：验证 Attention 机制
import torch
import torch.nn.functional as F

def simple_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attention = F.softmax(scores, dim=-1)
    return torch.matmul(attention, V)

# 验证代码
Q = torch.randn(2, 3, 4)
K = torch.randn(2, 3, 4)
V = torch.randn(2, 3, 4)
result = simple_attention(Q, K, V)
print(f"Output shape: {result.shape}")
```

### 预期输出
```
Output shape: torch.Size([2, 3, 4])
```

### 练习题
1. **练习1**：修改代码，添加 mask 支持
2. **练习2**：使用 matplotlib 可视化 attention 权重
````

### 5.3 实验失败处理

```
实验验证流程：

1. Experimenter 设计实验代码
2. 调用沙箱 API 验证
3. 判断结果：
   - 成功 → 保留代码，继续
   - 失败 → 修改代码，重试

重试逻辑：
- 最多重试 3 次
- 每次重试可修改代码或调整方法
- 3 次失败后：
  标记 experiments.md:
  > ⚠️ 此实验代码需要用户确认：{错误原因}
  → 继续其他工作
```

## 6. 容错机制

### 6.1 整体容错策略

```
原则：避免阻塞，确保流程流畅执行

容错场景：
┌────────────────────┬─────────────────────────────────────┐
│ 场景               │ 处理方式                            │
├────────────────────┼─────────────────────────────────────┤
│ 校审 3 轮不通过    │ 标记 "需用户确认"，继续下一篇文章    │
│ 实验 3 次失败      │ 标记 "需用户确认"，继续其他工作      │
│ Agent Team 启动失败│ 报告错误，建议用户重试              │
│ 沙箱不可用         │ 跳过实验验证，标记待确认            │
│ 文件读写错误       │ 报告错误，暂停等待用户介入          │
└────────────────────┴─────────────────────────────────────┘
```

### 6.2 状态恢复

```
文章状态持久化，支持中断后恢复：

恢复逻辑：
1. 读取 .article.yaml 获取当前状态
2. 根据 status 确定下一步操作：
   - exploring → 继续探索
   - proposed → 可以开始写作
   - composing → 继续写作（读取 tasks.md 确定进度）
   - reviewing → 继续校审
   - verified → 准备发布
   - published → 已完成

3. 读取 tasks.md 确定具体任务进度
4. 恢复执行
```

## 7. Skills 实现规范

### 7.1 Skill 文件结构

```
.claude/skills/writer-{name}/
├── SKILL.md           # Skill 定义
└── templates/         # 模板文件（可选）
    └── *.md
```

### 7.2 Skill 元数据格式

```yaml
---
name: writer-compose
description: 执行文章写作任务。启动 Agent Team 协调 Author、Reviewer、Experimenter 完成写作。
license: MIT
compatibility: Requires Claude Code Agent Teams feature enabled.
metadata:
  author: dmla
  version: "1.0"
---
```

## 8. 安全与权限

### 8.1 权限要求

- 文件系统读写权限（articles/ 目录）
- 网络访问权限（调用沙箱 API）
- Agent Teams 启动权限

### 8.2 数据安全

- 所有文章内容存储在本地文件系统
- 沙箱执行隔离在 Docker 容器中
- 不涉及外部 API 密钥存储

## 9. 监控与日志

### 9.1 日志记录

```
每个 skill 执行过程记录：
- 开始时间
- 文章名称
- 执行步骤
- Agent Team 状态
- 错误信息（如有）
- 结束时间
- 结果状态
```

### 9.2 进度追踪

```
通过 index.json 和各文章的 .article.yaml 追踪：
- 总文章数
- 各状态文章数
- 需用户确认的文章
- 最近更新时间
```