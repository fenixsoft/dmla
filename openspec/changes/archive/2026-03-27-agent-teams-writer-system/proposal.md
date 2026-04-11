# 提案：Agent Teams 文章写作系统

## 概述

构建一个基于 Claude Code Agent Teams 的文章写作系统，用于完成数百篇计算机、人工智能等领域的技术文章写作。系统采用四角色协作模式（经理、作者、校审、实验者），通过迭代式工作流确保文章质量，并与现有沙箱环境集成验证实验代码。

## 背景

### 当前状态

- 项目 DMLA (Design Machine Learning Applications) 是一个知识管理平台，支持 Markdown 文章发布
- 已有 Docker 沙箱支持 Python 代码执行和验证
- 用户计划撰写数百篇技术文章，每篇数百至数千字

### 问题陈述

- 手动写作大量文章效率低下
- 文章质量需要多维度校审（技术、逻辑、语言、完整性）
- 实验代码需要验证确保可运行
- 需要系统化的工作流程管理

## 目标

1. **自动化写作流程**：通过 Agent Teams 协作完成文章写作
2. **质量保证**：多角色校审确保技术准确性和内容完整性
3. **实验验证**：与沙箱集成，自动验证实验代码
4. **流程可追溯**：文章状态持久化，支持断点续写

## 范围

### 包含

- 4 个 Writer Skills：`/writer:explore`、`/writer:propose`、`/writer:compose`、`/writer:publish`
- 文章状态文件系统（参考 OpenSpec 结构）
- Agent Team 配置（经理、作者、校审、实验者）
- 校审清单和反馈机制
- 实验代码沙箱验证集成
- 容错机制（3轮限制 + 标记待确认）

### 不包含

- 多文章并行处理（当前版本顺序执行）
- 自动发布到生产环境（需用户验收）
- 文章内容自动翻译

## 关键决策

| 决策项 | 选择 | 原因 |
|--------|------|------|
| Skills 集成方式 | Skills 作为协调器，内部启动 Agent Teams | 简化用户交互，统一入口 |
| 状态持久化 | 文件系统 | 参考 OpenSpec 模式，便于版本控制 |
| 并行策略 | Author + Experimenter 并行，Reviewer 串行 | 提高效率，保证质量 |
| 迭代终止 | 最多 3 轮，标记后继续 | 避免阻塞，确保流程流畅 |
| 文章处理 | 顺序执行 | 简化实现，降低资源消耗 |

## 成功标准

1. 用户可通过 `/writer:explore` → `/writer:propose` → `/writer:compose` → `/writer:publish` 完成一篇文章
2. Agent Team 自动协调四个角色完成写作和校审
3. 实验代码在沙箱中验证通过
4. 文章状态持久化，支持中断后恢复
5. 校审迭代不超过 3 轮自动处理，超过则标记待确认

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Agent Team 消耗大量 token | 成本增加 | 控制团队规模（3个 teammate），顺序处理文章 |
| 校审迭代无法收敛 | 流程阻塞 | 最多 3 轮后标记待确认，继续下一篇 |
| 实验代码验证失败 | 内容不完整 | 最多 3 次重试，失败则标记待确认 |
| Agent Teams 功能变更 | 实现需调整 | 基于官方文档设计，预留适配空间 |

## 时间线

- **Phase 1**：Skills 基础结构 + 文件系统
- **Phase 2**：Agent Team 配置 + 协调逻辑
- **Phase 3**：校审机制 + 迭代循环
- **Phase 4**：沙箱集成 + 实验验证
- **Phase 5**：容错机制 + 测试验证

## 相关资源

- [Claude Code Agent Teams 文档](https://code.claude.com/docs/zh-CN/agent-teams)
- [OpenSpec Skills 参考](.claude/skills/openspec-*)
- [项目设计文档](docs/arch/design.md) - 沙箱设计