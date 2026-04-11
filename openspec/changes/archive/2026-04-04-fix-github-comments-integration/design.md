## Context

DMLA 设计文档（docs/arch/design.md）定义了完整的 GitHub Issues 评论系统架构，包括：
- 多层缓存（内存 30s → localStorage 5min → GitHub API）
- ETag 条件请求优化 API 限流
- IntersectionObserver 懒加载
- GitHub OAuth 登录发表评论

当前实现状态：
- ✅ Comments.vue 组件功能完整（docs/.vuepress/theme/components/Comments.vue）
- ✅ commentsPlugin 插件实现完整（docs/.vuepress/plugins/comments/）
- ✅ sync-issues.yml 工作流配置正确
- ❌ Layout.vue 未引入 Comments 组件
- ❌ config.js 未注册 commentsPlugin
- ❌ sync-issues.js 创建 Issue 后不回写编号到文章 frontmatter
- ❌ 文章 frontmatter 缺少 issue.number 字段

约束条件：
- 无后端服务（互联网部署模式），评论功能纯前端实现
- GitHub API 限流：未认证 60次/小时/IP，认证 5000次/小时/用户
- OAuth 需要代理服务交换 token（可选，无代理时跳转 GitHub 评论）

## Goals / Non-Goals

**Goals:**
- 在文章页面底部显示评论区，用户可查看现有评论
- 登录 GitHub 用户可在页面内发表评论
- Issue 同步脚本创建 Issue 后自动回写编号到文章 frontmatter
- 现有文章补充 issue.number 字段关联已创建的 Issue

**Non-Goals:**
- 不实现评论嵌套回复（GitHub Issues API 不原生支持）
- 不实现评论通知推送
- 不实现评论审核机制
- 不修改 GitHub OAuth 代理服务（使用现有方案或跳转 GitHub）

## Decisions

### D1: 评论组件集成方式

**决策**: 在 Layout.vue 的 `#page-content-bottom` slot 中添加 Comments 组件

**备选方案**:
- A: 在 ArticleFooter.vue 中添加（耦合度高）
- B: 创建独立的 PageBottom.vue slot 组件（过度设计）
- C: 直接在 Layout.vue slot 中引入（简单直接）✅

**理由**: Layout.vue 已有 `#page-content-bottom` slot 用于 ArticleFooter，Comments 组件紧随其后符合逻辑顺序。最小改动，最大复用。

### D2: 插件注册配置

**决策**: 在 config.js plugins 数组中注册 commentsPlugin，传入仓库信息

**配置**:
```javascript
commentsPlugin({
  repo: 'fenixsoft/dmla',
  clientId: process.env.GITHUB_CLIENT_ID || ''
})
```

**理由**: 仓库信息硬编码，OAuth ClientId 从环境变量读取（本地开发可配置，生产环境可选）。

### D3: Issue 编号回写策略

**决策**: sync-issues.js 创建 Issue 后更新文章 frontmatter

**备选方案**:
- A: 不回写，Comments 组件按标题搜索 Issue（API 开销大）
- B: 回写编号到 frontmatter（一次写入，后续高效）✅

**实现方式**: 使用 Node.js fs 模块替换 frontmatter 的 `---` 区域，添加 `issue.number: N` 字段。

**理由**: 回写后 Comments 组件可直接使用 `issueNumber` 加载评论，无需每次搜索 API，节省 API 配额。

### D4: 现有文章 Issue 补充方式

**决策**: 运行 sync-issues.js 时自动为缺失 issue.number 的文章补充编号

**流程**:
1. sync-issues.js 搜索已存在的 Issue（按标题匹配）
2. 找到则回写编号
3. 未找到则创建新 Issue 并回写编号

**理由**: 利用现有 sync-issues.yml 工作流，无需额外脚本。

## Risks / Trade-offs

**R1: GitHub API 限流风险** → Mitigation: Comments 组件已实现多层缓存和 ETag 条件请求，304 响应不计入限流

**R2: OAuth 代理缺失导致无法页面内发表评论** → Mitigation: 无代理时提供"在 GitHub 上评论"跳转链接，用户体验略有下降但功能可用

**R3: frontmatter 回写失败导致 Issue 创建但未关联** → Mitigation: sync-issues.js 添加错误处理，Issue 创建失败时不回写，回写失败时记录警告日志

**R4: 文章标题变更导致 Issue 关联断裂** → Mitigation: 设计文档已规定使用 `issue.title` 或 `issue.number` 固定关联，sync-issues.js 优先使用已配置的编号

## Migration Plan

**Step 1: 代码集成**（本次变更）
- 注册插件、集成组件、完善回写逻辑

**Step 2: 现有文章补充**（自动）
- 下次 CI 构建时 sync-issues.yml 自动运行，为现有文章补充 issue.number

**Step 3: 验证**（人工）
- 检查文章页面评论区显示正常
- 测试评论加载和发表功能

**回滚策略**: 若评论功能异常，可临时移除 Comments 组件引入，评论数据仍在 GitHub Issues 中不受影响。