## Context

当前评论系统基于 Gitalk，存在以下问题：
- **体积膨胀**: 开发模式下 ~7MB，生产构建 ~150KB（但仍在主 bundle 中）
- **OAuth 依赖**: 需代理服务器处理 GitHub OAuth，增加运维复杂度
- **过时依赖**: axios 0.19（安全漏洞）、date-fns 28MB（仅用 2 个函数）
- **无 ESM 支持**: CommonJS 模块，无法有效 tree-shake

Giscus 作为替代方案的优势：
- 基于 GitHub Discussions，无需 OAuth
- 官方 VuePress 插件支持（vuepress-plugin-giscus）
- 轻量级，懒加载支持
- 多语言、主题自动适配

## Goals / Non-Goals

**Goals:**
- 替换 Gitalk 为 Giscus，减少首屏加载体积
- 移除 OAuth 代理依赖，简化架构
- 保持现有评论数据（GitHub Issues → Discussions 迁移需手动处理）
- 保留错误处理 UI（OAuth 错误 → Giscus 加载错误）

**Non-Goals:**
- 不处理历史评论数据迁移（GitHub Issues 与 Discussions 结构不同）
- 不改变评论系统的视觉样式（仅替换底层实现）

## Decisions

### 1. 使用 vuepress-plugin-giscus 而非手动集成

**选择**: vuepress-plugin-giscus
**原因**:
- 官方维护，与 VuePress 生态兼容
- 自动处理懒加载、主题适配
- 减少 90% 的集成代码

**替代方案**:
- 手动集成 giscus.js: 需自行处理 Vue 生命周期、懒加载

### 2. 评论组件保留在 Comments.vue

**选择**: 重构 Comments.vue，不创建新组件
**原因**:
- 保持现有代码结构不变
- 便于后续维护者理解变更范围

### 3. GitHub Discussions 配置

**选择**: 使用 `pathname` mapping（基于 URL 路径关联 Discussion）
**原因**:
- 与现有 Gitalk 的 `id: location.pathname` 配置一致
- 便于后续文章与评论的关联

**替代方案**:
- `title` mapping: 与 frontmatter 标题关联，但标题可能变化
- `number` mapping: 需手动配置每个文章的 Discussion number

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| Discussions 未启用导致评论失败 | 文档说明启用步骤，组件显示启用提示 |
| 现有 Issues 评论无法迁移 | 提示用户手动迁移或保留 Issues 作为历史记录 |
| Giscus 依赖 GitHub 服务可用性 | 组件显示加载失败提示，不影响页面其他功能 |

## Migration Plan

1. **准备阶段**: 启用 GitHub Discussions，创建评论分类
2. **开发阶段**: 安装插件，重构 Comments.vue，测试功能
3. **部署阶段**: 构建验证，部署上线
4. **清理阶段**: 移除 Gitalk 依赖，关闭 OAuth 代理服务

**回滚策略**: 保留 Gitalk 代码分支，可快速切换回旧系统