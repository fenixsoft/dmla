## Why

Gitalk 评论系统存在显著的性能和架构问题：开发模式下体积膨胀至 ~7MB，依赖过时的库（axios 0.19、date-fns 28MB），且需要 GitHub OAuth 代理服务器才能正常工作。Giscus 基于 GitHub Discussions，无需 OAuth 代理，体积更小，且与 VuePress 生态有更好的集成（通过 vuepress-plugin-giscus）。

## What Changes

- 移除 `gitalk` 依赖及其 CSS 样式导入
- 添加 `vuepress-plugin-giscus` 作为替代方案
- 重构 `Comments.vue` 组件以使用 Giscus API
- 更新评论系统配置（repo、categoryId、mapping 等参数）
- 保留现有的 OAuth 错误处理逻辑（适配 Giscus 的错误场景）

## Capabilities

### New Capabilities

- `giscus-comments`: 基于 GitHub Discussions 的评论系统，支持多语言、主题适配、懒加载

### Modified Capabilities

无（评论系统作为全新能力引入，不修改现有 spec）

## Impact

- **前端代码**: `docs/.vuepress/theme/components/Comments.vue` 完全重构
- **配置文件**: `docs/.vuepress/config.js` 插件配置更新
- **依赖变更**: `package.json` 移除 gitalk，添加 vuepress-plugin-giscus
- **GitHub仓库**: 需在仓库中启用 Discussions 并创建评论分类