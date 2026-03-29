## Why

当前文章页面底部缺少关键元信息展示，用户无法快速了解文章的更新时间、内容篇幅，也无法便捷地表达对内容的认可（如 GitHub Star）。这降低了用户体验，也减少了项目与用户之间的互动机会。

按照 pencil 设计稿的规划，需要在文章底部统一展示：更新时间、字数统计、GitHub Star 按钮，形成完整的文章元信息区域。

## What Changes

- **新增文章底部元信息组件 (ArticleFooter)**：替换默认主题的 VPPageMeta，完全遵循设计稿样式
- **新增字数统计插件**：在构建时计算 Markdown 文章的字数，支持中英文混合统计
- **集成 GitHub Star 按钮**：使用 `github-buttons` npm 包实现实时 Star 数显示
- **隐藏默认 VPPageMeta**：自定义 Layout.vue 移除默认的页脚元信息组件

## Capabilities

### New Capabilities

- `article-footer`: 文章底部元信息展示组件，包含更新时间、字数统计、GitHub Star 按钮的完整 UI 和交互逻辑
- `word-count-plugin`: VuePress 插件，在构建时统计文章字数并存入 page data，支持中英文混合统计

### Modified Capabilities

- 无（本项目无现有 specs）

## Impact

**新增文件：**
- `docs/.vuepress/plugins/word-count/index.js` - 字数统计插件
- `docs/.vuepress/components/GithubButton.vue` - GitHub Star 按钮组件
- `docs/.vuepress/theme/components/ArticleFooter.vue` - 文章底部元信息组件

**修改文件：**
- `docs/.vuepress/theme/layouts/Layout.vue` - 在 content-bottom slot 插入 ArticleFooter
- `docs/.vuepress/config.js` - 添加 word-count 插件配置
- `package.json` - 添加 `github-buttons` 依赖

**依赖变更：**
- 新增 `github-buttons@^2.13.0` npm 包（用于实时 GitHub Star 按钮）