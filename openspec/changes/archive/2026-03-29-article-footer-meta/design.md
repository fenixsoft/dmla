## Context

当前项目使用 VuePress v2 + 自定义主题（继承 @vuepress/theme-default）。文章页面底部由默认主题的 `VPPageMeta` 组件渲染，显示编辑链接、更新时间、贡献者等信息。

根据 pencil 设计稿 `3YgV6` (Article Page) 中的 `wi52T` (articleFooter) 设计，需要：
- 左侧：字数统计（file-text icon + "文章字数：N"）和更新时间（calendar icon + "更新于 YYYY-MM-DD"）
- 右侧：GitHub Star 按钮（深色按钮 + Star 数徽章）
- 整体：顶部 1.5px 分割线，justify-content: space-between 布局

参考 icyfenix.cn 的实现方案：
- 字数统计：构建时通过 VuePress 插件的 `extendsPageData` 钩子计算
- GitHub Star：使用 `github-buttons` npm 包，客户端渲染，无需 API Token

## Goals / Non-Goals

**Goals:**
- 实现符合 pencil 设计稿的文章底部元信息 UI
- 构建时计算字数，避免客户端性能开销
- GitHub Star 实时显示，无需维护 Token
- 替换默认 VPPageMeta，避免重复信息

**Non-Goals:**
- 不实现阅读时长估算（仅显示精确字数）
- 不实现文章级别的反馈系统（如"有帮助/没帮助"）
- 不修改侧边栏、导航栏等其他组件

## Decisions

### 1. 字数统计算法选择

**决定**：采用 icyfenix.cn 的 `fnGetCpmisWords` 算法

**理由**：
- 该算法专为中文技术文章设计，正确处理中英文混合内容
- 将连续英文字符视为一个单词，中文按字符计数
- 已在 icyfenix.cn 生产环境验证

**备选方案**：
- 简单字符计数：不适合英文内容
- 正则 `\b` 单词匹配：不适合中文

### 2. GitHub Star 实现方式

**决定**：使用 `github-buttons` npm 包 + Vue 组件封装

**理由**：
- 官方 GitHub buttons，无需 API Token
- 客户端渲染，每个用户独立请求，不受 rate limit 影响
- 实时显示最新 Star 数
- 已有成熟组件封装方案（github-button.vue）

**备选方案**：
- iframe 嵌入 GitHub buttons 官方网站：加载慢，样式不可控
- 自建 GitHub API 调用：需要 Token，有 rate limit 问题
- 构建时获取 Star 数：数据不实时

### 3. VPPageMeta 处理方式

**决定**：完全隐藏默认 VPPageMeta，使用自定义 ArticleFooter

**理由**：
- VPPageMeta 的 editLink、contributors 功能在设计稿中未体现
- 自定义组件可以精确控制布局和样式
- 避免信息重复显示

**备选方案**：
- 在 VPPageMeta 后追加 ArticleFooter：会导致信息冗余
- 扩展 VPPageMeta slot：默认主题 slot 灵活性有限

## Risks / Trade-offs

### [github-buttons 加载失败] → 显示静态 Star 链接
- 客户端动态加载可能因网络问题失败
- Mitigation: 组件 fallback 为静态 `<a>` 链接，仍可跳转到 GitHub

### [字数统计准确性] → 接受约 5% 误差
- Markdown 语法残留（如代码块内的注释）可能被统计
- Mitigation: 统计前移除代码块和 LaTeX 公式，误差可接受

### [VPPageMeta 功能丢失] → 用户可能需要 editLink
- 默认主题的"编辑此页"功能将不再显示
- Mitigation: 如果用户反馈需要，可在后续版本添加