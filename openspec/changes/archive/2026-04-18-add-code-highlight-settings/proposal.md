## Why

当前设置页面标题为"沙箱设置"，仅包含沙箱服务地址配置，功能单一。用户无法自定义代码块的高亮风格，而代码块是本教程网站的核心内容元素，提供代码高亮风格选择功能可以提升用户阅读体验，满足不同用户对代码显示风格的偏好。

## What Changes

- 将设置弹窗标题从"沙箱设置"改为"设置"
- 新增代码高亮风格选择功能：
  - 下拉框展示可选的代码高亮风格（基于 PrismJS 主题）
  - 选择后实时预览样例代码片段的高亮效果
  - 配置保存到客户端 localStorage，全局生效
- 配置存储从 `sandbox-config` 改为 `site-config`，统一管理所有站点设置
- 动态加载 PrismJS 主题 CSS 文件，实现高亮风格切换

## Capabilities

### New Capabilities

- `code-highlight-settings`: 代码高亮风格选择与全局应用功能，包括：
  - 下拉框选择 PrismJS 高亮主题
  - 实时预览样例代码片段
  - localStorage 持久化配置
  - 动态 CSS 加载实现全局切换

### Modified Capabilities

无。本功能为新增能力，不修改现有 spec 级别的需求行为。

## Impact

- **前端组件**:
  - `docs/.vuepress/theme/components/Settings.vue` - 重构为通用设置组件
  - `docs/.vuepress/theme/components/NavbarSettings.vue` - 更新按钮提示文本
- **样式**:
  - `docs/.vuepress/theme/styles/index.scss` - 代码块样式变量化，支持主题切换
- **新增文件**:
  - PrismJS 主题 CSS 加载器（客户端脚本）
- **配置存储**: localStorage key 从 `sandbox-config` 改为 `site-config`