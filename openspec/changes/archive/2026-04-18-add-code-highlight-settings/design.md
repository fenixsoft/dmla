## Context

当前项目使用 VuePress + PrismJS 实现代码高亮。设置页面（`Settings.vue`）仅包含沙箱服务地址配置，标题为"沙箱设置"。代码块样式在 `index.scss` 中硬编码，使用了类似 One Dark Pro 的配色方案（背景 `#282C34`，文字 `#ABB2BF`）。

PrismJS 提供了 8 个内置主题：
- `prism.css` (default light)
- `prism-coy.css` (light, paper-like)
- `prism-dark.css` (dark)
- `prism-funky.css` (colored, funky)
- `prism-okaidia.css` (dark, similar to Monokai)
- `prism-solarizedlight.css` (Solarized Light)
- `prism-tomorrow.css` (dark, Tomorrow Night)
- `prism-twilight.css` (dark, Twilight)

用户无法自定义高亮风格，无法满足不同用户的阅读偏好。

## Goals / Non-Goals

**Goals:**
- 重构设置页面为通用设置组件，标题改为"设置"
- 实现代码高亮风格选择功能，下拉框展示可用主题
- 实时预览样例代码片段，选择后立即显示效果
- 配置持久化到 localStorage，刷新页面后保持选择
- 动态加载 PrismJS 主题 CSS，实现全局代码块高亮切换
- 保持原有沙箱配置功能不变

**Non-Goals:**
- 不提供自定义配色方案编辑功能（仅使用 PrismJS 预设主题）
- 不修改 runnable-code-block 插件的代码执行逻辑
- 不支持按页面/代码块单独设置高亮风格（全局统一）

## Decisions

### 1. 配置存储策略

**决策**: 将 localStorage key 从 `sandbox-config` 改为 `site-config`，统一存储所有站点设置。

**理由**:
- 单一 key 便于管理，避免配置分散
- 未来可能添加更多设置项（如字体大小、主题色等）
- 需要做数据迁移，读取旧的 `sandbox-config` 并合并到新 key

**替代方案**: 保持 `sandbox-config` 并新增 `highlight-config` key
- **缺点**: 配置分散，管理复杂，迁移历史数据困难

### 2. 主题 CSS 动态加载方式

**决策**: 使用 `<link>` 标签动态插入/替换 CSS 文件。

**理由**:
- PrismJS 主题是独立的 CSS 文件，适合动态加载
- VuePress 默认主题的代码块样式需要覆盖 PrismJS 主题
- 替换 `<link>` 标签可以立即生效，无需重新渲染组件

**实现**: 创建 `CodeHighlightLoader.vue` 组件，负责：
- 从 localStorage 读取配置
- 动态插入 `<link>` 标签加载 PrismJS 主题 CSS
- 使用自定义 CSS 变量覆盖部分样式（保持工具栏等 UI 统一）

### 3. 默认高亮风格

**决策**: 默认使用 `one-dark-pro`（自定义主题），非 PrismJS 预设。

**理由**:
- 当前网站的代码块样式是自定义的 One Dark Pro 风格
- 用户已经习惯了当前的显示效果
- PrismJS 预设主题中 `prism-tomorrow.css` 最接近当前风格

**实现**: 
- 默认选项为 "当前样式"（保持现有 CSS 不加载 PrismJS 主题）
- 其他选项对应 PrismJS 预设主题

### 4. 预览代码片段内容

**决策**: 预览使用固定样例代码，包含多种语法元素。

**理由**:
- 固定内容便于对比不同主题效果
- 包含关键字、函数、字符串、注释等典型元素
- 不依赖外部数据，实现简单

**样例代码**:
```python
# 这是一个示例代码片段
def hello_world(name: str) -> str:
    """返回问候语"""
    message = f"Hello, {name}!"
    print(message)
    return message

class User:
    def __init__(self, name: str):
        self.name = name

# 调用函数
result = hello_world("World")
```

## Risks / Trade-offs

### 1. 样式覆盖冲突 → 使用 CSS 优先级控制
- **风险**: PrismJS 主题 CSS 可能覆盖自定义代码块 UI 样式（工具栏、按钮等）
- **缓解**: 使用更高优先级的选择器或 `!important` 覆盖关键样式

### 2. 主题 CSS 加载延迟 → 预加载 + 加载指示器
- **风险**: 动态加载 CSS 可能导致短暂的样式闪烁
- **缓解**: 保存设置后显示"正在应用..."提示，CSS 加载完成后自动消失

### 3. localStorage 配置迁移 → 自动迁移脚本
- **风险**: 用户已有的 `sandbox-config` 数据丢失
- **缓解**: 首次加载时检查旧 key，合并到新 key 后删除旧数据

### 4. 移动端适配 → 响应式设计
- **风险**: 设置弹窗在移动端显示效果不佳
- **缓解**: 保持现有弹窗响应式样式，下拉框和预览区域适配小屏幕