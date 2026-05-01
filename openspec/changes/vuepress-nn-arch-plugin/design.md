## Context

DMLA 是一个 VuePress 2 + Vue 3 的深度学习教学网站，已有多个自定义插件（mermaid、runnable-code、math 等）。nn-arch 是一个纯前端的神经网络可视化 npm 包，输入 YAML 生成 SVG 架构图。

现有插件采用统一模式：
- `index.js`: 通过 `extendsMarkdown` 拦截 fence 代码块，将代码转换为带 data 属性的 HTML 元素
- `client.js`: 通过 `defineClientConfig` 在客户端动态加载库并渲染

## Goals / Non-Goals

**Goals:**
- 支持 Markdown 中 `nn-arch` 代码块语法
- 支持尺寸参数（`nn-arch width=800 height=400`）
- 复用现有 nn-arch npm 包 API
- 遵循现有插件架构模式

**Non-Goals:**
- 不支持实时编辑预览（非编辑器功能）
- 不创建独立的 nn-arch 编辑页面
- 不修改 nn-arch 包本身

## Decisions

### 1. 使用客户端渲染而非构建时渲染

**选择**: 客户端渲染（与 mermaid 插件一致）

**原因**:
- nn-arch 是纯前端库，无 Node.js API
- 客户端渲染支持动态尺寸调整
- 与现有插件架构保持一致
- 避免构建时引入额外的 SSR 复杂性

### 2. 尺寸参数解析方式

**选择**: 在代码块语言标识符后添加参数（如 `nn-arch width=800 height=400`）

**原因**:
- 与 mermaid 的 `mermaid small` 语法风格一致
- 用户直观易理解
- 正则解析简单可靠

**替代方案**:
- YAML 内定义尺寸：需要修改 YAML schema，侵入性强
- 单独配置文件：过于繁琐

### 3. SVG 嵌入方式

**选择**: 直接将 SVG 字符串嵌入 DOM

**原因**:
- nn-arch 返回 SVG 字符串
- 无需额外文件管理
- 支持 CSS 样式控制

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| nn-arch npm 包加载失败 | 添加错误处理，显示占位提示 |
| YAML 解析错误 | 捕获异常，显示错误信息而非崩溃 |
| 大型架构图性能问题 | 限制最大尺寸，提示用户简化 |
| 客户端渲染延迟页面显示 | 添加 loading 状态，异步渲染 |