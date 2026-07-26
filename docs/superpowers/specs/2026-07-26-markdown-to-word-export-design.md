# Markdown 文章批量导出 Word 文档 — 设计规格

## 目标

将 `docs/.vuepress/config.js` 中定义的中文文章批量转换为 Word (.docx) 文档，渲染格式尽可能接近 HTML 网页效果，尤其是 LaTeX 数学公式需兼顾视觉保真度和可编辑性。

## 技术选型

采用 **Pandoc + Lua 过滤器 + 预处理脚本** 方案。Pandoc 将 Markdown 直接转为 DOCX，LaTeX 公式默认转为 Word 原生 OMML 公式（可编辑、可缩放），通过 Node.js 预处理和 Lua 过滤器处理 VuePress 特有语法。

## 架构

```
config.js ──解析──▶ 文章清单 ──预处理──▶ 标准 Markdown ──Pandoc──▶ .docx
                                  ▲                        ▲
                            Lua 过滤器              参考模板 (reference.docx)
```

三个环节：

1. **文章清单解析器**（Node.js）：从 config.js 提取中文文章路径和章节归属
2. **Markdown 预处理**（Node.js）：将 VuePress 特有语法转为标准 Markdown
3. **Pandoc 转换**：调用 Pandoc，配合 Lua 过滤器和参考模板输出 .docx

## 文章清单解析

从 `themeConfig.locales['/'].sidebar` 递归提取所有叶子节点的 `link` 和 `text` 字段：

- `link` 如 `/maths/linear/vectors` → 文件路径 `docs/maths/linear/vectors.md`
- `text` 如「向量基础」→ 文章标题
- 聚合页（仅含 `link` 无 `children`，如 `contents`）过滤掉
- 相对路径（缺少前导 `/`）统一处理

**排除**：附录章节不生成 Word 文档。

## Markdown 预处理

### 自定义容器（`::: tip` 等）

转为 Pandoc fenced div 语法 `::: {.tip}`，由 Lua 过滤器处理为带样式的提示块。

### Runnable 代码块

移除 ` runnable` 后缀，代码块上方插入说明文字（如「*原文中为可运行代码块*」）。

### Mermaid 图表

` ```mermaid` 代码块调用 `mmdc`（Mermaid CLI）渲染为 PNG 图片，替换为 `![](image.png)`。

### 图片路径

Pandoc 以 Markdown 文件所在目录为基准解析相对路径，通常不需要额外处理。转换前校验所有图片文件存在。

### 文章间链接

`[向量](vectors.md)` 转为 `向量[^n]`，文末附加对应脚注 `[^n]: 参见「向量基础」`。Pandoc 在 DOCX 输出时自动将脚注渲染到页面底部。

### VuePress 特殊标记

- `[[toc]]` → 删除
- `{{eq:xxx}}` 公式交叉引用标记 → 替换为「公式（xxx）」文字引用。预处理时通过正则 `\{\{eq:[\w-]+\}\}` 匹配并替换。中文文章约 28 个文件使用此标记，数量有限，可在预处理时统一处理

## Pandoc 转换

### 调用方式

```bash
pandoc input.md -o output.docx \
  --from=markdown+footnotes \
  --reference-doc=articles/word/reference.docx \
  --lua-filter=filters/tip-block.lua \
  --lua-filter=filters/code-block.lua
```

### 数学公式

Pandoc 对 DOCX 输出的默认行为是将 `$...$` 和 `$$...$$` 转为 Word 原生 OMML 公式。支持 `\begin{cases}`、`\begin{bmatrix}`、`\begin{aligned}` 等环境，双击即可编辑。

### Lua 过滤器

- **提示块过滤器**：捕获 `::: {.tip}` fenced div，应用「注意」样式，生成带底色的段落
- **代码块过滤器**：给代码块应用「代码清单」样式（浅灰底色、等宽字体）

### 参考模板（reference.docx）

已从用户提供的 `articles/word/ref.docx` 生成。保留所有样式定义，清空内容和页眉页脚。关键样式：

| 样式名 | 用途 | 格式 |
|--------|------|------|
| Normal | 正文 | Times New Roman + 宋体, 10.5pt, 1.5 倍行距 |
| Heading 1-5 | 多级标题 | 18pt→14pt，1.5-2.4 倍行距 |
| 注意 | 提示/警告块 | 自定义段落样式 |
| 代码清单 | 代码块 | 等宽字体、浅灰底色 |
| 图题 | 图片标题 | 居中小号字 |
| 项目符号/编号 | 列表 | 标准列表样式 |

页面设置：A4（8.3"×11.7"），左右页边距约 1 英寸。

## 输出规范

### 目录结构

```
articles/word/
├── reference.docx          # Pandoc 样式模板
├── ref.docx                # 用户提供的原始参考
└── output/                 # 生成的 Word 文档（无子目录）
    ├── 0.1-about-me.docx
    ├── 0.2-about-dmla.docx
    ├── 1.1-vectors.docx
    ├── 1.2-matrices.docx
    └── ...
```

### 文件命名

格式：`<章节序号>.<文件序号>-<slug>.docx`

- 章节序号从 0 开始，文件序号从 1 开始
- slug 取自 `link` 路径最后一段
- 示例：`/maths/linear/vectors` → `1.1-vectors.docx`

### 章节序号映射

| 序号 | 章节 |
|------|------|
| 0 | 前言 |
| 1 | 机器学习数学基础 |
| 2 | 经典统计学习方法 |
| 3 | 神经网络与深度学习 |
| 4 | 语言模型的奇点 |
| 5 | AI 基础设施与工程化 |
| 6 | Agentic 应用系统 |

附录（序号 7）不导出。

## 依赖

- **Pandoc**：Markdown → DOCX 转换引擎
- **Node.js**：文章清单解析、Markdown 预处理脚本
- **python-docx**：reference.docx 模板生成与验证
- **@mermaid-js/mermaid-cli**（mmdc）：Mermaid 图表渲染为 PNG
- **Pandoc Lua**：内置，无需额外安装

## 风险与应对

1. **极端复杂 LaTeX 环境兼容性**：`\begin{align*}` 等多行对齐环境 Pandoc 的 OMML 还原度可能略低于 MathJax HTML 输出。如遇问题可针对性在 Lua 过滤器中做预处理替换。
2. **Mermaid CLI 依赖 Puppeteer**：`mmdc` 需要 Chrome/Chromium。若环境中没有，可降级为保留 Mermaid 源码并标注说明。
3. **图片文件缺失**：预处理时校验图片路径，缺失的图片替换为占位文字。
