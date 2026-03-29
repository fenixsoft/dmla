## ADDED Requirements

### Requirement: 构建时计算文章字数

系统 SHALL 在 VuePress 构建过程中计算每篇文章的字数。

字数计算通过 VuePress 插件的 `extendsPageData` 钩子实现，计算结果存入 `$page.wordCount`。

#### Scenario: 有内容的文章
- **WHEN** 文章 Markdown 内容不为空
- **THEN** `$page.wordCount` 包含计算后的字数

#### Scenario: 无内容的文章
- **WHEN** 文章 Markdown 内容为空或仅包含 frontmatter
- **THEN** `$page.wordCount` 为 0

---

### Requirement: 字数统计支持中英文混合

系统 SHALL 正确统计中英文混合内容的字数。

统计算法：
1. 移除 Markdown 语法标记（代码块、LaTeX 公式、链接、图片、标题标记等）
2. 将空白符（空格、换行、全角空格）替换为占位符
3. 英文字符（ASCII）标记为单词单位，连续英文视为一个单词
4. 中文字符按单个字符计数
5. 移除占位符后计算总长度

#### Scenario: 纯中文文章
- **WHEN** 文章内容为纯中文（如"线性代数是机器学习的语言"）
- **THEN** 每个汉字计为 1 字，上述示例应返回 10

#### Scenario: 纯英文文章
- **WHEN** 文章内容为纯英文（如"Linear algebra is the language of ML"）
- **THEN** 连续英文单词计为单词数，上述示例应返回 7（7 个单词）

#### Scenario: 中英文混合文章
- **WHEN** 文章内容包含中英文混合（如"理解 Vector 和 Matrix 是 ML 的基础"）
- **THEN** 中文按字符计，英文按单词计

---

### Requirement: 字数统计排除代码块

系统 SHALL 在统计字数时排除代码块内容。

代码块包括：
- Markdown fenced code block (```语言 ... ```)
- 行内代码 (`code`)

#### Scenario: 包含代码块的文章
- **WHEN** 文章包含代码块（如 Python 示例代码）
- **THEN** 代码块内的注释、变量名不计入字数

---

### Requirement: 字数统计排除 LaTeX 公式

系统 SHALL 在统计字数时排除 LaTeX 数学公式。

公式包括：
- 行内公式 `$公式$`
- 块级公式 `$$公式$$`

#### Scenario: 包含数学公式的文章
- **WHEN** 文章包含 LaTeX 公式（如 $\mathbf{v} = (3, 2)$）
- **THEN** 公式符号不计入字数