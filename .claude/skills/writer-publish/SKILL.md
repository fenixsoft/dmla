---
name: writer-publish
description: 验收文章并发布到文档目录，自动更新 VuePress 配置。保留章节结构，发布为独立目录。
license: MIT
compatibility: None
metadata:
  author: ideaspaces
  version: "2.1"
---

验收文章并发布到文档目录。

**输入**: 文章名称（可选，从上下文推断）

**输出**: 发布到 `docs/{article-name}/` 的文章目录，自动更新的 VuePress 配置，归档的工作文件

---

## 步骤

### 1. 确定文章

如果未指定文章名称：
- 检查 `articles/` 目录下的文章
- 找到 `status: verified` 的文章
- 如果有多个，使用 AskUserQuestion 让用户选择

**区分单篇文章和系列文章**：
- 单篇文章：`articles/{name}/` 目录
- 系列文章：`articles/{series-name}/` 目录，`.article.yaml` 中 `type: series`

### 2. 验证文章状态

读取 `.article.yaml` 确认状态：
- 如果 `status: composing` - 提示先运行 `/writer:compose`
- 如果 `status: reviewing` - 提示等待校审完成
- 如果 `flags.needsUserConfirmation: true` - 警告用户有未确认问题

### 3. 最终检查

检查以下内容是否完整：
- `draft/chapters/` 或 `draft/content.md` - 文章内容存在
- `draft/experiments.md` - 实验代码存在（如果需要）
- `reviews/` - 有通过的校审报告

如果有缺失或问题：
- 列出问题清单
- 使用 AskUserQuestion 询问是否继续

### 4. 用户验收

显示文章概想让用户确认：
- 文章标题
- 字数统计
- 章节列表
- 实验数量
- 校审通过情况

使用 AskUserQuestion 确认是否发布。

### 5. 创建发布目录

根据文章类型创建发布目录：

**单篇文章**：
```
docs/{article-name}/
├── README.md          # 文章入口（包含 frontmatter 和概述）
├── introduction.md    # 章节1（去掉编号前缀）
├── chapter.md         # 章节2（去掉编号前缀）
├── ...
└── experiments.md     # 实验与练习
```

**系列文章**：
```
docs/{series-name}/
├── README.md              # 系列入口（包含 frontmatter 和概述）
├── {name}.md              # 文章1（去掉编号前缀）
├── {name}.md              # 文章2（去掉编号前缀）
├── ...
└── assets/                # 图片资源（统一目录）
```

**关键规则**：
- **保留章节/文章结构**：不合并文件，每章/每篇独立文件
- **入口文件**：创建 `README.md` 作为文章首页，包含 frontmatter 和章节导航
- **文件命名**：**去掉编号前缀**，如 `01-introduction.md` → `introduction.md`

### 6. 创建文章入口

创建 `docs/{article-name}/README.md`：

```markdown
---
title: "{文章标题}"
date: {YYYY-MM-DD}
tags: [{标签列表}]
series:
  name: {系列名称}
  order: {序号}
---

# {文章标题}

{文章简介，来自 outline.md 的主题描述}

## 目录

- [1. 引言](./introduction.md)
- [2. {章节名称}](./{name}.md)
- [3. {章节名称}](./{name}.md)
- ...
- [实验与练习](./experiments.md)

## 目标读者

{目标读者描述}

## 系列信息

本文是「{系列名称}」系列的第 {order} 篇文章。
```

### 7. 复制文章文件

根据文章类型复制文件，**同时去掉文件名中的编号前缀**：

**单篇文章（章节模式）**：
```bash
# 复制章节文件，去掉编号前缀
# 例如: 01-introduction.md → introduction.md
for file in articles/{name}/draft/chapters/*.md; do
  filename=$(basename "$file")
  # 去掉编号前缀（如 01-、02- 等）
  newname=$(echo "$filename" | sed 's/^[0-9]*-//')
  cp "$file" "docs/{article-name}/$newname"
done

# 复制实验文件
cp articles/{name}/draft/experiments.md docs/{article-name}/experiments.md

# 复制资源文件（如果有）
cp -r articles/{name}/draft/assets docs/{article-name}/ 2>/dev/null || true
```

**系列文章**：
```bash
# 复制文章文件，去掉编号前缀
for file in articles/{series-name}/draft/*.md; do
  filename=$(basename "$file")
  newname=$(echo "$filename" | sed 's/^[0-9]*-//')
  cp "$file" "docs/{series-name}/$newname"
done

# 复制资源文件
cp -r articles/{series-name}/assets docs/{series-name}/ 2>/dev/null || true

# 复制代码示例（可选）
cp -r articles/{series-name}/code docs/{series-name}/ 2>/dev/null || true
```

**去掉编号前缀的规则**：
- `01-introduction.md` → `introduction.md`
- `02-probability-basics.md` → `probability-basics.md`
- `99-summary.md` → `summary.md`
- 使用正则表达式 `s/^[0-9]*-//` 匹配并删除开头的数字和连字符

**注意**：文件中的相对路径链接需要调整（如图片路径）。

### 8. 更新 VuePress 配置

更新 `docs/.vuepress/config.js`，添加新的侧边栏配置：

```javascript
// 在 sidebar 配置中添加
sidebar: {
  // ... 现有配置 ...
  '/{article-name}/': [
    {
      text: '{文章标题}',
      collapsible: true,
      children: [
        { text: '概述', link: '/{article-name}/' },
        { text: '1. 引言', link: '/{article-name}/introduction' },
        { text: '2. {章节名}', link: '/{article-name}/{name}' },
        // ... 其他章节 ...
        { text: '实验与练习', link: '/{article-name}/experiments' }
      ]
    }
  ]
}
```

**同时更新导航栏**（如果需要）：

```javascript
navbar: [
  // ... 现有配置 ...
  { text: '{分类名}', link: '/{article-name}/' }
]
```

### 9. 归档工作文件

将工作目录移动到归档：
```bash
mv articles/{name} articles/archive/{YYYY-MM-DD}-{name}/
```

### 10. 更新索引

更新 `articles/index.json`：
- 移除 `articles` 数组中的条目
- 增加 `stats.published` 计数

### 11. 更新归档元数据

更新归档目录中的 `.article.yaml`：
```yaml
status: published
publishedDate: {YYYY-MM-DD}
path: docs/{article-name}/
```

---

## 发布目录结构示例

```
docs/linear-algebra-vectors-matrices/
├── README.md                    # 文章入口（概述 + 目录）
├── introduction.md              # 第1章：引言（无编号前缀）
├── vectors.md                   # 第2章：向量基础
├── matrices.md                  # 第3章：矩阵基础
├── numpy.md                     # 第4章：NumPy 实践
├── applications.md              # 第5章：应用场景
├── summary.md                   # 第6章：总结
├── experiments.md               # 实验与练习
└── assets/                      # 图片等资源
    ├── vector_2d.png
    ├── vector_addition.png
    └── vector_dot_product.png
```

---

## 输出示例

```
## 文章发布完成

**文章**: linear-algebra-vectors-matrices
**标题**: 向量与矩阵运算基础

**发布位置**: docs/linear-algebra-vectors-matrices/
├── README.md (入口)
├── introduction.md
├── vectors.md
├── matrices.md
├── numpy.md
├── applications.md
├── summary.md
├── experiments.md
└── assets/

**归档位置**: articles/archive/2026-03-26-linear-algebra-vectors-matrices/

**统计**:
- 章节数: 6
- 实验数: 3
- 练习题: 3
- 校审轮数: 7

**VuePress 配置已更新** ✅

**下一步**: 运行 `npm run dev` 预览文章
```

---

## Guardrails

- 工作目录必须在 `articles/` 下，而不是 `docs/_work/`
- 必须有用户确认才能发布
- **保留章节/文章结构**：不合并为单文件
- **去掉编号前缀**：`01-introduction.md` → `introduction.md`
- 发布前检查所有必要文件存在
- 保留归档目录便于回溯
- **必须更新 VuePress 配置**：确保文章出现在侧边栏
- 如果有未确认问题，警告用户但允许继续
- 章节文件中的相对路径需要调整（如 `../assets/` → `./assets/`）
- 系列文章需要复制 `assets/` 和 `code/` 目录（如果存在），资源统一存放