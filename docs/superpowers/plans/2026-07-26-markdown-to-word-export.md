# Markdown 文章批量导出 Word 文档 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 config.js 中定义的中文 Markdown 文章批量转换为 Word (.docx) 文档

**Architecture:** 三步管线——Node.js 解析 config.js 获取文章清单 → 预处理 Markdown（VuePress 语法标准化） → Pandoc + Lua 过滤器转换为 DOCX

**Tech Stack:** Node.js, Pandoc (latest), Lua (Pandoc 内置), mmdc (Mermaid CLI), python-docx

## 全局约束

- 输出目录：`articles/word/output/`，无子目录
- 文件命名：`<章节序号>.<文件序号>-<slug>.docx`，章节序号从 0 开始，文件序号从 1 开始
- 仅处理中文 (`/`) locale 文章，排除附录章节
- LaTeX 公式转为 OMML（Word 原生），兼顾可编辑性和保真度
- 文章间链接转为脚注（`参见「xxx」` 格式）
- 使用 `articles/word/reference.docx` 作为样式模板（已生成）

## 文件结构

```
articles/word/
├── reference.docx              # Pandoc 样式模板（已生成）
├── ref.docx                    # 用户提供的原始参考
├── convert.js                  # 主入口：解析 → 预处理 → Pandoc 转换 → 输出
├── lib/
│   ├── parse-config.js         # 从 config.js 提取中文文章清单
│   ├── preprocess.js           # Markdown 预处理
│   └── pandoc-convert.js       # Pandoc 调用封装
├── filters/
│   ├── tip-block.lua           # 自定义容器 → 带样式段落
│   └── code-block.lua          # 代码块 →「代码清单」样式
├── output/                     # 生成的 .docx 文件
└── tmp/                        # 预处理中间产物（.md 临时文件）
```

---

### Task 1: 安装依赖

**Files:**
- Modify: 无（系统环境配置）

**Interfaces:**
- Produces: Pandoc ≥ 3.1 可用命令 `pandoc`，mmdc 可用命令 `npx mmdc`

- [ ] **Step 1: 安装最新 Pandoc**

```bash
# 下载最新 Pandoc deb 包并安装
PANDOC_VERSION=$(curl -s https://api.github.com/repos/jgm/pandoc/releases/latest | grep tag_name | head -1 | cut -d'"' -f4)
wget "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-1-amd64.deb" -O /tmp/pandoc.deb
dpkg -i /tmp/pandoc.deb
```

验证：`pandoc --version` 显示版本 ≥ 3.1

- [ ] **Step 2: 安装 Mermaid CLI**

```bash
cd /root/dmla/articles/word
npm init -y
npm install @mermaid-js/mermaid-cli
```

验证：`npx mmdc --version` 正常输出

- [ ] **Step 3: 创建目录结构**

```bash
mkdir -p /root/dmla/articles/word/{lib,filters,output,tmp}
```

- [ ] **Step 4: 提交**

```bash
git add articles/word/package.json articles/word/package-lock.json
git commit -m "chore: 初始化 Word 导出工具依赖和目录结构"
```

---

### Task 2: 文章清单解析器

**Files:**
- Create: `articles/word/lib/parse-config.js`

**Interfaces:**
- Produces: `parseArticleList()` → `Array<{filePath: string, title: string, chapterIndex: number, fileIndex: number, slug: string, chapterPath: string}>`

- [ ] **Step 1: 编写解析器**

```javascript
// articles/word/lib/parse-config.js
import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../../..');

/**
 * 从 config.js 提取中文文章清单
 * @returns {Array<{filePath: string, title: string, chapterIndex: number, fileIndex: number, slug: string, chapterPath: string}>}
 */
export function parseArticleList() {
  // 直接读取 config.js 源文本，手动解析 sidebar 结构
  // 避免 eval/import 复杂的 ES module 依赖
  const configPath = resolve(PROJECT_ROOT, 'docs/.vuepress/config.js');
  const content = readFileSync(configPath, 'utf-8');

  // 提取中文 sidebar 区域（locales['/'] 的 sidebar 数组）
  // 正则匹配：locales: { '/': { ... sidebar: [...] ... } }
  const zhMatch = content.match(/locales:\s*\{[^}]*'\/':\s*\{[\s\S]*?sidebar:\s*\[([\s\S]*?)\]\s*\}/);
  if (!zhMatch) throw new Error('无法定位中文 sidebar 配置');

  // 手动提取所有带 link 和 children 或 text 的条目
  // 处理嵌套结构
  const articles = [];
  let chapterIndex = -1;

  // 用简单的行级解析：匹配包含 link 的行
  const sidebarBlock = zhMatch[1];
  const lines = sidebarBlock.split('\n');

  let currentChapterTitle = '';
  let fileIndex = 1;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // 检测章节标题：顶级 text + children 结构
    // { text: '前言', collapsible: false, children: [ ... ] }
    const topLevelMatch = line.match(/text:\s*'([^']+)'/);
    const hasChildren = lines.slice(i, i + 5).join(' ').includes('children:');

    if (topLevelMatch && hasChildren) {
      // 检查后面几行是否有 children
      chapterIndex++;
      currentChapterTitle = topLevelMatch[1];
      fileIndex = 1; // 新章节重置文件序号

      // 排除附录
      if (currentChapterTitle.includes('附录') || chapterIndex >= 7) {
        chapterIndex = 'skip';
        continue;
      }
    }

    // 检测叶子文章节点
    // { text: '向量基础', link: '/maths/linear/vectors' }
    const leafMatch = line.match(/\{\s*text:\s*'([^']+)',\s*link:\s*'([^']+)'/);
    if (leafMatch && chapterIndex !== 'skip' && chapterIndex >= 0) {
      const title = leafMatch[1];
      let link = leafMatch[2];

      // 相对路径补全前导 /
      if (!link.startsWith('/') && !link.startsWith('http')) {
        link = '/' + link;
      }

      // 跳过外部链接和聚合页
      if (link.startsWith('http')) continue;

      const slug = link.split('/').filter(Boolean).pop();
      const filePath = resolve(PROJECT_ROOT, 'docs', link.slice(1) + '.md');

      articles.push({
        filePath,
        title,
        chapterIndex,
        fileIndex: fileIndex++,
        slug,
        chapterPath: currentChapterTitle,
      });
    }
  }

  return articles;
}
```

- [ ] **Step 2: 验证解析结果**

```bash
cd /root/dmla/articles/word
node -e "
import { parseArticleList } from './lib/parse-config.js';
const articles = parseArticleList();
console.log('总文章数:', articles.length);
console.log('章节分布:');
const byChapter = {};
for (const a of articles) {
  byChapter[a.chapterIndex] = (byChapter[a.chapterIndex] || 0) + 1;
}
for (const [ch, count] of Object.entries(byChapter)) {
  console.log('  章节', ch, ':', count, '篇');
}
console.log('前 5 篇:');
for (const a of articles.slice(0, 5)) {
  console.log('  ', a.chapterIndex + '.' + a.fileIndex, a.title, '->', a.slug);
}
"
```

预期：约 90 篇文章、7 个章节（0-6），无附录。每篇文章的文件路径对应的 .md 文件确实存在。

- [ ] **Step 3: 提交**

```bash
git add articles/word/lib/parse-config.js
git commit -m "feat: 实现文章清单解析器，从 config.js 提取中文文章列表"
```

---

### Task 3: Pandoc Lua 过滤器

**Files:**
- Create: `articles/word/filters/tip-block.lua`
- Create: `articles/word/filters/code-block.lua`

**Interfaces:**
- Consumes: Pandoc AST 节点（fenced div, code block）
- Produces: 修改后的 AST 节点（应用自定义 Word 样式）

- [ ] **Step 1: 编写自定义容器过滤器**

```lua
-- articles/word/filters/tip-block.lua
-- 将 Pandoc fenced div（::: {.tip} 等）转为带「注意」样式的段落

local CONTAINER_MAP = {
  tip = '注意',
  warning = '注意',
  danger = '注意',
  info = '注意',
  note = '注意',
  details = '注意',
}

function Div(el)
  -- 检查 fenced div 的 class
  for _, cls in ipairs(el.classes) do
    local style_name = CONTAINER_MAP[cls]
    if style_name then
      -- 创建一个带 custom-style 的 Div 包裹内容
      -- Pandoc 在 DOCX 输出时识别 custom-style 属性
      local new_div = pandoc.Div(el.content)
      new_div.attributes['custom-style'] = style_name
      return new_div
    end
  end
  return nil -- 不修改其他 Div
end
```

- [ ] **Step 2: 编写代码块过滤器**

```lua
-- articles/word/filters/code-block.lua
-- 给代码块应用「代码清单」样式

function CodeBlock(el)
  -- 给代码块添加 custom-style 属性
  -- Pandoc DOCX writer 会将此属性映射为 Word 段落样式
  el.attributes['custom-style'] = '代码清单'
  return el
end

function Code(el)
  -- 内联代码也做标记（可选，Word 对 inline code 样式支持有限）
  return el
end
```

- [ ] **Step 3: 验证过滤器语法**

```bash
pandoc --lua-filter=articles/word/filters/tip-block.lua --lua-filter=articles/word/filters/code-block.lua /dev/null -t native 2>&1
# 预期：无错误输出，过滤器加载成功
```

- [ ] **Step 4: 提交**

```bash
git add articles/word/filters/
git commit -m "feat: 添加 Pandoc Lua 过滤器，处理自定义容器和代码块样式"
```

---

### Task 4: Markdown 预处理

**Files:**
- Create: `articles/word/lib/preprocess.js`

**Interfaces:**
- Consumes: 原始 Markdown 字符串、文章元信息（`{filePath, title, slug}`）
- Produces: `preprocess(markdown, meta)` → `{processed: string, mermaidPngs: Map<string, string>}`

- [ ] **Step 1: 编写预处理函数框架**

```javascript
// articles/word/lib/preprocess.js
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { dirname, resolve } from 'path';
import { execSync } from 'child_process';

const TMP_DIR = resolve(import.meta.dirname, '../tmp');

/**
 * 预处理 Markdown，将 VuePress 特有语法转为 Pandoc 兼容形式
 */
export function preprocess(markdown, meta) {
  if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });

  let result = markdown;

  // 处理顺序很重要——依次执行所有转换
  result = convertVuePressContainers(result);
  result = convertRunnableBlocks(result);
  result = convertMermaidBlocks(result, meta);
  result = convertArticleLinks(result, meta);
  result = convertEquationRefs(result);
  result = removeTocMarkers(result);

  return result;
}
```

- [ ] **Step 2: 实现自定义容器转换**

```javascript
/**
 * ::: tip / ::: warning / ::: danger / ::: info / ::: note / ::: details
 * 转为 Pandoc fenced div: ::: {.tip}
 */
function convertVuePressContainers(markdown) {
  // 匹配 ::: type Title 开头，::: 结尾
  return markdown.replace(
    /^:::\s*(tip|warning|danger|info|note|details)\s*(.*)$\n([\s\S]*?)^:::$/gm,
    (match, type, title, content) => {
      const header = title.trim() ? `**${title.trim()}**\n\n` : '';
      return `::: {.${type}}\n${header}${content.trim()}\n:::\n`;
    }
  );
}
```

- [ ] **Step 3: 实现 runnable 代码块转换**

```javascript
/**
 * ```python runnable → ```python 并添加上方说明
 */
function convertRunnableBlocks(markdown) {
  return markdown.replace(
    /^```(\w+)\s+runnable\s*\n/gm,
    (match, lang) => {
      return '\n*原文中为可运行代码块*\n\n```' + lang + '\n';
    }
  );
}
```

- [ ] **Step 4: 实现 Mermaid 转 PNG**

```javascript
/**
 * ```mermaid ... ``` → ![mermaid](./tmp/xxx.png)
 */
function convertMermaidBlocks(markdown, meta) {
  const mermaidRegex = /^```mermaid\s*\n([\s\S]*?)```$/gm;
  const pngs = new Map();
  let counter = 0;

  const result = markdown.replace(mermaidRegex, (match, diagram) => {
    const pngName = `${meta.slug}-mermaid-${++counter}.png`;
    const pngPath = resolve(TMP_DIR, pngName);

    try {
      // 将 mermaid 源码写入临时文件
      const mmdPath = resolve(TMP_DIR, `${meta.slug}-mermaid-${counter}.mmd`);
      writeFileSync(mmdPath, diagram.trim(), 'utf-8');

      // 调用 mmdc 渲染
      execSync(
        `npx -y @mermaid-js/mermaid-cli mmdc -i "${mmdPath}" -o "${pngPath}" -b transparent`,
        { stdio: 'pipe', timeout: 30000 }
      );

      pngs.set(pngName, pngPath);
      // 图片相对路径（Pandoc 以 MD 文件所在目录为基准）
      // 这里先写输出到 tmp 的相对路径，调用方会复制到文章目录
      return `![Mermaid 图表](./tmp/${pngName})\n`;
    } catch (err) {
      // 降级：保留 mermaid 源码并添加说明
      console.warn(`  ⚠ Mermaid 渲染失败 (${meta.slug}-${counter}):`, err.message);
      return '```text\n[Mermaid 图表——原文为流程图，需要在线查看]\n```\n';
    }
  });

  return result;
}
```

- [ ] **Step 5: 实现文章间链接转脚注**

```javascript
/**
 * [文字](path/to/article.md) → 文字[^n]
 * 文末附加 [^n]: 参见「文章标题」
 */
function convertArticleLinks(markdown, meta) {
  let footnoteIndex = 0;
  const footnotes = [];

  // 匹配 Markdown 链接，且目标为 .md 文件（文章间链接）
  let result = markdown.replace(
    /\[([^\]]+)\]\(([^)]+\.md)\)/g,
    (match, text, target) => {
      footnoteIndex++;
      // 从路径提取目标文章 slug
      const targetSlug = target.split('/').pop().replace('.md', '');
      footnotes.push(`[^${footnoteIndex}]: 参见「${targetSlug}」`);
      return `${text}[^${footnoteIndex}]`;
    }
  );

  // 追加脚注定义到文末
  if (footnotes.length > 0) {
    result += '\n\n' + footnotes.join('\n');
  }

  return result;
}
```

- [ ] **Step 6: 实现 {{eq:xxx}} 公式引用处理**

```javascript
/**
 * {{eq:var-mul}} → 公式（var-mul）
 */
function convertEquationRefs(markdown) {
  return markdown.replace(
    /\{\{eq:([\w-]+)\}\}/g,
    (match, refId) => `公式（${refId}）`
  );
}
```

- [ ] **Step 7: 实现 [[toc]] 删除**

```javascript
/**
 * [[toc]] → 删除
 */
function removeTocMarkers(markdown) {
  return markdown.replace(/^\[\[toc\]\]\s*$/gm, '');
}
```

- [ ] **Step 8: 验证预处理（单元测试）**

```bash
cd /root/dmla/articles/word
node -e "
import { preprocess } from './lib/preprocess.js';

const testMd = \`::: tip 阅读建议
这是一条建议。
:::

[向量](vectors.md)是基础概念。

{{eq:var-mul}}

\`\`\`python runnable
print('hello')
\`\`\`
\`;

const meta = { slug: 'test', filePath: '/tmp/test.md' };
const result = preprocess(testMd, meta);
console.log(result);
console.log('---');
console.log('验证:');
console.log('  container:', result.includes('{.tip}') ? 'PASS' : 'FAIL');
console.log('  footnote:', result.includes('[^1]') ? 'PASS' : 'FAIL');
console.log('  eq-ref:', result.includes('公式（var-mul）') ? 'PASS' : 'FAIL');
console.log('  runnable:', result.includes('可运行代码块') ? 'PASS' : 'FAIL');
"
```

预期：所有验证输出 PASS。

- [ ] **Step 9: 提交**

```bash
git add articles/word/lib/preprocess.js
git commit -m "feat: 实现 Markdown 预处理器，处理 VuePress 特有语法"
```

---

### Task 5: Pandoc 转换封装

**Files:**
- Create: `articles/word/lib/pandoc-convert.js`

**Interfaces:**
- Consumes: 预处理后的 Markdown 文件路径、输出 .docx 路径
- Produces: `convertToDocx(mdPath, docxPath)` → `Promise<void>`

- [ ] **Step 1: 编写 Pandoc 调用封装**

```javascript
// articles/word/lib/pandoc-convert.js
import { execSync } from 'child_process';
import { existsSync } from 'fs';
import { resolve } from 'path';

const FILTER_DIR = resolve(import.meta.dirname, '../filters');
const REF_DOCX = resolve(import.meta.dirname, '../reference.docx');

/**
 * 调用 Pandoc 将 Markdown 转为 DOCX
 * @param {string} mdPath - 源 Markdown 文件路径
 * @param {string} docxPath - 目标 DOCX 文件路径
 */
export function convertToDocx(mdPath, docxPath) {
  if (!existsSync(mdPath)) {
    throw new Error(`Markdown 文件不存在: ${mdPath}`);
  }

  if (!existsSync(REF_DOCX)) {
    throw new Error(`参考模板不存在: ${REF_DOCX}`);
  }

  const args = [
    'pandoc',
    mdPath,
    '-o', docxPath,
    '--from=markdown+footnotes+pipe_tables+fenced_divs+bracketed_spans',
    `--reference-doc=${REF_DOCX}`,
    `--lua-filter=${resolve(FILTER_DIR, 'tip-block.lua')}`,
    `--lua-filter=${resolve(FILTER_DIR, 'code-block.lua')}`,
    '--wrap=none',
    '--quiet',
  ];

  execSync(args.join(' '), { stdio: 'pipe', timeout: 60000 });

  if (!existsSync(docxPath)) {
    throw new Error(`Pandoc 转换失败，未生成输出文件: ${docxPath}`);
  }
}
```

- [ ] **Step 2: 验证单文件转换**

```bash
cd /root/dmla/articles/word

# 用一篇含公式的文章做测试
node -e "
import { convertToDocx } from './lib/pandoc-convert.js';

// 测试转换一篇数学文章
const mdPath = '/root/dmla/docs/maths/linear/vectors.md';
const docxPath = '/root/dmla/articles/word/output/test-vectors.docx';

try {
  convertToDocx(mdPath, docxPath);
  console.log('转换成功:', docxPath);
} catch (err) {
  console.error('转换失败:', err.message);
}
"

# 检查输出文件
ls -lh /root/dmla/articles/word/output/test-vectors.docx
```

验证生成的 DOCX 文件存在且大小合理（> 10KB）。用 `python3 -c "from docx import Document; d=Document('articles/word/output/test-vectors.docx'); print('段落数:', len(d.paragraphs))"` 确认有内容。

- [ ] **Step 3: 提交**

```bash
git add articles/word/lib/pandoc-convert.js
git commit -m "feat: 实现 Pandoc 转换封装，支持参考模板和 Lua 过滤器"
```

---

### Task 6: 主入口脚本

**Files:**
- Create: `articles/word/convert.js`

**Interfaces:**
- Consumes: parseArticleList(), preprocess(), convertToDocx()
- Produces: `articles/word/output/*.docx`

- [ ] **Step 1: 编写主入口**

```javascript
#!/usr/bin/env node
// articles/word/convert.js
// 批量将中文 Markdown 文章转为 Word 文档

import { parseArticleList } from './lib/parse-config.js';
import { preprocess } from './lib/preprocess.js';
import { convertToDocx } from './lib/pandoc-convert.js';
import { readFileSync, writeFileSync, mkdirSync, existsSync, copyFileSync } from 'fs';
import { resolve, dirname } from 'path';

const OUTPUT_DIR = resolve(import.meta.dirname, 'output');
const TMP_DIR = resolve(import.meta.dirname, 'tmp');

function main() {
  // 确保输出目录存在
  if (!existsSync(OUTPUT_DIR)) mkdirSync(OUTPUT_DIR, { recursive: true });
  if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });

  // 1. 解析文章清单
  console.log('解析文章清单...');
  const articles = parseArticleList();
  console.log(`共 ${articles.length} 篇文章\n`);

  // 2. 逐篇处理
  let success = 0;
  let skipped = 0;
  const errors = [];

  for (const article of articles) {
    const outputFileName = `${article.chapterIndex}.${article.fileIndex}-${article.slug}.docx`;
    const outputPath = resolve(OUTPUT_DIR, outputFileName);

    console.log(`[${article.chapterIndex}.${article.fileIndex}] ${article.title}`);

    try {
      // 读取原始 Markdown
      if (!existsSync(article.filePath)) {
        console.log(`  ⚠ 跳过：文件不存在 ${article.filePath}`);
        skipped++;
        continue;
      }

      let markdown = readFileSync(article.filePath, 'utf-8');

      // 预处理
      console.log('  预处理...');
      markdown = preprocess(markdown, { slug: article.slug, filePath: article.filePath, title: article.title });

      // 写入临时 Markdown 文件
      const tmpMdPath = resolve(TMP_DIR, `${article.slug}.md`);
      writeFileSync(tmpMdPath, markdown, 'utf-8');

      // Pandoc 转换
      console.log('  转换中...');
      convertToDocx(tmpMdPath, outputPath);

      console.log(`  ✓ ${outputFileName}`);
      success++;
    } catch (err) {
      console.error(`  ✗ 错误: ${err.message}`);
      errors.push({ article: `${article.chapterIndex}.${article.fileIndex} ${article.title}`, error: err.message });
    }
  }

  // 3. 输出汇总
  console.log('\n' + '='.repeat(50));
  console.log(`完成：成功 ${success}，跳过 ${skipped}，失败 ${errors.length}`);
  if (errors.length > 0) {
    console.log('\n失败列表:');
    for (const e of errors) {
      console.log(`  - ${e.article}: ${e.error}`);
    }
  }
}

main();
```

- [ ] **Step 2: 运行批量转换**

```bash
cd /root/dmla/articles/word
node convert.js
```

- [ ] **Step 3: 验证输出完整性和质量**

```bash
# 检查输出文件数量
ls articles/word/output/*.docx | wc -l
echo "预期数量: ~90 个文件"

# 抽查一个数学密集文章
python3 -c "
from docx import Document
d = Document('articles/word/output/3.1-idea-origin.docx')
print(f'段落数: {len(d.paragraphs)}')
# 检查包含 OMML 公式（Word math namespace）
import zipfile, os
with zipfile.ZipFile('articles/word/output/3.1-idea-origin.docx', 'r') as z:
    has_math = any('math' in f.lower() for f in z.namelist())
    print(f'含 LaTeX/OMML 公式: {has_math}')
"
```

- [ ] **Step 4: 提交**

```bash
git add articles/word/convert.js articles/word/output/.gitkeep
git commit -m "feat: 实现批量转换主入口脚本，完整管线打通"
```

---

### Task 7: 端到端验证与清理

**Files:**
- 无新建文件（验证步骤）

**Interfaces:**
- 无（最终验证）

- [ ] **Step 1: 清理测试产物，重新完整运行**

```bash
rm -f articles/word/output/test-*.docx
rm -rf articles/word/tmp/*
cd /root/dmla/articles/word
node convert.js
```

- [ ] **Step 2: 抽查 3 篇文章的输出质量**

```bash
python3 << 'PYEOF'
from docx import Document

# 抽查：1 篇数学文章、1 篇含代码文章、1 篇含提示块文章
test_files = [
    ('数学文章', 'output/1.1-vectors.docx'),
    ('代码文章', 'output/1.9-boosting.docx'),
    ('综合文章', 'output/4.1-transformer-architecture.docx'),
]

for label, path in test_files:
    try:
        d = Document(path)
        heading_count = sum(1 for p in d.paragraphs if p.style.name.startswith('Heading'))
        normal_count = sum(1 for p in d.paragraphs if p.style.name == 'Normal')
        print(f'{label} ({path}): {len(d.paragraphs)} 段, {heading_count} 个标题, {normal_count} 个正文段落')
    except Exception as e:
        print(f'{label} ({path}): 错误 - {e}')
PYEOF
```

- [ ] **Step 3: 清理临时文件**

```bash
rm -rf articles/word/tmp
echo "tmp/" >> articles/word/.gitignore
```

- [ ] **Step 4: 最终提交**

```bash
git add articles/word/.gitignore
git commit -m "chore: 清理临时文件，添加 .gitignore"
```

---

## 自审清单

### 规格覆盖

| 规格要求 | 对应任务 |
|----------|----------|
| 从 config.js 提取中文文章清单 | Task 2 |
| 排除附录 | Task 2 |
| `::: tip/info/warning/danger/note/details` → fenced div | Task 4 Step 2 |
| fenced div → 样式段落 | Task 3 Step 1 |
| runnable 代码块 → 标注说明 | Task 4 Step 3 |
| Mermaid → PNG 图片 | Task 4 Step 4 |
| 文章间链接 → 脚注 | Task 4 Step 5 |
| `{{eq:xxx}}` → 公式（xxx） | Task 4 Step 6 |
| `[[toc]]` → 删除 | Task 4 Step 7 |
| Pandoc 调用 + Lua 过滤器 | Task 5 |
| 参考模板 reference.docx | Task 5（已生成） |
| 数学公式 OMML | Task 5（Pandoc 默认行为） |
| 文件命名 `<章节>.<序号>-<slug>.docx` | Task 6 |
| 输出到 `articles/word/output/` | Task 6 |
| `参考（reference.docx）样式应用` | Task 5（`--reference-doc`） |
| 代码块「代码清单」样式 | Task 3 Step 2 |
| 提示块「注意」样式 | Task 3 Step 1 |

### 占位符扫描

无 TBD/TODO/占位符。所有步骤包含可执行的命令或代码。

### 类型一致性

- `parseArticleList()` 返回对象字段名在 Task 6 中使用一致：`chapterIndex`, `fileIndex`, `slug`, `title`, `filePath`
- `preprocess()` 签名在 Task 4 定义、Task 6 调用一致
- `convertToDocx(mdPath, docxPath)` 在 Task 5 定义、Task 6 调用一致
