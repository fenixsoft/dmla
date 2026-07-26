import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { resolve } from 'path';
import { execSync } from 'child_process';

const TMP_DIR = resolve(import.meta.dirname, '../tmp');
const PUPPETEER_CONFIG = resolve(import.meta.dirname, '../puppeteer-config.json');

/** 收集 Mermaid 渲染的 PNG 文件信息 */
const _mermaidPngs = new Map();

/**
 * 预处理 Markdown，将 VuePress 特有语法转为 Pandoc 兼容形式
 * @param {string} markdown - 原始 Markdown 内容
 * @param {{slug: string, filePath: string, title: string}} meta - 文章元信息
 * @returns {{processed: string, mermaidPngs: Map<string, string>}}
 */
export function preprocess(markdown, meta) {
  if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });
  _mermaidPngs.clear();

  let result = markdown;

  // 处理顺序很重要——依次执行所有转换
  result = convertVuePressContainers(result);
  result = convertRunnableBlocks(result);
  result = convertMermaidBlocks(result, meta);
  result = convertArticleLinks(result, meta);
  result = convertEquationRefs(result);
  result = removeTocMarkers(result);

  return { processed: result, mermaidPngs: new Map(_mermaidPngs) };
}

// ─── 容器转换 ────────────────────────────────────────────────

/**
 * ::: tip / ::: warning / ::: danger / ::: info / ::: note / ::: details
 * 转为 Pandoc fenced div: ::: {.tip}
 * @param {string} markdown
 * @returns {string}
 */
function convertVuePressContainers(markdown) {
  return markdown.replace(
    /^:::\s*(tip|warning|danger|info|note|details)\s*(.*)$\n([\s\S]*?)^:::$/gm,
    (match, type, title, content) => {
      const header = title.trim() ? `**${title.trim()}**\n\n` : '';
      return `::: {.${type}}\n${header}${content.trim()}\n:::\n`;
    }
  );
}

// ─── Runnable 代码块转换 ────────────────────────────────────

/**
 * ```python runnable → ```python 并添加上方说明
 * @param {string} markdown
 * @returns {string}
 */
function convertRunnableBlocks(markdown) {
  return markdown.replace(
    /^```(\w+)\s+runnable\s*\n/gm,
    (match, lang) => {
      return '\n*原文中为可运行代码块*\n\n```' + lang + '\n';
    }
  );
}

// ─── Mermaid 图表转换 ───────────────────────────────────────

/**
 * ```mermaid ... ``` → ![mermaid](./tmp/xxx.png)
 * @param {string} markdown
 * @param {{slug: string}} meta
 * @returns {string}
 */
function convertMermaidBlocks(markdown, meta) {
  const mermaidRegex = /^```mermaid\s*\n([\s\S]*?)```$/gm;
  let counter = 0;

  const result = markdown.replace(mermaidRegex, (match, diagram) => {
    counter++;
    const pngName = `${meta.slug}-mermaid-${counter}.png`;
    const pngPath = resolve(TMP_DIR, pngName);

    try {
      // 将 mermaid 源码写入临时文件
      const mmdPath = resolve(TMP_DIR, `${meta.slug}-mermaid-${counter}.mmd`);
      writeFileSync(mmdPath, diagram.trim(), 'utf-8');

      // 调用 mmdc 渲染
      execSync(
        `npx mmdc -i "${mmdPath}" -o "${pngPath}" -b transparent -p "${PUPPETEER_CONFIG}"`,
        { stdio: 'pipe', timeout: 30000 }
      );

      _mermaidPngs.set(pngName, pngPath);
      return `![Mermaid 图表](./tmp/${pngName})\n`;
    } catch (err) {
      console.warn(`  [警告] Mermaid 渲染失败 (${meta.slug}-${counter}): ${err.message}`);
      return '\n```text\n[Mermaid 图表——原文为流程图，需要在线查看]\n```\n';
    }
  });

  return result;
}

// ─── 文章间链接转脚注 ──────────────────────────────────────

/**
 * [文字](path/to/article.md) → 文字[^n]
 * 文末附加 [^n]: 参见「文章标题」
 * @param {string} markdown
 * @param {{slug: string}} meta
 * @returns {string}
 */
function convertArticleLinks(markdown, meta) {
  let footnoteIndex = 0;
  const footnotes = [];

  const result = markdown.replace(
    /\[([^\]]+)\]\(([^)]+\.md)\)/g,
    (match, text, target) => {
      footnoteIndex++;
      const targetSlug = target.split('/').pop().replace('.md', '');
      footnotes.push(`[^${footnoteIndex}]: 参见「${targetSlug}」`);
      return `${text}[^${footnoteIndex}]`;
    }
  );

  if (footnotes.length > 0) {
    return result + '\n\n' + footnotes.join('\n');
  }

  return result;
}

// ─── 公式引用处理 ──────────────────────────────────────────

/**
 * {{eq:var-mul}} → 公式（var-mul）
 * @param {string} markdown
 * @returns {string}
 */
function convertEquationRefs(markdown) {
  return markdown.replace(
    /\{\{eq:([\w-]+)\}\}/g,
    (match, refId) => `公式（${refId}）`
  );
}

// ─── [[toc]] 删除 ───────────────────────────────────────────

/**
 * [[toc]] → 删除
 * @param {string} markdown
 * @returns {string}
 */
function removeTocMarkers(markdown) {
  return markdown.replace(/^\[\[toc\]\]\s*$/gm, '');
}
