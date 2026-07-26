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
 * @param {Record<string, string>} [titleMap] - slug → 文章标题映射，用于脚注显示标题而非 slug
 * @returns {{processed: string, mermaidPngs: Map<string, string>}}
 */
export function preprocess(markdown, meta, titleMap) {
  if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });
  _mermaidPngs.clear();

  let result = markdown;

  // 处理顺序很重要——依次执行所有转换
  result = convertVuePressContainers(result);
  result = convertRunnableBlocks(result);
  result = convertMermaidBlocks(result, meta);

  // 验证图片文件存在性（需在 Mermaid 转换之后执行）
  result = validateImages(result, meta);

  // 文章间链接转换需保护代码块，防止正则误匹配反引号内的内容
  const codeBlockGuard = protectCodeBlocks(result);
  codeBlockGuard.text = convertArticleLinks(codeBlockGuard.text, titleMap);
  result = restoreCodeBlocks(codeBlockGuard.text, codeBlockGuard.blocks);

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

      // 调用 mmdc 渲染（-y 跳过交互提示）
      execSync(
        `npx -y mmdc -i "${mmdPath}" -o "${pngPath}" -b transparent -p "${PUPPETEER_CONFIG}"`,
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

// ─── 图片文件验证 ────────────────────────────────────────

/**
 * 验证所有 Markdown 图片引用对应的文件是否存在
 * 缺失的图片替换为占位文字，避免 Word 中出现断裂的图片链接
 * @param {string} markdown
 * @param {{slug: string, filePath: string}} meta
 * @returns {string}
 */
function validateImages(markdown, meta) {
  const articleDir = resolve(meta.filePath, '..');
  return markdown.replace(
    /!\[([^\]]*)\]\(([^)]+)\)/g,
    (match, alt, path) => {
      // 去除可能的 title 属性（如 image.png "title"）
      const cleanPath = path.split(/\s+/)[0];

      // 跳过外部 URL 和 data URI
      if (cleanPath.startsWith('http://') || cleanPath.startsWith('https://') || cleanPath.startsWith('data:')) {
        return match;
      }
      // 跳过 Mermaid 渲染的临时图片（已在 convertMermaidBlocks 中生成）
      if (cleanPath.startsWith('./tmp/') || cleanPath.startsWith('tmp/')) {
        return match;
      }

      const fullPath = resolve(articleDir, cleanPath);
      if (!existsSync(fullPath)) {
        console.warn(`  [警告] 图片缺失: ${cleanPath}（${meta.slug}）`);
        return `[图片缺失: ${cleanPath}]`;
      }
      return match;
    }
  );
}

// ─── 代码块保护（防止 convertArticleLinks 误匹配） ────────

/**
 * 将代码块和行内代码替换为占位符，防止后续正则处理时误匹配
 * @param {string} text
 * @returns {{text: string, blocks: string[]}}
 */
function protectCodeBlocks(text) {
  const blocks = [];
  let index = 0;
  // 保护 fenced code blocks
  text = text.replace(/```[\s\S]*?```/g, (match) => {
    const placeholder = `\x00CODEBLOCK${index}\x00`;
    blocks[index++] = match;
    return placeholder;
  });
  // 保护 inline code (backticks)
  text = text.replace(/`[^`]+`/g, (match) => {
    const placeholder = `\x00CODEBLOCK${index}\x00`;
    blocks[index++] = match;
    return placeholder;
  });
  return { text, blocks };
}

/**
 * 将占位符恢复为原始代码块内容
 * @param {string} text
 * @param {string[]} blocks
 * @returns {string}
 */
function restoreCodeBlocks(text, blocks) {
  return text.replace(/\x00CODEBLOCK(\d+)\x00/g, (_, index) => blocks[parseInt(index)]);
}

// ─── 文章间链接转脚注 ──────────────────────────────────────

/**
 * [文字](path/to/article.md) → 文字[^n]
 * 文末附加 [^n]: 参见「文章标题」
 * @param {string} markdown
 * @param {Record<string, string>} [titleMap] - slug → 文章标题映射
 * @returns {string}
 */
function convertArticleLinks(markdown, titleMap) {
  let footnoteIndex = 0;
  const footnotes = [];

  const result = markdown.replace(
    /\[([^\]]+)\]\(([^)"'\s]+\.md)(?:\s+"[^"]*")?\)/g,
    (match, text, target) => {
      footnoteIndex++;
      const targetSlug = target.split('/').pop().replace('.md', '');
      const title = titleMap?.[targetSlug] || targetSlug;
      footnotes.push(`[^${footnoteIndex}]: 参见「${title}」`);
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
