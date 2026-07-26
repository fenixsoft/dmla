// articles/word/lib/html-to-docx.js
// 使用 Pandoc 将 HTML 内容转为 DOCX
import { execFileSync } from 'child_process';
import { existsSync, writeFileSync, mkdirSync } from 'fs';
import { resolve } from 'path';

const TMP_DIR = resolve(import.meta.dirname, '../tmp');
const REF_DOCX = resolve(import.meta.dirname, '../reference.docx');

/**
 * 检查 pandoc 命令是否可用
 */
function checkPandoc() {
  try {
    execFileSync('pandoc', ['--version'], { stdio: 'pipe', timeout: 5000 });
  } catch {
    throw new Error('Pandoc 未安装或不可用，请先运行 npm run setup');
  }
}

/**
 * 将 HTML 内容转为 DOCX 文件
 * @param {string} htmlContent - HTML 内容字符串
 * @param {string} title - 文章标题
 * @param {string} docxPath - 输出 DOCX 路径
 */
export function convertHtmlToDocx(htmlContent, title, docxPath) {
  checkPandoc();

  if (!existsSync(REF_DOCX)) {
    throw new Error(`参考模板不存在: ${REF_DOCX}`);
  }

  // 确保临时目录存在
  if (!existsSync(TMP_DIR)) {
    mkdirSync(TMP_DIR, { recursive: true });
  }

  // 包装为完整 HTML 文档，便于 Pandoc 解析
  const fullHtml = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>${escapeHtml(title)}</title>
<style>
  pre { background: #f5f5f5; padding: 1em; overflow-x: auto; font-family: Consolas, Monaco, monospace; font-size: 0.9em; line-height: 1.5; }
  code { font-family: Consolas, Monaco, monospace; font-size: 0.9em; background: #f0f0f0; padding: 0.1em 0.3em; }
  table { border-collapse: collapse; width: 100%; }
  table th, table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
  table th { background: #f5f5f5; font-weight: bold; }
  img { max-width: 100%; height: auto; }
  blockquote { border-left: 4px solid #ddd; margin: 1em 0; padding: 0.5em 1em; color: #666; }
  .custom-container { border-left: 4px solid; padding: 0.5em 1em; margin: 1em 0; }
  .custom-container.tip { border-color: #42b983; background: #f0faf5; }
  .custom-container.warning { border-color: #e7c000; background: #fffdf0; }
  .custom-container.danger { border-color: #c00; background: #fff0f0; }
  .custom-container.info { border-color: #0070f3; background: #f0f7ff; }
  .custom-container.details { border-color: #888; background: #f8f8f8; }
</style>
</head>
<body>
<h1>${escapeHtml(title)}</h1>
${htmlContent}
</body>
</html>`;

  // 写入临时 HTML 文件
  const tmpHtmlPath = resolve(TMP_DIR, `${sanitizeFileName(title)}.html`);
  writeFileSync(tmpHtmlPath, fullHtml, 'utf-8');

  // Pandoc HTML → DOCX
  // 使用 --from=html 让 Pandoc 解析 HTML 输入
  const pandocArgs = [
    tmpHtmlPath,
    '-o', docxPath,
    '--from=html',
    `--reference-doc=${REF_DOCX}`,
    '--wrap=none',
    '--quiet',
  ];

  execFileSync('pandoc', pandocArgs, { stdio: 'pipe', timeout: 60000 });

  if (!existsSync(docxPath)) {
    throw new Error(`Pandoc 转换失败，未生成输出文件: ${docxPath}`);
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function sanitizeFileName(name) {
  return name.replace(/[<>:"/\\|?*]/g, '_').substring(0, 50);
}
