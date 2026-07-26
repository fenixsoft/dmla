import { execFileSync } from 'child_process';
import { existsSync } from 'fs';
import { resolve } from 'path';

const FILTER_DIR = resolve(import.meta.dirname, '../filters');
const REF_DOCX = resolve(import.meta.dirname, '../reference.docx');

/** 检测 pandoc 是否可用 */
function checkPandoc() {
  try {
    execFileSync('pandoc', ['--version'], { stdio: 'ignore' });
  } catch {
    throw new Error('pandoc 命令不可用，请确认已安装 Pandoc >= 3.1');
  }
}

/**
 * 调用 Pandoc 将 Markdown 转为 DOCX
 * @param {string} mdPath - 源 Markdown 文件路径
 * @param {string} docxPath - 目标 DOCX 文件路径
 */
export function convertToDocx(mdPath, docxPath) {
  checkPandoc();

  if (!existsSync(mdPath)) {
    throw new Error(`Markdown 文件不存在: ${mdPath}`);
  }

  if (!existsSync(REF_DOCX)) {
    throw new Error(`参考模板不存在: ${REF_DOCX}`);
  }

  const args = [
    mdPath,
    '-o', docxPath,
    '--from=markdown+footnotes+pipe_tables+fenced_divs+bracketed_spans',
    `--reference-doc=${REF_DOCX}`,
    `--lua-filter=${resolve(FILTER_DIR, 'tip-block.lua')}`,
    `--lua-filter=${resolve(FILTER_DIR, 'code-block.lua')}`,
    '--wrap=none',
    '--quiet',
  ];

  execFileSync('pandoc', args, { stdio: 'pipe', timeout: 60000 });

  if (!existsSync(docxPath)) {
    throw new Error(`Pandoc 转换失败，未生成输出文件: ${docxPath}`);
  }
}
