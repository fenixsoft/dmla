#!/usr/bin/env node
// articles/word/convert.js
// 批量将中文 Markdown 文章转为 Word 文档
// 管线：VuePress 开发服务器 → Playwright 渲染 → Pandoc HTML→DOCX

import { parseArticleList } from './lib/parse-config.js';
import { renderPage, launchBrowser } from './lib/render-page.js';
import { convertHtmlToDocx } from './lib/html-to-docx.js';
import { existsSync, mkdirSync } from 'fs';
import { resolve } from 'path';

const OUTPUT_DIR = resolve(import.meta.dirname, 'output');

async function main() {
  // 确保输出目录存在
  if (!existsSync(OUTPUT_DIR)) mkdirSync(OUTPUT_DIR, { recursive: true });

  // 1. 解析文章清单
  console.log('解析文章清单...');
  const articles = parseArticleList();
  console.log(`共 ${articles.length} 篇文章\n`);

  // 2. 启动浏览器
  console.log('启动浏览器...');
  const browser = await launchBrowser();
  console.log('浏览器已就绪\n');

  // 3. 逐篇处理
  let success = 0;
  let skipped = 0;
  const errors = [];

  for (const article of articles) {
    const outputFileName = `${article.chapterIndex}.${article.fileIndex}-${article.slug}.docx`;
    const outputPath = resolve(OUTPUT_DIR, outputFileName);
    const pageUrl = `${article.link}.html`;

    console.log(`[${article.chapterIndex}.${article.fileIndex}] ${article.title}`);

    try {
      // Playwright 渲染 + 提取内容
      console.log('  渲染...');
      const { content, title } = await renderPage(pageUrl, browser);

      if (!content) {
        console.log('  ⚠ 跳过：页面无内容');
        skipped++;
        continue;
      }

      // HTML → DOCX
      console.log('  转换...');
      convertHtmlToDocx(content, title || article.title, outputPath);

      console.log(`  ✓ ${outputFileName}`);
      success++;
    } catch (err) {
      console.error(`  ✗ 错误: ${err.message}`);
      errors.push({
        article: `${article.chapterIndex}.${article.fileIndex} ${article.title}`,
        error: err.message
      });
    }
  }

  // 4. 关闭浏览器
  await browser.close();

  // 5. 输出汇总
  console.log('\n' + '='.repeat(50));
  console.log(`完成：成功 ${success}，跳过 ${skipped}，失败 ${errors.length}`);
  if (errors.length > 0) {
    console.log('\n失败列表:');
    for (const e of errors) {
      console.log(`  - ${e.article}: ${e.error}`);
    }
  }
}

main().catch(err => {
  console.error('致命错误:', err.message);
  process.exit(1);
});
