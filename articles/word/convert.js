#!/usr/bin/env node
// articles/word/convert.js
// 管线：VuePress → Playwright → Pandoc HTML→DOCX

import { parseArticleList } from './lib/parse-config.js';
import { renderPage, launchBrowser } from './lib/render-page.js';
import { convertHtmlToDocx } from './lib/html-to-docx.js';
import { existsSync, mkdirSync } from 'fs';
import { resolve, join } from 'path';

const OUTPUT_DIR = resolve(import.meta.dirname, 'output');

async function main() {
  const parts = parseArticleList();

  // Count total articles
  let totalArticles = 0;
  for (const part of parts)
    for (const ch of part.chapters)
      totalArticles += ch.articles.length;
  console.log(`解析文章清单: ${parts.length} 部分, ${totalArticles} 篇文章\n`);

  console.log('启动浏览器...');
  const browser = await launchBrowser();
  console.log('浏览器已就绪\n');

  let success = 0, skipped = 0;
  const errors = [];

  for (const part of parts) {
    // Create part directory
    const partDir = join(OUTPUT_DIR, part.partDir);
    if (!existsSync(partDir)) mkdirSync(partDir, { recursive: true });

    for (const ch of part.chapters) {
      const chapterDir = join(partDir, ch.chapterDir);
      if (!existsSync(chapterDir)) mkdirSync(chapterDir, { recursive: true });

      for (const article of ch.articles) {
        const fileName = `${article.sectionTitle}.docx`;
        const outputPath = join(chapterDir, fileName);
        const pageUrl = `${article.link}.html`;

        console.log(`[${part.partDir}/${ch.chapterDir}] ${article.title}`);

        try {
          console.log('  渲染...');
          const { content, title } = await renderPage(pageUrl, browser);

          if (!content) {
            console.log('  ⚠ 跳过：页面无内容');
            skipped++;
            continue;
          }

          console.log('  转换...');
          convertHtmlToDocx(content, title || article.title, outputPath);

          console.log(`  ✓ ${fileName}`);
          success++;
        } catch (err) {
          console.error(`  ✗ 错误: ${err.message}`);
          errors.push({ article: article.title, error: err.message });
        }
      }
    }
  }

  await browser.close();

  console.log('\n' + '='.repeat(50));
  console.log(`完成：成功 ${success}，跳过 ${skipped}，失败 ${errors.length}`);
  if (errors.length > 0) {
    console.log('\n失败列表:');
    for (const e of errors) console.log(`  - ${e.article}: ${e.error}`);
  }
}

main().catch(err => {
  console.error('致命错误:', err.message);
  process.exit(1);
});
