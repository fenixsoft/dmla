#!/usr/bin/env node
// articles/word/convert.js
// 批量将中文 Markdown 文章转为 Word 文档

import { parseArticleList } from './lib/parse-config.js';
import { preprocess } from './lib/preprocess.js';
import { convertToDocx } from './lib/pandoc-convert.js';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { resolve } from 'path';

const OUTPUT_DIR = resolve(import.meta.dirname, 'output');
const TMP_DIR = resolve(import.meta.dirname, 'tmp');

function main() {
  // 确保输出和临时目录存在
  if (!existsSync(OUTPUT_DIR)) mkdirSync(OUTPUT_DIR, { recursive: true });
  if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });

  // 1. 解析文章清单
  console.log('解析文章清单...');
  const articles = parseArticleList();
  console.log(`共 ${articles.length} 篇文章\n`);

  // 构建 slug → 标题映射，用于脚注显示文章标题
  const titleMap = Object.fromEntries(articles.map(a => [a.slug, a.title]));

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
      const processed = preprocess(markdown, {
        slug: article.slug,
        filePath: article.filePath,
        title: article.title,
      }, titleMap);
      markdown = processed.processed;

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
