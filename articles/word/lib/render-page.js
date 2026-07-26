// articles/word/lib/render-page.js
// 使用 Playwright + 系统 Chrome 渲染 VuePress 页面，提取最终 DOM
import { chromium } from 'playwright';
import { existsSync } from 'fs';

const BASE_URL = 'http://localhost:8080';
const BROWSER_PATH = '/usr/bin/google-chrome';

/**
 * 渲染一个 VuePress 页面并提取内容 HTML
 * @param {string} articlePath - 文章路径，如 /maths/linear/vectors.html
 * @param {object} browser - 复用的 Playwright browser 实例
 * @returns {Promise<{content: string, title: string}>}
 */
export async function renderPage(articlePath, browser) {
  const page = await browser.newPage();

  try {
    const url = `${BASE_URL}${articlePath}`;
    await page.goto(url, {
      waitUntil: 'networkidle',
      timeout: 15000
    });

    // 等待动态图表渲染（Mermaid、nn-arch 等）
    await page.waitForTimeout(2000);

    // 获取文章标题
    const pageTitle = await page.title();
    const h1Text = await page.evaluate(() => {
      const h1 = document.querySelector('h1');
      return h1 ? h1.textContent.trim() : '';
    });
    const title = h1Text || pageTitle;

    // 提取 KaTeX 公式的原始 LaTeX 并替换 HTML
    // KaTeX 的 annotation 元素包含原始 LaTeX 源码
    await page.evaluate(() => {
      // 1) 块级公式 (katex-display): 替换为 $$...$$ 供 Pandoc 转为 OMML
      document.querySelectorAll('.katex-display').forEach(el => {
        const annotation = el.querySelector('annotation[encoding="application/x-tex"]');
        if (annotation && annotation.textContent.trim()) {
          const tex = annotation.textContent.trim();
          const container = document.createElement('p');
          container.textContent = `$$${tex}$$`;
          el.replaceWith(container);
        }
      });

      // 2) 行内公式: 替换为 $...$
      // 注意：katex-display 内部的 .katex 已经被上面处理了
      document.querySelectorAll('.katex').forEach(el => {
        // 跳过已被替换的（父级是 katex-display 的情况）
        if (el.closest('.katex-display')) return;
        const annotation = el.querySelector('annotation[encoding="application/x-tex"]');
        if (annotation && annotation.textContent.trim()) {
          const tex = annotation.textContent.trim();
          // 检查是否已经是 $...$ 包裹的
          if (!tex.startsWith('$')) {
            const span = document.createElement('span');
            span.textContent = `$${tex}$`;
            el.replaceWith(span);
          }
        }
      });
    });

    // 提取内容区域 HTML
    const contentHtml = await page.evaluate(() => {
      const content = document.querySelector('[vp-content]');
      if (!content) return '';

      // 克隆内容以避免修改原始 DOM
      const clone = content.cloneNode(true);

      // 移除不需要的元素
      clone.querySelectorAll('.runnable-code-toolbar, .runnable-code-btn, '
        + '.code-demo, .giscus, .page-nav, .page-meta, '
        + 'script, style, .vuepress-plugin-search-pro').forEach(el => el.remove());

      return clone.innerHTML;
    });

    return { content: contentHtml, title };

  } finally {
    await page.close();
  }
}

/**
 * 启动共享的浏览器实例（整个批量转换共用一个浏览器）
 * @returns {Promise<import('playwright').Browser>}
 */
export async function launchBrowser() {
  return chromium.launch({
    headless: true,
    executablePath: BROWSER_PATH,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
    ],
  });
}
