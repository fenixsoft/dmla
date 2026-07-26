// articles/word/lib/render-page.js
// 使用 Playwright + 系统 Chrome 渲染 VuePress 页面，提取最终 DOM
import { chromium } from 'playwright';
import { existsSync } from 'fs';
import { resolve } from 'path';

const BASE_URL = 'http://localhost:8080';
const BROWSER_PATH = '/usr/bin/google-chrome';
// KaTeX 路径：VuePress 的 math 插件通过 ES module 加载，未暴露全局
// 需手动注入脚本以便在浏览器中使用 katex.renderToString()
const KATEX_PATH = resolve(import.meta.dirname, '../../../node_modules/katex/dist/katex.min.js');

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

    // 注入 KaTeX 全局函数（VuePress 通过 ES module 加载，未暴露全局）
    if (existsSync(KATEX_PATH)) {
      await page.addScriptTag({ path: KATEX_PATH });
    }

    // 将 KaTeX 公式转换为 MathML，Pandoc HTML reader 会转为 OMML
    // - 块级公式：<math display="block"> → m:oMathPara
    // - 行内公式：<math> → m:oMath
    const formulaStats = await page.evaluate(() => {
      if (typeof katex === 'undefined') return { error: 'KaTeX not loaded' };

      let displayDone = 0, inlineDone = 0, errors = 0;

      // Step 1: 块级 display 公式 → <math display="block">
      document.querySelectorAll('.katex-display').forEach(el => {
        const annotation = el.querySelector('annotation[encoding="application/x-tex"]');
        if (!annotation || !annotation.textContent.trim()) return;
        const tex = annotation.textContent.trim();
        try {
          const mathml = katex.renderToString(tex, {
            output: 'mathml', throwOnError: false, displayMode: true,
          });
          const tmp = document.createElement('div');
          tmp.innerHTML = mathml;
          const mathEl = tmp.querySelector('math');
          if (mathEl) {
            const p = document.createElement('p');
            p.appendChild(mathEl);
            el.replaceWith(p);
            displayDone++;
          }
        } catch (e) { errors++; }
      });

      // Step 2: 行内公式 → <math>（不带 display 属性）
      document.querySelectorAll('.katex').forEach(el => {
        if (el.closest('.katex-display')) return; // 已在 step 1 处理
        const annotation = el.querySelector('annotation[encoding="application/x-tex"]');
        if (!annotation || !annotation.textContent.trim()) return;
        const tex = annotation.textContent.trim();
        try {
          const mathml = katex.renderToString(tex, {
            output: 'mathml', throwOnError: false, displayMode: false,
          });
          const tmp = document.createElement('div');
          tmp.innerHTML = mathml;
          const mathEl = tmp.querySelector('math');
          if (mathEl) {
            el.replaceWith(mathEl);
            inlineDone++;
          }
        } catch (e) { errors++; }
      });

      return {
        displayDone, inlineDone, errors,
        katexLeft: document.querySelectorAll('.katex').length,
        mathCount: document.querySelectorAll('math').length,
      };
    });
    console.log(`  公式转换: display=${formulaStats.displayDone} inline=${formulaStats.inlineDone} errors=${formulaStats.errors} katex_left=${formulaStats.katexLeft} math=${formulaStats.mathCount}`);

    // 提取内容区域 HTML
    const contentHtml = await page.evaluate(() => {
      const content = document.querySelector('[vp-content]');
      if (!content) return '';

      const clone = content.cloneNode(true);

      // 移除不需要的元素
      clone.querySelectorAll(
        '.runnable-code-toolbar, .runnable-code-btn, '
        + '.code-demo, .giscus, .page-nav, .page-meta, '
        + 'script, style, .vuepress-plugin-search-pro'
      ).forEach(el => el.remove());

      return clone.innerHTML;
    });

    // 将相对图片路径转为绝对路径（Pandoc 需要文件系统路径）
    const docsDir = resolve(import.meta.dirname, '../../../docs');
    const fixedHtml = contentHtml.replace(
      /<img\s+[^>]*src="(\/[^"]+)"/g,
      (match, srcPath) => {
        const absPath = resolve(docsDir, srcPath.slice(1));
        return match.replace(srcPath, absPath);
      }
    );

    return { content: fixedHtml, title };

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
