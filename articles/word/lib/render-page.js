// articles/word/lib/render-page.js
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, writeFileSync, unlinkSync } from 'fs';
import { resolve } from 'path';
import { execFileSync } from 'child_process';

const BASE_URL = 'http://localhost:8080';
const BROWSER_PATH = '/usr/bin/google-chrome';
const KATEX_PATH = resolve(import.meta.dirname, '../../../node_modules/katex/dist/katex.min.js');
const TMP_DIR = resolve(import.meta.dirname, '../tmp');
const DOCS_DIR = resolve(import.meta.dirname, '../../../docs');
const PUPPETEER_CONFIG = resolve(import.meta.dirname, '../puppeteer-config.json');

export async function renderPage(articlePath, browser) {
  const page = await browser.newPage();

  try {
    const slug = articlePath.split('/').pop().replace('.html', '');
    if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });

    // ---- 0) mmdc 渲染 mermaid（在浏览器加载前完成） ----
    const mermaidImgs = renderMermaidBlocks(articlePath, slug);

    // ---- 1) 浏览器加载 ----
    await page.setViewportSize({ width: 2560, height: 3000 });
    const url = `${BASE_URL}${articlePath}`;
    await page.goto(url, { waitUntil: 'networkidle', timeout: 15000 });
    await page.waitForTimeout(3000);

    const pageHeight = await page.evaluate(() => Math.max(
      document.body.scrollHeight,
      document.documentElement.scrollHeight,
      document.querySelector('[vp-content]')?.scrollHeight || 0
    ) + 200);
    if (pageHeight > 3000) {
      await page.setViewportSize({ width: 2560, height: pageHeight });
      await page.waitForTimeout(500);
    }

    const pageTitle = await page.title();
    const title = pageTitle.split('|')[0].trim();

    // ---- 2) 替换页面中的 mermaid div 为 mmdc 生成的 PNG ----
    if (mermaidImgs.length > 0) {
      console.log(`  mmdc→PNG: ${mermaidImgs.length} 个图表`);
      await page.evaluate((imgs) => {
        const mermaidDivs = document.querySelectorAll('[vp-content] .mermaid');
        imgs.forEach((img, i) => {
          if (i < mermaidDivs.length) {
            const el = document.createElement('img');
            el.src = img.path;
            el.setAttribute('width', String(img.width));
            el.setAttribute('height', String(img.height));
            mermaidDivs[i].replaceWith(el);
          }
        });
      }, mermaidImgs);
    }

    // ---- 3) 注入 KaTeX + 公式转换 ----
    if (existsSync(KATEX_PATH)) {
      await page.addScriptTag({ path: KATEX_PATH });
    }
    const formulaStats = await page.evaluate(() => {
      let displayDone = 0, inlineDone = 0, errors = 0;

      // 块级公式：不使用 display=block（会导致编号换行），改为段落居中
      document.querySelectorAll('.katex-display').forEach(el => {
        try {
          const mathml = el.querySelector('.katex-mathml math');
          if (!mathml) return;
          const math = mathml.cloneNode(true);
          // 不用 display=block — 改为段落级别的居中
          // 不带 display 属性的 <math> 会被 Pandoc 转为 inline m:oMath

          const eqContainer = el.closest('.equation-numbered');
          const eqNumEl = eqContainer ? eqContainer.querySelector('.equation-number') : null;
          const eqNum = eqNumEl ? eqNumEl.textContent.trim() : '';

          if (eqNum) {
            // 有编号的公式：math + 编号放在同一居中段落
            const p = document.createElement('p');
            p.style.textAlign = 'center';
            p.appendChild(math);
            const numSpan = document.createElement('span');
            numSpan.textContent = '  ' + eqNum;
            p.appendChild(numSpan);
            eqContainer.replaceWith(p);
          } else {
            // 无编号公式：math 放在居中段落
            const p = document.createElement('p');
            p.style.textAlign = 'center';
            p.appendChild(math);
            el.replaceWith(p);
          }
          displayDone++;
        } catch (e) { errors++; }
      });

      // 行内公式：保留原有的 MathML
      document.querySelectorAll('.katex').forEach(el => {
        if (el.closest('.katex-display')) return;
        try {
          const mathml = el.querySelector('.katex-mathml math');
          if (mathml) {
            const math = mathml.cloneNode(true);
            el.replaceWith(math);
            inlineDone++;
          }
        } catch (e) { errors++; }
      });

      return { displayDone, inlineDone, errors };
    });
    console.log(`  公式: display=${formulaStats.displayDone} inline=${formulaStats.inlineDone} err=${formulaStats.errors}`);

    // ---- 4) nn-arch SVG → PNG 截图 ----
    let nnArchCount = 0;
    const svgDataList = await page.evaluate(() => {
      const list = [];
      document.querySelectorAll('[vp-content] svg').forEach((svg, i) => {
        const rect = svg.getBoundingClientRect();
        const inMermaid = svg.closest('.mermaid');
        const inFooter = svg.closest('.article-footer, .page-meta, .page-nav');
        if (inMermaid || inFooter || rect.width < 40) return;
        if (svg.parentElement?.closest('svg')) return;
        const origStyle = svg.getAttribute('style') || '';
        svg.setAttribute('style', origStyle.replace(/transform:\s*scale\([^)]+\);?/gi, ''));
        svg.setAttribute('data-orig-style', origStyle);
        const id = `__svg_to_img_${i}`;
        svg.setAttribute('data-svg-id', id);
        list.push({ index: i, id, width: svg.scrollWidth || Math.round(rect.width), height: svg.scrollHeight || Math.round(rect.height) });
      });
      return list;
    });

    for (const item of svgDataList) {
      try {
        const loc = page.locator(`svg[data-svg-id="${item.id}"]`);
        if (await loc.count() === 0) continue;
        await loc.scrollIntoViewIfNeeded();
        await page.waitForTimeout(300);
        const pngPath = resolve(TMP_DIR, `${slug}-nnarch-${item.index}.png`);
        await loc.screenshot({ path: pngPath, type: 'png' });
        nnArchCount++;
        await loc.evaluate((el, params) => {
          const orig = el.getAttribute('data-orig-style');
          if (orig) el.setAttribute('style', orig);
          const img = document.createElement('img');
          img.src = params.src;
          img.setAttribute('width', String(params.w));
          img.setAttribute('height', String(params.h));
          el.replaceWith(img);
        }, { src: pngPath, w: item.width, h: item.height });
      } catch (e) {
        console.warn(`    nn-arch截图失败 #${item.index}:`, e.message);
      }
    }
    if (nnArchCount > 0) console.log(`  nn-arch→PNG: ${nnArchCount} 个图表`);

    // ---- 5) 提取内容 ----
    const contentHtml = await page.evaluate(() => {
      const content = document.querySelector('[vp-content]');
      if (!content) return '';
      const clone = content.cloneNode(true);
      clone.querySelectorAll(
        '.article-footer, .floating-toolbar, .run-btn, '
        + '.code-demo, .giscus, .page-nav, .page-meta, '
        + '.code-demo-hint, script, style, .vuepress-plugin-search-pro'
      ).forEach(el => el.remove());
      clone.querySelectorAll('p').forEach(p => {
        if (/点击\s*Run|点击运行|点击代码/.test(p.textContent)) p.remove();
      });
      clone.querySelectorAll('div').forEach(div => {
        const t = div.textContent.trim();
        if ((t.includes('点击 Run') || t.includes('点击运行')) && t.length < 100) div.remove();
      });
      const h1 = clone.querySelector('h1');
      if (h1) h1.remove();
      clone.querySelectorAll('pre').forEach(pre => {
        if (!pre.closest('.__code_block__')) {
          const wrapper = document.createElement('div');
          wrapper.className = '__code_block__';
          pre.parentNode.insertBefore(wrapper, pre);
          wrapper.appendChild(pre);
        }
        // 移除 Python 代码块的 import 行
        const code = pre.querySelector('code');
        if (code) {
          const text = code.textContent || '';
          if (text.includes('import ') || text.includes('from ')) {
            const filtered = text.split('\n').filter(line => {
              const t = line.trim();
              return !t.startsWith('import ') && !t.startsWith('from ');
            });
            code.textContent = filtered.join('\n');
          }
        }
      });
      clone.querySelectorAll('.hint-container-title').forEach(el => {
        const strong = document.createElement('strong');
        strong.innerHTML = el.innerHTML;
        el.innerHTML = '';
        el.appendChild(strong);
      });
      clone.querySelectorAll('p').forEach(p => {
        if (p.textContent.trim().startsWith('图：')) p.setAttribute('data-figure', 'true');
      });
      clone.querySelectorAll('h1, h2, h3').forEach(h => {
        if (/练[习题]|习题|Exercise/i.test(h.textContent)) {
          let next = h.nextElementSibling;
          while (next && !/^H[1-6]$/.test(next.tagName)) {
            const toRemove = next;
            next = next.nextElementSibling;
            toRemove.remove();
          }
          h.remove();
        }
      });
      return clone.innerHTML;
    });

    // ---- 6) 图片路径转绝对路径 ----
    const fixedHtml = (contentHtml || '').replace(
      /<img\s+[^>]*src="(\/[^"]+)"/g,
      (match, srcPath) => {
        if (srcPath.startsWith('/root/')) return match;
        const absPath = resolve(DOCS_DIR, srcPath.slice(1));
        return match.replace(srcPath, absPath);
      }
    );

    return { content: fixedHtml, title };

  } finally {
    await page.close();
  }
}

// ---- mmdc 渲染 mermaid 为 PNG ----
function renderMermaidBlocks(articlePath, slug) {
  const mdPath = articlePathToMd(articlePath);
  if (!mdPath || !existsSync(mdPath)) return [];

  const mdContent = readFileSync(mdPath, 'utf-8');
  const blocks = extractMermaidBlocks(mdContent);
  if (blocks.length === 0) return [];

  const results = [];
  blocks.forEach((code, i) => {
    const mmdPath = resolve(TMP_DIR, `${slug}-mermaid-${i}.mmd`);
    const pngPath = resolve(TMP_DIR, `${slug}-mermaid-${i}.png`);
    writeFileSync(mmdPath, code, 'utf-8');
    try {
      execFileSync('npx', [
        'mmdc', '-i', mmdPath, '-o', pngPath,
        '-b', 'white', '-s', '2',
        '-p', PUPPETEER_CONFIG,
      ], { stdio: 'pipe', timeout: 30000, env: { ...process.env, PUPPETEER_EXECUTABLE_PATH: BROWSER_PATH } });
      // Get PNG dimensions
      if (existsSync(pngPath)) {
        const buf = readFileSync(pngPath);
        const w = buf.readUInt32BE(16);
        const h = buf.readUInt32BE(20);
        results.push({ path: pngPath, width: w, height: h });
      }
    } catch (e) {
      console.warn(`    mmdc渲染失败 #${i}:`, e.message);
    }
    try { unlinkSync(mmdPath); } catch {}
  });
  return results;
}

function articlePathToMd(articlePath) {
  // /maths/linear/vectors.html → docs/maths/linear/vectors.md
  const noExt = articlePath.replace('.html', '');
  return resolve(DOCS_DIR, noExt.slice(1) + '.md');
}

function extractMermaidBlocks(mdContent) {
  const blocks = [];
  const lines = mdContent.split('\n');
  let inBlock = false;
  let blockLines = [];
  for (const line of lines) {
    if (line.startsWith('```mermaid')) {
      inBlock = true;
      blockLines = [];
    } else if (inBlock && line.trim() === '```') {
      blocks.push(blockLines.join('\n'));
      inBlock = false;
    } else if (inBlock) {
      blockLines.push(line);
    }
  }
  return blocks;
}

export async function launchBrowser() {
  return chromium.launch({
    headless: true,
    executablePath: BROWSER_PATH,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
  });
}
