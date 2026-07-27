import { existsSync, readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../../..');

function findMatchingBracket(str, startPos, open, close) {
  let depth = 0, inSingle = false, inDouble = false, inBacktick = false;
  for (let i = startPos; i < str.length; i++) {
    const ch = str[i], prev = i > 0 ? str[i - 1] : '';
    if (ch === "'" && !inDouble && !inBacktick && prev !== '\\') inSingle = !inSingle;
    if (ch === '"' && !inSingle && !inBacktick && prev !== '\\') inDouble = !inDouble;
    if (ch === '`' && !inSingle && !inDouble && prev !== '\\') inBacktick = !inBacktick;
    if (!inSingle && !inDouble && !inBacktick) {
      if (ch === open) depth++;
      else if (ch === close) { depth--; if (depth === 0) return i; }
    }
  }
  return -1;
}

function extractBracedObjects(text) {
  const objects = [];
  let i = 0;
  while (i < text.length) {
    while (i < text.length && /[\s,]/.test(text[i])) i++;
    if (i >= text.length) break;
    const bracePos = text.indexOf('{', i);
    if (bracePos === -1) break;
    const endPos = findMatchingBracket(text, bracePos, '{', '}');
    if (endPos === -1) break;
    objects.push(text.substring(bracePos, endPos + 1));
    i = endPos + 1;
  }
  return objects;
}

function getText(objText) {
  const m = objText.match(/text:\s*'([^']+)'/);
  return m ? m[1] : null;
}

function getLink(objText) {
  const m = objText.match(/link:\s*'([^']+)'/);
  return m ? m[1] : null;
}

function getArrayContent(objText, arrayName) {
  const regex = new RegExp(`${arrayName}\\s*:\\s*\\[`);
  const m = objText.match(regex);
  if (!m) return null;
  const startPos = m.index + m[0].length - 1;
  const endPos = findMatchingBracket(objText, startPos, '[', ']');
  if (endPos === -1) return null;
  return objText.substring(startPos + 1, endPos);
}

function normalizeLink(link) {
  if (!link.startsWith('/') && !link.startsWith('http')) return '/' + link;
  return link;
}

function parseSidebar(sidebarContent) {
  const parts = [];
  const topEntries = extractBracedObjects(sidebarContent);

  for (const entry of topEntries) {
    const text = getText(entry);
    if (!text || text === '目录' || text.includes('附录')) continue;
    const childrenContent = getArrayContent(entry, 'children');
    if (!childrenContent) continue;

    const childEntries = extractBracedObjects(childrenContent);
    const chapters = [];

    // Detect whether child entries have their own children (sub-chapters)
    const hasSubChapters = childEntries.some(c => getArrayContent(c, 'children'));

    if (hasSubChapters) {
      for (const child of childEntries) {
        const chTitle = getText(child);
        if (!chTitle) continue;
        const subChildren = getArrayContent(child, 'children');
        if (!subChildren) continue;

        const articles = [];
        for (const leaf of extractBracedObjects(subChildren)) {
          const leafText = getText(leaf);
          const leafLink = getLink(leaf);
          if (leafText && leafLink) articles.push({ title: leafText, link: normalizeLink(leafLink) });
        }
        if (articles.length > 0) chapters.push({ chapterTitle: chTitle, articles });
      }
    } else {
      const articles = [];
      for (const child of childEntries) {
        const childText = getText(child);
        const childLink = getLink(child);
        if (childText && childLink) articles.push({ title: childText, link: normalizeLink(childLink) });
      }
      if (articles.length > 0) chapters.push({ chapterTitle: null, articles });
    }

    if (chapters.length > 0) parts.push({ partTitle: text, chapters });
  }
  return parts;
}

export function parseArticleList() {
  const configPath = resolve(PROJECT_ROOT, 'docs/.vuepress/config.js');
  const content = readFileSync(configPath, 'utf-8');

  const themeMatch = content.match(/dmlaTheme\s*\(/);
  if (!themeMatch) throw new Error('无法定位 dmlaTheme 配置');
  const themeBracePos = themeMatch.index + themeMatch[0].length - 1;
  const themeEndPos = findMatchingBracket(content, themeBracePos, '(', ')');
  if (themeEndPos === -1) throw new Error('无法匹配 dmlaTheme 括号');
  const themeBody = content.substring(themeMatch.index, themeEndPos + 1);

  const localesMatch = themeBody.match(/locales\s*:\s*\{/);
  if (!localesMatch) throw new Error('无法定位 locales 配置');
  const localesBracePos = localesMatch.index + localesMatch[0].length - 1;
  const localesEndPos = findMatchingBracket(themeBody, localesBracePos, '{', '}');
  if (localesEndPos === -1) throw new Error('无法匹配 locales 花括号');
  const localesBody = themeBody.substring(localesMatch.index, localesEndPos + 1);

  const zhLocaleMatch = localesBody.match(/'\/'\s*:/);
  if (!zhLocaleMatch) throw new Error('无法定位中文 locale');
  const afterKey = zhLocaleMatch.index + zhLocaleMatch[0].length;
  const zhBracePos = localesBody.indexOf('{', afterKey);
  if (zhBracePos === -1) throw new Error('无法定位中文 locale 花括号');
  const zhLocaleEnd = findMatchingBracket(localesBody, zhBracePos, '{', '}');
  const zhLocaleContent = localesBody.substring(zhBracePos + 1, zhLocaleEnd);

  const sidebarMatch = zhLocaleContent.match(/sidebar\s*:\s*\[/);
  if (!sidebarMatch) throw new Error('无法定位 sidebar');
  const sidebarBracePos = sidebarMatch.index + sidebarMatch[0].length - 1;
  const sidebarEnd = findMatchingBracket(zhLocaleContent, sidebarBracePos, '[', ']');
  const sidebarContent = zhLocaleContent.substring(sidebarBracePos + 1, sidebarEnd);

  // Build hierarchy: Part → Chapter → Section
  const parts = parseSidebar(sidebarContent);
  const result = [];
  let globalChapterNum = 0, globalSectionNum = 0;

  for (let pi = 0; pi < parts.length; pi++) {
    const part = parts[pi];
    const outChapters = [];

    for (const ch of part.chapters) {
      const outArticles = [];
      for (const art of ch.articles) {
        globalSectionNum++;
        const slug = art.link.split('/').filter(Boolean).pop();
        const filePath = resolve(PROJECT_ROOT, 'docs', art.link.slice(1) + '.md');
        outArticles.push({
          title: art.title,
          sectionNum: globalSectionNum,
          sectionTitle: `第${globalSectionNum}节 ${art.title}`,
          slug,
          filePath,
          link: art.link,
        });
      }
      if (ch.chapterTitle) {
        globalChapterNum++;
        outChapters.push({
          chapterTitle: ch.chapterTitle,
          chapterNum: globalChapterNum,
          chapterDir: `第${globalChapterNum}章 ${ch.chapterTitle}`,
          articles: outArticles,
        });
      } else {
        // 前言 —— no chapter, place articles directly in part dir
        outChapters.push({
          chapterTitle: part.partTitle,
          chapterNum: 0,
          chapterDir: part.partTitle,
          articles: outArticles,
        });
      }
    }
    result.push({
      partTitle: part.partTitle,
      partNum: pi + 1,
      partDir: `第${pi + 1}部分 ${part.partTitle}`,
      chapters: outChapters,
    });
  }

  return result;
}
