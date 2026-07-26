import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../../..');

/**
 * 在字符串中查找与 startPos 处 open 字符匹配的 close 字符位置
 * 正确处理单引号/双引号字符串，避免将字符串内的括号计入
 * @param {string} str
 * @param {number} startPos
 * @param {string} open
 * @param {string} close
 * @returns {number} 匹配的 close 位置，-1 表示未找到
 */
function findMatchingBracket(str, startPos, open, close) {
  let depth = 0;
  let inSingle = false;
  let inDouble = false;
  for (let i = startPos; i < str.length; i++) {
    const ch = str[i];
    const prev = i > 0 ? str[i - 1] : '';

    if (ch === "'" && !inDouble && prev !== '\\') inSingle = !inSingle;
    if (ch === '"' && !inSingle && prev !== '\\') inDouble = !inDouble;

    if (!inSingle && !inDouble) {
      if (ch === open) depth++;
      else if (ch === close) {
        depth--;
        if (depth === 0) return i;
      }
    }
  }
  return -1;
}

/**
 * 从 config.js 提取中文文章清单
 * @returns {Array<{filePath: string, title: string, chapterIndex: number, fileIndex: number, slug: string, chapterPath: string}>}
 */
export function parseArticleList() {
  // 直接读取 config.js 源文本，手动解析 sidebar 结构
  // 避免 eval/import 复杂的 ES module 依赖
  const configPath = resolve(PROJECT_ROOT, 'docs/.vuepress/config.js');
  const content = readFileSync(configPath, 'utf-8');

  // --- 第一步：定位中文 locale ('/') 的 sidebar 内容 ---
  // 使用括号匹配处理嵌套结构，避免简单正则的匹配错误
  //
  // config.js 中有两层 locales：
  //   1. 顶层 locales（line 33），不含 sidebar——需要跳过
  //   2. dmlaTheme({...}) 内部的 locales（line 70），包含 sidebar
  // 我们通过 dmlaTheme 标记来定位正确的上下文

  // 找到 dmlaTheme({ 的起始和结束范围
  const themeMatch = content.match(/dmlaTheme\s*\(/);
  if (!themeMatch) throw new Error('无法定位 dmlaTheme 配置');
  const themeBracePos = themeMatch.index + themeMatch[0].length - 1; // '(' 的位置

  const themeEndPos = findMatchingBracket(content, themeBracePos, '(', ')');
  if (themeEndPos === -1) throw new Error('无法匹配 dmlaTheme 括号');

  const themeBody = content.substring(themeMatch.index, themeEndPos + 1);

  // 在 dmlaTheme 主体内找到 locales 对象
  const localesMatch = themeBody.match(/locales\s*:\s*\{/);
  if (!localesMatch) throw new Error('无法定位 dmlaTheme 中的 locales 配置');
  const localesBracePos = localesMatch.index + localesMatch[0].length - 1;

  const localesEndPos = findMatchingBracket(themeBody, localesBracePos, '{', '}');
  if (localesEndPos === -1) throw new Error('无法匹配 locales 花括号');

  const localesBody = themeBody.substring(localesMatch.index, localesEndPos + 1);

  // 在 locales 内容中定位 '/' locale 块
  const zhLocaleMatch = localesBody.match(/'\/'\s*:/);
  if (!zhLocaleMatch) throw new Error('无法定位中文 locale 键');

  const afterKey = zhLocaleMatch.index + zhLocaleMatch[0].length;
  const zhBracePos = localesBody.indexOf('{', afterKey);
  if (zhBracePos === -1) throw new Error('无法定位中文 locale 起始花括号');

  const zhLocaleEnd = findMatchingBracket(localesBody, zhBracePos, '{', '}');
  if (zhLocaleEnd === -1) throw new Error('无法匹配中文 locale 花括号');

  const zhLocaleContent = localesBody.substring(zhBracePos + 1, zhLocaleEnd);

  // 在 '/' locale 内容中定位顶层 sidebar 数组
  const sidebarMatch = zhLocaleContent.match(/sidebar\s*:\s*\[/);
  if (!sidebarMatch) throw new Error('无法定位中文 sidebar 配置');

  const sidebarBracePos = sidebarMatch.index + sidebarMatch[0].length - 1;
  const sidebarEnd = findMatchingBracket(zhLocaleContent, sidebarBracePos, '[', ']');
  if (sidebarEnd === -1) throw new Error('无法匹配 sidebar 方括号');

  const sidebarContent = zhLocaleContent.substring(sidebarBracePos + 1, sidebarEnd);

  // --- 第二步：递归解析 nested {text, children, link} 结构 ---
  // 使用括号匹配感知嵌套层级，正确处理三层及以上结构
  // 只有顶层 sidebar 数组的直接子元素才计为章节

  /**
   * 从文本中提取所有顶层 {...} 对象
   * 自动跳过逗号、空白等分隔符
   */
  function extractBracedObjects(text) {
    const objects = [];
    let i = 0;
    while (i < text.length) {
      // 跳过空白和逗号
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

  /** 提取对象的 text 属性值 */
  function getText(objText) {
    const m = objText.match(/text:\s*'([^']+)'/);
    return m ? m[1] : null;
  }

  /** 提取对象的 link 属性值 */
  function getLink(objText) {
    const m = objText.match(/link:\s*'([^']+)'/);
    return m ? m[1] : null;
  }

  /**
   * 提取对象内指定数组属性的内容（介于 [...] 之间的文本）
   * 例如 getArrayContent(obj, 'children') 返回 children array 的内容
   */
  function getArrayContent(objText, arrayName) {
    const regex = new RegExp(`${arrayName}\\s*:\\s*\\[`);
    const m = objText.match(regex);
    if (!m) return null;
    const startPos = m.index + m[0].length - 1; // '[' 的位置
    const endPos = findMatchingBracket(objText, startPos, '[', ']');
    if (endPos === -1) return null;
    return objText.substring(startPos + 1, endPos);
  }

  /**
   * 递归提取数组内容中的所有叶子文章
   * @param {string} arrayContent - children 数组的文本内容（介于 [...] 内）
   * @param {number} chapterIndex - 当前所属章节序号
   * @param {string} chapterPath - 当前所属章节名称
   * @param {number} startFileIndex - 起始文件序号
   * @returns {Array}
   */
  function flattenArticles(arrayContent, chapterIndex, chapterPath, startFileIndex) {
    const result = [];
    let fileIndex = startFileIndex;

    const entries = extractBracedObjects(arrayContent);
    for (const entry of entries) {
      const text = getText(entry);
      if (!text) continue;

      const link = getLink(entry);
      const childrenContent = getArrayContent(entry, 'children');

      if (childrenContent) {
        // 子节（如"线性代数"下的 children），递归处理
        const subArticles = flattenArticles(
          childrenContent, chapterIndex, chapterPath, fileIndex
        );
        result.push(...subArticles);
        fileIndex += subArticles.length;
      } else if (link) {
        // 叶子文章节点
        let normalizedLink = link;
        if (!normalizedLink.startsWith('/') && !normalizedLink.startsWith('http')) {
          normalizedLink = '/' + normalizedLink;
        }
        if (normalizedLink.startsWith('http')) continue;

        const slug = normalizedLink.split('/').filter(Boolean).pop();
        const filePath = resolve(PROJECT_ROOT, 'docs', normalizedLink.slice(1) + '.md');

        result.push({
          filePath,
          title: text,
          chapterIndex,
          fileIndex: fileIndex++,
          slug,
          chapterPath,
        });
      }
    }

    return result;
  }

  // 提取 sidebar 顶层 {...} 对象，识别章节
  const articles = [];
  let chapterIndex = -1;

  const topLevelEntries = extractBracedObjects(sidebarContent);
  for (const entry of topLevelEntries) {
    const text = getText(entry);
    if (!text) continue;

    // 跳过"目录"（它仅是一个目录页链接，不是章节）
    if (text === '目录') continue;

    // 跳过"附录"及其所有子条目
    if (text.includes('附录')) continue;

    const childrenContent = getArrayContent(entry, 'children');
    if (!childrenContent) continue; // 既非章节又非附录的孤立条目（如目录）直接跳过

    // 这是一个章节
    chapterIndex++;
    const chapterArticles = flattenArticles(
      childrenContent, chapterIndex, text, 1
    );
    articles.push(...chapterArticles);
  }

  return articles;
}
