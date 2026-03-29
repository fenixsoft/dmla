/**
 * 字数统计插件
 * 在构建时计算 Markdown 文章的字数，支持中英文混合统计
 */
import { readFile } from 'node:fs/promises'

/**
 * 预处理 Markdown 内容，移除语法标记
 * @param {string} content Markdown 源内容
 * @returns {string} 处理后的纯文本
 */
function preprocessMarkdown(content) {
  if (!content) return ''

  let text = content

  // 移除 fenced code blocks (```lang ... ```)
  text = text.replace(/```[\s\S]*?```/g, '')

  // 移除行内代码 (`code`)
  text = text.replace(/`[^`]+`/g, '')

  // 移除 LaTeX 行内公式 ($formula$) - 需要小心处理避免误删
  // 只匹配明确的 $...$ 公式，不匹配货币符号
  text = text.replace(/\$([^\$\n]+?)\$/g, '')

  // 移除 LaTeX 块级公式 ($$formula$$)
  text = text.replace(/\$\$[\s\S]*?\$\$/g, '')

  // 移除链接 [text](url)
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')

  // 移除图片 ![alt](url)
  text = text.replace(/!\[[^\]]*\]\([^)]+\)/g, '')

  // 移除标题标记 (# ## ### 等)
  text = text.replace(/^#{1,6}\s+/gm, '')

  // 移除粗体/斜体标记 (** ***, _ _)
  text = text.replace(/\*\*?([^*]+)\*\*?/g, '$1')
  text = text.replace(/__?([^_]+)__?/g, '$1')

  // 移除 HTML 标签
  text = text.replace(/<[^>]+>/g, '')

  // 移除 frontmatter (--- ... ---)
  text = text.replace(/^---[\s\S]*?---/m, '')

  return text
}

/**
 * 计算字数（支持中英文混合）
 * 基于 icyfenix.cn 的 fnGetCpmisWords 算法
 * @param {string} str 纯文本内容
 * @returns {number} 字数
 */
function countWords(str) {
  if (!str) return 0

  let sLen = 0
  try {
    // 将空白符（空格、换行、全角空格）替换为占位符
    str = str.replace(/(\r\n+|\s+|　+)/g, '龘')

    // 英文字符（ASCII）标记为 'm'
    str = str.replace(/[\x00-\xff]/g, 'm')

    // 连续英文合并为一个单词
    str = str.replace(/m+/g, '*')

    // 移除占位符
    str = str.replace(/龘+/g, '')

    // 返回字数（中文按字符，英文按单词）
    sLen = str.length
  } catch (e) {
    console.error('字数统计出错:', e)
  }

  return sLen
}

/**
 * VuePress 插件：字数统计
 */
const wordCountPlugin = (options = {}) => {
  return {
    name: 'vuepress-plugin-word-count',

    // VuePress v2: 使用 onInitialized 钩子读取源文件
    onInitialized: async (app) => {
      for (const page of app.pages) {
        const frontmatter = page.frontmatter

        // 如果 frontmatter 中已有 wordCount，使用预设值
        if (frontmatter && frontmatter.wordCount) {
          page.data.wordCount = frontmatter.wordCount
          continue
        }

        // 通过 filePath 读取原始 Markdown 内容
        let rawContent = ''
        if (page.filePath) {
          try {
            rawContent = await readFile(page.filePath, 'utf-8')
          } catch (e) {
            // 读取失败时使用 page.content 作为 fallback
            rawContent = page.content || ''
          }
        } else {
          rawContent = page.content || ''
        }

        if (!rawContent) {
          page.data.wordCount = 0
          continue
        }

        // 预处理 Markdown
        const plainText = preprocessMarkdown(rawContent)

        // 计算字数
        const wordCount = countWords(plainText)

        // 存入 page data
        page.data.wordCount = wordCount
      }
    }
  }
}

export default wordCountPlugin