/**
 * 字数统计插件
 * 参考 awesome-fenix 的 read-time 插件实现
 * 在构建时计算 Markdown 文章的字数，支持中英文混合统计
 * 支持区分文字字数和代码字数
 */

// 全局字数统计对象（构建时收集所有页面的字数）
const globalWords = {}

/**
 * 预处理 Markdown 内容，分离文字和代码内容
 *
 * 字数统计规则：
 * - 代码块 ```...``` 和行内代码 `...`：计入代码字数（移除语法标记，保留内容）
 * - 其他内容：计入文字字数
 * - 行内公式 $...$ 和块级公式 $$...$$：计入字数（移除语法标记，保留内容）
 * - 链接、图片、HTML标签、frontmatter：不计入字数
 *
 * @returns {text: string, code: string} 分离后的文字和代码内容
 */
function preprocessMarkdownSeparate(content) {
  if (!content) return { text: '', code: '' }

  let textContent = content
  let codeContent = ''

  // 提取代码块内容（保存代码内容用于代码字数统计）
  const codeBlocks = []
  textContent = textContent.replace(/```[^\n]*\n([\s\S]*?)```/g, (match, code) => {
    codeBlocks.push(code)
    return ''  // 从文字内容中移除代码块
  })

  // 提取行内代码内容
  const inlineCodes = []
  textContent = textContent.replace(/`([^`]+)`/g, (match, code) => {
    inlineCodes.push(code)
    return ''  // 从文字内容中移除行内代码
  })

  // 合并代码内容
  codeContent = codeBlocks.join('\n') + '\n' + inlineCodes.join(' ')

  // 处理文字内容中的其他 Markdown 语法（不计入字数的内容）
  textContent = textContent.replace(/\$\$([\s\S]*?)\$\$/g, '$1')  // 块级公式：保留内容
  textContent = textContent.replace(/\$([^\$\n]+?)\$/g, '$1')     // 行内公式：保留内容
  textContent = textContent.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')  // 链接：保留显示文字
  textContent = textContent.replace(/!\[[^\]]*\]\([^)]+\)/g, '')     // 图片：完全移除
  textContent = textContent.replace(/<[a-zA-Z\/][^>]*>/g, '')        // HTML标签：只匹配真正的标签
  textContent = textContent.replace(/^---[\s\S]*?---/m, '')          // Frontmatter：完全移除
  textContent = textContent.replace(/^#{1,6}\s+/gm, '')               // 标题标记
  textContent = textContent.replace(/\*\*?([^*]+)\*\*?/g, '$1')       // 粗体/斜体
  textContent = textContent.replace(/__?([^_]+)__?/g, '$1')           // 粗体/斜体

  return { text: textContent, code: codeContent }
}

/**
 * 计算字数（支持中英文混合）
 * 基于 awesome-fenix 的 fnGetCpmisWords 算法
 */
function countWords(str) {
  if (!str) return 0

  let sLen = 0
  try {
    str = str.replace(/(\r\n+|\s+|　+)/g, '龘')
    str = str.replace(/[\x00-\xff]/g, 'm')
    str = str.replace(/m+/g, '*')
    str = str.replace(/龘+/g, '')
    sLen = str.length
  } catch (e) {
    console.error('字数统计出错:', e)
  }

  return sLen
}

/**
 * VuePress 2 插件：字数统计
 */
const wordCountPlugin = (options = {}) => {
  return {
    name: 'vuepress-plugin-word-count',

    // VuePress 2: 使用 onInitialized 钩子收集所有页面字数并写入临时文件
    async onInitialized(app) {
      // 第一遍：收集所有页面的字数（区分文字和代码）
      for (const page of app.pages) {
        let wordCount = 0
        let textWordCount = 0
        let codeWordCount = 0

        // 如果 frontmatter 中已有 wordCount，使用预设值
        if (page.frontmatter?.wordCount) {
          wordCount = page.frontmatter.wordCount
          textWordCount = wordCount  // 预设值时，全部计入文字
          codeWordCount = 0
        } else {
          const content = page.content || ''
          if (content) {
            const { text, code } = preprocessMarkdownSeparate(content)
            textWordCount = countWords(text)
            codeWordCount = countWords(code)
            wordCount = textWordCount + codeWordCount
          }
        }

        globalWords[page.path] = { wordCount, textWordCount, codeWordCount }
        page.data.wordCount = wordCount
        page.data.textWordCount = textWordCount
        page.data.codeWordCount = codeWordCount
      }

      // 第二遍：将 globalWords 注入到每个页面（用于页面级别显示）
      for (const page of app.pages) {
        page.data.readingTime = {
          words: page.data.wordCount,
          textWords: page.data.textWordCount,
          codeWords: page.data.codeWordCount,
          minutes: page.data.wordCount / 500,
          globalWords: { ...globalWords }  // 复制一份，确保每个页面都有完整数据
        }
      }

      // 构建 wordCountData 结构 {path: {title, wordCount, textWordCount, codeWordCount}}
      const wordCountData = {}
      for (const page of app.pages) {
        wordCountData[page.path] = {
          title: page.title,
          wordCount: page.data.wordCount || 0,
          textWordCount: page.data.textWordCount || 0,
          codeWordCount: page.data.codeWordCount || 0
        }
      }

      // 在 onInitialized 中写入临时文件（异步操作必须在此完成）
      await app.writeTemp(
        'word-count/data.js',
        `export const wordCountData = ${JSON.stringify(wordCountData)}`
      )

      await app.writeTemp(
        'word-count/client.js',
        `import { defineClientConfig } from 'vuepress/client'
import { wordCountData } from './data.js'

export default defineClientConfig({
  enhance({ app }) {
    app.provide('wordCountData', wordCountData)
  }
})`
      )
    },

    // clientConfigFile 只返回文件路径（文件已在 onInitialized 中写入）
    clientConfigFile(app) {
      return app.dir.temp('word-count/client.js')
    }
  }
}

export default wordCountPlugin