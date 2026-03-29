/**
 * Math 插件 - 支持 Markdown 中的 LaTeX 数学公式
 * 使用 KaTeX 在服务端渲染
 */
import katex from 'katex'
import { path } from 'vuepress/utils'

export default {
  name: 'vuepress-plugin-math-katex',

  extendsMarkdown: (md) => {
    // 解析块级公式 $$...$$
    const parseBlockMath = (state, startLine, endLine, silent) => {
      const start = state.bMarks[startLine] + state.tShift[startLine]
      const max = state.eMarks[startLine]

      // 检查是否以 $$ 开头
      if (state.src.slice(start, start + 2) !== '$$') {
        return false
      }

      // 检查单行公式 $$...$$
      const firstLineEnd = state.eMarks[startLine]
      const firstLineContent = state.src.slice(start + 2, firstLineEnd)
      const singleLineMatch = firstLineContent.indexOf('$$')

      if (singleLineMatch !== -1) {
        // 单行块级公式
        const content = firstLineContent.slice(0, singleLineMatch).trim()
        if (!silent) {
          const token = state.push('math_block', 'div', 0)
          token.content = content
          token.block = true
        }
        state.line = startLine + 1
        return true
      }

      // 多行块级公式，查找结束的 $$
      for (let nextLine = startLine + 1; nextLine < endLine; nextLine++) {
        const lineStart = state.bMarks[nextLine] + state.tShift[nextLine]
        const lineMax = state.eMarks[nextLine]
        const lineContent = state.src.slice(lineStart, lineMax)
        const trimmedContent = lineContent.trim()

        // 检查整行是否只有 $$
        if (trimmedContent === '$$') {
          // 收集内容
          const contentStart = state.bMarks[startLine] + state.tShift[startLine] + 2
          const contentEnd = lineStart
          const content = state.src.slice(contentStart, contentEnd).trim()

          if (!silent) {
            const token = state.push('math_block', 'div', 0)
            token.content = content
            token.block = true
          }
          state.line = nextLine + 1
          return true
        }

        // 检查行末是否有 $$（支持 \end{pmatrix} $$ 格式）
        if (trimmedContent.endsWith('$$')) {
          // 收集内容，不包括最后的 $$
          const contentStart = state.bMarks[startLine] + state.tShift[startLine] + 2
          const contentEnd = lineStart + lineContent.lastIndexOf('$$')
          const content = state.src.slice(contentStart, contentEnd).trim()

          if (!silent) {
            const token = state.push('math_block', 'div', 0)
            token.content = content
            token.block = true
          }
          state.line = nextLine + 1
          return true
        }
      }

      return false
    }

    // 渲染块级公式（服务端渲染）
    md.renderer.rules.math_block = (tokens, idx) => {
      const content = tokens[idx].content
      try {
        return `<div class="katex-display">${katex.renderToString(content, {
          throwOnError: false,
          displayMode: true,
          strict: "ignore"
        })}</div>`
      } catch (e) {
        console.warn('KaTeX block render error:', e.message)
        return `<div class="katex-error">$$${content}$$</div>`
      }
    }

    // 注册块级公式解析规则
    md.block.ruler.before('paragraph', 'math_block', parseBlockMath, {
      alt: ['paragraph', 'reference', 'blockquote', 'list']
    })

    // 处理行内公式 $...$
    md.inline.ruler.before('escape', 'math_inline', (state, silent) => {
      const start = state.pos
      const max = state.posMax

      // 检查是否以 $ 开头（且不是 $$）
      if (state.src.charCodeAt(start) !== 0x24 /* $ */) {
        return false
      }

      // 检查是否是 $$（块级公式已处理）
      if (start + 1 < max && state.src.charCodeAt(start + 1) === 0x24) {
        return false
      }

      // 查找结束的 $
      let pos = start + 1
      while (pos < max) {
        if (state.src.charCodeAt(pos) === 0x24 /* $ */) {
          break
        }
        pos++
      }

      if (pos >= max || pos === start + 1) {
        return false
      }

      const content = state.src.slice(start + 1, pos)

      // 检查内容是否有效
      if (!content.trim()) {
        return false
      }

      if (!silent) {
        const token = state.push('math_inline', 'span', 0)
        token.content = content
      }

      state.pos = pos + 1
      return true
    })

    // 渲染行内公式（服务端渲染）
    md.renderer.rules.math_inline = (tokens, idx) => {
      const content = tokens[idx].content
      try {
        return katex.renderToString(content, {
          throwOnError: false,
          displayMode: false,
          strict: "ignore"
        })
      } catch (e) {
        console.warn('KaTeX inline render error:', e.message)
        return `<span class="katex-error">$${content}$</span>`
      }
    }
  },

  // 客户端只需要加载 CSS
  clientConfigFile: path.resolve(import.meta.dirname, 'client.js')
}