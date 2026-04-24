/**
 * Math 插件 - 支持 Markdown 中的 LaTeX 数学公式
 * 使用 KaTeX 在服务端渲染，支持公式编号和引用
 */
import katex from 'katex'
import { path } from 'vuepress/utils'

// 全局公式计数器，用于自动编号
let equationCounter = 0
// 存储公式标签到编号的映射，用于引用
const equationLabels = new Map()

/**
 * 处理 HTML 标签内的数学公式
 * markdown-it 将 HTML 标签视为原始 HTML，不解析内部的 Markdown 内容
 * 此函数使用后处理方式，对已渲染的 HTML 中的 $...$ 进行解析
 */
const processMathInHtml = (html) => {
  // 先处理 \eqref{...} 和 \ref{...}
  html = html.replace(/\\(eq)?ref\{([^}]+)\}/g, (match, eqPrefix, label) => {
    const equationNumber = equationLabels.get(label)
    if (equationNumber) {
      const display = eqPrefix ? `(${equationNumber})` : `${equationNumber}`
      return `<a href="#eq-${label}" class="equation-reference">${display}</a>`
    }
    return `<span class="equation-reference-unknown">${label}?</span>`
  })

  // 处理行内公式 $...$（排除已渲染的 katex 内容）
  // 使用正则匹配不在 katex 相关标签内的 $...$
  return html.replace(/\$([^$\n]+)\$/g, (match, content) => {
    // 检查是否在 katex 标签内（避免重复处理）
    // 由于是全局替换，这里只是简单处理
    if (!content.trim()) return match
    try {
      return katex.renderToString(content, {
        throwOnError: false,
        displayMode: false,
        strict: "ignore"
      })
    } catch (e) {
      console.warn('KaTeX inline render error in HTML:', e.message)
      return match
    }
  })
}

export default {
  name: 'vuepress-plugin-math-katex',

  // 每个页面开始时重置计数器
  onPrepared: (app) => {
    // 页面切换时重置计数器
    equationCounter = 0
    equationLabels.clear()
  },

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

    // 解析 equation 环境 \begin{equation} ... \end{equation}
    const parseEquationEnv = (state, startLine, endLine, silent) => {
      const start = state.bMarks[startLine] + state.tShift[startLine]
      const max = state.eMarks[startLine]

      // 检查是否以 \begin{equation} 开头（Markdown 中写作 \begin，实际是单个反斜杠）
      const beginMarker = '\\begin{equation}'
      const lineContent = state.src.slice(start, max).trim()
      if (!lineContent.startsWith(beginMarker)) {
        return false
      }

      // 查找结束的 \end{equation}
      for (let nextLine = startLine; nextLine < endLine; nextLine++) {
        const lineStart = state.bMarks[nextLine] + state.tShift[nextLine]
        const lineMax = state.eMarks[nextLine]
        const lineContent = state.src.slice(lineStart, lineMax)

        // 跳过第一行（开始标记所在行）
        if (nextLine === startLine) {
          // 检查是否在同一行结束
          const endMarker = '\\end{equation}'
          const endIndex = lineContent.indexOf(endMarker, beginMarker.length)
          if (endIndex !== -1) {
            const content = lineContent.slice(beginMarker.length, endIndex).trim()
            if (!silent) {
              const token = state.push('math_equation', 'div', 0)
              token.content = content
              token.block = true
            }
            state.line = startLine + 1
            return true
          }
          continue
        }

        // 检查是否是结束行
        if (lineContent.trim().startsWith('\\end{equation}')) {
          const contentStart = state.bMarks[startLine] + state.tShift[startLine] + beginMarker.length
          const contentEnd = lineStart
          const content = state.src.slice(contentStart, contentEnd).trim()

          if (!silent) {
            const token = state.push('math_equation', 'div', 0)
            token.content = content
            token.block = true
          }
          state.line = nextLine + 1
          return true
        }

        // 检查行内是否有 \end{equation}
        const endMarker = '\\end{equation}'
        const endIndex = lineContent.indexOf(endMarker)
        if (endIndex !== -1) {
          const contentStart = state.bMarks[startLine] + state.tShift[startLine] + beginMarker.length
          const contentEnd = lineStart + endIndex
          const content = state.src.slice(contentStart, contentEnd).trim()

          if (!silent) {
            const token = state.push('math_equation', 'div', 0)
            token.content = content
            token.block = true
          }
          state.line = nextLine + 1
          return true
        }
      }

      return false
    }

    // 渲染 equation 环境（带编号）
    md.renderer.rules.math_equation = (tokens, idx) => {
      const content = tokens[idx].content
      equationCounter++
      const equationNumber = equationCounter

      // 提取 \label{...} 标签
      const labelMatch = content.match(/\\label\{([^}]+)\}/)
      const label = labelMatch ? labelMatch[1] : null

      // 移除 \label{...} 用于渲染
      const cleanContent = content.replace(/\\label\{[^}]+\}/g, '').trim()

      // 如果有标签，存储映射关系
      if (label) {
        equationLabels.set(label, equationNumber)
      }

      try {
        const html = katex.renderToString(cleanContent, {
          throwOnError: false,
          displayMode: true,
          strict: "ignore"
        })

        // 包装为带编号的公式，添加锚点以便引用跳转
        const anchorAttr = label ? `id="eq-${label}"` : ''
        return `<div class="katex-display equation-numbered" data-equation-number="${equationNumber}" data-label="${label || ''}" ${anchorAttr}>
          <div class="equation-content">${html}</div>
          <div class="equation-number">(${equationNumber})</div>
        </div>`
      } catch (e) {
        console.warn('KaTeX equation render error:', e.message)
        return `<div class="katex-error">\\begin{equation}${content}\\end{equation}</div>`
      }
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

    // 注册 equation 环境解析规则（优先级高于 math_block）
    md.block.ruler.before('math_block', 'math_equation', parseEquationEnv, {
      alt: ['paragraph', 'reference', 'blockquote', 'list']
    })

    // 处理行内公式 $...$ 和引用 \eqref{...} \ref{...}
    md.inline.ruler.before('escape', 'math_inline', (state, silent) => {
      const start = state.pos
      const max = state.posMax

      // 检查是否是 \eqref{...} 或 \ref{...}
      const refPattern = /^\\(eq)?ref\{([^}]+)\}/
      const remaining = state.src.slice(start)
      const refMatch = remaining.match(refPattern)

      if (refMatch) {
        if (!silent) {
          const refType = refMatch[1] ? 'eqref' : 'ref'
          const label = refMatch[2]
          const token = state.push(`math_${refType}`, 'span', 0)
          token.content = label
        }
        state.pos += refMatch[0].length
        return true
      }

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

    // 渲染 \eqref{...}
    md.renderer.rules.math_eqref = (tokens, idx) => {
      const label = tokens[idx].content
      const equationNumber = equationLabels.get(label)
      if (equationNumber) {
        return `<a href="#eq-${label}" class="equation-reference">(${equationNumber})</a>`
      }
      return `<span class="equation-reference-unknown">(${label}?)</span>`
    }

    // 渲染 \ref{...}
    md.renderer.rules.math_ref = (tokens, idx) => {
      const label = tokens[idx].content
      const equationNumber = equationLabels.get(label)
      if (equationNumber) {
        return `<a href="#eq-${label}" class="equation-reference">${equationNumber}</a>`
      }
      return `<span class="equation-reference-unknown">${label}?</span>`
    }

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

    // 处理 HTML 块内的数学公式和公式编号标记
    // markdown-it 将 HTML 标签视为原始内容，不解析内部的 Markdown
    // 这里我们在 core ruler 中添加一个规则，在解析完成后处理 HTML 块内的 $...$ 和公式编号标记
    md.core.ruler.push('process_math_in_html', (state) => {
      for (const token of state.tokens) {
        // 处理 html_block 和 html_inline 类型的 token
        if (token.type === 'html_block' || token.type === 'html_inline') {
          if (token.content && token.content.includes('$')) {
            token.content = processMathInHtml(token.content)
          }
        }

        // 处理段落中的公式编号 HTML 注释标记
        if (token.type === 'paragraph_open' || token.type === 'paragraph_close') {
          // 查找相邻的 html_inline token
          continue
        }
      }

      // 处理公式编号标记 <!-- equation:label=xxx --> ... <!-- end-equation -->
      // 以及 <!-- eqref:xxx --> 引用标记
      // 注意：HTML 注释可能是 html_block（独立行）或 html_inline（行内）
      let i = 0
      while (i < state.tokens.length) {
        const token = state.tokens[i]

        // 处理 inline token 的 children（公式引用可能在行内文本中）
        if (token.type === 'inline' && token.children) {
          for (const child of token.children) {
            if (child.type === 'html_inline' && child.content) {
              const eqrefMatch = child.content.match(/<!--\s*eqref:([^>]+)\s*-->/)
              if (eqrefMatch) {
                const label = eqrefMatch[1].trim()
                const equationNumber = equationLabels.get(label)
                if (equationNumber) {
                  child.type = 'html_inline'
                  child.content = `<a href="#eq-${label}" class="equation-reference">(${equationNumber})</a>`
                } else {
                  child.type = 'html_inline'
                  child.content = `<span class="equation-reference-unknown">(${label}?)</span>`
                }
              }
            }
          }
        }

        // 检查是否是公式编号开始标记（html_block）
        if (token.type === 'html_block' && token.content) {
          const labelMatch = token.content.match(/<!--\s*equation:label=([^>]+)\s*-->/)
          if (labelMatch) {
            const label = labelMatch[1].trim()
            equationCounter++
            const equationNumber = equationCounter
            equationLabels.set(label, equationNumber)

            // 查找下一个 math_block token（公式内容）和结束标记
            let mathBlockIndex = -1
            let endMarkerIndex = -1
            let j = i + 1
            while (j < state.tokens.length) {
              const nextToken = state.tokens[j]
              // 检查结束标记（html_block）
              if (nextToken.type === 'html_block' && nextToken.content && nextToken.content.includes('<!-- end-equation -->')) {
                endMarkerIndex = j
                break
              }
              if (nextToken.type === 'math_block' && mathBlockIndex === -1) {
                mathBlockIndex = j
              }
              j++
            }

            // 如果找到了 math_block，修改它的渲染
            if (mathBlockIndex !== -1) {
              const mathToken = state.tokens[mathBlockIndex]
              const originalContent = mathToken.content
              try {
                const html = katex.renderToString(originalContent, {
                  throwOnError: false,
                  displayMode: true,
                  strict: "ignore"
                })
                mathToken.type = 'html_block'
                mathToken.tag = 'div'
                mathToken.content = `<div class="katex-display equation-numbered" id="eq-${label}" data-equation-number="${equationNumber}" data-label="${label}">
                  <div class="equation-content">${html}</div>
                  <div class="equation-number">(${equationNumber})</div>
                </div>`
              } catch (e) {
                console.warn('KaTeX equation render error:', e.message)
              }
            }

            // 移除开始标记 token
            token.type = 'text'
            token.content = ''
            // 移除结束标记 token
            if (endMarkerIndex !== -1) {
              state.tokens[endMarkerIndex].type = 'text'
              state.tokens[endMarkerIndex].content = ''
            }
          }

          // 检查是否是公式引用标记（在独立行的 html_block 中）
          const eqrefMatch = token.content.match(/<!--\s*eqref:([^>]+)\s*-->/)
          if (eqrefMatch) {
            const label = eqrefMatch[1]
            const equationNumber = equationLabels.get(label)
            if (equationNumber) {
              token.content = `<a href="#eq-${label}" class="equation-reference">(${equationNumber})</a>`
            } else {
              token.content = `<span class="equation-reference-unknown">(${label}?)</span>`
            }
          }
        }
        i++
      }
    })
  },

  // 客户端只需要加载 CSS
  clientConfigFile: path.resolve(import.meta.dirname, 'client.js')
}