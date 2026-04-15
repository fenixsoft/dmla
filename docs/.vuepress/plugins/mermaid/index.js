/**
 * Mermaid 插件 - 支持 Markdown 中的流程图
 */
import { getDirname, path } from 'vuepress/utils'

const __dirname = getDirname(import.meta.url)

export default {
  name: 'vuepress-plugin-mermaid',

  extendsMarkdown: (md) => {
    const defaultFence = md.renderer.rules.fence

    md.renderer.rules.fence = (tokens, idx, options, env, self) => {
      const token = tokens[idx]
      const code = token.content.trim()
      const lang = token.info.trim().toLowerCase()

      // 支持大小修饰符：mermaid small, mermaid compact, mermaid tiny
      // small: zoom 0.85, compact: zoom 0.75, tiny: zoom 0.6
      if (lang.startsWith('mermaid')) {
        // 转义 HTML 特殊字符，保留原始代码用于调试
        const escapedCode = code
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')

        // 解析大小修饰符，只接受 small, compact, tiny
        const parts = lang.split(/\s+/)
        const sizeModifier = parts[1]
        const validSizes = ['small', 'compact', 'tiny']
        const sizeClass = validSizes.includes(sizeModifier) ? `mermaid-${sizeModifier}` : ''

        return `<pre class="mermaid ${sizeClass}"><code>${escapedCode}</code></pre>`
      }

      return defaultFence(tokens, idx, options, env, self)
    }
  },

  clientConfigFile: path.resolve(__dirname, 'client.js')
}