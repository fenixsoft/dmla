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

      if (lang === 'mermaid') {
        return `<div class="mermaid">${code}</div>`
      }

      return defaultFence(tokens, idx, options, env, self)
    }
  },

  clientConfigFile: path.resolve(__dirname, 'client.js')
}