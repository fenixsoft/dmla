/**
 * nn-arch 插件 - 支持 Markdown 中的神经网络架构可视化
 */
import { getDirname, path } from 'vuepress/utils'

const __dirname = getDirname(import.meta.url)

/**
 * 解析尺寸参数
 * @param {string} lang - 语言标识符，如 "nn-arch width=800 height=400"
 * @returns {{width: number|null, height: number|null}}
 */
function parseSizeParams(lang) {
  const result = { width: null, height: null }

  // 匹配 width=N 或 height=N
  const widthMatch = lang.match(/width=(\d+)/i)
  const heightMatch = lang.match(/height=(\d+)/i)

  if (widthMatch) {
    result.width = parseInt(widthMatch[1], 10)
  }
  if (heightMatch) {
    result.height = parseInt(heightMatch[1], 10)
  }

  return result
}

export default {
  name: 'vuepress-plugin-nn-arch',

  extendsMarkdown: (md) => {
    const defaultFence = md.renderer.rules.fence

    md.renderer.rules.fence = (tokens, idx, options, env, self) => {
      const token = tokens[idx]
      const code = token.content.trim()
      const lang = token.info.trim()

      // 检查是否为 nn-arch 代码块
      if (lang.toLowerCase().startsWith('nn-arch')) {
        // 解析尺寸参数
        const sizeParams = parseSizeParams(lang)

        // 构建 data 属性
        const dataAttrs = [
          `data-nn-arch="${encodeURIComponent(code)}"`,
          sizeParams.width ? `data-width="${sizeParams.width}"` : '',
          sizeParams.height ? `data-height="${sizeParams.height}"` : ''
        ].filter(Boolean).join(' ')

        // 返回占位元素
        return `<div class="nn-arch-container" ${dataAttrs}><div class="nn-arch-loading">正在加载神经网络架构图...</div></div>`
      }

      return defaultFence(tokens, idx, options, env, self)
    }
  },

  clientConfigFile: path.resolve(__dirname, 'client.js')
}