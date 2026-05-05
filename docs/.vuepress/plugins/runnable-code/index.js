/**
 * VuePress 可运行代码插件
 * 支持 Python 代码在本地模式下执行
 */
import { path } from 'vuepress/utils'
import { getDirname } from 'vuepress/utils'

const __dirname = getDirname(import.meta.url)

export const runnableCodePlugin = (options = {}) => {
  const {
    apiEndpoint = 'http://localhost:3001/api/sandbox/run',
    enabledInBuild = false  // 生产构建中是否启用
  } = options

  return {
    name: 'vuepress-plugin-runnable-code',

    define: {
      __RUNNABLE_API_ENDPOINT__: apiEndpoint,
      __RUNNABLE_ENABLED__: enabledInBuild
    },

    extendsMarkdown: (md) => {
      // 保存当前的 fence 规则（可能是其他插件已经处理过的，如 Prism.js）
      const rawFence = md.renderer.rules.fence

      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const info = token.info.trim()

        // 检查是否为 runnable 代码块
        if (info.includes('runnable')) {
          const infoLower = info.toLowerCase()
          // 移除所有标记参数，提取语言名称
          const language = info.replace(/runnable|gpu|gpuonly|timeout=\S+|extract-class="[^"]*"/gi, '').trim() || 'python'
          const useGpu = infoLower.includes('gpu')
          const gpuOnly = infoLower.includes('gpuonly')

          // 解析 timeout 参数
          const timeoutMatch = info.match(/timeout=(\d+|unlimited)/i)
          const timeout = timeoutMatch ? timeoutMatch[1] : null

          // 解析 extract-class 参数
          const extractMatch = info.match(/extract-class="([^"]+)"/i)
          const extractClass = extractMatch ? extractMatch[1] : null

          // 生成唯一 ID
          const id = `runnable-${idx}-${Date.now()}`

          // 调用原始 fence 规则获取语法高亮后的 HTML（已包含行号）
          const highlightedHtml = rawFence(tokens, idx, options, env, self)

          // 解析高亮后的 HTML，提取内部内容
          let codeHtml = highlightedHtml
          const divMatch = highlightedHtml.match(/^<div class="[^"]*"[^>]*>([\s\S]*)<\/div>$/)
          if (divMatch) {
            codeHtml = divMatch[1]
          }

          // 构建 data 属性
          const dataAttrs = [
            `data-lang="${language}"`,
            `data-gpu="${useGpu}"`,
            `data-gpu-only="${gpuOnly}"`,
            timeout ? `data-timeout="${timeout}"` : '',
            extractClass ? `data-extract-class="${extractClass}"` : ''
          ].filter(Boolean).join(' ')

          // 渲染按钮：gpuonly 时只显示 GPU 按钮，否则显示普通 Run 按钮（有 gpu 时也显示 GPU 按钮）
          const runBtn = gpuOnly ? '' : `<button class="run-btn" data-target="${id}">▶ Run</button>`
          const gpuBtn = (useGpu || gpuOnly) ? `<button class="run-btn gpu-btn" data-target="${id}" data-gpu="true">▶ Run on GPU</button>` : ''

          // 返回可编辑的代码块，保留语法高亮
          return `<div class="runnable-code-block" ${dataAttrs}>
  <div class="code-area">
    <div class="floating-toolbar">
      ${runBtn}${gpuBtn}
    </div>
    ${codeHtml}
  </div>
  <div class="output-area output-container" id="${id}">点击 Run 按钮执行代码</div>
</div>`
        }

        return rawFence(tokens, idx, options, env, self)
      }
    },

    clientConfigFile: path.resolve(__dirname, 'client.js')
  }
}

export default runnableCodePlugin