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
        const info = token.info.trim().toLowerCase()

        // 检查是否为 runnable 代码块
        if (info.includes('runnable')) {
          const language = info.replace('runnable', '').replace('gpu', '').trim() || 'python'
          const useGpu = info.includes('gpu')

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

          // 返回可编辑的代码块，保留语法高亮
          return `<div class="runnable-code-block" data-lang="${language}" data-gpu="${useGpu}">
  <div class="code-area">
    <div class="floating-toolbar">
      <button class="run-btn" data-target="${id}">▶ Run</button>
      ${useGpu ? `<button class="run-btn gpu-btn" data-target="${id}" data-gpu="true">▶ Run on GPU</button>` : ''}
    </div>
    ${codeHtml}
  </div>
  <div class="output-area" id="${id}">点击 Run 按钮执行代码</div>
</div>`
        }

        return rawFence(tokens, idx, options, env, self)
      }
    },

    clientConfigFile: path.resolve(__dirname, 'client.js')
  }
}

export default runnableCodePlugin