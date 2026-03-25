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
      // 自定义 fence 规则处理 runnable 代码块
      const defaultFence = md.renderer.rules.fence

      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const info = token.info.trim().toLowerCase()
        const code = token.content

        // 检查是否为 runnable 代码块
        if (info.includes('runnable')) {
          const language = info.replace('runnable', '').replace('gpu', '').trim() || 'python'
          const useGpu = info.includes('gpu')

          // 生成唯一 ID
          const id = `runnable-${idx}-${Date.now()}`

          // 返回自定义 HTML
          return `<div class="runnable-code-block" data-lang="${language}" data-gpu="${useGpu}" data-code="${encodeURIComponent(code)}">
  <div class="code-area">
    <pre><code class="language-${language}">${md.utils.escapeHtml(code)}</code></pre>
  </div>
  <div class="toolbar">
    <button class="run-btn" data-target="${id}">▶ Run</button>
    ${useGpu ? '<button class="run-btn gpu-btn" data-target="${id}" data-gpu="true">▶ Run on GPU</button>' : ''}
  </div>
  <div class="output-area" id="${id}"></div>
</div>`
        }

        return defaultFence(tokens, idx, options, env, self)
      }
    },

    clientConfigFile: path.resolve(__dirname, 'client.js')
  }
}

export default runnableCodePlugin