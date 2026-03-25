/**
 * 可运行代码插件客户端配置
 */
import { defineClientConfig } from 'vuepress/client'
import RunnableCode from './RunnableCode.vue'

export default defineClientConfig({
  enhance({ app }) {
    app.component('RunnableCode', RunnableCode)
  },

  setup() {
    // 初始化可运行代码块
    if (typeof window !== 'undefined') {
      initRunnableCodeBlocks()
    }
  }
})

function initRunnableCodeBlocks() {
  document.addEventListener('DOMContentLoaded', () => {
    const runButtons = document.querySelectorAll('.runnable-code-block .run-btn')

    runButtons.forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const block = e.target.closest('.runnable-code-block')
        const outputArea = block.querySelector('.output-area')
        const code = decodeURIComponent(block.dataset.code)
        const useGpu = e.target.classList.contains('gpu-btn') || block.dataset.gpu === 'true'

        // 禁用按钮
        const buttons = block.querySelectorAll('.run-btn')
        buttons.forEach(b => {
          b.disabled = true
          b.textContent = 'Running...'
        })

        // 显示加载状态
        outputArea.className = 'output-area loading'
        outputArea.textContent = '执行中...'

        try {
          const response = await fetch(window.__RUNNABLE_API_ENDPOINT__ || 'http://localhost:3001/api/sandbox/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, useGpu })
          })

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`)
          }

          const result = await response.json()

          if (result.success) {
            outputArea.className = 'output-area'
            outputArea.textContent = result.output || '(无输出)'
          } else {
            outputArea.className = 'output-area error'
            outputArea.textContent = result.error || '执行失败'
          }

          // 显示执行时间
          if (result.executionTime) {
            outputArea.textContent += `\n\n---\n执行时间: ${result.executionTime.toFixed(3)}s`
          }

        } catch (error) {
          outputArea.className = 'output-area error'
          if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            outputArea.textContent = '⚠️ 无法连接到沙箱服务\n\n请确保本地服务正在运行:\nnpm run local'
          } else {
            outputArea.textContent = `❌ 错误: ${error.message}`
          }
        } finally {
          // 恢复按钮
          buttons.forEach(b => {
            b.disabled = false
            if (b.classList.contains('gpu-btn')) {
              b.textContent = '▶ Run on GPU'
            } else {
              b.textContent = '▶ Run'
            }
          })
        }
      })
    })
  })
}