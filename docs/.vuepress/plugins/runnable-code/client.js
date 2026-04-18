/**
 * 可运行代码插件客户端配置
 */
import { defineClientConfig } from 'vuepress/client'
import { onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import RunnableCode from './RunnableCode.vue'
import { getSandboxEndpoint } from './sandbox-config.js'

// 导入 Prism.js 用于客户端语法高亮
import Prism from 'prismjs'
import 'prismjs/components/prism-python'

// 内联样式确保加载
const styleId = 'runnable-code-styles'
if (typeof document !== 'undefined' && !document.getElementById(styleId)) {
  const style = document.createElement('style')
  style.id = styleId
  style.textContent = `
/* 全局隐藏所有代码块的行号 */
.line-numbers {
  display: none !important;
}

/* 可运行代码块样式 */
.runnable-code-block {
  margin: 1rem 0;
  border: 1px solid var(--c-border, #eaecef);
  border-radius: 8px;
  overflow: hidden;
}

.runnable-code-block .code-area {
  position: relative;
  margin: 0;
}

/* 浮动工具栏 - 右上角 */
.runnable-code-block .floating-toolbar {
  position: absolute;
  top: 8px;
  right: 12px;
  display: flex;
  gap: 6px;
  align-items: center;
  z-index: 10;
}

/* 可编辑代码区域 */
.runnable-code-block pre.runnable-editable {
  margin: 0;
  padding: 16px;
  background: var(--code-bg-color, #282c34);
  overflow-x: auto;
  cursor: text;
  outline: none;
  min-height: 60px;
}

/* 编辑模式下的样式 */
.runnable-code-block pre.runnable-editable:focus {
  background: #1e1e1e;
  box-shadow: inset 0 0 0 2px rgba(62, 175, 124, 0.3);
}

/* 非编辑模式下隐藏光标 */
.runnable-code-block pre.runnable-editable code[data-editing="false"] {
  caret-color: transparent;
}

.runnable-code-block  code {
  font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
  font-size: 14px;
  line-height: 1.5;
  display: block;
  white-space: pre;
}

/* 工具栏 */
.runnable-code-block .toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 8px 16px;
  background: #1E1E1E;
  border-top: 1px solid #333333;
}

.runnable-code-block .run-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  font-size: 13px;
  font-weight: 500;
  color: #fff;
  background: #3eaf7c;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
  opacity: 0.9;
  margin: 7px 3px 0 0;
}

.runnable-code-block .run-btn:hover:not(:disabled) {
  background: #4abf8a;
  opacity: 1;
}

.runnable-code-block .run-btn:disabled {
  background: #666;
  cursor: not-allowed;
  opacity: 0.6;
}

.runnable-code-block .run-btn.gpu-btn {
  background: #2563eb;
}

.runnable-code-block .run-btn.gpu-btn:hover:not(:disabled) {
  background: #3b82f6;
}

/* 输出区域 */
.runnable-code-block .output-area {
  padding: 12px 16px;
  background: #1e1e1e;
  color: #d4d4d4;
  font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
  font-size: 13px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 1000px;
  overflow-y: auto;
}

.runnable-code-block .output-area:empty::before {
  content: '点击 Run 按钮执行代码';
  color: #666;
}

.runnable-code-block .output-area.loading {
  color: #888;
}

.runnable-code-block .output-area.error {
  color: #f48771;
}

/* 文本流输出 */
.runnable-code-block .output-stream {
  margin: 0;
  padding: 0;
  background: transparent;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: inherit;
  font-size: inherit;
}

.runnable-code-block .output-stream.stderr {
  color: #f48771;
}

/* 图片输出 */
.runnable-code-block .output-image {
  max-width: 100%;
  border-radius: 8px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  margin: 0.5rem 0;
  cursor: pointer;
  transition: transform 0.2s;
}

.runnable-code-block .output-image:hover {
  transform: scale(1.02);
}

/* 执行结果 */
.runnable-code-block .output-result {
  margin: 0;
  padding: 0;
  background: transparent;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: inherit;
  font-size: inherit;
}

/* 错误输出 */
.runnable-code-block .output-error {
  margin: 0.5rem 0;
  padding: 0.5rem;
  background: rgba(244, 135, 113, 0.1);
  border-left: 3px solid #f48771;
  border-radius: 0 4px 4px 0;
}

.runnable-code-block .error-header {
  color: #f48771;
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.runnable-code-block .error-traceback {
  margin: 0;
  padding: 0;
  background: transparent;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: inherit;
  font-size: 12px;
  color: #888;
}

.runnable-code-block .execution-time {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #333;
  color: #888;
  font-size: 12px;
}

/* HTML 表格输出（pandas DataFrame 等） */
.runnable-code-block .output-html {
  margin: 0.5rem 0;
  overflow-x: auto;
}

.runnable-code-block .output-html table {
  border-collapse: collapse;
  width: 100%;
  font-size: 12px;
}

.runnable-code-block .output-html th,
.runnable-code-block .output-html td {
  border: 1px solid #444;
  padding: 6px 12px;
  text-align: left;
}

.runnable-code-block .output-html th {
  background: #2d2d2d;
  font-weight: 600;
  color: #fff;
}

.runnable-code-block .output-html td {
  color: #d4d4d4;
}

.runnable-code-block .output-html tr:nth-child(even) td {
  background: #252525;
}

.runnable-code-block .output-html tr:hover td {
  background: #333;
}

/* JSON 输出 */
.runnable-code-block .output-json {
  margin: 0;
  padding: 0;
  background: transparent;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: inherit;
  font-size: inherit;
}
`
  document.head.appendChild(style)
}

// MutationObserver 实例
let observer = null

export default defineClientConfig({
  enhance({ app }) {
    app.component('RunnableCode', RunnableCode)
  },

  setup() {
    const route = useRoute()

    // 启动 MutationObserver 监听 DOM 变化
    function startObserver() {
      if (observer) return // 避免重复启动

      observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
          for (const node of mutation.addedNodes) {
            // 检查新添加的节点是否包含 runnable-code-block
            if (node.nodeType === Node.ELEMENT_NODE) {
              if (node.classList?.contains('runnable-code-block')) {
                initCodeBlock(node)
              }
              // 检查子元素
              const codeBlocks = node.querySelectorAll?.('.runnable-code-block')
              codeBlocks?.forEach(initCodeBlock)
            }
          }
        }
      })

      observer.observe(document.body, {
        childList: true,
        subtree: true
      })
    }

    // 停止 MutationObserver
    function stopObserver() {
      if (observer) {
        observer.disconnect()
        observer = null
      }
    }

    // 首次加载时启动 observer 并初始化现有代码块
    onMounted(() => {
      startObserver()
      // 初始化当前页面已存在的代码块
      setTimeout(() => {
        document.querySelectorAll('.runnable-code-block').forEach(initCodeBlock)
      }, 100)
    })

    // 清理
    onBeforeUnmount(() => {
      stopObserver()
    })

    // 路由变化时确保初始化
    watch(
      () => route.path,
      async () => {
        // 等待 Vue 完成 DOM 更新
        await nextTick()
        // 额外延迟确保 markdown 渲染完成
        setTimeout(() => {
          document.querySelectorAll('.runnable-code-block').forEach(initCodeBlock)
        }, 100)
      }
    )
  }
})

/**
 * 初始化单个代码块
 */
function initCodeBlock(block) {
  // 避免重复初始化
  if (block.dataset.initialized) return
  block.dataset.initialized = 'true'

  const codeArea = block.querySelector('.code-area')
  if (!codeArea) return

  const preElement = codeArea.querySelector('pre')
  const codeElement = preElement?.querySelector('code')

  if (!codeElement) return

  // 移除行号元素
  const lineNumbersDivs = codeArea.querySelectorAll('.line-numbers')
  lineNumbersDivs.forEach(div => div.remove())

  // 让 code 元素可编辑（但默认不显示焦点样式）
  codeElement.contentEditable = 'true'
  codeElement.spellcheck = false
  codeElement.dataset.editing = 'false'
  preElement.classList.add('runnable-editable')

  // 点击时进入编辑模式
  codeElement.addEventListener('focus', () => {
    codeElement.dataset.editing = 'true'
  })

  // 失焦时退出编辑模式
  codeElement.addEventListener('blur', () => {
    codeElement.dataset.editing = 'false'
  })

  // 处理 Tab 和 Enter 键输入
  codeElement.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault()
      document.execCommand('insertText', false, '    ')
    }
    if (e.key === 'Enter') {
      e.preventDefault()
      document.execCommand('insertText', false, '\n')
    }
  })

  // 处理粘贴：去除格式，只保留纯文本
  codeElement.addEventListener('paste', (e) => {
    e.preventDefault()
    const text = e.clipboardData.getData('text/plain')
    document.execCommand('insertText', false, text)
    codeElement.dataset.modified = 'true'
  })

  // 标记编辑状态
  codeElement.addEventListener('input', () => {
    codeElement.dataset.modified = 'true'
  })

  // 失焦时重新进行语法高亮
  codeElement.addEventListener('blur', () => {
    if (codeElement.dataset.modified === 'true') {
      // 获取当前纯文本代码
      const code = codeElement.textContent
      const language = block.dataset.lang || 'python'

      // 保存光标位置（近似）
      const selection = window.getSelection()
      const range = selection.rangeCount > 0 ? selection.getRangeAt(0) : null
      const offset = range ? getTextOffset(codeElement, range.startContainer, range.startOffset) : 0

      // 使用 Prism 重新高亮
      const highlighted = Prism.highlight(code, Prism.languages[language] || Prism.languages.python, language)
      codeElement.innerHTML = highlighted

      // 尝试恢复光标位置
      try {
        if (offset > 0) {
          const newRange = document.createRange()
          const result = findPosition(codeElement, offset)
          if (result) {
            newRange.setStart(result.node, result.offset)
            newRange.collapse(true)
            selection.removeAllRanges()
            selection.addRange(newRange)
          }
        }
      } catch (e) {
        // 光标恢复失败，忽略
      }

      codeElement.dataset.modified = 'false'
    }
  })

  // 绑定运行按钮事件
  const runButtons = block.querySelectorAll('.run-btn')
  runButtons.forEach(btn => {
    btn.addEventListener('click', async (e) => {
      const outputArea = block.querySelector('.output-area')
      // 从 contenteditable 元素获取纯文本代码
      const code = codeElement.textContent
      const useGpu = e.target.classList.contains('gpu-btn')

      // 禁用按钮
      runButtons.forEach(b => {
        b.disabled = true
        b.textContent = 'Running...'
      })

      // 显示加载状态
      outputArea.className = 'output-area loading'
      outputArea.textContent = '执行中...'

      const endpoint = getSandboxEndpoint() + '/api/sandbox/run'

      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code, useGpu })
        })

        // 解析响应内容，即使 HTTP 状态码不是 200
        const result = await response.json()

        if (!response.ok) {
          // 显示后端返回的具体错误消息
          outputArea.className = 'output-area error'
          outputArea.textContent = result.error || `请求失败 (HTTP ${response.status})`
          return
        }
        const outputs = result.outputs || []

        if (result.success) {
          outputArea.className = 'output-area'
          outputArea.innerHTML = ''

          if (outputs.length === 0) {
            outputArea.textContent = '(无输出)'
          } else {
            // 渲染每个输出项
            outputs.forEach(output => {
              if (output.type === 'stream') {
                // 文本流输出
                const pre = document.createElement('pre')
                pre.className = `output-stream ${output.name || 'stdout'}`
                pre.textContent = output.text || ''
                outputArea.appendChild(pre)
              } else if (output.type === 'display_data' && output.data && output.data['image/png']) {
                // 图片输出
                const img = document.createElement('img')
                img.className = 'output-image'
                img.src = 'data:image/png;base64,' + output.data['image/png']
                img.style.maxWidth = '100%'
                img.style.borderRadius = '8px'
                img.style.margin = '0.5rem 0'
                img.style.cursor = 'pointer'
                // 点击放大
                img.addEventListener('click', () => {
                  openImageModal(output.data['image/png'])
                })
                outputArea.appendChild(img)
              } else if (output.type === 'display_data' && output.data && output.data['text/html']) {
                // HTML 输出（如 pandas DataFrame 表格）
                const htmlDiv = document.createElement('div')
                htmlDiv.className = 'output-html'
                htmlDiv.innerHTML = output.data['text/html']
                outputArea.appendChild(htmlDiv)
              } else if (output.type === 'display_data' && output.data && output.data['application/json']) {
                // JSON 输出
                const pre = document.createElement('pre')
                pre.className = 'output-json'
                pre.textContent = JSON.stringify(output.data['application/json'], null, 2)
                outputArea.appendChild(pre)
              } else if (output.type === 'execute_result') {
                // 执行结果 - 检查是否有富输出格式
                if (output.data && output.data['text/html']) {
                  // HTML 表格输出（如 pandas DataFrame）
                  const htmlDiv = document.createElement('div')
                  htmlDiv.className = 'output-html'
                  htmlDiv.innerHTML = output.data['text/html']
                  outputArea.appendChild(htmlDiv)
                } else if (output.data && output.data['image/png']) {
                  // 图片输出
                  const img = document.createElement('img')
                  img.className = 'output-image'
                  img.src = 'data:image/png;base64,' + output.data['image/png']
                  img.style.maxWidth = '100%'
                  img.style.borderRadius = '8px'
                  img.style.margin = '0.5rem 0'
                  img.style.cursor = 'pointer'
                  img.addEventListener('click', () => {
                    openImageModal(output.data['image/png'])
                  })
                  outputArea.appendChild(img)
                } else {
                  // 默认文本输出
                  const pre = document.createElement('pre')
                  pre.className = 'output-result'
                  pre.textContent = output.data && output.data['text/plain']
                    ? output.data['text/plain']
                    : JSON.stringify(output.data, null, 2)
                  outputArea.appendChild(pre)
                }
              } else if (output.type === 'error') {
                // 错误输出
                const errorDiv = document.createElement('div')
                errorDiv.className = 'output-error'
                errorDiv.innerHTML = `<div class="error-header">${output.ename}: ${output.evalue}</div>`
                if (output.traceback && output.traceback.length) {
                  const tracebackPre = document.createElement('pre')
                  tracebackPre.className = 'error-traceback'
                  tracebackPre.textContent = Array.isArray(output.traceback)
                    ? output.traceback.join('\n')
                    : String(output.traceback)
                  errorDiv.appendChild(tracebackPre)
                }
                outputArea.appendChild(errorDiv)
              }
            })
          }
        } else {
          outputArea.className = 'output-area error'
          outputArea.textContent = result.error || '执行失败'
        }

        if (result.executionTime) {
          const timeDiv = document.createElement('div')
          timeDiv.className = 'execution-time'
          timeDiv.textContent = `--- 执行时间: ${result.executionTime.toFixed(3)}s`
          outputArea.appendChild(timeDiv)
        }

      } catch (error) {
        outputArea.className = 'output-area error'
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
          outputArea.textContent = '⚠️ 无法连接到沙箱服务\n\n请确保沙箱服务正在运行，或在设置中检查沙箱地址配置'
        } else {
          outputArea.textContent = `❌ 错误: ${error.message}`
        }
      } finally {
        runButtons.forEach(b => {
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
}

/**
 * 获取文本偏移量
 */
function getTextOffset(root, targetNode, targetOffset) {
  let offset = 0
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false)
  while (walker.nextNode()) {
    if (walker.currentNode === targetNode) {
      return offset + targetOffset
    }
    offset += walker.currentNode.textContent.length
  }
  return offset
}

/**
 * 根据偏移量找到位置
 */
function findPosition(root, offset) {
  let currentOffset = 0
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false)
  while (walker.nextNode()) {
    const nodeLength = walker.currentNode.textContent.length
    if (currentOffset + nodeLength >= offset) {
      return { node: walker.currentNode, offset: offset - currentOffset }
    }
    currentOffset += nodeLength
  }
  return null
}

/**
 * 打开图片模态框
 */
function openImageModal(base64Data) {
  // 移除已存在的模态框
  const existingModal = document.getElementById('runnable-image-modal')
  if (existingModal) {
    existingModal.remove()
  }

  // 创建模态框
  const modal = document.createElement('div')
  modal.id = 'runnable-image-modal'
  modal.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.9);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    cursor: pointer;
  `

  // 创建图片
  const img = document.createElement('img')
  img.src = 'data:image/png;base64,' + base64Data
  img.style.cssText = `
    max-width: 95vw;
    max-height: 95vh;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 0 40px rgba(0, 0, 0, 0.5);
  `

  // 创建关闭按钮
  const closeBtn = document.createElement('div')
  closeBtn.textContent = '×'
  closeBtn.style.cssText = `
    position: absolute;
    top: 20px;
    right: 20px;
    width: 40px;
    height: 40px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    color: white;
    cursor: pointer;
  `

  modal.appendChild(img)
  modal.appendChild(closeBtn)

  // 点击关闭
  modal.addEventListener('click', () => {
    modal.remove()
  })

  // ESC 键关闭
  const handleEsc = (e) => {
    if (e.key === 'Escape') {
      modal.remove()
      document.removeEventListener('keydown', handleEsc)
    }
  }
  document.addEventListener('keydown', handleEsc)

  document.body.appendChild(modal)
}