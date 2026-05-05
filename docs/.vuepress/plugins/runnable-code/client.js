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

/**
 * 去除 ANSI 转义码
 * IPython/Jupyter 的 traceback 包含彩色输出的 ANSI 转义码（如 [36m、[31m 等）
 * 在网页中需要过滤掉这些字符，只保留纯文本
 * @param {string} text - 包含 ANSI 转义码的文本
 * @returns {string} - 纯文本
 */
function stripAnsi(text) {
  // 匹配两种格式的 ANSI 转义码：
  // 1. \x1b[...m（完整的 ESC 序列）
  // 2. [...m（部分显示格式，如 [36m）
  return text.replace(/\x1b\[[0-9;]*m/g, '').replace(/\[[0-9;]*m/g, '')
}

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

/* 停止按钮样式 */
.runnable-code-block .stop-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 6px 12px;
  font-size: 13px;
  font-weight: 500;
  color: #fff;
  background: #dc2626;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
  margin: 7px 3px 0 0;
}

.runnable-code-block .stop-btn:hover {
  background: #ef4444;
}

/* 输出区域 */
.runnable-code-block .output-container {
  padding: 12px 16px;
  background: #1E1E1E;
  border-top: 1px solid #333333;
  font-family: 'Fira Code', monospace;
  font-size: 13px;
  line-height: 1.5;
  color: #ffffff;
  white-space: pre-wrap;
  max-height: 1000px;
  overflow-y: auto;
}

.runnable-code-block .output-area {
  padding: 12px 16px;
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

/* 进度条容器（固定在顶部，与文本输出分开） */
.runnable-code-block .progress-container {
  padding: 12px 16px;
  background: #1e1e1e;
  border-bottom: 1px solid #333;
  display: grid;
}

/* 进度条 */
.runnable-code-block .progress-bar {
  // margin-bottom: 1rem;
  display: grid;
}

.runnable-code-block .progress-header {
  color: #4ec9b0;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.runnable-code-block .progress-track {
  width: 100%;
  height: 8px;
  background: #333;
  border-radius: 4px;
  overflow: hidden;
  display: flex;
}

.runnable-code-block .progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3eaf7c, #4abf8a);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.runnable-code-block .progress-info {
  display: flex;
  justify-content: space-between;
  margin-top: 0.5rem;
  color: #888;
  font-size: 12px;
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

  const toolbar = block.querySelector('.floating-toolbar')  // 工具栏（用于添加停止按钮）

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
  const timeout = block.dataset.timeout || null

  // 格式化时间（秒转为 mm:ss）
  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // 处理流式消息（进度条和文本输出分开）
  function handleStreamMessage(msg, progressContainer, textOutput) {
    switch (msg.type) {
      case 'status':
        // 状态消息（starting/running/complete）
        if (msg.status === 'starting') {
          progressContainer.innerHTML = `
            <div class="progress-bar">
              <div class="progress-header">${msg.message || '正在启动容器...'}</div>
            </div>
          `
          textOutput.className = 'output-area loading'
          textOutput.textContent = ''
        } else if (msg.status === 'running') {
          // 运行中状态，更新进度条并清除加载提示
          progressContainer.innerHTML = `
            <div class="progress-bar">
              <div class="progress-header">${msg.message || '代码执行中...'}</div>
            </div>
          `
          textOutput.className = 'output-area'
          if (textOutput.textContent === '执行中...' || textOutput.textContent === '正在启动容器...') {
            textOutput.innerHTML = ''
          }
        }
        break

      case 'progress':
        // 进度更新消息 - 只更新进度条区域
        const percent = msg.percent || 0
        const elapsed = msg.elapsed_seconds || 0
        const remaining = msg.estimated_remaining || null

        progressContainer.innerHTML = `
          <div class="progress-bar">
            <div class="progress-header">${msg.description || '执行中...'}</div>
            <div class="progress-track">
              <div class="progress-fill" style="width: ${percent}%"></div>
            </div>
            <div class="progress-info">
              <span>${msg.message || `${msg.current_step || 0}/${msg.total_steps || 0}`}</span>
              <span>${formatTime(elapsed)}${remaining ? ' / ' + formatTime(remaining) : ''}</span>
            </div>
          </div>
        `
        // 清除文本区域的加载提示
        textOutput.className = 'output-area'
        if (textOutput.textContent === '执行中...' || textOutput.textContent === '正在启动容器...') {
          textOutput.innerHTML = ''
        }
        break

      case 'stream':
        // 文本流输出 - 只追加到文本区域，不影响进度条
        textOutput.className = 'output-area'
        if (textOutput.textContent === '执行中...' || textOutput.textContent === '正在启动容器...') {
          textOutput.innerHTML = ''
        }

        // 检查是否是嵌套的 progress JSON（来自 dmla_progress.py）
        const text = msg.text || ''
        if (text.trim().startsWith('{') && (text.includes('"type": "progress"') || text.includes('"type":"progress"'))) {
          try {
            const progressData = JSON.parse(text.trim())
            if (progressData.type === 'progress') {
              // 更新进度条区域，不追加到文本区域
              const pgPercent = progressData.percent || 0
              const pgElapsed = progressData.elapsed_seconds || 0
              const pgRemaining = progressData.estimated_remaining || null

              progressContainer.innerHTML = `
                <div class="progress-bar">
                  <div class="progress-header">${progressData.description || '执行中...'}</div>
                  <div class="progress-track">
                    <div class="progress-fill" style="width: ${pgPercent}%"></div>
                  </div>
                  <div class="progress-info">
                    <span>${progressData.message || `${progressData.current_step || 0}/${progressData.total_steps || 0}`}</span>
                    <span>${formatTime(pgElapsed)}${pgRemaining ? ' / ' + formatTime(pgRemaining) : ''}</span>
                  </div>
                </div>
              `
              break  // 不追加到文本区域
            }
          } catch (parseError) {
            // 解析失败，作为普通文本输出
          }
        }

        // 普通文本输出 - 追加到文本区域（去除 ANSI 转义码）
        const pre = document.createElement('pre')
        pre.className = `output-stream ${msg.name || 'stdout'}`
        pre.textContent = stripAnsi(msg.text || '')
        textOutput.appendChild(pre)
        break

      case 'display_data':
        // 富输出（图片、HTML 等）- 追加到文本区域
        textOutput.className = 'output-area'
        if (textOutput.textContent === '执行中...' || textOutput.textContent === '正在启动容器...') {
          textOutput.innerHTML = ''
        }

        if (msg.data && msg.data['image/png']) {
          const img = document.createElement('img')
          img.className = 'output-image'
          img.src = 'data:image/png;base64,' + msg.data['image/png']
          img.style.maxWidth = '100%'
          img.style.borderRadius = '8px'
          img.style.margin = '0.5rem 0'
          img.style.cursor = 'pointer'
          img.addEventListener('click', () => {
            openImageModal(msg.data['image/png'])
          })
          textOutput.appendChild(img)
        } else if (msg.data && msg.data['text/html']) {
          const htmlDiv = document.createElement('div')
          htmlDiv.className = 'output-html'
          htmlDiv.innerHTML = msg.data['text/html']
          textOutput.appendChild(htmlDiv)
        } else if (msg.data && msg.data['application/json']) {
          const pre = document.createElement('pre')
          pre.className = 'output-json'
          pre.textContent = JSON.stringify(msg.data['application/json'], null, 2)
          textOutput.appendChild(pre)
        }
        break

      case 'execute_result':
        // 执行结果 - 追加到文本区域
        textOutput.className = 'output-area'
        if (textOutput.textContent === '执行中...' || textOutput.textContent === '正在启动容器...') {
          textOutput.innerHTML = ''
        }

        if (msg.data && msg.data['text/html']) {
          const htmlDiv = document.createElement('div')
          htmlDiv.className = 'output-html'
          htmlDiv.innerHTML = msg.data['text/html']
          textOutput.appendChild(htmlDiv)
        } else if (msg.data && msg.data['image/png']) {
          const img = document.createElement('img')
          img.className = 'output-image'
          img.src = 'data:image/png;base64,' + msg.data['image/png']
          img.style.maxWidth = '100%'
          img.style.borderRadius = '8px'
          img.style.margin = '0.5rem 0'
          img.style.cursor = 'pointer'
          img.addEventListener('click', () => {
            openImageModal(msg.data['image/png'])
          })
          textOutput.appendChild(img)
        } else if (msg.data && msg.data['text/plain']) {
          const pre = document.createElement('pre')
          pre.className = 'output-result'
          pre.textContent = msg.data['text/plain']
          textOutput.appendChild(pre)
        }
        break

      case 'error':
        // 错误输出 - 追加到文本区域
        textOutput.className = 'output-area error'
        if (textOutput.textContent === '执行中...' || textOutput.textContent === '正在启动容器...') {
          textOutput.innerHTML = ''
        }

        const errorDiv = document.createElement('div')
        errorDiv.className = 'output-error'
        // 去除 ANSI 转义码，显示纯文本
        const cleanEvalue = stripAnsi(msg.evalue || '')
        errorDiv.innerHTML = `<div class="error-header">${msg.ename}: ${cleanEvalue}</div>`
        if (msg.traceback && msg.traceback.length) {
          const tracebackPre = document.createElement('pre')
          tracebackPre.className = 'error-traceback'
          // 去除 ANSI 转义码，显示纯文本 traceback
          const cleanTraceback = Array.isArray(msg.traceback)
            ? msg.traceback.map(stripAnsi).join('\n')
            : stripAnsi(String(msg.traceback))
          tracebackPre.textContent = cleanTraceback
          errorDiv.appendChild(tracebackPre)
        }
        textOutput.appendChild(errorDiv)
        break

      case 'result':
        // 最终结果汇总 - 根据成功状态更新进度条
        if (msg.success) {
          progressContainer.innerHTML = `
            <div class="progress-bar">
              <div class="progress-header">✅ 代码执行完毕</div>
            </div>
          `
        } else {
          progressContainer.innerHTML = `
            <div class="progress-bar">
              <div class="progress-header">❌ 执行失败</div>
            </div>
          `
        }
        if (msg.executionTime) {
          const timeDiv = document.createElement('div')
          timeDiv.className = 'execution-time'
          timeDiv.textContent = `--- 执行时间: ${msg.executionTime.toFixed(3)}s`
          textOutput.appendChild(timeDiv)
        }
        break
    }
  }

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

      // 创建 AbortController 用于中止请求
      const abortController = new AbortController()

      // 创建停止按钮
      const stopBtn = document.createElement('button')
      stopBtn.className = 'stop-btn'
      stopBtn.textContent = 'Stop'
      if (toolbar) toolbar.appendChild(stopBtn)

      // 创建两个独立的容器
      // 1. 进度条容器（固定在顶部）
      const progressContainer = document.createElement('div')
      progressContainer.className = 'progress-container'

      // 2. 文本输出容器（在进度条下方）
      const textOutput = document.createElement('div')
      textOutput.className = 'output-area'

      // 清空原有输出区域，插入新结构
      outputArea.innerHTML = ''
      outputArea.appendChild(progressContainer)
      outputArea.appendChild(textOutput)

      // 停止按钮点击事件
      stopBtn.addEventListener('click', async () => {
        // 中止 fetch 请求
        abortController.abort()

        // 调用后端中止 API
        try {
          const abortEndpoint = getSandboxEndpoint() + '/api/sandbox/abort'
          await fetch(abortEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
          })
        } catch {
          // 忽略中止 API 错误
        }

        // 移除停止按钮
        stopBtn.remove()

        // 恢复按钮状态
        runButtons.forEach(b => {
          b.disabled = false
          if (b.classList.contains('gpu-btn')) {
            b.textContent = '▶ Run on GPU'
          } else {
            b.textContent = '▶ Run'
          }
        })

        // 显示已中止状态
        progressContainer.innerHTML = ''
        const abortedMsg = document.createElement('pre')
        abortedMsg.className = 'output-stream stdout'
        abortedMsg.textContent = textOutput.textContent ? '\n--- 已中止 ---' : '已中止'
        textOutput.appendChild(abortedMsg)
      })

      // 显示初始状态
      progressContainer.innerHTML = `
        <div class="progress-bar">
          <div class="progress-header">正在启动容器...</div>
        </div>
      `
      textOutput.className = 'output-area loading'
      textOutput.textContent = ''

      // 使用流式端点
      const endpoint = getSandboxEndpoint() + '/api/sandbox/stream'

      // 连接超时检测（5秒）
      // 当服务未启动时，fetch 会长时间等待，需要主动超时
      const CONNECTION_TIMEOUT = 5000
      let timeoutId = null
      const connectionTimeoutPromise = new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
          reject(new Error('ConnectionTimeout'))
        }, CONNECTION_TIMEOUT)
      })

      try {
        // 使用 Promise.race 实现连接超时
        const response = await Promise.race([
          fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              code,
              useGpu,
              timeout: timeout === 'unlimited' ? null : (timeout ? parseInt(timeout, 10) : null)
            }),
            signal: abortController.signal
          }),
          connectionTimeoutPromise
        ])

        // 连接成功，清除超时计时器
        if (timeoutId) clearTimeout(timeoutId)

        if (!response.ok) {
          // HTTP 错误响应
          progressContainer.innerHTML = ''
          textOutput.className = 'output-area error'
          try {
            const errorResult = await response.json()
            textOutput.textContent = errorResult.error || `请求失败 (HTTP ${response.status})`
          } catch {
            textOutput.textContent = `请求失败 (HTTP ${response.status})`
          }
          return
        }

        // 流式读取响应
        const reader = response.body.getReader()
        const decoder = new TextDecoder()

        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })

          // 按行分割并处理
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''  // 保留不完整的行

          for (const line of lines) {
            if (!line.trim()) continue
            try {
              const msg = JSON.parse(line)
              handleStreamMessage(msg, progressContainer, textOutput)
            } catch (parseError) {
              // JSON 解析错误，可能是非标准输出，作为普通文本处理
              console.warn('Failed to parse JSON line:', line.substring(0, 100))
            }
          }
        }

        // 处理剩余的 buffer
        if (buffer.trim()) {
          try {
            const msg = JSON.parse(buffer)
            handleStreamMessage(msg, progressContainer, textOutput)
          } catch (parseError) {
            // 忽略
          }
        }

      } catch (error) {
        // 清除超时计时器（如果还存在）
        if (timeoutId) clearTimeout(timeoutId)

        // 处理中止错误
        if (error.name === 'AbortError') {
          // 中止不是错误，输出已由停止按钮处理
        } else if (error.message === 'ConnectionTimeout') {
          // 连接超时 - 服务未启动
          progressContainer.innerHTML = ''
          textOutput.className = 'output-area error'
          textOutput.textContent = '⚠️ 无法连接到沙箱服务（连接超时）\n\n请确保沙箱服务正在运行：\n• 源码模式：npm run server\n• CLI 模式：dmla start\n\n或在设置中检查沙箱地址配置'
        } else {
          progressContainer.innerHTML = ''
          textOutput.className = 'output-area error'
          if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            textOutput.textContent = '⚠️ 无法连接到沙箱服务\n\n请确保沙箱服务正在运行，或在设置中检查沙箱地址配置'
          } else {
            textOutput.textContent = `❌ 错误: ${error.message}`
          }
        }
      } finally {
        // 移除停止按钮（如果还存在）
        const existingStopBtn = toolbar.querySelector('.stop-btn')
        if (existingStopBtn) {
          existingStopBtn.remove()
        }

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