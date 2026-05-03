<template>
  <div class="runnable-code-block" :class="{ 'has-gpu': hasGpu }">
    <!-- 代码区域 -->
    <div class="code-area">
      <!-- 浮动工具栏 - 右上角 -->
      <div class="floating-toolbar">
        <button
          class="run-btn"
          :disabled="isRunning || !sandboxAvailable"
          @click="runCode(false)"
        >
          {{ isRunning ? 'Running...' : 'Run' }}
        </button>

        <button
          v-if="hasGpu"
          class="run-btn gpu-btn"
          :disabled="isRunning || !sandboxAvailable"
          @click="runCode(true)"
        >
          {{ isRunning ? 'Running...' : 'GPU' }}
        </button>

        <!-- 停止按钮 - 仅运行时显示 -->
        <button
          v-if="isRunning"
          class="stop-btn"
          @click="stopExecution"
        >
          Stop
        </button>

        <span v-if="!sandboxAvailable" class="sandbox-notice">
          ⚠️
        </span>
      </div>

      <pre><code :class="`language-${language}`">{{ code }}</code></pre>
    </div>

    <!-- 输出区域 -->
    <div
      class="output-area"
      :class="{
        loading: isRunning,
        error: hasError,
        success: !hasError && outputs.length > 0
      }"
    >
      <template v-if="isRunning">
        <!-- 进度条显示 -->
        <template v-if="showProgress && progress">
          <div class="progress-bar">
            <div class="progress-header">{{ progress.description || '执行中...' }}</div>
            <div class="progress-track">
              <div class="progress-fill" :style="{ width: `${progress.percent || 0}%` }"></div>
            </div>
            <div class="progress-info">
              <span>{{ progress.message || `${progress.current_step || 0}/${progress.total_steps || 0}` }}</span>
              <span v-if="progress.elapsed_seconds">
                {{ formatTime(progress.elapsed_seconds) }}
                <template v-if="progress.estimated_remaining"> / {{ formatTime(progress.estimated_remaining) }}</template>
              </span>
            </div>
          </div>
        </template>
        <template v-else>
          执行中...
        </template>
      </template>
      <template v-else-if="outputs.length > 0">
        <!-- 渲染每个输出项 -->
        <template v-for="(output, idx) in outputs" :key="idx">
          <!-- 文本流输出 -->
          <pre
            v-if="output.type === 'stream'"
            :class="['output-stream', output.name]"
          >{{ output.text }}</pre>

          <!-- 图片输出 -->
          <img
            v-else-if="output.type === 'display_data' && output.data && output.data['image/png']"
            :src="'data:image/png;base64,' + output.data['image/png']"
            class="output-image"
            @click="openImageModal(output.data['image/png'])"
          />

          <!-- HTML 表格输出（pandas DataFrame 等） -->
          <div
            v-else-if="output.type === 'display_data' && output.data && output.data['text/html']"
            class="output-html"
            v-html="output.data['text/html']"
          />

          <!-- JSON 输出 -->
          <pre
            v-else-if="output.type === 'display_data' && output.data && output.data['application/json']"
            class="output-json"
          >{{ JSON.stringify(output.data['application/json'], null, 2) }}</pre>

          <!-- 执行结果 -->
          <template v-else-if="output.type === 'execute_result'">
            <!-- HTML 表格输出（pandas DataFrame 等） -->
            <div
              v-if="output.data && output.data['text/html']"
              class="output-html"
              v-html="output.data['text/html']"
            />
            <!-- 图片输出 -->
            <img
              v-else-if="output.data && output.data['image/png']"
              :src="'data:image/png;base64,' + output.data['image/png']"
              class="output-image"
              @click="openImageModal(output.data['image/png'])"
            />
            <!-- 默认文本输出 -->
            <pre v-else class="output-result">{{ formatExecuteResult(output) }}</pre>
          </template>

          <!-- 错误输出 -->
          <div
            v-else-if="output.type === 'error'"
            class="output-error"
          >
            <div class="error-header">{{ output.ename }}: {{ output.evalue }}</div>
            <pre v-if="output.traceback && output.traceback.length" class="error-traceback">{{ formatTraceback(output.traceback) }}</pre>
          </div>
        </template>

        <!-- 执行时间 -->
        <div v-if="executionTime" class="execution-time">
          --- 执行时间: {{ executionTime.toFixed(3) }}s ---
        </div>
      </template>
      <template v-else>
        点击 Run 按钮执行代码
      </template>
    </div>

    <!-- 图片放大模态框 -->
    <div
      v-if="showImageModal"
      class="image-modal"
      @click="closeImageModal"
      @keydown.esc="closeImageModal"
    >
      <img
        :src="'data:image/png;base64,' + modalImageData"
        class="modal-image"
        @click.stop
      />
      <div class="modal-close">×</div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { getSandboxEndpoint, globalRunningState, startExecution, endExecution, abortCurrentExecution } from './sandbox-config.js'

const props = defineProps({
  code: {
    type: String,
    required: true
  },
  language: {
    type: String,
    default: 'python'
  },
  hasGpu: {
    type: Boolean,
    default: false
  },
  timeout: {
    type: String,
    default: null  // null, 数字字符串, 或 'unlimited'
  },
  apiEndpoint: {
    type: String,
    default: ''  // 留空则使用配置的端点
  }
})

// 当前使用的端点（响应式，会在配置变化时更新）
const currentEndpoint = ref(getSandboxEndpoint())

// 计算超时值（用于 API）
const timeoutValue = computed(() => {
  if (!props.timeout) return null
  if (props.timeout === 'unlimited') return null
  return parseInt(props.timeout, 10)
})

// 是否启用进度显示（timeout > 60 或 unlimited）
const showProgress = computed(() => {
  if (!props.timeout) return false
  if (props.timeout === 'unlimited') return true
  return parseInt(props.timeout, 10) > 60
})

// 获取 API 端点（响应式更新）
const resolvedEndpoint = computed(() => {
  return props.apiEndpoint || currentEndpoint.value + '/api/sandbox/run'
})

// 状态
const isRunning = ref(false)
const outputs = ref([])
const hasError = ref(false)
const executionTime = ref(null)
const sandboxAvailable = ref(true)
const aborted = ref(false)  // 新增：中止标记

// 进度状态
const progress = ref(null)  // { percent, message, status, elapsed_seconds, estimated_remaining }
const progressInterval = ref(null)

// 图片模态框状态
const showImageModal = ref(false)
const modalImageData = ref('')

// 监听配置变化事件
function handleConfigChange(event) {
  if (event.detail && event.detail.sandboxEndpoint) {
    currentEndpoint.value = event.detail.sandboxEndpoint
    // 配置变化后重新检查连接状态
    checkSandboxAvailability()
  }
}

// 运行代码
async function runCode(useGpu = false) {
  // 检查全局运行状态（防止并发执行）
  if (globalRunningState.isRunning) {
    outputs.value = [{
      type: 'error',
      ename: 'ConcurrencyError',
      evalue: '已有任务在运行，请先中止当前任务',
      traceback: ['点击 Stop 按钮中止当前正在运行的任务']
    }]
    hasError.value = true
    return
  }

  isRunning.value = true
  outputs.value = []
  hasError.value = false
  executionTime.value = null
  progress.value = null
  aborted.value = false

  // 创建 AbortController 用于中止 fetch 请求
  const abortController = new AbortController()

  // 启动进度轮询（如果启用）
  if (showProgress.value) {
    startProgressPolling()
  }

  try {
    const response = await fetch(resolvedEndpoint.value, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code: props.code,
        useGpu,
        timeout: timeoutValue.value
      }),
      signal: abortController.signal  // 关联 AbortController
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const result = await response.json()

    // 注册到全局状态（执行成功后不需要，因为已经结束）
    outputs.value = result.outputs || []
    hasError.value = !result.success || outputs.value.some(o => o.type === 'error')
    executionTime.value = result.executionTime

    // 保存 executionId 用于可能的中止（但此时任务已完成）
    // result.executionId 可用于日志记录

  } catch (error) {
    // 处理中止错误
    if (error.name === 'AbortError') {
      aborted.value = true
      // 显示已中止（如果已有输出则保留）
      if (outputs.value.length === 0) {
        outputs.value = [{ type: 'stream', name: 'stdout', text: '已中止' }]
      } else {
        outputs.value.push({ type: 'stream', name: 'stdout', text: '\n--- 已中止 ---' })
      }
      hasError.value = false  // 中止不是错误
    } else {
      hasError.value = true
      outputs.value = [{
        type: 'error',
        ename: 'ConnectionError',
        evalue: error.message.includes('Failed to fetch') || error.message.includes('NetworkError')
          ? '无法连接到沙箱服务'
          : error.message,
        traceback: error.message.includes('Failed to fetch') || error.message.includes('NetworkError')
          ? ['请确保沙箱服务正在运行，或在设置中检查沙箱地址配置']
          : [error.message]
      }]

      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        sandboxAvailable.value = false
      }
    }
  } finally {
    isRunning.value = false
    stopProgressPolling()
    // 清理全局状态（如果已注册）
    if (globalRunningState.abortController === abortController) {
      endExecution()
    }
  }
}

// 停止执行
async function stopExecution() {
  if (!isRunning.value) return

  try {
    // 调用全局中止函数
    await abortCurrentExecution(currentEndpoint.value)
    aborted.value = true
    isRunning.value = false
    stopProgressPolling()

    // 显示已中止（如果已有输出则保留）
    if (outputs.value.length === 0) {
      outputs.value = [{ type: 'stream', name: 'stdout', text: '已中止' }]
    } else {
      outputs.value.push({ type: 'stream', name: 'stdout', text: '\n--- 已中止 ---' })
    }
  } catch (error) {
    outputs.value = [{
      type: 'error',
      ename: 'AbortError',
      evalue: '中止失败',
      traceback: [error.message]
    }]
    hasError.value = true
  }
}

// 进度轮询
function startProgressPolling() {
  progressInterval.value = setInterval(async () => {
    try {
      const response = await fetch(resolvedEndpoint.value.replace('/run', '/progress'))
      if (response.ok) {
        const data = await response.json()
        // 只有 running/starting/complete 状态才更新进度
        // no_progress/no_task/error 状态不更新，避免显示无效进度
        if (data.status === 'running' || data.status === 'starting' || data.status === 'complete') {
          progress.value = data
        }
      }
    } catch {
      // 进度获取失败时忽略
    }
  }, 2000)
}

function stopProgressPolling() {
  if (progressInterval.value) {
    clearInterval(progressInterval.value)
    progressInterval.value = null
  }
}

// 格式化执行结果
function formatExecuteResult(output) {
  if (output.data && output.data['text/plain']) {
    return output.data['text/plain']
  }
  return JSON.stringify(output.data, null, 2)
}

// 格式化 traceback
function formatTraceback(traceback) {
  if (Array.isArray(traceback)) {
    return traceback.join('\n')
  }
  return String(traceback)
}

// 打开图片模态框
function openImageModal(base64Data) {
  modalImageData.value = base64Data
  showImageModal.value = true
  document.addEventListener('keydown', handleEscKey)
}

// 关闭图片模态框
function closeImageModal() {
  showImageModal.value = false
  modalImageData.value = ''
  document.removeEventListener('keydown', handleEscKey)
}

// 处理 ESC 键关闭模态框
function handleEscKey(event) {
  if (event.key === 'Escape') {
    closeImageModal()
  }
}

// 格式化时间（秒转为 mm:ss）
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

// 检查沙箱可用性
async function checkSandboxAvailability() {
  try {
    const response = await fetch(currentEndpoint.value + '/api/health', {
      method: 'GET'
    })
    sandboxAvailable.value = response.ok
  } catch {
    sandboxAvailable.value = false
  }
}

// 注册配置变化事件监听
onMounted(async () => {
  // 初始化端点
  currentEndpoint.value = getSandboxEndpoint()

  // 监听配置变化事件
  window.addEventListener('site-config-changed', handleConfigChange)

  // 检查连接状态
  await checkSandboxAvailability()
})

// 清理事件监听器
onUnmounted(() => {
  window.removeEventListener('keydown', handleEscKey)
  window.removeEventListener('site-config-changed', handleConfigChange)
  stopProgressPolling()
})
</script>

<style scoped>
.runnable-code-block {
  margin: 1rem 0;
  border: 1px solid var(--c-border, #eaecef);
  border-radius: 8px;
  overflow: hidden;
}

.code-area {
  position: relative;
  margin: 0;
}

.code-area pre {
  margin: 0;
  padding: 1rem;
  background: var(--code-bg-color, #282c34);
  overflow-x: auto;
}

.code-area code {
  font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
  font-size: 14px;
  line-height: 1.5;
}

/* 浮动工具栏 - 右上角 */
.floating-toolbar {
  position: absolute;
  top: 8px;
  right: 12px;
  display: flex;
  gap: 6px;
  align-items: center;
  z-index: 10;
}

.run-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 500;
  color: #fff;
  background: #3eaf7c;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
  opacity: 0.9;
}

.run-btn:hover:not(:disabled) {
  background: #4abf8a;
  opacity: 1;
}

.run-btn:disabled {
  background: #666;
  cursor: not-allowed;
  opacity: 0.6;
}

.gpu-btn {
  background: #2563eb;
}

.gpu-btn:hover:not(:disabled) {
  background: #3b82f6;
}

/* 停止按钮 */
.stop-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 500;
  color: #fff;
  background: #dc2626;  /* 红色 */
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.stop-btn:hover {
  background: #ef4444;
}

.sandbox-notice {
  font-size: 12px;
  color: #f48771;
}

.output-area {
  padding: 12px 16px;
  background: #1e1e1e;
  color: #d4d4d4;
  font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
  font-size: 13px;
  line-height: 1.5;
  max-height: 400px;
  overflow-y: auto;
}

.output-area.loading {
  color: #888;
}

.output-area.error {
  color: #f48771;
}

.output-area.success {
  color: #4ec9b0;
}

/* 文本流输出 */
.output-stream {
  margin: 0;
  padding: 0;
  background: transparent;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: inherit;
  font-size: inherit;
}

.output-stream.stderr {
  color: #f48771;
}

/* 图片输出 */
.output-image {
  max-width: 100%;
  border-radius: 8px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  margin: 0.5rem 0;
  cursor: pointer;
  transition: transform 0.2s;
}

.output-image:hover {
  transform: scale(1.02);
}

/* 执行结果 */
.output-result {
  margin: 0;
  padding: 0;
  background: transparent;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: inherit;
  font-size: inherit;
}

/* 错误输出 */
.output-error {
  margin: 0.5rem 0;
  padding: 0.5rem;
  background: rgba(244, 135, 113, 0.1);
  border-left: 3px solid #f48771;
  border-radius: 0 4px 4px 0;
}

.error-header {
  color: #f48771;
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.error-traceback {
  margin: 0;
  padding: 0;
  background: transparent;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: inherit;
  font-size: 12px;
  color: #888;
}

.execution-time {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #333;
  color: #888;
  font-size: 12px;
}

/* 进度条 */
.progress-bar {
  margin-bottom: 1rem;
}

.progress-header {
  color: #4ec9b0;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.progress-track {
  width: 100%;
  height: 8px;
  background: #333;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3eaf7c, #4abf8a);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.progress-info {
  display: flex;
  justify-content: space-between;
  margin-top: 0.5rem;
  color: #888;
  font-size: 12px;
}

/* HTML 表格输出（pandas DataFrame 等） */
.output-html {
  margin: 0.5rem 0;
  overflow-x: auto;
}

.output-html :deep(table) {
  border-collapse: collapse;
  width: 100%;
  font-size: 12px;
}

.output-html :deep(th),
.output-html :deep(td) {
  border: 1px solid #444;
  padding: 6px 12px;
  text-align: left;
}

.output-html :deep(th) {
  background: #2d2d2d;
  font-weight: 600;
  color: #fff;
}

.output-html :deep(td) {
  color: #d4d4d4;
}

.output-html :deep(tr:nth-child(even) td) {
  background: #252525;
}

.output-html :deep(tr:hover td) {
  background: #333;
}

/* JSON 输出 */
.output-json {
  margin: 0;
  padding: 0;
  background: transparent;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: inherit;
  font-size: inherit;
}

/* 图片模态框 */
.image-modal {
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
}

.modal-image {
  max-width: 95vw;
  max-height: 95vh;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 0 40px rgba(0, 0, 0, 0.5);
}

.modal-close {
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
  transition: background 0.2s;
}

.modal-close:hover {
  background: rgba(255, 255, 255, 0.3);
}
</style>