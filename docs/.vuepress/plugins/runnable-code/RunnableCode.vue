<template>
  <div class="runnable-code-block" :class="{ 'has-gpu': hasGpu }">
    <!-- 代码区域 -->
    <div class="code-area">
      <pre><code :class="`language-${language}`">{{ code }}</code></pre>
    </div>

    <!-- 工具栏 -->
    <div class="toolbar">
      <button
        class="run-btn"
        :disabled="isRunning || !sandboxAvailable"
        @click="runCode(false)"
      >
        {{ isRunning ? 'Running...' : '▶ Run' }}
      </button>

      <button
        v-if="hasGpu"
        class="run-btn gpu-btn"
        :disabled="isRunning || !sandboxAvailable"
        @click="runCode(true)"
      >
        {{ isRunning ? 'Running...' : '▶ Run on GPU' }}
      </button>

      <span v-if="!sandboxAvailable" class="sandbox-notice">
        (沙箱服务未连接)
      </span>
    </div>

    <!-- 输出区域 -->
    <div
      class="output-area"
      :class="{
        loading: isRunning,
        error: hasError,
        success: !hasError && output
      }"
    >
      <template v-if="isRunning">
        执行中...
      </template>
      <template v-else-if="output">
        {{ output }}
        <div v-if="executionTime" class="execution-time">
          --- 执行时间: {{ executionTime.toFixed(3) }}s ---
        </div>
      </template>
      <template v-else>
        点击 Run 按钮执行代码
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

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
  apiEndpoint: {
    type: String,
    default: 'http://localhost:3001/api/sandbox/run'
  }
})

// 状态
const isRunning = ref(false)
const output = ref('')
const hasError = ref(false)
const executionTime = ref(null)
const sandboxAvailable = ref(true)

// 运行代码
async function runCode(useGpu = false) {
  isRunning.value = true
  output.value = ''
  hasError.value = false
  executionTime.value = null

  try {
    const response = await fetch(props.apiEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code: props.code,
        useGpu
      })
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const result = await response.json()

    output.value = result.output || '(无输出)'
    hasError.value = !result.success

    if (result.error && !result.success) {
      output.value = result.error
    }

    executionTime.value = result.executionTime

  } catch (error) {
    hasError.value = true
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
      output.value = '⚠️ 无法连接到沙箱服务\n\n请确保本地服务正在运行:\nnpm run local'
      sandboxAvailable.value = false
    } else {
      output.value = `❌ 错误: ${error.message}`
    }
  } finally {
    isRunning.value = false
  }
}

// 检查沙箱可用性
onMounted(async () => {
  try {
    const response = await fetch(props.apiEndpoint.replace('/run', '/health'), {
      method: 'GET'
    })
    sandboxAvailable.value = response.ok
  } catch {
    sandboxAvailable.value = false
  }
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

.toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 8px 16px;
  background: var(--c-bg-light, #f3f4f5);
  border-top: 1px solid var(--c-border, #eaecef);
}

.run-btn {
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
  transition: background 0.2s;
}

.run-btn:hover:not(:disabled) {
  background: #4abf8a;
}

.run-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.gpu-btn {
  background: #2563eb;
}

.gpu-btn:hover:not(:disabled) {
  background: #3b82f6;
}

.sandbox-notice {
  font-size: 12px;
  color: #999;
  margin-left: 8px;
}

.output-area {
  padding: 12px 16px;
  background: #1e1e1e;
  color: #d4d4d4;
  font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
  font-size: 13px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 300px;
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

.execution-time {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #333;
  color: #888;
  font-size: 12px;
}
</style>