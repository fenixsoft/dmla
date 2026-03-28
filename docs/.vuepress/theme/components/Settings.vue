<template>
  <Teleport to="body">
    <div v-if="visible" class="settings-overlay" @click.self="close">
      <div class="settings-modal">
        <div class="settings-header">
          <h3>沙箱设置</h3>
          <button class="close-btn" @click="close">&times;</button>
        </div>

        <div class="settings-body">
          <div class="form-group">
            <label for="sandbox-endpoint">沙箱服务地址</label>
            <input
              id="sandbox-endpoint"
              v-model="endpoint"
              type="url"
              placeholder="http://localhost:3001"
              @input="resetStatus"
            />
            <p class="help-text">用于执行教程中的 Python 代码</p>
          </div>

          <div class="connection-status">
            <span class="status-label">连接状态:</span>
            <span class="status-value" :class="statusClass">
              <span class="status-dot"></span>
              {{ statusText }}
            </span>
          </div>
        </div>

        <div class="settings-footer">
          <button class="btn btn-secondary" @click="testConnection">
            {{ testing ? '检测中...' : '测试连接' }}
          </button>
          <button class="btn btn-primary" @click="save">
            保存设置
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['close', 'save'])

// 配置状态
const endpoint = ref('')
const connectionStatus = ref('unknown') // 'connected' | 'disconnected' | 'unknown' | 'testing'
const testing = ref(false)

// 从 localStorage 加载配置
function loadConfig() {
  try {
    const config = JSON.parse(localStorage.getItem('sandbox-config') || '{}')
    endpoint.value = config.endpoint || 'http://localhost:3001'
  } catch {
    endpoint.value = 'http://localhost:3001'
  }
}

// 重置状态
function resetStatus() {
  connectionStatus.value = 'unknown'
}

// 状态计算
const statusClass = computed(() => {
  return {
    'status-connected': connectionStatus.value === 'connected',
    'status-disconnected': connectionStatus.value === 'disconnected',
    'status-testing': connectionStatus.value === 'testing',
    'status-unknown': connectionStatus.value === 'unknown'
  }
})

const statusText = computed(() => {
  switch (connectionStatus.value) {
    case 'connected':
      return '已连接'
    case 'disconnected':
      return '未连接'
    case 'testing':
      return '检测中'
    default:
      return '未检测'
  }
})

// 测试连接
async function testConnection() {
  connectionStatus.value = 'testing'
  testing.value = true

  try {
    const response = await fetch(`${endpoint.value}/api/sandbox/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000) // 5秒超时
    })

    if (response.ok) {
      connectionStatus.value = 'connected'
    } else {
      connectionStatus.value = 'disconnected'
    }
  } catch {
    connectionStatus.value = 'disconnected'
  } finally {
    testing.value = false
  }
}

// 保存配置
function save() {
  const config = {
    endpoint: endpoint.value.trim() || 'http://localhost:3001'
  }

  localStorage.setItem('sandbox-config', JSON.stringify(config))

  // 更新全局配置
  if (typeof window !== 'undefined') {
    window.__SANDBOX_CONFIG__ = config
  }

  emit('save', config)
  close()
}

// 关闭弹窗
function close() {
  emit('close')
}

// 监听 visible 变化，加载配置
watch(() => props.visible, (newVal) => {
  if (newVal) {
    loadConfig()
    // 自动测试连接
    testConnection()
  }
})
</script>

<style scoped>
.settings-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.settings-modal {
  background: #fff;
  border-radius: 12px;
  width: 480px;
  max-width: 90vw;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
}

.settings-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  border-bottom: 1px solid #E4E4E7;
}

.settings-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #18181B;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  color: #71717A;
  cursor: pointer;
  padding: 0;
  line-height: 1;
}

.close-btn:hover {
  color: #18181B;
}

.settings-body {
  padding: 24px;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 500;
  color: #18181B;
}

.form-group input {
  width: 100%;
  padding: 10px 14px;
  border: 1.5px solid #E4E4E7;
  border-radius: 8px;
  font-size: 14px;
  color: #18181B;
  transition: border-color 0.2s;
  box-sizing: border-box;
}

.form-group input:focus {
  outline: none;
  border-color: #2563EB;
}

.help-text {
  margin: 8px 0 0;
  font-size: 12px;
  color: #71717A;
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-label {
  font-size: 14px;
  color: #71717A;
}

.status-value {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 500;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #A1A1AA;
}

.status-connected .status-dot {
  background: #22C55E;
}

.status-connected {
  color: #22C55E;
}

.status-disconnected .status-dot {
  background: #EF4444;
}

.status-disconnected {
  color: #EF4444;
}

.status-testing .status-dot {
  background: #F59E0B;
  animation: pulse 1s infinite;
}

.status-testing {
  color: #F59E0B;
}

.status-unknown {
  color: #A1A1AA;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.settings-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 24px;
  border-top: 1px solid #E4E4E7;
  background: #FAFAFA;
  border-radius: 0 0 12px 12px;
}

.btn {
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  border: none;
}

.btn-primary {
  background: #2563EB;
  color: #fff;
}

.btn-primary:hover {
  background: #1D4ED8;
}

.btn-secondary {
  background: #fff;
  color: #18181B;
  border: 1.5px solid #E4E4E7;
}

.btn-secondary:hover {
  background: #F4F4F5;
}
</style>