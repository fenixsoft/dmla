<template>
  <ClientOnly>
    <Teleport to="body">
      <div v-if="visible" class="settings-overlay" @click.self="close">
        <div class="settings-modal">
          <div class="settings-header">
            <h3>设置</h3>
            <button class="close-btn" @click="close">&times;</button>
          </div>

          <!-- Tab 页签 -->
          <div class="settings-tabs">
            <button
              class="tab-btn"
              :class="{ active: activeTab === 'sandbox' }"
              @click="activeTab = 'sandbox'"
            >
              沙箱服务
            </button>
            <button
              class="tab-btn"
              :class="{ active: activeTab === 'highlight' }"
              @click="activeTab = 'highlight'"
            >
              代码高亮
            </button>
          </div>

          <div class="settings-body">
            <!-- 沙箱服务配置 Tab -->
            <div v-show="activeTab === 'sandbox'" class="tab-content">
              <div class="form-group">
                <label>服务模式</label>
                <div class="mode-selector">
                  <label class="mode-option" :class="{ active: sandboxMode === 'fc' }">
                    <input
                      type="radio"
                      v-model="sandboxMode"
                      value="fc"
                      @change="onModeChange"
                    />
                    <span class="mode-label">FC（默认）</span>
                    <span class="mode-hint">云端 Serverless，闲置免费</span>
                  </label>
                  <label class="mode-option" :class="{ active: sandboxMode === 'custom' }">
                    <input
                      type="radio"
                      v-model="sandboxMode"
                      value="custom"
                      @change="onModeChange"
                    />
                    <span class="mode-label">自定义地址</span>
                    <span class="mode-hint">自建沙箱服务</span>
                  </label>
                </div>
              </div>

              <div v-if="sandboxMode === 'fc'" class="form-group">
                <label for="sandbox-endpoint">FC 沙箱地址</label>
                <input
                  id="sandbox-endpoint"
                  :value="FC_DEFAULT_URL"
                  type="text"
                  readonly
                  class="readonly-input"
                />
                <p class="help-text">阿里云函数计算提供，无需自行部署</p>
              </div>

              <div v-if="sandboxMode === 'custom'" class="form-group">
                <label for="sandbox-endpoint">服务地址</label>
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

            <!-- 代码高亮风格配置 Tab -->
            <div v-show="activeTab === 'highlight'" class="tab-content">
              <div class="form-group">
                <label for="highlight-theme">选择风格</label>
                <select id="highlight-theme" v-model="selectedTheme">
                  <option
                    v-for="theme in highlightThemes"
                    :key="theme.id"
                    :value="theme.id"
                  >
                    {{ theme.name }}
                  </option>
                </select>
                <p class="help-text">设置全站代码块的显示风格</p>
              </div>

              <!-- 实际页面预览 - 使用 iframe -->
              <div class="preview-section">
                <label class="preview-label">效果预览</label>
                <div class="preview-frame-container">
                  <iframe
                    ref="previewFrame"
                    :src="previewUrl"
                    class="preview-frame"
                    sandbox="allow-same-origin allow-scripts"
                    scrolling="no"
                    @load="onPreviewLoad"
                  ></iframe>
                </div>
              </div>
            </div>
          </div>

          <div class="settings-footer">
            <button v-if="activeTab === 'sandbox'" class="btn btn-secondary" @click="testConnection">
              {{ testing ? '检测中...' : '测试连接' }}
            </button>
            <button class="btn btn-primary" @click="save">
              保存设置
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </ClientOnly>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { ClientOnly } from 'vuepress/client'
import { HIGHLIGHT_THEMES, DEFAULT_THEME } from '../config/highlightThemes.js'
import { getSiteConfig, saveSiteConfig } from '../utils/configMigration.js'
import { FC_DEFAULT_URL } from '../../plugins/runnable-code/sandbox-config.js'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['close', 'save'])

// Tab 状态
const activeTab = ref('sandbox')

// 沙箱配置状态
const endpoint = ref('')
const connectionStatus = ref('unknown')
const testing = ref(false)

// FC 模式相关（FC_DEFAULT_URL 从 sandbox-config.js 导入）
// 沙箱模式: 'fc' | 'custom'
const sandboxMode = ref('fc')
// 分别记忆两种模式的地址，切换时不丢失
const customEndpoint = ref('http://localhost:3001')
const fcEndpoint = ref(FC_DEFAULT_URL)

// 高亮主题配置
const highlightThemes = HIGHLIGHT_THEMES
const selectedTheme = ref(DEFAULT_THEME)
const previewFrame = ref(null)

// 预览页面 URL（带 preview 参数，只显示内容）
const previewUrl = computed(() => {
  if (typeof window === 'undefined') return ''
  const baseUrl = window.location.origin
  return `${baseUrl}/settings-preview.html?preview=true&theme=${selectedTheme.value}`
})

// 从配置加载状态
function loadConfig() {
  const config = getSiteConfig()
  sandboxMode.value = config.sandboxMode || 'fc'
  // 恢复分别保存的地址
  customEndpoint.value = config.customEndpoint || 'http://localhost:3001'
  fcEndpoint.value = config.fcEndpoint || FC_DEFAULT_URL
  // 根据当前模式设置显示地址
  endpoint.value = sandboxMode.value === 'fc' ? fcEndpoint.value : customEndpoint.value

  selectedTheme.value = config.highlightTheme || DEFAULT_THEME
}

function onModeChange() {
  // 切换前保存当前地址，然后切换到新模式对应地址
  if (sandboxMode.value === 'fc') {
    customEndpoint.value = endpoint.value
    endpoint.value = fcEndpoint.value
  } else {
    fcEndpoint.value = endpoint.value
    endpoint.value = customEndpoint.value
  }
  resetStatus()
}

// 重置连接状态
function resetStatus() {
  connectionStatus.value = 'unknown'
}

// 状态计算
const statusClass = computed(() => ({
  'status-connected': connectionStatus.value === 'connected',
  'status-disconnected': connectionStatus.value === 'disconnected',
  'status-testing': connectionStatus.value === 'testing',
  'status-unknown': connectionStatus.value === 'unknown'
}))

const statusText = computed(() => {
  switch (connectionStatus.value) {
    case 'connected': return '已连接'
    case 'disconnected': return '未连接'
    case 'testing': return '检测中'
    default: return '未检测'
  }
})

// iframe 加载完成后的处理
function onPreviewLoad() {
  // iframe 加载完成，主题 CSS 会通过 CodeHighlightLoader 自动应用
}

// 测试连接
async function testConnection() {
  connectionStatus.value = 'testing'
  testing.value = true

  try {
    const response = await fetch(`${endpoint.value}/api/sandbox/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    })
    connectionStatus.value = response.ok ? 'connected' : 'disconnected'
  } catch {
    connectionStatus.value = 'disconnected'
  } finally {
    testing.value = false
  }
}

// 保存配置
function save() {
  const mode = sandboxMode.value
  // 保存前更新当前模式地址
  if (mode === 'fc') {
    fcEndpoint.value = endpoint.value
  } else {
    customEndpoint.value = endpoint.value.trim() || 'http://localhost:3001'
  }

  const config = {
    sandboxMode: mode,
    sandboxEndpoint: mode === 'fc' ? fcEndpoint.value : customEndpoint.value,
    fcEndpoint: fcEndpoint.value,
    customEndpoint: customEndpoint.value,
    highlightTheme: selectedTheme.value
  }

  saveSiteConfig(config)

  if (typeof window !== 'undefined') {
    window.__SITE_CONFIG__ = config
    window.dispatchEvent(new CustomEvent('site-config-changed', { detail: config }))
  }

  emit('save', config)
  close()
}

// 关闭弹窗
function close() {
  emit('close')
}

// 监听主题变化，立即更新 localStorage 用于 iframe 预览
watch(selectedTheme, (newTheme) => {
  if (typeof window !== 'undefined' && props.visible) {
    // 立即更新 localStorage，让 iframe 加载时能读取新主题
    const currentConfig = getSiteConfig()
    currentConfig.highlightTheme = newTheme
    localStorage.setItem('site-config', JSON.stringify(currentConfig))
  }
})

// 监听 visible 变化
watch(() => props.visible, (newVal) => {
  if (newVal) {
    loadConfig()
    testConnection()
  }
})

// 初始化
onMounted(() => {
  loadConfig()
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
  width: 600px;
  max-width: 90vw;
  max-height: 85vh;
  overflow-y: auto;
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

.settings-tabs {
  display: flex;
  border-bottom: 1px solid #E4E4E7;
  padding: 0 24px;
}

.tab-btn {
  padding: 12px 16px;
  border: none;
  background: transparent;
  font-size: 14px;
  font-weight: 500;
  color: #71717A;
  cursor: pointer;
  position: relative;
  transition: color 0.2s;
}

.tab-btn:hover {
  color: #18181B;
}

.tab-btn.active {
  color: #2563EB;
}

.tab-btn.active::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  right: 0;
  height: 2px;
  background: #2563EB;
}

.settings-body {
  padding: 24px;
}

.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 500;
  color: #18181B;
}

.form-group input,
.form-group select {
  width: 100%;
  padding: 10px 14px;
  border: 1.5px solid #E4E4E7;
  border-radius: 8px;
  font-size: 14px;
  color: #18181B;
  transition: border-color 0.2s;
  box-sizing: border-box;
  background: #fff;
}

.form-group input:focus,
.form-group select:focus {
  outline: none;
  border-color: #2563EB;
}

.form-group select {
  cursor: pointer;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2371717A' stroke-width='2'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 12px center;
  padding-right: 36px;
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

.status-connected .status-dot { background: #22C55E; }
.status-connected { color: #22C55E; }
.status-disconnected .status-dot { background: #EF4444; }
.status-disconnected { color: #EF4444; }
.status-testing .status-dot { background: #F59E0B; animation: pulse 1s infinite; }
.status-testing { color: #F59E0B; }
.status-unknown { color: #A1A1AA; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.preview-section {
  margin-top: 16px;
}

.preview-label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  color: #18181B;
}

.preview-frame-container {
  border-radius: 8px;
  background: #FAFAFA;
  overflow: hidden;
}

.preview-frame {
  width: 100%;
  height: 270px;
  border: none;
  display: block;
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

.mode-selector {
  display: flex;
  gap: 12px;
}

.mode-option {
  flex: 1;
  padding: 12px 16px;
  border: 2px solid #E4E4E7;
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
}

.mode-option input[type="radio"] {
  display: none;
}

.mode-option:hover {
  border-color: #A1A1AA;
}

.mode-option.active {
  border-color: #2563EB;
  background: #EFF6FF;
}

.mode-label {
  display: block;
  font-size: 14px;
  font-weight: 600;
  color: #18181B;
  margin-bottom: 4px;
}

.mode-hint {
  display: block;
  font-size: 12px;
  color: #71717A;
}

.readonly-input {
  background: #F4F4F5 !important;
  color: #71717A !important;
  cursor: not-allowed;
}
</style>