# Task 5: 更新 Settings.vue（FC/自定义模式选择）

## 目标

修改 `docs/.vuepress/theme/components/Settings.vue`，为沙箱服务设置增加 FC/自定义双模式选择。

## 文件

- Modify: `docs/.vuepress/theme/components/Settings.vue`

## 前置条件

- Task 4 已完成，FC 公网 URL 已写入 `/root/dmla/.superpowers/sdd/fc-url.txt`
- 读取该文件获取 FC URL 值，替换代码中 `FC_DEFAULT_URL` 的占位符

## 修改内容

### 1. Template 部分：替换"沙箱服务配置 Tab"区域

找到 `<!-- 沙箱服务配置 Tab -->` 注释及其后的整个 tab-content div，替换为新的双模式 UI。

**新 Template：**

```vue
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
```

### 2. Script 部分：新增 FC 相关变量

在 `const testing = ref(false)` 之后添加：

```javascript
// FC 模式相关
// 从 /root/dmla/.superpowers/sdd/fc-url.txt 读取实际 URL 替换此处
const FC_DEFAULT_URL = 'REPLACE_WITH_FC_URL'

// 沙箱模式: 'fc' | 'custom'
const sandboxMode = ref('fc')

// 原 endpoint 变量保留（用于自定义模式）
```

**重要：** 必须先用 `cat /root/dmla/.superpowers/sdd/fc-url.txt` 读取 URL，替换上面代码中的 `REPLACE_WITH_FC_URL`。

修改 `loadConfig()` 函数：

```javascript
function loadConfig() {
  const config = getSiteConfig()
  sandboxMode.value = config.sandboxMode || 'fc'
  endpoint.value = config.sandboxEndpoint || 'http://localhost:3001'

  if (sandboxMode.value === 'fc') {
    endpoint.value = FC_DEFAULT_URL
  }

  selectedTheme.value = config.highlightTheme || DEFAULT_THEME
}
```

在 `loadConfig` 之后新增 `onModeChange`：

```javascript
function onModeChange() {
  if (sandboxMode.value === 'fc') {
    endpoint.value = FC_DEFAULT_URL
  }
  resetStatus()
}
```

修改 `save()` 函数，在 config 对象中增加 sandboxMode：

```javascript
function save() {
  const config = {
    sandboxMode: sandboxMode.value,
    sandboxEndpoint: sandboxMode.value === 'fc' ? FC_DEFAULT_URL : (endpoint.value.trim() || 'http://localhost:3001'),
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
```

修改 `watch(() => props.visible, ...)` 块，确保打开设置页时自动检测连接：

```javascript
watch(() => props.visible, (newVal) => {
  if (newVal) {
    loadConfig()
    testConnection()
  }
})
```

### 3. Style 部分：新增模式选择器样式

在 `</style>` 之前添加：

```css
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
```

## 验证

```bash
cd /root/dmla
npm run build 2>&1 | tail -5
# 预期: 构建成功，无 Vue 模板编译错误
```

## 全局约束

- FC_DEFAULT_URL 从 fc-url.txt 读取，不能硬编码为占位符
- 默认模式为 'fc'
