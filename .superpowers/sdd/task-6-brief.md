# Task 6: 更新 sandbox-config.js 和 client.js

## 目标

修改两个前端文件：
1. `sandbox-config.js` — 支持 `sandboxMode` 字段的持久化
2. `client.js` — 调整连接超时

## 文件

- Modify: `docs/.vuepress/plugins/runnable-code/sandbox-config.js`
- Modify: `docs/.vuepress/plugins/runnable-code/client.js`

## 修改内容

### sandbox-config.js 改动

找到 `setSandboxConfig()` 函数（约第 118 行），将合并部分改为同时保存 `sandboxMode`：

```javascript
export function setSandboxConfig(config) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    const existing = localStorage.getItem(STORAGE_KEY)
    const existingConfig = existing ? JSON.parse(existing) : {}

    const newConfig = {
      ...existingConfig,
      sandboxEndpoint: config.endpoint || DEFAULT_ENDPOINT,
      sandboxMode: config.sandboxMode || 'custom'
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(newConfig))

    window.__SANDBOX_CONFIG__ = { endpoint: newConfig.sandboxEndpoint, mode: newConfig.sandboxMode }
    window.__SITE_CONFIG__ = newConfig
  } catch (error) {
    console.error('[Sandbox Config] 保存配置失败:', error)
  }
}
```

变更要点：
- 在 `newConfig` 中增加 `sandboxMode: config.sandboxMode || 'custom'`
- 在 `window.__SANDBOX_CONFIG__` 中增加 `mode: newConfig.sandboxMode`
- `getSandboxConfig()` 和 `getSandboxEndpoint()` 无需修改（继续从 localStorage 读 sandboxEndpoint）

### client.js 改动

找到 `CONNECTION_TIMEOUT` 常量定义，将其值从 `10000` 改为 `20000`：

```javascript
const CONNECTION_TIMEOUT = 20000  // 从 10000ms 调整为 20000ms，适应 FC 冷启动
```

## 验证

```bash
cd /root/dmla
npm run build 2>&1 | tail -5
# 预期: 构建成功
```

## 全局约束

- sandboxMode 默认值为 'custom'（sandbox-config.js 层面）
- 前端 Settings 默认值为 'fc'（Settings.vue 层面，在 Task 5 中设置）
- CONNECTION_TIMEOUT = 20000ms
