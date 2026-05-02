# Timeout Extension

超时扩展能力，支持自定义超时时长和取消超时限制。

## 功能需求

### Markdown 语法扩展

代码块语言标识符支持 timeout 参数：

```markdown
```python runnable timeout=600
# 10 分钟超时（600 秒）
```

```python runnable gpu timeout=unlimited
# 无超时限制，用于长时间训练
```
```

### 参数解析

timeout 参数格式：
- 数字：秒数，如 `timeout=600` 表示 10 分钟
- `unlimited`：取消超时限制

默认值：
- 未指定 timeout：60 秒（现有默认）
- timeout=unlimited：无限制，但必须配合 ProgressReporter 使用

### 前端处理

| timeout 值 | UI 显示 | 进度轮询 |
|:-----------|:--------|:---------|
| <= 60 | 不显示进度条 | 不启用 |
| > 60 | 显示进度条 | 启用 |
| unlimited | 显示进度条 | 启用 |

## 技术实现

### Markdown 解析

VuePress markdown-it 插件解析代码块属性：

```javascript
// 解析 ```python runnable timeout=600
const fenceParser = (tokens, idx, options, env, self) => {
  const token = tokens[idx]
  const lang = token.info.trim()
  
  // 解析参数
  const match = lang.match(/runnable\s+(gpu)?\s*(timeout=(\d+|unlimited))?/)
  
  if (match) {
    const useGpu = match[1] === 'gpu'
    const timeout = match[3] === 'unlimited' ? null : parseInt(match[3]) || 60
    
    // 添加到代码块属性
    token.attrPush(['data-use-gpu', useGpu])
    token.attrPush(['data-timeout', timeout || 'unlimited'])
  }
}
```

### 沙箱 API

sandbox.js 接收 timeout 参数：

```javascript
// API 调用
POST /api/sandbox/run
{
  "code": "...",
  "useGpu": true,
  "timeout": 600  // null 表示 unlimited
}

// 处理逻辑
async function runPythonCode(code, useGpu, timeout) {
  if (timeout === null) {
    // unlimited: 不设置超时，启用进度轮询
    containerConfig.Env.push('DMLA_NO_TIMEOUT=1')
    startProgressPolling(container)
  } else if (timeout > 60) {
    // 长超时: 设置自定义超时，启用进度轮询
    SANDBOX_CONFIG.timeout = timeout * 1000
    startProgressPolling(container)
  }
  // 默认 60 秒：不启用进度轮询
}
```

## 安全约束

- unlimited 超时需要用户在前端确认（显示警告提示）
- maximum timeout: 86400 秒（24 小时）
- 超时后强制终止容器，避免资源泄漏
- 进度轮询失败时，使用心跳检测判断容器状态