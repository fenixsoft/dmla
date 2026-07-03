# Task 6 报告: 更新 sandbox-config.js 和 client.js

## 修改概要

### 1. sandbox-config.js
- **setSandboxConfig()**：合并配置时新增 `sandboxMode: config.sandboxMode || 'custom'` 字段到 `newConfig`
- **window.__SANDBOX_CONFIG__**：更新时附带 `mode: newConfig.sandboxMode`

### 2. client.js
- **CONNECTION_TIMEOUT**：从 `10000`（10 秒）调整为 `20000`（20 秒），注释说明适应 FC 冷启动

## 验证结果

```bash
npm run build 2>&1 | tail -5
# 输出:
# ✔ Compiling with vite - done in 19.12s
# ✔ Rendering 102 pages - done in 1.83s
# success VuePress build completed in 26.30s!
```

构建成功，共 102 页，未出现任何错误或警告。

## 提交记录

```
b49ab18 Task 6: 更新 sandbox-config.js 和 client.js
 2 files changed, 5 insertions(+), 4 deletions(-)
```
