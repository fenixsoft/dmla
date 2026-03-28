# 沙箱运行时功能完善

## Why

当前沙箱运行时核心代码已实现，但存在多个阻碍正常使用的 Bug，且缺少 CPU 环境支持和沙箱地址配置功能。用户在无 GPU 环境下无法使用沙箱，且无法将沙箱服务部署到远程 GPU 服务器并通过前端配置连接地址。

## What Changes

### Bug 修复
- **修复 `checkGPUAvailable()` 函数**：当前返回对象而非字符串，导致 GPU 检测逻辑失效
- **修复 `runCommand()` 函数**：未正确获取容器输出，导致健康检查返回错误数据
- **添加 CPU 镜像支持**：当前只有 GPU 版本镜像，缺少 `ideaspaces-sandbox:latest` (CPU 版本)

### 新增功能
- **前端设置界面**：在导航栏添加设置按钮，支持配置沙箱服务地址
- **设置持久化**：使用 localStorage 保存用户配置的沙箱地址
- **连接状态检测**：实时显示沙箱服务连接状态

### 优化
- **简化 Dockerfile**：移除不需要的 TensorFlow 和 JAX，仅保留 PyTorch、NumPy、Matplotlib 等课程必需库
- **构建脚本增强**：支持构建 CPU 和 GPU 两种镜像

### 测试
- **API 端点测试**：验证 `/api/sandbox/run`、`/api/sandbox/health` 等接口
- **Docker 执行测试**：验证容器创建、代码执行、输出获取、容器销毁流程

## Capabilities

### New Capabilities
- `sandbox-settings`: 沙箱设置功能，包括前端设置界面、地址配置、持久化存储、连接状态检测

### Modified Capabilities
- `sandbox-execution`: 修复 Bug 并增强沙箱执行功能，支持 CPU/GPU 镜像选择、正确的 GPU 检测逻辑

## Impact

### 后端影响
- `local-server/src/sandbox.js` - 修复 Bug，添加 CPU 镜像支持
- `local-server/src/routes/sandbox.js` - 无变更
- `local-server/Dockerfile.sandbox` - 简化依赖，新增 CPU 版本构建参数
- `package.json` - 新增 `build:sandbox:cpu` 构建命令

### 前端影响
- `docs/.vuepress/plugins/runnable-code/` - 读取配置的沙箱地址
- `docs/.vuepress/theme/components/Navbar.vue` - 添加设置按钮
- `docs/.vuepress/theme/components/Settings.vue` - 新增设置弹窗组件

### 测试文件（新增）
- `local-server/tests/sandbox.test.js` - API 端点测试
- `local-server/tests/docker-execution.test.js` - Docker 执行测试