# 沙箱运行时功能完善 - 技术设计

## Context

### 当前架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据流                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  前端 (VuePress)                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Markdown 解析 → ```python runnable [gpu]               │   │
│  │  RunnableCode.vue → 点击运行 → POST /api/sandbox/run    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │ HTTP                                │
│                           ▼                                     │
│  后端 (Express :3001)                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  routes/sandbox.js → 路由处理                            │   │
│  │  sandbox.js → Docker 容器管理                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │ Docker API                         │
│                           ▼                                     │
│  Docker 容器                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  dmla-sandbox:gpu (GPU 版本)                       │   │
│  │  dmla-sandbox:latest (CPU 版本) ← 缺失!            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 现有问题

| 问题 | 位置 | 影响 |
|------|------|------|
| `checkGPUAvailable()` 返回对象而非字符串 | `sandbox.js:20-28` | GPU 检测失效 |
| `runCommand()` 未获取容器输出 | `sandbox.js:33-51` | 健康检查错误 |
| CPU 镜像不存在 | `sandbox.js:11` | 无 GPU 环境无法使用 |
| 沙箱地址硬编码 | 前端多处 | 无法连接远程沙箱 |
| Dockerfile 包含不需要的依赖 | `Dockerfile.sandbox` | 镜像体积过大 |

## Goals / Non-Goals

**Goals:**
- 修复所有阻碍正常使用的 Bug
- 支持 CPU 环境（无 GPU 机器）
- 允许用户配置远程沙箱服务地址
- 简化 Docker 镜像，仅保留课程必需库
- 添加基本测试保证功能稳定性

**Non-Goals:**
- 不实现代码持久化（如安装额外包）
- 不实现文件上传/下载
- 不实现多用户隔离
- 不实现代码编辑器功能增强

## Decisions

### 1. 沙箱地址配置方案

**决定**: 使用 localStorage 存储配置 + 全局状态管理

**方案对比**:

| 方案 | 优点 | 缺点 |
|------|------|------|
| localStorage | 简单、持久化、无需后端 | 仅客户端生效 |
| VuePress 配置文件 | 构建时可配置 | 需要重新构建 |
| URL 参数 | 灵活 | 每次都需要传递 |

**选择 localStorage 的理由**:
- 用户一次配置，永久生效
- 不需要修改后端
- 实现简单

**实现方式**:
```javascript
// 存储格式
localStorage.setItem('sandbox-config', JSON.stringify({
  endpoint: 'http://192.168.1.100:3001'
}))

// 全局访问
window.__SANDBOX_CONFIG__ = {
  getEndpoint: () => JSON.parse(localStorage.getItem('sandbox-config'))?.endpoint
    || 'http://localhost:3001'
}
```

### 2. CPU/GPU 镜像策略

**决定**: 单一 Dockerfile + 构建参数区分

**Dockerfile 改造**:
```dockerfile
# 构建参数
ARG GPU=true

# 条件安装 CUDA 基础镜像
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 AS gpu-base
FROM python:3.11-slim AS cpu-base

# 根据参数选择基础镜像
FROM ${GPU:-true} == "true" ? gpu-base : cpu-base
```

**构建命令**:
```bash
# GPU 版本
docker build -t dmla-sandbox:gpu . -f Dockerfile.sandbox

# CPU 版本
docker build -t dmla-sandbox:latest . -f Dockerfile.sandbox --build-arg GPU=false
```

### 3. GPU 检测修复方案

**当前代码问题**:
```javascript
// 错误: result 是 Docker 容器返回的对象，不是输出字符串
const result = await runCommand('nvidia/cuda:11.8-base', ['nvidia-smi', '-L'])
return result.includes('GPU')  // ❌ 对象没有 includes 方法
```

**修复方案**:
```javascript
export async function checkGPUAvailable() {
  try {
    // 直接使用 dockerode 的方式运行容器
    const container = await docker.createContainer({
      Image: 'nvidia/cuda:11.8-base',
      Cmd: ['nvidia-smi', '-L'],
      HostConfig: {
        DeviceRequests: [{
          Driver: 'nvidia',
          Count: -1,
          Capabilities: [['gpu']]
        }]
      }
    })

    await container.start()
    await container.wait()

    const logs = await container.logs({ stdout: true, stderr: true })
    const output = parseDockerLogs(logs)

    await container.remove()

    return output.includes('GPU')
  } catch {
    return false
  }
}
```

### 4. 设置界面设计

**UI 组件结构**:
```
Navbar.vue
├── 设置按钮 (齿轮图标)
│   └── @click → 打开设置弹窗
│
Settings.vue (新组件)
├── 设置弹窗 (Modal)
│   ├── 标题: 沙箱设置
│   ├── 输入框: 沙箱服务地址
│   ├── 状态指示: 连接状态
│   └── 按钮: 取消 / 保存设置
│
└── 连接测试逻辑
    └── GET {endpoint}/api/sandbox/health
```

### 5. 测试策略

**测试范围**:

| 测试类型 | 文件 | 内容 |
|---------|------|------|
| API 单元测试 | `sandbox.test.js` | 路由处理、参数验证 |
| 集成测试 | `docker-execution.test.js` | Docker 容器执行流程 |
| 前端测试 | Playwright E2E | 设置界面、代码运行 |

**测试框架**: Jest + Supertest (API) + Playwright (E2E)

## Risks / Trade-offs

### 风险 1: Docker 守护进程不可用
- **风险**: 用户机器未安装或未启动 Docker
- **缓解**: 启动时检测 Docker 状态，显示友好错误提示

### 风险 2: 远程沙箱网络延迟
- **风险**: 配置远程沙箱后，网络延迟影响体验
- **缓解**: 添加请求超时配置，显示加载动画

### 风险 3: 镜像构建时间长
- **风险**: GPU 镜像包含 PyTorch CUDA 版本，体积约 5GB
- **缓解**: 提供预构建镜像下载，文档说明首次构建时间

### 权衡: 简化 Dockerfile vs 完整性
- **选择**: 移除 TensorFlow/JAX，仅保留 PyTorch
- **理由**: 课程内容只使用 PyTorch，减少镜像体积和构建时间
- **代价**: 如果后续需要 TensorFlow，需要重新添加

## Migration Plan

### 部署步骤

1. **更新 Dockerfile**
   ```bash
   # 构建新镜像
   npm run build:sandbox      # GPU 版本
   npm run build:sandbox:cpu  # CPU 版本
   ```

2. **更新后端代码**
   ```bash
   cd local-server && npm install
   ```

3. **更新前端**
   ```bash
   npm run build
   ```

4. **验证**
   - 运行测试: `npm test`
   - 手动验证: 运行一个 runnable 代码块

### 回滚策略

- 后端代码通过 Git 回滚
- Docker 镜像保留旧版本标签 `dmla-sandbox:gpu-v1`
- 前端重新部署上一版本

## Open Questions

1. ~~是否需要支持 TensorFlow?~~ → 已决定不需要
2. 设置界面是否需要支持多套沙箱配置? → 暂不需要，单配置足够
3. 是否需要认证机制保护远程沙箱? → 暂不实现，本地/内网场景