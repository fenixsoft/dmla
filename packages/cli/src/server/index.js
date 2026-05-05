/**
 * Design Machine Learning Applications 本地服务
 * 提供 Python 代码沙箱执行 API
 *
 * 支持两种模式：
 * - Docker 模式（默认）：通过 Docker 容器执行代码
 * - Native 模式（DMLA_MODE=native）：直接在本机执行代码
 */
import express from 'express'
import cors from 'cors'
import { fileURLToPath } from 'url'
import { resolve } from 'path'
import sandboxRouter from './routes/sandbox.js'
import nativeRouter from './routes/native.js'
import { cleanupAllContainers } from './sandbox.js'
import { cleanupAllProcesses } from './native_executor.js'
import { checkNativeEnvironment } from './native_env_check.js'

// 检测运行模式
const isNativeMode = process.env.DMLA_MODE === 'native'

export const app = express()
const PORT = process.env.PORT || 3001

// 日志函数
function log(message) {
  const timestamp = new Date().toISOString()
  console.log(`[${timestamp}] ${message}`)
}

// 启动日志
log('Server initializing...')
log(`PORT: ${PORT}`)
log(`NODE_VERSION: ${process.version}`)
log(`PLATFORM: ${process.platform}`)
log(`DMLA_MODE: ${isNativeMode ? 'native' : 'docker'}`)
log(`DMLA_SYNC_MODE: ${process.env.DMLA_SYNC_MODE || 'false'}`)

// 中间件
app.use(cors())
app.use(express.json())

// 请求日志中间件
app.use((req, res, next) => {
  log(`Request: ${req.method} ${req.path}`)
  next()
})

// 健康检查
app.get('/api/health', (req, res) => {
  log('Health check request')
  res.json({ status: 'ok', timestamp: new Date().toISOString() })
})

// 停止服务（用于 CLI stop 命令）
app.post('/api/shutdown', (req, res) => {
  log('Shutdown request received')
  res.json({ status: 'shutting_down', timestamp: new Date().toISOString() })
  // 延迟关闭，确保响应发送完成
  setTimeout(() => {
    log('Exiting due to shutdown request')
    process.exit(0)
  }, 100)
})

// 沙箱 API（根据模式选择路由）
if (isNativeMode) {
  log('Using Native sandbox router')
  app.use('/api/sandbox', nativeRouter)
} else {
  log('Using Docker sandbox router')
  app.use('/api/sandbox', sandboxRouter)
}

// 错误处理
app.use((err, req, res, next) => {
  log(`Error: ${err.message}`)
  log(`Stack: ${err.stack}`)
  res.status(500).json({
    success: false,
    error: err.message || 'Internal Server Error'
  })
})

// 捕获未处理的异常
process.on('uncaughtException', (err) => {
  log(`UNCAUGHT EXCEPTION: ${err.message}`)
  log(`Stack: ${err.stack}`)
  process.exit(1)
})

// 捕获未处理的 Promise 拒绝
process.on('unhandledRejection', (reason, promise) => {
  log(`UNHANDLED REJECTION: ${reason}`)
})

// 捕获进程信号 - 优雅退出
process.on('SIGTERM', async () => {
  log('Received SIGTERM, cleaning up...')
  try {
    if (isNativeMode) {
      await cleanupAllProcesses()
      log('All processes cleaned up')
    } else {
      await cleanupAllContainers()
      log('All containers cleaned up')
    }
  } catch (e) {
    log(`Cleanup error: ${e.message}`)
  }
  process.exit(0)
})

process.on('SIGINT', async () => {
  log('Received SIGINT (Ctrl+C), cleaning up...')
  try {
    if (isNativeMode) {
      await cleanupAllProcesses()
      log('All processes cleaned up')
    } else {
      await cleanupAllContainers()
      log('All containers cleaned up')
    }
  } catch (e) {
    log(`Cleanup error: ${e.message}`)
  }
  process.exit(0)
})

// 启动服务器
// 条件1: 直接运行（入口点匹配）
// 条件2: 同步模式（DMLA_SYNC_MODE 环境变量）
const __filename = fileURLToPath(import.meta.url)
const entryPoint = resolve(process.argv[1] || '')
const shouldStart = __filename === entryPoint || process.env.DMLA_SYNC_MODE === 'true'

log(`Entry point check: __filename=${__filename}, entryPoint=${entryPoint}, match=${__filename === entryPoint}`)

if (shouldStart) {
  // Native 模式：等待环境检测完成后再启动服务
  if (isNativeMode) {
    checkNativeEnvironment().then(envResult => {
      if (!envResult.success) {
        log('Environment check failed!')
        console.log(envResult.error)
        process.exit(1)
      }
      log('Environment check passed')
      log(`Python: ${envResult.pythonVersion}`)
      log(`PyTorch: ${envResult.pytorchVersion}`)
      log(`GPU: ${envResult.gpuCapable ? 'available' : 'not available (CPU-only)'}`)

      // 环境检测通过，启动服务
      startServer()
    }).catch(err => {
      log(`Environment check error: ${err.message}`)
      process.exit(1)
    })
  } else {
    // Docker 模式：直接启动服务
    startServer()
  }
} else {
  log('Skipping server start (imported as module)')
}

// 启动服务函数
function startServer() {
  const server = app.listen(PORT, () => {
    log('Server started successfully')
    log(`API: http://localhost:${PORT}`)
    log(`Health: http://localhost:${PORT}/api/health`)
  })

  server.on('error', (err) => {
    log(`Server error: ${err.message}`)
  })

  server.on('close', () => {
    log('Server closed')
  })
}

export default app