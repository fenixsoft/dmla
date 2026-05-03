/**
 * Design Machine Learning Applications 本地服务
 * 提供 Python 代码沙箱执行 API
 */
import express from 'express'
import cors from 'cors'
import { fileURLToPath } from 'url'
import { resolve } from 'path'
import sandboxRouter from './routes/sandbox.js'
import { cleanupAllContainers } from './sandbox.js'

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

// 沙箱 API
app.use('/api/sandbox', sandboxRouter)

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

// 捕获进程信号 - 优雅退出，清理所有容器
process.on('SIGTERM', async () => {
  log('Received SIGTERM, cleaning up containers...')
  try {
    await cleanupAllContainers()
    log('All containers cleaned up')
  } catch (e) {
    log(`Cleanup error: ${e.message}`)
  }
  process.exit(0)
})

process.on('SIGINT', async () => {
  log('Received SIGINT (Ctrl+C), cleaning up containers...')
  try {
    await cleanupAllContainers()
    log('All containers cleaned up')
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
} else {
  log('Skipping server start (imported as module)')
}

export default app