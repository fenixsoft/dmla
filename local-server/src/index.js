/**
 * Design Machine Learning Applications 本地服务
 * 提供 Python 代码沙箱执行 API
 */
import express from 'express'
import cors from 'cors'
import { fileURLToPath } from 'url'
import { resolve } from 'path'
import sandboxRouter from './routes/sandbox.js'

export const app = express()
const PORT = process.env.PORT || 3001

// 中间件
app.use(cors())
app.use(express.json())

// 健康检查
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() })
})

// 停止服务（用于 CLI stop 命令）
app.post('/api/shutdown', (req, res) => {
  res.json({ status: 'shutting_down', timestamp: new Date().toISOString() })
  console.log('🛑 收到停止请求，服务即将关闭...')
  // 延迟关闭，确保响应发送完成
  setTimeout(() => {
    process.exit(0)
  }, 100)
})

// 沙箱 API
app.use('/api/sandbox', sandboxRouter)

// 错误处理
app.use((err, req, res, next) => {
  console.error('Error:', err)
  res.status(500).json({
    success: false,
    error: err.message || 'Internal Server Error'
  })
})

// 启动服务器
// 条件1: 直接运行（入口点匹配）
// 条件2: 同步模式（DMLA_SYNC_MODE 环境变量）
const __filename = fileURLToPath(import.meta.url)
const entryPoint = resolve(process.argv[1] || '')
const shouldStart = __filename === entryPoint || process.env.DMLA_SYNC_MODE === 'true'

if (shouldStart) {
  app.listen(PORT, () => {
    console.log(`🚀 DMLA 本地服务已启动`)
    console.log(`   API: http://localhost:${PORT}`)
    console.log(`   健康检查: http://localhost:${PORT}/api/health`)
  })
}

export default app