import { Router } from 'express'
import chatManager from '../chat-manager.cjs'

const router = Router()

/**
 * 查询对话服务状态
 * GET /api/chat/status
 */
router.get('/status', (req, res) => {
  res.json(chatManager.getStatus())
})

/**
 * 发送对话消息
 * POST /api/chat/send
 * Body: { message: string }
 */
router.post('/send', async (req, res) => {
  const { message } = req.body

  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: '消息不能为空' })
  }

  try {
    const response = await chatManager.send(message)
    res.json({ response })
  } catch (err) {
    const status = err.message === '对话服务未就绪' ? 503 : 500
    res.status(status).json({ error: err.message })
  }
})

export default router
