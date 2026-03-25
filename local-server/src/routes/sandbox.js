/**
 * 沙箱 API 路由
 */
import { Router } from 'express'
import { runPythonCode, checkImageExists, checkGPUAvailable } from '../sandbox.js'

const router = Router()

/**
 * 健康检查
 */
router.get('/health', async (req, res) => {
  try {
    const imageCpuExists = await checkImageExists(false)
    const imageGpuExists = await checkImageExists(true)
    const gpuAvailable = await checkGPUAvailable()

    res.json({
      status: 'ok',
      images: {
        cpu: imageCpuExists,
        gpu: imageGpuExists
      },
      gpu: gpuAvailable
    })
  } catch (error) {
    res.status(500).json({
      status: 'error',
      error: error.message
    })
  }
})

/**
 * 执行代码
 * POST /api/sandbox/run
 * Body: { code: string, useGpu?: boolean }
 */
router.post('/run', async (req, res) => {
  const { code, useGpu = false } = req.body

  // 验证请求
  if (!code || typeof code !== 'string') {
    return res.status(400).json({
      success: false,
      error: 'Missing or invalid code parameter'
    })
  }

  // 代码长度限制 (约 100KB)
  if (code.length > 100000) {
    return res.status(400).json({
      success: false,
      error: 'Code too long (max 100KB)'
    })
  }

  try {
    // 检查镜像是否存在
    const imageExists = await checkImageExists(useGpu)

    if (!imageExists) {
      return res.status(503).json({
        success: false,
        error: `Sandbox image not found. Run: npm run build:sandbox${useGpu ? ' (with GPU support)' : ''}`
      })
    }

    // 执行代码
    const result = await runPythonCode(code, useGpu)

    res.json(result)

  } catch (error) {
    console.error('Sandbox error:', error)
    res.status(500).json({
      success: false,
      error: error.message || 'Internal sandbox error'
    })
  }
})

/**
 * GPU 状态检查
 */
router.get('/gpu', async (req, res) => {
  try {
    const gpuAvailable = await checkGPUAvailable()

    res.json({
      available: gpuAvailable,
      message: gpuAvailable ? 'GPU is available' : 'No GPU detected'
    })
  } catch (error) {
    res.status(500).json({
      available: false,
      error: error.message
    })
  }
})

export default router