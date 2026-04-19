/**
 * 沙箱 API 路由
 */
import { Router } from 'express'
import sandbox, { runPythonCode, checkImageExists, checkGPUAvailable } from '../sandbox.js'

const { SANDBOX_CONFIG } = sandbox

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
    // 检查镜像是否存在，智能降级
    // GPU镜像包含CPU的全部功能，如果CPU镜像不存在但GPU镜像存在，可以使用GPU镜像执行CPU代码
    let actualUseGpu = useGpu
    let actualImage = null  // 指定使用的镜像
    let imageExists = await checkImageExists(useGpu)

    if (!imageExists && !useGpu) {
      // CPU镜像不存在，检查是否可以用GPU镜像替代
      const gpuImageExists = await checkImageExists(true)
      if (gpuImageExists) {
        imageExists = true
        actualUseGpu = false  // 不启用GPU设备
        actualImage = SANDBOX_CONFIG.imageGpu  // 使用GPU镜像
        console.log('[Sandbox] CPU镜像不存在，使用GPU镜像执行（不启用GPU设备）')
      }
    }

    if (!imageExists) {
      return res.status(503).json({
        success: false,
        error: useGpu
          ? 'GPU 镜像未安装。请运行以下命令安装：\n\nnpm run build:sandbox:gpu\n\n或使用 dmla CLI：\n\ndmla install --gpu'
          : '沙箱镜像未安装。请运行以下命令安装：\n\nnpm run build:sandbox:cpu\n\n或使用 dmla CLI：\n\ndmla install --cpu\n\n注意：如果您已安装 GPU 镜像，它也支持 CPU 执行'
      })
    }

    // 如果请求 GPU，检查 GPU 是否可用
    if (actualUseGpu) {
      const gpuAvailable = await checkGPUAvailable()
      if (!gpuAvailable) {
        return res.status(503).json({
          success: false,
          error: `GPU 硬件不可用。请确保系统安装了 NVIDIA GPU 驱动和 nvidia-container-toolkit。\n\n诊断步骤：\n1. 运行 nvidia-smi 检查 GPU 状态\n2. 运行 docker run --rm --gpus all ${SANDBOX_CONFIG.imageGpu} nvidia-smi 测试 Docker GPU 支持\n\n或使用 dmla doctor 进行环境诊断`
        })
      }
    }

    // 执行代码（使用确定后的镜像）
    const result = await runPythonCode(code, actualUseGpu, actualImage)

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