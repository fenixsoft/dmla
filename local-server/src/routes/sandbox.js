/**
 * 沙箱 API 路由
 */
import { Router } from 'express'
import Docker from 'dockerode'
import sandbox, { runPythonCode, checkImageExists, checkGPUAvailable, checkCUDACompatibility } from '../sandbox.js'

const { SANDBOX_CONFIG } = sandbox
const docker = new Docker()

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
 * Body: { code: string, useGpu?: boolean, timeout?: number|null }
 */
router.post('/run', async (req, res) => {
  const { code, useGpu = false, timeout = null } = req.body

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

  // 验证 timeout 参数
  if (timeout !== null && timeout !== undefined) {
    if (typeof timeout !== 'number' || timeout < 0 || timeout > 86400) {
      return res.status(400).json({
        success: false,
        error: 'Invalid timeout parameter (must be null or number 0-86400)'
      })
    }
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
    const result = await runPythonCode(code, actualUseGpu, actualImage, timeout)

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

/**
 * 进度查询
 * GET /api/sandbox/progress
 * 用于长时间任务的进度轮询
 */
router.get('/progress', async (req, res) => {
  try {
    // 进度文件在容器内的 /workspace/progress.json
    // 需要通过 docker exec 读取

    // 查找运行中的沙箱容器
    const containers = await docker.listContainers({ filters: { status: ['running'] } })
    const sandboxContainer = containers.find(c =>
      c.Names.some(name => name.includes('dmla')) ||
      c.Image.includes('dmla-sandbox')
    )

    if (!sandboxContainer) {
      return res.json({ status: 'no_task', message: 'No running task' })
    }

    // 在容器内读取进度文件
    const container = docker.getContainer(sandboxContainer.Id)
    const exec = await container.exec({
      Cmd: ['cat', '/workspace/progress.json'],
      AttachStdout: true
    })

    const stream = await exec.start()
    const chunks = []
    stream.on('data', chunk => chunks.push(chunk))

    await new Promise(resolve => stream.on('end', resolve))

    const output = Buffer.concat(chunks).toString()

    // 解析进度 JSON
    try {
      // 去除 Docker exec 的头部信息
      const jsonStart = output.indexOf('{')
      if (jsonStart !== -1) {
        const progressData = JSON.parse(output.substring(jsonStart))
        return res.json(progressData)
      }
    } catch {
      // JSON 解析失败
    }

    return res.json({ status: 'no_progress', message: 'Progress file not found' })

  } catch (error) {
    res.json({ status: 'error', message: error.message })
  }
})

/**
 * CUDA 兼容性检查
 * 返回详细的 CUDA 环境诊断信息
 */
router.get('/cuda-compat', async (req, res) => {
  try {
    const imageGpuExists = await checkImageExists(true)

    if (!imageGpuExists) {
      return res.json({
        status: 'error',
        message: 'GPU 镜像未安装',
        compatible: false,
        suggestion: '请运行 npm run build:sandbox:gpu 或 dmla install --gpu'
      })
    }

    const compatResult = await checkCUDACompatibility()

    res.json({
      status: compatResult.compatible ? 'ok' : 'error',
      compatible: compatResult.compatible,
      details: compatResult.details,
      issues: compatResult.issues,
      message: compatResult.compatible
        ? 'CUDA 环境完全兼容，GPU 加速可用'
        : 'CUDA 环境不兼容，请使用 CPU 模式或重新构建镜像'
    })
  } catch (error) {
    res.status(500).json({
      status: 'error',
      compatible: false,
      error: error.message
    })
  }
})

export default router