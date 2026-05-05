/**
 * Native 模式沙箱 API 路由
 *
 * 与 Docker 模式路由的差异：
 * - health: 返回环境检测结果而非镜像状态
 * - progress: 直接读取文件而非 docker exec
 * - abort: kill 进程而非停止容器
 */
import { Router } from 'express'
import fs from 'fs'
import path from 'path'
import os from 'os'
import { getCachedEnvironment, getDataPath, getProgressPath } from '../native_env_check.js'
import { runPythonCodeNative, runPythonCodeStreamingNative, abortExecutionNative } from '../native_executor.js'

const router = Router()

/**
 * 健康检查
 * GET /api/sandbox/health
 * 注意：返回格式兼容 Docker 模式，前端通过 images 字段判断环境可用性
 */
router.get('/health', (req, res) => {
  const envInfo = getCachedEnvironment()

  if (!envInfo) {
    return res.status(503).json({
      status: 'error',
      // 兼容 Docker 模式格式
      images: { cpu: false, gpu: false },
      gpu: false
    })
  }

  // 返回完全兼容 Docker 模式的格式
  res.json({
    status: envInfo.success ? 'ok' : 'error',
    images: {
      cpu: envInfo.success,  // Native 模式下，环境可用等同于 CPU 镜像可用
      gpu: envInfo.gpuCapable  // GPU 能力取决于 CUDA 检测
    },
    gpu: envInfo.gpuCapable
    // 不返回 mode 字段，让前端认为是 Docker 环境
    // Native 模式详细信息可通过其他 API 获取
  })
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

  // 代码长度限制
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
    const envInfo = getCachedEnvironment()

    // GPU 请求检查
    if (useGpu && (!envInfo || !envInfo.success || !envInfo.gpuCapable)) {
      return res.status(503).json({
        success: false,
        error: 'CUDA 不可用。本机未检测到 NVIDIA GPU 或 CUDA 驱动。\n\n' +
               '请使用 CPU 模式执行代码（不传递 useGpu 参数或设置为 false）。\n\n' +
               '环境信息:\n' +
               `  Python: ${envInfo?.pythonVersion || '未知'}\n` +
               `  PyTorch: ${envInfo?.pytorchVersion || '未知'}\n` +
               `  CUDA: 不可用`
      })
    }

    // 执行代码
    const result = await runPythonCodeNative(code, useGpu, timeout)
    res.json(result)

  } catch (error) {
    console.error('[Native Sandbox] Error:', error)
    res.status(500).json({
      success: false,
      error: error.message || 'Internal sandbox error'
    })
  }
})

/**
 * 流式执行代码
 * POST /api/sandbox/stream
 * Body: { code: string, useGpu?: boolean, timeout?: number|null }
 * 响应: JSON Lines 流式输出
 */
router.post('/stream', async (req, res) => {
  const { code, useGpu = false, timeout = null } = req.body

  // 验证请求
  if (!code || typeof code !== 'string') {
    return res.status(400).json({
      success: false,
      error: 'Missing or invalid code parameter'
    })
  }

  // 代码长度限制
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
    const envInfo = getCachedEnvironment()

    // GPU 请求检查
    if (useGpu && (!envInfo || !envInfo.success || !envInfo.gpuCapable)) {
      return res.status(503).json({
        success: false,
        error: 'CUDA 不可用。本机未检测到 NVIDIA GPU 或 CUDA 驱动。'
      })
    }

    // 流式执行
    await runPythonCodeStreamingNative(code, useGpu, res, timeout)

  } catch (error) {
    console.error('[Native Sandbox Stream] Error:', error)
    if (!res.headersSent) {
      res.status(500).json({
        success: false,
        error: error.message || 'Internal sandbox error'
      })
    }
  }
})

/**
 * GPU 状态检查
 * GET /api/sandbox/gpu
 */
router.get('/gpu', (req, res) => {
  const envInfo = getCachedEnvironment()

  res.json({
    available: envInfo?.gpuCapable || false,
    message: envInfo?.gpuCapable
      ? `GPU 可用: ${envInfo.deviceName} (CUDA ${envInfo.cudaVersion})`
      : 'GPU 不可用 (CPU-only 模式)',
    details: envInfo?.gpuCapable ? {
      cudaVersion: envInfo.cudaVersion,
      deviceName: envInfo.deviceName
    } : null
  })
})

/**
 * 进度查询
 * GET /api/sandbox/progress
 * 用于长时间任务的进度轮询
 */
router.get('/progress', (req, res) => {
  try {
    const progressPath = getProgressPath()

    if (!fs.existsSync(progressPath)) {
      return res.json({ status: 'no_task', message: 'No progress file found' })
    }

    const content = fs.readFileSync(progressPath, 'utf8')
    const progressData = JSON.parse(content)

    res.json(progressData)

  } catch (error) {
    res.json({ status: 'error', message: error.message })
  }
})

/**
 * CUDA 兼容性检查
 * GET /api/sandbox/cuda-compat
 * Native 模式下等同于 gpu 检查
 */
router.get('/cuda-compat', (req, res) => {
  const envInfo = getCachedEnvironment()

  res.json({
    status: envInfo?.gpuCapable ? 'ok' : 'error',
    compatible: envInfo?.gpuCapable || false,
    details: envInfo?.gpuCapable ? {
      pytorch_version: envInfo.pytorchVersion,
      cuda_version: envInfo.cudaVersion,
      device_name: envInfo.deviceName
    } : null,
    issues: envInfo?.gpuCapable ? [] : ['CUDA 不可用'],
    message: envInfo?.gpuCapable
      ? 'CUDA 环境完全兼容，GPU 加速可用'
      : 'CUDA 不可用，请使用 CPU 模式'
  })
})

/**
 * 中止执行
 * POST /api/sandbox/abort
 * Body: { executionId?: string }
 */
router.post('/abort', async (req, res) => {
  const { executionId = null } = req.body

  try {
    const result = await abortExecutionNative(executionId)

    res.json({
      success: result.success,
      stopped: result.stopped,
      message: result.stopped > 0
        ? `已中止 ${result.stopped} 个执行任务`
        : '没有正在运行的任务'
    })
  } catch (error) {
    console.error('[Native Sandbox] Abort error:', error)
    res.status(500).json({
      success: false,
      error: error.message || '中止失败'
    })
  }
})

export default router