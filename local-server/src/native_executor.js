/**
 * Native 模式执行器
 *
 * 通过子进程直接在本机执行 Python 代码，无需 Docker
 */
import { spawn } from 'child_process'
import path from 'path'
import { fileURLToPath } from 'url'
import os from 'os'
import fs from 'fs'
import chalk from 'chalk'
import { getCachedEnvironment, getKernelRunnerPath, getSharedModulesPath, getDataPath, getProgressPath } from './native_env_check.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// 日志函数
function log(message) {
  const timestamp = new Date().toISOString()
  console.log(`[${timestamp}] [NativeSandbox] ${message}`)
}

// 默认超时（秒）
const DEFAULT_TIMEOUT = 60

// ==================== 全局进程追踪 ====================
const activeProcesses = new Map()

/**
 * 生成唯一的执行 ID
 */
function generateExecutionId() {
  return `native-exec-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
}

/**
 * 注册活跃进程
 */
function registerProcess(executionId, proc) {
  activeProcesses.set(executionId, proc)
  log(`Process registered: ${executionId} (PID: ${proc.pid})`)
}

/**
 * 移除进程记录
 */
function unregisterProcess(executionId) {
  activeProcesses.delete(executionId)
  log(`Process unregistered: ${executionId}`)
}

/**
 * 中止指定执行或所有执行
 * @param {string|null} executionId - 执行 ID，null 表示中止所有
 * @returns {Promise<{success: boolean, stopped: number}>}
 */
export async function abortExecutionNative(executionId = null) {
  const processesToStop = executionId
    ? [activeProcesses.get(executionId)].filter(p => p)
    : Array.from(activeProcesses.values())

  log(`Aborting ${processesToStop.length} processes (executionId: ${executionId || 'all'})`)

  let stopped = 0
  for (const proc of processesToStop) {
    if (proc) {
      try {
        proc.kill('SIGTERM')
        const id = activeProcesses.entries().find(([k, v]) => v === proc)?.[0]
        if (id) activeProcesses.delete(id)
        stopped++
        log(`Process aborted: ${id}`)
      } catch (e) {
        log(`Failed to abort process: ${e.message}`)
      }
    }
  }

  return { success: true, stopped }
}

/**
 * 清理所有活跃进程
 */
export async function cleanupAllProcesses() {
  return abortExecutionNative(null)
}

/**
 * 执行 Python 代码（Native 模式）
 *
 * @param {string} code - Python 代码
 * @param {boolean} useGpu - 是否请求 GPU
 * @param {number|null} timeoutOverride - 超时时间（秒），null 表示 unlimited
 * @returns {Promise<{success: boolean, outputs: Array, executionTime: number, gpuUsed: boolean, executionId: string}>}
 */
export async function runPythonCodeNative(code, useGpu = false, timeoutOverride = null) {
  const startTime = Date.now()
  const executionId = generateExecutionId()

  // 获取环境检测结果
  const envInfo = getCachedEnvironment()
  if (!envInfo || !envInfo.success) {
    log('Environment check result not available')
    return {
      success: false,
      outputs: [{
        type: 'error',
        ename: 'EnvironmentError',
        evalue: '环境检测未完成或失败',
        traceback: ['请重新启动服务']
      }],
      executionTime: (Date.now() - startTime) / 1000,
      gpuUsed: false,
      executionId
    }
  }

  // GPU 请求检查
  if (useGpu && !envInfo.gpuCapable) {
    log(`GPU requested but not available (CPU-only mode)`)
    return {
      success: false,
      outputs: [{
        type: 'error',
        ename: 'GPUNotAvailable',
        evalue: 'CUDA 不可用',
        traceback: [
          '本机未检测到 NVIDIA GPU 或 CUDA 驱动。',
          '请使用 CPU 模式执行代码（不传递 useGpu 参数）。',
          '',
          '环境信息:',
          `  Python: ${envInfo.pythonVersion}`,
          `  PyTorch: ${envInfo.pytorchVersion}`,
          `  CUDA: 不可用`
        ]
      }],
      executionTime: (Date.now() - startTime) / 1000,
      gpuUsed: false,
      executionId
    }
  }

  // 计算超时时间
  const timeoutSeconds = timeoutOverride === null ? 86400 : (timeoutOverride || DEFAULT_TIMEOUT)
  log(`Executing code (length=${code.length}, timeout=${timeoutSeconds}s, useGpu=${useGpu})`)

  // 构建环境变量
  const env = {
    ...process.env,
    PYTHONPATH: getSharedModulesPath(),
    PYTHONUNBUFFERED: '1',
    DMLA_DATA_PATH: getDataPath(),
    DMLA_PROGRESS_PATH: getProgressPath()
  }

  // 确保数据目录存在
  const dataPath = getDataPath()
  if (!fs.existsSync(dataPath)) {
    log(`Creating data directory: ${dataPath}`)
    fs.mkdirSync(dataPath, { recursive: true })
  }

  // 确保进度文件目录存在
  const progressDir = path.dirname(getProgressPath())
  if (!fs.existsSync(progressDir)) {
    fs.mkdirSync(progressDir, { recursive: true })
  }

  const kernelRunnerPath = getKernelRunnerPath()
  log(`Kernel runner: ${kernelRunnerPath}`)
  log(`Shared modules: ${getSharedModulesPath()}`)
  log(`Data path: ${dataPath}`)

  let proc = null
  let timeoutId = null

  try {
    // 创建子进程
    proc = spawn('python3', [
      kernelRunnerPath,
      '--code', code,
      '--timeout', String(timeoutSeconds)
    ], { env })

    registerProcess(executionId, proc)

    // 设置超时
    const timeoutPromise = new Promise((_, reject) => {
      timeoutId = setTimeout(() => {
        log('Execution timeout triggered')
        if (proc) proc.kill('SIGKILL')
        reject(new Error(`Execution timeout after ${timeoutSeconds} seconds`))
      }, timeoutSeconds * 1000 + 5000)
    })

    // 收集输出
    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', (data) => {
      stdout += data.toString()
    })

    proc.stderr.on('data', (data) => {
      stderr += data.toString()
      // stderr 可能包含进度信息，实时输出
      log(`stderr: ${data.toString().substring(0, 100)}...`)
    })

    // 等待进程完成
    const execPromise = new Promise((resolve, reject) => {
      proc.on('close', (code) => {
        log(`Process exited with code ${code}`)
        if (timeoutId) clearTimeout(timeoutId)
        resolve({ stdout, stderr, exitCode: code })
      })

      proc.on('error', (err) => {
        log(`Process error: ${err.message}`)
        if (timeoutId) clearTimeout(timeoutId)
        reject(err)
      })
    })

    // 竞速：超时 vs 正常完成
    const result = await Promise.race([execPromise, timeoutPromise])

    const executionTime = (Date.now() - startTime) / 1000

    // 解析输出（与 Docker 模式格式一致）
    // stdout 可能包含日志 + JSON，需要提取 JSON 部分
    const jsonStart = stdout.indexOf('{')
    if (jsonStart === -1) {
      log('No JSON found in stdout')
      return {
        success: false,
        outputs: [{
          type: 'error',
          ename: 'OutputParseError',
          evalue: '输出解析失败',
          traceback: [stdout.substring(0, 1000) || stderr.substring(0, 1000)]
        }],
        executionTime,
        gpuUsed: useGpu,
        executionId
      }
    }

    const rawOutput = stdout.substring(jsonStart)
    log(`JSON extracted from position ${jsonStart}, length=${rawOutput.length}`)

    try {
      const parsedResult = JSON.parse(rawOutput)
      log('Output parsed successfully')

      return {
        success: parsedResult.success,
        outputs: parsedResult.outputs || [],
        executionTime: parsedResult.executionTime || executionTime,
        gpuUsed: useGpu,
        executionId
      }
    } catch (parseError) {
      log(`JSON parse error: ${parseError.message}`)
      return {
        success: false,
        outputs: [{
          type: 'error',
          ename: 'OutputParseError',
          evalue: 'JSON 解析失败',
          traceback: [rawOutput.substring(0, 500)]
        }],
        executionTime,
        gpuUsed: useGpu,
        executionId
      }
    }

  } catch (error) {
    log(`Execution error: ${error.message}`)
    if (timeoutId) clearTimeout(timeoutId)

    const executionTime = (Date.now() - startTime) / 1000

    return {
      success: false,
      outputs: [{
        type: 'error',
        ename: error.name || 'ExecutionError',
        evalue: error.message || 'Unknown error',
        traceback: [error.message || 'Unknown error']
      }],
      executionTime,
      gpuUsed: useGpu,
      executionId
    }

  } finally {
    unregisterProcess(executionId)
    if (proc) {
      try {
        // 确保进程已终止
        if (proc.pid) {
          try {
            process.kill(proc.pid, 0)  // 检查进程是否存活
          } catch {
            // 进程已不存在
          }
        }
      } catch {
        // 忽略清理错误
      }
    }
  }
}

/**
 * 流式执行 Python 代码
 *
 * @param {string} code - Python 代码
 * @param {boolean} useGpu - 是否请求 GPU
 * @param {object} res - Express 响应对象
 * @param {number|null} timeoutOverride - 超时时间（秒）
 */
export async function runPythonCodeStreamingNative(code, useGpu = false, res, timeoutOverride = null) {
  const startTime = Date.now()
  const executionId = generateExecutionId()

  // 获取环境检测结果
  const envInfo = getCachedEnvironment()

  // GPU 请求检查
  if (useGpu && (!envInfo || !envInfo.success || !envInfo.gpuCapable)) {
    res.write(JSON.stringify({
      type: 'error',
      ename: 'GPUNotAvailable',
      evalue: 'CUDA 不可用',
      traceback: ['本机未检测到 NVIDIA GPU 或 CUDA 驱动']
    }) + '\n')
    res.write(JSON.stringify({
      type: 'result',
      success: false,
      executionTime: 0
    }) + '\n')
    res.end()
    return
  }

  // 设置流式响应头
  res.setHeader('Content-Type', 'application/x-ndjson')
  res.setHeader('Transfer-Encoding', 'chunked')
  res.setHeader('X-Accel-Buffering', 'no')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')

  const timeoutSeconds = timeoutOverride === null ? 86400 : (timeoutOverride || DEFAULT_TIMEOUT)

  // 构建环境变量
  const env = {
    ...process.env,
    PYTHONPATH: getSharedModulesPath(),
    PYTHONUNBUFFERED: '1',
    DMLA_DATA_PATH: getDataPath(),
    DMLA_PROGRESS_PATH: getProgressPath()
  }

  const kernelRunnerPath = getKernelRunnerPath()

  // 输出启动状态
  res.write(JSON.stringify({
    type: 'status',
    status: 'starting',
    message: '正在启动 Python 进程...',
    executionId
  }) + '\n')

  let proc = null

  try {
    proc = spawn('python3', [
      kernelRunnerPath,
      '--code', code,
      '--timeout', String(timeoutSeconds),
      '--stream'
    ], { env })

    registerProcess(executionId, proc)

    // 输出运行状态
    res.write(JSON.stringify({
      type: 'status',
      status: 'running',
      message: '代码执行中...'
    }) + '\n')

    // 实时转发 stdout
    proc.stdout.on('data', (data) => {
      const lines = data.toString().split('\n').filter(l => l.trim())
      for (const line of lines) {
        // kernel_runner.py 输出的已经是 JSON 格式，直接转发
        if (line.trim().startsWith('{')) {
          res.write(line + '\n')
        } else {
          // 非 JSON 内容包装为 stream 消息
          res.write(JSON.stringify({
            type: 'stream',
            name: 'stdout',
            text: line
          }) + '\n')
        }
      }
    })

    // 实时转发 stderr（进度信息）
    proc.stderr.on('data', (data) => {
      const lines = data.toString().split('\n').filter(l => l.trim())
      for (const line of lines) {
        // 进度消息是 JSON 格式
        if (line.trim().startsWith('{') && line.includes('"type":')) {
          res.write(line + '\n')
        } else {
          // 其他 stderr 内容
          res.write(JSON.stringify({
            type: 'stream',
            name: 'stderr',
            text: line
          }) + '\n')
        }
      }
    })

    proc.on('error', (err) => {
      res.write(JSON.stringify({
        type: 'error',
        ename: 'ProcessError',
        evalue: err.message,
        traceback: [err.message]
      }) + '\n')
    })

    // 等待进程完成
    await new Promise((resolve) => {
      proc.on('close', resolve)
    })

    log('Streaming execution finished')

  } catch (error) {
    log(`Streaming execution error: ${error.message}`)
    res.write(JSON.stringify({
      type: 'error',
      ename: error.name || 'ExecutionError',
      evalue: error.message,
      traceback: [error.message]
    }) + '\n')
  } finally {
    unregisterProcess(executionId)
    res.end()
    log('Streaming response ended')
  }
}

export default {
  runPythonCodeNative,
  runPythonCodeStreamingNative,
  abortExecutionNative,
  cleanupAllProcesses
}