/**
 * 沙箱管理模块
 * 负责创建和管理 Docker 容器执行 Python 代码
 */
import Docker from 'dockerode'
import path from 'path'
import { fileURLToPath } from 'url'
import fs from 'fs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// 日志函数
function log(message) {
  const timestamp = new Date().toISOString()
  console.log(`[${timestamp}] [Sandbox] ${message}`)
}

// 启动时记录
log('Sandbox module initialized')

// 检测运行模式并计算正确的路径
// 开发模式: 从 local-server/src 运行，项目根目录在上两级
// 独立模式: 从 packages/cli/src/server 运行，无 shared_modules 目录
function detectProjectRoot() {
  // 尝试向上两级查找 local-server 目录（开发模式）
  const candidateRoot = path.resolve(__dirname, '..', '..')
  const localServerPath = path.join(candidateRoot, 'local-server')
  if (fs.existsSync(localServerPath)) {
    return candidateRoot
  }

  // 尝试向上三级查找（独立模式下的项目根目录）
  const standaloneRoot = path.resolve(__dirname, '..', '..', '..')
  const standaloneLocalServer = path.join(standaloneRoot, 'local-server')
  if (fs.existsSync(standaloneLocalServer)) {
    return standaloneRoot
  }

  // 独立安装模式，无项目根目录
  return null
}

const PROJECT_ROOT = detectProjectRoot()

// 共享模块目录（仅开发模式可用）
const DEFAULT_SHARED_MODULES_PATH = PROJECT_ROOT
  ? path.join(PROJECT_ROOT, 'local-server', 'shared_modules')
  : null

// kernel_runner.py 路径（开发模式下可用）
const DEFAULT_KERNEL_RUNNER_PATH = PROJECT_ROOT
  ? path.join(PROJECT_ROOT, 'local-server', 'src', 'kernel_runner.py')
  : null

const docker = new Docker()

// 沙箱配置
const SANDBOX_CONFIG = {
  imageCpu: 'dmla-sandbox:cpu',
  imageGpu: 'dmla-sandbox:gpu',
  timeout: 60000,           // 60 秒超时
  memory: 4 * 1024 * 1024 * 1024  // 4GB 内存
}

/**
 * 获取共享模块路径
 */
function getSharedModulesPath() {
  // 优先使用环境变量指定的路径
  if (process.env.SHARED_MODULES_PATH) {
    return process.env.SHARED_MODULES_PATH
  }
  // 开发模式下的默认路径
  return DEFAULT_SHARED_MODULES_PATH
}

/**
 * 获取 kernel_runner.py 路径
 */
function getKernelRunnerPath() {
  // 优先使用环境变量指定的路径
  if (process.env.KERNEL_RUNNER_PATH) {
    return process.env.KERNEL_RUNNER_PATH
  }
  // 开发模式下的默认路径
  return DEFAULT_KERNEL_RUNNER_PATH
}

/**
 * 检查是否启用 Volume Mount
 */
function shouldMountSharedModules() {
  return process.env.MOUNT_SHARED_MODULES !== 'false'
}

/**
 * 检查是否挂载本地 kernel_runner.py（开发模式）
 */
function shouldMountKernelRunner() {
  return process.env.MOUNT_KERNEL_RUNNER !== 'false' && PROJECT_ROOT !== null
}

/**
 * 检查 GPU 是否可用
 * 使用已安装的 GPU 镜像运行 nvidia-smi 命令检测 GPU 状态
 */
export async function checkGPUAvailable() {
  let container = null

  try {
    // 使用已配置的 GPU 镜像检测，而非硬编码的 nvidia/cuda 镜像
    container = await docker.createContainer({
      Image: SANDBOX_CONFIG.imageGpu,
      Cmd: ['nvidia-smi', '-L'],
      HostConfig: {
        DeviceRequests: [{
          Driver: 'nvidia',
          Count: -1,  // 使用所有 GPU
          Capabilities: [['gpu']]
        }]
      }
    })

    // 启动容器
    await container.start()

    // 等待执行完成
    await container.wait()

    // 获取输出日志
    const logs = await container.logs({
      stdout: true,
      stderr: true
    })

    // 解析输出
    const output = parseDockerLogs(logs)

    // 检查输出是否包含 GPU 信息
    return output.includes('GPU')
  } catch {
    // GPU 不可用或 Docker/nvidia-smi 执行失败
    return false
  } finally {
    // 清理容器
    if (container) {
      try {
        await container.remove({ force: true })
      } catch {
        // 忽略清理错误
      }
    }
  }
}

/**
 * 执行 Python 代码
 * 使用 IPython Kernel 执行代码，支持富输出（图片、文本、错误等）
 * @param {string} code - Python 代码
 * @param {boolean} useGpu - 是否使用 GPU
 * @returns {Promise<{success: boolean, outputs: Array, executionTime: number, gpuUsed: boolean}>}
 */
export async function runPythonCode(code, useGpu = false) {
  const startTime = Date.now()

  log(`runPythonCode called, useGpu=${useGpu}, code length=${code.length}`)

  // 选择镜像
  const image = useGpu ? SANDBOX_CONFIG.imageGpu : SANDBOX_CONFIG.imageCpu
  log(`Using image: ${image}`)

  // 创建容器配置 - 使用 kernel_runner.py 执行代码
  const containerConfig = {
    Image: image,
    Cmd: ['python3', '/workspace/kernel_runner.py', '--code', code, '--timeout', String(Math.floor(SANDBOX_CONFIG.timeout / 1000))],
    HostConfig: {
      Memory: SANDBOX_CONFIG.memory,
      AutoRemove: false  // 手动移除以获取日志
    },
    Env: [
      'PYTHONUNBUFFERED=1'
      // matplotlib 使用 IPython Kernel 的 inline 后端，自动发送 display_data
    ]
  }

  log('Container config created')

  // Volume Mount 配置
  const useMount = shouldMountSharedModules()
  const sharedModulesPath = getSharedModulesPath()
  const mountKernelRunner = shouldMountKernelRunner()
  const kernelRunnerPath = getKernelRunnerPath()

  // 收集所有需要挂载的路径
  const binds = []

  // 挂载共享模块
  if (useMount && sharedModulesPath && fs.existsSync(sharedModulesPath)) {
    binds.push(`${sharedModulesPath}:/usr/local/lib/python3.11/site-packages/shared:ro`)
    console.log(`[Sandbox] 共享模块 Volume Mount: ${sharedModulesPath}`)
  } else if (useMount && sharedModulesPath) {
    console.warn(`[Sandbox] 警告: 共享模块目录不存在: ${sharedModulesPath}`)
  }

  // 挂载 kernel_runner.py（开发模式调试）
  if (mountKernelRunner && kernelRunnerPath && fs.existsSync(kernelRunnerPath)) {
    binds.push(`${kernelRunnerPath}:/workspace/kernel_runner.py:ro`)
    console.log(`[Sandbox] kernel_runner.py Volume Mount: ${kernelRunnerPath}`)
  } else if (mountKernelRunner && kernelRunnerPath) {
    console.warn(`[Sandbox] 警告: kernel_runner.py 不存在: ${kernelRunnerPath}`)
  }

  // 设置 Binds
  if (binds.length > 0) {
    containerConfig.HostConfig.Binds = binds
  }

  if (!PROJECT_ROOT) {
    console.log('[Sandbox] 独立安装模式，无 Volume Mount')
  }

  // GPU 配置
  if (useGpu) {
    containerConfig.HostConfig.DeviceRequests = [{
      Driver: 'nvidia',
      Count: -1,  // 使用所有 GPU
      Capabilities: [['gpu']]
    }]
  }

  let container = null
  let timeoutId = null

  try {
    // 创建容器
    log('Creating container...')
    container = await docker.createContainer(containerConfig)
    log(`Container created: ${container.id}`)

    // 设置超时
    const timeoutPromise = new Promise((_, reject) => {
      timeoutId = setTimeout(() => {
        log('Execution timeout triggered')
        reject(new Error('Execution timeout'))
      }, SANDBOX_CONFIG.timeout + 10000)  // 额外 10 秒用于清理
    })

    // 启动容器
    log('Starting container...')
    await container.start()
    log('Container started')

    // 等待执行完成
    log('Waiting for container to finish...')
    const waitPromise = container.wait()

    // 竞速: 超时 vs 正常完成
    const result = await Promise.race([waitPromise, timeoutPromise])
    log(`Container finished, result: ${JSON.stringify(result)}`)

    // 清除超时
    if (timeoutId) clearTimeout(timeoutId)

    // 获取输出
    log('Getting container logs...')
    const logs = await container.logs({
      stdout: true,
      stderr: true
    })

    // 解析输出 - 分别处理 stdout 和 stderr
    const { stdout, stderr } = parseDockerLogsSeparate(logs)
    log(`Stdout length: ${stdout.length}`)
    log(`Stderr length: ${stderr.length}`)

    // stderr 可能包含 CUDA banner 等信息，记录用于调试
    if (stderr.length > 0) {
      log(`Stderr content preview: ${stderr.substring(0, 500)}`)
    }

    // 使用 stdout 解析 JSON
    const rawOutput = stdout
    log(`Stdout content preview: ${rawOutput.substring(0, 200)}`)

    // 解析 JSON 输出
    let parsedResult
    try {
      parsedResult = JSON.parse(rawOutput)
      log('Output parsed successfully')
    } catch (parseError) {
      log(`JSON parse error: ${parseError.message}`)
      // 如果 JSON 解析失败，返回原始输出作为错误
      const executionTime = (Date.now() - startTime) / 1000
      return {
        success: false,
        outputs: [{
          type: 'error',
          ename: 'OutputParseError',
          evalue: 'Failed to parse kernel output',
          traceback: [rawOutput]
        }],
        executionTime,
        gpuUsed: useGpu
      }
    }

    return {
      success: parsedResult.success,
      outputs: parsedResult.outputs || [],
      executionTime: parsedResult.executionTime || (Date.now() - startTime) / 1000,
      gpuUsed: useGpu
    }

  } catch (error) {
    log(`Execution error: ${error.message}`)
    log(`Error stack: ${error.stack}`)
    // 清除超时
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
      gpuUsed: useGpu
    }

  } finally {
    // 清理容器
    log('Cleaning up container...')
    if (container) {
      try {
        await container.remove({ force: true })
        log('Container removed')
      } catch (e) {
        log(`Container cleanup error: ${e.message}`)
      }
    }
  }
}

/**
 * 解析 Docker 日志输出
 * Docker 日志格式: [8字节头][数据]
 */
function parseDockerLogs(logs) {
  if (!logs || logs.length === 0) return ''

  // 如果是 Buffer
  if (Buffer.isBuffer(logs)) {
    let output = ''
    let offset = 0

    while (offset < logs.length) {
      // 跳过 8 字节头
      if (offset + 8 > logs.length) break

      const streamType = logs[offset]
      const length = logs.readUInt32BE(offset + 4)

      offset += 8

      if (offset + length > logs.length) break

      const chunk = logs.slice(offset, offset + length).toString('utf8')
      output += chunk
      offset += length
    }

    return output
  }

  // 如果是字符串
  return logs.toString()
}

/**
 * 解析 Docker 日志输出，分别返回 stdout 和 stderr
 * Docker 日志格式: [8字节头][数据]
 * 头部第一个字节: 0=stdin, 1=stdout, 2=stderr
 */
function parseDockerLogsSeparate(logs) {
  if (!logs || logs.length === 0) return { stdout: '', stderr: '' }

  // 如果是 Buffer
  if (Buffer.isBuffer(logs)) {
    let stdout = ''
    let stderr = ''
    let offset = 0

    while (offset < logs.length) {
      // 跳过 8 字节头
      if (offset + 8 > logs.length) break

      const streamType = logs[offset]  // 1=stdout, 2=stderr
      const length = logs.readUInt32BE(offset + 4)

      offset += 8

      if (offset + length > logs.length) break

      const chunk = logs.slice(offset, offset + length).toString('utf8')

      if (streamType === 1) {
        stdout += chunk
      } else if (streamType === 2) {
        stderr += chunk
      }

      offset += length
    }

    return { stdout, stderr }
  }

  // 如果是字符串，无法区分，全部作为 stdout
  return { stdout: logs.toString(), stderr: '' }
}

/**
 * 检查沙箱镜像是否存在
 */
export async function checkImageExists(useGpu = false) {
  const image = useGpu ? SANDBOX_CONFIG.imageGpu : SANDBOX_CONFIG.imageCpu

  try {
    await docker.getImage(image).inspect()
    return true
  } catch {
    return false
  }
}

/**
 * 拉取沙箱镜像
 */
export async function pullImage(useGpu = false) {
  const image = useGpu ? SANDBOX_CONFIG.imageGpu : SANDBOX_CONFIG.imageCpu

  return new Promise((resolve, reject) => {
    docker.pull(image, (err, stream) => {
      if (err) {
        reject(err)
        return
      }

      docker.modem.followProgress(stream, (err, output) => {
        if (err) {
          reject(err)
        } else {
          resolve(output)
        }
      })
    })
  })
}

export default {
  runPythonCode,
  checkGPUAvailable,
  checkImageExists,
  pullImage,
  SANDBOX_CONFIG
}