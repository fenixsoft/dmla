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
 * 检查是否启用 Volume Mount
 */
function shouldMountSharedModules() {
  return process.env.MOUNT_SHARED_MODULES !== 'false'
}

/**
 * 检查 GPU 是否可用
 * 通过运行 nvidia-smi 命令检测 GPU 状态
 */
export async function checkGPUAvailable() {
  let container = null

  try {
    // 创建容器运行 nvidia-smi 命令
    container = await docker.createContainer({
      Image: 'nvidia/cuda:11.8-base',
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

  // 选择镜像
  const image = useGpu ? SANDBOX_CONFIG.imageGpu : SANDBOX_CONFIG.imageCpu

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

  // Volume Mount 配置 - 挂载共享模块
  const useMount = shouldMountSharedModules()
  const sharedModulesPath = getSharedModulesPath()

  if (useMount && sharedModulesPath) {
    // 检查共享模块目录是否存在
    if (fs.existsSync(sharedModulesPath)) {
      containerConfig.HostConfig.Binds = [
        `${sharedModulesPath}:/usr/local/lib/python3.11/site-packages/shared:ro`
      ]
      console.log(`[Sandbox] Volume Mount 已启用: ${sharedModulesPath}`)
    } else {
      console.warn(`[Sandbox] 警告: 共享模块目录不存在: ${sharedModulesPath}`)
      console.warn('[Sandbox] 提示: 运行 npm run extract:shared 生成共享模块')
    }
  } else if (!sharedModulesPath) {
    console.log('[Sandbox] 独立安装模式，无共享模块目录')
  } else {
    console.log('[Sandbox] Volume Mount 已禁用 (MOUNT_SHARED_MODULES=false)')
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
    container = await docker.createContainer(containerConfig)

    // 设置超时
    const timeoutPromise = new Promise((_, reject) => {
      timeoutId = setTimeout(() => {
        reject(new Error('Execution timeout'))
      }, SANDBOX_CONFIG.timeout + 10000)  // 额外 10 秒用于清理
    })

    // 启动容器
    await container.start()

    // 等待执行完成
    const waitPromise = container.wait()

    // 竞速: 超时 vs 正常完成
    const result = await Promise.race([waitPromise, timeoutPromise])

    // 清除超时
    if (timeoutId) clearTimeout(timeoutId)

    // 获取输出
    const logs = await container.logs({
      stdout: true,
      stderr: true
    })

    // 解析输出
    const rawOutput = parseDockerLogs(logs)

    // 解析 JSON 输出
    let parsedResult
    try {
      parsedResult = JSON.parse(rawOutput)
    } catch (parseError) {
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
    if (container) {
      try {
        await container.remove({ force: true })
      } catch (e) {
        console.warn('Failed to remove container:', e.message)
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
  pullImage
}