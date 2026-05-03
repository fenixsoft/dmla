/**
 * 沙箱管理模块
 * 负责创建和管理 Docker 容器执行 Python 代码
 */
import Docker from 'dockerode'
import path from 'path'
import { fileURLToPath } from 'url'
import fs from 'fs'
import os from 'os'

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

// DMLA 配置文件路径
const DMLA_CONFIG_DIR = path.join(os.homedir(), '.dmla')
const DMLA_CONFIG_FILE = path.join(DMLA_CONFIG_DIR, 'config.json')
const DEFAULT_DATA_DIR = path.join(os.homedir(), 'dmla-data')

/**
 * 读取 DMLA 配置文件
 */
function readDmlaConfig() {
  try {
    if (fs.existsSync(DMLA_CONFIG_FILE)) {
      const content = fs.readFileSync(DMLA_CONFIG_FILE, 'utf8')
      return JSON.parse(content)
    }
  } catch (error) {
    log(`配置文件读取失败: ${error.message}`)
  }
  return { dataVolumePath: DEFAULT_DATA_DIR }
}

/**
 * 获取数据卷路径
 */
function getDataVolumePath() {
  const config = readDmlaConfig()
  return config.dataVolumePath || DEFAULT_DATA_DIR
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
 * 获取 dmla_progress.py 路径（开发模式挂载）
 */
function getProgressReporterPath() {
  if (process.env.PROGRESS_REPORTER_PATH) {
    return process.env.PROGRESS_REPORTER_PATH
  }
  return DEFAULT_PROGRESS_REPORTER_PATH
}

// dmla_progress.py 默认路径
const DEFAULT_PROGRESS_REPORTER_PATH = PROJECT_ROOT
  ? path.join(PROJECT_ROOT, 'local-server', 'src', 'dmla_progress.py')
  : null

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
 * 检查 CUDA 兼容性
 * 在 GPU 镜像中运行简单的 CUDA 操作测试，验证 PyTorch 与 GPU 兼容
 * @returns {Promise<{compatible: boolean, issues: string[], details: object}>}
 */
export async function checkCUDACompatibility() {
  let container = null

  const testCode = `
import torch
import json

result = {
    'pytorch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_version': str(torch.version.cuda) if torch.cuda.is_available() else None,
    'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'compatible': True,
    'test_passed': False,
    'error': None
}

if torch.cuda.is_available():
    try:
        x = torch.randn(100, 100, device='cuda')
        y = x + x
        torch.cuda.synchronize()
        result['test_passed'] = True
    except RuntimeError as e:
        result['compatible'] = False
        result['error'] = str(e)
        if 'no kernel image' in str(e) or 'CUDA error' in str(e):
            result['error_type'] = 'compatibility'

print(json.dumps(result))
`

  try {
    container = await docker.createContainer({
      Image: SANDBOX_CONFIG.imageGpu,
      Cmd: ['python3', '-c', testCode],
      HostConfig: {
        DeviceRequests: [{
          Driver: 'nvidia',
          Count: -1,
          Capabilities: [['gpu']]
        }],
        AutoRemove: false
      },
      Env: ['PYTHONUNBUFFERED=1']
    })

    await container.start()
    await container.wait()

    const logs = await container.logs({
      stdout: true,
      stderr: true
    })

    const { stdout, stderr } = parseDockerLogsSeparate(logs)

    // 尝试解析 JSON 输出
    const jsonStart = stdout.indexOf('{')
    if (jsonStart !== -1) {
      try {
        const result = JSON.parse(stdout.substring(jsonStart))
        return {
          compatible: result.compatible && result.test_passed,
          issues: result.error ? [result.error] : [],
          details: result
        }
      } catch {
        // JSON 解析失败
      }
    }

    // 如果无法解析，检查 stderr 是否有 CUDA 错误
    if (stderr.includes('no kernel image') || stderr.includes('CUDA error')) {
      return {
        compatible: false,
        issues: [stderr],
        details: { raw_output: stderr }
      }
    }

    // 默认返回未知状态
    return {
      compatible: true,  // 假设兼容，让实际执行来验证
      issues: [],
      details: { stdout, stderr }
    }

  } catch (error) {
    return {
      compatible: false,
      issues: [error.message],
      details: { error: error.message }
    }
  } finally {
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
 * @param {boolean} useGpu - 是否启用 GPU 设备
 * @param {string|null} imageOverride - 可选，指定使用的镜像名称（覆盖默认选择）
 * @param {number|null} timeoutOverride - 可选，超时时间（秒），null 表示 unlimited
 * @returns {Promise<{success: boolean, outputs: Array, executionTime: number, gpuUsed: boolean}>}
 */
export async function runPythonCode(code, useGpu = false, imageOverride = null, timeoutOverride = null) {
  const startTime = Date.now()

  // 计算实际超时时间
  const actualTimeout = timeoutOverride === null
    ? null  // unlimited
    : (timeoutOverride || Math.floor(SANDBOX_CONFIG.timeout / 1000))

  log(`runPythonCode called, useGpu=${useGpu}, code length=${code.length}, imageOverride=${imageOverride}, timeout=${actualTimeout === null ? 'unlimited' : actualTimeout}`)

  // 选择镜像：优先使用指定的镜像，否则根据 useGpu 选择
  const image = imageOverride || (useGpu ? SANDBOX_CONFIG.imageGpu : SANDBOX_CONFIG.imageCpu)
  log(`Using image: ${image}`)

  // GPU 兼容性预检查
  if (useGpu) {
    log('GPU mode: running CUDA compatibility pre-check...')
    const compatResult = await checkCUDACompatibility()
    log(`CUDA compatibility check result: ${JSON.stringify(compatResult)}`)

    if (!compatResult.compatible) {
      log('CUDA compatibility check failed')
      const executionTime = (Date.now() - startTime) / 1000

      // 构建详细的错误信息
      const errorDetails = compatResult.details || {}
      const errorType = errorDetails.error_type || 'unknown'

      let errorMessage = 'CUDA 兼容性错误：PyTorch CUDA 版本与您的 GPU 不兼容\n\n'

      if (errorType === 'compatibility' || compatResult.issues.some(i => i.includes('no kernel image'))) {
        errorMessage += `诊断详情:\n`
        errorMessage += `- PyTorch 版本: ${errorDetails.pytorch_version || '未知'}\n`
        errorMessage += `- CUDA 版本: ${errorDetails.cuda_version || '未知'}\n`
        errorMessage += `- GPU 设备: ${errorDetails.device_name || '未知'}\n`
        errorMessage += `- 错误类型: CUDA kernel 不兼容\n\n`
        errorMessage += `解决方案:\n`
        errorMessage += `1. 使用 CPU 模式运行代码（在前端选择 "Run on CPU"）\n`
        errorMessage += `2. 在代码开头添加: device = torch.device('cpu')\n`
        errorMessage += `3. 重新构建兼容的 Docker 镜像（修改 Dockerfile.sandbox 使用 CUDA 12.x）\n\n`
        errorMessage += `更多诊断信息请运行: dmla doctor`
      } else {
        errorMessage += `错误详情: ${compatResult.issues.join('\n')}\n\n`
        errorMessage += `建议使用 CPU 模式运行代码。`
      }

      return {
        success: false,
        outputs: [{
          type: 'error',
          ename: 'CUDACompatError',
          evalue: 'CUDA 兼容性错误',
          traceback: [errorMessage]
        }],
        executionTime,
        gpuUsed: false
      }
    }
    log('CUDA compatibility check passed')
  }

  // 创建容器配置 - 使用 kernel_runner.py 执行代码
  const timeoutSeconds = actualTimeout === null ? 86400 : actualTimeout  // unlimited 使用 24 小时
  const containerConfig = {
    Image: image,
    Cmd: ['python3', '/workspace/kernel_runner.py', '--code', code, '--timeout', String(timeoutSeconds)],
    HostConfig: {
      Memory: SANDBOX_CONFIG.memory,
      AutoRemove: false  // 手动移除以获取日志
    },
    // matplotlib 使用 IPython Kernel 的 inline 后端，自动发送 display_data
    // PYTHONPATH 添加 /workspace 以支持导入 volume-mounted 的模块
    Env: [
      'PYTHONUNBUFFERED=1',
      'PYTHONPATH=/workspace',
      actualTimeout === null ? 'DMLA_NO_TIMEOUT=1' : ''
    ].filter(e => e)  // 过滤空字符串
  }

  log('Container config created')

  // Volume Mount 配置
  const useMount = shouldMountSharedModules()
  const sharedModulesPath = getSharedModulesPath()
  const mountKernelRunner = shouldMountKernelRunner()
  const kernelRunnerPath = getKernelRunnerPath()

  // 收集所有需要挂载的路径
  const binds = []

  // 挂载数据目录
  const dataVolumePath = getDataVolumePath()
  if (dataVolumePath && fs.existsSync(dataVolumePath)) {
    binds.push(`${dataVolumePath}:/data`)
    console.log(`[Sandbox] 数据目录 Volume Mount: ${dataVolumePath}`)
  } else if (dataVolumePath) {
    console.warn(`[Sandbox] 警告: 数据目录不存在: ${dataVolumePath}`)
    console.log(`[Sandbox] 提示: 运行 'dmla data' 创建数据目录`)
  }

  // 挂载共享模块到 /workspace/shared（而非 site-packages）
  // 原因：PYTHONPATH=/workspace，这样 Python 可以直接导入 shared.xxx
  // 避免 Windows Docker 的 site-packages 路径兼容性问题
  if (useMount && sharedModulesPath && fs.existsSync(sharedModulesPath)) {
    binds.push(`${sharedModulesPath}:/workspace/shared:ro`)
    console.log(`[Sandbox] 共享模块 Volume Mount: ${sharedModulesPath} -> /workspace/shared`)
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

  // 挂载 dmla_progress.py（开发模式新增文件，无需重建镜像）
  const progressReporterPath = getProgressReporterPath()
  if (mountKernelRunner && progressReporterPath && fs.existsSync(progressReporterPath)) {
    binds.push(`${progressReporterPath}:/workspace/dmla_progress.py:ro`)
    console.log(`[Sandbox] dmla_progress.py Volume Mount: ${progressReporterPath}`)
  } else if (mountKernelRunner && progressReporterPath) {
    console.warn(`[Sandbox] 警告: dmla_progress.py 不存在: ${progressReporterPath}`)
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

    // 设置超时（使用动态计算的超时时间，unlimited 时为 24 小时）
    const containerTimeoutMs = timeoutSeconds * 1000 + 10000  // 转换为毫秒，额外 10 秒用于清理
    const timeoutPromise = new Promise((_, reject) => {
      timeoutId = setTimeout(() => {
        log('Execution timeout triggered')
        reject(new Error('Execution timeout'))
      }, containerTimeoutMs)
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

    // stderr 可能包含调试信息
    if (stderr.length > 0) {
      log(`Stderr content preview: ${stderr.substring(0, 500)}`)
    }

    // stdout 可能包含 CUDA banner + JSON，需要提取 JSON 部分
    // 找到第一个 '{' 作为 JSON 开始
    const jsonStart = stdout.indexOf('{')
    if (jsonStart === -1) {
      log(`No JSON found in stdout`)
      const executionTime = (Date.now() - startTime) / 1000
      return {
        success: false,
        outputs: [{
          type: 'error',
          ename: 'OutputParseError',
          evalue: 'No JSON output found',
          traceback: [stdout.substring(0, 1000)]
        }],
        executionTime,
        gpuUsed: useGpu
      }
    }

    const rawOutput = stdout.substring(jsonStart)
    log(`JSON extracted from position ${jsonStart}, length: ${rawOutput.length}`)

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
  checkCUDACompatibility,
  checkImageExists,
  pullImage,
  SANDBOX_CONFIG
}