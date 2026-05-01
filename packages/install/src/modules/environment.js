/**
 * 环境检测模块
 */
import { execSync } from 'child_process'
import Docker from 'dockerode'

const docker = new Docker()

/**
 * 检查 Docker 环境
 */
async function checkDocker() {
  try {
    const info = await docker.info()
    return {
      installed: true,
      version: info.ServerVersion || null
    }
  } catch {
    return {
      installed: false,
      version: null
    }
  }
}

/**
 * 检查 Node.js 环境
 */
function checkNode() {
  try {
    const version = execSync('node --version', {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore']
    }).trim()
    return {
      installed: true,
      version: version.replace('v', '')
    }
  } catch {
    return {
      installed: false,
      version: null
    }
  }
}

/**
 * 检查 GPU 环境
 * @returns {{ available: boolean, info: string|null, driverVersion: string|null }}
 */
function checkGPU() {
  try {
    // stdio: ['ignore', 'pipe', 'ignore'] 隐藏 stderr 输出
    const output = execSync('nvidia-smi', {
      timeout: 5000,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore']
    })

    // 解析驱动版本：Driver Version: xxx.xx
    const driverMatch = output.match(/Driver Version:\s*(\d+\.\d+)/)
    const driverVersion = driverMatch ? driverMatch[1] : null

    // 解析 CUDA 版本上限：CUDA Version: xxx.xx
    const cudaMatch = output.match(/CUDA Version:\s*(\d+\.\d+)/)
    const cudaVersion = cudaMatch ? cudaMatch[1] : null

    if (output.includes('GPU')) {
      // 提取 GPU 名称
      const lines = output.split('\n').filter(l => l.includes('GPU'))
      const gpuInfo = lines[0] || '检测到 GPU'
      return {
        available: true,
        info: gpuInfo.trim(),
        driverVersion,
        cudaVersion
      }
    }
  } catch {}

  return {
    available: false,
    info: null,
    driverVersion: null,
    cudaVersion: null
  }
}

/**
 * 检查端口是否可用
 */
function checkPort(port) {
  try {
    execSync(`nc -z localhost ${port}`, {
      timeout: 1000,
      stdio: ['ignore', 'pipe', 'ignore']
    })
    return false // 端口被占用
  } catch {
    return true // 端口可用
  }
}

/**
 * 综合环境检测
 */
export async function checkEnvironment() {
  const dockerEnv = await checkDocker()
  const nodeEnv = checkNode()
  const gpuEnv = checkGPU()

  return {
    docker: dockerEnv.installed,
    dockerVersion: dockerEnv.version,
    node: nodeEnv.installed,
    nodeVersion: nodeEnv.version,
    gpu: gpuEnv.available,
    gpuInfo: gpuEnv.info,
    gpuDriverVersion: gpuEnv.driverVersion,
    gpuCudaVersion: gpuEnv.cudaVersion
  }
}