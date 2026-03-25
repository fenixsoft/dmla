/**
 * 沙箱管理模块
 * 负责创建和管理 Docker 容器执行 Python 代码
 */
import Docker from 'dockerode'

const docker = new Docker()

// 沙箱配置
const SANDBOX_CONFIG = {
  imageCpu: 'ideaspaces-sandbox:latest',
  imageGpu: 'ideaspaces-sandbox:gpu',
  timeout: 60000,           // 60 秒超时
  memory: 4 * 1024 * 1024 * 1024  // 4GB 内存
}

/**
 * 检查 GPU 是否可用
 */
async function checkGPUAvailable() {
  try {
    // 尝试运行 nvidia-smi
    const result = await runCommand('nvidia/cuda:11.8-base', ['nvidia-smi', '-L'])
    return result.includes('GPU')
  } catch {
    return false
  }
}

/**
 * 运行命令并获取输出
 */
async function runCommand(image, cmd) {
  return new Promise((resolve, reject) => {
    docker.run(image, cmd, (err, data, container) => {
      if (err) {
        reject(err)
        return
      }

      container.remove()

      if (data.StatusCode !== 0) {
        reject(new Error(`Command failed with status ${data.StatusCode}`))
        return
      }

      resolve(data)
    })
  })
}

/**
 * 执行 Python 代码
 * @param {string} code - Python 代码
 * @param {boolean} useGpu - 是否使用 GPU
 * @returns {Promise<{success: boolean, output: string, error: string|null, executionTime: number}>}
 */
export async function runPythonCode(code, useGpu = false) {
  const startTime = Date.now()

  // 选择镜像
  const image = useGpu ? SANDBOX_CONFIG.imageGpu : SANDBOX_CONFIG.imageCpu

  // 创建容器配置
  const containerConfig = {
    Image: image,
    Cmd: ['python3', '-c', code],
    HostConfig: {
      Memory: SANDBOX_CONFIG.memory,
      AutoRemove: false  // 手动移除以获取日志
    },
    Env: [
      'PYTHONUNBUFFERED=1'
    ]
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
      }, SANDBOX_CONFIG.timeout)
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
    const output = parseDockerLogs(logs)
    const executionTime = (Date.now() - startTime) / 1000

    return {
      success: result.StatusCode === 0,
      output: output.trim(),
      error: result.StatusCode !== 0 ? output.trim() : null,
      executionTime,
      gpuUsed: useGpu
    }

  } catch (error) {
    // 清除超时
    if (timeoutId) clearTimeout(timeoutId)

    const executionTime = (Date.now() - startTime) / 1000

    return {
      success: false,
      output: '',
      error: error.message || 'Unknown error',
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