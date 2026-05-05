/**
 * 服务管理命令
 */
import chalk from 'chalk'
import Docker from 'dockerode'
import { spawn } from 'child_process'
import http from 'http'
import path from 'path'
import { fileURLToPath, pathToFileURL } from 'url'
import fs from 'fs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const docker = new Docker()

// 配置
const CONFIG = {
  imageCpu: 'dmla-sandbox:cpu',
  imageGpu: 'dmla-sandbox:gpu',
  defaultPort: 3001
}

/**
 * 从 Docker 镜像标签解析日期
 * 标签格式: YYYY.M.D-HHMM (如 2026.4.21-2025)
 * @returns {Date|null}
 */
function parseImageTagDate(tag) {
  const match = tag.match(/^(\d{4})\.(\d{1,2})\.(\d{1,2})-(\d{4})$/)
  if (!match) return null

  const [, year, month, day, time] = match
  const hour = time.substring(0, 2)
  const minute = time.substring(2, 4)

  return new Date(`${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}T${hour}:${minute}:00+08:00`)
}

/**
 * 获取 Docker 镜像的最新标签日期
 * @param {string} imageType - 'cpu' 或 'gpu'
 * @returns {Promise<Date|null>}
 */
async function getImageLatestDate(imageType) {
  const imageName = imageType === 'gpu' ? 'dmla-sandbox' : 'dmla-sandbox'

  try {
    const images = await docker.listImages()
    const targetTag = imageType === 'gpu' ? ':gpu' : ':cpu'

    for (const image of images) {
      if (image.RepoTags) {
        for (const tag of image.RepoTags) {
          if (tag.includes(imageName) && tag.includes(targetTag)) {
            // 尝试从标签解析日期
            const tagPart = tag.split(':')[1]
            const date = parseImageTagDate(tagPart)
            if (date) return date

            // 如果标签不是日期格式，使用镜像创建时间
            return new Date(image.Created * 1000)
          }
        }
      }
    }
    return null
  } catch {
    return null
  }
}

/**
 * 获取 CLI 包的构建日期
 * @returns {Date|null}
 */
function getCliBuildDate() {
  try {
    // 尝试读取 version.json（build 时生成）
    const versionPath = path.resolve(__dirname, '../../version.json')
    if (fs.existsSync(versionPath)) {
      const versionInfo = JSON.parse(fs.readFileSync(versionPath, 'utf8'))
      return new Date(versionInfo.buildTime)
    }

    // 如果没有 version.json，使用 package.json 的修改时间作为参考
    const pkgPath = path.resolve(__dirname, '../../package.json')
    if (fs.existsSync(pkgPath)) {
      const stats = fs.statSync(pkgPath)
      return stats.mtime
    }

    return null
  } catch {
    return null
  }
}

/**
 * 检查 --dev 模式的版本兼容性
 * 比较镜像标签日期和 CLI 包构建日期
 */
async function checkDevModeCompatibility(imageType) {
  const imageDate = await getImageLatestDate(imageType)
  const cliDate = getCliBuildDate()

  if (!imageDate || !cliDate) {
    // 无法获取日期，不警告
    return { compatible: true, imageDate: null, cliDate: null }
  }

  // 镜像日期比 CLI 包日期新，说明用户可能更新了镜像但 CLI 包是旧版本
  // 在 --dev 模式下，本地代码会覆盖镜像中的代码，可能导致代码版本不匹配
  const imageNewer = imageDate > cliDate

  return {
    compatible: !imageNewer,
    imageDate,
    cliDate,
    warning: imageNewer ? '镜像版本比 CLI 包更新，本地代码可能覆盖了新版本镜像中的代码' : null
  }
}

/**
 * 检查端口是否可用
 */
async function checkPortAvailable(port) {
  return new Promise((resolve) => {
    const server = http.createServer()
    server.once('error', () => resolve(false))
    server.once('listening', () => {
      server.close()
      resolve(true)
    })
    server.listen(port)
  })
}

/**
 * 检查镜像是否存在
 */
async function checkImageExists(type) {
  const image = type === 'gpu' ? CONFIG.imageGpu : CONFIG.imageCpu
  try {
    await docker.getImage(image).inspect()
    return true
  } catch {
    return false
  }
}

/**
 * 检查可用镜像并自动选择
 * @returns {Object} { imageType: 'cpu'|'gpu', message: string }
 */
async function resolveImageType(useGpu) {
  const cpuExists = await checkImageExists('cpu')
  const gpuExists = await checkImageExists('gpu')

  // 用户明确指定了 GPU
  if (useGpu) {
    if (gpuExists) {
      return { imageType: 'gpu', message: 'GPU' }
    }
    // 用户想要 GPU 但 GPU 镜像不存在
    if (cpuExists) {
      console.log(chalk.yellow('⚠️ GPU 镜像不存在，将使用 CPU 镜像'))
      return { imageType: 'cpu', message: 'CPU (降级)' }
    }
    return { imageType: null, message: '无可用镜像' }
  }

  // 用户未指定，自动选择
  if (cpuExists) {
    return { imageType: 'cpu', message: 'CPU' }
  }

  // CPU 镜像不存在
  if (gpuExists) {
    console.log(chalk.yellow('⚠️ CPU 镜像不存在，自动使用 GPU 镜像'))
    return { imageType: 'gpu', message: 'GPU (自动)' }
  }

  return { imageType: null, message: '无可用镜像' }
}

/**
 * 检查 GPU 是否可用
 */
async function checkGPUAvailable() {
  try {
    // 尝试运行 nvidia-smi 命令
    const result = await new Promise((resolve, reject) => {
      const proc = spawn('nvidia-smi', ['-L'], { timeout: 5000 })
      let output = ''
      proc.stdout.on('data', (data) => output += data.toString())
      proc.stderr.on('data', (data) => output += data.toString())
      proc.on('close', (code) => {
        if (code === 0) resolve(output)
        else reject(new Error('nvidia-smi failed'))
      })
      proc.on('error', reject)
    })
    return result.includes('GPU')
  } catch {
    return false
  }
}

/**
 * 检查 GPU 驱动兼容性
 * @returns {Promise<{compatible: boolean, driverVersion: string|null, cudaVersion: string|null}>}
 */
async function checkGPUDriverCompatibility() {
  const minDriverForCuda128 = 570

  try {
    const result = await new Promise((resolve, reject) => {
      const proc = spawn('nvidia-smi', [], { timeout: 5000 })
      let output = ''
      proc.stdout.on('data', (data) => output += data.toString())
      proc.stderr.on('data', (data) => output += data.toString())
      proc.on('close', (code) => {
        if (code === 0) resolve(output)
        else reject(new Error('nvidia-smi failed'))
      })
      proc.on('error', reject)
    })

    // 解析驱动版本
    const driverMatch = result.match(/Driver Version:\s*(\d+\.\d+)/)
    const driverVersion = driverMatch ? driverMatch[1] : null

    // 解析 CUDA 兼容上限
    const cudaMatch = result.match(/CUDA Version:\s*(\d+\.\d+)/)
    const cudaVersion = cudaMatch ? cudaMatch[1] : null

    // 判断兼容性
    const driverNum = parseFloat(driverVersion || '0')
    const compatible = driverNum >= minDriverForCuda128

    return { compatible, driverVersion, cudaVersion }
  } catch {
    return { compatible: false, driverVersion: null, cudaVersion: null }
  }
}

/**
 * 检查服务是否运行
 */
async function checkServiceRunning(port) {
  return new Promise((resolve) => {
    const req = http.request({
      hostname: 'localhost',
      port: port,
      path: '/api/health',
      method: 'GET',
      timeout: 2000
    }, (res) => {
      resolve(res.statusCode === 200)
    })
    req.on('error', () => resolve(false))
    req.on('timeout', () => {
      req.destroy()
      resolve(false)
    })
    req.end()
  })
}

/**
 * 查找运行中的服务容器
 */
async function findServiceContainer() {
  try {
    const containers = await docker.listContainers({ all: true })
    // 查找 dmla 服务容器
    for (const container of containers) {
      if (container.Names.some(name => name.includes('dmla-server'))) {
        return container
      }
    }
    return null
  } catch {
    return null
  }
}

/**
 * 查找服务器入口文件
 */
function findServerPath() {
  // 开发环境路径：packages/cli/src/commands -> ../../../local-server/src/index.js
  const serverPath = path.resolve(__dirname, '../../../local-server/src/index.js')
  // npm 包路径：packages/cli/src/commands -> ../server/index.js
  const standaloneServerPath = path.resolve(__dirname, '../server/index.js')

  // 检查 __dirname 是否正确（调试用）
  const cliPackageRoot = path.resolve(__dirname, '../..')
  const expectedServerDir = path.resolve(cliPackageRoot, 'src/server')

  if (fs.existsSync(serverPath)) {
    return serverPath
  }

  if (fs.existsSync(standaloneServerPath)) {
    return standaloneServerPath
  }

  // 调试输出：显示路径信息帮助诊断
  console.log(chalk.yellow('⚠️ 服务入口文件查找失败'))
  console.log(chalk.gray(`   __dirname: ${__dirname}`))
  console.log(chalk.gray(`   开发路径: ${serverPath} (${fs.existsSync(serverPath) ? '存在' : '不存在'})`))
  console.log(chalk.gray(`   npm路径: ${standaloneServerPath} (${fs.existsSync(standaloneServerPath) ? '存在' : '不存在'})`))
  console.log(chalk.gray(`   CLI包根目录: ${cliPackageRoot}`))

  // 检查 CLI 包根目录下的文件结构
  const srcDir = path.resolve(cliPackageRoot, 'src')
  if (fs.existsSync(srcDir)) {
    console.log(chalk.gray(`   src目录内容: ${fs.readdirSync(srcDir).join(', ')}`))
    if (fs.existsSync(expectedServerDir)) {
      console.log(chalk.gray(`   server目录内容: ${fs.readdirSync(expectedServerDir).join(', ')}`))
    }
  }

  return null
}

/**
 * 查找 kernel_runner.py 路径
 * --dev 模式下需要挂载此文件
 */
function findKernelRunnerPath() {
  // 开发环境路径：packages/cli/src/commands -> ../../../local-server/src/kernel_runner.py
  const devPath = path.resolve(__dirname, '../../../local-server/src/kernel_runner.py')
  // npm 包路径：packages/cli/src/commands -> ../server/kernel_runner.py（构建后）
  const npmPath = path.resolve(__dirname, '../server/kernel_runner.py')

  if (fs.existsSync(devPath)) {
    return devPath
  }
  if (fs.existsSync(npmPath)) {
    return npmPath
  }
  return null
}

/**
 * 查找 dmla_progress.py 路径
 * --dev 模式下需要挂载此文件
 */
function findProgressReporterPath() {
  // 开发环境路径：packages/cli/src/commands -> ../../../local-server/src/dmla_progress.py
  const devPath = path.resolve(__dirname, '../../../local-server/src/dmla_progress.py')
  // npm 包路径：packages/cli/src/commands -> ../server/dmla_progress.py（构建后）
  const npmPath = path.resolve(__dirname, '../server/dmla_progress.py')

  if (fs.existsSync(devPath)) {
    return devPath
  }
  if (fs.existsSync(npmPath)) {
    return npmPath
  }
  return null
}

/**
 * 查找共享模块目录
 * --dev 模式下需要挂载此目录
 */
function findSharedModulesPath() {
  // 开发环境路径：packages/cli/src/commands -> ../../../local-server/shared_modules
  const devPath = path.resolve(__dirname, '../../../local-server/shared_modules')
  // npm 包路径：packages/cli/src/commands -> ../../shared_modules（构建后）
  const npmPath = path.resolve(__dirname, '../../shared_modules')
  // CLI 包根目录下的 shared_modules（构建后）
  const cliRootPath = path.resolve(__dirname, '../../shared_modules')

  // 优先使用开发环境路径（如果 local-server 存在）
  if (fs.existsSync(devPath) && fs.readdirSync(devPath).length > 0) {
    return devPath
  }

  // 其次使用 npm 包路径
  if (fs.existsSync(npmPath) && fs.readdirSync(npmPath).length > 0) {
    return npmPath
  }

  // 最后检查 CLI 包根目录
  if (fs.existsSync(cliRootPath) && fs.readdirSync(cliRootPath).length > 0) {
    return cliRootPath
  }

  return null
}

/**
 * 同步启动服务（在当前进程运行，用于调试）
 * @param {number} port - 服务端口
 * @param {boolean} useGpu - 是否使用 GPU（可选，自动检测）
 * @param {boolean} dev - 开发模式（挂载本地代码）
 * @param {number} shmSize - Docker 共享内存大小（MB）
 */
export async function startServerSync(port, useGpu = false, dev = false, shmSize = 64) {
  // 检查端口
  const portAvailable = await checkPortAvailable(port)
  if (!portAvailable) {
    console.log(chalk.red(`❌ 端口 ${port} 已被占用`))
    console.log(chalk.yellow('提示: 使用 --port 选项指定其他端口'))
    return
  }

  // 智能选择镜像
  const imageResolution = await resolveImageType(useGpu)
  if (!imageResolution.imageType) {
    console.log(chalk.red('❌ 无可用镜像'))
    console.log(chalk.yellow('提示: 运行 dmla install 安装镜像'))
    return
  }
  const resolvedUseGpu = imageResolution.imageType === 'gpu'

  // --dev 模式版本检查
  if (dev) {
    const compat = await checkDevModeCompatibility(imageResolution.imageType)
    if (!compat.compatible) {
      console.log(chalk.yellow('⚠️ 开发模式版本兼容性警告'))
      console.log(chalk.gray(`   镜像构建时间: ${compat.imageDate?.toLocaleString('zh-CN') || '未知'}`))
      console.log(chalk.gray(`   CLI 包构建时间: ${compat.cliDate?.toLocaleString('zh-CN') || '未知'}`))
      console.log(chalk.yellow(`   风险: ${compat.warning}`))
      console.log(chalk.gray('   说明: --dev 模式会挂载本地代码到容器，可能覆盖镜像中的新版本代码'))
      console.log(chalk.gray('   建议: 如需使用镜像最新功能，请退出 --dev 模式，或更新 CLI 包'))
      console.log()
    }
  }

  // 检查服务是否已运行
  const alreadyRunning = await checkServiceRunning(port)
  if (alreadyRunning) {
    console.log(chalk.green(`✅ 服务已在端口 ${port} 运行`))
    return
  }

  // GPU 驱动兼容性预检
  if (resolvedUseGpu) {
    const driverCheck = await checkGPUDriverCompatibility()
    if (!driverCheck.compatible && driverCheck.driverVersion) {
      console.log(chalk.yellow(`⚠️ GPU 驱动兼容性警告`))
      console.log(chalk.gray(`   当前驱动: ${driverCheck.driverVersion}`))
      console.log(chalk.gray(`   CUDA 12.8 需要: 驱动 >= 570`))
      console.log(chalk.yellow('   解决方案：'))
      console.log(chalk.gray('      1. 升级 NVIDIA 驾动到 570+ 版本'))
      console.log(chalk.gray('      2. 使用 CPU 模式: dmla start'))
      console.log()
      console.log(chalk.gray('   继续启动 GPU 模式（可能会失败）...'))
      console.log()
    }
  }

  // 查找服务器入口
  const actualServerPath = findServerPath()
  if (!actualServerPath) {
    console.log(chalk.red('❌ 找不到服务入口文件'))
    console.log(chalk.yellow('提示: 保正确安装了 @icyfenix-dmla/cli'))
    return
  }

  // 查找共享模块路径（--dev 模式需要）
  const sharedModulesPath = dev ? findSharedModulesPath() : null
  // 查找 kernel_runner.py 路径（--dev 模式需要）
  const kernelRunnerPath = dev ? findKernelRunnerPath() : null
  // 查找 dmla_progress.py 路径（--dev 模式需要）
  const progressReporterPath = dev ? findProgressReporterPath() : null

  if (dev && !sharedModulesPath) {
    console.log(chalk.yellow('⚠️ --dev 模式需要共享模块目录'))
    console.log(chalk.gray('   未找到 shared_modules，将仅使用镜像内置模块'))
  }
  if (dev && !kernelRunnerPath) {
    console.log(chalk.yellow('⚠️ --dev 模式需要 kernel_runner.py'))
    console.log(chalk.gray('   未找到 kernel_runner.py，将仅使用镜像内置版本'))
  }
  if (dev && !progressReporterPath) {
    console.log(chalk.yellow('⚠️ --dev 模式需要 dmla_progress.py'))
    console.log(chalk.gray('   未找到 dmla_progress.py，将仅使用镜像内置版本'))
  }

  console.log(chalk.gray(`   镜像类型: ${imageResolution.message}`))
  console.log(chalk.gray('   同步模式启动...'))
  console.log(chalk.gray(`   服务入口: ${actualServerPath}`))
  if (dev && sharedModulesPath) {
    console.log(chalk.gray(`   共享模块: ${sharedModulesPath}`))
  }
  if (dev && kernelRunnerPath) {
    console.log(chalk.gray(`   执行器: ${kernelRunnerPath}`))
  }
  if (dev && progressReporterPath) {
    console.log(chalk.gray(`   进度报告: ${progressReporterPath}`))
  }
  console.log()

  // 设置环境变量
  process.env.PORT = port.toString()
  process.env.USE_GPU = resolvedUseGpu ? 'true' : 'false'
  process.env.DMLA_SYNC_MODE = 'true'  // 标记同步模式，让服务器在 import 时启动
  process.env.DMLA_SHM_SIZE = shmSize.toString()  // Docker 共享内存大小（MB）

  // --dev 模式：启用 Volume Mount
  if (dev) {
    process.env.MOUNT_SHARED_MODULES = 'true'
    process.env.MOUNT_KERNEL_RUNNER = 'true'
    if (sharedModulesPath) {
      process.env.SHARED_MODULES_PATH = sharedModulesPath
    }
    if (kernelRunnerPath) {
      process.env.KERNEL_RUNNER_PATH = kernelRunnerPath
    }
    if (progressReporterPath) {
      process.env.PROGRESS_REPORTER_PATH = progressReporterPath
    }
  }

  // 动态 import 服务器模块并直接运行
  // 服务器模块会在 import 时自动启动（因为入口点检测逻辑）
  // Windows 需要将路径转换为 file:// URL 格式
  try {
    const serverURL = pathToFileURL(actualServerPath).href
    await import(serverURL)
  } catch (error) {
    console.log(chalk.red(`❌ 服务启动失败: ${error.message}`))
    console.log(chalk.gray(error.stack))
  }
}

/**
 * 启动服务（异步模式，spawn 子进程）
 * @param {number} port - 服务端口
 * @param {boolean} useGpu - 是否使用 GPU（可选，自动检测）
 * @param {boolean} dev - 开发模式（挂载本地代码）
 * @param {number} shmSize - Docker 共享内存大小（MB）
 */
export async function startServer(port, useGpu = false, dev = false, shmSize = 64) {
  // 检查端口
  const portAvailable = await checkPortAvailable(port)
  if (!portAvailable) {
    console.log(chalk.red(`❌ 端口 ${port} 已被占用`))
    console.log(chalk.yellow('提示: 使用 --port 选项指定其他端口'))
    return
  }

  // 智能选择镜像
  const imageResolution = await resolveImageType(useGpu)
  if (!imageResolution.imageType) {
    console.log(chalk.red('❌ 无可用镜像'))
    console.log(chalk.yellow('提示: 运行 dmla install 安装镜像'))
    return
  }
  const resolvedUseGpu = imageResolution.imageType === 'gpu'

  // --dev 模式版本检查
  if (dev) {
    const compat = await checkDevModeCompatibility(imageResolution.imageType)
    if (!compat.compatible) {
      console.log(chalk.yellow('⚠️ 开发模式版本兼容性警告'))
      console.log(chalk.gray(`   镜像构建时间: ${compat.imageDate?.toLocaleString('zh-CN') || '未知'}`))
      console.log(chalk.gray(`   CLI 包构建时间: ${compat.cliDate?.toLocaleString('zh-CN') || '未知'}`))
      console.log(chalk.yellow(`   风险: ${compat.warning}`))
      console.log(chalk.gray('   说明: --dev 模式会挂载本地代码到容器，可能覆盖镜像中的新版本代码'))
      console.log(chalk.gray('   建议: 如需使用镜像最新功能，请退出 --dev 模式，或更新 CLI 包'))
      console.log()
    }
  }

  // 检查服务是否已运行
  const alreadyRunning = await checkServiceRunning(port)
  if (alreadyRunning) {
    console.log(chalk.green(`✅ 服务已在端口 ${port} 运行`))
    return
  }

  // GPU 驱动兼容性预检
  if (resolvedUseGpu) {
    const driverCheck = await checkGPUDriverCompatibility()
    if (!driverCheck.compatible && driverCheck.driverVersion) {
      console.log(chalk.yellow(`⚠️ GPU 驱动兼容性警告`))
      console.log(chalk.gray(`   当前驱动: ${driverCheck.driverVersion}`))
      console.log(chalk.gray(`   CUDA 12.8 需要: 驱动 >= 570`))
      console.log(chalk.yellow('   解决方案：'))
      console.log(chalk.gray('      1. 升级 NVIDIA 驾动到 570+ 版本'))
      console.log(chalk.gray('      2. 使用 CPU 模式: dmla start'))
      console.log()
      console.log(chalk.gray('   继续启动 GPU 模式（可能会失败）...'))
      console.log()
    }
  }

  // 启动服务
  console.log(chalk.gray(`   镜像类型: ${imageResolution.message}`))
  console.log(chalk.gray('   正在启动...'))

  try {
    const actualServerPath = findServerPath()

    if (!actualServerPath) {
      console.log(chalk.red('❌ 找不到服务入口文件'))
      console.log(chalk.yellow('提示: 确保正确安装了 @icyfenix-dmla/cli'))
      return
    }

    // 查找共享模块路径（--dev 模式需要）
    const sharedModulesPath = dev ? findSharedModulesPath() : null
    // 查找 kernel_runner.py 路径（--dev 模式需要）
    const kernelRunnerPath = dev ? findKernelRunnerPath() : null
    // 查找 dmla_progress.py 路径（--dev 模式需要）
    const progressReporterPath = dev ? findProgressReporterPath() : null

    if (dev && sharedModulesPath) {
      console.log(chalk.gray(`   共享模块: ${sharedModulesPath}`))
    }
    if (dev && kernelRunnerPath) {
      console.log(chalk.gray(`   执行器: ${kernelRunnerPath}`))
    }
    if (dev && progressReporterPath) {
      console.log(chalk.gray(`   进度报告: ${progressReporterPath}`))
    }

    // 日志文件路径
    const logDir = path.resolve(__dirname, '../../logs')
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true })
    }
    const logFile = path.join(logDir, 'server.log')
    const errorLogFile = path.join(logDir, 'server-error.log')

    console.log(chalk.gray(`   日志文件: ${logFile}`))

    // 创建日志文件流
    const logStream = fs.openSync(logFile, 'a')
    const errorLogStream = fs.openSync(errorLogFile, 'a')

    const env = {
      ...process.env,
      PORT: port.toString(),
      USE_GPU: resolvedUseGpu ? 'true' : 'false',
      DMLA_LOG_FILE: logFile  // 传递日志文件路径给服务端
    }

    // --dev 模式：启用 Volume Mount
    if (dev) {
      env.MOUNT_SHARED_MODULES = 'true'
      env.MOUNT_KERNEL_RUNNER = 'true'
      if (sharedModulesPath) {
        env.SHARED_MODULES_PATH = sharedModulesPath
      }
      if (kernelRunnerPath) {
        env.KERNEL_RUNNER_PATH = kernelRunnerPath
      }
      if (progressReporterPath) {
        env.PROGRESS_REPORTER_PATH = progressReporterPath
      }
    }

    // 写入启动日志
    const timestamp = new Date().toISOString()
    fs.writeSync(logStream, `[${timestamp}] Server starting...\n`)
    fs.writeSync(logStream, `[${timestamp}] Server path: ${actualServerPath}\n`)
    fs.writeSync(logStream, `[${timestamp}] Port: ${port}\n`)
    fs.writeSync(logStream, `[${timestamp}] GPU: ${resolvedUseGpu} (${imageResolution.message})\n`)
    if (dev) {
      fs.writeSync(logStream, `[${timestamp}] Dev mode: enabled (volume mount)\n`)
      if (sharedModulesPath) {
        fs.writeSync(logStream, `[${timestamp}] Shared modules: ${sharedModulesPath}\n`)
      }
    }

    // 使用 spawn 启动 server 进程
    // 重要：stdio 必须是 'ignore' 或管道，不能是 'inherit'
    // 因为 'inherit' 会让子进程依赖父进程的 stdout，父进程退出后子进程也会退出
    const serverProcess = spawn('node', [actualServerPath], {
      env,
      stdio: ['ignore', logStream, errorLogStream],  // stdin: ignore, stdout: log file, stderr: error log
      detached: true,
      windowsHide: true  // Windows 下隐藏窗口
    })

    // 监听子进程事件（调试用）
    serverProcess.on('error', (err) => {
      fs.writeSync(errorLogStream, `[${new Date().toISOString()}] Spawn error: ${err.message}\n`)
    })

    serverProcess.on('exit', (code, signal) => {
      const msg = `[${new Date().toISOString()}] Process exited: code=${code}, signal=${signal}\n`
      fs.writeSync(logStream, msg)
      fs.writeSync(errorLogStream, msg)
    })

    serverProcess.unref()

    // 关闭父进程中的文件描述符（子进程会保留自己的副本）
    fs.closeSync(logStream)
    fs.closeSync(errorLogStream)

    // 等待服务启动
    console.log(chalk.gray('   等待服务就绪...'))
    let attempts = 0
    const maxAttempts = 30

    while (attempts < maxAttempts) {
      const running = await checkServiceRunning(port)
      if (running) {
        console.log(chalk.green(`✅ 服务已启动: http://localhost:${port}`))
        console.log(chalk.gray(`   健康检查: http://localhost:${port}/api/health`))
        console.log(chalk.gray(`   日志查看: ${logFile}`))
        if (dev) {
          console.log(chalk.cyan('   开发模式: 已启用 Volume Mount'))
        }
        return
      }
      await new Promise(resolve => setTimeout(resolve, 500))
      attempts++
    }

    console.log(chalk.yellow('⚠️ 服务启动超时'))
    console.log(chalk.gray(`   请查看日志: ${logFile}`))
    console.log(chalk.gray(`   或使用 --sync 模式调试`))
  } catch (error) {
    console.log(chalk.red(`❌ 启动失败: ${error.message}`))
    console.log(chalk.gray(error.stack))
  }
}

/**
 * 停止服务
 * @param {number} port - 服务端口
 */
export async function stopServer(port = CONFIG.defaultPort) {
  // 首先尝试通过 API 停止服务
  const running = await checkServiceRunning(port)

  if (running) {
    try {
      // 调用 shutdown API
      await new Promise((resolve, reject) => {
        const req = http.request({
          hostname: 'localhost',
          port: port,
          path: '/api/shutdown',
          method: 'POST',
          timeout: 5000
        }, (res) => {
          if (res.statusCode === 200) {
            console.log(chalk.green('✅ 服务已停止'))
            resolve()
          } else {
            reject(new Error(`HTTP ${res.statusCode}`))
          }
        })
        req.on('error', (e) => reject(e))
        req.on('timeout', () => {
          req.destroy()
          reject(new Error('Timeout'))
        })
        req.end()
      })

      // 等待服务完全关闭
      let attempts = 0
      while (attempts < 10) {
        const stillRunning = await checkServiceRunning(port)
        if (!stillRunning) break
        await new Promise(r => setTimeout(r, 200))
        attempts++
      }
      return
    } catch (error) {
      console.log(chalk.yellow(`⚠️ 通过 API 停止失败: ${error.message}`))
    }
  }

  // 尝试查找并停止 Docker 容器
  const container = await findServiceContainer()

  if (container) {
    try {
      const containerObj = docker.getContainer(container.Id)
      await containerObj.stop()
      await containerObj.remove()
      console.log(chalk.green('✅ 服务容器已停止'))
    } catch (error) {
      console.log(chalk.red(`❌ 停止容器失败: ${error.message}`))
    }
  } else if (!running) {
    console.log(chalk.gray('   服务未运行'))
  } else {
    // API 停止失败，尝试直接 kill 进程
    console.log(chalk.yellow('⚠️ API 停止失败，尝试强制终止进程...'))
    try {
      // 查找监听该端口的进程
      const result = execSync(`lsof -ti:${port} -sTCP:LISTEN 2>/dev/null || echo ""`, { encoding: 'utf8' })
      const pids = result.trim().split('\n').filter(p => p)
      if (pids.length > 0) {
        for (const pid of pids) {
          execSync(`kill -9 ${pid} 2>/dev/null || true`)
          console.log(chalk.gray(`   已终止进程 PID: ${pid}`))
        }
        console.log(chalk.green('✅ 进程已强制终止'))
      } else {
        console.log(chalk.yellow('⚠️ 无法找到监听端口的进程'))
        console.log(chalk.gray(`   提示: 手动终止端口 ${port} 上的进程`))
      }
    } catch (e) {
      console.log(chalk.yellow('⚠️ 无法停止服务'))
      console.log(chalk.gray(`   提示: 手动终止端口 ${port} 上的进程`))
    }
  }
}

/**
 * 获取状态
 */
export async function getStatus() {
  console.log()

  // 检查 npm 包版本
  console.log(chalk.bold('npm 包版本'))
  try {
    // __dirname 是 src/commands，需要向上两级到包根目录
    const pkgPath = path.resolve(__dirname, '../../package.json')
    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'))
    console.log(chalk.gray(`   @icyfenix-dmla/cli: ${pkg.version}`))
  } catch {
    console.log(chalk.gray('   版本信息不可用'))
  }

  console.log()

  // 检查镜像
  console.log(chalk.bold('Docker 镜像'))
  const cpuExists = await checkImageExists('cpu')
  const gpuExists = await checkImageExists('gpu')
  console.log(chalk.gray(`   CPU: ${cpuExists ? chalk.green('已安装') : chalk.red('未安装')}`))
  console.log(chalk.gray(`   GPU: ${gpuExists ? chalk.green('已安装') : chalk.red('未安装')}`))

  console.log()

  // 检查 GPU
  console.log(chalk.bold('GPU 状态'))
  const gpuAvailable = await checkGPUAvailable()
  if (gpuAvailable) {
    console.log(chalk.green('   GPU 可用'))
    try {
      const proc = spawn('nvidia-smi', ['-L'])
      proc.stdout.on('data', (data) => {
        const lines = data.toString().split('\n').filter(l => l.trim())
        lines.forEach(line => console.log(chalk.gray(`   ${line}`)))
      })
    } catch {}
  } else {
    console.log(chalk.gray('   GPU 不可用'))
  }

  console.log()

  // 检查服务
  console.log(chalk.bold('服务状态'))
  const running = await checkServiceRunning(CONFIG.defaultPort)
  if (running) {
    console.log(chalk.green(`   服务运行中 (端口 ${CONFIG.defaultPort})`))
    try {
      // 获取详细状态
      const healthUrl = `http://localhost:${CONFIG.defaultPort}/api/sandbox/health`
      http.get(healthUrl, (res) => {
        let data = ''
        res.on('data', (chunk) => data += chunk)
        res.on('end', () => {
          try {
            const health = JSON.parse(data)
            if (health.images) {
              console.log(chalk.gray(`   CPU 镜像: ${health.images.cpu ? '就绪' : '未就绪'}`))
              console.log(chalk.gray(`   GPU 镜像: ${health.images.gpu ? '就绪' : '未就绪'}`))
            }
          } catch {}
        })
      })
    } catch {}
  } else {
    console.log(chalk.gray('   服务未运行'))
    console.log(chalk.yellow('   提示: 运行 dmla start 启动服务'))
  }
}

// ==================== Native 模式 ====================

/**
 * 同步启动 Native 服务（在当前进程运行，用于调试）
 * @param {number} port - 服务端口
 */
export async function startNativeServerSync(port) {
  // 检查端口
  const portAvailable = await checkPortAvailable(port)
  if (!portAvailable) {
    console.log(chalk.red(`❌ 端口 ${port} 已被占用`))
    console.log(chalk.yellow('提示: 使用 --port 选项指定其他端口'))
    return
  }

  // 检查服务是否已运行
  const alreadyRunning = await checkServiceRunning(port)
  if (alreadyRunning) {
    console.log(chalk.green(`✅ 服务已在端口 ${port} 运行`))
    return
  }

  // 查找服务器入口文件（Native 模式使用相同的服务入口）
  const serverPath = findNativeServerPath()
  if (!serverPath) {
    console.log(chalk.red('❌ 找不到服务入口文件'))
    console.log(chalk.yellow('提示: 确保正确安装了 @icyfenix-dmla/cli'))
    return
  }

  console.log(chalk.gray('   同步模式启动...'))
  console.log(chalk.gray(`   服务入口: ${serverPath}`))
  console.log()

  // 设置环境变量（标记 Native 模式）
  process.env.PORT = port.toString()
  process.env.DMLA_MODE = 'native'
  process.env.DMLA_SYNC_MODE = 'true'

  // 动态 import 服务器模块
  try {
    const serverURL = pathToFileURL(serverPath).href
    await import(serverURL)
  } catch (error) {
    console.log(chalk.red(`❌ 服务启动失败: ${error.message}`))
    console.log(chalk.gray(error.stack))
  }
}

/**
 * 启动 Native 服务（异步模式，spawn 子进程）
 * @param {number} port - 服务端口
 */
export async function startNativeServer(port) {
  // 检查端口
  const portAvailable = await checkPortAvailable(port)
  if (!portAvailable) {
    console.log(chalk.red(`❌ 端口 ${port} 已被占用`))
    console.log(chalk.yellow('提示: 使用 --port 选项指定其他端口'))
    return
  }

  // 检查服务是否已运行
  const alreadyRunning = await checkServiceRunning(port)
  if (alreadyRunning) {
    console.log(chalk.green(`✅ 服务已在端口 ${port} 运行`))
    return
  }

  // 查找服务器入口文件
  const serverPath = findNativeServerPath()
  if (!serverPath) {
    console.log(chalk.red('❌ 找不到服务入口文件'))
    console.log(chalk.yellow('提示: 确保正确安装了 @icyfenix-dmla/cli'))
    return
  }

  console.log(chalk.gray('   正在启动...'))

  try {
    // 日志文件路径
    const logDir = path.resolve(__dirname, '../../logs')
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true })
    }
    const logFile = path.join(logDir, 'native-server.log')
    const errorLogFile = path.join(logDir, 'native-server-error.log')

    console.log(chalk.gray(`   日志文件: ${logFile}`))

    // 创建日志文件流
    const logStream = fs.openSync(logFile, 'a')
    const errorLogStream = fs.openSync(errorLogFile, 'a')

    const env = {
      ...process.env,
      PORT: port.toString(),
      DMLA_MODE: 'native',
      DMLA_LOG_FILE: logFile
    }

    // 写入启动日志
    const timestamp = new Date().toISOString()
    fs.writeSync(logStream, `[${timestamp}] Native Server starting...\n`)
    fs.writeSync(logStream, `[${timestamp}] Server path: ${serverPath}\n`)
    fs.writeSync(logStream, `[${timestamp}] Port: ${port}\n`)
    fs.writeSync(logStream, `[${timestamp}] Mode: native\n`)

    // 使用 spawn 启动 server 进程
    const serverProcess = spawn('node', [serverPath], {
      env,
      stdio: ['ignore', logStream, errorLogStream],
      detached: true,
      windowsHide: true
    })

    // 监听子进程事件
    serverProcess.on('error', (err) => {
      fs.writeSync(errorLogStream, `[${new Date().toISOString()}] Spawn error: ${err.message}\n`)
    })

    serverProcess.on('exit', (code, signal) => {
      const msg = `[${new Date().toISOString()}] Process exited: code=${code}, signal=${signal}\n`
      fs.writeSync(logStream, msg)
      fs.writeSync(errorLogStream, msg)
    })

    serverProcess.unref()

    // 关闭父进程中的文件描述符
    fs.closeSync(logStream)
    fs.closeSync(errorLogStream)

    // 等待服务启动
    console.log(chalk.gray('   等待服务就绪...'))
    let attempts = 0
    const maxAttempts = 30

    while (attempts < maxAttempts) {
      const running = await checkServiceRunning(port)
      if (running) {
        console.log(chalk.green(`✅ 服务已启动: http://localhost:${port}`))
        console.log(chalk.gray(`   健康检查: http://localhost:${port}/api/sandbox/health`))
        console.log(chalk.gray(`   日志查看: ${logFile}`))
        console.log(chalk.cyan('   模式: Native（本机执行）'))
        return
      }
      await new Promise(resolve => setTimeout(resolve, 500))
      attempts++
    }

    console.log(chalk.yellow('⚠️ 服务启动超时'))
    console.log(chalk.gray(`   请查看日志: ${logFile}`))
    console.log(chalk.gray(`   或使用 --sync 模式调试`))
  } catch (error) {
    console.log(chalk.red(`❌ 启动失败: ${error.message}`))
    console.log(chalk.gray(error.stack))
  }
}

/**
 * 查找 Native 服务入口文件
 */
function findNativeServerPath() {
  // Native 模式使用相同的服务入口，只是路由不同
  // packages/cli/src/commands -> ../server/index.js
  const serverPath = path.resolve(__dirname, '../server/index.js')

  if (fs.existsSync(serverPath)) {
    return serverPath
  }

  return null
}