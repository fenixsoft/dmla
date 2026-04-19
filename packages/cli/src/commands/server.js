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
 * 同步启动服务（在当前进程运行，用于调试）
 * @param {number} port - 服务端口
 * @param {boolean} useGpu - 是否使用 GPU
 */
export async function startServerSync(port, useGpu = false) {
  // 检查端口
  const portAvailable = await checkPortAvailable(port)
  if (!portAvailable) {
    console.log(chalk.red(`❌ 端口 ${port} 已被占用`))
    console.log(chalk.yellow('💡 提示: 使用 --port 选项指定其他端口'))
    return
  }

  // 检查镜像
  const imageType = useGpu ? 'gpu' : 'cpu'
  const imageExists = await checkImageExists(imageType)
  if (!imageExists) {
    console.log(chalk.red(`❌ 镜像 ${useGpu ? CONFIG.imageGpu : CONFIG.imageCpu} 不存在`))
    console.log(chalk.yellow('💡 提示: 运行 dmla install 安装镜像'))
    return
  }

  // 检查服务是否已运行
  const alreadyRunning = await checkServiceRunning(port)
  if (alreadyRunning) {
    console.log(chalk.green(`✅ 服务已在端口 ${port} 运行`))
    return
  }

  // 查找服务器入口
  const actualServerPath = findServerPath()
  if (!actualServerPath) {
    console.log(chalk.red('❌ 找不到服务入口文件'))
    console.log(chalk.yellow('💡 提示: 确保正确安装了 @icyfenix-dmla/cli'))
    return
  }

  console.log(chalk.gray('   同步模式启动...'))
  console.log(chalk.gray(`   服务入口: ${actualServerPath}`))
  console.log()

  // 设置环境变量
  process.env.PORT = port.toString()
  process.env.USE_GPU = useGpu ? 'true' : 'false'
  process.env.DMLA_SYNC_MODE = 'true'  // 标记同步模式，让服务器在 import 时启动

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
 */
export async function startServer(port, useGpu = false) {
  // 检查端口
  const portAvailable = await checkPortAvailable(port)
  if (!portAvailable) {
    console.log(chalk.red(`❌ 端口 ${port} 已被占用`))
    console.log(chalk.yellow('💡 提示: 使用 --port 选项指定其他端口'))
    return
  }

  // 检查镜像
  const imageType = useGpu ? 'gpu' : 'cpu'
  const imageExists = await checkImageExists(imageType)
  if (!imageExists) {
    console.log(chalk.red(`❌ 镜像 ${useGpu ? CONFIG.imageGpu : CONFIG.imageCpu} 不存在`))
    console.log(chalk.yellow('💡 提示: 运行 dmla install 安装镜像'))
    return
  }

  // 检查服务是否已运行
  const alreadyRunning = await checkServiceRunning(port)
  if (alreadyRunning) {
    console.log(chalk.green(`✅ 服务已在端口 ${port} 运行`))
    return
  }

  // 启动服务
  console.log(chalk.gray('   正在启动...'))

  try {
    const actualServerPath = findServerPath()

    if (!actualServerPath) {
      console.log(chalk.red('❌ 找不到服务入口文件'))
      console.log(chalk.yellow('💡 提示: 确保正确安装了 @icyfenix-dmla/cli'))
      return
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
      USE_GPU: useGpu ? 'true' : 'false',
      DMLA_LOG_FILE: logFile  // 传递日志文件路径给服务端
    }

    // 写入启动日志
    const timestamp = new Date().toISOString()
    fs.writeSync(logStream, `[${timestamp}] Server starting...\n`)
    fs.writeSync(logStream, `[${timestamp}] Server path: ${actualServerPath}\n`)
    fs.writeSync(logStream, `[${timestamp}] Port: ${port}\n`)
    fs.writeSync(logStream, `[${timestamp}] GPU: ${useGpu}\n`)

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
 */
export async function stopServer() {
  // 首先尝试通过 API 停止服务
  const port = CONFIG.defaultPort
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
    console.log(chalk.yellow('⚠️ 无法停止服务'))
    console.log(chalk.gray('   提示: 手动终止端口 3001 上的进程'))
  }
}

/**
 * 获取状态
 */
export async function getStatus() {
  console.log()

  // 检查 npm 包版本
  console.log(chalk.bold('📦 npm 包版本'))
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
  console.log(chalk.bold('🖼️  Docker 镜像'))
  const cpuExists = await checkImageExists('cpu')
  const gpuExists = await checkImageExists('gpu')
  console.log(chalk.gray(`   CPU: ${cpuExists ? chalk.green('已安装') : chalk.red('未安装')}`))
  console.log(chalk.gray(`   GPU: ${gpuExists ? chalk.green('已安装') : chalk.red('未安装')}`))

  console.log()

  // 检查 GPU
  console.log(chalk.bold('🎮 GPU 状态'))
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
  console.log(chalk.bold('🚀 服务状态'))
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