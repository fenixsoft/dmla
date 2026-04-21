/**
 * npm 包安装和验证模块
 */
import chalk from 'chalk'
import { execSync, spawn } from 'child_process'
import http from 'http'

/**
 * 安装 npm 包
 */
export async function installNpmPackage() {
  console.log(chalk.gray('执行 npm install -g @icyfenix-dmla/cli...'))

  try {
    execSync('npm install -g @icyfenix-dmla/cli', { stdio: 'inherit' })
    console.log(chalk.green('✔ npm 包安装完成'))
  } catch (error) {
    console.log(chalk.yellow('⚠️ npm 包安装失败'))
    console.log(chalk.yellow('请手动执行: npm install -g @icyfenix-dmla/cli'))
  }

  // 验证命令可用
  console.log(chalk.gray('验证 dmla 命令...'))
  try {
    const version = execSync('dmla --version', { encoding: 'utf8' }).trim()
    console.log(chalk.green(`✔ @icyfenix-dmla/cli ${version} 已安装`))
  } catch {
    console.log(chalk.yellow('⚠️ dmla 命令暂不可用，可能需要重新打开终端'))
  }
}

/**
 * 验证安装并启动服务
 */
export async function verifyInstallation(port = 3001, useGpu = false) {
  console.log(chalk.gray('启动服务...'))

  // 检查 dmla 命令是否可用
  let dmlaAvailable = false
  try {
    execSync('dmla --version', { encoding: 'utf8' })
    dmlaAvailable = true
  } catch {
    // dmla 命令暂不可用，使用 npx 启动
    console.log(chalk.gray('dmla 命令暂不可用，使用 npx 启动...'))
  }

  // 构建启动参数
  const gpuArg = useGpu ? ['--gpu'] : []
  const args = dmlaAvailable
    ? ['start', '--port', port.toString(), ...gpuArg]
    : ['@icyfenix-dmla/cli', 'start', '--port', port.toString(), ...gpuArg]

  const cmd = dmlaAvailable ? 'dmla' : 'npx'

  // Windows 需要 shell: true，Linux/macOS 不需要
  const isWindows = process.platform === 'win32'
  const spawnOptions = {
    stdio: 'inherit',
    ...(isWindows ? { shell: true } : {})
  }

  const serverProcess = spawn(cmd, args, spawnOptions)

  // 处理启动错误
  serverProcess.on('error', (err) => {
    console.log(chalk.yellow('⚠️ 服务启动失败'))
    console.log(chalk.yellow(`💡 请手动执行: ${cmd} ${args.join(' ')}`))
    return false
  })

  // 等待服务就绪
  console.log(chalk.gray('等待服务就绪...'))

  let attempts = 0
  const maxAttempts = 30

  while (attempts < maxAttempts) {
    try {
      const result = await checkHealth(port)
      if (result) {
        console.log(chalk.green(`✔ 服务已启动: http://localhost:${port}`))
        return true
      }
    } catch {}

    await new Promise(resolve => setTimeout(resolve, 500))
    attempts++
  }

  console.log(chalk.yellow('⚠️ 服务启动超时'))
  console.log(chalk.yellow(`💡 请手动执行: ${cmd} ${args.join(' ')}`))
  return false
}

/**
 * 健康检查
 */
async function checkHealth(port) {
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: 'localhost',
      port: port,
      path: '/api/health',
      method: 'GET',
      timeout: 2000
    }, (res) => {
      if (res.statusCode === 200) {
        resolve(true)
      } else {
        reject(new Error('Health check failed'))
      }
    })
    req.on('error', reject)
    req.on('timeout', () => {
      req.destroy()
      reject(new Error('Timeout'))
    })
    req.end()
  })
}