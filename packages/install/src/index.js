/**
 * DMLA 安装 TUI 入口
 */
import chalk from 'chalk'
import pkg from 'enquirer'
const { prompt } = pkg
import net from 'net'
import { checkEnvironment } from './modules/environment.js'
import { pullImages } from './modules/docker.js'
import { installNpmPackage, verifyInstallation } from './modules/install.js'

/**
 * 检测端口是否可用
 */
function isPortAvailable(port) {
  return new Promise((resolve) => {
    const server = net.createServer()
    server.once('error', () => resolve(false))
    server.once('listening', () => {
      server.close()
      resolve(true)
    })
    server.listen(port)
  })
}

/**
 * 显示退出信息并终止程序
 */
function gracefulExit() {
  console.log()
  console.log(chalk.yellow('安装已取消'))
  console.log(chalk.gray('您可以稍后重新运行安装程序'))
  console.log()
  process.exit(0)
}

// 监听 Ctrl+C 信号
process.on('SIGINT', gracefulExit)

// 监听 enquirer 的取消事件
process.on('uncaughtException', (err) => {
  if (err.code === 'ERR_USE_AFTER_CLOSE') {
    gracefulExit()
  } else {
    throw err
  }
})

console.log()
console.log(chalk.cyan(' ______   ____    ____  _____          _       '))
console.log(chalk.cyan('|_   _ `.|_   \\  /   _||_   _|        / \\      '))
console.log(chalk.cyan('  | | `. \\ |   \\/   |    | |         / _ \\     '))
console.log(chalk.cyan('  | |  | | | |\\  /| |    | |   _    / ___ \\    '))
console.log(chalk.cyan(' _| |_.\' /_| |_\\/_| |_  _| |__/ | _/ /   \\ \\_  '))
console.log(chalk.cyan('|______.\'|_____||_____||________||____| |____| '))
console.log(chalk.blue('== Designing Machine Learning Applications =='))
console.log()

async function main() {
  try {
    // ─────────────────────────────────────────────────────────────
    // 步骤 1: 环境检测
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('🔍 环境检测'))
    console.log()

    const env = await checkEnvironment()

    if (!env.docker) {
      console.log(chalk.red('❌ Docker 未安装或未运行'))
      console.log(chalk.yellow('💡 请先安装 Docker: https://docs.docker.com/get-docker/'))
      process.exit(1)
    }

    console.log(chalk.green(`✔ Docker ${env.dockerVersion || ''} 已安装`))

    if (!env.node) {
      console.log(chalk.red('❌ Node.js 未安装'))
      console.log(chalk.yellow('💡 请先安装 Node.js: https://nodejs.org/'))
      process.exit(1)
    }

    console.log(chalk.green(`✔ Node.js ${env.nodeVersion} 已安装`))

    if (env.gpu) {
      console.log(chalk.green(`✔ GPU: ${env.gpuInfo || '检测到'}`))
    } else {
      console.log(chalk.gray('   GPU: 未检测到'))
    }

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 2: 选择镜像仓库
    // ─────────────────────────────────────────────────────────────
    const registryChoice = await prompt({
      type: 'select',
      name: 'registry',
      message: '请选择镜像仓库',
      initial: 2,  // 默认选择 'auto'
      choices: [
        { name: 'dockerhub', message: 'Docker Hub (全球访问)' },
        { name: 'acr', message: '阿里云 ACR (国内加速)' },
        { name: 'auto', message: '自动选择 (根据网络延迟)' }
      ]
    })

    let registry = registryChoice.registry
    if (registry === 'auto') {
      // 简化的自动选择逻辑
      console.log(chalk.gray('   检测网络延迟...'))
      // 默认使用 ACR（国内用户更常见）
      registry = 'acr'
      console.log(chalk.gray(`   已选择: ${registry === 'acr' ? '阿里云 ACR' : 'Docker Hub'}`))
    }

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 3: 选择镜像类型
    // ─────────────────────────────────────────────────────────────
    const choices = [
      { name: 'all', message: '全部安装 (CPU + GPU)' },
      { name: 'cpu', message: '仅 CPU 版本 (~1.5GB)' },
      { name: 'gpu', message: '仅 GPU 版本 (~2.5GB)' }
    ]

    // 如果检测到GPU，添加推荐选项并设为默认
    if (env.gpu) {
      choices.push({ name: 'gpu-recommended', message: `仅 GPU 版本 (推荐，已检测到 GPU)` })
    }

    const typeChoice = await prompt({
      type: 'select',
      name: 'imageType',
      message: '请选择要安装的镜像',
      initial: env.gpu ? 3 : 1,  // 有GPU默认选推荐项，无GPU默认选CPU
      choices
    })

    let imageTypes = []
    const selectedType = typeChoice.imageType
    if (selectedType === 'all') imageTypes = ['cpu', 'gpu']
    else if (selectedType === 'gpu-recommended') imageTypes = ['gpu']
    else imageTypes = [selectedType]

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 4: 配置端口
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('🔌 配置服务端口'))
    console.log()

    const defaultPort = 3001
    let port = defaultPort

    // 检测默认端口是否可用
    const portAvailable = await isPortAvailable(defaultPort)

    if (portAvailable) {
      console.log(chalk.green(`✔ 端口 ${defaultPort} 可用，将使用默认端口`))
    } else {
      console.log(chalk.yellow(`⚠️  端口 ${defaultPort} 已被占用`))

      const portChoice = await prompt({
        type: 'input',
        name: 'port',
        message: '请输入服务端口',
        initial: '3002',
        validate: (value) => {
          const p = parseInt(value, 10)
          if (isNaN(p) || p < 1 || p > 65535) {
            return '请输入有效的端口 (1-65535)'
          }
          return true
        }
      })

      port = parseInt(portChoice.port, 10)
    }

    console.log(chalk.gray(`   端口: ${port}`))

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 5: 拉取镜像
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('拉取 Docker 镜像'))
    console.log()

    await pullImages(imageTypes, registry)

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 6: 安装 npm 包
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('安装 npm 包'))
    console.log()

    await installNpmPackage()

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 7: 验证安装
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('✔ 验证安装'))
    console.log()

    const startNow = await prompt({
      type: 'select',
      name: 'start',
      message: '安装完成！是否立即启动服务？',
      choices: [
        { name: 'yes', message: '是，立即启动' },
        { name: 'no', message: '否，稍后手动启动' }
      ]
    })

    if (startNow.start === 'yes') {
      await verifyInstallation(port)
    }

    // ─────────────────────────────────────────────────────────────
    // 完成
    // ─────────────────────────────────────────────────────────────
    console.log()
    console.log(chalk.cyan(' ______   ____    ____  _____          _       '))
    console.log(chalk.cyan('|_   _ `.|_   \\  /   _||_   _|        / \\      '))
    console.log(chalk.cyan('  | | `. \\ |   \\/   |    | |         / _ \\     '))
    console.log(chalk.cyan('  | |  | | | |\\  /| |    | |   _    / ___ \\    '))
    console.log(chalk.cyan(' _| |_.\' /_| |_\\/_| |_  _| |__/ | _/ /   \\ \\_  '))
    console.log(chalk.cyan('|______.\'|_____||_____||________||____| |____| '))
    console.log(chalk.blue('== Designing Machine Learning Applications =='))
    console.log()
    console.log(chalk.green('DMLA Sandbox 安装成功'))
    console.log(chalk.gray('常用命令:'))
    console.log(chalk.gray('  dmla start      启动服务'))
    console.log(chalk.gray('  dmla status     查看状态'))
    console.log(chalk.gray('  dmla update     更新版本'))
    console.log(chalk.gray('  dmla doctor     环境诊断'))
    console.log()
    console.log(chalk.gray(`服务地址: http://localhost:${port}`))
    console.log(chalk.gray(`健康检查: http://localhost:${port}/api/health`))
    console.log()

  } catch (error) {
    // 用户取消安装（Ctrl+C）
    if (error.message && error.message.includes('cancel')) {
      gracefulExit()
      return
    }
    console.log()
    console.log(chalk.red(`❌ 安装失败: ${error.message}`))
    console.log(chalk.yellow('💡 请运行 dmla doctor 检查环境'))
    process.exit(1)
  }
}

main()