/**
 * DMLA 安装 TUI 入口
 */
import chalk from 'chalk'
import pkg from 'enquirer'
const { prompt } = pkg
import net from 'net'
import path from 'path'
import { fileURLToPath } from 'url'
import fs from 'fs'
import { checkEnvironment } from './modules/environment.js'
import { pullImages } from './modules/docker.js'
import { installNpmPackage, verifyInstallation } from './modules/install.js'

// 导出供 CLI 的 images 命令使用
export { pullImages }

// 从 package.json 读取版本号
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const pkgPath = path.resolve(__dirname, '../package.json')
const VERSION = JSON.parse(fs.readFileSync(pkgPath, 'utf8')).version

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
 * 显示 Banner（右下角带版本号）
 */
function showBanner() {
  console.log()
  console.log(chalk.cyan(' ______   ____    ____  _____          _       '))
  console.log(chalk.cyan('|_   _ `.|_   \\  /   _||_   _|        / \\      '))
  console.log(chalk.cyan('  | | `. \\ |   \\/   |    | |         / _ \\     '))
  console.log(chalk.cyan('  | |  | | | |\\  /| |    | |   _    / ___ \\    '))
  console.log(chalk.cyan(' _| |_.\' /_| |_\\/_| |_  _| |__/ | _/ /   \\ \\_  '))
  console.log(chalk.cyan('|______.\'|_____||_____||________||____| |____| '))
  console.log(chalk.blue('== Designing Machine Learning Applications =='))
  console.log(chalk.blue('                                 '+ chalk.gray('v' + VERSION) ))
  console.log()
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

/**
 * 设置信号监听
 */
function setupSignalHandlers() {
  process.on('SIGINT', gracefulExit)

  process.on('uncaughtException', (err) => {
    if (err.code === 'ERR_USE_AFTER_CLOSE') {
      gracefulExit()
    } else {
      throw err
    }
  })
}

/**
 * 运行安装 TUI 主流程
 */
export async function runInstallTUI() {
  // 设置信号监听
  setupSignalHandlers()

  // 显示 Banner
  showBanner()

  try {
    // ─────────────────────────────────────────────────────────────
    // 步骤 1: 环境检测
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('环境检测'))
    console.log()

    const env = await checkEnvironment()
    let dockerAvailable = true

    if (!env.docker) {
      dockerAvailable = false
      console.log(chalk.yellow('⚠️  Docker 未安装或未运行'))
      console.log(chalk.gray('   CLI 工具可先安装，稍后在有 Docker 的环境中使用'))
      console.log(chalk.gray('   Docker 安装指南: https://docs.docker.com/get-docker/'))
    } else {
      console.log(chalk.green(`✔ Docker ${env.dockerVersion || ''} 已安装`))
    }

    if (!env.node) {
      console.log(chalk.red('❌ Node.js 未安装'))
      console.log(chalk.yellow('请先安装 Node.js: https://nodejs.org/'))
      process.exit(1)
    }

    console.log(chalk.green(`✔ Node.js ${env.nodeVersion} 已安装`))

    // Git 检测
    if (env.git) {
      console.log(chalk.green(`✔ Git ${env.gitVersion || ''} 已安装`))
    } else {
      console.log(chalk.yellow('⚠️  Git 未安装'))
      console.log(chalk.gray('   下载数据集功能需要 Git，建议安装'))
    }

    // Git LFS 检测
    if (env.gitLfs) {
      console.log(chalk.green(`✔ Git LFS ${env.gitLfsVersion || ''} 已安装`))
    } else {
      console.log(chalk.gray('   Git LFS: 未安装 (部分大型数据集可能需要)'))
    }

    // GPU 信息显示（仅当 Docker 可用时）
    if (dockerAvailable && env.gpu) {
      console.log(chalk.green(`✔ GPU: ${env.gpuInfo || '检测到'}`))

      // 显示驱动版本和 CUDA 兼容上限
      if (env.gpuDriverVersion) {
        console.log(chalk.gray(`   驱动版本: ${env.gpuDriverVersion}`))
      }
      if (env.gpuCudaVersion) {
        console.log(chalk.gray(`   CUDA 兼容上限: ${env.gpuCudaVersion}`))
      }

      // GPU 镜像兼容性检查（CUDA 12.8 需要驱动 >= 570）
      const minDriverForGpuImage = 570
      const driverVersion = parseFloat(env.gpuDriverVersion || '0')

      if (driverVersion >= minDriverForGpuImage) {
        console.log(chalk.green(`   GPU 镜像: 兼容 (驱动 >= ${minDriverForGpuImage})`))
      } else if (driverVersion > 0) {
        console.log(chalk.yellow(`   GPU 镜像: 不兼容 (需要驱动 >= ${minDriverForGpuImage})`))
        console.log(chalk.yellow(`   💡 建议：升级 NVIDIA 驱动到 ${minDriverForGpuImage}+ 或使用 CPU 模式`))
      }
    } else if (!dockerAvailable) {
      console.log(chalk.gray('   GPU: 检测已跳过（Docker 不可用）'))
    } else {
      console.log(chalk.gray('   GPU: 未检测到'))
    }

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 2: 选择镜像仓库（或跳过拉取）
    // ─────────────────────────────────────────────────────────────

    // Docker 不可用时，强制跳过镜像拉取
    let registry = 'skip'
    let skipPull = true

    if (!dockerAvailable) {
      console.log(chalk.bold('镜像安装'))
      console.log()
      console.log(chalk.gray('   Docker 未安装，将仅安装 CLI 工具'))
      console.log(chalk.gray('   稍后可在有 Docker 的环境中运行: dmla install'))
      console.log()
    } else {
      const registryChoice = await prompt({
        type: 'select',
        name: 'registry',
        message: '请选择镜像仓库',
        initial: 0,  // 默认选择 'auto'（第一项）
        choices: [
          { name: 'auto', message: '自动选择 (根据网络延迟)' },
          { name: 'dockerhub', message: 'Docker Hub (全球访问)' },
          { name: 'acr', message: '阿里云 ACR (国内加速)' },
          { name: 'skip', message: '暂不拉取镜像 (仅安装 CLI)' }
        ]
      })

      registry = registryChoice.registry
      skipPull = false

      if (registry === 'skip') {
        skipPull = true
        console.log(chalk.gray('   将仅安装 CLI 工具'))
      } else if (registry === 'auto') {
        console.log(chalk.gray('   检测网络延迟...'))
        registry = 'acr'
        console.log(chalk.gray(`   已选择: ${registry === 'acr' ? '阿里云 ACR' : 'Docker Hub'}`))
      }

      console.log()
    }

    // ─────────────────────────────────────────────────────────────
    // 步骤 3: 选择镜像类型（如果需要拉取）
    // ─────────────────────────────────────────────────────────────
    let imageTypes = []

    if (!skipPull) {
      const choices = [
        { name: 'auto', message: '自动选择 (根据环境)' },
        { name: 'all', message: '全部安装 (CPU + GPU)' },
        { name: 'cpu', message: '仅 CPU 版本 (~ 683MB)' },
        { name: 'gpu', message: '仅 GPU 版本 (~ 7.93GB)' }
      ]

      const typeChoice = await prompt({
        type: 'select',
        name: 'imageType',
        message: '请选择要安装的镜像',
        initial: 0,  // 默认选择第一项（自动选择）
        choices
      })

      const selectedType = typeChoice.imageType
      if (selectedType === 'auto') {
        // 根据环境自动选择
        imageTypes = env.gpu ? ['gpu'] : ['cpu']
        console.log(chalk.gray(`   已选择: ${env.gpu ? 'GPU 版本' : 'CPU 版本'}`))
      } else if (selectedType === 'all') {
        imageTypes = ['cpu', 'gpu']
      } else {
        imageTypes = [selectedType]
      }

      console.log()
    }

    // ─────────────────────────────────────────────────────────────
    // 步骤 4: 配置端口
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('配置服务端口'))
    console.log()

    const defaultPort = 3001
    let port = defaultPort

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
    let pullResults = {}

    if (skipPull) {
      console.log(chalk.bold('跳过镜像拉取'))
      console.log()
      console.log(chalk.gray('   将仅安装 CLI 工具，稍后可通过以下命令拉取镜像：'))
      console.log(chalk.cyan('   dmla install'))
      console.log()
    } else {
      console.log(chalk.bold('拉取 Docker 镜像'))
      console.log()

      pullResults = await pullImages(imageTypes, registry)

      // 检查是否有失败的镜像
      const failedImages = Object.entries(pullResults)
        .filter(([type, result]) => !result.success)
        .map(([type, result]) => ({ type, ...result }))

      if (failedImages.length > 0) {
        console.log(chalk.yellow('⚠️  部分镜像拉取失败'))
        console.log()

        // 检查是否有可用的镜像
        const availableImages = Object.entries(pullResults)
          .filter(([type, result]) => result.success)
          .map(([type]) => type)

        if (availableImages.length > 0) {
          console.log(chalk.green(`✔ 已成功拉取镜像: ${availableImages.join(', ')}`))
          console.log(chalk.gray('   您可以使用这些镜像启动服务'))
        }

        console.log()
        console.log(chalk.yellow('手动拉取镜像命令：'))
        for (const failed of failedImages) {
          console.log(chalk.cyan(`   docker pull ${failed.manualCommand}`))
          console.log(chalk.gray(`   docker tag ${failed.manualCommand} dmla-sandbox:${failed.type}`))
        }
        console.log()

        // 询问是否继续
        const continueChoice = await prompt({
          type: 'select',
          name: 'action',
          message: '是否继续安装 CLI？',
          choices: [
            { name: 'continue', message: '继续安装（稍后手动拉取镜像）' },
            { name: 'exit', message: '退出安装' }
          ]
        })

        if (continueChoice.action === 'exit') {
          gracefulExit()
          return
        }

        console.log(chalk.gray('继续安装...'))
        console.log()
      } else {
        // 全部成功
        const successCount = Object.keys(pullResults).length
        console.log(chalk.green(`✔ ${successCount} 个镜像已准备就绪`))
        console.log()
      }
    }

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

    // 如果跳过了镜像拉取，不询问是否启动服务
    if (!skipPull) {
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
        // 根据安装的镜像类型决定是否使用 GPU
        const useGpu = imageTypes.includes('gpu')
        await verifyInstallation(port, useGpu)
      }
    }

    // ─────────────────────────────────────────────────────────────
    // 完成
    // ─────────────────────────────────────────────────────────────
    showBanner()
    console.log(chalk.green('DMLA Sandbox 安装成功'))
    console.log(chalk.gray('常用命令:'))
    console.log(chalk.gray('  dmla start        启动服务'))
    console.log(chalk.gray('  dmla status       查看状态'))
    console.log(chalk.gray('  dmla install      安装/更新镜像'))
    console.log(chalk.gray('  dmla doctor       环境诊断'))
    console.log()
    console.log(chalk.gray(`服务地址: http://localhost:${port}`))
    console.log(chalk.gray(`健康检查: http://localhost:${port}/api/health`))
    console.log()

  } catch (error) {
    if (error.message && error.message.includes('cancel')) {
      gracefulExit()
      return
    }
    console.log()
    console.log(chalk.red(`❌ 安装失败: ${error.message}`))
    console.log(chalk.yellow('请运行 dmla doctor 检查环境'))
    process.exit(1)
  }
}

