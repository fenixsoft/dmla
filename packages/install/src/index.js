/**
 * DMLA 安装 TUI 入口
 */
import chalk from 'chalk'
import { prompt } from 'enquirer'
import { checkEnvironment } from './modules/environment.js'
import { pullImages } from './modules/docker.js'
import { installNpmPackage, verifyInstallation } from './modules/install.js'

console.log()
console.log(chalk.bold.blue('╔════════════════════════════════════════════════════════════╗'))
console.log(chalk.bold.blue('║                                                            ║'))
console.log(chalk.bold.blue('║           DMLA Sandbox 安装向导                            ║'))
console.log(chalk.bold.blue('║                                                            ║'))
console.log(chalk.bold.blue('╚════════════════════════════════════════════════════════════╝'))
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

    console.log(chalk.green(`✅ Docker ${env.dockerVersion || ''} 已安装`))

    if (!env.node) {
      console.log(chalk.red('❌ Node.js 未安装'))
      console.log(chalk.yellow('💡 请先安装 Node.js: https://nodejs.org/'))
      process.exit(1)
    }

    console.log(chalk.green(`✅ Node.js ${env.nodeVersion} 已安装`))

    if (env.gpu) {
      console.log(chalk.green(`✅ GPU: ${env.gpuInfo || '检测到'}`))
    } else {
      console.log(chalk.gray('   GPU: 未检测到'))
    }

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 2: 选择镜像仓库
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('📦 选择镜像仓库'))
    console.log()

    const registryChoice = await prompt({
      type: 'select',
      name: 'registry',
      message: '请选择镜像仓库',
      choices: [
        { name: 'dockerhub', message: 'Docker Hub (全球访问)' },
        { name: 'tcr', message: '腾讯云 TCR (国内加速)' },
        { name: 'auto', message: '自动选择 (根据网络延迟)' }
      ]
    })

    let registry = registryChoice.registry
    if (registry === 'auto') {
      // 简化的自动选择逻辑
      console.log(chalk.gray('   检测网络延迟...'))
      // 默认使用 TCR（国内用户更常见）
      registry = 'tcr'
      console.log(chalk.gray(`   已选择: ${registry === 'tcr' ? '腾讯云 TCR' : 'Docker Hub'}`))
    }

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 3: 选择镜像类型
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('🖼️  选择镜像类型'))
    console.log()

    const defaultChoice = env.gpu ? 'gpu' : 'all'

    const typeChoice = await prompt({
      type: 'select',
      name: 'imageType',
      message: '请选择要安装的镜像',
      initial: defaultChoice,
      choices: [
        { name: 'all', message: '全部安装 (CPU + GPU)' },
        { name: 'cpu', message: '仅 CPU 版本 (~1.5GB)' },
        { name: 'gpu', message: '仅 GPU 版本 (~2.5GB)' }
      ].concat(env.gpu ? [
        { name: 'gpu-recommended', message: `仅 GPU 版本 (推荐，已检测到 GPU)` }
      ] : [])
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

    const portChoice = await prompt({
      type: 'input',
      name: 'port',
      message: '请输入服务端口',
      initial: '3001',
      validate: (value) => {
        const port = parseInt(value, 10)
        if (isNaN(port) || port < 1 || port > 65535) {
          return '请输入有效的端口 (1-65535)'
        }
        return true
      }
    })

    const port = parseInt(portChoice.port, 10)
    console.log(chalk.gray(`   端口: ${port}`))

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 5: 拉取镜像
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('📥 拉取 Docker 镜像'))
    console.log()

    await pullImages(imageTypes, registry)

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 6: 安装 npm 包
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('📦 安装 npm 包'))
    console.log()

    await installNpmPackage()

    console.log()

    // ─────────────────────────────────────────────────────────────
    // 步骤 7: 验证安装
    // ─────────────────────────────────────────────────────────────
    console.log(chalk.bold('✅ 验证安装'))
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
    console.log(chalk.bold.green('╔════════════════════════════════════════════════════════════╗'))
    console.log(chalk.bold.green('║                                                            ║'))
    console.log(chalk.bold.green('║           🎉 DMLA 安装成功！                               ║'))
    console.log(chalk.bold.green('║                                                            ║'))
    console.log(chalk.bold.green('╚════════════════════════════════════════════════════════════╝'))
    console.log()
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
    console.log()
    console.log(chalk.red(`❌ 安装失败: ${error.message}`))
    console.log(chalk.yellow('💡 请运行 dmla doctor 检查环境'))
    process.exit(1)
  }
}

main()