/**
 * DMLA CLI 入口
 * 沙箱服务命令行管理工具
 */
import { program } from 'commander'
import chalk from 'chalk'
import { startServer, stopServer, getStatus } from './commands/server.js'
import { installImages, updateAll, runDoctor } from './commands/manage.js'

const VERSION = '0.0.0' // 将在发布时由 workflow 更新

program
  .name('dmla')
  .description('DMLA 沙箱服务命令行管理工具')
  .version(VERSION)

// ─────────────────────────────────────────────────────────────
// start 命令
// ─────────────────────────────────────────────────────────────
program
  .command('start')
  .description('启动沙箱服务')
  .option('-p, --port <number>', '服务端口', '3001')
  .option('--gpu', '使用 GPU 镜像')
  .action(async (options) => {
    const port = parseInt(options.port, 10)
    const useGpu = options.gpu
    console.log(chalk.blue('🚀 启动 DMLA 沙箱服务...'))
    console.log(chalk.gray(`   端口: ${port}`))
    console.log(chalk.gray(`   镜像: ${useGpu ? 'GPU' : 'CPU'}`))
    await startServer(port, useGpu)
  })

// ─────────────────────────────────────────────────────────────
// stop 命令
// ─────────────────────────────────────────────────────────────
program
  .command('stop')
  .description('停止运行中的沙箱服务')
  .action(async () => {
    console.log(chalk.blue('🛑 停止 DMLA 沙箱服务...'))
    await stopServer()
  })

// ─────────────────────────────────────────────────────────────
// status 命令
// ─────────────────────────────────────────────────────────────
program
  .command('status')
  .description('查看服务状态')
  .action(async () => {
    console.log(chalk.blue('📊 DMLA 沙箱服务状态'))
    await getStatus()
  })

// ─────────────────────────────────────────────────────────────
// install 命令
// ─────────────────────────────────────────────────────────────
program
  .command('install')
  .description('安装 Docker 镜像')
  .option('--cpu', '仅安装 CPU 版本')
  .option('--gpu', '仅安装 GPU 版本')
  .option('--all', '安装所有镜像（默认）')
  .option('-r, --registry <type>', '镜像仓库 (dockerhub/acr)', 'dockerhub')
  .action(async (options) => {
    const registry = options.registry
    let types = []

    if (options.cpu) types.push('cpu')
    if (options.gpu) types.push('gpu')
    if (types.length === 0 || options.all) types = ['cpu', 'gpu']

    console.log(chalk.blue('📦 安装 DMLA Docker 镜像...'))
    console.log(chalk.gray(`   仓库: ${registry}`))
    console.log(chalk.gray(`   类型: ${types.join(', ')}`))

    await installImages(types, registry)
  })

// ─────────────────────────────────────────────────────────────
// update 命令
// ─────────────────────────────────────────────────────────────
program
  .command('update')
  .description('更新 npm 包和 Docker 镜像')
  .option('-r, --registry <type>', '镜像仓库 (dockerhub/acr)', 'dockerhub')
  .action(async (options) => {
    console.log(chalk.blue('🔄 更新 DMLA...'))
    await updateAll(options.registry)
  })

// ─────────────────────────────────────────────────────────────
// doctor 命令
// ─────────────────────────────────────────────────────────────
program
  .command('doctor')
  .description('诊断安装环境')
  .action(async () => {
    console.log(chalk.blue('🔍 DMLA 环境诊断'))
    await runDoctor()
  })

program.parse()