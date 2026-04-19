/**
 * DMLA CLI 入口
 * 沙箱服务命令行管理工具
 */
import { program } from 'commander'
import chalk from 'chalk'
import { startServer, startServerSync, stopServer, getStatus } from './commands/server.js'
import { updateAll, runDoctor } from './commands/manage.js'
import { runInstallTUI } from '@icyfenix-dmla/install'

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
  .option('--sync', '同步模式：在当前进程运行，日志直接输出（用于调试）')
  .action(async (options) => {
    const port = parseInt(options.port, 10)
    const useGpu = options.gpu
    const sync = options.sync

    console.log(chalk.blue('🚀 启动 DMLA 沙箱服务...'))
    console.log(chalk.gray(`   端口: ${port}`))
    console.log(chalk.gray(`   镜像: ${useGpu ? 'GPU' : 'CPU'}`))
    if (sync) {
      console.log(chalk.yellow(`   模式: 同步（调试模式）`))
    }

    if (sync) {
      await startServerSync(port, useGpu)
    } else {
      await startServer(port, useGpu)
    }
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
  .description('启动安装向导')
  .action(async () => {
    await runInstallTUI()
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