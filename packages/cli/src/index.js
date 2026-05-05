/**
 * DMLA CLI 入口
 * 沙箱服务命令行管理工具
 */
import { program, Help } from 'commander'
import chalk from 'chalk'
import path from 'path'
import { fileURLToPath } from 'url'
import fs from 'fs'
import { startServer, startServerSync, stopServer, getStatus } from './commands/server.js'
import { runDoctor } from './commands/manage.js'
import { runDataTUI, runDataCommand } from './commands/data.js'
import { runImagesTUI } from './commands/images.js'
import { runUpdate } from './commands/update.js'

// 从 package.json 读取版本号
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const pkgPath = path.resolve(__dirname, '../package.json')
const VERSION = JSON.parse(fs.readFileSync(pkgPath, 'utf8')).version

// 重写 Help 类的方法以输出中文标题
Help.prototype.padWidth = function(cmd, helper) {
  return 20
}

// 重写帮助信息格式化方法
Help.prototype.formatHelp = function(cmd, helper) {
  const indent = '  '
  const itemIndent = '  '

  let output = []

  // 用法（中文）
  output.push('用法:')
  output.push(indent + helper.commandUsage(cmd))

  // 说明（中文）
  if (cmd.description()) {
    output.push('')
    output.push('说明:')
    output.push(indent + cmd.description())
  }

  // 参数（中文）
  const args = helper.visibleArguments(cmd)
  if (args.length > 0) {
    output.push('')
    output.push('参数:')
    args.forEach(arg => {
      output.push(itemIndent + arg.name())
    })
  }

  // 选项（中文）
  const options = helper.visibleOptions(cmd)
  if (options.length > 0) {
    output.push('')
    output.push('选项:')
    options.forEach(opt => {
      const term = helper.optionTerm(opt)
      const description = helper.optionDescription(opt)
      output.push(itemIndent + term.padEnd(20) + description)
    })
  }

  // 命令（中文）
  const commands = helper.visibleCommands(cmd)
  if (commands.length > 0) {
    output.push('')
    output.push('命令:')
    commands.forEach(subcmd => {
      const term = helper.subcommandTerm(subcmd)
      const description = helper.subcommandDescription(subcmd)
      output.push(itemIndent + term.padEnd(20) + description)
    })
  }

  return output.join('\n')
}

program
  .name('dmla')
  .description('DMLA 沙箱服务命令行管理工具')
  .version(VERSION, '-v, --version', '显示版本号')
  .helpOption('-h, --help', '显示帮助信息')
  .addHelpCommand('help [command]', '显示命令帮助信息')

// ─────────────────────────────────────────────────────────────
// start 命令
// ─────────────────────────────────────────────────────────────
program
  .command('start')
  .description('启动沙箱服务')
  .option('-p, --port <number>', '服务端口', '3001')
  .option('--gpu', '使用 GPU 镜像')
  .option('--sync', '同步模式：在当前进程运行，日志直接输出（用于调试）')
  .option('--dev', '开发模式：挂载本地代码到容器，无需重建镜像')
  .option('--shm-size <size>', 'Docker 共享内存大小（MB），用于 DataLoader 多线程。GPU 模式默认 1024MB，CPU 模式默认 64MB')
  .action(async (options) => {
    const port = parseInt(options.port, 10)
    const useGpu = options.gpu
    const sync = options.sync
    const dev = options.dev
    // GPU 模式默认 1GB shm，CPU 模式默认 64MB
    const defaultShm = useGpu ? 1024 : 64
    const shmSize = options.shmSize ? parseInt(options.shmSize, 10) : defaultShm

    console.log(chalk.blue('启动 DMLA 沙箱服务...'))
    console.log(chalk.gray(`   端口: ${port}`))
    console.log(chalk.gray(`   请求类型: ${useGpu ? 'GPU' : '自动选择'}`))
    console.log(chalk.gray(`   共享内存: ${shmSize}MB（DataLoader 多线程需要足够 shm）`))
    if (sync) {
      console.log(chalk.yellow(`   模式: 同步（调试模式）`))
    }
    if (dev) {
      console.log(chalk.cyan(`   模式: 开发（挂载本地代码）`))
    }

    if (sync) {
      await startServerSync(port, useGpu, dev, shmSize)
    } else {
      await startServer(port, useGpu, dev, shmSize)
    }
  })

// ─────────────────────────────────────────────────────────────
// stop 命令
// ─────────────────────────────────────────────────────────────
program
  .command('stop')
  .description('停止运行中的沙箱服务')
  .action(async () => {
    console.log(chalk.blue('停止 DMLA 沙箱服务...'))
    await stopServer()
  })

// ─────────────────────────────────────────────────────────────
// status 命令
// ─────────────────────────────────────────────────────────────
program
  .command('status')
  .description('查看服务状态')
  .action(async () => {
    console.log(chalk.blue('DMLA 沙箱服务状态'))
    await getStatus()
  })

// ─────────────────────────────────────────────────────────────
// images 命令
// ─────────────────────────────────────────────────────────────
program
  .command('images')
  .description('拉取 Docker 镜像')
  .action(async () => {
    await runImagesTUI()
  })

// ─────────────────────────────────────────────────────────────
// doctor 命令
// ─────────────────────────────────────────────────────────────
program
  .command('doctor')
  .description('诊断安装环境')
  .action(async () => {
    console.log(chalk.blue('DMLA 环境诊断'))
    await runDoctor()
  })

// ─────────────────────────────────────────────────────────────
// data 命令
// ─────────────────────────────────────────────────────────────
program
  .command('data [subcommand]')
  .description('数据管理（挂载、下载、清理等）')
  .option('-p, --path <path>', '设置挂载路径')
  .action(async (subcommand, options) => {
    if (subcommand) {
      await runDataCommand(subcommand, options)
    } else {
      await runDataTUI()
    }
  })

// ─────────────────────────────────────────────────────────────
// update 命令
// ─────────────────────────────────────────────────────────────
program
  .command('update')
  .description('更新程序（npm 自动更新）')
  .action(async () => {
    await runUpdate()
  })

program.parse()