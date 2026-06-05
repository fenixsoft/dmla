/**
 * DMLA CLI update 命令
 * 通过 npm 更新程序
 */
import chalk from 'chalk'
import { execSync } from '../verbose.js'
import path from 'path'
import { fileURLToPath } from 'url'
import fs from 'fs'

/**
 * 运行 update 命令
 */
export async function runUpdate() {
  console.log(chalk.blue('更新 DMLA...'))
  console.log()

  try {
    // 直接执行 npm 更新
    console.log(chalk.cyan('正在从 npm 更新...'))
    execSync('npm update -g @icyfenix-dmla/cli @icyfenix-dmla/install', {
      encoding: 'utf-8',
      stdio: 'inherit'
    })

    console.log()
    console.log(chalk.green('✓ 更新完成'))

    // 重新读取版本号（npm update 后 package.json 已更新）
    const __filename = fileURLToPath(import.meta.url)
    const __dirname = path.dirname(__filename)
    const pkgPath = path.resolve(__dirname, '../package.json')
    const newVersion = JSON.parse(fs.readFileSync(pkgPath, 'utf8')).version
    console.log(chalk.cyan(`当前版本: ${newVersion}`))

    // 提示用户更新 Docker 镜像（可选）
    console.log()
    console.log(chalk.gray('提示: 如需更新 Docker 镜像，请运行: dmla install'))

  } catch (error) {
    console.error(chalk.red('更新失败:'))
    console.error(chalk.red(error.message))
    console.log()
    console.log(chalk.yellow('建议手动执行: npm update -g @icyfenix-dmla/cli @icyfenix-dmla/install'))
    process.exit(1)
  }
}