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
 * 获取当前包的版本号
 */
function getCurrentVersion() {
  const __filename = fileURLToPath(import.meta.url)
  const __dirname = path.dirname(__filename)
  const pkgPath = path.resolve(__dirname, '../package.json')
  return JSON.parse(fs.readFileSync(pkgPath, 'utf8')).version
}

/**
 * 运行 update 命令
 */
export async function runUpdate() {
  console.log(chalk.blue('更新 DMLA...'))
  console.log()

  // 在 npm update 之前读取当前版本（更新后旧文件可能被删除导致读取失败）
  let oldVersion
  try {
    oldVersion = getCurrentVersion()
    console.log(chalk.gray(`当前版本: ${oldVersion}`))
  } catch {
    // 读取失败时继续执行更新
  }

  try {
    // 直接执行 npm 更新
    console.log(chalk.cyan('正在从 npm 更新...'))
    execSync('npm update -g @icyfenix-dmla/cli @icyfenix-dmla/install', {
      encoding: 'utf-8',
      stdio: 'inherit'
    })

    console.log()
    console.log(chalk.green('✓ 更新完成'))

    // 尝试读取更新后的版本号
    // npm update 可能删除了当前进程的旧文件，需要用 npm list 获取新版本
    try {
      const newVersion = execSync('npm list -g @icyfenix-dmla/cli --depth=0 --json', {
        encoding: 'utf-8'
      })
      const parsed = JSON.parse(newVersion)
      const version = parsed?.dependencies?.['@icyfenix-dmla/cli']?.version
      if (version) {
        console.log(chalk.cyan(`更新后版本: ${version}`))
        if (oldVersion && version !== oldVersion) {
          console.log(chalk.green(`${oldVersion} → ${version}`))
        } else if (oldVersion && version === oldVersion) {
          console.log(chalk.gray('已是最新版本'))
        }
      }
    } catch {
      // npm list 读取失败时静默跳过，不影响更新结果
    }

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