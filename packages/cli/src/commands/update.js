/**
 * DMLA CLI update 命令
 * 通过 npm 更新程序
 */
import chalk from 'chalk'
import { execSync } from 'child_process'

/**
 * 运行 update 命令
 */
export async function runUpdate() {
  console.log(chalk.blue('更新 DMLA...'))
  console.log()

  try {
    // 检查当前版本
    const currentVersion = execSync('npm list -g @icyfenix-dmla/cli --depth=0 2>/dev/null | grep @icyfenix-dmla/cli', { encoding: 'utf-8' })
      .trim()
      .split('@')[2] || '未知'

    console.log(chalk.gray(`当前版本: ${currentVersion}`))
    console.log()

    // 执行 npm 更新
    console.log(chalk.cyan('正在从 npm 更新...'))
    const output = execSync('npm update -g @icyfenix-dmla/cli @icyfenix-dmla/install', {
      encoding: 'utf-8',
      stdio: 'pipe'
    })

    console.log(output)

    // 检查更新后的版本
    const newVersion = execSync('npm list -g @icyfenix-dmla/cli --depth=0 2>/dev/null | grep @icyfenix-dmla/cli', { encoding: 'utf-8' })
      .trim()
      .split('@')[2] || '未知'

    if (newVersion !== currentVersion) {
      console.log()
      console.log(chalk.green(`✓ 更新成功！`))
      console.log(chalk.gray(`  ${currentVersion} → ${newVersion}`))
    } else {
      console.log()
      console.log(chalk.yellow('已是最新版本'))
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