/**
 * Docker 镜像拉取模块
 */
import chalk from 'chalk'
import Docker from 'dockerode'
import cliProgress from 'cli-progress'
import pkg from 'enquirer'
const { prompt } = pkg

const docker = new Docker()

// 配置
const CONFIG = {
  imageCpu: 'dmla-sandbox:cpu',
  imageGpu: 'dmla-sandbox:gpu',
  dockerhubRegistry: 'icyfenix',
  acrRegistry: 'crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft',
  imageName: 'dmla-sandbox'
}

/**
 * 获取镜像仓库地址
 */
function getRegistryUrl(registry) {
  if (registry === 'acr') {
    return `${CONFIG.acrRegistry}/${CONFIG.imageName}`
  }
  return `${CONFIG.dockerhubRegistry}/${CONFIG.imageName}`
}

/**
 * 检查本地镜像是否存在
 */
async function checkImageExists(imageName) {
  try {
    await docker.getImage(imageName).inspect()
    return true
  } catch {
    return false
  }
}

/**
 * 拉取镜像并显示进度
 * @returns {Object} 拉取结果 { cpu: {success: boolean, error?: string}, gpu: {success: boolean, error?: string} }
 */
export async function pullImages(types, registry = 'dockerhub') {
  const registryUrl = getRegistryUrl(registry)
  const registryName = registry === 'acr' ? '阿里云 ACR' : 'Docker Hub'

  console.log(chalk.gray(`从 ${registryName} 拉取镜像`))
  console.log()

  const results = {}

  for (const type of types) {
    const localImage = type === 'gpu' ? CONFIG.imageGpu : CONFIG.imageCpu
    const remoteImage = `${registryUrl}:${type}`

    console.log(chalk.bold(`${type.toUpperCase()} 版本`))

    // 检查本地镜像是否已存在
    const exists = await checkImageExists(localImage)

    if (exists) {
      console.log(chalk.yellow(`⚠️  本地已存在 ${localImage} 镜像`))

      const updateChoice = await prompt({
        type: 'select',
        name: 'action',
        message: '是否更新镜像？',
        choices: [
          { name: 'update', message: '更新镜像（重新拉取）' },
          { name: 'skip', message: '跳过（使用现有镜像）' }
        ]
      })

      if (updateChoice.action === 'skip') {
        console.log(chalk.green(`✔ 跳过 ${type.toUpperCase()} 镜像，使用现有版本`))
        console.log()
        results[type] = { success: true, skipped: true }
        continue
      }

      console.log(chalk.gray(`更新镜像...`))
    }

    // 拉取镜像（支持无限重试循环）
    let pullSuccess = false
    while (!pullSuccess) {
      try {
        await pullImageWithProgress(remoteImage, 2) // 自动重试 2 次

        // Tag 为本地名称
        console.log(chalk.gray(`重命名为 ${localImage}...`))
        const image = docker.getImage(remoteImage)
        await image.tag({ repo: CONFIG.imageName, tag: type })

        console.log(chalk.green(`✔ ${type.toUpperCase()} 镜像拉取完成`))
        console.log()
        results[type] = { success: true }
        pullSuccess = true
      } catch (error) {
        console.log(chalk.red(`❌ ${type.toUpperCase()} 镜像拉取失败: ${error.message}`))
        console.log()

        // 询问用户是否继续重试
        const retryChoice = await prompt({
          type: 'select',
          name: 'action',
          message: '拉取失败，是否继续重试？',
          choices: [
            { name: 'retry', message: '继续重试（自动重试 2 次）' },
            { name: 'skip', message: '跳过此镜像' },
            { name: 'exit', message: '退出整个流程' }
          ]
        })

        if (retryChoice.action === 'retry') {
          console.log(chalk.gray('继续重试...'))
          continue // 继续循环，再次尝试
        } else if (retryChoice.action === 'skip') {
          // 跳过，记录失败结果
          console.log(chalk.yellow('您可以手动拉取镜像：'))
          console.log(chalk.cyan(`   docker pull ${remoteImage}`))
          console.log(chalk.gray(`   docker tag ${remoteImage} ${localImage}`))
          console.log()

          results[type] = { success: false, error: error.message, manualCommand: remoteImage }

          // 如果本地已有镜像，可以继续使用
          if (exists) {
            console.log(chalk.gray('   但本地已有镜像，可继续使用'))
            console.log()
          }
          break // 跳出循环，处理下一个镜像
        } else {
          // 用户选择退出
          console.log(chalk.yellow('已取消镜像拉取'))
          results[type] = { success: false, error: 'User cancelled', manualCommand: remoteImage }
          return results // 直接返回，结束整个流程
        }
      }
    }
  }

  return results
}

/**
 * 带重试的镜像拉取
 * @param {string} imageName 镜像名称
 * @param {number} maxRetries 最大重试次数（默认 2 次）
 */
async function pullImageWithProgress(imageName, maxRetries = 2) {
  for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
    try {
      await doPullImageWithProgress(imageName)
      return // 成功则退出
    } catch (error) {
      if (attempt <= maxRetries) {
        console.log(chalk.yellow(`  拉取失败，正在重试 (${attempt}/${maxRetries})...`))
        // 等待 1 秒后重试
        await new Promise(resolve => setTimeout(resolve, 1000))
      } else {
        // 最后一次尝试也失败了，抛出错误
        throw error
      }
    }
  }
}

/**
 * 带进度显示的镜像拉取（单次尝试）
 */
async function doPullImageWithProgress(imageName) {
  return new Promise((resolve, reject) => {
    console.log(chalk.gray(`  正在拉取: ${imageName}`))

    // 保存每个 layer 的最新状态
    const layers = {}
    let outputLines = 0
    let hasUpdate = false

    docker.pull(imageName, (err, stream) => {
      if (err) {
        console.log(chalk.red(`  拉取失败: ${err.message}`))
        reject(err)
        return
      }

      stream.on('data', (chunk) => {
        try {
          const lines = chunk.toString().split('\n').filter(Boolean)
          for (const line of lines) {
            const event = JSON.parse(line)
            if (!event.id) continue

            // 更新 layer 状态
            const existing = layers[event.id] || {}
            layers[event.id] = {
              id: event.id,
              status: event.status,
              progress: event.progress || existing.progress || '',
              progressDetail: event.progressDetail || existing.progressDetail || {}
            }

            // 标记需要刷新显示
            if (event.status === 'Downloading' || event.status === 'Pull complete' || event.status === 'Already exists') {
              hasUpdate = true
            }
          }

          // 刷新显示
          if (hasUpdate) {
            hasUpdate = false
            // 移动光标到顶部并清除
            if (outputLines > 0) {
              process.stdout.write(`\x1b[${outputLines}A\x1b[0J`)
            }

            outputLines = 0
            for (const [layerId, layer] of Object.entries(layers)) {
              const output = formatLayerOutput(layer)
              process.stdout.write(output + '\n')
              outputLines++
            }
          }
        } catch (e) {
          // 忽略解析错误
        }
      })

      stream.on('error', (err) => {
        console.log(chalk.red(`  Stream 错误: ${err.message}`))
        reject(err)
      })

      stream.on('end', () => {
        // 最终输出所有完成状态
        if (outputLines > 0) {
          process.stdout.write(`\x1b[${outputLines}A\x1b[0J`)
        }
        for (const [layerId, layer] of Object.entries(layers)) {
          console.log(formatLayerOutput(layer))
        }
        console.log(chalk.green(`  ✔ 拉取完成`))
        resolve()
      })
    })
  })
}

/**
 * 格式化单个 layer 输出，类似 Docker 原生格式
 */
function formatLayerOutput(layer) {
  const id = layer.id.substring(0, 12)

  if (layer.status === 'Already exists') {
    return `${id}: Already exists `
  }

  if (layer.status === 'Pull complete') {
    return `${id}: Pull complete `
  }

  if (layer.status === 'Verifying Checksum') {
    return `${id}: Verifying Checksum`
  }

  if (layer.status === 'Download complete') {
    return `${id}: Download complete`
  }

  if (layer.status === 'Downloading') {
    const detail = layer.progressDetail || {}
    const current = detail.current || 0
    const total = detail.total || 0

    // 如果有 progressDetail，生成进度条
    if (total > 0) {
      const barWidth = 50
      const percent = Math.round((current / total) * 100)
      const filled = Math.min(Math.round((current / total) * barWidth), barWidth)
      const arrow = filled < barWidth ? '>' : ''
      const bar = '='.repeat(filled) + arrow + ' '.repeat(barWidth - filled - (arrow ? 1 : 0))

      const currentStr = formatSize(current)
      const totalStr = formatSize(total)

      return `${id}: Downloading [${bar}]  ${currentStr}/${totalStr}`
    }

    // 没有 progressDetail，只显示状态
    return `${id}: Downloading ${layer.progress}`
  }

  return `${id}: ${layer.status} ${layer.progress || ''}`
}

/**
 * 解析大小字符串
 */
function parseSize(sizeStr) {
  const units = { 'B': 1, 'KB': 1024, 'MB': 1024 * 1024, 'GB': 1024 * 1024 * 1024 }
  const match = sizeStr.match(/(\d+\.?\d*)([KMGT]?B)/)
  if (match) {
    return parseFloat(match[1]) * (units[match[2]] || 1)
  }
  return 0
}

/**
 * 格式化大小
 */
function formatSize(bytes) {
  if (bytes < 1024) return bytes + 'B'
  if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + 'KB'
  if (bytes < 1024 * 1024 * 1024) return Math.round(bytes / 1024 / 1024) + 'MB'
  return Math.round(bytes / 1024 / 1024 / 1024) + 'GB'
}