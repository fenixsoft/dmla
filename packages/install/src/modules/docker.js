/**
 * Docker 镜像拉取模块
 */
import chalk from 'chalk'
import Docker from 'dockerode'
import cliProgress from 'cli-progress'

const docker = new Docker()

// 配置
const CONFIG = {
  imageCpu: 'dmla-sandbox:cpu',
  imageGpu: 'dmla-sandbox:gpu',
  dockerhubRegistry: 'icyfenix',
  acrRegistry: 'crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/icyfenix',
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
 * 拉取镜像并显示进度
 */
export async function pullImages(types, registry = 'dockerhub') {
  const registryUrl = getRegistryUrl(registry)
  const registryName = registry === 'acr' ? '阿里云 ACR' : 'Docker Hub'

  console.log(chalk.gray(`从 ${registryName} 拉取镜像`))
  console.log()

  // 创建进度条
  const multibar = new cliProgress.MultiBar({
    format: `{type} [{bar}] {percentage}% | {downloaded}`,
    hideCursor: true,
    barsIncompleteChar: '░',
    barsCompleteChar: '█',
  })

  for (const type of types) {
    const remoteImage = `${registryUrl}:${type}`
    const localImage = type === 'gpu' ? CONFIG.imageGpu : CONFIG.imageCpu

    console.log(chalk.bold(`${type.toUpperCase()} 版本`))

    try {
      await pullImageWithProgress(remoteImage, type, multibar)

      // Tag 为本地名称
      console.log(chalk.gray(`重命名为 ${localImage}...`))
      const image = docker.getImage(remoteImage)
      await image.tag({ repo: CONFIG.imageName, tag: type })

      console.log(chalk.green(`✔ ${type.toUpperCase()} 镜像拉取完成`))
      console.log()
    } catch (error) {
      multibar.stop()
      console.log(chalk.red(`❌ ${type.toUpperCase()} 镜像拉取失败: ${error.message}`))
      throw error
    }
  }

  multibar.stop()
}

/**
 * 带进度显示的镜像拉取
 */
async function pullImageWithProgress(imageName, type, multibar) {
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