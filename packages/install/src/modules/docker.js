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
  tcrRegistry: 'ccr.ccs.tencentyun.com/icyfenix',
  imageName: 'dmla-sandbox'
}

/**
 * 获取镜像仓库地址
 */
function getRegistryUrl(registry) {
  if (registry === 'tcr') {
    return `${CONFIG.tcrRegistry}/${CONFIG.imageName}`
  }
  return `${CONFIG.dockerhubRegistry}/${CONFIG.imageName}`
}

/**
 * 拉取镜像并显示进度
 */
export async function pullImages(types, registry = 'dockerhub') {
  const registryUrl = getRegistryUrl(registry)
  const registryName = registry === 'tcr' ? '腾讯云 TCR' : 'Docker Hub'

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

      console.log(chalk.green(`✅ ${type.toUpperCase()} 镜像拉取完成`))
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
    // 进度跟踪
    const layers = {}
    let overallBar = multibar.create(100, 0, {
      type: chalk.bold(type.toUpperCase()),
      downloaded: '准备中...'
    })

    docker.pull(imageName, (err, stream) => {
      if (err) {
        reject(err)
        return
      }

      docker.modem.followProgress(stream, (err, output) => {
        if (err) {
          reject(err)
        } else {
          overallBar.update(100, { downloaded: '完成' })
          resolve(output)
        }
      }, (event) => {
        // 更新进度
        if (event.id && event.progress) {
          layers[event.id] = event.progress

          // 计算总体进度
          const totalLayers = Object.keys(layers).length
          let completed = 0
          let totalSize = 0
          let downloadedSize = 0

          for (const [id, progress] of Object.entries(layers)) {
            // 解析进度字符串 "xMB/yMB"
            const match = progress.match(/(\d+\.?\d*[KMGT]?B)\/(\d+\.?\d*[KMGT]?B)/)
            if (match) {
              downloadedSize += parseSize(match[1])
              totalSize += parseSize(match[2])
            }
            if (progress.includes('complete') || progress.includes('Pull complete')) {
              completed++
            }
          }

          const percentage = totalSize > 0 ? Math.round((downloadedSize / totalSize) * 100) : 0
          const downloadedStr = formatSize(downloadedSize) + '/' + formatSize(totalSize)

          overallBar.update(percentage, {
            type: chalk.bold(type.toUpperCase()),
            downloaded: downloadedStr
          })
        }

        // 处理完成状态
        if (event.status === 'Download complete' || event.status === 'Pull complete') {
          // 不做特殊处理，让进度自然更新
        }
      })
    })
  })
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