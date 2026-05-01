/**
 * 管理命令（安装、更新、诊断）
 */
import chalk from 'chalk'
import Docker from 'dockerode'
import { spawn, execSync } from 'child_process'
import http from 'http'
import path from 'path'
import { fileURLToPath } from 'url'
import fs from 'fs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const docker = new Docker()

// 配置
const CONFIG = {
  imageCpu: 'dmla-sandbox:cpu',
  imageGpu: 'dmla-sandbox:gpu',
  dockerhubRegistry: 'icyfenix',
  acrRegistry: 'crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft',
  imageName: 'dmla-sandbox',
  defaultPort: 3001
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
 * 安装镜像
 */
export async function installImages(types, registry = 'dockerhub') {
  const registryUrl = getRegistryUrl(registry)

  console.log(chalk.gray(`   从 ${registry === 'acr' ? '阿里云 ACR' : 'Docker Hub'} 拉取镜像`))

  for (const type of types) {
    console.log()
    console.log(chalk.bold(`📥 拉取 ${type.toUpperCase()} 版本镜像...`))

    const remoteImage = `${registryUrl}:${type}`
    const localImage = type === 'gpu' ? CONFIG.imageGpu : CONFIG.imageCpu

    try {
      // 拉取镜像
      await pullImageWithProgress(remoteImage)

      // Tag 为本地名称
      console.log(chalk.gray(`   重命名为 ${localImage}...`))
      const image = docker.getImage(remoteImage)
      await image.tag({ repo: CONFIG.imageName, tag: type })

      console.log(chalk.green(`✅ ${type.toUpperCase()} 镜像安装完成`))
    } catch (error) {
      console.log(chalk.red(`❌ ${type.toUpperCase()} 镜像安装失败: ${error.message}`))
    }
  }

  console.log()
  console.log(chalk.green('🎉 镜像安装完成'))
  console.log(chalk.yellow('提示: 运行 dmla start 启动服务'))
}

/**
 * 带进度显示的镜像拉取
 */
async function pullImageWithProgress(imageName) {
  return new Promise((resolve, reject) => {
    docker.pull(imageName, (err, stream) => {
      if (err) {
        reject(err)
        return
      }

      // 解析进度
      docker.modem.followProgress(stream, (err, output) => {
        if (err) {
          reject(err)
        } else {
          resolve(output)
        }
      }, (event) => {
        // 显示进度
        if (event.status) {
          let progress = event.status
          if (event.progress) {
            progress += ` ${event.progress}`
          }
          if (event.id) {
            console.log(chalk.gray(`   [${event.id}] ${progress}`))
          } else {
            console.log(chalk.gray(`   ${progress}`))
          }
        }
      })
    })
  })
}

/**
 * 环境诊断
 */
export async function runDoctor() {
  console.log()
  const issues = []

  // ───────────────────────────────────────────────────────────
  // Docker 检查
  // ───────────────────────────────────────────────────────────
  console.log(chalk.bold('🐳 Docker 环境'))

  try {
    const dockerInfo = await docker.info()
    console.log(chalk.green('   ✅ Docker 已安装'))
    console.log(chalk.gray(`   版本: ${dockerInfo.ServerVersion || '未知'}`))

    // 检查版本是否满足要求
    const minVersion = '20.10'
    if (dockerInfo.ServerVersion && dockerInfo.ServerVersion < minVersion) {
      issues.push(`Docker 版本过低，建议升级到 ${minVersion} 或更高`)
    }
  } catch (error) {
    console.log(chalk.red('   ❌ Docker 未安装或未运行'))
    issues.push('请安装 Docker 并确保服务正在运行')
  }

  console.log()

  // ───────────────────────────────────────────────────────────
  // 镜像检查
  // ───────────────────────────────────────────────────────────
  console.log(chalk.bold('Docker 镜像'))

  const cpuImage = CONFIG.imageCpu
  const gpuImage = CONFIG.imageGpu

  let cpuExists = false
  let gpuExists = false

  try {
    const cpuInfo = await docker.getImage(cpuImage).inspect()
    cpuExists = true
    console.log(chalk.green(`   ✅ CPU 镜像已安装`))
    console.log(chalk.gray(`   大小: ${Math.round(cpuInfo.Size / 1024 / 1024)} MB`))
  } catch {
    console.log(chalk.red(`   ❌ CPU 镜像未安装`))
    issues.push('运行 dmla install --cpu 安装 CPU 镜像')
  }

  try {
    const gpuInfo = await docker.getImage(gpuImage).inspect()
    gpuExists = true
    console.log(chalk.green(`   ✅ GPU 镜像已安装`))
    console.log(chalk.gray(`   大小: ${Math.round(gpuInfo.Size / 1024 / 1024)} MB`))
  } catch {
    console.log(chalk.yellow(`   ⚠️ GPU 镜像未安装`))
    console.log(chalk.gray('   (可选，仅在需要 GPU 时安装)'))
  }

  console.log()

  // ───────────────────────────────────────────────────────────
  // GPU 检查
  // ───────────────────────────────────────────────────────────
  console.log(chalk.bold('GPU 驱动'))

  try {
    // 获取完整的 nvidia-smi 输出以解析驱动和 CUDA 版本
    const output = execSync('nvidia-smi', { timeout: 5000, encoding: 'utf8' })

    // 解析驱动版本
    const driverMatch = output.match(/Driver Version:\s*(\d+\.\d+)/)
    const driverVersion = driverMatch ? driverMatch[1] : null

    // 解析 CUDA 兼容上限
    const cudaMatch = output.match(/CUDA Version:\s*(\d+\.\d+)/)
    const cudaVersion = cudaMatch ? cudaMatch[1] : null

    if (output.includes('GPU')) {
      console.log(chalk.green('   ✅ NVIDIA GPU 可用'))

      // 显示驱动版本和 CUDA 兼容上限
      if (driverVersion) {
        console.log(chalk.gray(`   驱动版本: ${driverVersion}`))
      }
      if (cudaVersion) {
        console.log(chalk.gray(`   CUDA 兼容上限: ${cudaVersion}`))
      }

      // GPU 镜像兼容性检查（CUDA 12.8 需要驱动 >= 570）
      const minDriverForGpuImage = 570
      const driverNum = parseFloat(driverVersion || '0')

      // 显示 GPU 设备信息
      const lines = output.split('\n').filter(l => l.trim())
      lines.slice(0, 20).forEach(line => {
        if (line.includes('GPU') && !line.includes('Driver Version') && !line.includes('CUDA Version')) {
          console.log(chalk.gray(`   ${line.trim()}`))
        }
      })

      console.log()

      // GPU 镜像兼容性诊断
      console.log(chalk.bold('GPU 镜像兼容性'))

      if (gpuExists) {
        if (driverNum >= minDriverForGpuImage) {
          console.log(chalk.green(`   ✅ GPU 镜像可用 (驱动 ${driverVersion} >= ${minDriverForGpuImage})`))
        } else {
          console.log(chalk.red(`   ❌ GPU 镜像不兼容 (驱动 ${driverVersion} < ${minDriverForGpuImage})`))
          console.log(chalk.yellow(`   💡 CUDA 12.8 需要驱动 >= ${minDriverForGpuImage}`))
          console.log(chalk.yellow('   解决方案：'))
          console.log(chalk.gray('      1. 升级 NVIDIA 驱动到 570+ 版本'))
          console.log(chalk.gray('      2. 或在前端选择 "Run on CPU" 模式'))
          issues.push('GPU 镜像不兼容，请升级驱动或使用 CPU 模式')
        }
      } else if (!gpuExists) {
        console.log(chalk.yellow('   ⚠️ GPU 镜像未安装'))
        if (driverNum >= minDriverForGpuImage) {
          console.log(chalk.green(`   ✅ 驱动兼容，可以安装 GPU 镜像`))
          console.log(chalk.gray('   运行 dmla install --gpu 安装'))
        } else {
          console.log(chalk.yellow(`   ⚠️ 驱动 ${driverVersion} 不兼容 CUDA 12.8`))
          console.log(chalk.yellow(`   💡 需要驱动 >= ${minDriverForGpuImage} 才能使用 GPU 镜像`))
        }
      }

      // 检查 GPU 镜像
      if (!gpuExists && driverNum >= minDriverForGpuImage) {
        console.log(chalk.yellow('   💡 检测到兼容 GPU，建议安装 GPU 镜像'))
        issues.push('运行 dmla install --gpu 安装 GPU 镜像')
      }
    } else {
      console.log(chalk.gray('   GPU 不可用'))
    }
  } catch {
    console.log(chalk.gray('   GPU 不可用'))
    console.log(chalk.gray('   (如果需要 GPU，请安装 NVIDIA 驱动)'))
  }

  console.log()

  // ───────────────────────────────────────────────────────────
  // 端口检查
  // ───────────────────────────────────────────────────────────
  console.log(chalk.bold('端口可用性'))

  const port = CONFIG.defaultPort
  const portAvailable = await checkPortAvailable(port)

  if (portAvailable) {
    console.log(chalk.green(`   ✅ 端口 ${port} 可用`))
  } else {
    console.log(chalk.red(`   ❌ 端口 ${port} 已被占用`))
    issues.push(`端口 ${port} 已被占用，使用 --port 指定其他端口`)
  }

  console.log()

  // ───────────────────────────────────────────────────────────
  // 网络连通性
  // ───────────────────────────────────────────────────────────
  console.log(chalk.bold('🌐 网络连通性'))

  // 测试 Docker Hub
  console.log(chalk.gray('   测试 Docker Hub 连接...'))
  try {
    execSync('docker pull icyfenix/dmla-sandbox:cpu --quiet', { timeout: 10000 })
    console.log(chalk.green('   ✅ Docker Hub 连接正常'))
  } catch {
    console.log(chalk.yellow('   ⚠️ Docker Hub 连接超时或受限'))
  }

  // 测试 ACR
  console.log(chalk.gray('   测试阿里云 ACR 连接...'))
  try {
    execSync('docker pull crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:cpu --quiet', { timeout: 10000 })
    console.log(chalk.green('   ✅ ACR 连接正常'))
  } catch {
    console.log(chalk.yellow('   ⚠️ ACR 连接超时或受限'))
  }

  console.log()

  // ───────────────────────────────────────────────────────────
  // 问题汇总
  // ───────────────────────────────────────────────────────────
  if (issues.length > 0) {
    console.log(chalk.bold.red('❌ 发现以下问题：'))
    console.log()
    issues.forEach((issue, i) => {
      console.log(chalk.red(`   ${i + 1}. ${issue}`))
    })
    console.log()
    console.log(chalk.yellow('请根据上述提示解决问题后再次运行 dmla doctor'))
  } else {
    console.log(chalk.bold.green('✅ 所有检查通过，环境正常'))
    console.log()
    console.log(chalk.gray('运行 dmla start 启动服务'))
  }
}

/**
 * 检查端口是否可用
 */
async function checkPortAvailable(port) {
  return new Promise((resolve) => {
    const server = http.createServer()
    server.once('error', () => resolve(false))
    server.once('listening', () => {
      server.close()
      resolve(true)
    })
    server.listen(port)
  })
}