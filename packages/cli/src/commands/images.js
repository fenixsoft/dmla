/**
 * DMLA CLI images 命令
 * 用于拉取 Docker 镜像（仅拉取，不安装 CLI）
 */
import chalk from 'chalk'
import pkg from 'enquirer'
const { prompt } = pkg
import Docker from 'dockerode'
import { pullImages } from '@icyfenix-dmla/install'

const docker = new Docker()

// 配置
const CONFIG = {
  imageCpu: 'dmla-sandbox:cpu',
  imageGpu: 'dmla-sandbox:gpu'
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
 * 显示镜像状态
 */
async function showImageStatus() {
  console.log(chalk.bold('本地镜像状态'))
  console.log()

  const cpuExists = await checkImageExists(CONFIG.imageCpu)
  const gpuExists = await checkImageExists(CONFIG.imageGpu)

  if (cpuExists) {
    try {
      const cpuInfo = await docker.getImage(CONFIG.imageCpu).inspect()
      console.log(chalk.green(`   ✅ CPU 镜像已安装 (${Math.round(cpuInfo.Size / 1024 / 1024)} MB)`))
    } catch {
      console.log(chalk.green(`   ✅ CPU 镜像已安装`))
    }
  } else {
    console.log(chalk.yellow(`   ⚠️ CPU 镜像未安装 (~ 683MB)`))
  }

  if (gpuExists) {
    try {
      const gpuInfo = await docker.getImage(CONFIG.imageGpu).inspect()
      console.log(chalk.green(`   ✅ GPU 镜像已安装 (${Math.round(gpuInfo.Size / 1024 / 1024)} MB)`))
    } catch {
      console.log(chalk.green(`   ✅ GPU 镜像已安装`))
    }
  } else {
    console.log(chalk.yellow(`   ⚠️ GPU 镜像未安装 (~ 7.93GB)`))
  }

  console.log()
}

/**
 * 运行 images 命令 TUI
 */
export async function runImagesTUI() {
  console.log(chalk.blue('DMLA 镜像管理'))
  console.log()

  // 检查 Docker 是否可用
  try {
    await docker.ping()
  } catch {
    console.log(chalk.red('❌ Docker 未安装或未运行'))
    console.log(chalk.yellow('请先安装 Docker: https://docs.docker.com/get-docker/'))
    return
  }

  // 显示当前镜像状态
  await showImageStatus()

  // 步骤 1: 选择镜像仓库
  const registryChoice = await prompt({
    type: 'select',
    name: 'registry',
    message: '请选择镜像仓库',
    initial: 0,
    choices: [
      { name: 'dockerhub', message: 'Docker Hub (全球访问)' },
      { name: 'acr', message: '阿里云 ACR (国内加速)' }
    ]
  })

  const registry = registryChoice.registry
  const registryName = registry === 'acr' ? '阿里云 ACR' : 'Docker Hub'
  console.log(chalk.gray(`已选择: ${registryName}`))
  console.log()

  // 步骤 2: 选择镜像类型
  const typeChoice = await prompt({
    type: 'select',
    name: 'imageType',
    message: '请选择要拉取的镜像',
    initial: 0,
    choices: [
      { name: 'all', message: '全部 (CPU + GPU, ~ 8.6GB)' },
      { name: 'cpu', message: '仅 CPU 版本 (~ 683MB)' },
      { name: 'gpu', message: '仅 GPU 版本 (~ 7.93GB)' }
    ]
  })

  const selectedType = typeChoice.imageType
  let imageTypes = []

  if (selectedType === 'all') {
    imageTypes = ['cpu', 'gpu']
  } else {
    imageTypes = [selectedType]
  }

  console.log(chalk.gray(`将拉取: ${imageTypes.map(t => t.toUpperCase()).join(', ')} 镜像`))
  console.log()

  // 步骤 3: 拉取镜像（复用 install 包的 pullImages，包含重试逻辑）
  console.log(chalk.bold('拉取 Docker 镜像'))
  console.log()

  const results = await pullImages(imageTypes, registry)

  // 步骤 4: 显示最终结果
  console.log()
  console.log(chalk.bold('拉取结果汇总'))
  console.log()

  const successCount = Object.values(results).filter(r => r.success).length
  const failCount = Object.values(results).filter(r => !r.success).length

  if (successCount > 0) {
    const successTypes = Object.entries(results)
      .filter(([type, r]) => r.success)
      .map(([type]) => type.toUpperCase())
    console.log(chalk.green(`   ✅ 成功: ${successTypes.join(', ')}`))
  }

  if (failCount > 0) {
    const failTypes = Object.entries(results)
      .filter(([type, r]) => !r.success)
      .map(([type]) => type.toUpperCase())
    console.log(chalk.red(`   ❌ 失败: ${failTypes.join(', ')}`))
  }

  console.log()

  // 再次显示镜像状态
  await showImageStatus()

  // 提示下一步操作
  if (successCount > 0) {
    console.log(chalk.green('🎉 镜像拉取完成'))
    console.log(chalk.gray('提示: 运行 dmla start 启动服务'))
  } else {
    console.log(chalk.yellow('⚠️ 部分镜像拉取失败'))
    console.log(chalk.gray('请检查网络连接后再次尝试: dmla images'))
  }

  console.log()
}