/**
 * 数据管理命令
 * 提供数据卷挂载、数据集下载、数据管理等功能的 TUI 界面
 */
import chalk from 'chalk'
import pkg from 'enquirer'
const { prompt } = pkg
import fs from 'fs'
import path from 'path'
import os from 'os'
import { spawn } from 'child_process'
import { execSync } from 'child_process'

// 配置文件路径
const DMLA_CONFIG_DIR = path.join(os.homedir(), '.dmla')
const DMLA_CONFIG_FILE = path.join(DMLA_CONFIG_DIR, 'config.json')

// 默认数据目录
const DEFAULT_DATA_DIR = path.join(os.homedir(), 'dmla-data')

// 数据集配置（使用 ModelScope 国内镜像，下载速度更快）
const DATASETS = [
  {
    id: 'tiny-imagenet-200',
    name: 'Tiny ImageNet 200',
    url: 'https://www.modelscope.cn/datasets/icyfenix/Tiny_ImageNet_200.git',
    size: '247MB',
    format: 'git',
    targetDir: 'datasets/tiny-imagenet-200',
    source: 'ModelScope (icyfenix)'
  },
  {
    id: 'cifar-10',
    name: 'CIFAR-10',
    url: 'https://www.modelscope.cn/datasets/icyfenix/CIFAR_10.git',
    size: '163MB',
    format: 'git',
    targetDir: 'datasets/cifar-10',
    source: 'ModelScope (icyfenix)'
  },
  {
    id: 'mnist',
    name: 'MNIST',
    url: 'https://www.modelscope.cn/datasets/icyfenix/MNIST.git',
    size: '11MB',
    format: 'git',
    targetDir: 'datasets/mnist',
    source: 'ModelScope (icyfenix)'
  }
]

/**
 * 显示 Banner
 */
function showBanner() {
  console.log()
  console.log(chalk.cyan('DMLA 数据管理'))
  console.log(chalk.cyan('================'))
  console.log()
}

/**
 * 读取配置文件
 */
function readConfig() {
  try {
    if (fs.existsSync(DMLA_CONFIG_FILE)) {
      const content = fs.readFileSync(DMLA_CONFIG_FILE, 'utf8')
      return JSON.parse(content)
    }
  } catch (error) {
    console.log(chalk.yellow(`警告: 配置文件读取失败: ${error.message}`))
  }
  return { dataVolumePath: DEFAULT_DATA_DIR }
}

/**
 * 写入配置文件
 */
function writeConfig(config) {
  try {
    // 确保配置目录存在
    if (!fs.existsSync(DMLA_CONFIG_DIR)) {
      fs.mkdirSync(DMLA_CONFIG_DIR, { recursive: true })
    }

    config.lastModified = new Date().toISOString()
    fs.writeFileSync(DMLA_CONFIG_FILE, JSON.stringify(config, null, 2))
  } catch (error) {
    console.log(chalk.red(`配置文件写入失败: ${error.message}`))
  }
}

/**
 * 获取数据卷路径
 */
function getDataVolumePath() {
  const config = readConfig()
  return config.dataVolumePath || DEFAULT_DATA_DIR
}

/**
 * 确保数据目录结构存在
 */
function ensureDataDirStructure(dataPath) {
  const subDirs = [
    'datasets',
    'datasets/custom',
    'models',
    'models/alexnet/checkpoints',
    'models/alexnet/final',
    'models/vgg',
    'models/resnet',
    'models/gan',
    'models/llm',
    'models/pretrained',
    'outputs',
    'outputs/training_logs',
    'outputs/visualizations',
    'outputs/exports',
    'cache',
    'cache/downloads',
    'cache/preprocessing',
    'cache/torch_hub'
  ]

  for (const subDir of subDirs) {
    const fullPath = path.join(dataPath, subDir)
    if (!fs.existsSync(fullPath)) {
      fs.mkdirSync(fullPath, { recursive: true })
    }
  }
}

/**
 * 统计目录信息
 */
function getDirectoryStats(dataPath) {
  const stats = {
    datasets: 0,
    models: 0,
    totalSize: 0
  }

  try {
    // 统计已下载的数据集
    const datasetsPath = path.join(dataPath, 'datasets')
    if (fs.existsSync(datasetsPath)) {
      const dirs = fs.readdirSync(datasetsPath).filter(d => {
        const fullPath = path.join(datasetsPath, d)
        return fs.statSync(fullPath).isDirectory() && d !== 'custom'
      })
      stats.datasets = dirs.length
    }

    // 统计模型文件数量
    const modelsPath = path.join(dataPath, 'models')
    if (fs.existsSync(modelsPath)) {
      const countModelFiles = (dir) => {
        let count = 0
        const items = fs.readdirSync(dir)
        for (const item of items) {
          const fullPath = path.join(dir, item)
          const stat = fs.statSync(fullPath)
          if (stat.isDirectory()) {
            count += countModelFiles(fullPath)
          } else if (item.endsWith('.pth') || item.endsWith('.pt') || item.endsWith('.onnx')) {
            count++
          }
        }
        return count
      }
      stats.models = countModelFiles(modelsPath)
    }
  } catch (error) {
    // 忽略统计错误
  }

  return stats
}

/**
 * 检查数据集是否已下载
 */
function isDatasetDownloaded(dataPath, datasetId) {
  const dataset = DATASETS.find(d => d.id === datasetId)
  if (!dataset) return false

  const targetPath = path.join(dataPath, dataset.targetDir)
  return fs.existsSync(targetPath)
}

/**
 * 显示主菜单
 */
async function showMainMenu(dataPath) {
  const stats = getDirectoryStats(dataPath)

  console.log(chalk.gray(`当前挂载路径: ${dataPath}`))
  console.log(chalk.gray(`数据集: ${stats.datasets} 个已下载`))
  console.log(chalk.gray(`模型: ${stats.models} 个已保存`))
  console.log()
  console.log(chalk.gray('------------------------------------'))
  console.log()

  const choices = [
    { name: '1', message: '挂载路径设置        ' + chalk.gray(`[当前: ${dataPath}]`) },
    { name: '2', message: '下载数据集' },
    { name: '3', message: '查看数据集列表' },
    { name: '4', message: '清空数据内容' },
    { name: '5', message: '删除数据卷' },
    { name: '6', message: '退出' }
  ]

  const { action } = await prompt({
    type: 'select',
    name: 'action',
    message: '选择操作',
    choices: choices.map(c => c.message)
  })

  // 解析选择
  const selectedIndex = choices.findIndex(c => c.message === action)
  return selectedIndex + 1
}

/**
 * 挂载路径设置
 */
async function mountPath() {
  const currentPath = getDataVolumePath()

  console.log()
  console.log(chalk.bold('挂载路径设置'))
  console.log(chalk.gray(`当前路径: ${currentPath}`))
  console.log()

  const { newPath } = await prompt({
    type: 'input',
    name: 'newPath',
    message: '输入新的挂载路径 (留空保持当前)',
    initial: currentPath
  })

  if (!newPath || newPath.trim() === '') {
    console.log(chalk.yellow('路径未修改'))
    return
  }

  const resolvedPath = path.resolve(newPath.trim())

  // 检查路径是否存在
  if (!fs.existsSync(resolvedPath)) {
    const { create } = await prompt({
      type: 'confirm',
      name: 'create',
      message: `路径 ${resolvedPath} 不存在，是否创建?`,
      initial: true
    })

    if (!create) {
      console.log(chalk.yellow('操作已取消'))
      return
    }

    fs.mkdirSync(resolvedPath, { recursive: true })
  }

  // 创建完整目录结构
  ensureDataDirStructure(resolvedPath)

  // 保存配置
  writeConfig({ dataVolumePath: resolvedPath })

  console.log(chalk.green(`挂载路径已更新: ${resolvedPath}`))
  console.log(chalk.yellow('提示: 需要重启沙箱服务才能生效 (dmla stop && dmla start)'))
}

/**
 * 清空数据内容
 */
async function clearData() {
  const dataPath = getDataVolumePath()

  console.log()
  console.log(chalk.bold('清空数据内容'))
  console.log(chalk.red('警告: 此操作将删除所有数据集、模型和输出文件!'))
  console.log()

  const { confirm } = await prompt({
    type: 'confirm',
    name: 'confirm',
    message: '确认清空数据内容?',
    initial: false
  })

  if (!confirm) {
    console.log(chalk.yellow('操作已取消'))
    return
  }

  // 清空子目录内容但保留目录结构
  const dirsToClear = ['datasets', 'models', 'outputs', 'cache']

  for (const dir of dirsToClear) {
    const fullPath = path.join(dataPath, dir)
    if (fs.existsSync(fullPath)) {
      const items = fs.readdirSync(fullPath)
      for (const item of items) {
        const itemPath = path.join(fullPath, item)
        fs.rmSync(itemPath, { recursive: true, force: true })
      }
    }
  }

  // 重新创建目录结构
  ensureDataDirStructure(dataPath)

  console.log(chalk.green('数据内容已清空'))
}

/**
 * 删除数据卷
 */
async function removeData() {
  const dataPath = getDataVolumePath()

  console.log()
  console.log(chalk.bold('删除数据卷'))
  console.log(chalk.red('警告: 此操作将删除整个数据目录和所有数据!'))
  console.log()

  const { confirm } = await prompt({
    type: 'confirm',
    name: 'confirm',
    message: '确认删除数据卷?',
    initial: false
  })

  if (!confirm) {
    console.log(chalk.yellow('操作已取消'))
    return
  }

  // 删除整个目录
  if (fs.existsSync(dataPath)) {
    fs.rmSync(dataPath, { recursive: true, force: true })
  }

  // 清除配置
  writeConfig({ dataVolumePath: null })

  console.log(chalk.green('数据卷已删除'))
}

/**
 * 查看数据集列表
 */
function listDatasets() {
  const dataPath = getDataVolumePath()

  console.log()
  console.log(chalk.bold('已下载的数据集'))
  console.log()

  for (const dataset of DATASETS) {
    const downloaded = isDatasetDownloaded(dataPath, dataset.id)
    const status = downloaded ? chalk.green('[已下载]') : chalk.gray('[未下载]')
    console.log(`${status} ${dataset.name} (${dataset.size})`)
  }

  console.log()
}

/**
 * 下载数据集子菜单
 */
async function downloadDatasets() {
  const dataPath = getDataVolumePath()

  console.log()
  console.log(chalk.bold('下载数据集'))
  console.log()

  // 检查数据目录是否存在
  if (!fs.existsSync(dataPath)) {
    console.log(chalk.yellow(`数据目录不存在: ${dataPath}`))
    console.log(chalk.yellow('请先设置挂载路径'))
    return
  }

  // 检查 Git 环境
  try {
    execSync('git --version', { stdio: 'pipe' })
  } catch {
    console.log(chalk.red('❌ Git 未安装'))
    console.log(chalk.yellow('下载数据集需要 Git，请先安装: https://git-scm.com/downloads'))
    return
  }

  // 构建选项列表
  const choices = DATASETS.map((dataset, index) => {
    const downloaded = isDatasetDownloaded(dataPath, dataset.id)

    let message = `${dataset.name} (${dataset.size})`
    if (downloaded) {
      message += ' [已下载]'
    }

    return {
      name: index.toString(),
      message,
      disabled: downloaded
    }
  })

  // 操作提示
  console.log(chalk.gray('操作: 上下键移动，空格勾选/取消，回车确认，ESC 返回'))
  console.log()

  try {
    const { selected } = await prompt({
      type: 'multiselect',
      name: 'selected',
      message: '选择要下载的数据集',
      choices,
      hint: '空格选择，回车确认下载',
      warn: '已下载'
    })

    if (!selected || selected.length === 0) {
      console.log(chalk.yellow('未选择任何数据集'))
      return
    }

    // 下载选中的数据集
    for (const indexStr of selected) {
      const index = parseInt(indexStr)
      const dataset = DATASETS[index]

      console.log()
      console.log(chalk.cyan(`────────────────────────────────────`))

    // 检查是否已下载
    if (isDatasetDownloaded(dataPath, dataset.id)) {
      console.log(chalk.yellow(`${dataset.name} 已下载，跳过`))
      continue
    }

    await downloadDataset(dataPath, dataset)
  }

  console.log()
  console.log(chalk.cyan(`────────────────────────────────────`))
  console.log(chalk.green('所有选中的数据集已处理完成'))
  } catch (error) {
    // 用户按 ESC 或 Ctrl+C 取消
    if (error.message && error.message.includes('cancel')) {
      console.log(chalk.gray('返回上一级'))
      return
    }
    throw error
  }
}

/**
 * 下载单个数据集
 */
async function downloadDataset(dataPath, dataset) {
  console.log()
  console.log(chalk.bold(`下载 ${dataset.name}...`))
  console.log(chalk.gray(`来源: ${dataset.source || 'ModelScope'}`))
  console.log(chalk.gray(`URL: ${dataset.url}`))

  const targetDir = path.join(dataPath, dataset.targetDir)

  console.log(chalk.gray(`目标: ${targetDir}`))
  console.log()

  // 确保目标目录的父目录存在
  const parentDir = path.dirname(targetDir)
  if (!fs.existsSync(parentDir)) {
    fs.mkdirSync(parentDir, { recursive: true })
  }

  try {
    if (dataset.format === 'git') {
      // 使用 git clone 下载 ModelScope 数据集
      console.log(chalk.gray('开始 git clone...'))
      console.log()

      // 先安装 git lfs（如果需要）
      try {
        execSync('git lfs install', { stdio: 'pipe' })
      } catch {
        // git lfs 可能未安装，但大多数数据集不需要
      }

      // 执行 git clone
      await new Promise((resolve, reject) => {
        const git = spawn('git', ['clone', dataset.url, targetDir], { stdio: 'inherit' })

        git.on('close', (code) => {
          if (code === 0) {
            resolve()
          } else {
            reject(new Error(`git clone exited with code ${code}`))
          }
        })

        git.on('error', (err) => {
          reject(err)
        })
      })

      console.log()
      console.log(chalk.green('下载完成'))

    } else {
      // 原有的 curl/wget 下载逻辑（保留兼容性）
      const cacheDir = path.join(dataPath, 'cache', 'downloads')
      const downloadFile = path.join(cacheDir, `${dataset.id}.${dataset.format}`)

      if (!fs.existsSync(cacheDir)) {
        fs.mkdirSync(cacheDir, { recursive: true })
      }

      const curlArgs = [
        '-L',
        '-o', downloadFile,
        '--progress-bar',
        dataset.url
      ]

      console.log(chalk.gray('开始下载...'))
      console.log()

      await new Promise((resolve, reject) => {
        const curl = spawn('curl', curlArgs, { stdio: 'inherit' })

        curl.on('close', (code) => {
          if (code === 0) {
            resolve()
          } else {
            reject(new Error(`curl exited with code ${code}`))
          }
        })

        curl.on('error', (err) => {
          reject(err)
        })
      })

      console.log()
      console.log(chalk.green('下载完成'))
      console.log()

      // 解压
      console.log(chalk.gray('正在解压...'))

      if (dataset.format === 'zip') {
        const unzipDir = path.join(cacheDir, dataset.id)
        fs.mkdirSync(unzipDir, { recursive: true })
        execSync(`unzip -o "${downloadFile}" -d "${unzipDir}"`, { stdio: 'inherit' })

        const extractedDir = path.join(unzipDir, dataset.id)
        if (fs.existsSync(extractedDir)) {
          fs.cpSync(extractedDir, targetDir, { recursive: true })
          fs.rmSync(unzipDir, { recursive: true, force: true })
        }

      } else if (dataset.format === 'tar.gz') {
        execSync(`tar -xzf "${downloadFile}" -C "${path.join(dataPath, 'datasets')}"`, { stdio: 'inherit' })
      }

      // 清理下载文件
      fs.rmSync(downloadFile, { force: true })
    }

    console.log()
    console.log(chalk.green(`数据集已保存到 ${targetDir}`))

    // 更新配置
    const config = readConfig()
    if (!config.installedDatasets) {
      config.installedDatasets = []
    }
    if (!config.installedDatasets.includes(dataset.id)) {
      config.installedDatasets.push(dataset.id)
    }
    writeConfig(config)

  } catch (error) {
    console.log()
    console.log(chalk.red(`下载失败: ${error.message}`))
    console.log(chalk.yellow('您可以稍后重试'))
  }
}

/**
 * 运行数据管理 TUI
 */
export async function runDataTUI() {
  showBanner()

  let dataPath = getDataVolumePath()

  // 确保配置目录存在
  if (!fs.existsSync(DMLA_CONFIG_DIR)) {
    fs.mkdirSync(DMLA_CONFIG_DIR, { recursive: true })
  }

  // 如果数据目录不存在，提示创建
  if (!fs.existsSync(dataPath)) {
    console.log(chalk.yellow(`数据目录不存在: ${dataPath}`))
    console.log()

    const { create } = await prompt({
      type: 'confirm',
      name: 'create',
      message: '是否创建数据目录?',
      initial: true
    })

    if (create) {
      ensureDataDirStructure(dataPath)
      console.log(chalk.green(`数据目录已创建: ${dataPath}`))
    }
  }

  // 主循环
  while (true) {
    console.log()
    const action = await showMainMenu(dataPath)

    switch (action) {
      case 1:
        await mountPath()
        break
      case 2:
        await downloadDatasets()
        break
      case 3:
        listDatasets()
        break
      case 4:
        await clearData()
        break
      case 5:
        await removeData()
        break
      case 6:
        console.log()
        console.log(chalk.gray('已退出数据管理'))
        console.log()
        return
    }

    // 刷新路径（可能在操作中修改了）
    dataPath = getDataVolumePath()
  }
}

/**
 * CLI 命令入口 (非 TUI 模式)
 */
export async function runDataCommand(subCommand, options) {
  const dataPath = getDataVolumePath()

  switch (subCommand) {
    case 'path':
      console.log(dataPath)
      break
    case 'mount':
      if (options.path) {
        const resolvedPath = path.resolve(options.path)
        ensureDataDirStructure(resolvedPath)
        writeConfig({ dataVolumePath: resolvedPath })
        console.log(chalk.green(`挂载路径已设置: ${resolvedPath}`))
      } else {
        console.log(chalk.yellow('请指定路径: dmla data mount <path>'))
      }
      break
    case 'clear':
      await clearData()
      break
    case 'remove':
      await removeData()
      break
    case 'download':
      await downloadDatasets()
      break
    default:
      // 无子命令时进入 TUI
      await runDataTUI()
  }
}

export default {
  runDataTUI,
  runDataCommand,
  getDataVolumePath,
  DATASETS
}