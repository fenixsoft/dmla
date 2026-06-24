/**
 * 模型管理命令
 * 提供模型下载、删除、列表等功能的 TUI 界面
 */
import chalk from 'chalk'
import pkg from 'enquirer'
const { prompt } = pkg
import fs from 'fs'
import path from 'path'
import os from 'os'
import { spawn, execSync } from '../verbose.js'

// 配置文件路径
const DMLA_CONFIG_DIR = path.join(os.homedir(), '.dmla')
const DMLA_CONFIG_FILE = path.join(DMLA_CONFIG_DIR, 'config.json')

// 默认数据目录
const DEFAULT_DATA_DIR = path.join(os.homedir(), 'dmla-data')

// 模型配置（使用 ModelScope 国内镜像，下载速度更快）
const MODELS = [
  {
    id: 'qwen3.5-0.8b-instruct',
    name: 'Qwen3.5-0.8B-Instruct',
    description: '通义千问 3.5 0.8B 指令微调模型',
    url: 'https://www.modelscope.cn/icyfenix/Qwen3.5-0.8B-Instruct.git',
    size: '~1.6GB',
    format: 'git',
    targetDir: 'models/llm/qwen3.5-0.8b-instruct',
    source: 'ModelScope (icyfenix)',
    framework: 'Transformers',
    task: '对话生成'
  },
  {
    id: 'minimind',
    name: 'MiniMind',
    description: 'MiniMind 小型语言模型 (0.2B, dim=768, layers=8)，支持预训练/SFT/DPO全流程',
    url: 'https://www.modelscope.cn/icyfenix/minimind.git',
    size: '~276MB (2×138MB)',
    format: 'git',
    targetDir: 'models/llm/minimind',
    source: 'ModelScope (icyfenix)',
    framework: 'Transformers',
    task: '对话生成'
  },
  {
    id: 'alexnet',
    name: 'AlexNet',
    description: 'AlexNet 卷积神经网络，ImageNet 预训练权重',
    url: 'https://www.modelscope.cn/icyfenix/alexnet.git',
    size: '236MB',
    format: 'git',
    targetDir: 'models/alexnet',
    source: 'ModelScope (icyfenix)',
    framework: 'PyTorch',
    task: '图像分类'
  }
]

/**
 * 检查是否是用户取消操作（ESC 或 Ctrl+C）
 * enquirer 可能抛出空字符串错误或包含 'cancel' 的消息
 */
function isUserCancel(error) {
  return !error ||
         !error.message ||
         error.message === '' ||
         error.message.includes('cancel') ||
         error.code === 'ERR_USE_AFTER_CLOSE'
}

/**
 * 显示 Banner
 */
function showBanner() {
  console.log()
  console.log(chalk.cyan(' ______   ____    ____  _____          _       '))
  console.log(chalk.cyan('|_   _ `.|_   \\  /   _||_   _|        / \\      '))
  console.log(chalk.cyan('  | | `. \\ |   \\/   |    | |         / _ \\     '))
  console.log(chalk.cyan('  | |  | | | |\\  /| |    | |   _    / ___ \\    '))
  console.log(chalk.cyan(' _| |_.\' /_| |_\\/_| |_  _| |__/ | _/ /   \\ \\_  '))
  console.log(chalk.cyan('|______.\'|_____||_____||________||____| |____| '))
  console.log(chalk.blue('== Designing Machine Learning Applications =='))
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
 * 确保模型目录结构存在
 */
function ensureModelDirStructure(dataPath) {
  const modelDirs = [
    'models/llm',
    'models/alexnet/checkpoints',
    'models/alexnet/final',
    'models/vgg',
    'models/resnet',
    'models/gan',
    'models/pretrained'
  ]

  for (const dir of modelDirs) {
    const fullPath = path.join(dataPath, dir)
    if (!fs.existsSync(fullPath)) {
      fs.mkdirSync(fullPath, { recursive: true })
    }
  }
}

/**
 * 检查模型是否已下载（且完整可用）
 */
function isModelDownloaded(dataPath, modelId) {
  const model = MODELS.find(m => m.id === modelId)
  if (!model) return false

  const targetPath = path.join(dataPath, model.targetDir)
  if (!fs.existsSync(targetPath)) return false

  // 检查是否有 LFS 不完整标记文件
  const incompleteMarker = path.join(targetPath, '.lfs-incomplete')
  if (fs.existsSync(incompleteMarker)) {
    return false
  }

  return true
}

/**
 * 检查模型目录是否已存在（不管是否完整）
 */
function isModelExists(dataPath, modelId) {
  const model = MODELS.find(m => m.id === modelId)
  if (!model) return false

  const targetPath = path.join(dataPath, model.targetDir)
  return fs.existsSync(targetPath)
}

/**
 * 检查模型是否不完整（LFS 未拉取）
 */
function isModelIncomplete(dataPath, modelId) {
  const model = MODELS.find(m => m.id === modelId)
  if (!model) return false

  const targetPath = path.join(dataPath, model.targetDir)
  if (!fs.existsSync(targetPath)) return false

  const incompleteMarker = path.join(targetPath, '.lfs-incomplete')
  return fs.existsSync(incompleteMarker)
}

/**
 * 统计模型信息
 */
function getModelStats(dataPath) {
  const stats = {
    downloaded: 0,
    incomplete: 0
  }

  try {
    for (const model of MODELS) {
      if (isModelDownloaded(dataPath, model.id)) {
        stats.downloaded++
      } else if (isModelIncomplete(dataPath, model.id)) {
        stats.incomplete++
      }
    }
  } catch (error) {
    // 忽略统计错误
  }

  return stats
}

/**
 * 显示主菜单
 */
async function showMainMenu(dataPath) {
  const stats = getModelStats(dataPath)

  console.log(chalk.gray(`当前挂载路径: ${dataPath}`))
  if (stats.incomplete > 0) {
    console.log(chalk.gray(`模型: ${stats.downloaded} 个可用, ${chalk.red(`${stats.incomplete} 个不完整`)}`))
  } else {
    console.log(chalk.gray(`模型: ${stats.downloaded} 个已下载`))
  }
  console.log()
  console.log(chalk.gray('------------------------------------'))
  console.log()

  const choices = [
    { name: '1', message: '下载模型' },
    { name: '2', message: '删除模型' },
    { name: '3', message: '查看模型列表' },
    { name: '4', message: '退出' }
  ]

  const { action } = await prompt({
    type: 'select',
    name: 'action',
    message: '选择操作',
    choices: choices.map(c => c.message),
    styles: {
      primary: chalk.cyan.bold
    }
  })

  // 解析选择
  const selectedIndex = choices.findIndex(c => c.message === action)
  return selectedIndex + 1
}

/**
 * 查看模型列表
 */
function listModels() {
  const dataPath = getDataVolumePath()

  console.log()
  console.log(chalk.bold('模型列表'))
  console.log()

  for (const model of MODELS) {
    const downloaded = isModelDownloaded(dataPath, model.id)
    const exists = isModelExists(dataPath, model.id)
    const incomplete = isModelIncomplete(dataPath, model.id)

    if (downloaded) {
      console.log(`${chalk.green('[可用]')} ${model.name} (${model.size})`)
    } else if (incomplete) {
      console.log(`${chalk.red('[不完整]')} ${model.name} (${model.size}) - 请安装 Git LFS 后执行 git lfs pull`)
    } else if (exists) {
      console.log(`${chalk.yellow('[存在]')} ${model.name} (${model.size}) - 状态未知`)
    } else {
      console.log(`${chalk.gray('[未下载]')} ${model.name} (${model.size})`)
    }
    console.log(chalk.gray(`  描述: ${model.description}`))
    console.log(chalk.gray(`  框架: ${model.framework}  |  任务: ${model.task}`))
  }

  console.log()
}

/**
 * 下载模型子菜单
 */
async function downloadModels() {
  const dataPath = getDataVolumePath()

  console.log()
  console.log(chalk.bold('下载模型'))
  console.log()

  // 检查数据目录是否存在
  if (!fs.existsSync(dataPath)) {
    console.log(chalk.yellow(`数据目录不存在: ${dataPath}`))
    console.log(chalk.yellow('请先设置挂载路径: dmla data mount <path>'))
    return
  }

  // 检查 Git 环境
  try {
    execSync('git --version', { stdio: 'pipe' })
  } catch {
    console.log(chalk.red('❌ Git 未安装'))
    console.log(chalk.yellow('下载模型需要 Git，请先安装: https://git-scm.com/downloads'))
    return
  }

  // 构建选项列表
  const choices = MODELS.map((model, index) => {
    const downloaded = isModelDownloaded(dataPath, model.id)
    const incomplete = isModelIncomplete(dataPath, model.id)

    let message = `${model.name} (${model.size})`
    if (downloaded) {
      message += ' [已下载]'
    } else if (incomplete) {
      message += ' [不完整-可重新下载]'
    }

    return {
      name: index.toString(),
      message,
      disabled: downloaded  // 完整下载的才禁用，不完整的可以重新下载
    }
  })

  // 操作提示
  console.log(chalk.gray('操作: 上下键移动，空格勾选/取消，回车确认，ESC 返回'))
  console.log()

  try {
    const { selected } = await prompt({
      type: 'multiselect',
      name: 'selected',
      message: '选择要下载的模型',
      choices,
      hint: '空格选择，回车确认下载',
      warn: '已下载',
      styles: {
        primary: chalk.cyan.bold
      }
    })

    if (!selected || selected.length === 0) {
      console.log(chalk.yellow('未选择任何模型'))
      return
    }

    // 下载选中的模型
    for (const indexStr of selected) {
      const index = parseInt(indexStr)
      const model = MODELS[index]

      console.log()
      console.log(chalk.cyan('────────────────────────────────────'))

      // 检查是否已完整下载
      if (isModelDownloaded(dataPath, model.id)) {
        console.log(chalk.yellow(`${model.name} 已完整下载，跳过`))
        continue
      }

      // 检查是否有不完整的数据，需要先删除
      if (isModelIncomplete(dataPath, model.id)) {
        console.log(chalk.yellow(`${model.name} 存在不完整数据，将删除后重新下载...`))
        const targetDir = path.join(dataPath, model.targetDir)
        fs.rmSync(targetDir, { recursive: true, force: true })
      }

      await downloadModel(dataPath, model)
    }

    console.log()
    console.log(chalk.cyan('────────────────────────────────────'))
    console.log(chalk.green('所有选中的模型已处理完成'))
  } catch (error) {
    // 用户按 ESC 或 Ctrl+C 取消
    if (isUserCancel(error)) {
      console.log(chalk.gray('返回上一级'))
      return
    }
    throw error
  }
}

/**
 * 下载单个模型
 */
async function downloadModel(dataPath, model) {
  console.log()
  console.log(chalk.bold(`下载 ${model.name}...`))
  console.log(chalk.gray(`来源: ${model.source || 'ModelScope'}`))
  console.log(chalk.gray(`URL: ${model.url}`))
  console.log(chalk.gray(`描述: ${model.description}`))

  const targetDir = path.join(dataPath, model.targetDir)

  console.log(chalk.gray(`目标: ${targetDir}`))
  console.log()

  // 确保目标目录的父目录存在
  const parentDir = path.dirname(targetDir)
  if (!fs.existsSync(parentDir)) {
    fs.mkdirSync(parentDir, { recursive: true })
  }

  let hasGitLfs = false

  try {
    if (model.format === 'git') {
      // 使用 git clone 下载 ModelScope 模型
      console.log(chalk.gray('开始 git clone...'))
      console.log()

      // 检查并安装 Git LFS
      try {
        execSync('git lfs install', { stdio: 'pipe' })
        hasGitLfs = true
      } catch {
        console.log(chalk.red('❌ Git LFS 未安装'))
        console.log(chalk.yellow('模型使用 Git LFS 存储大文件，未安装 LFS 时只能下载指针文件'))
        console.log(chalk.yellow('模型将不可用！'))
        console.log()
        console.log(chalk.yellow('建议安装 Git LFS 后重新下载:'))
        console.log(chalk.gray('  Ubuntu/Debian: sudo apt install git-lfs'))
        console.log(chalk.gray('  macOS: brew install git-lfs'))
        console.log(chalk.gray('  Windows: https://git-lfs.github.com/'))
        console.log()

        // 提供选项：继续下载（留待后续手动拉取）或中止
        const { choice } = await prompt({
          type: 'select',
          name: 'choice',
          message: '如何处理?',
          choices: [
            '中止下载（删除不完整数据）',
            '继续下载（安装 LFS 后手动拉取: git lfs pull）'
          ],
          styles: {
            primary: chalk.cyan.bold
          }
        })

        if (choice === '中止下载（删除不完整数据）') {
          console.log(chalk.yellow('下载已中止'))
          // 创建目标目录以便后续重试（标记为不完整）
          if (!fs.existsSync(targetDir)) {
            fs.mkdirSync(targetDir, { recursive: true })
          }
          // 写入标记文件，表明数据不完整
          fs.writeFileSync(path.join(targetDir, '.lfs-incomplete'), 'Git LFS 未安装，数据不完整')
          return
        }

        console.log(chalk.gray('继续下载指针文件...'))
        console.log(chalk.yellow('⚠ 提醒: 下载完成后需安装 Git LFS 并执行 git lfs pull 才能使用模型'))
        console.log()
      }

      // 执行 git clone
      // --progress: 强制显示 git 进度（即使非 TTY）
      // GIT_LFS_SKIP_SMUDGE=1: 跳过 clone 时自动下载 LFS 文件，改为后续 git lfs pull 单独下载
      await new Promise((resolve, reject) => {
        const git = spawn('git', ['clone', '--progress', model.url, targetDir], {
          stdio: 'inherit',
          env: { ...process.env, GIT_LFS_SKIP_SMUDGE: '1' }
        })

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

      // 根据 LFS 状态显示不同提示
      if (hasGitLfs) {
        console.log(chalk.green('Git 仓库克隆完成'))

        // 拉取 Git LFS 文件（GIT_LFS_FORCE_PROGRESS 强制显示下载进度）
        console.log()
        console.log(chalk.gray('下载 LFS 大文件（模型权重）...'))
        try {
          execSync('git lfs pull', { cwd: targetDir, stdio: 'inherit', env: { ...process.env, GIT_LFS_FORCE_PROGRESS: '1' } })
          console.log(chalk.green('LFS 文件拉取完成'))
        } catch (lfsError) {
          console.log(chalk.yellow(`⚠ LFS 拉取失败: ${lfsError.message}`))
          console.log(chalk.yellow('模型可能包含未下载的大文件，请手动执行: git lfs pull'))
        }
      } else {
        console.log(chalk.yellow('指针文件下载完成（数据不完整）'))
        console.log()
        console.log(chalk.red('⚠ 模型当前不可用！'))
        console.log(chalk.yellow('请按以下步骤完成下载:'))
        console.log(chalk.gray('  1. 安装 Git LFS'))
        console.log(chalk.gray(`  2. 进入目录: cd ${targetDir}`))
        console.log(chalk.gray('  3. 拉取数据: git lfs pull'))
        console.log()
      }

    } else {
      // 其他格式的下载逻辑（保留兼容性）
      const cacheDir = path.join(dataPath, 'cache', 'downloads')
      const downloadFile = path.join(cacheDir, `${model.id}.${model.format}`)

      if (!fs.existsSync(cacheDir)) {
        fs.mkdirSync(cacheDir, { recursive: true })
      }

      const curlArgs = [
        '-L',
        '-o', downloadFile,
        '--progress-bar',
        model.url
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
    }

    // 根据 LFS 状态显示不同的完成提示
    if (hasGitLfs) {
      console.log()
      console.log(chalk.green(`模型已保存到 ${targetDir}`))
    } else {
      console.log()
      console.log(chalk.yellow(`模型目录: ${targetDir} (数据不完整，暂不可用)`))
    }

    // 更新配置
    const config = readConfig()
    if (!config.installedModels) {
      config.installedModels = []
    }
    if (!config.installedModels.includes(model.id)) {
      config.installedModels.push(model.id)
    }
    // 如果 LFS 未安装，标记模型状态为不完整
    if (!hasGitLfs) {
      config.incompleteModels = config.incompleteModels || []
      if (!config.incompleteModels.includes(model.id)) {
        config.incompleteModels.push(model.id)
      }
    }
    writeConfig(config)

  } catch (error) {
    console.log()
    console.log(chalk.red(`下载失败: ${error.message}`))

    // 检查是否有不完整的数据残留
    const hasIncompleteData = fs.existsSync(targetDir)

    if (hasIncompleteData) {
      console.log(chalk.yellow(`目录 ${targetDir} 存在不完整数据`))
    }

    // 提供重试或删除选项
    while (true) {
      try {
        const choices = ['重试下载']
        if (hasIncompleteData) {
          choices.push('删除不完整数据并返回')
        }
        choices.push('保留现状并返回')

        const { choice } = await prompt({
          type: 'select',
          name: 'choice',
          message: '如何处理?',
          choices,
          styles: {
            primary: chalk.cyan.bold
          }
        })

        if (choice === '重试下载') {
          // 删除不完整数据后重试
          if (hasIncompleteData && fs.existsSync(targetDir)) {
            console.log(chalk.gray('删除不完整数据...'))
            fs.rmSync(targetDir, { recursive: true, force: true })
          }
          console.log(chalk.gray('重新下载...'))
          console.log()
          // 递归调用自身重试
          await downloadModel(dataPath, model)
          return  // 重试成功后退出
        } else if (choice === '删除不完整数据并返回') {
          fs.rmSync(targetDir, { recursive: true, force: true })
          console.log(chalk.gray('不完整数据已删除'))
          return
        } else {
          // 保留现状并返回
          if (!fs.existsSync(targetDir)) {
            fs.mkdirSync(targetDir, { recursive: true })
          }
          fs.writeFileSync(path.join(targetDir, '.lfs-incomplete'), `下载失败: ${error.message}`)
          console.log(chalk.yellow('已保留不完整数据，可稍后重试'))
          return
        }
      } catch (promptError) {
        if (isUserCancel(promptError)) {
          console.log(chalk.gray('返回上一级'))
          return
        }
        throw promptError
      }
    }
  }
}

/**
 * 删除模型子菜单
 */
async function deleteModels() {
  const dataPath = getDataVolumePath()

  console.log()
  console.log(chalk.bold('删除模型'))
  console.log()

  // 收集已下载（含不完整）的模型
  const existingModels = MODELS.filter(m => isModelExists(dataPath, m.id))

  if (existingModels.length === 0) {
    console.log(chalk.yellow('没有已下载的模型'))
    return
  }

  // 构建选项列表
  const choices = existingModels.map((model) => {
    const downloaded = isModelDownloaded(dataPath, model.id)
    const incomplete = isModelIncomplete(dataPath, model.id)

    let message = `${model.name} (${model.size})`
    if (downloaded) {
      message += ' [可用]'
    } else if (incomplete) {
      message += ' [不完整]'
    } else {
      message += ' [存在]'
    }

    return {
      name: model.id,
      message
    }
  })

  console.log(chalk.gray('操作: 上下键移动，空格勾选/取消，回车确认，ESC 返回'))
  console.log()

  try {
    const { selected } = await prompt({
      type: 'multiselect',
      name: 'selected',
      message: '选择要删除的模型',
      choices,
      hint: '空格选择，回车确认删除',
      styles: {
        primary: chalk.cyan.bold
      }
    })

    if (!selected || selected.length === 0) {
      console.log(chalk.yellow('未选择任何模型'))
      return
    }

    // 确认删除
    const selectedNames = selected.map(id => {
      const m = existingModels.find(d => d.id === id)
      return m.name
    })

    console.log()
    console.log(chalk.red(`将删除以下模型: ${selectedNames.join(', ')}`))

    const { confirm } = await prompt({
      type: 'confirm',
      name: 'confirm',
      message: '确认删除?',
      initial: false
    })

    if (!confirm) {
      console.log(chalk.yellow('操作已取消'))
      return
    }

    // 执行删除
    for (const modelId of selected) {
      const model = MODELS.find(m => m.id === modelId)
      const targetDir = path.join(dataPath, model.targetDir)

      if (fs.existsSync(targetDir)) {
        fs.rmSync(targetDir, { recursive: true, force: true })
        console.log(chalk.green(`已删除: ${model.name}`))
      }

      // 更新配置
      const config = readConfig()
      if (config.installedModels) {
        config.installedModels = config.installedModels.filter(id => id !== modelId)
      }
      if (config.incompleteModels) {
        config.incompleteModels = config.incompleteModels.filter(id => id !== modelId)
      }
      writeConfig(config)
    }

    console.log()
    console.log(chalk.green('删除完成'))
  } catch (error) {
    if (isUserCancel(error)) {
      console.log(chalk.gray('返回上一级'))
      return
    }
    throw error
  }
}

/**
 * 运行模型管理 TUI
 */
export async function runModelTUI() {
  showBanner()

  // 处理 enquirer 在 Ctrl+C 时抛出的 ERR_USE_AFTER_CLOSE
  const handleUncaught = (err) => {
    if (err.code === 'ERR_USE_AFTER_CLOSE') {
      console.log()
      console.log(chalk.gray('已退出模型管理'))
      console.log()
      process.exit(0)
    }
    throw err
  }
  process.on('uncaughtException', handleUncaught)

  let dataPath = getDataVolumePath()

  // 确保配置目录存在
  if (!fs.existsSync(DMLA_CONFIG_DIR)) {
    fs.mkdirSync(DMLA_CONFIG_DIR, { recursive: true })
  }

  // 如果数据目录不存在，提示创建
  if (!fs.existsSync(dataPath)) {
    console.log(chalk.yellow(`数据目录不存在: ${dataPath}`))
    console.log()

    try {
      const { create } = await prompt({
        type: 'confirm',
        name: 'create',
        message: '是否创建数据目录?',
        initial: true
      })

      if (create) {
        fs.mkdirSync(dataPath, { recursive: true })
        ensureModelDirStructure(dataPath)
        console.log(chalk.green(`数据目录已创建: ${dataPath}`))
      }
    } catch (error) {
      if (isUserCancel(error)) {
        console.log(chalk.gray('已退出模型管理'))
        console.log()
        return
      }
      throw error
    }
  } else {
    // 确保模型子目录存在
    ensureModelDirStructure(dataPath)
  }

  // 主循环
  while (true) {
    console.log()
    try {
      const action = await showMainMenu(dataPath)

      switch (action) {
        case 1:
          try {
            await downloadModels()
          } catch (error) {
            if (isUserCancel(error)) {
              console.log(chalk.gray('返回主菜单'))
            } else {
              throw error
            }
          }
          break
        case 2:
          try {
            await deleteModels()
          } catch (error) {
            if (isUserCancel(error)) {
              console.log(chalk.gray('返回主菜单'))
            } else {
              throw error
            }
          }
          break
        case 3:
          listModels()
          break
        case 4:
          console.log()
          console.log(chalk.gray('已退出模型管理'))
          console.log()
          process.off('uncaughtException', handleUncaught)
          return
      }

      // 刷新路径（可能在操作中修改了）
      dataPath = getDataVolumePath()
    } catch (error) {
      // 主菜单按 ESC 取消 -> 退出程序
      if (isUserCancel(error)) {
        console.log()
        console.log(chalk.gray('已退出模型管理'))
        console.log()
        process.off('uncaughtException', handleUncaught)
        return
      }
      throw error
    }
  }
}

/**
 * CLI 命令入口 (非 TUI 模式)
 */
export async function runModelCommand(subCommand, options) {
  switch (subCommand) {
    case 'list':
      listModels()
      break
    case 'download':
      await downloadModels()
      break
    case 'delete':
      await deleteModels()
      break
    default:
      // 无子命令时进入 TUI
      await runModelTUI()
  }
}

export default {
  runModelTUI,
  runModelCommand,
  MODELS,
  // 导出辅助函数供测试使用
  isModelDownloaded,
  isModelExists,
  isModelIncomplete,
  getModelStats
}
