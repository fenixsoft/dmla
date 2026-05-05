/**
 * Native 模式环境检测模块
 *
 * 检测 Python、PyTorch、CUDA 环境，自动安装软依赖
 */
import { spawn } from 'child_process'
import chalk from 'chalk'
import path from 'path'
import { fileURLToPath } from 'url'
import os from 'os'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// 软依赖列表（缺失时自动安装）
const SOFT_DEPS = [
  'numpy',
  'pandas',
  'matplotlib',
  'scipy',
  'scikit-learn',
  'pillow',
  'opencv-python-headless',
  'jupyter_client',
  'ipykernel',
  'lmdb',
  'requests'
]

// 环境检测结果缓存
let cachedResult = null

// Python/Pip 命令名缓存（跨平台兼容）
let cachedPythonCommand = null
let cachedPipCommand = null

/**
 * 检测可用的 Python 命令名
 * Windows 下通常只有 'python'，Linux/macOS 通常有 'python3'
 * @returns {Promise<string>}
 */
export async function detectPythonCommand() {
  if (cachedPythonCommand) {
    return cachedPythonCommand
  }

  // Windows 下 spawn 需要使用 shell 才能通过 PATH 查找命令
  const spawnOptions = process.platform === 'win32'
    ? { timeout: 5000, shell: true, windowsVerbatimArguments: true }
    : { timeout: 5000 }

  console.log(`[DEBUG] Platform: ${process.platform}, spawnOptions: ${JSON.stringify(spawnOptions)}`)

  // 先尝试 python3（Linux/macOS 常用）
  console.log('[DEBUG] Trying python3 --version...')
  const tryPython3 = await new Promise((resolve) => {
    const proc = spawn('python3', ['--version'], spawnOptions)
    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', (data) => stdout += data.toString())
    proc.stderr.on('data', (data) => stderr += data.toString())

    proc.on('close', (code) => {
      console.log(`[DEBUG] python3 exit code: ${code}, stdout: "${stdout.trim()}", stderr: "${stderr.trim()}"`)
      resolve(code === 0)
    })
    proc.on('error', (err) => {
      console.log(`[DEBUG] python3 error: ${err.message}`)
      resolve(false)
    })
  })

  if (tryPython3) {
    cachedPythonCommand = 'python3'
    console.log('[DEBUG] Detected: python3')
    return cachedPythonCommand
  }

  // 再尝试 python（Windows 常用）
  console.log('[DEBUG] Trying python --version...')
  const tryPython = await new Promise((resolve) => {
    const proc = spawn('python', ['--version'], spawnOptions)
    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', (data) => stdout += data.toString())
    proc.stderr.on('data', (data) => stderr += data.toString())

    proc.on('close', (code) => {
      console.log(`[DEBUG] python exit code: ${code}, stdout: "${stdout.trim()}", stderr: "${stderr.trim()}"`)
      resolve(code === 0)
    })
    proc.on('error', (err) => {
      console.log(`[DEBUG] python error: ${err.message}`)
      resolve(false)
    })
  })

  if (tryPython) {
    cachedPythonCommand = 'python'
    console.log('[DEBUG] Detected: python')
    return cachedPythonCommand
  }

  // 都不可用
  console.log('[DEBUG] No Python command detected')
  return null
}

/**
 * 检测可用的 pip 命令名
 * Windows 下可能需要使用 'python -m pip'
 * @returns {Promise<string>}
 */
async function detectPipCommand() {
  if (cachedPipCommand) {
    return cachedPipCommand
  }

  // Windows 下 spawn 需要使用 shell 才能通过 PATH 查找命令
  const spawnOptions = process.platform === 'win32'
    ? { timeout: 5000, shell: true, windowsVerbatimArguments: true }
    : { timeout: 5000 }

  // 先尝试 pip
  const tryPip = await new Promise((resolve) => {
    const proc = spawn('pip', ['--version'], spawnOptions)
    proc.on('close', (code) => resolve(code === 0))
    proc.on('error', () => resolve(false))
  })

  if (tryPip) {
    cachedPipCommand = 'pip'
    return cachedPipCommand
  }

  // 尝试 python -m pip（Windows 常用）
  const pythonCmd = await detectPythonCommand()
  if (pythonCmd) {
    cachedPipCommand = `${pythonCmd} -m pip`
    return cachedPipCommand
  }

  return null
}

/**
 * 获取已缓存的 Python 命令名
 * @returns {string|null}
 */
export function getPythonCommand() {
  return cachedPythonCommand
}

/**
 * 获取已缓存的 Pip 命令名
 * @returns {string|null}
 */
export function getPipCommand() {
  return cachedPipCommand
}

/**
 * 执行 Python 命令并获取输出
 * @param {string[]} args - Python 命令参数
 * @param {number} timeout - 超时时间（毫秒）
 * @returns {Promise<{success: boolean, output: string, error: string}>}
 */
async function runPythonCommand(args, timeout = 10000) {
  // 确保 Python 命令已检测
  const pythonCmd = await detectPythonCommand()
  if (!pythonCmd) {
    return {
      success: false,
      output: '',
      error: 'Python 未安装或不在 PATH 中'
    }
  }

  // Windows 下 spawn 需要使用 shell 才能通过 PATH 查找命令
  const spawnOptions = process.platform === 'win32'
    ? { timeout, shell: true, windowsVerbatimArguments: true }
    : { timeout }

  return new Promise((resolve) => {
    const proc = spawn(pythonCmd, args, spawnOptions)
    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', (data) => stdout += data.toString())
    proc.stderr.on('data', (data) => stderr += data.toString())

    proc.on('close', (code) => {
      resolve({
        success: code === 0,
        output: stdout.trim(),
        error: stderr.trim()
      })
    })

    proc.on('error', (err) => {
      resolve({
        success: false,
        output: '',
        error: err.message
      })
    })
  })
}

/**
 * 执行 pip 命令
 * @param {string[]} args - pip 命令参数
 * @returns {Promise<{success: boolean, output: string}>}
 */
async function runPipCommand(args) {
  // 确保 pip 命令已检测
  const pipCmd = await detectPipCommand()
  if (!pipCmd) {
    return { success: false, output: 'pip 未安装或不在 PATH 中' }
  }

  // 处理 'python -m pip' 形式
  const cmdParts = pipCmd.split(' ')
  const pipArgs = cmdParts.length > 1 ? [...cmdParts.slice(1), ...args] : args
  const pipBin = cmdParts[0]

  // Windows 下 spawn 需要使用 shell 才能通过 PATH 查找命令
  const spawnOptions = process.platform === 'win32'
    ? { stdio: 'inherit', shell: true, windowsVerbatimArguments: true }
    : { stdio: 'inherit' }

  return new Promise((resolve) => {
    const proc = spawn(pipBin, pipArgs, spawnOptions)

    proc.on('close', (code) => {
      resolve({ success: code === 0, output: '' })
    })

    proc.on('error', (err) => {
      resolve({ success: false, output: err.message })
    })
  })
}

/**
 * 检测 Python 版本
 * @returns {Promise<{success: boolean, version: string|null, error: string|null}>}
 */
async function checkPython() {
  const result = await runPythonCommand(['--version'])

  if (!result.success) {
    return {
      success: false,
      version: null,
      error: 'Python3 未安装或不在 PATH 中。请安装 Python 3.x:\n' +
             '  Ubuntu/Debian: sudo apt install python3\n' +
             '  macOS: brew install python3\n' +
             '  Windows: 从 python.org 下载安装'
    }
  }

  // 解析版本号（Python 3.11.5）
  const versionMatch = result.output.match(/Python\s+(\d+\.\d+\.\d+)/)
  const version = versionMatch ? versionMatch[1] : result.output

  // 检查版本是否 >= 3.8
  const majorMinor = version.split('.').slice(0, 2).map(Number)
  if (majorMinor[0] < 3 || (majorMinor[0] === 3 && majorMinor[1] < 8)) {
    return {
      success: false,
      version: version,
      error: `Python 版本过低 (${version})，需要 >= 3.8。请升级 Python。`
    }
  }

  return { success: true, version, error: null }
}

/**
 * 检测 PyTorch
 * @returns {Promise<{success: boolean, version: string|null, error: string|null}>}
 */
async function checkPyTorch() {
  const code = 'import torch; print(torch.__version__)'
  const result = await runPythonCommand(['-c', code])

  if (!result.success) {
    return {
      success: false,
      version: null,
      error: 'PyTorch 未安装。请安装 PyTorch:\n' +
             '  CPU: pip install torch torchvision torchaudio\n' +
             '  GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128'
    }
  }

  return { success: true, version: result.output, error: null }
}

/**
 * 检测 CUDA 可用性
 * @returns {Promise<{available: boolean, version: string|null, deviceName: string|null}>}
 */
async function checkCUDA() {
  const code = `
import torch
import json
result = {
    'available': torch.cuda.is_available(),
    'version': str(torch.version.cuda) if torch.cuda.is_available() else None,
    'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
}
print(json.dumps(result))
`
  const result = await runPythonCommand(['-c', code])

  if (!result.success) {
    return { available: false, version: null, deviceName: null }
  }

  try {
    const info = JSON.parse(result.output)
    return {
      available: info.available,
      version: info.version,
      deviceName: info.device_name
    }
  } catch {
    return { available: false, version: null, deviceName: null }
  }
}

/**
 * 检测单个 Python 包
 * @param {string} pkgName - 包名
 * @returns {Promise<boolean>}
 */
async function checkPackage(pkgName) {
  // 特殊处理：opencv-python-headless 的导入名是 cv2
  const importName = pkgName === 'opencv-python-headless' ? 'cv2' : pkgName.replace('-', '_')
  const result = await runPythonCommand(['-c', `import ${importName}`], 5000)
  return result.success
}

/**
 * 安装 Python 包
 * @param {string} pkgName - 包名
 * @returns {Promise<boolean>}
 */
async function installPackage(pkgName) {
  console.log(chalk.gray(`   安装 ${pkgName}...`))
  const result = await runPipCommand(['install', pkgName, '--quiet'])
  return result.success
}

/**
 * 检测并安装软依赖
 * @returns {Promise<{installed: string[], failed: string[]}>}
 */
async function ensureSoftDependencies() {
  const installed = []
  const failed = []

  console.log(chalk.blue('检查 Python 依赖包...'))

  for (const pkg of SOFT_DEPS) {
    const exists = await checkPackage(pkg)
    if (!exists) {
      const success = await installPackage(pkg)
      if (success) {
        installed.push(pkg)
      } else {
        failed.push(pkg)
      }
    }
  }

  if (installed.length > 0) {
    console.log(chalk.green(`   已安装: ${installed.join(', ')}`))
  }

  if (failed.length > 0) {
    console.log(chalk.yellow(`   安装失败: ${failed.join(', ')}`))
    console.log(chalk.gray('   可手动安装: pip install ' + failed.join(' ')))
  }

  return { installed, failed }
}

/**
 * 执行完整环境检测
 * @returns {Promise<{success: boolean, pythonVersion: string, pytorchVersion: string,
 *                     gpuCapable: boolean, cudaVersion: string|null, deviceName: string|null,
 *                     installedPackages: string[], warnings: string[], error: string|null}>}
 */
export async function checkNativeEnvironment() {
  // 使用缓存
  if (cachedResult) {
    return cachedResult
  }

  console.log(chalk.blue('环境检测...'))
  console.log()

  const warnings = []

  // 1. 检测 Python
  console.log(chalk.gray('   检测 Python...'))
  const pythonResult = await checkPython()
  if (!pythonResult.success) {
    console.log(chalk.red('   ❌ Python 检测失败'))
    cachedResult = {
      success: false,
      pythonVersion: null,
      pytorchVersion: null,
      gpuCapable: false,
      cudaVersion: null,
      deviceName: null,
      installedPackages: [],
      warnings: [],
      error: pythonResult.error
    }
    return cachedResult
  }
  console.log(chalk.green(`   ✓ Python ${pythonResult.version}`))

  // 2. 检测 PyTorch
  console.log(chalk.gray('   检测 PyTorch...'))
  const pytorchResult = await checkPyTorch()
  if (!pytorchResult.success) {
    console.log(chalk.red('   ❌ PyTorch 检测失败'))
    cachedResult = {
      success: false,
      pythonVersion: pythonResult.version,
      pytorchVersion: null,
      gpuCapable: false,
      cudaVersion: null,
      deviceName: null,
      installedPackages: [],
      warnings: [],
      error: pytorchResult.error
    }
    return cachedResult
  }
  console.log(chalk.green(`   ✓ PyTorch ${pytorchResult.version}`))

  // 3. 检测 CUDA
  console.log(chalk.gray('   检测 CUDA...'))
  const cudaResult = await checkCUDA()
  const gpuCapable = cudaResult.available
  if (gpuCapable) {
    console.log(chalk.green(`   ✓ CUDA ${cudaResult.version} (${cudaResult.deviceName})`))
  } else {
    console.log(chalk.yellow('   ⚠ CUDA 不可用，将使用 CPU 模式'))
    warnings.push('CUDA 不可用，GPU 请求将被拒绝')
  }

  // 4. 自动安装软依赖
  console.log()
  const depsResult = await ensureSoftDependencies()
  console.log()

  // 构建结果
  cachedResult = {
    success: true,
    pythonVersion: pythonResult.version,
    pytorchVersion: pytorchResult.version,
    gpuCapable: gpuCapable,
    cudaVersion: cudaResult.version,
    deviceName: cudaResult.deviceName,
    installedPackages: depsResult.installed,
    warnings: warnings,
    error: null
  }

  return cachedResult
}

/**
 * 获取缓存的环境检测结果
 * @returns {object|null}
 */
export function getCachedEnvironment() {
  return cachedResult
}

/**
 * 清除缓存（用于重新检测）
 */
export function clearCache() {
  cachedResult = null
}

/**
 * 获取共享模块路径
 * @returns {string}
 */
export function getSharedModulesPath() {
  // CLI 包中的 shared_modules 目录
  return path.resolve(__dirname, '../../shared_modules')
}

/**
 * 获取 kernel_runner.py 路径
 * @returns {string}
 */
export function getKernelRunnerPath() {
  return path.resolve(__dirname, 'kernel_runner.py')
}

/**
 * 获取数据目录路径
 * @returns {string}
 */
export function getDataPath() {
  // 默认 ~/dmla-data，可通过 DMLA_DATA_PATH 环境变量覆盖
  return process.env.DMLA_DATA_PATH || path.join(os.homedir(), 'dmla-data')
}

/**
 * 获取进度文件路径
 * @returns {string}
 */
export function getProgressPath() {
  return path.join(getDataPath(), 'progress.json')
}

export default {
  checkNativeEnvironment,
  getCachedEnvironment,
  clearCache,
  getSharedModulesPath,
  getKernelRunnerPath,
  getDataPath,
  getProgressPath,
  getPythonCommand,
  getPipCommand,
  detectPythonCommand
}