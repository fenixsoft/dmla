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

/**
 * 执行 Python 命令并获取输出
 * @param {string[]} args - Python 命令参数
 * @param {number} timeout - 超时时间（毫秒）
 * @returns {Promise<{success: boolean, output: string, error: string}>}
 */
async function runPythonCommand(args, timeout = 10000) {
  return new Promise((resolve) => {
    const proc = spawn('python3', args, { timeout })
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
  return new Promise((resolve) => {
    const proc = spawn('pip', args, { stdio: 'inherit' })

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
  getProgressPath
}