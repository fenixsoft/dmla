/**
 * 沙箱配置管理模块
 * 提供沙箱服务地址的读取和设置功能
 *
 * 注意：配置存储在 site-config 键中（与 Settings.vue 共用）
 */

import { reactive } from 'vue'

const STORAGE_KEY = 'site-config'  // 使用统一的存储键
const FC_DEFAULT_URL = 'https://sandbox-cpu-dcheerjqde.cn-hangzhou.fcapp.run'
const LOCAL_DEFAULT_URL = 'http://localhost:3001'

// ==================== 全局运行状态 ====================
// 用于防止多段代码同时运行，支持中止操作
export const globalRunningState = reactive({
  isRunning: false,
  currentExecutionId: null,
  abortController: null
})

/**
 * 开始执行时设置全局状态
 * @param {string} executionId - 执行 ID
 * @param {AbortController} abortController - 用于中止 fetch 的控制器
 * @throws {Error} 如果已有任务在运行
 */
export function startExecution(executionId, abortController) {
  if (globalRunningState.isRunning) {
    throw new Error('已有任务在运行，请先中止')
  }
  globalRunningState.isRunning = true
  globalRunningState.currentExecutionId = executionId
  globalRunningState.abortController = abortController
}

/**
 * 结束执行时清理全局状态
 */
export function endExecution() {
  globalRunningState.isRunning = false
  globalRunningState.currentExecutionId = null
  globalRunningState.abortController = null
}

/**
 * 中止当前执行
 * @param {string} endpoint - 沙箱服务端点
 * @returns {Promise<{success: boolean, stopped: number}>}
 */
export async function abortCurrentExecution(endpoint) {
  if (!globalRunningState.isRunning) {
    return { success: true, stopped: 0 }
  }

  // 1. 中止 fetch 请求
  if (globalRunningState.abortController) {
    globalRunningState.abortController.abort()
  }

  // 2. 调用后端中止 API（中止所有，因为用户可能不知道 executionId）
  try {
    const response = await fetch(endpoint + '/api/sandbox/abort', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})  // 不传 executionId，中止所有
    })

    const result = await response.json()

    // 3. 重置状态
    endExecution()

    return result
  } catch (error) {
    // 即使 API 调用失败，也重置状态
    endExecution()
    throw error
  }
}

/**
 * 获取沙箱配置
 * @returns {{ endpoint: string, mode: string }}
 */
export function getSandboxConfig() {
  if (typeof window === 'undefined') {
    return { endpoint: FC_DEFAULT_URL, mode: 'fc' }
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      const config = JSON.parse(stored)
      const mode = config.sandboxMode || 'fc'
      // 根据模式选择对应地址，优先使用已保存的值
      const endpoint = mode === 'fc'
        ? (config.fcEndpoint || FC_DEFAULT_URL)
        : (config.customEndpoint || config.sandboxEndpoint || config.endpoint || LOCAL_DEFAULT_URL)
      return { endpoint, mode }
    }
  } catch {
    // 忽略解析错误
  }

  // 无已保存配置，默认 FC 模式
  return { endpoint: FC_DEFAULT_URL, mode: 'fc' }
}

/**
 * 获取沙箱 API 端点
 * @returns {string}
 */
export function getSandboxEndpoint() {
  return getSandboxConfig().endpoint
}

/**
 * 设置沙箱配置，分别保存 FC 和自定义地址
 * @param {{ endpoint: string, sandboxMode?: string }} config
 */
export function setSandboxConfig(config) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    const existing = localStorage.getItem(STORAGE_KEY)
    const existingConfig = existing ? JSON.parse(existing) : {}

    const mode = config.sandboxMode || 'fc'
    const endpoint = config.endpoint || (mode === 'fc' ? FC_DEFAULT_URL : LOCAL_DEFAULT_URL)

    // 合并配置，分别记忆两种模式的地址
    const newConfig = {
      ...existingConfig,
      sandboxMode: mode,
      sandboxEndpoint: endpoint,
    }
    if (mode === 'fc') {
      newConfig.fcEndpoint = endpoint
    } else {
      newConfig.customEndpoint = endpoint
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(newConfig))

    // 更新全局配置
    window.__SANDBOX_CONFIG__ = { endpoint, mode }
    window.__SITE_CONFIG__ = newConfig
  } catch (error) {
    console.error('[Sandbox Config] 保存配置失败:', error)
  }
}

/**
 * 初始化全局沙箱配置
 */
export function initSandboxConfig() {
  if (typeof window === 'undefined') {
    return
  }

  const config = getSandboxConfig()
  window.__SANDBOX_CONFIG__ = config
}

// 自动初始化
initSandboxConfig()

export { FC_DEFAULT_URL }

export default {
  getSandboxConfig,
  getSandboxEndpoint,
  setSandboxConfig,
  initSandboxConfig,
  globalRunningState,
  startExecution,
  endExecution,
  abortCurrentExecution,
  FC_DEFAULT_URL
}