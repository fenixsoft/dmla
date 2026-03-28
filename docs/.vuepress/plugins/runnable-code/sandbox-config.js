/**
 * 沙箱配置管理模块
 * 提供沙箱服务地址的读取和设置功能
 */

const STORAGE_KEY = 'sandbox-config'
const DEFAULT_ENDPOINT = 'http://localhost:3001'

/**
 * 获取沙箱配置
 * @returns {{ endpoint: string }}
 */
export function getSandboxConfig() {
  if (typeof window === 'undefined') {
    return { endpoint: DEFAULT_ENDPOINT }
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      const config = JSON.parse(stored)
      return {
        endpoint: config.endpoint || DEFAULT_ENDPOINT
      }
    }
  } catch {
    // 忽略解析错误
  }

  return { endpoint: DEFAULT_ENDPOINT }
}

/**
 * 获取沙箱 API 端点
 * @returns {string}
 */
export function getSandboxEndpoint() {
  return getSandboxConfig().endpoint
}

/**
 * 设置沙箱配置
 * @param {{ endpoint: string }} config
 */
export function setSandboxConfig(config) {
  if (typeof window === 'undefined') {
    return
  }

  localStorage.setItem(STORAGE_KEY, JSON.stringify(config))

  // 更新全局配置
  window.__SANDBOX_CONFIG__ = config
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

export default {
  getSandboxConfig,
  getSandboxEndpoint,
  setSandboxConfig,
  initSandboxConfig
}