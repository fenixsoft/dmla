/**
 * 命令执行封装
 * --verbose 模式下打印所有执行的外部命令，方便调试
 */
import { execSync as nodeExecSync, spawn as nodeSpawn } from 'child_process'

// 全局 verbose 开关
let verboseEnabled = false

/**
 * 启用/禁用 verbose 模式
 */
export function setVerbose(enabled) {
  verboseEnabled = !!enabled
}

/**
 * 查询 verbose 模式状态
 */
export function isVerbose() {
  return verboseEnabled
}

/**
 * 打印 verbose 日志（仅在 verbose 模式下输出）
 */
function verboseLog(cmd, args) {
  if (!verboseEnabled) return
  const fullCmd = args && args.length > 0
    ? `${cmd} ${args.map(a => `'${a}'`).join(' ')}`
    : cmd
  console.log(`[verbose] $ ${fullCmd}`)
}

/**
 * 封装 execSync，verbose 模式下打印命令
 * 参数与原生 execSync 完全一致
 */
export function execSync(command, options) {
  verboseLog(command)
  return nodeExecSync(command, options)
}

/**
 * 封装 spawn，verbose 模式下打印命令
 * 参数与原生 spawn 完全一致
 */
export function spawn(command, args, options) {
  verboseLog(command, args)
  return nodeSpawn(command, args, options)
}
