/**
 * 构建脚本：将 local-server 代码复制到 CLI 包中
 * 用于 npm 发布时包含完整的服务器代码和共享模块
 */
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const rootDir = path.resolve(__dirname, '../../..')
const localServerSrc = path.resolve(rootDir, 'local-server/src')
const localServerShared = path.resolve(rootDir, 'local-server/shared_modules')
const cliServerDest = path.resolve(__dirname, '../src/server')
const cliSharedDest = path.resolve(__dirname, '../shared_modules')

console.log('📦 构建 CLI 包...')

// 递归复制目录
function copyDir(src, dest, filter = null) {
  if (!fs.existsSync(src)) {
    console.error(`❌ 源目录不存在: ${src}`)
    return false
  }

  // 创建目标目录
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true })
  }

  const entries = fs.readdirSync(src, { withFileTypes: true })

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name)
    const destPath = path.join(dest, entry.name)

    if (entry.isDirectory()) {
      copyDir(srcPath, destPath, filter)
    } else if (entry.isFile()) {
      // 应用过滤器
      if (filter && !filter(entry.name)) {
        continue
      }
      fs.copyFileSync(srcPath, destPath)
      console.log(`   ✓ ${path.relative(rootDir, srcPath)} → ${path.relative(__dirname, destPath)}`)
    }
  }
  return true
}

// 复制服务器代码（只复制 .js 文件）
console.log('\n📋 复制服务器代码...')
console.log(`   源目录: ${localServerSrc}`)
console.log(`   目标目录: ${cliServerDest}`)
copyDir(localServerSrc, cliServerDest, (name) => name.endsWith('.js'))

// 复制共享模块（复制所有 .py 文件和 __init__.py）
console.log('\n📋 复制共享模块...')
console.log(`   源目录: ${localServerShared}`)
console.log(`   目标目录: ${cliSharedDest}`)
const sharedCopied = copyDir(localServerShared, cliSharedDest, (name) => {
  // 复制 Python 文件和初始化文件
  return name.endsWith('.py') || name === '__init__.py'
})

if (!sharedCopied) {
  console.log('⚠️ 共享模块目录不存在，跳过')
}

// 创建版本信息文件（用于 --dev 模式的版本比较）
const versionInfo = {
  buildTime: new Date().toISOString(),
  // 从 package.json 读取版本
  cliVersion: JSON.parse(fs.readFileSync(path.resolve(__dirname, '../package.json'), 'utf8')).version
}
fs.writeFileSync(
  path.resolve(__dirname, '../version.json'),
  JSON.stringify(versionInfo, null, 2)
)
console.log('\n✅ 版本信息已生成')

console.log('\n✅ CLI 包构建完成')