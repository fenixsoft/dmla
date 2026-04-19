/**
 * 构建脚本：将 local-server 代码复制到 CLI 包中
 * 用于 npm 发布时包含完整的服务器代码
 */
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const rootDir = path.resolve(__dirname, '../../..')
const localServerSrc = path.resolve(rootDir, 'local-server/src')
const cliServerDest = path.resolve(__dirname, '../src/server')

console.log('📦 构建 CLI 包...')
console.log(`   源目录: ${localServerSrc}`)
console.log(`   目标目录: ${cliServerDest}`)

// 递归复制目录
function copyDir(src, dest) {
  if (!fs.existsSync(src)) {
    console.error(`❌ 源目录不存在: ${src}`)
    process.exit(1)
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
      copyDir(srcPath, destPath)
    } else if (entry.isFile() && entry.name.endsWith('.js')) {
      fs.copyFileSync(srcPath, destPath)
      console.log(`   ✓ 复制: ${entry.name}`)
    }
  }
}

// 执行复制
copyDir(localServerSrc, cliServerDest)

console.log('✅ 服务器代码已复制到 CLI 包')