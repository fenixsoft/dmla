/**
 * CLI 命令单元测试
 */
import { describe, it, expect, beforeAll } from '@jest/globals'

// 基础测试：确保模块可导入
describe('CLI Module', () => {
  beforeAll(() => {
    // 设置测试环境
  })

  it('should have correct package.json', async () => {
    const fs = await import('fs')
    const path = await import('path')
    const pkgPath = path.resolve(__dirname, '../package.json')
    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'))

    expect(pkg.name).toBe('@icyfenix-dmla/cli')
    expect(pkg.bin).toBeDefined()
    expect(pkg.bin.dmla).toBe('./bin/dmla.js')
  })

  it('should have commander dependency', async () => {
    const fs = await import('fs')
    const path = await import('path')
    const pkgPath = path.resolve(__dirname, '../package.json')
    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'))

    expect(pkg.dependencies.commander).toBeDefined()
  })
})