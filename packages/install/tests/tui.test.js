/**
 * TUI 模块单元测试
 */
import { describe, it, expect, beforeAll } from '@jest/globals'

describe('TUI Module', () => {
  beforeAll(() => {
    // 设置测试环境
  })

  it('should have correct package.json', async () => {
    const fs = await import('fs')
    const path = await import('path')
    const pkgPath = path.resolve(__dirname, '../package.json')
    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'))

    expect(pkg.name).toBe('@dmla/install')
    expect(pkg.dependencies.enquirer).toBeDefined()
  })

  it('environment module should export checkEnvironment', async () => {
    const envModule = await import('../src/modules/environment.js')
    expect(envModule.checkEnvironment).toBeDefined()
    expect(typeof envModule.checkEnvironment).toBe('function')
  })

  it('docker module should export pullImages', async () => {
    const dockerModule = await import('../src/modules/docker.js')
    expect(dockerModule.pullImages).toBeDefined()
    expect(typeof dockerModule.pullImages).toBe('function')
  })

  it('install module should export installNpmPackage', async () => {
    const installModule = await import('../src/modules/install.js')
    expect(installModule.installNpmPackage).toBeDefined()
    expect(typeof installModule.installNpmPackage).toBe('function')
  })
})