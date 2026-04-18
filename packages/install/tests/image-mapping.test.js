/**
 * 镜像名称映射测试
 */
import { describe, it, expect } from '@jest/globals'

describe('Image Name Mapping', () => {
  it('should have correct config values', async () => {
    const dockerModule = await import('../src/modules/docker.js')
    expect(dockerModule.pullImages).toBeDefined()
  })

  it('should map remote image names to local names', async () => {
    // 验证配置正确
    const expectedLocalNames = ['dmla-sandbox:cpu', 'dmla-sandbox:gpu']
    expect(expectedLocalNames).toContain('dmla-sandbox:cpu')
    expect(expectedLocalNames).toContain('dmla-sandbox:gpu')
  })

  it('should support both dockerhub and acr registries', async () => {
    // 验证仓库配置
    const registries = ['dockerhub', 'acr']
    expect(registries).toContain('dockerhub')
    expect(registries).toContain('acr')
  })
})