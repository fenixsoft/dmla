/**
 * API 端点测试
 * 测试沙箱 API 的基本功能
 */
import request from 'supertest'
import { app } from '../src/index.js'

// 跳过需要 Docker 的测试（在 CI 环境中可能没有 Docker）
const describeIfDocker = process.env.DOCKER_AVAILABLE === 'true' ? describe : describe.skip

describe('API 健康检查', () => {
  test('GET /api/health 应返回 200', async () => {
    const response = await request(app).get('/api/health')

    expect(response.status).toBe(200)
    expect(response.body).toHaveProperty('status', 'ok')
    expect(response.body).toHaveProperty('timestamp')
  })
})

describe('沙箱健康检查', () => {
  test('GET /api/sandbox/health 应返回状态信息', async () => {
    const response = await request(app).get('/api/sandbox/health')

    expect(response.status).toBe(200)
    expect(response.body).toHaveProperty('status')
    expect(response.body).toHaveProperty('images')
    expect(response.body.images).toHaveProperty('cpu')
    expect(response.body.images).toHaveProperty('gpu')
    expect(response.body).toHaveProperty('gpu')
  })
})

describe('GPU 状态检查', () => {
  test('GET /api/sandbox/gpu 应返回 GPU 可用性', async () => {
    const response = await request(app).get('/api/sandbox/gpu')

    expect(response.status).toBe(200)
    expect(response.body).toHaveProperty('available')
    expect(typeof response.body.available).toBe('boolean')
    expect(response.body).toHaveProperty('message')
  })
})

describe('代码执行接口', () => {
  test('POST /api/sandbox/run 无代码参数应返回 400', async () => {
    const response = await request(app)
      .post('/api/sandbox/run')
      .send({})

    expect(response.status).toBe(400)
    expect(response.body).toHaveProperty('success', false)
    expect(response.body.error).toContain('Missing or invalid code')
  })

  test('POST /api/sandbox/run 空代码应返回 400', async () => {
    const response = await request(app)
      .post('/api/sandbox/run')
      .send({ code: '' })

    expect(response.status).toBe(400)
    expect(response.body).toHaveProperty('success', false)
  })

  test('POST /api/sandbox/run 代码过长应返回 400', async () => {
    const longCode = 'x'.repeat(100001)

    const response = await request(app)
      .post('/api/sandbox/run')
      .send({ code: longCode })

    expect(response.status).toBe(400)
    expect(response.body.error).toContain('Code too long')
  })

  // 仅在 Docker 可用时运行
  describeIfDocker('Docker 执行测试', () => {
    test('POST /api/sandbox/run 执行简单 Python 代码（新格式）', async () => {
      const response = await request(app)
        .post('/api/sandbox/run')
        .send({ code: 'print("Hello, World!")', useGpu: false })

      // 如果镜像不存在，返回 503
      if (response.status === 503) {
        console.log('沙箱镜像不存在，跳过测试')
        return
      }

      expect(response.status).toBe(200)
      expect(response.body).toHaveProperty('success', true)
      expect(response.body).toHaveProperty('outputs')
      expect(response.body).toHaveProperty('executionTime')
      expect(Array.isArray(response.body.outputs)).toBe(true)

      // 检查输出内容
      const stdoutOutputs = response.body.outputs.filter(
        o => o.type === 'stream' && o.name === 'stdout'
      )
      expect(stdoutOutputs.length).toBeGreaterThan(0)
      expect(stdoutOutputs[0].text).toContain('Hello, World!')
    })

    test('POST /api/sandbox/run 执行 NumPy 计算', async () => {
      const code = `
import numpy as np
arr = np.array([1, 2, 3])
print(f"Sum: {np.sum(arr)}")
`
      const response = await request(app)
        .post('/api/sandbox/run')
        .send({ code, useGpu: false })

      if (response.status === 503) {
        console.log('沙箱镜像不存在，跳过测试')
        return
      }

      expect(response.status).toBe(200)
      expect(response.body).toHaveProperty('success', true)
      expect(response.body).toHaveProperty('outputs')

      // 合并所有 stdout 输出
      const stdoutText = response.body.outputs
        .filter(o => o.type === 'stream' && o.name === 'stdout')
        .map(o => o.text)
        .join('')

      expect(stdoutText).toContain('Sum: 6')
    })

    test('POST /api/sandbox/run 执行错误代码应返回结构化错误', async () => {
      const response = await request(app)
        .post('/api/sandbox/run')
        .send({ code: '1/0', useGpu: false })

      if (response.status === 503) {
        console.log('沙箱镜像不存在，跳过测试')
        return
      }

      expect(response.status).toBe(200)
      expect(response.body).toHaveProperty('outputs')

      // 检查有错误输出
      const errorOutputs = response.body.outputs.filter(o => o.type === 'error')
      expect(errorOutputs.length).toBeGreaterThan(0)

      const error = errorOutputs[0]
      expect(error).toHaveProperty('ename')
      expect(error).toHaveProperty('evalue')
      expect(error).toHaveProperty('traceback')
      expect(error.ename).toBe('ZeroDivisionError')
    })

    test('POST /api/sandbox/run matplotlib 图片输出', async () => {
      const code = `
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.savefig('/tmp/test.png')
print("Image saved")
`
      const response = await request(app)
        .post('/api/sandbox/run')
        .send({ code, useGpu: false })

      if (response.status === 503) {
        console.log('沙箱镜像不存在，跳过测试')
        return
      }

      expect(response.status).toBe(200)
      expect(response.body).toHaveProperty('success', true)
    }, 30000)  // 增加超时时间
  })
})