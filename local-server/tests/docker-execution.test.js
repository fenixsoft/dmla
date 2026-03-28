/**
 * Docker 执行测试
 * 测试沙箱容器的创建、执行和销毁
 *
 * 注意：这些测试需要 Docker 守护进程运行
 * 运行前确保：DOCKER_AVAILABLE=true
 */
import { runPythonCode, checkGPUAvailable, checkImageExists } from '../src/sandbox.js'

// 跳过整个测试文件如果没有 Docker
const describeIfDocker = process.env.DOCKER_AVAILABLE === 'true' ? describe : describe.skip

// 辅助函数：从 outputs 中提取 stdout 文本
function getStdoutText(outputs) {
  return outputs
    .filter(o => o.type === 'stream' && o.name === 'stdout')
    .map(o => o.text)
    .join('')
}

// 辅助函数：检查是否有错误输出
function hasErrorOutput(outputs) {
  return outputs.some(o => o.type === 'error')
}

// 辅助函数：获取错误信息
function getErrorOutput(outputs) {
  const error = outputs.find(o => o.type === 'error')
  return error || null
}

describeIfDocker('Docker 沙箱执行', () => {
  describe('镜像检查', () => {
    test('checkImageExists 应返回布尔值', async () => {
      const cpuExists = await checkImageExists(false)
      const gpuExists = await checkImageExists(true)

      expect(typeof cpuExists).toBe('boolean')
      expect(typeof gpuExists).toBe('boolean')
    })
  })

  describe('GPU 检测', () => {
    test('checkGPUAvailable 应返回布尔值', async () => {
      const result = await checkGPUAvailable()

      expect(typeof result).toBe('boolean')
    })
  })

  describe('代码执行（新格式）', () => {
    // 跳过如果 CPU 镜像不存在
    const testIfImage = async (testFn) => {
      const exists = await checkImageExists(false)
      if (!exists) {
        console.log('CPU 镜像不存在，跳过测试')
        return
      }
      await testFn()
    }

    test('执行简单打印语句（结构化输出）', async () => {
      const exists = await checkImageExists(false)
      if (!exists) {
        console.log('CPU 镜像不存在，跳过测试')
        return
      }

      const result = await runPythonCode('print("test output")', false)

      expect(result).toHaveProperty('success', true)
      expect(result).toHaveProperty('outputs')
      expect(Array.isArray(result.outputs)).toBe(true)
      expect(result).toHaveProperty('executionTime')
      expect(result.executionTime).toBeGreaterThan(0)
      expect(result.gpuUsed).toBe(false)

      // 检查输出内容
      const stdoutText = getStdoutText(result.outputs)
      expect(stdoutText).toContain('test output')
    })

    test('执行语法错误的代码（结构化错误输出）', async () => {
      const exists = await checkImageExists(false)
      if (!exists) {
        console.log('CPU 镜像不存在，跳过测试')
        return
      }

      const result = await runPythonCode('print("missing quote', false)

      // 执行完成，但输出包含错误
      expect(result).toHaveProperty('outputs')

      // 检查有错误输出
      const error = getErrorOutput(result.outputs)
      expect(error).not.toBeNull()
      expect(error).toHaveProperty('ename')
      expect(error).toHaveProperty('evalue')
      expect(error).toHaveProperty('traceback')
    })

    test('执行运行时错误代码', async () => {
      const exists = await checkImageExists(false)
      if (!exists) {
        console.log('CPU 镜像不存在，跳过测试')
        return
      }

      const result = await runPythonCode('1/0', false)

      expect(result).toHaveProperty('outputs')
      const error = getErrorOutput(result.outputs)
      expect(error).not.toBeNull()
      expect(error.ename).toBe('ZeroDivisionError')
    })

    test('执行 NumPy 数组操作', async () => {
      const exists = await checkImageExists(false)
      if (!exists) {
        console.log('CPU 镜像不存在，跳过测试')
        return
      }

      const code = `
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Dot product: {np.dot(a, b)}")
`
      const result = await runPythonCode(code, false)

      expect(result.success).toBe(true)
      const stdoutText = getStdoutText(result.outputs)
      expect(stdoutText).toContain('Dot product: 32')
    })

    test('执行 matplotlib 绑图代码', async () => {
      const exists = await checkImageExists(false)
      if (!exists) {
        console.log('CPU 镜像不存在，跳过测试')
        return
      }

      const code = `
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 9])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test Plot')
plt.savefig('/tmp/test_plot.png')
print("Plot saved successfully")
`
      const result = await runPythonCode(code, false)

      expect(result.success).toBe(true)
      const stdoutText = getStdoutText(result.outputs)
      expect(stdoutText).toContain('Plot saved successfully')
    }, 30000) // 增加超时时间

    test('执行超时检测', async () => {
      const exists = await checkImageExists(false)
      if (!exists) {
        console.log('CPU 镜像不存在，跳过测试')
        return
      }

      // 无限循环，应该超时
      const result = await runPythonCode('while True: pass', false)

      // 超时应该返回错误输出
      expect(result).toHaveProperty('outputs')
      const error = getErrorOutput(result.outputs)
      expect(error).not.toBeNull()
    }, 70000) // 增加超时时间到 70 秒
  })
})