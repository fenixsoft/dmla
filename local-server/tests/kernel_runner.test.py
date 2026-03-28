"""
kernel_runner.py 单元测试
"""

import json
import subprocess
import sys
import os

# 获取 kernel_runner.py 的路径
KERNEL_RUNNER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'kernel_runner.py')


def run_kernel(code: str, timeout: int = 30) -> dict:
    """运行 kernel_runner.py 并返回结果"""
    result = subprocess.run(
        [sys.executable, KERNEL_RUNNER_PATH, '--code', code, '--timeout', str(timeout)],
        capture_output=True,
        text=True,
        timeout=timeout + 10
    )

    if result.returncode != 0:
        raise RuntimeError(f"kernel_runner.py failed: {result.stderr}")

    return json.loads(result.stdout)


def test_simple_print():
    """测试简单的 print 语句"""
    result = run_kernel('print("Hello, World!")')

    assert result['success'] == True
    assert len(result['outputs']) >= 1

    # 找到 stdout stream 输出
    stdout_outputs = [o for o in result['outputs'] if o['type'] == 'stream' and o['name'] == 'stdout']
    assert len(stdout_outputs) >= 1
    assert 'Hello, World!' in stdout_outputs[0]['text']

    print("✓ test_simple_print passed")


def test_expression_result():
    """测试表达式结果"""
    result = run_kernel('1 + 1')

    assert result['success'] == True
    assert len(result['outputs']) >= 1

    # 检查有 execute_result 或 stream 输出
    has_result = any(o['type'] in ('execute_result', 'stream') for o in result['outputs'])
    assert has_result

    print("✓ test_expression_result passed")


def test_error_handling():
    """测试错误处理"""
    result = run_kernel('1/0')

    assert result['success'] == True  # 执行成功，但输出包含错误
    assert len(result['outputs']) >= 1

    # 找到 error 输出
    error_outputs = [o for o in result['outputs'] if o['type'] == 'error']
    assert len(error_outputs) >= 1
    assert error_outputs[0]['ename'] == 'ZeroDivisionError'

    print("✓ test_error_handling passed")


def test_multiple_prints():
    """测试多次 print"""
    result = run_kernel('print("Line 1"); print("Line 2")')

    assert result['success'] == True

    # 合并所有 stdout 输出
    stdout_text = ''.join(
        o['text'] for o in result['outputs']
        if o['type'] == 'stream' and o['name'] == 'stdout'
    )

    assert 'Line 1' in stdout_text
    assert 'Line 2' in stdout_text

    print("✓ test_multiple_prints passed")


def test_execution_time():
    """测试执行时间记录"""
    result = run_kernel('import time; time.sleep(0.1)')

    assert result['success'] == True
    assert 'executionTime' in result
    assert result['executionTime'] >= 0.1

    print("✓ test_execution_time passed")


def test_matplotlib_import():
    """测试 matplotlib 导入（不生成图片，只测试导入）"""
    result = run_kernel('import matplotlib.pyplot as plt; print("matplotlib imported")')

    assert result['success'] == True

    stdout_text = ''.join(
        o['text'] for o in result['outputs']
        if o['type'] == 'stream' and o['name'] == 'stdout'
    )
    assert 'matplotlib imported' in stdout_text

    print("✓ test_matplotlib_import passed")


if __name__ == '__main__':
    print("Running kernel_runner.py tests...")
    print()

    test_simple_print()
    test_expression_result()
    test_error_handling()
    test_multiple_prints()
    test_execution_time()
    test_matplotlib_import()

    print()
    print("All tests passed! ✓")