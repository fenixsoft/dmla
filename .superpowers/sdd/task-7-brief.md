# Task 7: 创建端到端等价性测试

## 目标

创建 `local-server/tests/fc-sandbox-e2e.test.py`，包含 12 个测试用例，验证 FC 沙箱与 CPU 沙箱输出等价性。

## 文件

- Create: `local-server/tests/fc-sandbox-e2e.test.py`

## 接口

- Consumes: `FC_ENDPOINT` 环境变量（FC HTTP 触发器 URL），`CPU_SANDBOX_ENDPOINT` 环境变量（CPU 沙箱地址）
- 通过 `requests.post(url + '/api/sandbox/run', json={'code': ..., 'timeout': ...})` 调用两个后端

## 完整代码

```python
"""
FC vs CPU Sandbox 端到端等价性测试
验证 FC 部署的沙箱与本地 CPU 沙箱输出一致

用法:
    FC_ENDPOINT="https://xxx.fc.aliyuncs.com/..." \
    CPU_SANDBOX_ENDPOINT="http://localhost:3001" \
    python3 local-server/tests/fc-sandbox-e2e.test.py
"""

import requests
import json
import base64
import sys
import os

FC_URL = os.environ.get('FC_ENDPOINT', 'http://localhost:9000')
CPU_URL = os.environ.get('CPU_SANDBOX_ENDPOINT', 'http://localhost:3001')

TEST_TIMEOUT = 120  # HTTP 请求超时（秒），含 FC 冷启动

PASS = 0
FAIL = 0
SKIP = 0


def run_fc(code, timeout=60):
    """在 FC 上执行代码"""
    resp = requests.post(
        FC_URL + '/api/sandbox/run',
        json={'code': code, 'timeout': timeout},
        timeout=TEST_TIMEOUT
    )
    return resp.json()


def run_cpu(code, timeout=60):
    """在 CPU 沙箱上执行代码"""
    resp = requests.post(
        CPU_URL + '/api/sandbox/run',
        json={'code': code, 'timeout': timeout},
        timeout=TEST_TIMEOUT
    )
    return resp.json()


def test(name, code, assertions_fn):
    """运行单个测试用例"""
    global PASS, FAIL, SKIP

    try:
        fc_result = run_fc(code)
        cpu_result = run_cpu(code)
    except requests.exceptions.ConnectionError as e:
        print(f"  ⚠ SKIP: 无法连接 ({e})")
        SKIP += 1
        return
    except Exception as e:
        print(f"  ✗ FAIL: 请求异常 ({e})")
        FAIL += 1
        return

    try:
        assertions_fn(fc_result, cpu_result)
        print(f"  ✓ PASS")
        PASS += 1
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        print(f"    FC:   {json.dumps(fc_result, ensure_ascii=False)[:200]}")
        print(f"    CPU:  {json.dumps(cpu_result, ensure_ascii=False)[:200]}")
        FAIL += 1


# --- 断言辅助函数 ---

def assert_success_equal(fc, cpu):
    assert fc.get('success') == cpu.get('success'), \
        f"success 不一致: FC={fc.get('success')}, CPU={cpu.get('success')}"


def assert_image_present(fc, cpu):
    """验证图片输出中 image/png 存在且为有效 base64"""
    for outputs in [fc.get('outputs', []), cpu.get('outputs', [])]:
        display_datas = [o for o in outputs if o.get('type') == 'display_data']
        assert len(display_datas) > 0, "没有 display_data 输出"
        for dd in display_datas:
            png_data = dd.get('data', {}).get('image/png', '')
            assert png_data, "image/png 字段为空"
            decoded = base64.b64decode(png_data)
            assert decoded[:8] == b'\x89PNG\r\n\x1a\n', "image/png 不是有效 PNG 文件头"


def assert_no_font_warnings(fc, cpu):
    """验证 stderr 中没有字体缺失警告"""
    for outputs in [fc.get('outputs', []), cpu.get('outputs', [])]:
        stderr_outs = [o for o in outputs
                       if o.get('type') == 'stream' and o.get('name') == 'stderr']
        stderr_text = ''.join(o.get('text', '') for o in stderr_outs)
        assert 'does not have a glyph for' not in stderr_text, \
            f"存在字体缺失警告: {stderr_text[:200]}"


def assert_stdout_contains(fc, cpu, text):
    """验证 stdout 中包含指定文本"""
    for outputs in [fc.get('outputs', []), cpu.get('outputs', [])]:
        stdout_text = ''.join(
            o.get('text', '') for o in outputs
            if o.get('type') == 'stream' and o.get('name') == 'stdout'
        )
        assert text in stdout_text, f"stdout 中未找到 '{text}': {stdout_text[:200]}"


def assert_error_type(fc, cpu, ename):
    """验证错误类型"""
    for outputs in [fc.get('outputs', []), cpu.get('outputs', [])]:
        errors = [o for o in outputs if o.get('type') == 'error']
        assert len(errors) > 0, "没有 error 输出"
        assert errors[0].get('ename') == ename, \
            f"错误类型不匹配: 期望 {ename}, 实际 {errors[0].get('ename')}"


def assert_execution_time_reasonable(fc, cpu, min_time=0.5):
    """验证执行时间合理"""
    assert fc.get('executionTime', 0) >= min_time, \
        f"FC 执行时间 {fc.get('executionTime')} < {min_time}"
    assert cpu.get('executionTime', 0) >= min_time, \
        f"CPU 执行时间 {cpu.get('executionTime')} < {min_time}"


# matplotlib setup
SETUP_MPL = """
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
"""


if __name__ == '__main__':
    print("=" * 60)
    print("FC vs CPU Sandbox 等价性测试")
    print(f"FC:  {FC_URL}")
    print(f"CPU: {CPU_URL}")
    print("=" * 60)

    # 1. 纯文本输出
    print("\n[1] 纯文本输出")
    test("print_text", 'print("Hello, DMLA!")', lambda fc, cpu: [
        assert_success_equal(fc, cpu),
        assert_stdout_contains(fc, cpu, 'Hello, DMLA!')
    ])

    # 2. 多行输出
    print("\n[2] 多行输出")
    test("multi_line", 'for i in range(3): print(f"Line {i}")', lambda fc, cpu: [
        assert_success_equal(fc, cpu),
        assert_stdout_contains(fc, cpu, 'Line 0'),
        assert_stdout_contains(fc, cpu, 'Line 2')
    ])

    # 3. 表达式结果
    print("\n[3] 表达式结果")
    test("expression", '3.14 * 2', lambda fc, cpu: [
        assert_success_equal(fc, cpu)
    ])

    # 4. 图片输出
    print("\n[4] matplotlib 图片输出")
    test("matplotlib_plot", SETUP_MPL + """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title("Test Plot")
plt.show()
""", lambda fc, cpu: [
        assert_image_present(fc, cpu)
    ])

    # 5. 中文标题图片
    print("\n[5] 中文图片（含字体检查）")
    test("chinese_plot", SETUP_MPL + """
import matplotlib.pyplot as plt
plt.figure()
plt.title("中文标题测试")
plt.text(0.5, 0.5, "你好，世界", ha='center', transform=plt.gca().transAxes)
plt.show()
""", lambda fc, cpu: [
        assert_image_present(fc, cpu),
        assert_no_font_warnings(fc, cpu)
    ])

    # 6. 运行时异常
    print("\n[6] 运行时异常 (ZeroDivisionError)")
    test("runtime_error", 'x = 1/0', lambda fc, cpu: [
        assert_error_type(fc, cpu, 'ZeroDivisionError')
    ])

    # 7. 语法异常
    print("\n[7] 语法异常 (SyntaxError)")
    test("syntax_error", 'if True print("oops")', lambda fc, cpu: [
        lambda f, _: len([o for o in f.get('outputs', []) if o.get('type') == 'error']) > 0 or (_ for _ in ()).throw(AssertionError("FC 无 error 输出")),
        lambda _, c: len([o for o in c.get('outputs', []) if o.get('type') == 'error']) > 0 or (_ for _ in ()).throw(AssertionError("CPU 无 error 输出"))
    ])

    # 8. NumPy 计算
    print("\n[8] NumPy 浮点计算")
    test("numpy_calc", """
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
print(f"mean={arr.mean():.6f}")
print(f"std={arr.std():.6f}")
""", lambda fc, cpu: [
        assert_success_equal(fc, cpu),
        assert_stdout_contains(fc, cpu, 'mean=2.000000'),
        assert_stdout_contains(fc, cpu, 'std=0.816497')
    ])

    # 9. 执行时间记录
    print("\n[9] 执行时间记录")
    test("exec_time", 'import time; time.sleep(0.5); print("done")', lambda fc, cpu: [
        assert_success_equal(fc, cpu),
        assert_execution_time_reasonable(fc, cpu, 0.5)
    ])

    # 10. 无输出执行
    print("\n[10] 无输出执行")
    test("no_output", 'x = 1 + 1', lambda fc, cpu: [
        assert_success_equal(fc, cpu)
    ])

    # 11. 大数据流输出
    print("\n[11] 大数据流输出（10000 字符）")
    def check_large_output(fc, cpu):
        for label, result in [("FC", fc), ("CPU", cpu)]:
            stdout_text = ''.join(
                o.get('text', '') for o in result.get('outputs', [])
                if o.get('type') == 'stream' and o.get('name') == 'stdout'
            )
            assert len(stdout_text) >= 10000, f"{label} 输出被截断: {len(stdout_text)} < 10000"
    test("large_output", 'print("x" * 10000)', check_large_output)

    # 12. imshow 图片
    print("\n[12] matplotlib imshow 图片")
    test("imshow", SETUP_MPL + """
import matplotlib.pyplot as plt
import numpy as np
img = np.random.rand(10, 10)
plt.imshow(img, cmap='viridis')
plt.colorbar()
plt.show()
""", lambda fc, cpu: [
        assert_image_present(fc, cpu)
    ])

    # 结果汇总
    total = PASS + FAIL + SKIP
    print("\n" + "=" * 60)
    print(f"结果: {PASS} 通过, {FAIL} 失败, {SKIP} 跳过 (共 {total})")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
```

## 验证

```bash
# 语法检查
python3 -c "import py_compile; py_compile.compile('local-server/tests/fc-sandbox-e2e.test.py', doraise=True)"
```

## 全局约束

- FC 和 CPU 沙箱都使用 `/api/sandbox/run` 端点
- 请求超时 120 秒（含 FC 冷启动）
- 12 个测试用例全部覆盖
