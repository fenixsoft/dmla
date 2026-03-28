#!/usr/bin/env python3
"""
IPython Kernel 执行器

通过 Jupyter 消息协议执行 Python 代码，支持富输出（图片、文本、错误等）。

使用方式:
    python3 kernel_runner.py --code "print('hello')"

输出:
    JSON 格式的执行结果，包含 outputs 数组
"""

import sys
import json
import argparse
import time
import traceback
from typing import Optional

# 超时时间（秒）
DEFAULT_TIMEOUT = 60


def run_code(code: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """
    使用 IPython Kernel 执行代码

    Args:
        code: 要执行的 Python 代码
        timeout: 执行超时时间（秒）

    Returns:
        包含 success, outputs, executionTime 的字典
    """
    from jupyter_client import KernelManager

    start_time = time.time()
    deadline = start_time + timeout
    km = None
    kc = None
    outputs = []
    timed_out = False

    try:
        # 1. 启动 Kernel
        km = KernelManager()
        km.start_kernel()
        kc = km.client()
        kc.start_channels()

        # 等待 Kernel 就绪（最多 30 秒）
        ready_timeout = min(30, timeout // 2) if timeout > 2 else 1
        kc.wait_for_ready(timeout=ready_timeout)

        # 2. 执行代码
        msg_id = kc.execute(code, allow_stdin=False)

        # 3. 收集输出
        while True:
            # 检查是否已超时
            remaining = deadline - time.time()
            if remaining <= 0:
                timed_out = True
                break

            try:
                # 使用剩余时间作为超时
                msg = kc.get_iopub_msg(timeout=max(1, remaining))
            except Exception as e:
                # EmptyQueueError 或其他超时异常
                # 检查是否真的超时了
                if time.time() >= deadline:
                    timed_out = True
                break

            msg_type = msg['header']['msg_type']
            content = msg['content']

            # 忽略状态消息，只关注 idle 作为结束标志
            if msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break
                continue

            # 处理不同类型的输出
            if msg_type == 'stream':
                outputs.append({
                    'type': 'stream',
                    'name': content.get('name', 'stdout'),
                    'text': content.get('text', '')
                })

            elif msg_type == 'display_data':
                outputs.append({
                    'type': 'display_data',
                    'data': content.get('data', {}),
                    'metadata': content.get('metadata', {})
                })

            elif msg_type == 'execute_result':
                outputs.append({
                    'type': 'execute_result',
                    'data': content.get('data', {}),
                    'metadata': content.get('metadata', {}),
                    'execution_count': content.get('execution_count')
                })

            elif msg_type == 'error':
                outputs.append({
                    'type': 'error',
                    'ename': content.get('ename', 'UnknownError'),
                    'evalue': content.get('evalue', ''),
                    'traceback': content.get('traceback', [])
                })

            elif msg_type == 'clear_output':
                pass

        execution_time = time.time() - start_time

        if timed_out:
            return {
                'success': False,
                'outputs': [{
                    'type': 'error',
                    'ename': 'TimeoutError',
                    'evalue': f'Execution timed out after {timeout} seconds',
                    'traceback': [f'Execution timed out after {timeout} seconds']
                }],
                'executionTime': round(execution_time, 3)
            }

        return {
            'success': True,
            'outputs': outputs,
            'executionTime': round(execution_time, 3)
        }

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            'success': False,
            'outputs': [{
                'type': 'error',
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': traceback.format_exc().split('\n')
            }],
            'executionTime': round(execution_time, 3)
        }

    finally:
        # 清理资源
        if kc:
            try:
                kc.stop_channels()
            except Exception:
                pass

        if km:
            try:
                km.shutdown_kernel(now=True)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description='IPython Kernel 执行器')
    parser.add_argument('--code', type=str, required=True, help='要执行的 Python 代码')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='执行超时时间（秒）')

    args = parser.parse_args()

    result = run_code(args.code, args.timeout)

    # 输出 JSON 结果到 stdout
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()