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


def output_json(data):
    """
    输出 JSON 到 stdout（原子输出，防止大 JSON 被分割）
    使用 sys.stdout.write 确保一次性输出，避免 Python print 的分割问题
    """
    try:
        json_str = json.dumps(data, ensure_ascii=False)
        sys.stdout.write(json_str + '\n')
        sys.stdout.flush()
    except Exception as e:
        # 输出失败，尝试打印警告
        print(f"Warning: Failed to output JSON: {e}")

# 调试日志文件（容器内路径）
DEBUG_LOG = '/tmp/kernel_runner.log'

def log_debug(message):
    """写入调试日志"""
    try:
        with open(DEBUG_LOG, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
    except:
        pass

# CUDA 兼容性错误诊断
CUDA_COMPAT_ERROR_SIGNATURES = [
    "no kernel image is available for execution on the device",
    "CUDA error: device-side assert triggered",
    "CUDA error: invalid device ordinal",
    "RuntimeError: CUDA error",
]

CUDA_COMPAT_SOLUTION = """
================================================================================
CUDA 兼容性错误诊断
================================================================================

错误原因: PyTorch CUDA 版本与 GPU 硬件/驱动不兼容

您遇到的错误表明 Docker 容器中的 PyTorch 版本编译时使用的 CUDA 版本
与您宿主机的 GPU 或 NVIDIA 驱动版本不匹配。

常见原因:
  1. GPU 的 Compute Capability 高于 PyTorch 支持的版本
     (例如: RTX 4090 需要 CUDA 11.8+ 的完整支持)
  2. NVIDIA 驱动版本过低，不支持容器内的 CUDA 版本
  3. PyTorch 编译时未包含您 GPU 架构的 CUDA kernel

解决方案:
  选项 1: 使用 CPU 模式运行代码
    在代码开头添加:
      device = torch.device('cpu')
    或在前端选择 "Run on CPU" 而非 "Run on GPU"

  选项 2: 重新构建兼容的 Docker 镜像
    检查您的 GPU Compute Capability:
      nvidia-smi --query-gpu=name,compute_cap --format=csv

    根据结果选择合适的 PyTorch 版本:
      - RTX 30 系列 (Ampere, sm_80): CUDA 11.1+ 即可
      - RTX 40 系列 (Ada Lovelace, sm_89): 需要 CUDA 11.8+ 或 12.x
      - H100 (Hopper, sm_90): 需要 CUDA 12.x

    修改 local-server/Dockerfile.sandbox:
      # 将 CUDA 11.8 改为 CUDA 12.1
      FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

      # 安装对应版本的 PyTorch
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

  选项 3: 升级 NVIDIA 驱动
    检查当前驱动版本:
      nvidia-smi

    如果驱动版本过低，请升级到支持 CUDA 12.x 的版本
    (通常需要 Driver Version >= 525.x)

================================================================================
"""

# 抑制导入时的 stdout 输出，避免污染 JSON 结果
import io
import os

# 将 stdout 重定向到临时缓冲区，抑制导入时的输出
_original_stdout = sys.stdout
_original_stderr = sys.stderr
_suppress_buffer = io.StringIO()

def suppress_stdout():
    """抑制 stdout 输出"""
    sys.stdout = _suppress_buffer
    sys.stderr = _suppress_buffer

def restore_stdout():
    """恢复 stdout 输出"""
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr

log_debug('Starting kernel_runner.py')
log_debug(f'Arguments: {sys.argv}')

# 在导入 matplotlib 前抑制输出
suppress_stdout()
log_debug('Suppressing stdout for imports')

# 配置 matplotlib 中文字体支持（在导入 matplotlib 之前）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['font.monospace'] = ['WenQuanYi Micro Hei', 'DejaVu Sans Mono']
matplotlib.rcParams['axes.unicode_minus'] = False

# 强制重建 matplotlib 字体缓存，确保中文字体正确识别
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

# 导入完成后恢复 stdout
restore_stdout()
log_debug('Imports completed, stdout restored')

# 超时时间（秒）
DEFAULT_TIMEOUT = 60


def is_cuda_compat_error(error_message: str) -> bool:
    """检测是否为 CUDA 兼容性错误"""
    for sig in CUDA_COMPAT_ERROR_SIGNATURES:
        if sig in error_message:
            return True
    return False


def enrich_cuda_error(error_dict: dict) -> dict:
    """增强 CUDA 兼容性错误的输出，添加诊断信息"""
    error_value = error_dict.get('evalue', '')
    if is_cuda_compat_error(error_value):
        # 添加诊断信息到 traceback
        enriched_traceback = error_dict.get('traceback', [])
        enriched_traceback.append("")
        enriched_traceback.append(CUDA_COMPAT_SOLUTION)

        return {
            'type': 'error',
            'ename': 'CUDACompatError',
            'evalue': f"CUDA 兼容性错误: {error_value}",
            'traceback': enriched_traceback
        }
    return error_dict


def run_code(code: str, timeout: int = DEFAULT_TIMEOUT, stream: bool = False) -> dict:
    """
    使用 IPython Kernel 执行代码

    Args:
        code: 要执行的 Python 代码
        timeout: 执行超时时间（秒）
        stream: 是否启用流式输出模式（实时输出每个消息）

    Returns:
        包含 success, outputs, executionTime 的字典（stream 模式下返回空字典）
    """
    log_debug(f'run_code called, code length={len(code)}, timeout={timeout}, stream={stream}')

    # 注意：不再在执行期间抑制 stdout
    # stdout 只在导入阶段抑制（避免 matplotlib 等导入输出污染结果）
    # 执行代码阶段需要恢复 stdout，让 print 输出实时发送到 iopub channel

    from jupyter_client import KernelManager

    start_time = time.time()
    deadline = start_time + timeout
    km = None
    kc = None
    outputs = []  # stream 模式下不使用，保留用于非流式模式
    timed_out = False
    msg_count = 0
    has_error = False
    final_outputs = []  # 用于最终 result 消息的 outputs 汇总

    try:
        # 1. 启动 Kernel（抑制 stdout 避免启动输出污染）
        log_debug('Creating KernelManager')
        suppress_stdout()
        log_debug('stdout suppressed for kernel startup')

        km = KernelManager()
        log_debug('Starting kernel')
        km.start_kernel()
        log_debug('Kernel started, creating client')
        kc = km.client()
        kc.start_channels()
        log_debug('Channels started')

        # 等待 Kernel 就绪（最多 30 秒）
        ready_timeout = min(30, timeout // 2) if timeout > 2 else 1
        log_debug(f'Waiting for kernel ready, timeout={ready_timeout}')
        kc.wait_for_ready(timeout=ready_timeout)
        log_debug('Kernel ready')

        # 2. 恢复 stdout，开始执行代码
        # 重要：恢复 stdout 后，print 输出可以实时发送到 iopub channel
        restore_stdout()
        log_debug('stdout restored for code execution')

        # 3. 执行代码
        log_debug('Executing code')
        msg_id = kc.execute(code, allow_stdin=False)
        log_debug(f'Code execution started, msg_id={msg_id}')

        # 4. 收集输出
        log_debug('Collecting outputs')
        while True:
            # 检查是否已超时
            remaining = deadline - time.time()
            if remaining <= 0:
                log_debug('Execution timeout reached')
                timed_out = True
                break

            try:
                # 使用剩余时间作为超时
                msg = kc.get_iopub_msg(timeout=max(1, remaining))
                msg_count += 1
            except Exception as e:
                # EmptyQueueError 或其他超时异常
                log_debug(f'get_iopub_msg exception: {type(e).__name__}: {e}')
                # 检查是否真的超时了
                if time.time() >= deadline:
                    timed_out = True
                break

            msg_type = msg['header']['msg_type']
            content = msg['content']
            log_debug(f'Message {msg_count}: type={msg_type}')

            # 忽略状态消息，只关注 idle 作为结束标志
            if msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    log_debug('Kernel idle, execution complete')
                    break
                continue

            # 处理不同类型的输出
            if msg_type == 'stream':
                stream_output = {
                    'type': 'stream',
                    'name': content.get('name', 'stdout'),
                    'text': content.get('text', '')
                }

                if stream:
                    # 流式模式：立即输出（使用原子输出防止大 JSON 被分割）
                    output_json(stream_output)
                else:
                    outputs.append(stream_output)
                log_debug(f'Stream output: {content.get("name")} len={len(content.get("text", ""))}')

            elif msg_type == 'display_data':
                display_output = {
                    'type': 'display_data',
                    'data': content.get('data', {}),
                    'metadata': content.get('metadata', {})
                }

                if stream:
                    # 流式模式：立即输出（使用原子输出防止大 JSON 被分割）
                    output_json(display_output)
                else:
                    outputs.append(display_output)
                final_outputs.append(display_output)  # 汇总到最终结果
                log_debug(f'Display data: keys={list(content.get("data", {}).keys())}')

            elif msg_type == 'execute_result':
                result_output = {
                    'type': 'execute_result',
                    'data': content.get('data', {}),
                    'metadata': content.get('metadata', {}),
                    'execution_count': content.get('execution_count')
                }

                if stream:
                    # 流式模式：立即输出（使用原子输出防止大 JSON 被分割）
                    output_json(result_output)
                else:
                    outputs.append(result_output)
                final_outputs.append(result_output)  # 汇总到最终结果
                log_debug(f'Execute result: keys={list(content.get("data", {}).keys())}')

            elif msg_type == 'error':
                # 检查是否为 CUDA 兼容性错误，增强错误信息
                error_output = {
                    'type': 'error',
                    'ename': content.get('ename', 'UnknownError'),
                    'evalue': content.get('evalue', ''),
                    'traceback': content.get('traceback', [])
                }

                # 增强 CUDA 兼容性错误
                if is_cuda_compat_error(content.get('evalue', '')):
                    error_output = enrich_cuda_error(error_output)
                    log_debug(f'Detected CUDA compatibility error, enriched output')

                has_error = True

                if stream:
                    # 流式模式：立即输出（使用原子输出防止大 JSON 被分割）
                    output_json(error_output)
                else:
                    outputs.append(error_output)
                final_outputs.append(error_output)  # 汇总到最终结果
                log_debug(f'Error: {content.get("ename")}: {content.get("evalue")}')

            elif msg_type == 'clear_output':
                log_debug('Clear output received')

        execution_time = time.time() - start_time
        log_debug(f'Execution finished, time={execution_time:.3f}s, outputs={len(outputs)}, timed_out={timed_out}')

        # 确保 stdout 已恢复（已在执行前恢复，这里是防御性调用）
        restore_stdout()
        log_debug('stdout restored for JSON output')

        # 读取 suppressed buffer 内容，记录到日志
        suppressed_content = _suppress_buffer.getvalue()
        if suppressed_content:
            log_debug(f'Suppressed output length: {len(suppressed_content)}')
            log_debug(f'Suppressed output preview: {suppressed_content[:500]}')

        if timed_out:
            timeout_result = {
                'success': False,
                'outputs': [{
                    'type': 'error',
                    'ename': 'TimeoutError',
                    'evalue': f'Execution timed out after {timeout} seconds',
                    'traceback': [f'Execution timed out after {timeout} seconds']
                }],
                'executionTime': round(execution_time, 3)
            }
            if stream:
                # 流式模式：输出超时消息（使用原子输出）
                output_json({'type': 'error', 'ename': 'TimeoutError',
                             'evalue': f'Execution timed out after {timeout} seconds',
                             'traceback': [f'Execution timed out after {timeout} seconds']})
                output_json({'type': 'result', 'success': False,
                             'executionTime': round(execution_time, 3)})
                return {}
            return timeout_result

        success = not has_error

        if stream:
            # 流式模式：输出最终结果消息（使用原子输出）
            result_msg = {
                'type': 'result',
                'success': success,
                'outputs': final_outputs,
                'executionTime': round(execution_time, 3)
            }
            output_json(result_msg)
            return {}

        return {
            'success': success,
            'outputs': outputs,
            'executionTime': round(execution_time, 3)
        }

    except Exception as e:
        execution_time = time.time() - start_time
        log_debug(f'Exception in run_code: {type(e).__name__}: {e}')
        log_debug(f'Traceback: {traceback.format_exc()}')
        restore_stdout()

        # 检查是否为 CUDA 兼容性错误
        error_output = {
            'type': 'error',
            'ename': type(e).__name__,
            'evalue': str(e),
            'traceback': traceback.format_exc().split('\n')
        }

        if is_cuda_compat_error(str(e)):
            error_output = enrich_cuda_error(error_output)
            log_debug(f'Detected CUDA compatibility error in exception, enriched output')

        if stream:
            # 流式模式：输出错误消息和结果消息（使用原子输出）
            output_json(error_output)
            output_json({'type': 'result', 'success': False,
                         'outputs': [error_output],
                         'executionTime': round(execution_time, 3)})
            return {}

        return {
            'success': False,
            'outputs': [error_output],
            'executionTime': round(execution_time, 3)
        }

    finally:
        # 确保 stdout 已恢复
        restore_stdout()
        log_debug('stdout restored in finally')

        # 清理资源
        if kc:
            try:
                kc.stop_channels()
                log_debug('Channels stopped')
            except Exception as e:
                log_debug(f'Error stopping channels: {e}')

        if km:
            try:
                km.shutdown_kernel(now=True)
                log_debug('Kernel shutdown')
            except Exception as e:
                log_debug(f'Error shutting down kernel: {e}')


def check_cuda_compatibility():
    """
    快速检查 CUDA 兼容性

    Returns:
        dict: 包含兼容性状态和详细信息
    """
    result = {
        'compatible': True,
        'cuda_available': False,
        'driver_version': None,
        'pytorch_cuda_version': None,
        'device_name': None,
        'issues': []
    }

    try:
        import torch

        result['pytorch_version'] = torch.__version__
        result['cuda_available'] = torch.cuda.is_available()

        if result['cuda_available']:
            result['pytorch_cuda_version'] = torch.version.cuda
            result['device_name'] = torch.cuda.get_device_name(0)

            # 尝试简单的 CUDA 操作
            try:
                x = torch.randn(10, 10, device='cuda')
                y = x + x
                torch.cuda.synchronize()
                result['test_passed'] = True
            except RuntimeError as e:
                if is_cuda_compat_error(str(e)):
                    result['compatible'] = False
                    result['test_passed'] = False
                    result['issues'].append(str(e))
                else:
                    result['compatible'] = False
                    result['test_passed'] = False
                    result['issues'].append(f"CUDA operation failed: {e}")
        else:
            result['compatible'] = False
            result['issues'].append("CUDA not available in PyTorch")

    except ImportError:
        result['compatible'] = False
        result['issues'].append("PyTorch not installed")
    except Exception as e:
        result['compatible'] = False
        result['issues'].append(f"Check failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description='IPython Kernel 执行器')
    parser.add_argument('--code', type=str, required=True, help='要执行的 Python 代码')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='执行超时时间（秒）')
    parser.add_argument('--check-cuda', action='store_true', help='仅检查 CUDA 兼容性')
    parser.add_argument('--stream', action='store_true', help='启用流式输出模式（实时输出每个消息）')

    args = parser.parse_args()

    # CUDA 兼容性检查模式
    if args.check_cuda:
        result = check_cuda_compatibility()
        print(json.dumps(result, ensure_ascii=False))
        return

    result = run_code(args.code, args.timeout, stream=args.stream)

    # 非流式模式：输出 JSON 结果到 stdout
    if not args.stream:
        output_json = json.dumps(result, ensure_ascii=False)
        log_debug(f'Final JSON output length: {len(output_json)}')
        log_debug(f'Final JSON preview: {output_json[:500]}')
        print(output_json)


if __name__ == '__main__':
    main()