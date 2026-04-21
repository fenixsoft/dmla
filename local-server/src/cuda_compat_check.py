#!/usr/bin/env python3
"""
CUDA 兼容性检查脚本

在容器启动时检查 PyTorch CUDA 版本与宿主机 GPU 驱动的兼容性。
如果不兼容，给出明确警告和建议。

兼容性问题通常由以下原因导致：
1. PyTorch 编译时的 CUDA 版本与宿主机 CUDA 驱动版本不匹配
2. GPU 的 compute capability 高于 PyTorch 支持的版本
3. NVIDIA 驱动版本过低
"""

import sys
import os

# ANSI 颜色代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header():
    """打印检查标题"""
    print()
    print(BOLD + BLUE + "=" * 70 + RESET)
    print(BOLD + BLUE + "  DMLA Sandbox - CUDA 兼容性检查" + RESET)
    print(BOLD + BLUE + "=" * 70 + RESET)
    print()

def check_nvidia_driver():
    """检查 NVIDIA 驱动版本"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None, "nvidia-smi 命令执行失败"

        # 解析驱动版本
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line:
                # 格式: "Driver Version: 535.104.05 CUDA Version: 12.2"
                parts = line.split()
                driver_version = None
                cuda_version = None
                for i, part in enumerate(parts):
                    if part == 'Driver' and i + 2 < len(parts):
                        driver_version = parts[i + 2]
                    if part == 'CUDA' and i + 2 < len(parts):
                        cuda_version = parts[i + 2]
                return driver_version, cuda_version

        return None, "无法解析驱动版本"
    except FileNotFoundError:
        return None, "nvidia-smi 未找到，请确保 NVIDIA 驱动已安装"
    except subprocess.TimeoutExpired:
        return None, "nvidia-smi 执行超时"
    except Exception as e:
        return None, f"检查驱动时出错: {e}"

def check_gpu_info():
    """检查 GPU 信息"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []

        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    compute_cap = parts[1].strip()
                    gpu_info.append({'name': name, 'compute_cap': compute_cap})
        return gpu_info
    except Exception as e:
        return []

def check_pytorch_cuda():
    """检查 PyTorch CUDA 版本"""
    try:
        import torch

        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device) if gpu_count > 0 else 'N/A'
            device_cap = torch.cuda.get_device_capability(current_device) if gpu_count > 0 else (0, 0)

            return {
                'pytorch_version': pytorch_version,
                'cuda_version': cuda_version,
                'cuda_available': True,
                'gpu_count': gpu_count,
                'device_name': device_name,
                'device_capability': device_cap
            }
        else:
            return {
                'pytorch_version': pytorch_version,
                'cuda_version': 'N/A',
                'cuda_available': False,
                'gpu_count': 0,
                'reason': 'CUDA not available'
            }
    except ImportError:
        return {
            'pytorch_version': 'N/A',
            'cuda_version': 'N/A',
            'cuda_available': False,
            'reason': 'PyTorch not installed'
        }
    except Exception as e:
        return {
            'pytorch_version': 'N/A',
            'cuda_version': 'N/A',
            'cuda_available': False,
            'reason': str(e)
        }

def test_cuda_operation():
    """测试实际 CUDA 操作"""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA 不可用"

        # 尝试简单的 CUDA 操作
        device = torch.device('cuda')
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = x + y

        # 强制同步
        torch.cuda.synchronize()

        return True, "CUDA 操作测试成功"
    except RuntimeError as e:
        if "no kernel image is available" in str(e):
            return False, f"CUDA kernel 不兼容: {e}"
        return False, f"CUDA 操作失败: {e}"
    except Exception as e:
        return False, f"测试出错: {e}"

def analyze_compatibility(driver_info, pytorch_info, test_result):
    """分析兼容性并返回诊断结果"""
    issues = []
    suggestions = []

    # 检查 1: PyTorch 是否可用
    if pytorch_info.get('reason') == 'PyTorch not installed':
        issues.append("PyTorch 未安装")
        suggestions.append("请安装 PyTorch: pip install torch")
        return issues, suggestions, 'error'

    # 检查 2: CUDA 是否可用
    if not pytorch_info.get('cuda_available', False):
        issues.append(f"CUDA 不可用: {pytorch_info.get('reason', '未知原因')}")

        # 检查是否是兼容性问题
        if 'no kernel image' in str(pytorch_info.get('reason', '')):
            suggestions.append("这是 PyTorch CUDA 版本与 GPU 不兼容的典型问题")
        else:
            suggestions.append("请检查 NVIDIA 驱动是否正确安装")
            suggestions.append("确保 docker run 时使用了 --gpus all 参数")

        return issues, suggestions, 'error'

    # 检查 3: CUDA 版本兼容性
    driver_cuda = driver_info[1] if driver_info[1] else None
    pytorch_cuda = pytorch_info.get('cuda_version', None)

    if driver_cuda and pytorch_cuda:
        try:
            driver_major = float(driver_cuda.split('.')[0])
            pytorch_major = float(pytorch_cuda.split('.')[0])

            # CUDA 驱动版本必须 >= PyTorch CUDA 版本
            if driver_major < pytorch_major:
                issues.append(f"CUDA 版本不兼容: 驱动支持 CUDA {driver_cuda}, PyTorch 需要 CUDA {pytorch_cuda}")
                suggestions.append(f"请升级 NVIDIA 驱动以支持 CUDA {pytorch_cuda}+")
                suggestions.append(f"或使用较低 CUDA 版本的 PyTorch (如 cu{int(driver_major)}{int(driver_cuda.split('.')[1])})")
        except:
            pass

    # 检查 4: Compute Capability
    gpu_info = check_gpu_info()
    if gpu_info and pytorch_info.get('device_capability'):
        device_cap = pytorch_info['device_capability']
        device_cap_str = f"{device_cap[0]}.{device_cap[1]}"

        # 检查 PyTorch 是否支持该 compute capability
        # PyTorch cu118 支持最高到 sm_90 (H100), 但某些版本可能不支持更新的架构
        try:
            cap_num = device_cap[0] * 10 + device_cap[1]
            # cu118 支持的架构: 37, 50, 60, 70, 80, 90
            supported_caps = [37, 50, 60, 70, 80, 90]

            if cap_num not in supported_caps and cap_num > 90:
                issues.append(f"GPU Compute Capability ({device_cap_str}) 可能不被当前 PyTorch CUDA 版本支持")
                suggestions.append(f"您的 GPU ({gpu_info[0]['name'] if gpu_info else '未知'}) 计算能力为 {device_cap_str}")
                suggestions.append("请尝试以下解决方案:")
                suggestions.append("  1. 使用更新版本的 PyTorch (如 CUDA 12.1+)")
                suggestions.append("  2. 或使用 CPU 模式运行代码")
        except:
            pass

    # 检查 5: 实际操作测试
    if test_result[0] is False:
        issues.append(f"CUDA 操作测试失败: {test_result[1]}")
        if "no kernel image" in test_result[1]:
            suggestions.append("这是典型的 GPU 架构与 PyTorch 不兼容问题")
            suggestions.append("解决方案:")
            suggestions.append("  1. 重新构建镜像，使用与您 GPU 兼容的 PyTorch 版本")
            suggestions.append("  2. 在代码中使用 CPU 模式: device = torch.device('cpu')")
            suggestions.append("  3. 或降级到支持的 CUDA 版本")
        return issues, suggestions, 'error'

    # 确定状态
    if len(issues) == 0:
        return [], [], 'ok'
    elif any('操作测试失败' in i for i in issues):
        return issues, suggestions, 'error'
    else:
        return issues, suggestions, 'warning'

def print_report(driver_info, pytorch_info, gpu_info, test_result, issues, suggestions, status):
    """打印完整的检查报告"""

    # 基本信息
    print(BOLD + "系统信息:" + RESET)
    print("-" * 50)

    # 驱动信息
    if driver_info[0]:
        print(f"  NVIDIA 驱动版本:   {GREEN}{driver_info[0]}{RESET}")
        print(f"  驱动支持 CUDA:     {GREEN}{driver_info[1]}{RESET}")
    else:
        print(f"  NVIDIA 驱动:       {YELLOW}未检测到 ({driver_info[1]}){RESET}")

    # GPU 信息
    if gpu_info:
        print(f"  检测到 GPU 数量:   {GREEN}{len(gpu_info)}{RESET}")
        for i, gpu in enumerate(gpu_info):
            print(f"    GPU {i}: {gpu['name']} (Compute Cap: {gpu['compute_cap']})")
    else:
        print(f"  GPU 信息:          {YELLOW}未检测到{RESET}")

    print()
    print(BOLD + "PyTorch 信息:" + RESET)
    print("-" * 50)

    print(f"  PyTorch 版本:      {pytorch_info.get('pytorch_version', 'N/A')}")

    if pytorch_info.get('cuda_available'):
        print(f"  CUDA 可用:         {GREEN}是{RESET}")
        print(f"  PyTorch CUDA 版本: {GREEN}{pytorch_info.get('cuda_version', 'N/A')}{RESET}")
        print(f"  GPU 设备:          {GREEN}{pytorch_info.get('device_name', 'N/A')}{RESET}")
        print(f"  Compute Cap:       {GREEN}{pytorch_info.get('device_capability', (0,0))}{RESET}")
    else:
        print(f"  CUDA 可用:         {RED}否{RESET}")
        if pytorch_info.get('reason'):
            print(f"  原因:              {RED}{pytorch_info.get('reason')}{RESET}")

    print()
    print(BOLD + "操作测试:" + RESET)
    print("-" * 50)

    if test_result[0]:
        print(f"  CUDA 操作测试:     {GREEN}{test_result[1]}{RESET}")
    else:
        print(f"  CUDA 操作测试:     {RED}失败{RESET}")
        print(f"  错误信息:          {RED}{test_result[1]}{RESET}")

    # 兼容性状态
    print()
    print(BOLD + "兼容性状态:" + RESET)
    print("-" * 50)

    if status == 'ok':
        print(GREEN + BOLD + "  ✓ CUDA 环境完全兼容，GPU 加速可用" + RESET)
    elif status == 'warning':
        print(YELLOW + BOLD + "  ⚠ CUDA 环境存在潜在问题，建议检查" + RESET)
        for issue in issues:
            print(f"    - {YELLOW}{issue}{RESET}")
    else:
        print(RED + BOLD + "  ✗ CUDA 环境不兼容，GPU 加速不可用" + RESET)
        for issue in issues:
            print(f"    - {RED}{issue}{RESET}")

    # 建议
    if suggestions:
        print()
        print(BOLD + "建议解决方案:" + RESET)
        print("-" * 50)
        for i, sug in enumerate(suggestions, 1):
            print(f"  {i}. {sug}")

    # CPU fallback 提示
    if status == 'error':
        print()
        print(BLUE + BOLD + "提示: 您仍然可以使用 CPU 模式运行代码" + RESET)
        print("  在代码中设置: device = torch.device('cpu')")
        print("  或在运行时选择 'Run on CPU' 而非 'Run on GPU'")

    print()
    print(BOLD + BLUE + "=" * 70 + RESET)
    print()

def main():
    """主检查流程"""
    print_header()

    # 执行各项检查
    driver_info = check_nvidia_driver()
    pytorch_info = check_pytorch_cuda()
    gpu_info = check_gpu_info()
    test_result = test_cuda_operation()

    # 分析兼容性
    issues, suggestions, status = analyze_compatibility(driver_info, pytorch_info, test_result)

    # 打印报告
    print_report(driver_info, pytorch_info, gpu_info, test_result, issues, suggestions, status)

    # 返回状态码 (0=ok, 1=warning, 2=error)
    if status == 'ok':
        return 0
    elif status == 'warning':
        return 1
    else:
        return 2

if __name__ == '__main__':
    exit_code = main()

    # 允许容器继续运行，即使 CUDA 不兼容（用户可能只用 CPU）
    # 但设置环境变量供其他程序检查
    if exit_code == 2:
        os.environ['DMLA_CUDA_COMPAT'] = 'error'
        print("容器将继续运行，但 GPU 加速不可用")
    elif exit_code == 1:
        os.environ['DMLA_CUDA_COMPAT'] = 'warning'
    else:
        os.environ['DMLA_CUDA_COMPAT'] = 'ok'

    # 不退出，让容器继续运行
    # sys.exit(exit_code)