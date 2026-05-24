# 测试页面

## 环境检查

```python runnable gpu
import importlib

# 检查 LLM 实验所需的 Python 包
required_packages = {
    'transformers': 'HuggingFace Transformers',
    'tokenizers': 'HuggingFace Tokenizers',
    'datasets': 'HuggingFace Datasets',
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'json': 'JSON (标准库)',
}

print("=== Python 包检查 ===")
for pkg, desc in required_packages.items():
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', '内置')
        print(f"  ✅ {pkg:20s} {desc:30s} {version}")
    except ModuleNotFoundError:
        print(f"  ❌ {pkg:20s} {desc:30s} 未安装")

# 检查 transformers 内的 AutoTokenizer 是否可用
try:
    from transformers import AutoTokenizer
    print("\n✅ AutoTokenizer 可正常导入")
except ImportError as e:
    print(f"\n❌ AutoTokenizer 导入失败: {e}")

# 检查 Python 版本和运行模式
import sys
import os
print(f"\nPython: {sys.version}")
print(f"运行模式: {'Docker' if os.path.exists('/.dockerenv') else 'Native'}")
print(f"DATA_DIR: {os.environ.get('DMLA_DATA_PATH', '/data')}")
```
