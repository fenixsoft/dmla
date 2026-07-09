# 构建沙箱环境

## 部署前准备

确保你的系统：

- 已经部署好了 [Docker 环境](https://docs.docker.com/engine/install)。
- 已经部署好了 [NodeJS 20.x+ 环境](https://nodejs.org/en/download)。
- **可选**：文章中的代码片段无需任何设置即可运行。但模型工程实训章节的实验需使用 GPU 支持，环境需具备 NVIDIA GPU 且已经安装了 [NVIDIA 驱动](https://www.nvidia.com/en-us/drivers/)、满足 CUDA 13.0 GA 的驱动版本要求、磁盘空间等条件，具体为：
    - NVIDIA 驱动版本要求 >= 580。
    - Docker GPU 支持：需要在宿主机安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，使 Docker 容器能够访问 GPU 硬件。
        - Windows 用户使用 Docker Desktop 时，Container Toolkit 已自动集成，无需额外安装。
        - Linux 用户（包括 WSL2 中直接安装 Docker Engine 的情况）需要手动安装。
            <details>
            <summary>安装方法</summary>
            
            ```bash
            # 配置 apt 仓库
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
                gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

            # 安装并配置
            apt-get update && apt-get install -y nvidia-container-toolkit
            nvidia-ctk runtime configure --runtime=docker
            systemctl restart docker

            # 验证安装（应输出 GPU 型号信息）
            docker run --rm --gpus all dmla-sandbox:gpu nvidia-smi -L
            ```
            </details>
    - 存储与内存：
        - CPU 镜像约为 880 MB；GPU 镜像约 11.1 GB（其中 CUDA 官方镜像 ≈ 4 GB，PyTorch GPU 版本 ≈ 2.7 GB, vLLM ≈ 3.1 GB）。
        - CPU 镜像内存上限为 4 GB；GPU 镜像无内存上限，但模型训练通常至少需要 16 GB 显存（具体评估见训练章节）。
        - 宿主机还应预留 20 GB 以上空间，用于存放模型、Checkpoint、数据集、预处理缓存等内容。
    - 其他工具要求：
        - [Git LFS](https://git-scm.com/install/)：模型训练/评估的数据集需使用 Git LFS 下载。
- 其余依赖（如 Jupyter Notebook Kernel、Python、NumPy、PyTorch、CUDA 等）均通过 Docker 镜像自动提供，无需单独安装。

## 快速开始

本文档内包含大量的代码用于演示机器学习算法以及进行模型训练，因此部署一套沙箱环境用于练习是有必要的。
- 如果你使用的是互联网上部署的文档（[https://ai.icyfenix.cn](https://ai.icyfenix.cn)），默认可直接在 Serverless 服务支持下运行文章中的代码片。但强烈建议在本机部署沙箱，让网站上的全部代码（文章代码片段、工程实训章节）能够在你本地执行。可使用 `DMLA-CLI` 部署沙箱环境：

    ``` shell
    npx @icyfenix-dmla/install@latest
    ```

    部署后，可使用如下命令启动沙箱服务：
    - **CPU 模式**：默认模式，以 CPU Docker 镜像运行代码，服务能力可满足文章内代码片段的运行需要。
    - **GPU 模式**：以 GPU Docker 镜像运行代码，服务能力可满足所有代码（含代码片段、工程实训章节完整实验）的需要。
    - **Native 模式**：不使用 Docker 镜像，直接以本机原始环境运行代码，服务能力取决于本机的软硬件环境。Python、PyTorch、CUDA 需要用户自行准备，其他 PIP 依赖会自动安装。

    ``` bash
    # 启动服务
    dmla start                 # 默认端口 3001，自动选择镜像，CPU 优先
    dmla start --gpu           # GPU 模式
    dmla start --native        # Native 模式

    dmla start --help          # 查看其他功能，如设置端口、设置同步模式、设置开发模式等
    ```

    除启动服务外，`DMLA-CLI` 的其余功能还包括停止服务、查看服务状态、下载 Docker 镜像、下载/管理数据集、诊断环境等，如下所示：

    ``` bash
    # 停止服务、查看状态、环境诊断
    dmla [stop|status|doctor]

    # 部署镜像、模型、数据集
    dmla [images|model|data]
    ```

- 如果你使用的是源码部署（[https://github.com/fenixsoft/dmla](https://github.com/fenixsoft/dmla)），除 `DMLA-CLI` 外，也可以直接拉取或者编译 Docker 镜像、用本地源码启动和调试服务。
    <details>
    <summary>以源码编译启动</summary>
    
    ``` shell
    # 启动沙箱（需先在仓库根目录执行 npm install 安装依赖）
    npm run server

    # 启动文档服务和沙箱
    npm run local

    # 拉取镜像
    # 从 Docker Hub 拉取（全球用户），拉取后需重命名为本地镜像名
    docker pull icyfenix/dmla-sandbox:gpu
    docker tag icyfenix/dmla-sandbox:gpu dmla-sandbox:gpu

    # 或从阿里云 ACR 拉取（国内加速）
    docker pull crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:gpu
    docker tag crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:gpu dmla-sandbox:gpu

    # 本地编译镜像
    npm run build:sandbox:[cpu|gpu|all]
    ```
    </details>

## 环境建议

- 当前 Docker GPU 镜像支持 NVIDIA RTX 20/30/40/50 系列显卡，A100/A800/H100/H800 专业计算卡。如果你的硬件不在此范畴，需要自行下载源码，调整 PyTorch 版本后重新编译镜像（譬如 AMD 显卡要自己处理 PyTorch + ROCm）。

- 本项目所有代码均可在 Windows / Linux 环境下正常运行（功能完整，性能有差距），但笔者强烈建议在 **Linux** 宿主环境下完成模型训练实验。macOS 或非 NVIDIA 硬件环境（如昇腾）可能需要额外适配。

- 如本地硬件不满足要求，可考虑租用云服务商的 GPU 异构计算服务，以按用量付费方式部署沙箱来完成练习（以 AutoDL 的 GeForce RTX 3090 GPU 约 1.6 元 / 小时计算，完成所有模型训练预计花费在十五元左右）。

- 沙箱环境默认为 `http://localhost:3001`，如果你选择了其他端口或者非本机的沙箱（譬如云服务），请点击文档右上角设置图标 <a href="javascript:document.getElementsByTagName('button')[0].click()"><svg data-v-9eec72c3="" class="settings-icon" style="width:18px; height:18px; color:#000" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle data-v-9eec72c3="" cx="12" cy="12" r="3"></circle><path data-v-9eec72c3="" d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg></a> 手动填入沙箱地址。

::: danger 安全提示

由于沙箱的功能是从外部接收并执行 Python 代码，安全防护主要依赖 Docker 容器的 Linux 命名空间隔离（PID、网络、文件系统等）、cgroups 资源限制以及默认的 seccomp 系统调用过滤，**将沙箱服务直接暴露在公网环境可能会带来安全风险**。建议你优先考虑将沙箱运行于本地或者无敏感数据的云服务中。
:::

## 数据管理

为便于管理实验数据、复用训练记录，本项目提供数据持久化功能，支持自动/手动数据集下载、模型保存等。数据目录可通过 `dmla data` 自定义，如未设置默认为宿主机的 `~/dmla-data` 目录。以下为该目录的完整数据结构（目录均会按需自动创建，无需手动处理）：

```
~/dmla-data/
├── datasets/                          # 数据集目录
│   ├── tiny-imagenet-200/             # Tiny ImageNet-200
│   ├── cifar-10/                      # CIFAR-10
│   ├── cifar-100/                     # CIFAR-100
│   ├── mnist/                         # MNIST
│   └── custom/                        # 用户自定义数据集
│   │   …
│
├── models/                            # 模型目录
│   ├── alexnet/                       # AlexNet 相关模型
│   │   ├── checkpoints/               # 训练中间 checkpoint
│   │   └── final/                     # 最终模型
│   ├── vgg/                           # VGG 系列模型
│   ├── resnet/                        # ResNet 系列模型
│   ├── gan/                           # GAN 模型
│   ├── llm/                           # 大语言模型
│   └── pretrained/                    # 预训练模型下载
│   │   …
│
├── outputs/                           # 输出目录
│   ├── training_logs/                 # 训练日志
│   ├── visualizations/                # 可视化结果
│   └── exports/                       # 导出文件 (ONNX等)
│
└── cache/                             # 缓存目录
    ├── downloads/                     # 数据集下载临时文件
    ├── preprocessing/                 # 预处理缓存
    └── torch_hub/                     # torch hub 缓存
```

## 检查环境

使用以下示例代码检查沙箱环境是否可用。代码可编辑，点击 Run 或 Run on GPU 按钮即可运行：

```python runnable gpu
import importlib

# 检查沙箱环境中的 Python 包
required_packages = {
    # 基础库
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'scipy': 'SciPy',
    'sklearn': 'scikit-learn',
    'requests': 'Requests',
    'PIL': 'Pillow',
    'cv2': 'OpenCV',
    'lmdb': 'LMDB',
    # PyTorch
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'torchaudio': 'TorchAudio',
    # HuggingFace
    'transformers': 'HuggingFace Transformers',
    'tokenizers': 'HuggingFace Tokenizers',
    'datasets': 'HuggingFace Datasets',
    # Jupyter
    'ipykernel': 'IPyKernel',
    'jupyter_client': 'Jupyter Client',
    'ipywidgets': 'IPyWidgets',
}

print("=== Python 包检查 ===")
for pkg, desc in required_packages.items():
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', '内置')
        print(f"  ✅ {pkg:20s} {desc:30s} {version}")
    except ModuleNotFoundError:
        print(f"  ❌ {pkg:20s} {desc:30s} 未安装")

# 检查 Python 版本和运行模式
import sys
import os
print(f"\nPython: {sys.version}")
print(f"运行模式: {'Docker' if os.path.exists('/.dockerenv') else 'Native'}")
print(f"DATA_DIR: {DATA_DIR}")

# 检查 shared 包位置和内容
print(f"\n=== Shared 包检查 ===")
is_docker = os.path.exists('/.dockerenv')
shared_path = None
shared_source = None

if is_docker:
    # Docker 模式：通过 DMLA_SHARED_INFO 环境变量获取宿主机路径
    shared_path = '/workspace/shared'
    if os.path.isdir(shared_path):
        shared_info = os.environ.get('DMLA_SHARED_INFO', '')
        if 'mounted=true' in shared_info:
            # 从 host_path=xxx 中提取宿主机路径
            import re
            host_match = re.search(r'host_path=([^,]+)', shared_info)
            host_path = host_match.group(1) if host_match else '未知'
            shared_source = f'Volume Mount（宿主机: {host_path}）'
        else:
            shared_source = '镜像内置（Volume Mount 已禁用）'
else:
    # Native 模式：从 PYTHONPATH 中查找
    python_paths = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    for p in python_paths:
        candidate = os.path.join(p, 'shared')
        if os.path.isdir(candidate):
            shared_path = candidate
            shared_source = f'PYTHONPATH: {p}'
            break

if shared_path:
    print(f"  ✅ Shared 包路径: {shared_path}")
    print(f"     来源: {shared_source}")
    # 列出 shared 包中的子模块
    submodules = sorted([
        d for d in os.listdir(shared_path)
        if os.path.isdir(os.path.join(shared_path, d))
        and not d.startswith('_')
        and os.path.exists(os.path.join(shared_path, d, '__init__.py'))
    ])
    if submodules:
        print(f"     子模块: {', '.join(submodules)}")
        # 列出每个子模块中的类
        for mod in submodules:
            mod_path = os.path.join(shared_path, mod)
            classes = sorted([
                f[:-3] for f in os.listdir(mod_path)
                if f.endswith('.py') and f != '__init__.py'
            ])
            if classes:
                print(f"       {mod}: {', '.join(classes)}")
else:
    print(f"  ⚠️ 未找到 shared 包（部分章节的代码将无法复用类定义）")


# 检查硬件信息
import multiprocessing
import torch

print("\n=== 硬件信息 ===")
print(f"CPU 核心数: {multiprocessing.cpu_count()}")
try:
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                mem_gb = int(line.split()[1]) / 1024 / 1024
                print(f"内存: {mem_gb:.1f} GB")
                break
except Exception:
    pass

if torch.cuda.is_available():
    print(f"\n=== GPU 信息 ===")
    print(f"CUDA 版本: {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")
else:
    print("GPU: 不可用（当前为 CPU 模式）")
```
