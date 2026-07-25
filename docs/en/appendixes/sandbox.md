# Building the Sandbox Environment

## Prerequisites

Ensure your system:

- Has a [Docker environment](https://docs.docker.com/engine/install) set up.
- Has a [NodeJS 20.x+ environment](https://nodejs.org/en/download) set up.
- **Optional**: The code snippets in the articles run without any setup. However, experiments in the Model Engineering Practice chapter require GPU support, which needs an NVIDIA GPU with [NVIDIA drivers](https://www.nvidia.com/en-us/drivers/) installed, meeting the driver version requirements for CUDA 13.0 GA, sufficient disk space, etc. Specifically:
    - NVIDIA driver version >= 580.
    - Docker GPU support: The host must have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed so that Docker containers can access GPU hardware.
        - Windows users using Docker Desktop have the Container Toolkit integrated automatically; no additional installation is needed.
        - Linux users (including those who install Docker Engine directly in WSL2) need to install it manually.
            <details>
            <summary>Installation Method</summary>

            ```bash
            # Configure the apt repository
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
                gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

            # Install and configure
            apt-get update && apt-get install -y nvidia-container-toolkit
            nvidia-ctk runtime configure --runtime=docker
            systemctl restart docker

            # Verify the installation (should output GPU model info)
            docker run --rm --gpus all dmla-sandbox:gpu nvidia-smi -L
            ```
            </details>
    - Storage and Memory:
        - CPU image is approximately 880 MB; GPU image approximately 11.1 GB (CUDA official image ≈ 4 GB, PyTorch GPU version ≈ 2.7 GB, vLLM ≈ 3.1 GB).
        - CPU image memory limit is 4 GB; GPU image has no memory limit, but model training typically requires at least 16 GB of VRAM (see the training chapter for details).
        - The host should also reserve at least 20 GB of disk space for storing models, checkpoints, datasets, preprocessing cache, etc.
    - Other tool requirements:
        - [Git LFS](https://git-scm.com/install/): Datasets for model training/evaluation need to be downloaded using Git LFS.
- Other dependencies (such as Jupyter Notebook Kernel, Python, NumPy, PyTorch, CUDA, etc.) are automatically provided via the Docker image and do not need to be installed separately.

## Quick Start

This document contains a large number of code examples for demonstrating machine learning algorithms and performing model training, so setting up a sandbox environment for practice is necessary.
- If you are using the online version of the documentation ([https://ai.icyfenix.cn](https://ai.icyfenix.cn)), you can run the code snippets in the articles directly with Serverless support by default. However, it is highly recommended to deploy the sandbox locally so that all code on the site (article code snippets, engineering practice chapters) can be executed on your machine. Use `DMLA-CLI` to deploy the sandbox environment:

    ``` shell
    npx @icyfenix-dmla/install@latest
    ```

    After deployment, use the following commands to start the sandbox service:
    - **CPU Mode**: Default mode, runs code using the CPU Docker image. The service capability meets the needs of running the code snippets in the articles.
    - **GPU Mode**: Runs code using the GPU Docker image. The service capability meets the needs of all code (including snippets and full experiments in the engineering practice chapters).
    - **Native Mode**: Runs code directly on the host environment without using Docker images. The service capability depends on the host's software and hardware configuration. Python, PyTorch, and CUDA need to be prepared by the user; other PIP dependencies are installed automatically.

    ``` bash
    # Start the service
    dmla start                 # Default port 3001, auto-select image, CPU preferred
    dmla start --gpu           # GPU mode
    dmla start --native        # Native mode

    dmla start --help          # View other options, such as setting the port, sync mode, development mode, etc.
    ```

    Besides starting the service, `DMLA-CLI` also offers features including stopping the service, viewing service status, downloading Docker images, downloading/managing datasets, diagnosing the environment, etc., as shown below:

    ``` bash
    # Stop the service, view status, diagnose the environment
    dmla [stop|status|doctor]

    # Deploy images, models, datasets
    dmla [images|model|data]
    ```

- If you are using the source code deployment ([https://github.com/fenixsoft/dmla](https://github.com/fenixsoft/dmla)), in addition to `DMLA-CLI`, you can also pull or build Docker images directly, and start and debug the service with the local source code.
    <details>
    <summary>Build and Start from Source</summary>

    ``` shell
    # Start the sandbox (run npm install in the repository root first)
    npm run server

    # Start the documentation service and sandbox
    npm run local

    # Pull the image
    # Pull from Docker Hub (global users), rename to the local image name after pulling
    docker pull icyfenix/dmla-sandbox:gpu
    docker tag icyfenix/dmla-sandbox:gpu dmla-sandbox:gpu

    # Or pull from Alibaba Cloud ACR (faster in China)
    docker pull crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:gpu
    docker tag crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:gpu dmla-sandbox:gpu

    # Build the image locally
    npm run build:sandbox:[cpu|gpu|all]
    ```
    </details>

## Environment Recommendations

- The current Docker GPU image supports NVIDIA RTX 20/30/40/50 series graphics cards, and A100/A800/H100/H800 professional computing cards. If your hardware is not within this range, you need to download the source code, adjust the PyTorch version, and rebuild the image (e.g., for AMD graphics cards, you need to handle PyTorch + ROCm yourself).

- All code in this project can run normally on Windows/Linux (full functionality, with some performance differences), but the author strongly recommends completing model training experiments on a **Linux** host environment. macOS or non-NVIDIA hardware environments (such as Ascend) may require additional adaptation.

- If your local hardware does not meet the requirements, consider renting GPU heterogeneous computing services from cloud providers, paying by usage to deploy the sandbox for practice (based on AutoDL's GeForce RTX 3090 GPU at approximately 1.6 RMB / hour, completing all model training is expected to cost around fifteen RMB).

- The sandbox environment defaults to `http://localhost:3001`. If you have chosen a different port or a non-local sandbox (e.g., cloud service), click the settings icon in the upper right corner of the documentation <a href="javascript:document.getElementsByTagName('button')[0].click()"><svg data-v-9eec72c3="" class="settings-icon" style="width:18px; height:18px; color:#000" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle data-v-9eec72c3="" cx="12" cy="12" r="3"></circle><path data-v-9eec72c3="" d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg></a> to manually enter the sandbox address.

::: danger Security Notice

Since the sandbox's function is to receive and execute Python code from external sources, its security primarily relies on Docker container's Linux namespace isolation (PID, network, filesystem, etc.), cgroups resource limits, and the default seccomp system call filtering. **Exposing the sandbox service directly to the public internet may pose security risks.** It is recommended that you run the sandbox locally or on a cloud service without sensitive data.
:::

## Data Management

To facilitate experiment data management and reuse of training records, this project provides data persistence functionality, supporting automatic/manual dataset downloads, model saving, etc. The data directory can be customized via `dmla data`; if not set, it defaults to the `~/dmla-data` directory on the host machine. Below is the complete directory structure (directories are created automatically as needed, no manual intervention required):

```
~/dmla-data/
├── datasets/                          # Dataset directory
│   ├── tiny-imagenet-200/             # Tiny ImageNet-200
│   ├── cifar-10/                      # CIFAR-10
│   ├── cifar-100/                     # CIFAR-100
│   ├── mnist/                         # MNIST
│   └── custom/                        # User-defined datasets
│   │   …
│
├── models/                            # Model directory
│   ├── alexnet/                       # AlexNet-related models
│   │   ├── checkpoints/               # Training intermediate checkpoints
│   │   └── final/                     # Final models
│   ├── vgg/                           # VGG series models
│   ├── resnet/                        # ResNet series models
│   ├── gan/                           # GAN models
│   ├── llm/                           # Large language models
│   └── pretrained/                    # Pretrained model downloads
│   │   …
│
├── outputs/                           # Output directory
│   ├── training_logs/                 # Training logs
│   ├── visualizations/                # Visualization results
│   └── exports/                       # Export files (ONNX, etc.)
│
└── cache/                             # Cache directory
    ├── downloads/                     # Temporary dataset download files
    ├── preprocessing/                 # Preprocessing cache
    └── torch_hub/                     # Torch Hub cache
```

## Environment Check

Use the following sample code to check whether the sandbox environment is ready. The code is editable; click the Run or Run on GPU button to execute it:

```python runnable gpu
import importlib

# Check Python packages in the sandbox environment
required_packages = {
    # Basic libraries
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

print("=== Python Package Check ===")
for pkg, desc in required_packages.items():
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', 'built-in')
        print(f"  {pkg:20s} {desc:30s} {version}")
    except ModuleNotFoundError:
        print(f"  {pkg:20s} {desc:30s} not installed")

# Check Python version and runtime mode
import sys
import os
print(f"\nPython: {sys.version}")
print(f"Runtime mode: {'Docker' if os.path.exists('/.dockerenv') else 'Native'}")
print(f"DATA_DIR: {DATA_DIR}")

# Check shared package location and contents
print(f"\n=== Shared Package Check ===")
is_docker = os.path.exists('/.dockerenv')
shared_path = None
shared_source = None

if is_docker:
    # Docker mode: get host path from DMLA_SHARED_INFO environment variable
    shared_path = '/workspace/shared'
    if os.path.isdir(shared_path):
        shared_info = os.environ.get('DMLA_SHARED_INFO', '')
        if 'mounted=true' in shared_info:
            # Extract the host path from host_path=xxx
            import re
            host_match = re.search(r'host_path=([^,]+)', shared_info)
            host_path = host_match.group(1) if host_match else 'unknown'
            shared_source = f'Volume Mount (host: {host_path})'
        else:
            shared_source = 'Built-in (Volume Mount disabled)'
else:
    # Native mode: look up from PYTHONPATH
    python_paths = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    for p in python_paths:
        candidate = os.path.join(p, 'shared')
        if os.path.isdir(candidate):
            shared_path = candidate
            shared_source = f'PYTHONPATH: {p}'
            break

if shared_path:
    print(f"  Shared package path: {shared_path}")
    print(f"     Source: {shared_source}")
    # List submodules in the shared package
    submodules = sorted([
        d for d in os.listdir(shared_path)
        if os.path.isdir(os.path.join(shared_path, d))
        and not d.startswith('_')
        and os.path.exists(os.path.join(shared_path, d, '__init__.py'))
    ])
    if submodules:
        print(f"     Submodules: {', '.join(submodules)}")
        # List classes in each submodule
        for mod in submodules:
            mod_path = os.path.join(shared_path, mod)
            classes = sorted([
                f[:-3] for f in os.listdir(mod_path)
                if f.endswith('.py') and f != '__init__.py'
            ])
            if classes:
                print(f"       {mod}: {', '.join(classes)}")
else:
    print(f"  Shared package not found (some chapter code will not be able to reuse class definitions)")


# Check hardware information
import multiprocessing
import torch

print("\n=== Hardware Information ===")
print(f"CPU cores: {multiprocessing.cpu_count()}")
try:
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                mem_gb = int(line.split()[1]) / 1024 / 1024
                print(f"Memory: {mem_gb:.1f} GB")
                break
except Exception:
    pass

if torch.cuda.is_available():
    print(f"\n=== GPU Information ===")
    print(f"CUDA version: {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"   VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"   Compute capability: {props.major}.{props.minor}")
else:
    print("GPU: Not available (currently in CPU mode)")
```
