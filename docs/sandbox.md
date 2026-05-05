# 构建沙箱环境

## 部署前准备

确保你的系统：

- 已经部署好了 [Docker 环境](https://docs.docker.com/engine/install)。
- 已经部署好了 [NodeJS 20.x+ 环境](https://nodejs.org/en/download)。
- **可选**：各章节内容中的代码片段与练习题使用 CPU 即可完成练习，但模型训练的内容需使用 GPU，应具备 NVIDIA GPU 且已经安装了 [NVIDIA 驱动](https://www.nvidia.com/en-us/drivers/) 、满足 CUDA 12.8 GA 的驱动版本要求、磁盘空间等要求，具体为：
    - 驱动版本要求：
        - Linux >= 570.26
        - Windows: >= 570.65
    - 存储空间要求：
        - CPU 镜像约为 650 MB；GPU 镜像约 7.43 GB（CUDA 的官方镜像超过 4GB，PyTorch GPU 版本接近 3GB）。
        - CPU 镜像的内存上限约束为 4GB；GPU 镜像未设置内存上限，在模型训练的章节中会有内存评估的内容，通常至少需要 16GB 内存和 8GB 显存。
        - 宿主机中还应预留大于 100GB 空间，用于存放各类数据集、预处理缓存及模型 Checkpoint 等内容。
    - 其他工具要求：
        - [Git](https://git-scm.com/install/)LFS：模型训练/评估的数据集使用需使用 Git LFS 下载。
- 其余依赖（如 Jupyter Notebook Kernel、Python、Numpy、PyTorch、CUDA 等）均通过 Docker 镜像来使用，不需要单独安装。

## 快速开始

本文档内包含大量的代码用于演示机器学习算法以及进行模型训练，因此部署一套沙箱环境用于练习是有必要的。
- 如果你使用的是互联网上部署的文档（[https://ai.icyfenix.cn](https://ai.icyfenix.cn)），可以在本地运行如下命令，使用`DMLA-CLI`部署沙箱环境，让网站上的代码能够在你本地执行：

    ``` shell
    npx @icyfenix-dmla/install@latest
    ```

    部署后，使用使用如下命令启动沙箱服务：
    ``` bash
    # 启动服务
    dmla start                 # 默认端口 3001，CPU 模式
    dmla start --port 3002     # 自定义端口
    dmla start --gpu           # GPU 模式

    # 停止服务
    dmla stop

    # 查看状态
    dmla status

    # 安装镜像
    dmla install

    # 环境诊断
    dmla doctor

    # 数据管理
    dmla data                    # 进入数据管理 TUI
    ```

- 如果你使用的是源码部署（[https://github.com/fenixsoft/dmla](https://github.com/fenixsoft/dmla)），除`DMLA-CLI`外，也可以直接拉取或者编译 Docker 镜像：
    ``` shell
    # 启动沙箱
    npm run server

    # 启动文档服务和沙箱
    npm run local

    # 拉取镜像
    # 从 Docker Hub 拉取（全球用户）
    docker pull icyfenix/dmla-sandbox:gpu
    docker tag icyfenix/dmla-sandbox:gpu dmla-sandbox:gpu

    # 或从阿里云 ACR 拉取（国内加速）
    docker pull crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:gpu
    docker tag crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:gpu dmla-sandbox:gpu

    # 本地编译镜像
    npm run build:sandbox:[cpu/gpu/all]
    ```

## 环境建议

- 对于文章内容的代码片段和课后练习题算法，只需纯 CPU 环境即可运行。在进入深度学习部分后，会出现专门的模型工程训练章节，它们需要有 GPU 异构计算环境的支持，当前 Docker 镜像使用的是 PyTorch with [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)，支持 20/30/40/50 系显卡，A100/A800/H100/H800 专业计算卡。如果你的硬件不在此范畴，需要自行下载代码，调整 PyTorch 版本后重新编译镜像（譬如 AMD 显卡要自己处理 PyTorch + ROCm）。

- 基于以下无法绕过的约束，笔者建议在 **Linux** 宿主环境下进行使用 GPU 进行模型训练
    - [NVIDIA NVML](https://developer.nvidia.com/management-library-nvml) 约束：本项目的模型训练虽不会使用 NVML 去调整 GPU 硬件参数。但会用到像 [DALI](https://developer.nvidia.com/dali) 这类数据处理库，需要 NVML 支持。Windows 有明确不受支持的 NVML API，即使可用的 API 也需要用户态代理将 NVML 请求翻译成宿主机的 API 调用，性能受限。
    - Docker SHM 约束：PyTorch 的 `_new_shared` 函数在 Windows Docker 内无法正常工作，即使手动设置了 SHM，DataLoader 依然无法启用多线程数据加载。
    - WSL 2 的跨宿主机 I/O 约束：[数据管理](#数据管理)中提到，训练数据和模型是存放在宿主机，通过 Volume Mount 到容器的，由于宿主机是 NTSF 磁盘格式，要通过 9P + DrvFs 翻译，会带来大量 I/O 损耗。
    
    即便存在上述限制，本文档仍保证了代码能在 Windows / Linux 环境上测试通过（能跑，性能有差距），支持跨平台运行，在 MacOS 或者非 NVIDIA 硬件环境（譬如昇腾环境）有可能需要额外的处理。

- 如本地硬件不满足要求，可考虑以按用量付费方式通过云服务商的 GPU 异构计算服务部署沙箱来完成练习（按 GeForce RTX 3090 GPU / 2 元 / 小时，完成所有模型训练不会超过一顿饭钱）。沙箱环境默认为 `http://localhost:3001`，如果你选择了其他端口或者非本机的沙箱（譬如云服务），请点击文档右上角设置图标 <a href="javascript:document.getElementsByTagName('button')[0].click()"><svg data-v-9eec72c3="" class="settings-icon" style="width:18px; height:18px; color:#000" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle data-v-9eec72c3="" cx="12" cy="12" r="3"></circle><path data-v-9eec72c3="" d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg></a> 手动填入沙箱地址。

::: danger 安全提示

由于沙箱的功能是从外部接收并执行 Python 代码，唯一的安全防护只有 cgroups 隔离，**将沙箱服务直接暴露在公网环境可能会带来安全风险**。建议你优先考虑将沙箱运行于本地或者无敏感数据的云服务中。
:::

## 数据管理

为便于观察大型实验的数据、复用训练记录，本项目提供数据持久化功能，支持自动/手动数据集下载、模型保存等。数据目录可通过 `dmla data` 自定义，如未设置默认为宿主机的 `~/dmla-data` 目录，。以下为该目录的完整数据结构（目录均会按需自动创建，无需手动处理）：

```
~/dmla-data/
├── datasets/                          # 数据集目录
│   ├── tiny-imagenet-200/             # Tiny ImageNet-200
│   ├── cifar-10/                      # CIFAR-10
│   ├── cifar-100/                     # CIFAR-100
│   ├── mnist/                         # MNIST
│   └── custom/                        # 用户自定义数据集
│   └── ……
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
│   └── ……
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

你可以使用以下示例代码实际检查沙箱环境是否已经可用，代码可编辑，点击 Run 或者 Run on GPU 按钮运行代码：

```python runnable gpu
import importlib.util
import sys

def check_package(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"❌ {package_name} 未安装")
        return False
    else:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', '未知版本')
        print(f"✅ {package_name} 已安装，版本: {version}")
        return True

check_package("numpy")
check_package("torch")
```
