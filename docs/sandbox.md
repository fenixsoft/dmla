::: danger 安全提示

由于沙箱的功能是从外部接收并执行 Python 代码，唯一的安全防护只有 cgroups 隔离，**将沙箱服务直接暴露在公网环境可能会带来安全风险**。建议你优先考虑将沙箱运行于本地或者无敏感数据的云服务中。
:::

# 构建沙箱环境

## 部署前准备

确保你的系统：

- 已经部署好了 [Docker 环境](https://docs.docker.com/engine/install)。
- 已经部署好了 [NodeJS 20.x+ 环境](https://nodejs.org/en/download)。
- 可选：如需使用 GPU 训练，应具备 NVIDIA GPU 且已经安装了 [NVIDIA 驱动](https://www.nvidia.com/en-us/drivers/) 。
- 其余依赖（如 Jupyter Notebook Kernel、Python、Numpy、PyTorch、CUDA 等）均通过 Docker 镜像来使用，不需要单独安装。

## 快速开始

本文档内包含大量的代码用于演示机器学习算法以及进行模型训练，因此部署一套沙箱环境用于练习是十分有用的。
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

## 相关事项

- 本文档的沙箱服务与代码基于 Linux 软件环境 / x86_64 + NVIDIA 硬件环境上测试通过，支持跨平台运行，但 Windows、WSL、MacOS 等不同的软硬件环境有可能需要额外的处理。
- 对于第一部分数学基础和第二部分经典统计学习算法，只需纯 CPU 环境即可运行。
- 对于深度学习及之后的内容，有部分实验需要有 GPU 异构计算环境的支持，当前 Docker 镜像使用的是 PyTorch with [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)，支持 20/30/40/50 系显卡，A100/A800/H100/H800 专业计算卡。如果你的硬件不在此范畴，需要自行下载代码，调整 PyTorch 版本后重新编译镜像。
- 沙箱环境默认端口为 3001，如果你选择了其他端口，或者非本机的沙箱（如云服务），请点击文档右上角设置图标 <a href="javascript:document.getElementsByTagName('button')[0].click()"><svg data-v-9eec72c3="" class="settings-icon" style="width:18px; height:18px; color:#000" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle data-v-9eec72c3="" cx="12" cy="12" r="3"></circle><path data-v-9eec72c3="" d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg></a> 手动填入沙箱地址。
- 包括大语言模型的完整训练在内，本文档所有练习都可以在 16G 级别的 NVIDIA 显卡上顺利完成。如本地硬件不满足要求，也可考虑以按用量付费方式，通过云服务商的 GPU 异构计算服务部署沙箱来完成练习（按 GeForce RTX 3090 GPU / 2 元 / 小时，成本在 10 元以内）。

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
