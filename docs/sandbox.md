
# 构建沙箱环境

::: danger 安全提示

由于沙箱的功能是从外部接收并执行 Python 代码，唯一的安全防护只有 CGroups 隔离，**将沙箱服务直接暴露在公网环境可能会带来安全风险**。建议你优先考虑将沙箱运行于本地或者无敏感数据的云服务中。
:::

## 部署前准备

确保你的系统：

- 已经[部署](https://docs.docker.com/engine/install)好了 Docker 环境。
- 已经[部署](https://nodejs.org/en/download)好了 NodeJS 22.x+ 环境。
- 可选：如需使用 GPU 训练，应具备 Nvidia GPU 且已经[安装](https://www.nvidia.com/en-us/drivers/) 了 Nvidia 驱动。
- 其余依赖（如 Jupyter Notebook 环境、Python、Numpy、Pytorch、CUDA 等）均通过 Docker 镜像来使用，不需要单独安装。

## 快速开始

本文档内包含大量的代码用于演示机器学习算法以及进行模型训练，因此部署一套沙箱环境用于练习是十分有用的。
- 如果你使用的是互联网上部署的文档（[https://ai.icyfenix.cn](https://ai.icyfenix.cn)），可以在本地运行如下命令，使用`DMLA-CLI`部署沙箱环境，让网站上的代码能够在你本地执行：

    ``` shell
    curl -fsSL https://ai.icyfenix.cn/install.sh | sh
    # 或者
    npx @icyfenix-dmla/install
    ```

    部署后，使用使用如下命令启动沙箱服务：
    ``` bash
    # 启动服务
    dmla start                 # 默认端口 3001
    dmla start --port 3002     # 自定义端口
    dmla start --gpu           # GPU 模式

    # 停止服务
    dmla stop

    # 查看状态
    dmla status

    # 安装镜像
    dmla install               # 安装所有镜像
    dmla install --cpu         # 仅 CPU 版本
    dmla install --gpu         # 仅 GPU 版本
    dmla install --registry acr  # 从阿里云 ACR 安装镜像（国内加速）

    # 更新
    dmla update                # 更新 npm 包和镜像
    dmla update --registry acr

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
    docker pull icyfenix/dmla-sandbox:cpu
    或者
    docker pull icyfenix/dmla-sandbox:gpu

    # 使用国内的镜像缓存
    docker pull crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/icyfenix/dmla-sandbox:[cpu/gpu]

    # 本地编译镜像
    npm run build:sandbox:[cpu/gpu/all]
    ```

## 相关事项

- 本文档的沙箱服务与代码基于 Linux 软件环境 / AMD64+Nvidia 硬件环境上测试通过，Windows、WSL、MacOS 等不同的软硬件环境有可能需要额外的处理。
- 对于第一部分数学基础和第二部分经典统计学习算法，只需纯 CPU 环境即可运行。
- 对于深度学习及之后的内容，需要有 GPU 异构计算环境的支持（目前是 Pytorch with CUDA 11.8）。
- 沙箱环境默认端口为 3001，如果你选择了其他端口，或者非本机的沙箱（如云服务），请点击文档右上角设置图标 <a href="javascript:document.getElementsByTagName('button')[0].click()"><svg data-v-9eec72c3="" class="settings-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle data-v-9eec72c3="" cx="12" cy="12" r="3"></circle><path data-v-9eec72c3="" d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg></a> 手动填入沙箱地址。
- 包括大语言模型的完整训练在内，本文档所有练习都可以在 16G 级别的 Nvidia 显卡上顺利完成。如本地硬件不满足，可考虑以按用量付费方式，通过云服务商的 GPU 异构计算服务部署沙箱来完成练习（按 3090 GPU / 2元 / 小时，成本 10 元以内）。

## 检查环境

你可以使用以下示例代码实际检查沙箱环境是否已经可用：

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
