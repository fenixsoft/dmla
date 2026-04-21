# 安装指南

DMLA 提供多种安装方式，选择最适合你的方式。

## 前置要求

- **Docker**：20.10+ 版本
- **Node.js**：18+ 或 20+ 版本
- **GPU**（可选）：NVIDIA GPU + nvidia-smi 驱动

## 方式一：一键安装（推荐）

```bash
curl -fsSL https://ai.icyfenix.cn/install.sh | sh
```

安装脚本会：
1. 检测 Docker 和 Node.js 环境
2. 启动交互式安装向导
3. 自动安装镜像和 CLI 工具

## 方式二：TUI 安装向导

如果你已安装 Node.js，可以直接运行安装向导：

```bash
npx @icyfenix-dmla/install
```

安装向导提供：
- 镜像仓库选择（Docker Hub / 阿里云 ACR）
- 镜像类型选择（CPU / GPU）
- 端口配置
- 安装进度显示

## 方式三：手动安装

### 1. 安装 CLI 工具

```bash
npm install -g @icyfenix-dmla/cli
```

### 2. 安装 Docker 镜像

```bash
# 默认从 Docker Hub 安装
dmla install

# 从阿里云 ACR 安装（国内加速）
dmla install --registry acr

# 仅安装 CPU 版本
dmla install --cpu

# 仅安装 GPU 版本
dmla install --gpu
```

### 3. 启动服务

```bash
dmla start
```

## 验证安装

运行诊断命令检查环境：

```bash
dmla doctor
```

诊断内容包括：
- Docker 版本和状态
- Node.js 版本
- 镜像完整性
- GPU 可用性
- 端口可用性
- 网络连通性

## 更新

```bash
dmla update
```

此命令会：
1. 更新 npm 包到最新版本
2. 检查并更新 Docker 镜像（如有新版本）

## 国内用户加速

国内用户推荐使用阿里云 ACR 镜像仓库：

```bash
# 安装时指定 ACR
dmla install --registry acr

# 更新时指定 ACR
dmla update --registry acr
```

镜像地址：
- `crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:cpu`
- `crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:gpu`

## 故障排除

### Docker 未安装

请先安装 Docker：
- macOS: https://docs.docker.com/desktop/install/mac-install/
- Linux: https://docs.docker.com/engine/install/
- Windows: https://docs.docker.com/desktop/install/windows-install/

### Node.js 未安装

请先安装 Node.js 18+：
- 下载地址：https://nodejs.org/
- 推荐版本：Node.js 20 LTS

### 端口被占用

默认端口 3001 被占用时：

```bash
# 使用其他端口启动
dmla start --port 8080
```

### GPU 不可用

确保已安装 NVIDIA 驾动：

```bash
# 检查 GPU
nvidia-smi
```

如果 nvidia-smi 命令失败，说明 GPU 驱动未正确安装。