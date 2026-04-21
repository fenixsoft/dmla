# CLI 命令参考

`@icyfenix-dmla/cli` 提供完整的沙箱服务管理命令。

## 安装 CLI

```bash
npm install -g @icyfenix-dmla/cli
```

安装后，`dmla` 命令可在终端直接调用。

## 命令概览

```bash
dmla <command> [options]
```

| 命令 | 描述 |
|------|------|
| `start` | 启动沙箱服务 |
| `stop` | 停止运行中的服务 |
| `status` | 查看服务状态 |
| `install` | 安装 Docker 镜像 |
| `update` | 更新 npm 包和镜像 |
| `doctor` | 环境诊断 |

## dmla start

启动沙箱服务。

```bash
dmla start [options]
```

**选项：**

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `-p, --port <number>` | 服务端口 | 3001 |
| `--gpu` | 使用 GPU 镜像 | false |

**示例：**

```bash
# 默认端口启动
dmla start

# 自定义端口
dmla start --port 8080

# GPU 模式
dmla start --gpu

# 组合使用
dmla start --port 8080 --gpu
```

## dmla stop

停止运行中的沙箱服务。

```bash
dmla stop
```

## dmla status

查看服务状态。

```bash
dmla status
```

显示信息包括：
- npm 包版本
- Docker 镜像状态（CPU/GPU）
- GPU 可用性
- 服务运行状态

## dmla install

安装 Docker 镜像。

```bash
dmla install [options]
```

**选项：**

| 选项 | 描述 |
|------|------|
| `--cpu` | 仅安装 CPU 版本 |
| `--gpu` | 仅安装 GPU 版本 |
| `--all` | 安装所有镜像（默认） |
| `-r, --registry <type>` | 镜像仓库：dockerhub 或 acr |

**示例：**

```bash
# 安装所有镜像（默认从 Docker Hub）
dmla install

# 仅安装 CPU 版本
dmla install --cpu

# 从 ACR 安装（国内加速）
dmla install --registry acr

# 仅安装 GPU 版本
dmla install --gpu --registry acr
```

## dmla update

更新 npm 包和 Docker 镜像。

```bash
dmla update [options]
```

**选项：**

| 选项 | 描述 |
|------|------|
| `-r, --registry <type>` | 镜像仓库：dockerhub 或 acr |

**示例：**

```bash
# 更新所有（默认从 Docker Hub）
dmla update

# 更新并指定 ACR
dmla update --registry acr
```

## dmla doctor

诊断安装环境。

```bash
dmla doctor
```

检查项目：
- Docker 安装和版本
- Docker 镜像完整性
- GPU 驿动和设备
- 端口可用性
- 网络连通性（Docker Hub / ACR）

## dmla --help

显示帮助信息。

```bash
dmla --help
dmla <command> --help
```

## dmla --version

显示版本号。

```bash
dmla --version
```

## 镜像仓库说明

### Docker Hub

- 地址：`icyfenix/dmla-sandbox`
- 无需登录
- 全球访问

### 阿里云 ACR

- 地址：`crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox`
- 无需登录（公开镜像）
- 国内访问更快

## 镜像版本说明

| 镜像 | 大小 | 说明 |
|------|------|------|
| `dmla-sandbox:cpu` | ~650MB | CPU 版本，无 GPU 支持 |
| `dmla-sandbox:gpu` | ~5.62GB | GPU 版本，支持 CUDA 11.8 |

## 端点说明

服务启动后提供以下端点：

| 端点 | 描述 |
|------|------|
| `http://localhost:3001/api/health` | 健康检查 |
| `http://localhost:3001/api/sandbox/run` | 执行 Python 代码 |
| `http://localhost:3001/api/sandbox/health` | 沙箱状态 |
| `http://localhost:3001/api/sandbox/gpu` | GPU 状态 |