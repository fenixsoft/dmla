# @icyfenix-dmla/cli

DMLA 沙箱服务命令行工具。

## 安装

```bash
npm install -g @icyfenix-dmla/cli
```

## 使用

```bash
# 启动服务
dmla start                 # 默认端口 3001
dmla start --port 8080     # 自定义端口
dmla start --gpu           # GPU 模式

# 停止服务
dmla stop

# 查看状态
dmla status

# 安装镜像
dmla install               # 安装所有镜像（默认从 Docker Hub）
dmla install --cpu         # 仅 CPU 版本
dmla install --gpu         # 仅 GPU 版本
dmla install --registry acr  # 从阿里云 ACR 安装（国内加速）

# 更新
dmla update                # 更新 npm 包和镜像
dmla update --registry acr

# 环境诊断
dmla doctor
```

## 要求

- Node.js >= 18.0.0
- Docker

## 许可证

MIT