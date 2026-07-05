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
dmla images               # 安装所有镜像（默认从 Docker Hub）

# 环境诊断
dmla doctor
```

## 要求

- Node.js >= 18.0.0
- Docker

## 许可证

MIT