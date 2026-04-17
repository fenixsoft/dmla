# DMLA (Designing Machine Learning Applications)

机器学习教育平台，提供交互式 Python 代码沙箱执行环境。

## 🚀 快速安装

### 一键安装

```bash
curl -fsSL https://ai.icyfenix.cn/install.sh | sh
```

### 手动安装

1. 安装 Docker：https://docs.docker.com/get-docker/
2. 安装 Node.js 18+：https://nodejs.org/
3. 安装 CLI：
   ```bash
   npm install -g @icyfenix-dmla/cli
   ```

4. 安装镜像：
   ```bash
   dmla install
   ```

## 📦 常用命令

```bash
dmla start              # 启动服务（默认端口 3001）
dmla start --port 8080  # 自定义端口启动
dmla start --gpu        # 使用 GPU 镜像
dmla stop               # 停止服务
dmla status             # 查看状态
dmla install            # 安装/更新 Docker 镜像
dmla update             # 更新 npm 包和镜像
dmla doctor             # 环境诊断
```

## 🌐 镜像仓库选择

支持两个镜像仓库：

- **Docker Hub**：全球访问，无需登录
  ```bash
  dmla install --registry dockerhub
  ```

- **腾讯云 TCR**：国内加速，无需登录（公开镜像）
  ```bash
  dmla install --registry tcr
  ```

## 📄 许可证

MIT License