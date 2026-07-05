# DMLA CLI 命令参考

**沙箱模式启动方式**：

```bash
# 启动服务
dmla start                 # 默认端口 3001
dmla start --port 8080     # 自定义端口
dmla start --gpu           # GPU 模式
dmla start --dev           # 开发模式（挂载本地代码，无需重建镜像）
dmla start --gpu --dev     # GPU + 开发模式
dmla start --sync --dev    # 同步模式（前台运行，便于调试）
```

**Native 模式启动方式**：

```bash
# 方式 1：源码目录直接启动（开发环境推荐）
lsof -ti:3001 -sTCP:LISTEN | xargs kill 2>/dev/null
DMLA_MODE=native node local-server/src/index.js

# 方式 2：通过 CLI 启动
dmla start --native

# 方式 3：同步模式（前台运行，便于观察依赖安装过程）
dmla start --native --sync --dev
```

**其他命令**：

```bash
# 停止服务
dmla stop

# 查看状态
dmla status

# 安装镜像
dmla images               # 安装所有镜像（默认从 Docker Hub）

# 环境诊断
dmla doctor

# 数据管理（TUI 菜单）
dmla data                    # 进入数据管理 TUI
dmla data path               # 显示数据卷路径
dmla data mount <path>       # 设置挂载路径
dmla data download           # 下载数据集
```

**Docker 沙箱**：
```bash
# 构建 GPU 版本沙箱镜像
npm run build:sandbox:gpu

# 构建 CPU 版本（无 GPU 支持）
npm run build:sandbox:cpu

# 测试沙箱镜像
docker run --rm dmla-sandbox:gpu python3 -c "print('Hello')"
```
