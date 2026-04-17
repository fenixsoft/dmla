# DMLA 系统架构设计

## 系统概览

DMLA（Design Machine Learning Applications）是一个机器学习教育平台，提供交互式 Python 代码沙箱执行环境。

```
┌─────────────────────────────────────────────────────────────────┐
│                        DMLA 系统架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │  VuePress 文档  │     │   用户浏览器    │                   │
│  │  (docs/)        │────▶│                 │                   │
│  └─────────────────┘     └─────────────────┘                   │
│                                 │                               │
│                                 │ HTTP请求                      │
│                                 ▼                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   DMLA CLI 服务                           │   │
│  │  (packages/cli)                                          │   │
│  │                                                          │   │
│  │  ├─ Express API Server                                   │   │
│  │  │   ├─ /api/health                                      │   │
│  │  │   ├─ /api/sandbox/run                                 │   │
│  │  │   ├─ /api/sandbox/health                              │   │
│  │  │   └─ /api/sandbox/gpu                                 │   │
│  │  │                                                       │   │
│  │  ├─ Docker 管理                                          │   │
│  │  │   ├─ 容器创建/销毁                                    │   │
│  │  │   ├─ GPU 设备映射                                     │   │
│  │  │   └─ Volume Mount                                     │   │
│  │  │                                                       │   │
│  │  └─ 命令管理                                             │   │
│  │      ├─ start/stop                                       │   │
│  │      ├─ status                                           │   │
│  │      ├─ install/update                                   │   │
│  │      └─ doctor                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                 │                               │
│                                 │ Docker API                    │
│                                 ▼                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Docker 容器沙箱                           │   │
│  │  (dmla-sandbox:cpu / dmla-sandbox:gpu)                   │   │
│  │                                                          │   │
│  │  ├─ Python 3.11                                          │   │
│  │  ├─ NumPy, Pandas, Matplotlib                            │   │
│  │  ├─ PyTorch (CPU/CUDA 11.8)                              │   │
│  │  ├─ scikit-learn                                         │   │
│  │  ├─ IPython Kernel                                       │   │
│  │  └─ 共享模块 (/workspace/shared)                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
dmla/
├── docs/                        # VuePress 文档站点
│   ├── .vuepress/
│   │   └ public/
│   │    └─ install.sh           # 一键安装脚本
│   └ getting-started/           # 安装指南
│   └ arch/                      # 架构文档
│   └── statistical-learning/    # 机器学习教程
│
├── packages/                    # npm 包（monorepo）
│   ├── cli/                     # @icyfenix-dmla/cli - 命令行工具
│   │   ├── bin/dmla.js          # CLI 入口
│   │   └ src/
│   │    ├─ index.js             # 命令解析
│   │    └─ commands/            # 命令实现
│   │   └ tests/                 # 单元测试
│   │   └ package.json
│   │
│   └── install/                 # @icyfenix-dmla/install - TUI 安装向导
│       ├── bin/dmla-install.js
│       ├── src/
│       │  ├─ index.js           # TUI 入口
│       │  └─ modules/           # 功能模块
│       └ tests/
│       └ package.json
│
├── local-server/                # 服务核心（workspaces）
│   ├── src/
│   │  ├─ index.js               # Express 服务入口
│   │  ├─ sandbox.js             # Docker 管理
│   │  ├─ kernel_runner.py       # Python Kernel 执行器
│   │  └─ routes/                # API 路由
│   │
│   ├── shared_modules/          # Python 共享模块
│   │  ├─ linear/                # 线性模型
│   │  ├─ bayesian/              # 贝叶斯方法
│   │  ├─ svm/                   # SVM
│   │  ├─ tree/                  # 决策树/集成
│   │  └─ unsupervised/          # 无监督学习
│   │
│   ├── Dockerfile.sandbox       # GPU 镜像
│   ├── Dockerfile.sandbox.cpu   # CPU 镜像
│   └ package.json
│
├── .github/workflows/           # GitHub Actions
│   ├─ deploy.yml                # VuePress 部署
│   ├─ publish.yml               # npm/Docker 发布
│   └─ auto-tag.yml              # 自动版本 Tag
│
├── package.json                 # 根配置（workspaces）
└─ CLAUDE.md                     # Claude Code 配置
```

## 发布流程

### 自动版本发布

```
┌─────────────────────────────────────────────────────────────────┐
│                     版本发布流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  代码变更 ──────────────────────────────────────────────────── │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                   │
│  │  路径检测                │                                   │
│  │  ├─ Dockerfile.sandbox   │ 触发 ✓                          │
│  │  ├─ Dockerfile.sandbox.cpu│ 触发 ✓                          │
│  │  ├─ shared_modules/**    │ 触发 ✓                          │
│  │  ├─ kernel_runner.py     │ 触发 ✓                          │
│  │  ├─ docs/**              │ 不触发 ✗                        │
│  │  └─ 其他                 │ 不触发 ✗                        │
│  └─────────────────────────┘                                   │
│       │                                                         │
│       ▼ 触发                                                    │
│  ┌─────────────────────────┐                                   │
│  │  auto-tag.yml            │                                   │
│  │  生成时间戳 Tag           │                                   │
│  │  2026.4.17-1503           │                                   │
│  └─────────────────────────┘                                   │
│       │                                                         │
│       ▼ Tag 推送                                                │
│  ┌─────────────────────────┐                                   │
│  │  publish.yml             │                                   │
│  │                                                          │   │
│  │  Job: publish-npm        │                                   │
│  │    ├─ @icyfenix-dmla/cli          │                                   │
│  │    └─ @icyfenix-dmla/install      │                                   │
│  │                                                          │   │
│  │  Job: build-images       │                                   │
│  │    ├─ CPU 镜像           │                                   │
│  │    └─ GPU 镜像           │                                   │
│  │                                                          │   │
│  │  Job: push-dockerhub     │                                   │
│  │    └─ icyfenix/dmla-sandbox                              │   │
│  │                                                          │   │
│  │  Job: push-tcr           │                                   │
│  │    └─ ccr.ccs.tencentyun.com/icyfenix/dmla-sandbox       │   │
│  └─────────────────────────┘                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 版本号策略

- **格式**：时间戳 `YYYY.M.D-HHMM`（如 `2026.4.17-1503`）
- **Git Tag**：`2026.4.17-1503`
- **npm 包版本**：`2026.4.17-1503`
- **Docker 镜像 Tag**：`2026.4.17-1503-cpu` / `2026.4.17-1503-gpu`

### 镜像仓库

| 仓库 | 镜像地址 | 说明 |
|------|----------|------|
| Docker Hub | `icyfenix/dmla-sandbox:cpu` | 全球访问 |
| 腾讯云 TCR | `ccr.ccs.tencentyun.com/icyfenix/dmla-sandbox:cpu` | 国内加速 |

## 安装流程

### install.sh 启动流程

```bash
curl -fsSL https://ai.icyfenix.cn/install.sh | sh
```

1. 检测 Docker 环境
2. 检测 Node.js 环境
3. 检测 GPU（可选）
4. 调用 `npx @icyfenix-dmla/install`

### TUI 安装向导流程

```
┌─────────────────────────────────────────────────────────────────┐
│                  TUI 安装向导流程                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 环境检测                                                     │
│     ├─ Docker 版本                                              │
│     ├─ Node.js 版本                                             │
│     └─ GPU 可用性                                               │
│                                                                 │
│  2. 选择镜像仓库                                                 │
│     ├─ Docker Hub (全球)                                        │
│     ├─ 腾讯云 TCR (国内加速)                                     │
│     └─ 自动选择 (网络延迟测试)                                   │
│                                                                 │
│  3. 选择镜像类型                                                 │
│     ├─ 全部安装 (CPU + GPU)                                     │
│     ├─ 仅 CPU 版本                                              │
│     └─ 仅 GPU 版本                                              │
│                                                                 │
│  4. 配置端口                                                     │
│     ├─ 默认: 3001                                               │
│     └─ 自定义端口                                               │
│                                                                 │
│  5. 拉取镜像                                                     │
│     ├─ docker pull                                              │
│     ├─ 进度显示                                                 │
│     └─ docker tag 重命名                                        │
│                                                                 │
│  6. 安装 npm 包                                                  │
│     └─ npm install -g @icyfenix-dmla/cli                         │
│                                                                 │
│  7. 验证安装                                                     │
│     ├─ 启动服务                                                 │
│     └─ 健康检查                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/health` | GET | 服务健康检查 |
| `/api/sandbox/run` | POST | 执行 Python 代码 |
| `/api/sandbox/health` | GET | 沙箱状态（镜像、GPU） |
| `/api/sandbox/gpu` | GET | GPU 可用性检查 |

## 安全考虑

- 容器隔离执行
- 60 秒执行超时
- 4GB 内存限制
- 代码长度限制（100KB）
- 镜像只读挂载

## 技术栈

- **前端**：VuePress v2 + Vue 3
- **后端**：Express + Dockerode
- **沙箱**：Python 3.11 + IPython Kernel
- **CI/CD**：GitHub Actions
- **发布**：npm + Docker Hub + 腾讯云 TCR