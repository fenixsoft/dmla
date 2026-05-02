# CLAUDE.md

## 前言：交互语言
使用**中文**进行信息描述和文档编写，包括：
 - **GIT Commit：** GIT提交时的变更信息，内容使用中文描述
 - **Proposal、Design和归档后形成的Spec文档：** 所有涉及到的SDD文档，内容使用中文描述
 - **程序注释：** 程序代码中的注释，内容使用中文描述

## 第一部分：核心编程原则
这是我们合作的顶层思想，指导所有具体的行为。

### 基础设计原则
- **可读性优先：** 始终牢记"代码是写给人看的，只是恰好机器可以执行"。清晰度高于一切。
- **DRY (Don't Repeat Yourself)：** 绝不复制代码片段。通过抽象（如函数、类、模块）来封装和复用通用逻辑。
- **高内聚，低耦合：** 功能高度相关的代码应该放在一起（高内聚），而模块之间应尽量减少依赖（低耦合），以增强模块独立性和可维护性。

### 开发方法论
- **测试驱动开发：** 每完成一个功能模块就立即编写相应的测试，确保代码质量和稳定性。
- **渐进式开发策略：** 每写一个单元就进行一轮测试，避免后期全局修改和发现系统性问题。
- **后端单元测试检查：** 后端所有对 public 方法的修改（包括方法签名变更、参数增减、返回值类型变化等）都必须检查并更新相关的单元测试，确保测试覆盖率不被破坏。
- **前端页面测试：** 对于前端页面的修改，完毕后要通过 playwright-cli 完成实际测试后才能关闭任务。

## 第二部分：任务运作约束

### 项目运作模式
- Goal: 在**不中断**的前提下，充分并行子任务，直到产出完整、可验证的交付物。
- Autonomy: 允许在本仓库内创建/修改/删除文件；禁止写入 `node_modules`, `.git`, `dist`, `build`, `~` 与上级目录。
- Parallelism: 同时运行 ≤ 3 个并行任务。
- Checkpointing: 每个任务完成后，**必须**更新`tasks.md`，以便断点续跑。
- Validation First: 任何子任务完成后，都要跑对应校验（构建/测试/脚本），不通过则自我修复后再进入下一个任务。

### 项目与代码维护
- **统一文档维护：** 严禁为每个独立任务（如重构、功能实现）创建新的总结文档（例如 CODE_REFACTORING_SUMMARY.md）。在任务完成后，必须优先检查项目中已有的相关文档（如 README.md、既有的设计文档等），并将新的总结、变更或补充内容直接整合到现有文档中，维护其完整性和时效性。
- **及时清理：** 在完成开发任务时，如果发现任何已无用（过时）的代码、文件或注释，应主动提出清理建议。
- **测试截图统一存放：** 所有测试产生的 Chrome 浏览器截图图片统一存放在 `.history` 目录，便于追踪测试历史和对比结果。
- **禁止修改 node_modules：** 严禁直接修改 `node_modules` 目录下的任何文件。`node_modules` 是依赖包的缓存目录，任何修改都会在 `npm install` 时被覆盖。如需修改第三方库的行为，应：
  1. Fork 该库并发布为独立包
  2. 将该库源码复制到项目中作为本地模块
  3. 使用 patch-package 创建补丁（需在 package.json 中配置 postinstall 脚本）

### 数据库安全规则
- **禁止删除数据库文件：** 严禁执行 `rm`、`mv` 或其他任何删除/移动数据库文件（`*.db`）的操作。数据库包含用户数据、工作流、运行历史等重要信息，一旦删除将导致不可逆的数据丢失。
- **数据库迁移优先：** 遇到数据库结构问题时，应通过数据迁移脚本（如 `DataMigrationService`）修复，而非删除数据库。
- **备份优先原则：** 如确需对数据库进行高风险操作，必须先备份现有数据库文件，并征得用户确认后方可执行。

### 镜像开发约束（Volume Mount 机制）

Docker 沙箱镜像高达 7GB，编译耗时约 15-30 分钟。为避免频繁重编译，**所有涉及镜像内容的改动必须先用 Volume Mount 机制验证通过后才能触发编译**。

**Volume Mount 机制说明**：

开发模式下，以下文件通过 Volume Mount 自动挂载到容器，修改无需重建镜像：
- `kernel_runner.py`: 挂载到 `/workspace/kernel_runner.py`
- `dmla_progress.py`: 挂载到 `/workspace/dmla_progress.py`
- `shared_modules/`: 挂载到 `/usr/local/lib/python3.11/site-packages/shared`

**验证流程**：

1. **修改 Python 文件**：编辑 `local-server/src/kernel_runner.py`、`dmla_progress.py` 或 `shared_modules/` 下的文件
2. **启动服务测试**：运行 `npm run server` 或 `cd local-server && npm start`
3. **验证功能正确**：通过 API 或前端执行代码块，确认新功能正常工作
4. **更新 tasks.md**：标记验证通过的子任务
5. **触发镜像编译**：**仅在所有验证通过后**执行 `npm run build:sandbox:gpu`

**环境变量控制**：

```bash
# 默认启用 Volume Mount（开发模式）
MOUNT_KERNEL_RUNNER=true    # 挂载 kernel_runner.py
MOUNT_SHARED_MODULES=true   # 挂载共享模块

# 禁用挂载（生产模式，使用镜像内置文件）
MOUNT_KERNEL_RUNNER=false
MOUNT_SHARED_MODULES=false
```

**适用范围**：

必须先验证后编译的改动类型：
- 新增 Python 模块（如 `dmla_progress.py`）
- 修改 `kernel_runner.py` 执行逻辑
- 修改 `shared_modules/` 下的共享类
- 修改 Dockerfile 中的 Python 脚本引用
- 新增数据目录挂载点

无需验证可直接编译的改动：
- 修改系统依赖（apt 包）
- 修改 Python 版本
- 修改 PyTorch/CUDA 版本
- 修改基础镜像（FROM 指令）

**验证检查清单**：

```bash
# 1. 确认服务启动时 Volume Mount 生效
# 日志应显示：
# [Sandbox] kernel_runner.py Volume Mount: /path/to/local-server/src/kernel_runner.py
# [Sandbox] dmla_progress.py Volume Mount: /path/to/local-server/src/dmla_progress.py

# 2. 测试新模块导入
curl -s -X POST http://localhost:3001/api/sandbox/run \
  -H "Content-Type: application/json" \
  -d '{"code": "from dmla_progress import ProgressReporter\nprint(\"OK\")"}'

# 3. 测试共享模块导入
curl -s -X POST http://localhost:3001/api/sandbox/run \
  -H "Content-Type: application/json" \
  -d '{"code": "from shared.linear.logistic_regression import LogisticRegression\nprint(\"OK\")"}'
```

### 进度报告机制（ProgressReporter）

长时间训练任务可通过 `ProgressReporter` 向前端报告实时进度，前端通过轮询 `/workspace/progress.json` 获取进度信息并显示进度条。

**模块位置**：`local-server/src/dmla_progress.py`

**使用方式**：

```python
from dmla_progress import ProgressReporter

# 初始化进度报告器
progress = ProgressReporter(
    total_steps=100,        # 总步数（如 epoch 数）
    description="训练 AlexNet"
)

# 训练循环中更新进度
for epoch in range(100):
    train_one_epoch()
    loss = calculate_loss()
    progress.update(
        step=epoch + 1,
        message=f"Epoch {epoch+1}: Loss={loss:.4f}",
        extra_data={"loss": loss, "accuracy": accuracy}
    )

# 训练完成
progress.complete(
    message="训练完成",
    extra_data={"final_accuracy": 0.85}
)
```

**进度数据结构**：

```json
{
  "description": "训练 AlexNet",
  "total_steps": 100,
  "current_step": 50,
  "percent": 50.0,
  "message": "Epoch 50: Loss=0.32",
  "status": "running",
  "start_time": "2026-05-02T12:00:00Z",
  "elapsed_seconds": 120,
  "estimated_remaining": 120,
  "extra_data": {"loss": 0.32, "accuracy": 0.78}
}
```

**状态类型**：
- `starting`: 任务开始
- `running`: 任务进行中
- `complete`: 任务完成
- `error`: 任务出错

**辅助函数**：

```python
from dmla_progress import get_progress, clear_progress

# 读取当前进度
progress = get_progress()

# 清除进度文件
clear_progress()
```

**注意事项**：
- 进度文件路径固定为 `/workspace/progress.json`
- 写入失败不会影响训练，仅打印警告
- `extra_data` 可携带自定义指标（loss、accuracy 等）

## 第三部分：项目概览
- **设计文档**: `/docs/arch/design.md`
- **技术栈**: VuePress v2 + Vue 3 + Express + Docker

---

## 第四部分：常用命令

### 开发运行

#### 互联网模式（仅前端预览）
```bash
# 启动 VuePress 开发服务器
npm run dev
# 访问地址: http://localhost:8080
```

#### 本地模式（前端 + 沙箱服务）
```bash
# 首次运行需构建 Docker 沙箱镜像
npm run build:sandbox:gpu

# 启动完整本地服务（VuePress + API 服务器）
npm run local
# 网站页面: http://localhost:8080
# API 服务: http://localhost:3001
```

#### 单独启动后端服务
```bash
# 仅启动沙箱 API 服务器
cd local-server && npm start
# API 地址: http://localhost:3001
```

### 构建与部署

```bash
# 构建生产版本
npm run build
# 输出目录: docs/.vuepress/dist

# 刷新腾讯云 CDN（需配置环境变量）
npm run cdn:refresh
```

### Docker 沙箱

```bash
# 构建 GPU 版本沙箱镜像
npm run build:sandbox:gpu

# 构建 CPU 版本（无 GPU 支持）
npm run build:sandbox:cpu

# 测试沙箱镜像
docker run --rm dmla-sandbox:gpu python3 -c "print('Hello')"
```

### DMLA CLI（用户安装后使用）

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

# 数据管理（TUI 菜单）
dmla data                    # 进入数据管理 TUI
dmla data path               # 显示数据卷路径
dmla data mount <path>       # 设置挂载路径
dmla data download           # 下载数据集
```

### 共享模块（可复用 Python 类）

文档中多个 runnable code 块可通过共享模块复用类定义：

```bash
# 从文档提取标记的类定义到共享模块
npm run extract:shared

# 构建镜像时自动执行提取
npm run build:sandbox:gpu
```

**使用方式**：

1. 在文档中标记需要提取的类：
   ```markdown
   ```python runnable extract-class="LogisticRegression"
   class LogisticRegression:
       ...
   ```
   ```

2. 在其他代码块中导入使用：
   ```python
   from shared.linear.logistic_regression import LogisticRegression
   model = LogisticRegression()
   ```

**目录结构**：
- 文档：`docs/statistical-learning/linear-models/*.md`
- 共享模块：`local-server/shared_modules/linear/*.py`

**开发模式**：Volume Mount 自动启用，修改代码无需重建镜像
**生产模式**：设置 `MOUNT_SHARED_MODULES=false` 禁用挂载

### 数据目录结构

用户宿主机数据目录（默认 `~/dmla-data`，可通过 `dmla data mount` 自定义）：

```
~/dmla-data/
├── datasets/                          # 数据集目录
│   ├── tiny-imagenet-200/             # Tiny ImageNet (200 类)
│   ├── cifar-10/                      # CIFAR-10
│   ├── cifar-100/                     # CIFAR-100
│   ├── mnist/                         # MNIST
│   └── custom/                        # 用户自定义数据集
│
├── models/                            # 模型目录
│   ├── alexnet/                       # AlexNet 相关模型
│   │   ├── checkpoints/               # 训练中间 checkpoint
│   │   └── final/                     # 最终模型
│   ├── vgg/                           # VGG 系列模型
│   ├── resnet/                        # ResNet 系列模型
│   ├── gan/                           # GAN 模型 (预留)
│   ├── llm/                           # 大语言模型 (预留)
│   └── pretrained/                    # 预训练模型下载
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

**使用方式**：
- 容器内访问：`/data/` 目录映射到宿主机数据目录
- 代码中保存模型：`torch.save(model, '/data/models/alexnet/final/model.pth')`
- 代码中加载数据集：`torchvision.datasets.ImageFolder('/data/datasets/tiny-imagenet-200/train')`

**数据集下载**：
运行 `dmla data` 进入 TUI 菜单，选择"下载数据集"，支持：
- Tiny ImageNet 200 (250MB)
- CIFAR-10 (170MB)
- CIFAR-100 (170MB)
- MNIST (11MB，通过 torchvision 自动下载)

### 端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| VuePress Dev | 8080 | 前端开发服务器 |
| Local Server API | 3001 | 沙箱 API 服务 |

### 环境变量

```bash
# GitHub OAuth（评论功能）
GITHUB_CLIENT_ID=your_client_id

# 腾讯云 CDN 刷新
TENCENT_SECRET_ID=your_secret_id
TENCENT_SECRET_KEY=your_secret_key
CDN_DOMAIN=your_domain.com
```

### GitHub Actions 发布流程

项目使用独立的 npm 和 Docker 发布流程：

| Workflow | 触发条件 | Tag 格式 | 说明 |
|----------|----------|----------|------|
| `auto-tag-npm.yml` | packages/ 目录变更 | `npm-YYYY.M.D-HHMM` | npm 包自动打 Tag |
| `auto-tag.yml` | local-server/ Docker 相关变更 | `YYYY.M.D-HHMM` | Docker 镜像自动打 Tag |
| `publish-npm.yml` | npm- 开头的 Tag | - | 发布 npm 包 |
| `publish-docker.yml` | 非 npm- 开头的 Tag | - | 发布 Docker 镜像 |

**镜像仓库**：
- Docker Hub: `icyfenix/dmla-sandbox`（全球用户）
- 阿里云 ACR: `crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox`（国内加速）

**Secrets 配置**：
- `NPM_TOKEN`: npm 发布认证
- `DOCKER_USERNAME/DOCKER_PASSWORD`: Docker Hub 认证
- `ACR_USERNAME/ACR_PASSWORD`: 阿里云 ACR 认证