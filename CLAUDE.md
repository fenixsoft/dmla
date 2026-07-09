# CLAUDE.md

## 交互语言

使用**中文**进行信息描述和文档编写，包括：
- **GIT Commit：** GIT提交时的变更信息，内容使用中文描述
- **Proposal、Design和归档后形成的Spec文档：** 所有涉及到的SDD文档，内容使用中文描述
- **程序注释：** 程序代码中的注释，内容使用中文描述

### 避免 AI 语言习惯

- **禁止使用破折号 "——"**
- 不要滥用"核心"、"关键"、"重要"、"洞察"等空洞强调词
- 不要滥用分号 "；"、冒号 "："
- 不要以短句罗列知识点来凑篇幅。写自然、连贯的中文，像给人讲解而非堆砌要点。
- 不要使用"第一，……第二，……第三，……"或"分两步/三步"等序号罗列式写法。把内容的层次关系融入自然叙述中，而非用数字标签隔开。

## 任务运作约束

### 项目运作模式

- Autonomy: 允许在本仓库内创建/修改/删除文件；禁止写入 `node_modules`, `.git`, `dist`, `build`, `~` 与上级目录。
- **Git 推送控制：** 除非得到用户的明确指令，否则禁止自动推送代码到 git 仓库。完成代码修改后，应告知用户变更内容并等待用户确认后再执行 `git push`。
- **GitHub 工作流与部署流水线查询：** 当需要检查 GitHub Actions 工作流状态、查看部署流水线运行情况时，使用 `gh` 命令查询相关信息（如 `gh run list`、`gh workflow view` 等），而非直接读取本地文件。
- **网页打开与 URL 访问：** 当用户要求打开网页、贴出网址 URL 时，使用 `playwright-cli` skill 调用浏览器打开页面。

### 项目与代码维护

- **前端沙箱连接超时配置**：Run / Run on GPU 按钮的连接超时时间定义在 `docs/.vuepress/plugins/runnable-code/client.js` 的 `CONNECTION_TIMEOUT` 常量（当前值 10000ms）。首次运行时 Docker 启动容器较慢，若超时时间过短会导致"无法连接到沙箱服务（连接超时）"误报。
- **及时清理：** 在完成开发任务时，如果发现任何已无用（过时）的代码、文件或注释，应主动提出清理建议。
- **测试截图统一存放：** 所有测试产生的 Chrome 浏览器截图图片统一存放在 `.history` 目录，便于追踪测试历史和对比结果。
- **禁止修改 node_modules：** 严禁直接修改 `node_modules` 目录下的任何文件。`node_modules` 是依赖包的缓存目录，任何修改都会在 `npm install` 时被覆盖。如需修改第三方库的行为，应：
  1. Fork 该库并发布为独立包
  2. 将该库源码复制到项目中作为本地模块
  3. 使用 patch-package 创建补丁（需在 package.json 中配置 postinstall 脚本）

### 进程管理规则

- **禁止广泛杀进程：** 严禁使用 `killall node`、`pkill -f node`、`killall python` 等广泛匹配的命令终止进程。这些命令会误杀 VSCode 远程连接进程、其他用户的进程或系统关键服务。
- **精确进程管理：** 需要终止进程时，必须使用精确匹配方式：
  1. **通过端口定位（监听状态）**：`lsof -ti:<PORT> -sTCP:LISTEN | xargs kill`（仅终止监听端口的进程，不杀连接状态的进程）
     - **重要**：8080 端口必须加 `-sTCP:LISTEN` 参数！VSCode Server 的端口转发功能会连接到 8080，若不加此参数会误杀 VSCode 远程连接
     - 正确命令：`lsof -ti:8080 -sTCP:LISTEN | xargs kill 2>/dev/null`
     - 错误命令：`lsof -ti:8080 | xargs kill`（会误杀 VSCode Server）
  2. 通过 PID 文件：如果服务有 PID 文件，读取后精确 kill
  3. 通过进程名全路径匹配：`pkill -f "local-server/src/index.js"`（包含完整路径）
- **服务管理优先：** 优先使用服务自带的停止命令（如 `npm run stop`）而非直接 kill 进程。
- **检查影响范围：** 执行任何进程终止命令前，先用 `ps aux | grep <pattern>` 确认匹配到的进程列表，确保不会误杀。

## 开发规范

### CLI 包开发规则

> **📄 参考文件**: [references/cli-package-dev.md](references/cli-package-dev.md)
> **用途**: CLI 包的源码-构建产物映射关系：构建脚本 `packages/cli/scripts/build.js` 从 `local-server/src/` 复制文件到 `packages/cli/src/server/`，因此所有服务端代码修改必须在 `local-server/src/` 源目录进行，禁止直接修改 `packages/cli/src/server/` 构建产物目录。文件中列出了各关键文件（`native_env_check.js`、`native_executor.js`、`kernel_runner.py`、`index.js`）的正确修改位置对照表。
> **触发场景**: 修改 CLI 服务端代码（沙箱执行、环境检测、Python 内核运行器等）时，必须参考此文件确认正确的源码目录，避免修改被构建脚本覆盖。

### 依赖管理

> **📄 参考文件**: [references/image-deps-and-dev.md](references/image-deps-and-dev.md)
> **用途**: 项目 Python 依赖的三处统一定义位置（`native_env_check.js` 的 `SOFT_DEPS` 数组、`Dockerfile.sandbox`、`Dockerfile.sandbox.cpu`），新增包时三者需同步更新。文件中包含完整的依赖变更验证流程（修改 SOFT_DEPS → 构建 CLI → Native 模式启动观察自动安装 → Python 导入验证 → 清理端口）和当前 SOFT_DEPS 完整列表。
> **触发场景**: 新增或移除 Python 依赖包时，必须读取此文件确认三处依赖定义位置，并按验证流程确认 Native 模式下的自动安装正常工作。

### 镜像开发约束（Volume Mount 机制）

> **📄 参考文件**: [references/image-deps-and-dev.md](references/image-deps-and-dev.md)
> **用途**: Docker 沙箱镜像高达 10GB、编译耗时 15-30 分钟，因此所有镜像内容改动必须先用 Volume Mount 机制验证。文件中详述了 Volume Mount 自动挂载的文件列表（`kernel_runner.py`、`dmla_progress.py`、`shared_modules/`）、两种开发模式启动方式（源码目录 vs CLI 开发模式）、环境变量控制（`MOUNT_KERNEL_RUNNER`/`MOUNT_SHARED_MODULES`）、必须先验证后编译的改动类型，以及验证检查清单。
> **触发场景**: 修改 `kernel_runner.py`、`dmla_progress.py`、`shared_modules/` 或 Dockerfile 中 Python 脚本引用时，应参考此文件按 Volume Mount 验证流程操作，验证通过后才能触发 `npm run build:sandbox:gpu` 编译镜像。

### DMLA CLI 命令

> **📄 参考文件**: [references/dmla-cli-commands.md](references/dmla-cli-commands.md)
> **用途**: DMLA CLI 工具的全部子命令及其参数说明，包括沙箱模式和 Native 模式启动方式、`stop`、`status`、`images`、`doctor`、`data` 子命令，以及 Docker 沙箱构建命令。
> **触发场景**: 需要启动/停止 DMLA 服务、切换 Native/GPU/开发模式、管理数据挂载路径、下载数据集、诊断运行环境时，应读取此文件获取完整命令列表。

### 训练实验支持

> **📄 参考文件**: [references/training-experiment-support.md](references/training-experiment-support.md)
> **用途**: 包含两个关联主题：(1) `DATA_DIR` 数据路径自动注入机制，实现 Docker 与 Native 模式的数据路径自动适配，以及文档代码块中应使用 `os.path.join(DATA_DIR, ...)` 的规范；(2) 文档 runnable 代码块通过 `extract-class` 标记提取可复用类定义到 `local-server/shared/` 目录的共享模块机制，包含标记语法、导入路径约定和禁止手动创建共享模块的约束。
> **触发场景**: 在文档或训练代码中引用数据集/模型/缓存路径、编写 runnable 代码块需要定义可复用类、运行 `npm run extract:shared` 提取类定义时，读取此文件获取正确用法。

### 进度报告机制

> **📄 参考文件**: [references/progress-reporter.md](references/progress-reporter.md)
> **用途**: 长时间训练任务通过 `ProgressReporter`（`local-server/src/dmla_progress.py`）向前端报告实时进度的完整 API 参考，包含 `ProgressReporter` 初始化/`update()`/`complete()` 用法示例、进度 JSON 数据结构、四种状态类型和辅助函数。
> **触发场景**: 编写需要向前端报告训练进度的 Python 代码时，读取此文件获取 `ProgressReporter` 的正确用法和 API 参考。

### 数据目录结构

> **📄 参考文件**: [references/data-directory.md](references/data-directory.md)
> **用途**: 用户宿主机数据目录（`~/dmla-data`）的完整目录树结构说明，涵盖 datasets/、models/、outputs/、cache/ 四个子目录的用途和约定路径，以及数据集下载列表（Tiny ImageNet、CIFAR-10/100、MNIST、Cartoon Face）和数据集配置维护位置（`packages/cli/src/commands/data.js` 的 `DATASETS` 数组）。
> **触发场景**: 需要在代码中保存/加载模型、下载或引用数据集、查找训练输出或缓存文件路径、了解数据卷挂载映射关系（容器内 `/data/` ↔ 宿主机 `~/dmla-data`）时，应读取此文件获取准确的目录结构和命名约定。

### GitHub Actions 发布流程

> **📄 参考文件**: [references/github-actions-publish.md](references/github-actions-publish.md)
> **用途**: 项目的 npm 和 Docker 自动化发布流程，包含四个 Workflow（`auto-tag-npm`/`auto-tag`/`publish-npm`/`publish-docker`）的触发条件和 Tag 格式约定，以及 Docker Hub、阿里云 ACR 镜像仓库地址和全部 Secrets 配置项。
> **触发场景**: 配置 GitHub Actions Secrets、排查发布流程问题、或了解 Tag 触发机制时，读取此文件获取完整的 Workflow 矩阵和仓库配置。

### 阿里云 FC 函数计算管理

> **📄 参考文件**: [references/aliyun-fc-management.md](references/aliyun-fc-management.md)
> **用途**: 阿里云函数计算（FC）的完整运维手册，包含认证凭据配置、已部署资源清单（sandbox-cpu 函数、HTTP 触发器、ACR 镜像）、函数配置参数（vCPU/内存/磁盘/超时）、FC 3.0 API 使用方法、镜像预热机制说明，以及每次代码变更后的完整镜像构建-推送-部署流程和 tag 缓存策略。
> **触发场景**: 需要管理 FC 函数（查看状态、更新镜像、查看触发器 URL）、配置 aliyun CLI 凭据、构建推送 FC 镜像、排查 FC 部署问题时，应读取此文件获取详细操作命令。