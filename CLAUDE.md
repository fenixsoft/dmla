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
npm run build:sandbox

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
npm run build:sandbox

# 构建 CPU 版本（无 GPU 支持）
docker build -t ideaspaces-sandbox:latest . -f local-server/Dockerfile.sandbox --build-arg GPU=false

# 测试沙箱镜像
docker run --rm ideaspaces-sandbox:gpu python3 -c "print('Hello')"
```

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