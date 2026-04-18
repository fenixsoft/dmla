## Why

当前 DMLA 的安装流程复杂，用户需要手动构建 Docker 镜像（涉及 GPU/CPU 版本选择、共享模块提取等），且 npm 包未发布到公共仓库。这导致：
1. **安装门槛高**：用户需要具备 Docker 构建知识和完整源码
2. **版本管理困难**：没有统一的版本发布机制
3. **国内访问慢**：镜像只能本地构建，无法利用镜像仓库加速

现在实施此变更，可以显著降低用户安装门槛，同时建立规范的版本发布流程，便于后续维护和用户升级。

## What Changes

### GitHub Actions 发布自动化
- 创建统一的发布工作流，响应 Git Tag 触发
- 自动发布 `@dmla/cli` 到 npm 公共仓库
- 自动构建并推送 Docker 镜像到 Docker Hub 和腾讯云 TCR
- 镜像同时推送稳定版别名（`:cpu`、`:gpu`）和版本号标签

### 自动版本 Tag 机制
- 创建独立的自动打 Tag 工作流
- 仅在 Dockerfile 或 shared_modules 变化时触发
- 使用时间戳格式：`2026.4.17-1503`
- 其他目录（如 docs）变化不触发版本发布

### npm CLI 包改造
- 包名从 `dmla-local-server` 改为 `@dmla/cli`
- 新增 CLI 命令：`start`、`stop`、`status`、`install`、`update`、`doctor`
- 提供 `update` 命令同时更新 npm 包和 Docker 镜像

### TUI 安装系统
- 创建轻量级 `install.sh` 启动脚本，托管于 VuePress public 目录
- 创建 `@dmla/install` npm 包提供完整 TUI 体验
- 支持镜像仓库选择（Docker Hub / Tencent TCR）
- 支持镜像类型选择（CPU / GPU）
- 支持端口配置和进度显示

## Capabilities

### New Capabilities
- `npm-publishing`: npm 包自动发布到公共仓库，版本号与 Git Tag 同步
- `docker-image-publishing`: Docker 镜像自动构建并推送到 Docker Hub 和 TCR，支持 CPU/GPU 双版本
- `installation-cli`: `@dmla/cli` 命令行工具，提供 start/stop/status/install/update/doctor 命令
- `installation-tui`: TUI 安装向导，支持仓库选择、镜像类型选择、端口配置、进度显示
- `auto-version-tagging`: 自动版本 Tag 生成机制，仅响应镜像相关文件变更

### Modified Capabilities
- `sandbox-settings`: 新增镜像仓库来源配置能力，支持从远程仓库拉取镜像而非仅本地构建

## Impact

### 新增文件
- `.github/workflows/publish.yml` - 统一发布工作流
- `.github/workflows/auto-tag.yml` - 自动打 Tag 工作流
- `packages/cli/` - `@dmla/cli` npm 包源码目录
- `packages/install/` - `@dmla/install` TUI 安装包源码目录
- `docs/.vuepress/public/install.sh` - 一键安装脚本

### 修改文件
- `local-server/package.json` - 包名改为 `@dmla/cli`，新增 bin 字段
- `local-server/src/` - 新增 CLI 相关模块
- `package.json` - 新增 workspaces 配置

### 外部依赖
- Docker Hub 账户（`icyfenix/dmla-sandbox` 命名空间）
- 腾讯云 TCR 命名空间（`ccr.ccs.tencentyun.com/icyfenix/dmla-sandbox`）
- npm 组织账户（`@dmla`）

### GitHub Secrets 需求
- `NPM_TOKEN` - npm 发布令牌
- `DOCKER_USERNAME` / `DOCKER_PASSWORD` - Docker Hub 认证
- `TCR_USERNAME` / `TCR_PASSWORD` - 腾讯云 TCR 认证