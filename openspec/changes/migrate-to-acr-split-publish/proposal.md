## Why

当前 `auto-tag.yml` 和 `publish.yml` 将 npm 包和 Docker 镜像绑定在同一个版本发布流程中，导致：
1. **发布耦合**：npm 包变更和 Docker 镜像变更使用相同版本号，无法独立迭代
2. **触发混乱**：当前 auto-tag 仅监控 Docker 相关文件变更，npm 包变更无法自动触发发布
3. **速度问题**：腾讯云 TCR 国内用户反馈推送速度过慢，影响用户体验

## What Changes

- 拆分为独立的 npm 和 Docker 发布流程，各自维护版本号
- 新增 npm 版本自动打 Tag workflow（`auto-tag-npm.yml`）
- Docker 版本打 Tag workflow（`auto-tag.yml`）保持独立
- 将腾讯云 TCR 替换为阿里云 ACR，提升国内镜像拉取速度

## Capabilities

### New Capabilities

- `npm-auto-tagging`: npm 包版本独立自动打 Tag，监控 packages/ 目录变更
- `acr-image-publishing`: Docker 镜像推送到阿里云容器镜像服务 ACR

### Modified Capabilities

- `auto-version-tagging`: 仅用于 Docker 镜像版本，移除与 npm 的关联
- `docker-image-publishing`: 移除腾讯云 TCR，改为阿里云 ACR
- `npm-publishing`: 使用独立的 npm 版本 Tag，不再与 Docker 版本绑定

## Impact

- **新增文件**：
  - `.github/workflows/auto-tag-npm.yml` — npm 版本自动打 Tag
  - `.github/workflows/publish-npm.yml` — npm 包独立发布流程
  - `.github/workflows/publish-docker.yml` — Docker 镜像独立发布流程
- **删除文件**：
  - `.github/workflows/publish.yml` — 拆分为两个独立 workflow
- **修改文件**：
  - `.github/workflows/auto-tag.yml` — 仅用于 Docker 版本
- **Secrets 配置**：
  - 新增 `ACR_USERNAME`、`ACR_PASSWORD` 用于阿里云镜像仓库认证
  - 保留 `TCR_USERNAME`、`TCR_PASSWORD`（可后续清理）