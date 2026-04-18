## MODIFIED Requirements

### Requirement: 触发路径限定

系统 SHALL 仅在 Docker 相关特定路径变化时触发自动打 Tag。

#### Scenario: Dockerfile 变化触发
- **WHEN** `local-server/Dockerfile.sandbox` 或 `Dockerfile.sandbox.cpu` 文件变更
- **THEN** 触发自动打 Tag workflow

#### Scenario: shared_modules 变化触发
- **WHEN** `local-server/shared_modules/**` 目录下任何文件变更
- **THEN** 触发自动打 Tag workflow

#### Scenario: kernel_runner 变化触发
- **WHEN** `local-server/src/kernel_runner.py` 文件变更
- **THEN** 触发自动打 Tag workflow

#### Scenario: 其他路径不触发
- **WHEN** docs、packages、README.md 或其他非触发路径变更
- **THEN** 不触发自动打 Tag workflow

#### Scenario: 主分支限定
- **WHEN** 变化推送到的分支不是 main
- **THEN** 不触发自动打 Tag workflow

### Requirement: Tag 推送触发发布

系统 SHALL 在 Docker Tag 推送时触发 Docker 镜像发布流程（不再触发 npm 发布）。

#### Scenario: Tag 推送事件
- **WHEN** Tag 推送到 GitHub 仓库
- **THEN** publish-docker workflow 自动触发
- **AND** 仅执行 Docker 镜像构建和推送

#### Scenario: 版本号传递
- **WHEN** Tag 为 `2026.4.17-1503`
- **THEN** Docker 镜像版本 tag 为 `2026.4.17-1503-cpu` 和 `2026.4.17-1503-gpu`
- **AND** 不影响 npm 包版本

## REMOVED Requirements

### Requirement: Tag 推送触发 npm 和 Docker 发布

**Reason**: npm 和 Docker 发布流程已拆分为独立 workflow，各自维护版本号

**Migration**: npm 发布使用 `npm-YYYY.M.D-HHMM` 格式的 Tag 触发独立的 publish-npm workflow