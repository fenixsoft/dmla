# 自动版本 Tag 功能规范

## ADDED Requirements

### Requirement: 自动生成时间戳 Tag

系统 SHALL 在触发条件满足时自动生成时间戳格式的 Git Tag。

#### Scenario: 时间戳格式
- **WHEN** 自动生成 Tag
- **THEN** Tag 格式为 `YYYY.M.D-HHMM`
- **AND** 使用当前 UTC+8 时间

#### Scenario: Tag 示例
- **WHEN** 时间为 2026年4月17日15:03
- **THEN** 生成的 Tag 为 `2026.4.17-1503`

#### Scenario: Tag 唯一性检查
- **WHEN** 尝试创建 Tag
- **THEN** 检查相同 Tag 是否已存在
- **AND** 如已存在则跳过创建

### Requirement: 触发路径限定

系统 SHALL 仅在特定路径变化时触发自动打 Tag。

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
- **WHEN** docs、.vuepress、README.md 或其他非触发路径变更
- **THEN** 不触发自动打 Tag workflow

#### Scenario: 主分支限定
- **WHEN** 变化推送到的分支不是 main
- **THEN** 不触发自动打 Tag workflow

### Requirement: Tag 推送触发发布

系统 SHALL 在 Tag 推送时触发 npm 和 Docker 发布流程。

#### Scenario: Tag 推送事件
- **WHEN** Tag 推送到 GitHub 仓库
- **THEN** publish workflow 自动触发
- **AND** 执行 npm 发布和 Docker 镜像构建推送

#### Scenario: 版本号传递
- **WHEN** Tag 为 `2026.4.17-1503`
- **THEN** npm 包版本设置为 `2026.4.17-1503`
- **AND** Docker 镜像版本 tag 为 `2026.4.17-1503-cpu` 和 `2026.4.17-1503-gpu`

### Requirement: workflow 并发控制

系统 SHALL 防止同时运行多个版本发布 workflow。

#### Scenario: 同一 Tag 并发控制
- **WHEN** 相同 Tag 的 workflow 已在运行
- **THEN** 新的 workflow 被中止
- **AND** 显示并发冲突提示

#### Scenario: 版本序列控制
- **WHEN** 多个 Tag 同时推送
- **THEN** 按 Tag 时间顺序依次执行
- **AND** 前一个完成后执行下一个

### Requirement: Tag 创建通知

系统 SHALL 在 Tag 创建成功后提供通知。

#### Scenario: workflow 日志
- **WHEN** Tag 创建完成
- **THEN** workflow 日志显示创建的 Tag 名称
- **AND** 显示触发变更的文件列表

#### Scenario: 失败通知
- **WHEN** Tag 创建失败
- **THEN** workflow 标记为失败
- **AND** 显示错误原因

### Requirement: 手动触发支持

系统 SHALL 支持手动触发版本发布。

#### Scenario: workflow_dispatch 触发
- **WHEN** 用户在 GitHub Actions 界面手动触发 publish workflow
- **THEN** workflow 执行发布流程
- **AND** 使用当前时间自动生成 Tag

#### Scenario: 无变更手动发布
- **WHEN** 手动触发且无实际代码变更
- **THEN** 仍执行发布流程
- **AND** 发布版本与上次内容相同但版本号不同