## ADDED Requirements

### Requirement: npm 包版本自动生成 Tag

系统 SHALL 在 packages/ 目录变更时自动生成 npm 版本的 Git Tag。

#### Scenario: 触发路径限定
- **WHEN** `packages/cli/` 或 `packages/install/` 目录下任何文件变更推送到 main 分支
- **THEN** 触发 npm-auto-tag workflow
- **AND** 生成时间戳格式的 Git Tag

#### Scenario: 时间戳格式
- **WHEN** 自动生成 npm Tag
- **THEN** Tag 格式为 `npm-YYYY.M.D-HHMM`
- **AND** 使用当前 UTC+8 时间
- **AND** Tag 唯一性检查防止重复

#### Scenario: 非触发路径不触发
- **WHEN** docs、local-server 或其他非 packages 目录变更
- **THEN** 不触发 npm-auto-tag workflow

### Requirement: Tag 推送触发 npm 发布

系统 SHALL 在 npm Tag 推送时触发 npm 包发布流程。

#### Scenario: Tag 格式匹配
- **WHEN** Tag 格式为 `npm-YYYY.M.D-HHMM` 推送到仓库
- **THEN** publish-npm workflow 自动触发
- **AND** npm 包版本设置为 `YYYY.M.D-HHMM`（不含 npm- 前缀）

#### Scenario: 手动触发发布
- **WHEN** 用户在 GitHub Actions 界面手动触发 publish-npm workflow
- **THEN** workflow 生成当前时间戳版本并发布
- **AND** 创建对应的 Git Tag

### Requirement: workflow 并发控制

系统 SHALL 防止同时运行多个 npm 发布 workflow。

#### Scenario: 同一版本并发控制
- **WHEN** 相同 npm Tag 的 workflow 已在运行
- **THEN** 新的 workflow 等待或中止
- **AND** 显示并发状态提示