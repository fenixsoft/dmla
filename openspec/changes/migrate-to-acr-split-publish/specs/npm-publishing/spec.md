## MODIFIED Requirements

### Requirement: 自动发布 npm 包

系统 SHALL 在 npm 格式的 Git Tag 推送时自动将 npm 包发布到 npm 公共仓库。

#### Scenario: Tag 触发发布
- **WHEN** Git Tag 格式为 `npm-YYYY.M.D-HHMM`（如 `npm-2026.4.17-1503`）推送到仓库
- **THEN** GitHub Actions 自动执行 npm 发布流程
- **AND** 包版本号与 Tag 版本号一致（不含 npm- 前缀）

#### Scenario: 发布成功验证
- **WHEN** npm 发布流程完成
- **THEN** 用户可通过 `npm install -g @icyfenix-dmla/cli@2026.4.17-1503` 安装指定版本
- **AND** 用户可通过 `npm view @icyfenix-dmla/cli versions` 查看所有已发布版本

#### Scenario: 手动触发发布
- **WHEN** 用户在 GitHub Actions 界面手动触发 publish-npm workflow
- **THEN** 系统使用当前最新 commit 执行发布流程
- **AND** 自动生成当前时间戳版本的 Tag

### Requirement: 发布失败处理

系统 SHALL 在发布失败时提供明确的错误信息并中止后续步骤。

#### Scenario: npm 认证失败
- **WHEN** NPM_TOKEN Secret 无效或过期
- **THEN** workflow 失败并显示认证错误信息

#### Scenario: 版本已存在
- **WHEN** 尝试发布的版本号已存在于 npm 仓库
- **THEN** workflow 失败并显示版本冲突错误
- **AND** 提示用户检查 Tag 是否正确

## REMOVED Requirements

### Requirement: 与 Docker 镜像发布绑定

**Reason**: npm 和 Docker 发布流程已拆分为独立 workflow

**Migration**: Docker 镜像使用独立的 publish-docker workflow，通过 Docker Tag（无 npm- 前缀）触发