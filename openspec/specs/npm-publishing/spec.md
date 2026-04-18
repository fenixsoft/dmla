# npm 发布功能规范

## ADDED Requirements

### Requirement: 自动发布 npm 包

系统 SHALL 在 Git Tag 推送时自动将 `@dmla/cli` 包发布到 npm 公共仓库。

#### Scenario: Tag 触发发布
- **WHEN** Git Tag 格式为 `YYYY.M.D-HHMM`（如 `2026.4.17-1503`）推送到仓库
- **THEN** GitHub Actions 自动执行 npm 发布流程
- **AND** 包版本号与 Tag 版本号一致

#### Scenario: 发布成功验证
- **WHEN** npm 发布流程完成
- **THEN** 用户可通过 `npm install -g @dmla/cli@2026.4.17-1503` 安装指定版本
- **AND** 用户可通过 `npm view @dmla/cli versions` 查看所有已发布版本

#### Scenario: 手动触发发布
- **WHEN** 用户在 GitHub Actions 界面手动触发 publish workflow
- **THEN** 系统使用当前最新 commit 执行发布流程
- **AND** 自动生成当前时间戳版本的 Tag

### Requirement: 包元数据完整性

系统 SHALL 确保 npm 包包含完整的元数据信息。

#### Scenario: 包信息完整
- **WHEN** 用户查看 npm 包信息
- **THEN** 包含 name、version、description、author、license、repository、keywords 字段
- **AND** 包含正确的 bin 字段指向 CLI 入口

#### Scenario: 包依赖声明
- **WHEN** 用户安装 npm 包
- **THEN** 自动安装所有声明的外部依赖
- **AND** 依赖版本锁定，确保一致性

### Requirement: 发布失败处理

系统 SHALL 在发布失败时提供明确的错误信息并中止后续步骤。

#### Scenario: npm 认证失败
- **WHEN** NPM_TOKEN Secret 无效或过期
- **THEN** workflow 失败并显示认证错误信息
- **AND** 不执行后续的 Docker 镜像推送步骤

#### Scenario: 版本已存在
- **WHEN** 尝试发布的版本号已存在于 npm 仓库
- **THEN** workflow 失败并显示版本冲突错误
- **AND** 提示用户检查 Tag 是否正确

### Requirement: CLI 入口配置

系统 SHALL 配置 npm 包的 bin 入口，使用户安装后可直接运行 `dmla` 命令。

#### Scenario: 命令可用
- **WHEN** 用户执行 `npm install -g @dmla/cli`
- **THEN** `dmla` 命令在终端可直接调用
- **AND** `dmla --help` 显示帮助信息

#### Scenario: 命令执行入口
- **WHEN** 用户运行 `dmla` 命令
- **THEN** 系统执行 bin 字段指定的入口脚本
- **AND** 正确解析命令行参数