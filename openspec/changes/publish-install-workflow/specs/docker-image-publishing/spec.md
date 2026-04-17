# Docker 镜像发布功能规范

## ADDED Requirements

### Requirement: 自动构建 Docker 镜像

系统 SHALL 在 Git Tag 推送时自动构建 CPU 和 GPU 两个版本的 Docker 镜像。

#### Scenario: CPU 镜像构建
- **WHEN** publish workflow 执行
- **THEN** 使用 `Dockerfile.sandbox.cpu` 构建镜像
- **AND** 镜像名为 `dmla-sandbox:cpu`

#### Scenario: GPU 镜像构建
- **WHEN** publish workflow 执行
- **THEN** 使用 `Dockerfile.sandbox` 构建镜像
- **AND** 镜像名为 `dmla-sandbox:gpu`

#### Scenario: 共享模块提取
- **WHEN** 镜像构建开始
- **THEN** 先执行 `npm run extract:shared` 提取共享模块
- **AND** 共享模块文件复制到镜像中

### Requirement: 推送到 Docker Hub

系统 SHALL 将构建的镜像推送到 Docker Hub 公开仓库。

#### Scenario: 稳定版推送
- **WHEN** 镜像构建完成
- **THEN** 推送 `icyfenix/dmla-sandbox:cpu` 到 Docker Hub
- **AND** 推送 `icyfenix/dmla-sandbox:gpu` 到 Docker Hub

#### Scenario: 版本号推送
- **WHEN** Tag 版本为 `2026.4.17-1503`
- **THEN** 推送 `icyfenix/dmla-sandbox:2026.4.17-1503-cpu`
- **AND** 推送 `icyfenix/dmla-sandbox:2026.4.17-1503-gpu`

#### Scenario: 公开访问
- **WHEN** 镜像推送成功
- **THEN** 用户无需登录即可拉取镜像
- **AND** `docker pull icyfenix/dmla-sandbox:cpu` 命令可用

### Requirement: 推送到腾讯云 TCR

系统 SHALL 将构建的镜像推送到腾讯云容器镜像服务 TCR。

#### Scenario: TCR 稳定版推送
- **WHEN** Docker Hub 推送完成
- **THEN** 推送 `ccr.ccs.tencentyun.com/icyfenix/dmla-sandbox:cpu` 到 TCR
- **AND** 推送 `ccr.ccs.tencentyun.com/icyfenix/dmla-sandbox:gpu` 到 TCR

#### Scenario: TCR 版本号推送
- **WHEN** Tag 版本为 `2026.4.17-1503`
- **THEN** 推送带版本号的镜像到 TCR
- **AND** 镜像名格式与 Docker Hub 一致

#### Scenario: 国内用户加速
- **WHEN** 国内用户拉取镜像
- **THEN** 使用 TCR 地址可获得更快的下载速度
- **AND** 无需登录即可拉取公开镜像

### Requirement: 镜像完整性验证

系统 SHALL 在推送前验证镜像构建完整性。

#### Scenario: 镜像测试运行
- **WHEN** 镜像构建完成
- **THEN** 执行简单测试命令验证镜像可用
- **AND** `docker run --rm dmla-sandbox:cpu python3 -c "print('OK')"` 输出 "OK"

#### Scenario: 镜像大小检查
- **WHEN** 镜像构建完成
- **THEN** 检查镜像大小在预期范围内
- **AND** CPU 版本小于 1.5GB，GPU 版本小于 2.5GB

### Requirement: 发布失败回滚

系统 SHALL 在镜像推送失败时不影响已成功的推送。

#### Scenario: Docker Hub 失败
- **WHEN** Docker Hub 推送失败
- **THEN** workflow 显示错误信息
- **AND** 不执行后续 TCR 推送

#### Scenario: TCR 推送失败
- **WHEN** Docker Hub 成功但 TCR 失败
- **THEN** Docker Hub 镜像已可访问
- **AND** workflow 标记为部分成功，显示 TCR 错误

### Requirement: 镜像内容一致性

系统 SHALL 确保 CPU 和 GPU 镜像的 Python 环境和共享模块内容一致。

#### Scenario: Python 版本一致
- **WHEN** 用户在 CPU 或 GPU 镜像中运行 Python
- **THEN** Python 版本均为 3.11
- **AND** pip 安装的包列表一致

#### Scenario: 共享模块一致
- **WHEN** 用户导入共享模块
- **THEN** CPU 和 GPU 镜像中的 shared 模块内容相同
- **AND** 模块路径为 `/usr/local/lib/python3.11/site-packages/shared`