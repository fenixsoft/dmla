## ADDED Requirements

### Requirement: 推送到阿里云 ACR

系统 SHALL 将构建的 Docker 镜像推送到阿里云容器镜像服务 ACR。

#### Scenario: ACR 认证配置
- **WHEN** workflow 执行镜像推送
- **THEN** 使用 ACR_USERNAME 和 ACR_PASSWORD Secrets 进行认证
- **AND** Registry 地址为 `crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com`

#### Scenario: ACR 稳定版推送
- **WHEN** 镜像构建完成
- **THEN** 推送镜像到 ACR，格式为 `crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/icyfenix/dmla-sandbox:cpu`
- **AND** 推送 GPU 版本镜像

#### Scenario: ACR 版本号推送
- **WHEN** Tag 版本为 `2026.4.17-1503`
- **THEN** 推送带版本号的镜像到 ACR
- **AND** 镜像名格式与 Docker Hub 一致

#### Scenario: 国内用户加速
- **WHEN** 国内用户拉取镜像
- **THEN** 使用 ACR 地址可获得更快的下载速度
- **AND** 无需登录即可拉取公开镜像