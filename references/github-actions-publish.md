# GitHub Actions 发布流程

项目使用独立的 npm 和 Docker 发布流程：

| Workflow | 触发条件 | Tag 格式 | 说明 |
|----------|----------|----------|------|
| `auto-tag-npm.yml` | packages/ 目录变更 | `npm-YYYY.M.D-HHMM` | npm 包自动打 Tag |
| `auto-tag.yml` | local-server/ Docker 相关变更 | `YYYY.M.D-HHMM` | Docker 镜像自动打 Tag |
| `publish-npm.yml` | npm- 开头的 Tag | - | 发布 npm 包 |
| `publish-docker.yml` | 非 npm- 开头的 Tag | - | 发布 Docker 镜像 |

**镜像仓库**：
- Docker Hub: `icyfenix/dmla-sandbox`（全球用户）
- 阿里云 ACR: `crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox`（国内加速）

**Secrets 配置**：
- `NPM_TOKEN`: npm 发布认证
- `DOCKER_USERNAME/DOCKER_PASSWORD`: Docker Hub 认证
- `ACR_USERNAME/ACR_PASSWORD`: 阿里云 ACR 认证
- `ALIBABA_CLOUD_ACCESS_KEY_ID/ALIBABA_CLOUD_ACCESS_KEY_SECRET`：阿里云认证信息
- `TENCENT_SECRET_ID/TENCENT_SECRET_KEY`：腾讯云认证信息
