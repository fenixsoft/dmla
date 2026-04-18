## Context

当前项目使用单一 `publish.yml` workflow 处理 npm 包和 Docker 镜像的发布，两者绑定在相同的 Git Tag 版本号上。`auto-tag.yml` 仅监控 Docker 相关文件变更（Dockerfile、kernel_runner），npm 包变更无法自动触发版本发布。

阿里云 ACR 个人版镜像仓库地址：`crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com`

## Goals / Non-Goals

**Goals:**
- npm 包和 Docker 镜像独立版本管理，各自维护时间戳版本号
- npm 包变更（packages/ 目录）自动触发 npm 版本发布
- Docker 相关变更继续使用现有 auto-tag 机制
- Docker 镜像推送到阿里云 ACR，替代腾讯云 TCR

**Non-Goals:**
- 不改变现有版本号格式（YYYY.M.D-HHMM）
- 不改变 Docker Hub 推送流程
- 不实现 npm 和 Docker 版本的同步机制

## Decisions

### 1. Workflow 拆分策略

**决定**: 将 `publish.yml` 拆分为 `publish-npm.yml` 和 `publish-docker.yml`

**理由**:
- 独立的 workflow 允许独立触发和版本管理
- 当前 publish.yml 的 job 依赖关系（npm → Docker）造成不必要的耦合
- 拆分后每个 workflow 更简洁，易于维护

**替代方案**:
- 保持单一 workflow，通过条件判断分离执行 → 版本号仍会耦合，不采用

### 2. npm 自动打 Tag 机制

**决定**: 新增 `auto-tag-npm.yml`，监控 `packages/` 目录变更

**理由**:
- npm 包变更有独立的版本迭代需求
- 与 Docker auto-tag 使用相同的时间戳格式，避免混淆

### 3. 阿里云 ACR 替代腾讯云 TCR

**决定**: 使用阿里云 ACR 个人版镜像仓库

**理由**:
- 用户反馈 TCR 国内推送速度过慢
- ACR 国内节点覆盖更广，杭州地域稳定
- 个人版免费，满足项目需求

**Secrets 配置**:
- `ACR_USERNAME`: `icyfenix`
- `ACR_PASSWORD`: 从阿里云控制台获取
- Registry: `crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com`

### 4. 版本号策略

**决定**: npm 和 Docker 版本号各自独立生成，不强制同步

**理由**:
- npm 包和 Docker 镜像发布频率不同
- 时间戳格式天然保证唯一性，无需协调
- 用户可通过版本号判断发布类型（npm-only 或 docker-only）

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| 两个 workflow 同时触发可能导致相同版本号 | 时间戳精度为分钟级，实际冲突概率极低；如冲突可手动触发 |
| ACR 认证失败导致镜像无法推送 | 保留 Docker Hub 作为主要仓库，ACR 为国内加速备选 |
| 旧版本 publish.yml 删除后历史 workflow 记录丢失 | Git 历史保留删除前的 workflow 定义，不影响已执行的 release 记录 |

## Migration Plan

1. 创建新的 workflow 文件（auto-tag-npm.yml、publish-npm.yml、publish-docker.yml）
2. 在 GitHub Settings → Secrets 中配置 ACR_USERNAME 和 ACR_PASSWORD
3. 测试新 workflow 的手动触发功能
4. 删除旧的 publish.yml
5. 更新 README 和 CLAUDE.md 中的发布命令说明