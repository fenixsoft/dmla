## Context

DMLA（Design Machine Learning Applications）是一个机器学习教育平台，包含 VuePress 文档站点和 Python 代码沙箱执行服务。当前架构：

```
dmla/
├── docs/                    # VuePress 文档站点
├── local-server/            # 沙箱 API 服务
│   ├── src/                 # Express API + Docker 管理
│   ├── shared_modules/      # Python 共享模块
│   └── Dockerfile.sandbox   # GPU 镜像
│   └── Dockerfile.sandbox.cpu # CPU 镜像
└── .github/workflows/       # CI/CD（仅 VuePress 部署）
```

**当前问题**：
1. 用户安装需要完整源码 + 本地构建镜像（约 10-30 分钟）
2. 无版本发布机制，无法追踪变更
3. 国内用户访问 Docker Hub 速度慢

**约束条件**：
- npm 包和 Docker 镜像版本号各自独立演进
- 镜像名称统一为 `dmla-sandbox:cpu/gpu`
- 时间戳版本格式：`YYYY.M.D-HHMM`
- 仅 Dockerfile/shared_modules 变化触发版本发布

## Goals / Non-Goals

**Goals:**
- 建立自动化发布流程，npm 包和 Docker 镜像同步发布
- 降低用户安装门槛，从 30+ 分钟降至 5 分钟以内
- 提供国内加速选项（腾讯云 TCR）
- 提供友好的 TUI 安装体验和 CLI 管理工具
- 建立版本追溯机制，支持用户更新

**Non-Goals:**
- 不改变沙箱核心执行逻辑
- 不改变 API 接口定义
- 不支持离线安装（仍需网络拉取镜像）
- 不提供 Web 界面安装向导

## Decisions

### 1. 版本触发策略

**决策**：仅 Dockerfile 和 shared_modules 变化触发自动打 Tag

**原因**：
- 镜像内容 = Dockerfile + shared_modules + kernel_runner.py
- API 路由变更不影响镜像，无需重新发布
- 文档变更频繁，不应触发版本发布
- 减少不必要的版本发布噪音

**触发路径**：
```yaml
paths:
  - 'local-server/Dockerfile.sandbox'
  - 'local-server/Dockerfile.sandbox.cpu'
  - 'local-server/shared_modules/**'
  - 'local-server/src/kernel_runner.py'
```

**排除路径**：docs/**, .vuepress/**, README.md, 其他文件

**替代方案**：全路径触发 → 拒绝，因为会产生过多无用版本

### 2. 版本号格式

**决策**：时间戳格式 `YYYY.M.D-HHMM`（如 `2026.4.17-1503`）

**原因**：
- 简单直观，无需人工维护版本号
- 符合 npm 语义化版本规范（major.minor.patch-prerelease）
- 自动生成，避免版本号冲突
- Git Tag、npm 版本、镜像版本统一

**替代方案**：
- 语义化版本（如 1.2.3）→ 拒绝，需人工判断 major/minor/patch
- Git commit hash → 拒绝，不直观，难以追溯时间

### 3. 镜像仓库策略

**决策**：同时推送到 Docker Hub 和腾讯云 TCR

**镜像命名**：
| 仓库 | 镜像名 | 示例 |
|------|--------|------|
| Docker Hub | `icyfenix/dmla-sandbox:cpu` | `icyfenix/dmla-sandbox:2026.4.17-1503-cpu` |
| TCR | `ccr.ccs.tencentyun.com/icyfenix/dmla-sandbox:gpu` | 同上 |

**原因**：
- Docker Hub：全球标准，无需登录
- TCR：国内加速，腾讯云用户友好
- 双推送确保不同地区用户都能快速访问

**替代方案**：
- 仅 Docker Hub → 拒绝，国内访问慢
- 仅 TCR → 拒绝，国际用户访问受限

### 4. 镜像名称统一

**决策**：仓库命名空间改为 `dmla-sandbox`，与本地镜像名一致

**原因**：
- 避免名称映射复杂性
- 用户本地和远程镜像名一致
- 减少安装脚本中的 tag 重命名步骤

**实施**：
- Docker Hub：`icyfenix/dmla-sandbox`
- TCR：`ccr.ccs.tencentyun.com/icyfenix/dmla-sandbox`
- 本地：`dmla-sandbox:cpu/gpu`

### 5. npm 包结构

**决策**：使用 monorepo 结构，packages 目录下分离 CLI 和安装包

```
packages/
├── cli/          # @dmla/cli - 服务运行和命令管理
└── install/      # @dmla/install - TUI 安装向导
```

**原因**：
- 功能分离，职责清晰
- 可独立安装（用户可选安装 CLI 或仅使用 install）
- 便于维护和测试

**替代方案**：
- 单包包含所有功能 → 拒绝，职责混乱
- 保持 local-server 目录 → 拒绝，不利于 npm 发布

### 6. CLI 命令设计

**决策**：提供完整的生命周期管理命令

```bash
dmla start [--port 3001] [--gpu]  # 启动服务
dmla stop                          # 停止服务
dmla status                        # 状态查看
dmla install [--cpu|--gpu|--all]   # 安装镜像
dmla update                        # 更新 npm 包和镜像
dmla doctor                        # 环境诊断
```

**原因**：
- 覆盖用户完整使用场景
- update 命令同时更新 npm 包和镜像，简化升级流程
- doctor 命令帮助用户排查环境问题

### 7. TUI 技术选型

**决策**：使用 Node.js + enquirer，通过 npx 运行

**原因**：
- enquirer 提供美观的交互界面
- npx 无需预先安装
- 与 npm 包生态一致

**替代方案**：
- Bash + dialog → 拒绝，TUI 功能有限，进度显示复杂
- Python + rich → 拒绝，与项目 Node.js 技术栈不一致

### 8. install.sh 部署位置

**决策**：放置于 VuePress public 目录，URL 为 `https://ai.icyfenix.cn/install.sh`

**原因**：
- 与文档站点同域名，无需额外配置
- VuePress 构建自动同步
- CDN 加速访问

**替代方案**：
- GitHub raw URL → 拒绝，国内访问不稳定
- 独立域名 → 拒绝，增加配置复杂度

## Risks / Trade-offs

### Risk 1: 时间戳版本冲突
- **风险**：极端情况下同一分钟多次推送
- **缓解**：GitHub Actions 并发控制，同一 tag 已存在时跳过

### Risk 2: npm 发布失败
- **风险**：npm 发布失败但镜像已推送，版本不一致
- **缓解**：workflow 中 npm 发布先于镜像推送，失败时中止后续步骤

### Risk 3: TCR 认证过期
- **风险**：TCR Secrets 过期导致推送失败
- **缓解**：定期检查 Secrets 有效期，workflow 失败时告警

### Risk 4: 镜像拉取中断
- **风险**：大镜像（GPU 版本约 2GB）拉取中断
- **缓解**：TUI 显示进度，支持断点续传提示

### Risk 5: 用户本地镜像版本混乱
- **风险**：用户本地可能有多个版本镜像
- **缓解**：`dmla update` 自动清理旧版本镜像

## Migration Plan

### Phase 1: 基础设施准备
1. 创建 Docker Hub 命名空间 `icyfenix/dmla-sandbox`
2. 创建腾讯云 TCR 命名空间
3. 创建 npm 组织 `@dmla`
4. 配置 GitHub Secrets

### Phase 2: 代码改造
1. 创建 packages 目录结构
2. 实现 CLI 命令模块
3. 实现 TUI 安装模块
4. 创建 GitHub Actions workflow

### Phase 3: 发布验证
1. 手动触发 workflow 测试
2. 验证 npm 包发布
3. 验证 Docker Hub 推送
4. 验证 TCR 推送
5. 测试安装脚本

### Phase 4: 文档更新
1. 更新 README 安装说明
2. 创建用户安装指南

### Rollback Strategy
- GitHub workflow 可随时禁用
- npm 包可撤回已发布版本（72小时内）
- Docker 镜像可删除特定 tag
- install.sh 可替换为旧版本说明

## Open Questions

1. **GPU 检测准确性**：TUI 中 GPU 检测依赖 `nvidia-smi`，虚拟机环境可能不准确。是否需要用户手动确认？

2. **镜像清理策略**：是否在 `dmla update` 时自动清理旧版本镜像？清理策略是什么（保留最近 N 个版本）？

3. **TCR 登录需求**：TCR 公开镜像是否需要用户登录？文档说明需要明确。

4. **本地开发兼容**：开发者本地构建的镜像名是否保持 `dmla-sandbox:cpu/gpu`？这会与远程拉取的镜像名冲突。