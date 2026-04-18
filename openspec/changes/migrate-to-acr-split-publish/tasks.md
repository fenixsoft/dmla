## 1. 创建 npm 自动打 Tag Workflow

- [x] 1.1 创建 `.github/workflows/auto-tag-npm.yml`
- [x] 1.2 配置触发路径为 `packages/cli/**` 和 `packages/install/**`
- [x] 1.3 实现 npm-前缀的时间戳 Tag 格式

## 2. 创建 npm 独立发布 Workflow

- [x] 2.1 创建 `.github/workflows/publish-npm.yml`
- [x] 2.2 配置 Tag 触发条件为 `npm-[0-9]+.[0-9]+.[0-9]+-*`
- [x] 2.3 实现 npm 包版本更新和发布逻辑
- [x] 2.4 添加 workflow_dispatch 手动触发支持

## 3. 创建 Docker 独立发布 Workflow

- [x] 3.1 创建 `.github/workflows/publish-docker.yml`
- [x] 3.2 配置 Tag 触发条件为 `[0-9]+.[0-9]+.[0-9]+-*`（不含 npm- 前缀）
- [x] 3.3 实现 Docker 镜像构建和推送逻辑
- [x] 3.4 添加 workflow_dispatch 手动触发支持

## 4. 配置阿里云 ACR 集成

- [x] 4.1 在 publish-docker.yml 中添加 ACR 登录步骤
- [x] 4.2 配置 ACR Registry 地址 `crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com`
- [x] 4.3 实现镜像 Tag 和推送到 ACR 的逻辑
- [x] 4.4 移除 TCR 推送相关代码

## 5. 清理旧 Workflow

- [x] 5.1 删除 `.github/workflows/publish.yml`
- [x] 5.2 更新 `auto-tag.yml` 确保 Docker 版本格式不变

## 6. Secrets 配置和文档更新

- [ ] 6.1 在 GitHub Settings 中配置 ACR_USERNAME Secret（需用户手动操作）
- [ ] 6.2 在 GitHub Settings 中配置 ACR_PASSWORD Secret（需用户手动操作）
- [x] 6.3 更新 CLAUDE.md 中的发布流程说明
- [ ] 6.4 测试 workflow_dispatch 手动触发功能（需用户手动操作）

## 7. CLI 工具 ACR 支持（额外发现）

- [x] 7.1 更新 `packages/install/src/modules/docker.js` 中的 registry 配置
- [x] 7.2 更新 `packages/cli/src/commands/manage.js` 中的 registry 配置
- [x] 7.3 更新 `packages/cli/src/index.js` 命令参数描述
- [x] 7.4 更新 `packages/install/tests/image-mapping.test.js` 测试文件
- [x] 7.5 运行测试验证修改正确

## 8. TUI 安装界面更新

- [x] 8.1 更新 `packages/install/src/index.js` 中的 registry 选择选项

## 9. 文档更新

- [x] 9.1 更新 `docs/sandbox.md`
- [x] 9.2 更新 `docs/getting-started/installation.md`
- [x] 9.3 更新 `docs/getting-started/cli-reference.md`
- [x] 9.4 运行构建验证文档修改

## 10. 活跃 Spec 文件更新

- [x] 10.1 更新 `openspec/specs/installation-cli/spec.md`
- [x] 10.2 更新 `openspec/specs/installation-tui/spec.md`
- [x] 10.3 更新 `openspec/specs/sandbox-settings/spec.md`