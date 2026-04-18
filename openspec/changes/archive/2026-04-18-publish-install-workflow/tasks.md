## 1. 基础设施准备

- [ ] 1.1 创建 Docker Hub 命名空间 `icyfenix/dmla-sandbox` ⚠️ 需手动操作
- [ ] 1.2 创建腾讯云 TCR 命名空间 `ccr.ccs.tencentyun.com/icyfenix/dmla-sandbox` ⚠️ 需手动操作
- [ ] 1.3 创建 npm 组织 `@dmla` 并注册账户 ⚠️ 需手动操作
- [ ] 1.4 配置 GitHub Secrets：NPM_TOKEN、DOCKER_USERNAME、DOCKER_PASSWORD、TCR_USERNAME、TCR_PASSWORD ⚠️ 需手动操作
- [x] 1.5 更新 Dockerfile.sandbox 中镜像名称为 `dmla-sandbox:gpu`
- [x] 1.6 更新 Dockerfile.sandbox.cpu 中镜像名称为 `dmla-sandbox:cpu`
- [x] 1.7 更新 sandbox.js 中镜像名称定义

## 2. Monorepo 结构搭建

- [x] 2.1 创建 packages 目录结构
- [x] 2.2 创建 packages/cli 目录和基础文件
- [x] 2.3 创建 packages/install 目录和基础文件
- [x] 2.4 更新根 package.json 添加 workspaces 配置
- [x] 2.5 创建 packages/cli/package.json（包名 @dmla/cli）
- [x] 2.6 创建 packages/install/package.json（包名 @dmla/install）
- [x] 2.7 配置 packages/cli/bin 字段指向 CLI 入口

## 3. GitHub Actions 发布工作流

- [x] 3.1 创建 .github/workflows/publish.yml 统一发布工作流
- [x] 3.2 配置 workflow 触发条件（push tags: 'v*' 和 workflow_dispatch）
- [x] 3.3 实现 publish-npm job：npm 版本设置和发布
- [x] 3.4 实现 build-images job：使用 matrix 构建 CPU 和 GPU 镜像
- [x] 3.5 实现 push-dockerhub job：推送到 Docker Hub（稳定版 + 版本号）
- [x] 3.6 实现 push-tcr job：推送到腾讯云 TCR（稳定版 + 版本号）
- [x] 3.7 配置 job 依赖顺序：npm → build → dockerhub → tcr
- [x] 3.8 添加镜像测试运行验证步骤

## 4. 自动 Tag 生成工作流

- [x] 4.1 创建 .github/workflows/auto-tag.yml 自动打 Tag 工作流
- [x] 4.2 配置触发路径限定（Dockerfile、shared_modules、kernel_runner.py）
- [x] 4.3 配置分支限定（仅 main 分支触发）
- [x] 4.4 实现时间戳 Tag 生成逻辑（YYYY.M.D-HHMM）
- [x] 4.5 实现 Tag 唯一性检查（已存在则跳过）
- [x] 4.6 配置并发控制（防止重复触发）
- [x] 4.7 添加 workflow 输出显示创建的 Tag 名称

## 5. CLI 命令实现

- [x] 5.1 创建 packages/cli/src/index.js CLI 入口
- [x] 5.2 实现 dmla start 命令（默认端口、自定义端口、GPU 选项）
- [x] 5.3 实现 dmla stop 命令
- [x] 5.4 实现 dmla status 命令（服务状态、GPU 信息、版本信息）
- [x] 5.5 实现 dmla install 命令（--cpu、--gpu、--all、--registry 选项）
- [x] 5.6 实现 dmla update 命令（npm 更新 + 镜像检查更新）
- [x] 5.7 实现 dmla doctor 命令（Docker 检查、镜像检查、GPU 检查、端口检查、网络检查）
- [x] 5.8 实现 dmla --help 和 dmla --version 命令
- [x] 5.9 创建命令行参数解析模块（使用 commander 或 yargs）
- [x] 5.10 创建服务管理模块（启动、停止、状态检测）

## 6. TUI 安装模块实现

- [x] 6.1 创建 packages/install/src/index.js TUI 入口
- [x] 6.2 实现环境检测模块（Docker、Node.js、GPU）
- [x] 6.3 实现镜像仓库选择界面（使用 enquirer）
- [x] 6.4 实现镜像类型选择界面
- [x] 6.5 实现端口配置界面（含可用性检测）
- [x] 6.6 实现 docker pull 输出解析模块
- [x] 6.7 实现进度条显示模块
- [x] 6.8 实现镜像名称映射模块（docker tag 操作）
- [x] 6.9 实现 npm 包安装模块
- [x] 6.10 实现安装验证模块（服务启动测试、健康检查）
- [x] 6.11 实现安装中断恢复逻辑

## 7. install.sh 启动脚本

- [x] 7.1 创建 docs/.vuepress/public/install.sh 文件
- [x] 7.2 实现 Docker 环境检测
- [x] 7.3 实现 Node.js 环境检测
- [x] 7.4 实现 npx @dmla/install 调用
- [x] 7.5 实现错误处理和用户提示
- [x] 7.6 验证脚本在常见 Linux 发行版和 macOS 上运行

## 8. 本地镜像名称兼容性

- [x] 8.1 更新 local-server/src/sandbox.js 镜像名称定义
- [x] 8.2 确保 CPU 镜像名为 dmla-sandbox:cpu（去除 :latest 别名）
- [x] 8.3 确保 GPU 镜像名为 dmla-sandbox:gpu
- [x] 8.4 更新 package.json 中 build:sandbox 脚本镜像名称
- [x] 8.5 更新 package.json 中 build:sandbox:cpu 脚本镜像名称

## 9. 单元测试

- [x] 9.1 创建 packages/cli/tests/ 目录
- [x] 9.2 编写 CLI 命令单元测试
- [x] 9.3 创建 packages/install/tests/ 目录
- [x] 9.4 编写 TUI 模块单元测试
- [x] 9.5 编写 docker pull 输出解析测试
- [x] 9.6 编写镜像名称映射测试
- [x] 9.7 配置 CI 中运行测试

## 10. 文档更新

- [x] 10.1 更新 README.md 安装说明
- [x] 10.2 创建 docs/getting-started/installation.md 安装指南
- [x] 10.3 创建 docs/getting-started/cli-reference.md CLI 命令参考
- [x] 10.4 更新 docs/arch/design.md 架构文档
- [x] 10.5 更新 CLAUDE.md 常用命令部分

## 11. 发布验证

- [ ] 11.1 手动触发 publish workflow 测试 ⚠️ 需手动操作
- [ ] 11.2 验证 npm 包发布成功并可安装 ⚠️ 需手动操作
- [ ] 11.3 验证 Docker Hub 镜像推送成功并可拉取 ⚠️ 需手动操作
- [ ] 11.4 验证 TCR 镜像推送成功并可拉取 ⚠️ 需手动操作
- [ ] 11.5 测试 install.sh 一键安装流程 ⚠️ 需手动操作
- [ ] 11.6 测试 npx @dmla/install 安装流程 ⚠️ 需手动操作
- [ ] 11.7 测试 dmla update 命令功能 ⚠️ 需手动操作
- [ ] 11.8 测试国内用户通过 TCR 安装的速度 ⚠️ 需手动操作