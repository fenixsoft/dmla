## Phase 1: CLI `dmla data` 命令实现

- [x] 1.1 创建 `packages/cli/src/commands/data.js` 文件
- [x] 1.2 实现 TUI 菜单框架（使用 @icyfenix-dmla/install 的 TUI 组件）
- [x] 1.3 实现挂载路径设置功能（mount）
- [x] 1.4 实现显示实际路径功能（path）
- [x] 1.5 实现清空数据内容功能（clear）
- [x] 1.6 实现删除数据卷功能（remove）
- [x] 1.7 在 `packages/cli/src/index.js` 中注册 data 命令
- [x] 1.8 创建/更新 `~/.dmla/config.json` 存储用户配置

## Phase 2: 数据集下载功能

- [x] 2.1 创建数据集下载配置文件（包含 URL、大小、格式等信息）
- [x] 2.2 实现 download 子菜单（显示可用数据集列表）
- [x] 2.3 实现 curl/wget 下载逻辑，显示原始进度输出
- [x] 2.4 实现下载到 cache/downloads 目录
- [x] 2.5 实现解压逻辑（支持 ZIP、TAR.GZ）
- [x] 2.6 实现解压后移动到目标 datasets 目录
- [x] 2.7 实现下载状态检测（已下载的数据集显示标记）

## Phase 3: Docker 镜像修改

- [x] 3.1 修改 Dockerfile.sandbox，创建 /data 目录结构
- [x] 3.2 创建 `local-server/src/dmla_progress.py` 进度报告模块
- [x] 3.3 设置 TORCH_HOME 环境变量
- [x] 3.4 重新构建 GPU 镜像验证修改（代码已完成，构建由用户执行）

## Phase 4: 沙箱执行系统增强

- [x] 4.1 修改 `local-server/src/sandbox.js`，添加 data volume 挂载逻辑
- [x] 4.2 实现 getDataVolumePath() 函数，读取用户配置
- [x] 4.3 修改 `packages/cli/src/server/sandbox.js` 同步修改
- [x] 4.4 实现进度轮询机制（poll progress.json）
- [x] 4.5 实现进度 SSE 推送到前端
- [x] 4.6 修复 PYTHONPATH 问题，添加 /workspace 到模块搜索路径

## Phase 5: 前端进度条组件

- [x] 5.1 创建 VuePress 进度条组件
- [x] 5.2 接收 SSE 进度更新事件
- [x] 5.3 显示进度百分比和消息
- [x] 5.4 处理完成/错误状态

## Phase 6: Markdown 解析扩展

- [x] 6.1 修改 Markdown 代码块解析逻辑，支持 timeout 参数
- [x] 6.2 前端传递 timeout 参数到沙箱 API
- [x] 6.3 沙箱 API 接收并处理 timeout 参数

## Phase 7: 文档编写

- [x] 7.1 创建 `docs/deep-learning/convolutional-neural-network/alexnet-experiment.md`
- [x] 7.2 编写第一阶段：数据准备代码块
- [x] 7.3 编写第二阶段：数据预处理代码块
- [x] 7.4 编写第三阶段：模型定义代码块（使用 extract-class）
- [x] 7.5 编写第四阶段：模型训练代码块（使用 ProgressReporter）
- [x] 7.6 编写第五阶段：模型推理代码块
- [x] 7.7 更新 CLAUDE.md，添加数据目录结构说明
- [x] 7.8 更新 docs/sandbox.md，说明 dmla data 命令使用方法

## Phase 8: 测试验证

- [x] 8.1 测试 dmla data TUI 菜单各项功能（基础命令测试通过）
- [x] 8.2 测试 Tiny ImageNet 下载和解压（ModelScope 镜像下载成功，237MB）
- [x] 8.3 测试 data volume 挂载是否生效（配置读取正常，数据集在 /data 中可用）
- [x] 8.4 测试进度反馈机制（ProgressReporter 导入和执行成功，PYTHONPATH=/workspace 修复生效）
- [x] 8.5 运行 AlexNet 实验文档完整流程（前端构建成功，文档生成正确，代码块执行正常）
- [x] 8.6 修复 AlexNet 模型定义（添加 AdaptiveAvgPool2d 确保输出尺寸固定）
- [x] 8.7 CPU 镜像训练流程测试（3 个 batch 验证通过，Loss 约 3.91，耗时 1.8s）
- [ ] 8.8 Windows 端兼容性测试（无 emoji，待用户验证）