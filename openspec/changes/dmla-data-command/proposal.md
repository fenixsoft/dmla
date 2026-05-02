## Why

DMLA 项目是一个深度学习教学网站，包含大量需要在沙箱环境中执行的 Python 训练代码。现有系统存在以下问题：

1. **数据持久化缺失**：Docker 容器每次执行完毕后销毁，下载的数据集、训练的模型等数据无法保存，每次执行都需要重新下载，浪费时间和带宽。

2. **训练任务超时限制**：当前 60 秒超时限制无法支持长时间训练任务（如 AlexNet 训练可能需要数十分钟）。

3. **缺乏进度反馈**：长时间任务执行时用户无法看到进度，体验较差。

4. **数据集管理分散**：不同文档中的数据集下载代码各自实现，缺乏统一管理。

为支持 AlexNet PyTorch 训练实验文档，以及后续 GAN、大语言模型等实验文档，需要建立统一的数据持久化和管理机制。

## What Changes

### CLI 新增 `dmla data` 命令

- TUI 菜单式交互，支持挂载路径设置、数据集下载、数据管理等功能
- 支持常用数据集（Tiny ImageNet、CIFAR-10、CIFAR-100、MNIST）的自动下载
- 下载时显示进度条，使用 curl/wget 等工具的原始进度输出
- 不使用 emoji，确保 Windows CMD 环境兼容性

### Docker 镜像新增数据目录

- 创建 `/data` 目录，映射到宿主机用户指定路径
- 规划子目录结构：datasets、models、outputs、cache

### 沙箱执行系统增强

- 支持代码块 `timeout` 参数，单位为秒，支持 `unlimited` 取值
- 实现进度反馈机制，容器内 Python 代码可调用 API 更新进度条
- 前端实时显示训练进度

### 文档新增

- 新增 AlexNet PyTorch 训练实验文档，采用分块接力模式
- 分阶段展示：数据准备、预处理、模型定义、训练、推理

## Capabilities

### New Capabilities

- `data-volume-management`: 数据卷管理能力，CLI 命令支持挂载、下载、清空、删除等操作
- `dataset-auto-download`: 数据集自动下载能力，支持 Tiny ImageNet、CIFAR-10/100、MNIST
- `progress-reporting`: 进度反馈能力，长时间任务可实时显示进度条
- `timeout-extension`: 超时扩展能力，支持自定义超时时长和 unlimited 模式

### Modified Capabilities

- `sandbox-execution`: 修改沙箱执行逻辑，增加 data volume 挂载和进度轮询

## Impact

- 修改 Dockerfile.sandbox：新增 `/data` 目录结构和进度报告模块
- 修改 sandbox.js：新增 data volume 挂载逻辑、进度轮询逻辑
- 修改 packages/cli：新增 data 命令实现
- 新增文档：docs/deep-learning/convolutional-neural-network/alexnet-experiment.md
- 更新 CLAUDE.md：新增数据目录结构说明