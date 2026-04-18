# 安装 CLI 功能规范

## ADDED Requirements

### Requirement: 启动服务命令

系统 SHALL 提供 `dmla start` 命令启动沙箱服务。

#### Scenario: 默认端口启动
- **WHEN** 用户执行 `dmla start`
- **THEN** 服务在端口 3001 启动
- **AND** 输出服务地址信息

#### Scenario: 自定义端口启动
- **WHEN** 用户执行 `dmla start --port 8080`
- **THEN** 服务在端口 8080 启动
- **AND** 输出自定义端口信息

#### Scenario: GPU 模式启动
- **WHEN** 用户执行 `dmla start --gpu`
- **THEN** 服务使用 GPU 镜像运行代码
- **AND** 显示 GPU 使用提示

#### Scenario: 服务已运行检测
- **WHEN** 用户在服务已运行时执行 `dmla start`
- **THEN** 系统提示服务已在运行
- **AND** 显示当前运行端口

### Requirement: 停止服务命令

系统 SHALL 提供 `dmla stop` 命令停止运行中的沙箱服务。

#### Scenario: 正常停止
- **WHEN** 用户执行 `dmla stop`
- **THEN** 服务停止运行
- **AND** 输出停止确认信息

#### Scenario: 服务未运行
- **WHEN** 用户在服务未运行时执行 `dmla stop`
- **THEN** 系统提示服务未在运行
- **AND** 命令正常退出

### Requirement: 状态查看命令

系统 SHALL 提供 `dmla status` 命令查看服务状态信息。

#### Scenario: 服务运行状态
- **WHEN** 用户执行 `dmla status` 且服务正在运行
- **THEN** 显示服务运行端口、运行时长
- **AND** 显示镜像版本信息（CPU/GPU）

#### Scenario: GPU 可用性状态
- **WHEN** 用户执行 `dmla status`
- **THEN** 显示 GPU 是否可用
- **AND** 显示检测到的 GPU 设备信息（如有）

#### Scenario: npm 包版本状态
- **WHEN** 用户执行 `dmla status`
- **THEN** 显示当前安装的 @dmla/cli 版本号
- **AND** 显示 npm 包名称和版本

#### Scenario: 服务未运行状态
- **WHEN** 用户执行 `dmla status` 且服务未运行
- **THEN** 显示"服务未运行"提示
- **AND** 显示已安装的镜像信息

### Requirement: 安装镜像命令

系统 SHALL 提供 `dmla install` 命令拉取和安装 Docker 镜像。

#### Scenario: 安装全部镜像
- **WHEN** 用户执行 `dmla install`
- **THEN** 同时拉取 CPU 和 GPU 镜像
- **AND** 显示拉取进度

#### Scenario: 仅安装 CPU 镜像
- **WHEN** 用户执行 `dmla install --cpu`
- **THEN** 仅拉取 CPU 版本镜像
- **AND** GPU 镜像不受影响

#### Scenario: 仅安装 GPU 镜像
- **WHEN** 用户执行 `dmla install --gpu`
- **THEN** 仅拉取 GPU 版本镜像
- **AND** CPU 镜像不受影响

#### Scenario: 指定镜像仓库
- **WHEN** 用户执行 `dmla install --registry acr`
- **THEN** 从阿里云 ACR 拉取镜像
- **AND** 自动执行 tag 重命名为本地名称

### Requirement: 更新命令

系统 SHALL 提供 `dmla update` 命令更新 npm 包和 Docker 镜像。

#### Scenario: npm 包更新
- **WHEN** 用户执行 `dmla update`
- **THEN** 执行 `npm update -g @dmla/cli`
- **AND** 显示更新前后版本对比

#### Scenario: 镜像检查更新
- **WHEN** 用户执行 `dmla update`
- **THEN** 检查远程镜像是否有新版本
- **AND** 显示当前镜像版本和远程最新版本

#### Scenario: 镜像自动更新
- **WHEN** 远程镜像有新版本
- **THEN** 自动拉取最新镜像
- **AND** 显示更新摘要

#### Scenario: 无需更新
- **WHEN** npm 包和镜像均为最新版本
- **THEN** 显示"已是最新版本"提示
- **AND** 命令正常退出

### Requirement: 环境诊断命令

系统 SHALL 提供 `dmla doctor` 命令诊断安装环境。

#### Scenario: Docker 环境检查
- **WHEN** 用户执行 `dmla doctor`
- **THEN** 检查 Docker 是否安装
- **AND** 显示 Docker 版本信息
- **AND** 检查 Docker 版本是否满足最低要求（≥ 20.10）

#### Scenario: 镜像完整性检查
- **WHEN** 用户执行 `dmla doctor`
- **THEN** 检查 dmla-sandbox:cpu 和 :gpu 镜像是否存在
- **AND** 显示镜像大小和创建时间

#### Scenario: GPU 驱动检查
- **WHEN** 用户执行 `dmla doctor`
- **THEN** 检查 NVIDIA GPU 是否可用
- **AND** 显示 GPU 驱动版本（如有）
- **AND** 显示 nvidia-smi 输出摘要

#### Scenario: 端口可用性检查
- **WHEN** 用户执行 `dmla doctor`
- **THEN** 检查端口 3001 是否可用
- **AND** 显示端口状态（可用/已占用）

#### Scenario: 网络连通性检查
- **WHEN** 用户执行 `dmla doctor`
- **THEN** 测试 Docker Hub 和 ACR 的网络连通性
- **AND** 显示网络延迟测试结果

#### Scenario: 问题汇总报告
- **WHEN** 诊断发现任何问题
- **THEN** 显示问题汇总列表
- **AND** 提供修复建议

### Requirement: 命令行帮助

系统 SHALL 提供完整的命令行帮助信息。

#### Scenario: 全局帮助
- **WHEN** 用户执行 `dmla --help`
- **THEN** 显示所有可用命令列表
- **AND** 显示每个命令的简要说明

#### Scenario: 命令帮助
- **WHEN** 用户执行 `dmla start --help`
- **THEN** 显示 start 命令的详细用法
- **AND** 显示支持的选项和参数

#### Scenario: 版本信息
- **WHEN** 用户执行 `dmla --version`
- **THEN** 显示 @dmla/cli 版本号