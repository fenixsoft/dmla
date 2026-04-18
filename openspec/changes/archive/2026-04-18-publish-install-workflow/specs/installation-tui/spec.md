# TUI 安装功能规范

## ADDED Requirements

### Requirement: 安装脚本启动器

系统 SHALL 提供 install.sh 脚本作为安装入口，托管于 VuePress public 目录。

#### Scenario: curl 一键安装
- **WHEN** 用户执行 `curl -fsSL https://ai.icyfenix.cn/install.sh | sh`
- **THEN** 脚本开始执行安装流程
- **AND** 显示欢迎信息

#### Scenario: 环境检查
- **WHEN** install.sh 开始执行
- **THEN** 检查 Docker 是否已安装
- **AND** 检查 Node.js 是否已安装
- **AND** 显示环境检测结果

#### Scenario: Node.js 未安装提示
- **WHEN** 检测到 Node.js 未安装
- **THEN** 提示用户需要安装 Node.js
- **AND** 提供 Node.js 安装建议
- **AND** 询问是否继续（如果 Docker 已安装）

#### Scenario: npx 调用 TUI
- **WHEN** 环境检查通过
- **THEN** 执行 `npx @dmla/install` 启动 TUI
- **AND** 显示 TUI 安装向导界面

### Requirement: 镜像仓库选择

系统 SHALL 在 TUI 中提供镜像仓库选择界面。

#### Scenario: 仓库选项显示
- **WHEN** TUI 显示镜像仓库选择步骤
- **THEN** 提供选项：Docker Hub、Tencent TCR、自动选择
- **AND** 每个选项附带说明

#### Scenario: Docker Hub 选择
- **WHEN** 用户选择 Docker Hub
- **THEN** 使用 `icyfenix/dmla-sandbox` 作为镜像源
- **AND** 拉取时不需要登录认证

#### Scenario: TCR 选择
- **WHEN** 用户选择 Tencent TCR
- **THEN** 使用 `ccr.ccs.tencentyun.com/icyfenix/dmla-sandbox` 作为镜像源
- **AND** 拉取时不需要登录认证（公开镜像）

#### Scenario: 自动选择
- **WHEN** 用户选择自动选择
- **THEN** 测试两个仓库的网络延迟
- **AND** 自动选择延迟更低的仓库

### Requirement: 镜像类型选择

系统 SHALL 在 TUI 中提供镜像类型选择界面。

#### Scenario: 类型选项显示
- **WHEN** TUI 显示镜像类型选择步骤
- **THEN** 提供选项：全部安装、仅 CPU、仅 GPU
- **AND** 显示每个选项的说明和镜像大小预估

#### Scenario: GPU 检测推荐
- **WHEN** 系统检测到 NVIDIA GPU
- **THEN** 推荐安装 GPU 版本
- **AND** 显示检测到的 GPU 设备名称

#### Scenario: 全部安装
- **WHEN** 用户选择全部安装
- **THEN** 拉取 CPU 和 GPU 两个镜像
- **AND** 显示总下载大小预估

### Requirement: 端口配置

系统 SHALL 在 TUI 中提供端口配置界面。

#### Scenario: 默认端口显示
- **WHEN** TUI 显示端口配置步骤
- **THEN** 默认显示端口 3001
- **AND** 提示用户可输入自定义端口

#### Scenario: 端口可用性检测
- **WHEN** 用户输入端口
- **THEN** 检测端口是否已被占用
- **AND** 显示检测结果

#### Scenario: 端口被占用提示
- **WHEN** 用户输入的端口已被占用
- **THEN** 显示警告信息
- **AND** 建议使用其他端口或显示占用进程

### Requirement: 镜像拉取进度显示

系统 SHALL 在镜像拉取时显示实时进度。

#### Scenario: 分层进度显示
- **WHEN** 镜像拉取进行中
- **THEN** 显示每层的下载进度
- **AND** 显示已下载大小和总大小

#### Scenario: 总体进度显示
- **WHEN** 拉取多个镜像
- **THEN** 显示当前镜像和总体进度
- **AND** 显示预计剩余时间

#### Scenario: Docker pull 输出解析
- **WHEN** 执行 docker pull 命令
- **THEN** 解析 JSON 流输出
- **AND** 转换为用户友好的进度条显示

#### Scenario: 拉取失败处理
- **WHEN** 镜像拉取失败
- **THEN** 显示错误信息
- **AND** 提供重试选项

### Requirement: 镜像名称映射

系统 SHALL 在拉取后将远程镜像 tag 为本地名称。

#### Scenario: Docker Hub 镜像映射
- **WHEN** 从 Docker Hub 拉取完成
- **THEN** 执行 `docker tag icyfenix/dmla-sandbox:cpu dmla-sandbox:cpu`
- **AND** 执行 `docker tag icyfenix/dmla-sandbox:gpu dmla-sandbox:gpu`

#### Scenario: TCR 镜像映射
- **WHEN** 从 TCR 拉取完成
- **THEN** 执行 `docker tag ccr.ccs.tencentyun.com/.../dmla-sandbox:cpu dmla-sandbox:cpu`
- **AND** 执行 `docker tag ccr.ccs.tencentyun.com/.../dmla-sandbox:gpu dmla-sandbox:gpu`

#### Scenario: 映射确认
- **WHEN** tag 操作完成
- **THEN** 验证本地镜像存在
- **AND** 显示映射成功信息

### Requirement: npm 包安装

系统 SHALL 在镜像安装完成后安装 npm CLI 包。

#### Scenario: npm 全局安装
- **WHEN** 镜像安装完成
- **THEN** 执行 `npm install -g @dmla/cli`
- **AND** 显示安装进度

#### Scenario: 安装成功验证
- **WHEN** npm 安装完成
- **THEN** 执行 `dmla --version` 验证命令可用
- **AND** 显示安装的版本号

### Requirement: 安装完成验证

系统 SHALL 在安装完成后执行验证测试。

#### Scenario: 服务启动测试
- **WHEN** 安装完成
- **THEN** 提示用户是否立即启动服务
- **AND** 用户选择启动后执行 `dmla start`

#### Scenario: 健康检查
- **WHEN** 服务启动
- **THEN** 调用 `/api/health` 接口验证
- **AND** 显示服务状态

#### Scenario: 安装成功报告
- **WHEN** 所有验证通过
- **THEN** 显示安装成功摘要
- **AND** 显示使用提示和常用命令列表

### Requirement: npx 直接安装

系统 SHALL 支持用户通过 npx 直接运行安装程序。

#### Scenario: npx 启动安装
- **WHEN** 用户执行 `npx @dmla/install`
- **THEN** 显示完整的 TUI 安装向导
- **AND** 执行所有安装步骤

#### Scenario: npx 无需预安装
- **WHEN** 用户执行 npx 命令
- **THEN** 自动下载 @dmla/install 包
- **AND** 无需用户预先安装该包

### Requirement: 安装中断恢复

系统 SHALL 支持安装中断后恢复。

#### Scenario: 镜像拉取中断
- **WHEN** 镜像拉取因网络中断
- **THEN** Docker 自动支持断点续传
- **AND** 提示用户重新执行安装命令

#### Scenario: 已安装部分跳过
- **WHEN** 用户重新执行安装
- **THEN** 检测已存在的镜像和 npm 包
- **AND** 跳过已安装的部分