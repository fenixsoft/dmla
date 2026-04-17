# 沙箱设置功能规范（变更）

## MODIFIED Requirements

### Requirement: 用户可以配置沙箱服务地址

系统 SHALL 允许用户通过设置界面配置沙箱服务的 URL 地址，默认值为 `http://localhost:3001`。同时支持配置镜像仓库来源。

#### Scenario: 查看当前沙箱地址
- **WHEN** 用户打开设置界面
- **THEN** 系统显示当前配置的沙箱地址
- **AND** 显示当前配置的镜像仓库来源

#### Scenario: 修改沙箱地址
- **WHEN** 用户输入新的沙箱地址并点击保存
- **THEN** 系统将地址保存到 localStorage
- **AND** 后续代码执行请求使用新地址

#### Scenario: 恢复默认地址
- **WHEN** 用户清空沙箱地址输入框并保存
- **THEN** 系统恢复使用默认地址 `http://localhost:3001`

#### Scenario: 配置镜像仓库来源
- **WHEN** 用户在设置界面选择镜像仓库
- **THEN** 系统保存镜像仓库选择（dockerhub/tcr）
- **AND** 后续镜像拉取使用选择的仓库

## ADDED Requirements

### Requirement: 镜像仓库来源配置

系统 SHALL 允许用户配置镜像拉取来源（Docker Hub 或腾讯云 TCR）。

#### Scenario: 默认仓库设置
- **WHEN** 用户首次安装或未配置仓库
- **THEN** 默认使用 Docker Hub
- **AND** 镜像地址为 `icyfenix/dmla-sandbox`

#### Scenario: 选择 TCR 仓库
- **WHEN** 用户选择腾讯云 TCR
- **THEN** 镜像地址为 `ccr.ccs.tencentyun.com/icyfenix/dmla-sandbox`
- **AND** 国内用户获得加速访问

#### Scenario: 仓库配置持久化
- **WHEN** 用户配置镜像仓库
- **THEN** 配置保存到 localStorage
- **AND** dmla install 和 update 命令使用配置的仓库

### Requirement: 镜像版本信息显示

系统 SHALL 在状态查看中显示已安装的镜像版本信息。

#### Scenario: 显示镜像版本
- **WHEN** 用户执行 `dmla status`
- **THEN** 显示已安装镜像的版本 tag
- **AND** 显示镜像创建时间

#### Scenario: 显示镜像来源
- **WHEN** 用户执行 `dmla status`
- **THEN** 显示镜像拉取来源（Docker Hub / TCR）
- **AND** 显示远程最新可用版本

### Requirement: 镜像本地名称统一

系统 SHALL 确保远程拉取的镜像在本地使用统一名称。

#### Scenario: 本地镜像名称
- **WHEN** 从任何仓库拉取镜像
- **THEN** 本地镜像名称统一为 `dmla-sandbox:cpu` 和 `dmla-sandbox:gpu`
- **AND** 沙箱执行使用统一的本地镜像名

#### Scenario: 镜像名称与现有代码兼容
- **WHEN** 沙箱执行代码
- **THEN** sandbox.js 使用 `dmla-sandbox:cpu/gpu` 作为镜像名
- **AND** 无需修改现有执行逻辑