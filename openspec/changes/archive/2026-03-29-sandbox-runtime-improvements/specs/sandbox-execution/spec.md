# 沙箱执行功能规范（修改版）

## MODIFIED Requirements

### Requirement: 支持 CPU 和 GPU 两种执行环境

系统 SHALL 根据运行环境自动选择 CPU 或 GPU 镜像执行代码，当 GPU 不可用时自动回退到 CPU 镜像。

#### Scenario: GPU 环境执行代码
- **WHEN** GPU 可用且用户选择 GPU 执行
- **THEN** 系统使用 `ideaspaces-sandbox:gpu` 镜像执行代码
- **AND** 返回结果中 `gpuUsed` 为 `true`

#### Scenario: CPU 环境执行代码
- **WHEN** GPU 不可用或用户选择 CPU 执行
- **THEN** 系统使用 `ideaspaces-sandbox:latest` 镜像执行代码
- **AND** 返回结果中 `gpuUsed` 为 `false`

#### Scenario: 镜像不存在时的错误提示
- **WHEN** 请求的镜像（CPU 或 GPU）不存在
- **THEN** 系统返回 503 错误
- **AND** 错误信息包含构建命令提示

### Requirement: 正确检测 GPU 可用性

系统 SHALL 通过执行 `nvidia-smi` 命令检测 GPU 是否可用，返回布尔值结果。

#### Scenario: GPU 可用
- **WHEN** 系统有 NVIDIA GPU 且驱动正常
- **THEN** `checkGPUAvailable()` 返回 `true`

#### Scenario: GPU 不可用
- **WHEN** 系统无 NVIDIA GPU 或驱动异常
- **THEN** `checkGPUAvailable()` 返回 `false`

#### Scenario: Docker 守护进程异常
- **WHEN** Docker 守护进程未运行
- **THEN** `checkGPUAvailable()` 捕获异常并返回 `false`

## ADDED Requirements

### Requirement: 使用用户配置的沙箱地址

系统 SHALL 从 localStorage 读取用户配置的沙箱地址，而非使用硬编码的默认地址。

#### Scenario: 使用自定义沙箱地址
- **WHEN** 用户配置了自定义沙箱地址（如 `http://192.168.1.100:3001`）
- **THEN** 代码执行请求发送到自定义地址

#### Scenario: 使用默认沙箱地址
- **WHEN** 用户未配置沙箱地址
- **THEN** 代码执行请求发送到默认地址 `http://localhost:3001`

### Requirement: 简化的 Docker 镜像依赖

Docker 镜像 SHALL 仅包含课程必需的 Python 库：NumPy、Pandas、Matplotlib、SciPy、Scikit-learn、PyTorch。

#### Scenario: GPU 镜像包含 PyTorch CUDA 版本
- **WHEN** 构建 GPU 版本镜像
- **THEN** 镜像包含 PyTorch with CUDA 11.8 支持

#### Scenario: CPU 镜像使用标准 PyTorch
- **WHEN** 构建 CPU 版本镜像
- **THEN** 镜像包含标准 PyTorch（无 CUDA 依赖）

#### Scenario: 镜像不包含不需要的库
- **WHEN** 检查镜像内容
- **THEN** 镜像不包含 TensorFlow 和 JAX