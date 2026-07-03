# Task 2 报告: 创建 FC 专用 Dockerfile

## Status: DONE

## Commits
- `df0195a` — 创建 FC 函数计算专用 Dockerfile (dmla-sandbox:fc)

## 创建的文件
- `local-server/Dockerfile.sandbox.fc` — FC 函数计算精简 Docker 镜像定义

## 变更概要
- 基于 `Dockerfile.sandbox.cpu`，移除 pandas 和 lmdb pip 包
- 新增 `fc_handler.py` 复制指令
- 新增 `FC_SERVER_PORT=9000` 环境变量
- 新增 HEALTHCHECK（30s 间隔，9000 端口）
- CMD 改为 `["python3", "/workspace/fc_handler.py"]`
- 保留其余所有配置（PyTorch CPU、HuggingFace 生态、matplotlib 中文字体、共享模块等）

## 验证结果
- Docker 语法检查通过: `docker build --check` 无警告
- 依赖文件 fc_handler.py 已存在（来自 Task 1）

## 后续步骤
- 任务 3 中需要对此镜像进行构建验证
