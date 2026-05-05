---
name: CLI Images 命令重构
description: 将 install 命令移至 install 包独占，CLI 新增 images 命令用于镜像拉取
type: project
---

## 背景

- CLI 的 `install` 命令目前调用 `@icyfenix-dmla/install` 包的完整安装流程
- install 包的安装逻辑（拉取镜像 + 安装 cli 工具）应只在 npm 安装时使用，不对外暴露
- CLI 需要独立的 `images` 命令，仅用于拉取镜像（无安装动作）

## 需求

1. CLI 移除 `install` 命令
2. CLI 新增 `images` 命令 - 仅拉取镜像
3. 拉取失败时：自动重试 2 次 → 询问用户是否继续重试
4. 更新镜像大小提示：CPU 683MB, GPU 7.93GB

## 实现方案

### 方案 A：复用 install 包的 docker 模块

- install 包导出 `pullImages` 函数
- CLI 新增 `images` 命令调用它
- 增强 `pullImages` 的重试逻辑

### 修改清单

1. **packages/install/src/index.js** - 导出 `pullImages`
2. **packages/install/src/modules/docker.js** - 增强重试逻辑（自动2次 + 用户询问），更新镜像大小
3. **packages/install/src/modules/install.js** - 更新镜像大小提示
4. **packages/cli/src/commands/images.js** - 新建 images 命令模块
5. **packages/cli/src/index.js** - 移除 install 命令，新增 images 命令

### 重试逻辑设计

```
拉取镜像 → 自动重试 2 次 → 失败后询问：
  - 继续重试（再次自动2次 + 询问）
  - 跳过此镜像
  - 退出整个流程
```