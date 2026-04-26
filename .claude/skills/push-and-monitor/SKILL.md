---
name: push-and-monitor
description: Use when submitting changes to main branch and need to monitor GitHub Actions deployment (deploy.yml), Docker image publishing (publish-docker.yml triggered by auto-tag.yml), and npm package publishing (publish-npm.yml triggered by auto-tag-npm.yml) workflows
---

# 推送与工作流监控

## 概述

提交代码修改到 main 分支后，自动监控三个层面的 CI/CD 流程：VuePress 站点部署、Docker 镜像发布、npm 包发布。在工作流各阶段输出状态提示，遇到失败时自动诊断并尝试修复。

## 工作流触发条件

| 工作流 | 触发条件 | 监控要点 |
|--------|----------|----------|
| deploy.yml | push 到 main | 站点构建 → GitHub Pages 部署 → CDN 刷新 |
| auto-tag.yml | local-server Dockerfile/kernel_runner 变化 | 自动创建时间戳 Tag |
| auto-tag-npm.yml | packages/cli 或 packages/install 变化 | 自动创建 npm- 前缀 Tag |
| publish-docker.yml | 非 npm- 前缀 Tag 推送 | 构建 → 测试 → 推送到 Docker Hub 和 ACR |
| publish-npm.yml | npm- 前缀 Tag 推送 | 构建 → 测试 → 发布到 npm registry |

## 执行流程

```dot
digraph monitor_flow {
    rankdir=TB;
    node [shape=box];

    "开始: git commit + push" [shape=ellipse];

    subgraph cluster_deploy {
        label="deploy.yml 监控";
        style=dashed;
        "检查 deploy 工作流状态" -> "等待完成或失败";
    }

    subgraph cluster_tags {
        label="Tag 自动生成检查";
        style=dashed;
        "检查是否有 Docker 相关变更";
        "检查是否有 npm 包相关变更";
        "等待 auto-tag 工作流完成";
    }

    subgraph cluster_docker {
        label="Docker 发布监控";
        style=dashed;
        "检查新 Tag 是否触发 publish-docker";
        "监控 Docker 构建和推送";
    }

    subgraph cluster_npm {
        label="npm 发布监控";
        style=dashed;
        "检查 npm- Tag 是否触发 publish-npm";
        "监控 npm 构建和发布";
    }

    "开始: git commit + push" -> "检查 deploy 工作流状态";

    "等待完成或失败" -> "检查是否有 Docker 相关变更" [label="deploy 成功"];
    "等待完成或失败" -> "诊断 deploy 失败原因" [label="deploy 失败"];

    "检查是否有 Docker 相关变更" -> "等待 auto-tag 工作流完成" [label="有变更"];
    "检查是否有 Docker 相关变更" -> "检查是否有 npm 包相关变更" [label="无变更"];

    "等待 auto-tag 工作流完成" -> "检查新 Tag 是否触发 publish-docker";

    "检查新 Tag 是否触发 publish-docker" -> "监控 Docker 构建和推送" [label="已触发"];
    "监控 Docker 构建和推送" -> "检查是否有 npm 包相关变更" [label="完成"];

    "检查是否有 npm 包相关变更" -> "等待 auto-tag 工作流完成" [label="有变更" style=dashed];
    "检查是否有 npm 包相关变更" -> "结束: 输出完整报告" [label="无变更"];

    subgraph 的等待 auto-tag 工作流完成 -> "检查 npm- Tag 是否触发 publish-npm";

    "检查 npm- Tag 是否触发 publish-npm" -> "监控 npm 构建和发布" [label="已触发"];
    "监控 npm 构建和发布" -> "结束: 输出完整报告";
}
```

## 监控命令

### 查看工作流运行状态

```bash
# 列出最近的工作流运行
gh run list --limit 10

# 查看特定工作流的运行状态
gh run list --workflow=deploy.yml --limit 3
gh run list --workflow=auto-tag.yml --limit 3
gh run list --workflow=auto-tag-npm.yml --limit 3
gh run list --workflow=publish-docker.yml --limit 3
gh run list --workflow=publish-npm.yml --limit 3

# 实时监控特定运行
gh run watch <run-id>

# 获取失败运行的详细信息
gh run view <run-id> --log-failed
```

### 检查 Tag 触发

```bash
# 查看最新的 Tag
gh release list --limit 5
git tag --sort=-created:refname | head -5

# 检查特定 Tag 的工作流触发
gh run list --workflow=publish-docker.yml --commit=<tag-name>
gh run list --workflow=publish-npm.yml --commit=<tag-name>
```

## 阶段提示信息模板

### 推送阶段
```
📤 正在提交并推送修改...
✅ 推送完成，开始监控工作流
```

### Deploy 工作流
```
🚀 deploy.yml 工作流已触发 (run-id: XXX)
⏳ 正在构建 VuePress 站点...
✅ 构建完成，部署到 GitHub Pages
✅ CDN 刷新完成
🌐 站点已更新: https://icyfenix.github.io/dmla/
```

### Docker 发布
```
🐳 auto-tag.yml 已触发（检测到 Dockerfile 变更）
📦 新 Tag 已创建: YYYY.M.D-HHMM
🚢 publish-docker.yml 已触发 (run-id: XXX)
✅ Docker 镜像已推送到 Docker Hub: icyfenix/dmla-sandbox:gpu
✅ Docker 镜像已推送到阿里云 ACR（国内加速）
```

### npm 发布
```
📦 auto-tag-npm.yml 已触发（检测到 packages 变更）
🏷️ 新 npm Tag 已创建: npm-YYYY.M.D-HHMM
🚀 publish-npm.yml 已触发 (run-id: XXX)
✅ npm 包已发布: @icyfenix-dmla/cli@VERSION
✅ npm 包已发布: @icyfenix-dmla/install@VERSION
```

## 失败诊断与修复

### Deploy 失败常见原因

| 错误类型 | 诊断命令 | 修复方法 |
|----------|----------|----------|
| 构建失败 | `gh run view <run-id> --log-failed` | 检查 VuePress 构建错误，修复代码 |
| Pages 部署失败 | `gh run view <run-id> --log-failed` | 检查 Pages 配置 |
| CDN 刷新失败 | 查看日志中的 TENCENT_SECRET | 检查 Secrets 配置，此步骤已设置 continue-on-error |

### Docker 发布失败常见原因

| 错误类型 | 诊断命令 | 修复方法 |
|----------|----------|----------|
| 构建失败 | `gh run view <run-id> --log-failed` | 检查 Dockerfile 语法、依赖问题 |
| 测试失败 | 查看日志中的 `python3 -c` 输出 | 检查镜像内的 Python 环境 |
| 推送失败 | 查看日志中的 login 错误 | 检查 DOCKER_USERNAME/DOCKER_PASSWORD Secrets |

### npm 发布失败常见原因

| 错误类型 | 诊断命令 | 修复方法 |
|----------|----------|----------|
| 测试失败 | `gh run view <run-id> --log-failed` | 修复测试代码 |
| 构建失败 | 查看日志中的 npm run build 输出 | 修复构建配置 |
| 发布失败 | 查看日志中的 NODE_AUTH_TOKEN 错误 | 检查 NPM_TOKEN Secret 配置 |
| 版本冲突 | 查看 npm registry 错误 | 使用新时间戳重新触发 |

## 自动修复策略

1. **构建失败**: 在本地重现问题，修复代码后重新提交
2. **Secrets 问题**: 提示用户检查 GitHub Repository Settings → Secrets
3. **工作流配置问题**: 提示用户检查 workflow 文件语法
4. **并发冲突**: 工作流已设置并发控制，等待当前运行完成

## 完整报告示例

```
📊 推送与发布报告

├─ Git 推送
│  └─ ✅ 提交已推送至 main 分支

├─ Deploy 工作流
│  ├─ ✅ 构建完成
│  ├─ ✅ GitHub Pages 部署完成
│  └─ ✅ CDN 刷新完成
│  └─ 🌐 https://icyfenix.github.io/dmla/

├─ Docker 发布
│  ├─ ✅ auto-tag.yml 完成 (Tag: 2026.4.26-1430)
│  ├─ ✅ publish-docker.yml 完成
│  ├─ 🐳 icyfenix/dmla-sandbox:gpu
│  └─ 🐳 icyfenix/dmla-sandbox:cpu

├─ npm 发布
│  ├─ ⏭️ 无 packages 变更，未触发

└─ 🎉 所有工作流执行完成
```

## 注意事项

- deploy.yml 与 auto-tag.yml 会并行执行，需同时监控
- publish-docker.yml 和 publish-npm.yml 等待对应的 auto-tag 完成后才会触发
- 工作流设置了并发控制，短时间内多次推送可能取消之前的运行
- npm 发布验证需要等待 registry 同步（约 60 秒）