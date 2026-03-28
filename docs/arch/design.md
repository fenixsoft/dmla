# IdeaSpaces 知识管理平台设计文档

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [文章与 Issue 关联机制](#3-文章与-issue-关联机制)
4. [评论系统设计](#4-评论系统设计)
5. [Python 沙箱设计](#5-python-沙箱设计)
6. [CI/CD 部署流程](#6-cicd-部署流程)
7. [技术选型说明](#7-技术选型说明)

---

## 1. 项目概述

### 1.1 项目背景

IdeaSpaces 是一个交互式知识类 Web 应用，旨在为个人知识管理和公开教学提供一个统一的平台。

### 1.2 核心目标

- **个人知识库**：整理和管理个人知识内容
- **公开教学平台**：将知识内容公开发布，供他人学习
- **双模式部署**：支持互联网部署和本地部署两种模式

### 1.3 部署模式对比

| 特性 | 互联网部署 | 本地部署 |
|------|-----------|---------|
| 基础设施 | GitHub Pages | 本地 Node.js 服务 |
| 后端需求 | 无 | Express API 服务器 |
| 文章管理 | Git 提交 | 本地文件操作 |
| 评论系统 | GitHub Issues | GitHub Issues |
| 代码沙箱 | 不支持 | Docker + GPU 支持 |
| 流程图 | Mermaid | Mermaid |
| 代码高亮 | Prism.js/Shiki | Prism.js/Shiki |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                    Architecture Overview                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   Content Layer (内容层)                     │ │
│  │                                                              │ │
│  │   Markdown 文章 → Git 版本控制 → GitHub 仓库                │ │
│  │                                                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Build & Deploy Layer (构建部署层)               │ │
│  │                                                              │ │
│  │   Git Push → GitHub Actions → VuePress Build               │ │
│  │        │                                                     │ │
│  │        ├──────────────────┐                                 │ │
│  │        ▼                  ▼                                 │ │
│  │   GitHub Pages      腾讯云 CDN 刷新                         │ │
│  │   (静态 HTML)       (缓存刷新)                               │ │
│  │                                                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Runtime Features (运行时功能层)               │ │
│  │                                                              │ │
│  │   ┌─────────────────────┐    ┌─────────────────────────┐   │ │
│  │   │   Internet Mode     │    │      Local Mode         │   │ │
│  │   │   (互联网模式)       │    │      (本地模式)          │   │ │
│  │   ├─────────────────────┤    ├─────────────────────────┤   │ │
│  │   │ • 静态 HTML 展示     │    │ • VuePress 开发服务器    │   │ │
│  │   │ • 评论系统 (GitHub   │    │ • 评论系统 (GitHub API   │   │ │
│  │   │   API + ETag 缓存)  │    │   + ETag 缓存)          │   │ │
│  │   │ • Mermaid 流程图     │    │ • Mermaid 流程图         │   │ │
│  │   │ • 代码语法高亮       │    │ • 代码语法高亮           │   │ │
│  │   │ • 运行按钮隐藏       │    │ • GPU 沙箱 (Docker)      │   │ │
│  │   └─────────────────────┘    │ • 添加文章 (文件操作)    │   │ │
│  │                              └─────────────────────────┘   │ │
│  │                                                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │            API Rate Limit Handling (API 限流处理层)          │ │
│  │                                                              │ │
│  │   多层缓存: 内存 (30秒) → LocalStorage (5分钟) → GitHub API │ │
│  │   ETag 条件请求                                              │ │
│  │   懒加载 (IntersectionObserver)                              │ │
│  │   优雅降级 UI                                                 │ │
│  │                                                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 项目目录结构

```
ideaspaces/
├── docs/                              # Markdown 文章目录
│   ├── README.md                      # 首页
│   ├── arch/                          # 架构设计文档
│   │   └── design.md                  # 本设计文档
│   ├── python/                        # Python 分类
│   │   ├── README.md                  # 分类索引
│   │   └── *.md                       # 文章文件
│   └── javascript/                    # JavaScript 分类
│       └── *.md
│
├── .vuepress/                         # VuePress 配置目录
│   ├── config.js                      # 主配置文件
│   ├── theme/                         # 自定义主题
│   │   └── index.js
│   ├── components/                    # Vue 组件
│   │   ├── Comments.vue               # 评论组件
│   │   └── RunnableCode.vue           # 可运行代码组件
│   ├── plugins/                       # 自定义插件
│   │   ├── comments/                  # GitHub Issues 评论插件
│   │   │   ├── index.js
│   │   │   └── Comments.vue
│   │   └── runnable-code/             # 代码沙箱插件
│   │       ├── index.js
│   │       └── RunnableBlock.vue
│   └── styles/                        # 样式文件
│       └── index.styl
│
├── local-server/                      # 本地模式后端服务
│   ├── package.json
│   ├── src/
│   │   ├── index.js                   # Express 服务器入口
│   │   ├── sandbox.js                 # Docker 沙箱管理
│   │   └── routes/
│   │       └── sandbox.js             # API 路由定义
│   └── Dockerfile.sandbox             # Python 沙箱镜像
│
├── scripts/                           # 构建脚本
│   ├── cdn-refresh.js                 # CDN 刷新脚本
│   └── sync-issues.js                 # Issue 同步脚本
│
├── .github/                           # GitHub 配置
│   └── workflows/
│       ├── deploy.yml                 # 部署工作流
│       └── sync-issues.yml            # Issue 同步工作流
│
├── package.json                       # 项目配置
├── package-lock.json
└── README.md                          # 项目说明
```

---

## 3. 文章与 Issue 关联机制

### 3.1 Frontmatter 元数据格式

每篇文章的 YAML frontmatter 支持以下 Issue 相关字段：

```yaml
---
title: "文章标题"
date: 2024-03-24
tags: [标签1, 标签2]
issue:                        # 可选字段
  title: "自定义 Issue 标题"   # 自定义 Issue 标题
  number: 42                  # 或直接指定 Issue 编号
---
```

### 3.2 Issue 关联解析流程图

```
                    文章页面加载
                         │
                         ▼
            ┌────────────────────────┐
            │ 检查 frontmatter.      │
            │ issue.number           │
            └────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼                           ▼
        存在                         不存在
           │                           │
           ▼                           ▼
    ┌──────────────┐      ┌────────────────────────┐
    │ 直接使用该    │      │ 检查 frontmatter.      │
    │ Issue 编号   │      │ issue.title            │
    └──────────────┘      └────────────────────────┘
                                     │
                       ┌─────────────┴─────────────┐
                       │                           │
                       ▼                           ▼
                    存在                         不存在
                       │                           │
                       ▼                           ▼
          ┌────────────────────┐      ┌────────────────────┐
          │ 按标题搜索 Issue   │      │ 使用文章标题作为    │
          │                    │      │ Issue 标题         │
          └────────────────────┘      └────────────────────┘
                       │                           │
                       ▼                           ▼
          ┌────────────────────┐      ┌────────────────────┐
          │ 找到 → 使用该 Issue│      │ 搜索或创建 Issue   │
          │ 未找到 → 创建新的  │      │                    │
          └────────────────────┘      └────────────────────┘
```

### 3.3 Issue 命名约定

| 项目 | 约定 |
|------|------|
| Issue 标题格式 | `[Comments] {文章标题}` |
| Issue 标签 | `comments`, `article` |
| 示例 | 文章 "Python 装饰器详解" → Issue "[Comments] Python 装饰器详解" |

### 3.4 标题变更处理

当文章标题变更时，为保留已有评论：

1. **推荐方式**：在 frontmatter 中设置 `issue.title` 或 `issue.number`
2. **自动处理**：GitHub Action 可在检测到标题变更时，更新 Issue 标题

---

## 4. 评论系统设计

### 4.1 组件架构图

```
┌─────────────────────────────────────────────────────────────────┐
│  Comments Component (评论组件)                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    标题区域                              │   │
│  │  💬 讨论                        [在 GitHub 上查看]      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  登录状态区域                            │   │
│  │                                                         │   │
│  │  未登录: [🔐 使用 GitHub 登录参与讨论]                   │   │
│  │  已登录: 显示用户头像和名称                              │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  评论输入区域 (已登录)                   │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │                                                 │   │   │
│  │  │  [Markdown 支持的文本输入框]                     │   │   │
│  │  │                                                 │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  │                              [发表评论]                 │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    评论列表区域                          │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  👤 用户名  ·  时间戳                            │   │   │
│  │  │  评论内容...                                     │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  👤 用户名  ·  时间戳                            │   │   │
│  │  │  评论内容...                                     │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 GitHub API 限流说明

| 认证状态 | 请求限制 | 适用场景 |
|---------|---------|---------|
| 未认证 | 60 次/小时/IP | 匿名访问者 |
| 已认证 | 5000 次/小时/用户 | 登录用户 |

### 4.3 多层缓存架构

```
┌─────────────────────────────────────────────────────────────────┐
│  多层缓存架构                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │   浏览器内存     │ ← 第一层缓存                               │
│  │   (Session)      │   TTL: 30 秒                               │
│  │                  │   存储: Map 对象                           │
│  └────────┬────────┘                                            │
│           │ 未命中                                               │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │   localStorage  │ ← 第二层缓存                               │
│  │                  │   TTL: 5 分钟                              │
│  │                  │   存储: 持久化 JSON                        │
│  └────────┬────────┘                                            │
│           │ 未命中                                               │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │   GitHub API    │ ← 数据源                                   │
│  │   (带 ETag)      │   支持 304 Not Modified                    │
│  │                  │                                           │
│  └─────────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 ETag 条件请求流程

```
┌─────────────────────────────────────────────────────────────────┐
│  ETag 条件请求流程                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  首次请求:                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  GET /repos/owner/repo/issues/42/comments               │   │
│  │                                                         │   │
│  │  Response:                                              │   │
│  │    Status: 200 OK                                       │   │
│  │    ETag: "aabc123def456..."                             │   │
│  │    Body: [{comment1}, {comment2}, ...]                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  缓存存储:                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  localStorage.setItem('gh:comments:42', {               │   │
│  │    etag: "aabc123def456...",                            │   │
│  │    data: [{comment1}, {comment2}],                      │   │
│  │    timestamp: Date.now()                                │   │
│  │  });                                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  后续请求:                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  GET /repos/owner/repo/issues/42/comments               │   │
│  │  Headers:                                               │   │
│  │    If-None-Match: "aabc123def456..."                    │   │
│  │                                                         │   │
│  │  Response (未修改):                                      │   │
│  │    Status: 304 Not Modified                             │   │
│  │    → 不计入 API 限制！使用缓存数据                        │   │
│  │                                                         │   │
│  │  Response (已修改):                                      │   │
│  │    Status: 200 OK                                       │   │
│  │    ETag: "new-etag-789..."                              │   │
│  │    → 更新缓存                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 懒加载实现

```javascript
// 使用 IntersectionObserver 实现评论区懒加载
// 仅在用户滚动到评论区时才加载评论数据

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      loadComments();
      observer.unobserve(entry.target);
    }
  });
}, {
  rootMargin: '100px'  // 提前 100px 开始加载
});

observer.observe(document.querySelector('.comments-section'));
```

### 4.6 优雅降级 UI

当 API 限流或加载失败时显示：

```
┌─────────────────────────────────────────────────────────────┐
│  💬 讨论区                                                   │
│                                                             │
│  ⚠️ 评论暂时无法加载                                        │
│                                                             │
│  API 请求次数已达上限，请稍后再试。                          │
│  或者 [登录 GitHub] 获取更高的请求配额。                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  [在 GitHub 上查看讨论]                              │   │
│  │  直接跳转到 GitHub Issue 页面参与讨论                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.7 GitHub OAuth 流程

```
┌─────────────────────────────────────────────────────────────────┐
│  GitHub OAuth 授权流程                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  用户点击 "使用 GitHub 登录"                                     │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  重定向到 GitHub 授权页面                                │   │
│  │                                                         │   │
│  │  https://github.com/login/oauth/authorize               │   │
│  │    ?client_id={CLIENT_ID}                               │   │
│  │    &redirect_uri={REDIRECT_URI}                         │   │
│  │    &scope=public_repo                                   │   │
│  │    &state={random_state}                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  用户在 GitHub 页面授权                                         │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  重定向回应用，携带授权码                                │   │
│  │                                                         │   │
│  │  https://your-site.com/callback?code={CODE}             │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  通过代理服务交换 Access Token                          │   │
│  │                                                         │   │
│  │  方案1: Cloudflare Worker 代理                          │   │
│  │  方案2: 公开客户端 OAuth (无需 secret)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  存储 Token 到 localStorage                                     │
│           │                                                     │
│           ▼                                                     │
│  使用 Token 调用 GitHub API                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Python 沙箱设计

### 5.1 安全模型说明

```
┌─────────────────────────────────────────────────────────────────┐
│  安全模型：信任本地用户                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ❌ 移除的限制:                                                 │
│     • 网络隔离 (NetworkMode: 'none')                           │
│     • 严格的资源限制 (CPU/内存限制放宽)                         │
│     • 只读文件系统                                              │
│     • PID 限制                                                  │
│                                                                 │
│  ✅ 保留的配置:                                                 │
│     • 基础容器隔离 (自动清理)                                   │
│     • 执行超时 (防止无限循环)                                   │
│     • 非 root 用户运行 (可选)                                   │
│                                                                 │
│  原因: 本地部署场景，用户为可信用户（自己）                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Docker 镜像设计

提供两种镜像版本：GPU 版本和 CPU 版本。

**GPU 版本 (Dockerfile.sandbox)**:
```dockerfile
# 基于 NVIDIA CUDA 11.8
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# 安装 Python 3.11 和科学计算库
# 包括: numpy, pandas, matplotlib, scipy, scikit-learn
# 以及 PyTorch GPU 版本

# 构建命令: npm run build:sandbox
```

**CPU 版本 (Dockerfile.sandbox.cpu)**:
```dockerfile
# 基于 Python 3.11 slim 镜像
FROM python:3.11-slim

# 安装科学计算库
# 包括: numpy, pandas, matplotlib, scipy, scikit-learn
# 以及 PyTorch CPU 版本

# 构建命令: npm run build:sandbox:cpu
```

**构建命令**:
```bash
# 构建 GPU 版本
npm run build:sandbox

# 构建 CPU 版本
npm run build:sandbox:cpu

# 构建所有版本
npm run build:sandbox:all
```

**预装库列表**:
- numpy - 数值计算
- pandas - 数据处理
- matplotlib - 数据可视化
- scipy - 科学计算
- scikit-learn - 机器学习
- torch - 深度学习 (PyTorch)
- pillow - 图像处理
- opencv-python-headless - 计算机视觉
- ipykernel - IPython Kernel（支持富输出）
- jupyter_client - Jupyter 客户端（Kernel 通信）

### 5.3 容器执行架构

#### 5.3.1 IPython Kernel 架构

沙箱采用 IPython Kernel 方案执行代码，支持 Jupyter 消息协议的富输出格式。

```
┌─────────────────────────────────────────────────────────────────┐
│  IPython Kernel 执行架构                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  前端 RunnableCode.vue                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  outputs: [                                              │   │
│  │    { type: 'stream', name: 'stdout', text: '...' },     │   │
│  │    { type: 'display_data', data: { 'image/png': '...' } }│   │
│  │    { type: 'error', ename: 'NameError', traceback: [...] }│   │
│  │  ]                                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │ HTTP (JSON)                         │
│                           ▼                                     │
│  后端 Express (Node.js)                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  POST /api/sandbox/run                                  │   │
│  │  → 调用 Docker 容器执行 kernel_runner.py                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │ Docker API                          │
│                           ▼                                     │
│  Docker 容器内                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  kernel_runner.py                                        │   │
│  │  1. 启动 IPython Kernel                                   │   │
│  │  2. 发送 execute_request                                  │   │
│  │  3. 收集 IOPub 消息 (stream, display_data, error)        │   │
│  │  4. 输出 JSON 到 stdout                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │ ZeroMQ (localhost)                  │
│                           ▼                                     │
│  IPython Kernel (独立进程)                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  - matplotlib inline 后端                                 │   │
│  │  - plt.show() → display_data 消息                        │   │
│  │  - 支持所有 Jupyter 富输出                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 5.3.2 执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│  沙箱执行流程                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  用户请求 "运行代码"                                             │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  API Server (Express)                                   │   │
│  │                                                         │   │
│  │  POST /api/sandbox/run                                 │   │
│  │  Body: { code: "...", useGPU: true/false }             │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Sandbox Manager (sandbox.js)                           │   │
│  │                                                         │   │
│  │  1. 检测 GPU 可用性                                      │   │
│  │  2. 选择镜像 (GPU/CPU)                                   │   │
│  │  3. 创建容器                                             │   │
│  │  4. 执行 kernel_runner.py                                │   │
│  │  5. 解析 JSON 输出                                       │   │
│  │  6. 销毁容器                                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Docker Container                                       │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  IPython Kernel + Python 3.11                   │   │   │
│  │  │                                                 │   │   │
│  │  │  预装库:                                        │   │   │
│  │  │  • numpy, pandas, matplotlib                    │   │   │
│  │  │  • scikit-learn, scipy                          │   │   │
│  │  │  • torch, ipykernel, jupyter_client             │   │   │
│  │  │                                                 │   │   │
│  │  │  资源配置:                                       │   │   │
│  │  │  • 内存: 4GB                                    │   │   │
│  │  │  • 超时: 60秒                                   │   │   │
│  │  │  • GPU: 全部可用 GPU (GPU镜像)                  │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Response (结构化输出)                                   │   │
│  │                                                         │   │
│  │  {                                                      │   │
│  │    "success": true,                                     │   │
│  │    "outputs": [                                         │   │
│  │      { "type": "stream", "name": "stdout",              │   │
│  │        "text": "Hello, World!\n" },                     │   │
│  │      { "type": "display_data",                          │   │
│  │        "data": { "image/png": "base64..." } }           │   │
│  │    ],                                                   │   │
│  │    "executionTime": 1.23,                              │   │
│  │    "gpuUsed": true                                      │   │
│  │  }                                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 5.3.3 Kernel 生命周期

| 策略 | 说明 |
|------|------|
| 生命周期 | 每次请求启动新 Kernel，执行完成后立即关闭 |
| 隔离性 | 完全隔离，无状态污染 |
| 超时控制 | 60 秒执行超时，自动终止 Kernel |
| 内存限制 | 容器内存限制 4GB |

### 5.4 Markdown 语法扩展

**普通可运行代码块：**

````markdown
```python runnable
import numpy as np

# 创建数组并计算
arr = np.array([1, 2, 3, 4, 5])
print(f"数组: {arr}")
print(f"平均值: {np.mean(arr)}")
```
````

**GPU 可运行代码块：**

````markdown
```python runnable gpu
import torch

# 检查 GPU 状态
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")

    # GPU 计算
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print(f"计算结果形状: {y.shape}")
```
````

### 5.5 渲染效果示意图

```
┌─────────────────────────────────────────────────────────────────┐
│  Python 代码块                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1 │ import torch                                              │
│  2 │                                                           │
│  3 │ print(f"CUDA: {torch.cuda.is_available()}")               │
│  4 │ print(f"Device: {torch.cuda.get_device_name(0)}")         │
│  5 │                                                           │
│  6 │ x = torch.randn(1000, 1000).cuda()                        │
│  7 │ y = torch.matmul(x, x)                                    │
│  8 │ print(f"Shape: {y.shape}")                                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  [▶ Run on CPU]  [▶ Run on GPU]                                │
├─────────────────────────────────────────────────────────────────┤
│  Output:                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CUDA: True                                             │   │
│  │  Device: NVIDIA GeForce RTX 3080                        │   │
│  │  Shape: torch.Size([1000, 1000])                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.6 API 接口定义

**请求端点**

```
POST /api/sandbox/run
Content-Type: application/json
```

**请求体**

```json
{
  "code": "print('Hello, World!')",
  "useGpu": false
}
```

**成功响应（结构化输出）**

```json
{
  "success": true,
  "outputs": [
    {
      "type": "stream",
      "name": "stdout",
      "text": "Hello, World!\n"
    }
  ],
  "executionTime": 0.156,
  "gpuUsed": false
}
```

**图片输出响应**

```json
{
  "success": true,
  "outputs": [
    {
      "type": "display_data",
      "data": {
        "image/png": "iVBORw0KGgo..."
      },
      "metadata": {
        "image/png": {
          "width": 640,
          "height": 480
        }
      }
    }
  ],
  "executionTime": 1.234,
  "gpuUsed": false
}
```

**错误响应**

```json
{
  "success": true,
  "outputs": [
    {
      "type": "error",
      "ename": "ZeroDivisionError",
      "evalue": "division by zero",
      "traceback": [
        "ZeroDivisionError: division by zero",
        "  File \"<string>\", line 1, in <module>"
      ]
    }
  ],
  "executionTime": 0.023,
  "gpuUsed": false
}
```

### 5.7 输出类型说明

| 类型 | 字段 | 说明 |
|------|------|------|
| stream | name, text | 标准输出流（stdout/stderr） |
| display_data | data, metadata | 富输出（图片、HTML等） |
| execute_result | data, metadata, execution_count | 表达式执行结果 |
| error | ename, evalue, traceback | 错误信息 |

**支持的 MIME 类型**:
- `image/png` - PNG 图片
- `image/jpeg` - JPEG 图片
- `text/plain` - 纯文本
- `text/html` - HTML 内容
- `application/json` - JSON 数据

---

## 6. CI/CD 部署流程

### 6.1 部署流程图

```
┌─────────────────────────────────────────────────────────────────┐
│  CI/CD 部署流程                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  开发者推送代码到 main 分支                                      │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  GitHub Actions 触发                                     │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  Step 1: Checkout 代码                          │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                       │                                 │   │
│  │                       ▼                                 │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  Step 2: 安装 Node.js 依赖                       │   │   │
│  │  │  npm ci                                         │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                       │                                 │   │
│  │                       ▼                                 │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  Step 3: VuePress 构建                          │   │   │
│  │  │  npm run build                                  │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                       │                                 │   │
│  │                       ▼                                 │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  Step 4: 部署到 GitHub Pages                    │   │   │
│  │  │  peaceiris/actions-gh-pages@v3                  │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                       │                                 │   │
│  │                       ▼                                 │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  Step 5: 刷新腾讯云 CDN                         │   │   │
│  │  │  npm run cdn:refresh                            │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  部署完成                                                │   │
│  │                                                         │   │
│  │  • GitHub Pages: https://username.github.io/repo        │   │
│  │  • 自定义域名: https://blog.example.com (CDN 加速)      │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 GitHub Actions 工作流配置

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:  # 支持手动触发

# 设置 GITHUB_TOKEN 的权限
permissions:
  contents: read
  pages: write
  id-token: write

# 只允许一个并发部署
concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 获取完整历史，用于 Git 日志

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build VuePress site
        run: npm run build

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./docs/.vuepress/dist

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4

  cdn-refresh:
    runs-on: ubuntu-latest
    needs: deploy
    if: success()
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies for CDN refresh
        run: npm ci

      - name: Refresh Tencent CDN
        run: npm run cdn:refresh
        env:
          TENCENT_SECRET_ID: ${{ secrets.TENCENT_SECRET_ID }}
          TENCENT_SECRET_KEY: ${{ secrets.TENCENT_SECRET_KEY }}
          CDN_DOMAIN: ${{ secrets.CDN_DOMAIN }}
        continue-on-error: true  # CDN 刷新失败不影响部署状态
```

### 6.3 Issue 同步工作流

```yaml
# .github/workflows/sync-issues.yml
name: Sync Issues for Articles

on:
  push:
    branches: [main]
    paths:
      - 'docs/**/*.md'
  workflow_dispatch:

permissions:
  contents: write
  issues: write

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Sync Issues
        run: npm run sync:issues
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 6.4 腾讯云 CDN 刷新脚本

```javascript
// scripts/cdn-refresh.js
const tencentcloud = require("tencentcloud-sdk-nodejs");
const CdnClient = tencentcloud.cdn.v20180606.Client;

/**
 * 刷新腾讯云 CDN 缓存
 */
async function refreshCDN() {
  // 验证环境变量
  const secretId = process.env.TENCENT_SECRET_ID;
  const secretKey = process.env.TENCENT_SECRET_KEY;
  const cdnDomain = process.env.CDN_DOMAIN;

  if (!secretId || !secretKey || !cdnDomain) {
    console.log("⚠️  腾讯云 CDN 配置不完整，跳过刷新");
    console.log("需要配置以下 Secrets:");
    console.log("  - TENCENT_SECRET_ID");
    console.log("  - TENCENT_SECRET_KEY");
    console.log("  - CDN_DOMAIN");
    return;
  }

  // 创建客户端
  const client = new CdnClient({
    credential: {
      secretId,
      secretKey,
    },
    region: "",
    profile: {
      httpProfile: {
        endpoint: "cdn.tencentcloudapi.com",
      },
    },
  });

  try {
    // 全站刷新
    const result = await client.PurgePathsCache({
      Paths: [`https://${cdnDomain}/`],
      FlushType: "flush",  // flush: 刷新目录; delete: 刷新文件
    });

    console.log("✅ CDN 刷新任务已提交:");
    console.log(`   任务 ID: ${result.TaskId}`);
    console.log(`   刷新路径: https://${cdnDomain}/`);
  } catch (error) {
    console.error("❌ CDN 刷新失败:", error.message);
    throw error;
  }
}

// 增量刷新 (仅刷新变更的文件)
async function refreshChangedFiles(changedFiles) {
  const urls = changedFiles
    .filter((f) => f.endsWith(".md"))
    .map((f) => {
      const urlPath = f
        .replace("docs/", "")
        .replace("README.md", "index.html")
        .replace(".md", ".html");
      return `https://${process.env.CDN_DOMAIN}/${urlPath}`;
    });

  if (urls.length === 0) {
    console.log("无需刷新的文件");
    return;
  }

  const client = new CdnClient({
    credential: {
      secretId: process.env.TENCENT_SECRET_ID,
      secretKey: process.env.TENCENT_SECRET_KEY,
    },
    region: "",
    profile: {
      httpProfile: {
        endpoint: "cdn.tencentcloudapi.com",
      },
    },
  });

  try {
    const result = await client.PurgeUrlsCache({
      Urls: urls,
    });

    console.log("✅ 增量刷新任务已提交:");
    console.log(`   任务 ID: ${result.TaskId}`);
    console.log(`   刷新 URL 数量: ${urls.length}`);
  } catch (error) {
    console.error("❌ 增量刷新失败:", error.message);
    throw error;
  }
}

// 执行刷新
refreshCDN().catch((error) => {
  console.error("CDN 刷新出错:", error);
  process.exit(0);  // 不阻断部署流程
});
```

### 6.5 GitHub Secrets 配置

| Secret 名称 | 获取方式 | 用途 |
|------------|---------|------|
| `TENCENT_SECRET_ID` | 腾讯云控制台 → 访问管理 → API 密钥 | 腾讯云 API 认证 |
| `TENCENT_SECRET_KEY` | 同上 | 腾讯云 API 认证 |
| `CDN_DOMAIN` | 自定义域名 (如: `blog.example.com`) | CDN 刷新目标域名 |

---

## 7. 技术选型说明

### 7.1 技术栈总览

| 层级 | 技术选型 | 版本要求 |
|------|---------|---------|
| 静态站点生成 | VuePress | v2.x (Vue 3) |
| 前端框架 | Vue | 3.x |
| 构建工具 | Vite | 5.x |
| 代码高亮 | Shiki | 内置于 VuePress |
| 流程图 | Mermaid | 通过插件支持 |
| 本地后端 | Express | 4.x |
| 沙箱运行时 | Docker | 20.x+ |
| Python 版本 | Python | 3.11 |
| CUDA 版本 | NVIDIA CUDA | 11.8 |
| CI/CD | GitHub Actions | - |

### 7.2 选型理由

#### VuePress v2

- **Vue 3 生态**: 组件化开发，TypeScript 支持完善
- **Vite 构建**: 开发启动快，热更新迅速
- **插件系统**: 易于扩展评论系统和代码沙箱
- **Markdown 增强**: 内置 frontmatter、代码高亮支持

#### Shiki (vs Prism.js)

- **精准高亮**: 使用 VS Code 同款语法引擎
- **主题一致**: 与编辑器主题保持一致
- **性能优异**: 编译时高亮，无运行时开销

#### GitHub Issues 作为评论系统

- **无后端**: 符合互联网部署无后端的要求
- **用户体系**: 借助 GitHub OAuth，无需自建用户系统
- **数据归属**: 评论数据存储在用户自己的仓库

#### Docker + CUDA

- **环境一致**: 开发和生产环境统一
- **GPU 支持**: NVIDIA 官方镜像，GPU 透传简单
- **隔离性**: 容器自动清理，无状态管理

### 7.3 开发命令一览

| 命令 | 说明 | 模式 |
|------|------|------|
| `npm run dev` | 启动 VuePress 开发服务器 | 互联网/本地 |
| `npm run build` | 构建生产版本 | 互联网/本地 |
| `npm run local` | 启动本地完整服务 (VuePress + API) | 本地 |
| `npm run build:sandbox` | 构建 Docker 沙箱镜像 | 本地 (首次) |
| `npm run cdn:refresh` | 刷新腾讯云 CDN | CI/CD |

### 7.4 浏览器兼容性

| 浏览器 | 最低版本 |
|--------|---------|
| Chrome | 90+ |
| Firefox | 88+ |
| Safari | 14+ |
| Edge | 90+ |

---

## 附录 A: 快速开始

### 互联网部署

```bash
# 1. 克隆仓库
git clone https://github.com/username/ideaspaces.git
cd ideaspaces

# 2. 安装依赖
npm install

# 3. 本地预览
npm run dev

# 4. 推送到 GitHub 触发自动部署
git push origin main
```

### 本地部署 (带沙箱)

```bash
# 1. 安装依赖
npm install

# 2. 构建 Docker 沙箱镜像 (首次或更新依赖后)
npm run build:sandbox

# 3. 启动本地服务
npm run local

# 4. 访问
# - 网站页面: http://localhost:8080
# - API 服务: http://localhost:3001
```

---

## 附录 B: 文章模板

```markdown
---
title: "文章标题"
date: 2024-03-24
tags: [标签1, 标签2]
issue:
  title: "自定义评论标题"
---

# 文章标题

文章简介...

## 章节一

正文内容...

### 代码示例

\`\`\`python runnable
print("Hello, IdeaSpaces!")
\`\`\`

### 流程图

\`\`\`mermaid
graph TD
    A[开始] --> B[处理]
    B --> C[结束]
\`\`\`

## 总结

...
```

---

## 变更历史

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| 1.0 | 2026-03-25 | 初始设计文档 |