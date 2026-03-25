# Design: Knowledge Web Application

## 1. 系统架构

### 1.1 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    Architecture Overview                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   Content Layer                              │ │
│  │   Markdown (VuePress) → Git → GitHub                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 Build & Deploy Pipeline                      │ │
│  │                                                              │ │
│  │   Git Push → GitHub Actions → VuePress Build               │ │
│  │        │                                                     │ │
│  │        ├──────────────────┐                                 │ │
│  │        ▼                  ▼                                 │ │
│  │   GitHub Pages      Tencent CDN Refresh                     │ │
│  │   (static HTML)     (cache invalidation)                    │ │
│  │                                                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Runtime Features                          │ │
│  │                                                              │ │
│  │   ┌─────────────────────┐    ┌─────────────────────────┐   │ │
│  │   │   Internet Mode     │    │      Local Mode         │   │ │
│  │   ├─────────────────────┤    ├─────────────────────────┤   │ │
│  │   │ • Static HTML       │    │ • VuePress Dev Server   │   │ │
│  │   │ • Comments (GitHub  │    │ • Comments (GitHub API  │   │ │
│  │   │   API + ETag cache) │    │   + ETag cache)         │   │ │
│  │   │ • Mermaid diagrams  │    │ • Mermaid diagrams      │   │ │
│  │   │ • Code highlighting │    │ • Code highlighting     │   │ │
│  │   │ • Run button hidden │    │ • GPU Sandbox (Docker)  │   │ │
│  │   └─────────────────────┘    │ • Add articles (files)  │   │ │
│  │                              └─────────────────────────┘   │ │
│  │                                                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 项目结构

```
ideaspaces/
├── docs/                              # Markdown 文章
│   ├── README.md                      # 首页
│   ├── arch/                          # 架构设计文档
│   │   └── design.md                  # 整体设计文档
│   ├── python/                        # 分类目录
│   │   ├── README.md                  # 分类索引
│   │   └── decorators.md              # 文章
│   └── javascript/
│       └── ...
│
├── .vuepress/
│   ├── config.js                      # VuePress 配置
│   ├── theme/                         # 自定义主题
│   ├── components/                    # Vue 组件
│   │   ├── Comments.vue               # 评论组件
│   │   └── RunnableCode.vue           # 沙箱组件
│   ├── plugins/
│   │   ├── comments/                  # GitHub Issues 插件
│   │   └── runnable-code/             # 沙箱插件
│   └── styles/
│       └── index.styl
│
├── local-server/                      # 本地模式后端
│   ├── package.json
│   ├── src/
│   │   ├── index.js                   # Express 服务器
│   │   ├── sandbox.js                 # Docker 集成
│   │   └── routes/
│   │       └── sandbox.js             # API 路由
│   └── Dockerfile.sandbox             # Python 沙箱镜像
│
├── scripts/
│   └── cdn-refresh.js                 # CDN 刷新脚本
│
├── .github/
│   └── workflows/
│       ├── deploy.yml                 # 构建部署
│       └── sync-issues.yml            # 自动创建 Issues
│
└── package.json
```

---

## 2. 文章与 Issue 关联机制

### 2.1 Frontmatter 元数据格式

```yaml
---
title: "Python 装饰器详解"
date: 2024-03-24
tags: [python, advanced]
issue:                        # 可选
  title: "装饰器讨论"          # 自定义 Issue 标题
  number: 42                  # 或已存在的 Issue 编号
---
```

### 2.2 Issue 关联解析流程

```
文章页面加载
    │
    ▼
检查 frontmatter.issue.number
    │
    ├── 存在 → 直接使用该 Issue 编号
    │
    └── 不存在
        │
        ▼
    检查 frontmatter.issue.title
        │
        ├── 存在 → 按标题搜索 Issue
        │       │
        │       ├── 找到 → 使用该 Issue
        │       │
        │       └── 未找到 → 创建新 Issue
        │
        └── 不存在 → 使用文章标题作为 Issue 标题
                    │
                    ▼
                搜索或创建 Issue
```

### 2.3 Issue 命名约定

- Issue 标题格式：`[Comments] {文章标题}`
- Issue 标签：`comments`, `article`
- 示例：文章 "Python 装饰器详解" → Issue "[Comments] Python 装饰器详解"

---

## 3. 评论系统设计

### 3.1 组件架构

```
┌─────────────────────────────────────────────────────────────────┐
│  Comments Component (Vue)                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  <template>                                                     │
│    <div class="comments-section">                              │
│      <!-- 标题区域 -->                                          │
│      <div class="comments-header">                             │
│        <h3>💬 讨论</h3>                                        │
│        <a :href="issueUrl" target="_blank">                    │
│          在 GitHub 上查看                                       │
│        </a>                                                     │
│      </div>                                                     │
│                                                                 │
│      <!-- 登录提示 -->                                          │
│      <div v-if="!isLoggedIn">                                  │
│        <button @click="loginWithGitHub">                       │
│          🔐 使用 GitHub 登录参与讨论                            │
│        </button>                                                │
│      </div>                                                     │
│                                                                 │
│      <!-- 评论表单（已登录） -->                                 │
│      <div v-if="isLoggedIn">                                   │
│        <textarea v-model="newComment" />                       │
│        <button @click="submitComment">发表评论</button>         │
│      </div>                                                     │
│                                                                 │
│      <!-- 评论列表 -->                                          │
│      <div class="comments-list">                               │
│        <CommentItem v-for="comment in comments" />             │
│      </div>                                                     │
│    </div>                                                       │
│  </template>                                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 GitHub API 限流缓解策略

#### 限流说明

| 认证状态 | 请求限制 |
|---------|---------|
| 未认证 | 60 次/小时/IP |
| 已认证 | 5000 次/小时/用户 |

#### 缓解措施

**1. 多层缓存**

```
浏览器内存缓存 (TTL: 30秒)
       │
       ▼
localStorage 缓存 (TTL: 5分钟)
       │
       ▼
GitHub API (带 ETag 支持)
```

**2. ETag 条件请求**

```javascript
// 首次请求
GET /repos/owner/repo/issues/42/comments
Response:
  Status: 200 OK
  ETag: "aabc123..."
  Body: [{...}, {...}]

// 后续请求
GET /repos/owner/repo/issues/42/comments
Headers:
  If-None-Match: "aabc123..."

Response (未修改):
  Status: 304 Not Modified
  → 不计入 API 限制！

Response (已修改):
  Status: 200 OK
  ETag: "new-etag..."
```

**3. 懒加载**

```javascript
// 使用 IntersectionObserver
// 仅在用户滚动到评论区时加载
const observer = new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting) {
    loadComments();
    observer.disconnect();
  }
});
observer.observe(document.querySelector('.comments'));
```

**4. 优雅降级**

当 API 限流时显示：

```
┌─────────────────────────────────────────────────┐
│  💬 讨论区                                       │
│                                                 │
│  ⚠️ 评论暂时无法加载                            │
│                                                 │
│  API 请求次数已达上限，请稍后再试。              │
│  或者 [登录 GitHub] 获取更高的请求配额。         │
│                                                 │
│  [在 GitHub 上查看讨论]                         │
└─────────────────────────────────────────────────┘
```

### 3.3 GitHub OAuth 流程

```
用户点击 "使用 GitHub 登录"
       │
       ▼
重定向到 GitHub 授权页面
https://github.com/login/oauth/authorize
  ?client_id={CLIENT_ID}
  &redirect_uri={REDIRECT_URI}
  &scope=public_repo
  &state={random_state}
       │
       ▼
用户在 GitHub 授权
       │
       ▼
重定向回应用，携带 ?code={CODE}
       │
       ▼
通过代理服务交换 token
(使用 Cloudflare Worker 或 OAuth 代理)
       │
       ▼
存储 token 到 localStorage
       │
       ▼
使用 token 调用 API
```

---

## 4. Python 沙箱设计

### 4.1 Docker 镜像

```dockerfile
# 基于 NVIDIA CUDA 镜像
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# 安装 Python
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 预装常用库
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    requests \
    beautifulsoup4 \
    scikit-learn \
    # GPU 相关库
    torch --index-url https://download.pytorch.org/whl/cu118 \
    tensorflow \
    jax[cuda11_local]

WORKDIR /workspace
```

### 4.2 容器执行配置

```javascript
const container = await docker.createContainer({
  Image: 'ideaspaces-sandbox:gpu',
  Cmd: ['python3', '-c', code],
  HostConfig: {
    // GPU 透传
    DeviceRequests: [{
      Driver: 'nvidia',
      Count: -1,  // 使用所有可用 GPU
      Capabilities: [['gpu']]
    }],
    // 基本限制（信任本地用户，无需严格隔离）
    Memory: 4 * 1024 * 1024 * 1024,  // 4GB
  }
});
```

### 4.3 Markdown 语法扩展

**普通可运行代码块：**

````markdown
```python runnable
import numpy as np
print(np.array([1, 2, 3]))
```
````

**GPU 可运行代码块：**

````markdown
```python runnable gpu
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```
````

### 4.4 渲染效果

```
┌─────────────────────────────────────────────────────────────┐
│  [代码块 + 语法高亮]                                         │
│                                                             │
│  [▶ Run on CPU]  [▶ Run on GPU]                            │
│                                                             │
│  Output:                                                    │
│  CUDA available: True                                       │
│  Device: NVIDIA GeForce RTX 3080                            │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 API 接口

**请求：**
```
POST /api/sandbox/run
Content-Type: application/json

{
  "code": "print('hello')",
  "useGPU": false
}
```

**响应：**
```json
{
  "success": true,
  "output": "hello\n",
  "error": null,
  "executionTime": 0.23
}
```

---

## 5. CI/CD 部署流程

### 5.1 GitHub Actions 工作流

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/.vuepress/dist

      - name: Refresh Tencent CDN
        run: npm run cdn:refresh
        env:
          TENCENT_SECRET_ID: ${{ secrets.TENCENT_SECRET_ID }}
          TENCENT_SECRET_KEY: ${{ secrets.TENCENT_SECRET_KEY }}
          CDN_DOMAIN: ${{ secrets.CDN_DOMAIN }}
```

### 5.2 腾讯云 CDN 刷新

```javascript
// scripts/cdn-refresh.js
const tencentcloud = require("tencentcloud-sdk-nodejs");
const CdnClient = tencentcloud.cdn.v20180606.Client;

async function refreshCDN() {
  const client = new CdnClient({
    credential: {
      secretId: process.env.TENCENT_SECRET_ID,
      secretKey: process.env.TENCENT_SECRET_KEY,
    },
    region: "",
    profile: {
      httpProfile: { endpoint: "cdn.tencentcloudapi.com" }
    }
  });

  const domain = process.env.CDN_DOMAIN;

  // 刷新域名下所有内容
  const result = await client.PurgePathsCache({
    Paths: [`https://${domain}/`],
    FlushType: "flush"
  });

  console.log("CDN refresh task submitted:", result);
}
```

### 5.3 GitHub Secrets 配置

| Secret 名称 | 说明 |
|------------|------|
| `TENCENT_SECRET_ID` | 腾讯云 API 密钥 ID |
| `TENCENT_SECRET_KEY` | 腾讯云 API 密钥 Key |
| `CDN_DOMAIN` | CDN 域名 (如: blog.example.com) |

---

## 6. 运行命令

### package.json Scripts

```json
{
  "scripts": {
    "dev": "vuepress dev docs",
    "build": "vuepress build docs",
    "local": "concurrently \"npm run dev\" \"npm run server\"",
    "server": "cd local-server && npm start",
    "build:sandbox": "docker build -t ideaspaces-sandbox:gpu . -f local-server/Dockerfile.sandbox",
    "cdn:refresh": "node scripts/cdn-refresh.js"
  }
}
```

### 使用方式

```bash
# 互联网开发模式（无沙箱）
npm run dev

# 构建生产版本
npm run build

# 本地模式（带沙箱）
npm run build:sandbox   # 首次需构建 Docker 镜像
npm run local           # 启动 VuePress + API 服务器
```

---

## 7. 技术决策总结

| 模块 | 技术选型 | 理由 |
|------|---------|------|
| 静态站点生成 | VuePress v2 | Vue 生态，Vite 构建，插件丰富 |
| 代码高亮 | Prism.js / Shiki | VuePress 原生支持 |
| 流程图 | Mermaid | VuePress 插件支持，语法简洁 |
| 评论系统 | GitHub Issues | 无需后端，与用户体系集成 |
| 本地沙箱 | Docker + CUDA | GPU 支持，环境隔离 |
| CI/CD | GitHub Actions | 与 GitHub Pages 原生集成 |
| CDN | 腾讯云 CDN | 国内访问优化 |