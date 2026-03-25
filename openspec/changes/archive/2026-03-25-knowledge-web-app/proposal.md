# Proposal: Knowledge Web Application

## 概述

构建一个交互式知识类 Web 应用，以 Markdown 文章为核心内容，支持互联网和本地两种部署模式。

## 目标

- 创建个人知识库，同时作为公开教学平台
- 支持互联网部署（GitHub Pages）和本地部署（Node.js 服务）
- 提供文章评论功能（基于 GitHub Issues）
- 本地部署支持代码沙箱执行（Python + GPU）

## 核心功能

### 互联网部署模式

- 文章编译为静态 HTML（VuePress 框架）
- 通过 GitHub Actions 发布到 GitHub Pages
- 支持嵌入 Mermaid 流程图
- 支持代码片段高亮展示
- 评论系统：GitHub 用户认证 + Issues 存储
- 无需后端服务

### 本地部署模式

- 包含互联网部署全部功能
- Python 代码沙箱执行（Docker + GPU 支持）
- 支持添加用户自己的文章（纯文件操作）
- VuePress 开发服务器 + API 服务

## 非目标

- 不支持 Obsidian 格式（与 VuePress 格式存在冲突）
- 不实现用户系统（单用户场景）
- 本地部署不实现 Web Admin UI（纯文件操作）
- 沙箱不支持动态安装包（仅预安装库）

## 技术栈

- **前端**: VuePress v2 + Vue 3 + TypeScript
- **代码高亮**: Prism.js 或 Shiki
- **流程图**: Mermaid
- **本地后端**: Node.js + Express
- **沙箱**: Docker + NVIDIA CUDA
- **部署**: GitHub Actions + GitHub Pages + 腾讯云 CDN

## 成功指标

- 互联网部署：完全静态，无后端依赖
- 本地部署：一键启动（npm run local）
- 评论加载：通过多层缓存和 ETag 优化 API 调用
- 沙箱执行：支持 GPU 加速的 Python 代码运行

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| GitHub API 限流 | 多层缓存 + ETag + 懒加载 + 优雅降级 |
| OAuth 无后端 | 使用公开客户端 OAuth 或 Cloudflare Worker 代理 |
| CDN 缓存刷新 | GitHub Actions 自动调用腾讯云 CDN API |

## 范围

本项目为一个完整的知识管理 + 教学平台，包含：

- 文章管理（Markdown + VuePress）
- 评论系统（GitHub Issues）
- 代码执行沙箱（本地部署）
- CI/CD 部署流程