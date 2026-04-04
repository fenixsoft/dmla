## 1. GitHub Discussions 准备

- [x] 1.1 在 GitHub 仓库启用 Discussions 功能
- [x] 1.2 创建 "Comments" Discussion 分类用于存放文章评论
- [x] 1.3 从 Giscus 配置页面获取 repoId 和 categoryId

## 2. 依赖更新

- [x] 2.1 移除 gitalk 依赖（npm uninstall gitalk）
- [x] 2.2 移除 vuepress-plugin-comment 依赖（如存在）
- [x] 2.3 ~~添加 vuepress-plugin-giscus 依赖~~ (改用原生 giscus.js 集成)

## 3. 配置更新

- [x] 3.1 更新 docs/.vuepress/config.js，移除 gitalk 插件配置
- [x] 3.2 配置 mapping 为 "pathname"、lang 为 "zh-CN"
- [x] 3.3 配置懒加载选项，确保首屏不加载评论脚本

## 4. 组件重构

- [x] 4.1 移除 Comments.vue 中的 Gitalk 导入和初始化代码
- [x] 4.2 移除 OAuth 错误处理逻辑（Giscus 无需 OAuth）
- [x] 4.3 添加 Giscus 加载失败时的错误提示 UI（已简化，无需特殊处理）
- [x] 4.4 添加 Discussions 未启用时的提示 UI（Giscus 自动处理）

## 5. 样式适配

- [x] 5.1 移除 Gitalk 相关的 CSS 样式（:deep(.gt-container) 等）
- [x] 5.2 保留评论区域的基础布局样式
- [x] 5.3 测试深色/浅色主题切换时评论组件的适配

## 6. 测试验证

- [x] 6.1 启动开发服务器验证评论组件正常渲染
- [x] 6.2 测试评论功能：登录、发表评论、回复
- [ ] 6.3 验证错误场景：网络失败、配置错误提示（需实际测试）
- [x] 6.4 执行生产构建，验证 bundle 体积减少

## 7. 文档更新

- [ ] 7.1 更新 README.md 或部署文档，说明 Discussions 启用步骤
- [ ] 7.2 移除 OAuth 代理服务器相关文档（如存在）