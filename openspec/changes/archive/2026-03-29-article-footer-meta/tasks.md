## 1. 依赖与环境准备

- [x] 1.1 添加 `github-buttons` npm 包依赖到 package.json
- [x] 1.2 创建 `docs/.vuepress/plugins/word-count/` 插件目录结构

## 2. 字数统计插件实现

- [x] 2.1 实现字数计算核心函数 `countWords()`，支持中英文混合统计
- [x] 2.2 实现 Markdown 内容预处理函数，移除代码块、LaTeX 公式等语法标记
- [x] 2.3 实现 VuePress 插件 `extendsPageData` 钩子，将字数存入 `$page.wordCount`
- [x] 2.4 在 `docs/.vuepress/config.js` 中注册 word-count 插件

## 3. GitHub Star 组件实现

- [x] 3.1 创建 `docs/.vuepress/components/GithubButton.vue` 组件
- [x] 3.2 实现 github-buttons 动态加载逻辑（mounted 钩子 import）
- [x] 3.3 配置仓库地址为 `https://github.com/fenixsoft/ideaspaces`
- [x] 3.4 实现加载失败的 fallback 为静态链接

## 4. 文章底部组件实现

- [x] 4.1 创建 `docs/.vuepress/theme/components/ArticleFooter.vue` 组件
- [x] 4.2 实现左侧元信息区域（字数统计 + 更新时间）
- [x] 4.3 实现右侧 GitHub Star 区域（Star 按钮 + Star 数徽章）
- [x] 4.4 添加 lucide 图标（file-text、calendar、star）的 SVG 实现
- [x] 4.5 实现符合设计稿的样式（分割线、间距、字号、颜色）

## 5. Layout 集成

- [x] 5.1 修改 `docs/.vuepress/theme/layouts/Layout.vue`，在 content-bottom slot 插入 ArticleFooter
- [x] 5.2 验证 ArticleFooter 正确替换默认 VPPageMeta 显示效果
- [x] 5.3 添加响应式样式支持（移动端适配）

## 6. 测试与验证

- [x] 6.1 启动开发服务器验证字数统计显示正确
- [x] 6.2 验证更新时间从 Git 历史正确获取并显示
- [x] 6.3 验证 GitHub Star 按钮渲染并显示实时 Star 数
- [x] 6.4 执行构建命令验证无编译错误
- [x] 6.5 使用 playwright-cli 测试前端页面渲染效果