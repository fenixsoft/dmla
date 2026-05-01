## 1. 插件基础结构

- [x] 1.1 创建插件目录 `docs/.vuepress/plugins/nn-arch/`
- [x] 1.2 安装 nn-arch npm 包依赖 `@icyfenix-dmla/nn-arch`
- [x] 1.3 在 `config.js` 中注册插件

## 2. Markdown 代码块处理

- [x] 2.1 实现 `index.js` 的 `extendsMarkdown` 钩子
- [x] 2.2 解析代码块语言标识符，识别 `nn-arch` 及尺寸参数
- [x] 2.3 将 YAML 内容存储到 HTML 元素的 data 属性中
- [x] 2.4 生成带尺寸信息的占位 HTML 元素

## 3. 客户端渲染逻辑

- [x] 3.1 实现 `client.js` 的 `defineClientConfig`
- [x] 3.2 动态加载 nn-arch npm 包
- [x] 3.3 实现 SVG 渲染函数，调用 `NNArch.generateFromYaml()`
- [x] 3.4 处理渲染错误，显示友好错误提示
- [x] 3.5 应用尺寸参数到生成的 SVG
- [x] 3.6 添加路由切换监听，重新渲染页面图表

## 4. 样式与测试

- [x] 4.1 添加 SVG 居中显示样式
- [x] 4.2 编写测试 Markdown 文档验证基础功能
- [x] 4.3 测试带尺寸参数的代码块
- [x] 4.4 测试无效 YAML 的错误处理
- [x] 4.5 测试路由切换后的重新渲染