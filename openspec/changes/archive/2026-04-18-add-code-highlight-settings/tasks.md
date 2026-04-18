## 1. 准备工作

- [x] 1.1 创建高亮主题配置常量文件，定义 PrismJS 主题列表（id、名称、CSS 路径）
- [x] 1.2 创建配置迁移脚本，将 `sandbox-config` 数据迁移到 `site-config`

## 2. UI 组件重构

- [x] 2.1 重构 Settings.vue 组件：
  - 修改标题从"沙箱设置"改为"设置"
  - 添加"代码高亮风格"设置区域
  - 添加下拉框选择器（绑定主题列表）
  - 添加样例代码预览区域
  - 实现选择后实时更新预览效果
- [x] 2.2 更新 NavbarSettings.vue 组件：
  - 修改设置按钮的 title 属性从"沙箱设置"改为"设置"
- [x] 2.3 实现配置保存逻辑：
  - 修改 localStorage key 从 `sandbox-config` 改为 `site-config`
  - 保存时包含 `highlightTheme` 和 `sandboxEndpoint` 字段

## 3. 高亮风格全局加载器

- [x] 3.1 创建 CodeHighlightLoader.vue 客户端组件：
  - 从 localStorage 读取 `site-config.highlightTheme`
  - 动态创建/替换 `<link>` 标签加载对应 PrismJS 主题 CSS
  - 默认不加载任何主题（保持当前自定义样式）
- [x] 3.2 在主题 client.js 中注册 CodeHighlightLoader 组件，确保全局加载

## 4. 样式适配

- [x] 4.1 调整 index.scss 代码块样式：
  - 确保自定义样式与 PrismJS 主题兼容
  - 保持工具栏、按钮等 UI 元素样式不变

## 5. 测试验证

- [x] 5.1 启动本地开发服务器，手动测试设置弹窗功能
- [x] 5.2 使用 playwright-cli 测试前端功能：
  - 打开设置弹窗
  - 选择不同高亮风格，验证预览效果
  - 保存设置，刷新页面验证配置持久化
  - 验证全站代码块高亮风格切换