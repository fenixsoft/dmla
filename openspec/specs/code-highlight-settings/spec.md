## ADDED Requirements

### Requirement: 用户可以选择代码高亮风格

系统 SHALL 提供代码高亮风格选择功能，允许用户从下拉列表中选择不同的 PrismJS 高亮主题。

#### Scenario: 打开设置弹窗查看高亮风格选项
- **WHEN** 用户点击导航栏的设置按钮
- **THEN** 系统显示设置弹窗，标题为"设置"
- **AND** 弹窗包含"代码高亮风格"设置项
- **AND** 下拉列表包含以下选项：当前样式、Default、Coy、Dark、Funky、Okaidia、Solarized Light、Tomorrow Night、Twilight

#### Scenario: 选择高亮风格并实时预览
- **WHEN** 用户在下拉框中选择一个高亮风格
- **THEN** 预览区域的样例代码片段立即更新显示对应的语法高亮效果

### Requirement: 高亮风格配置持久化存储

系统 SHALL 将用户选择的高亮风格保存到客户端 localStorage，并在页面刷新后自动应用。

#### Scenario: 保存高亮风格配置
- **WHEN** 用户选择高亮风格后点击"保存设置"按钮
- **THEN** 系统将配置保存到 localStorage 的 `site-config` key
- **AND** 配置包含 `highlightTheme` 字段，值为选中的主题标识

#### Scenario: 页面刷新后自动应用保存的风格
- **WHEN** 用户刷新页面或重新访问网站
- **THEN** 系统从 localStorage 读取 `site-config`
- **AND** 如果存在 `highlightTheme` 配置，系统加载对应的高亮主题 CSS
- **AND** 全站代码块使用保存的高亮风格渲染

### Requirement: 高亮风格全局生效

系统 SHALL 将选中的高亮风格应用到全站所有代码块，包括 Markdown 渲染的代码块和可运行代码块。

#### Scenario: 高亮风格应用于普通代码块
- **WHEN** 用户保存高亮风格配置
- **THEN** 所有 Markdown 代码块（```python``` 等）使用选中的高亮风格渲染

#### Scenario: 高亮风格应用于可运行代码块
- **WHEN** 用户保存高亮风格配置
- **THEN** 所有 runnable-code-block 组件中的代码使用选中的高亮风格渲染

### Requirement: 沙箱配置功能保持不变

系统 SHALL 在设置弹窗中保留原有的沙箱服务地址配置功能。

#### Scenario: 配置沙箱服务地址
- **WHEN** 用户在设置弹窗中修改沙箱服务地址
- **THEN** 系统将沙箱配置保存到 `site-config` 的 `sandboxEndpoint` 字段
- **AND** 保存后沙箱功能正常使用新地址执行代码

### Requirement: 配置数据迁移

系统 SHALL 自动将旧的 `sandbox-config` 数据迁移到新的 `site-config` key。

#### Scenario: 首次访问时自动迁移旧配置
- **WHEN** 用户首次访问更新后的网站
- **AND** localStorage 中存在旧的 `sandbox-config` 数据
- **THEN** 系统读取 `sandbox-config` 内容
- **AND** 将数据迁移到 `site-config`
- **AND** 删除旧的 `sandbox-config` key