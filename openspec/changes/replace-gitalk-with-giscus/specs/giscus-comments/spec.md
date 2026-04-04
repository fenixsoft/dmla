## ADDED Requirements

### Requirement: Giscus 评论组件渲染

系统 SHALL 在文章页面底部渲染 Giscus 评论组件，基于 GitHub Discussions 实现评论功能。

#### Scenario: 评论组件正常加载
- **WHEN** 用户访问包含评论配置的文章页面
- **THEN** Giscus 评论组件在页面加载完成后懒加载显示
- **AND** 评论组件显示与文章关联的 GitHub Discussion 内容

#### Scenario: Discussions 未启用时的提示
- **WHEN** GitHub Discussions 未在仓库中启用
- **THEN** 评论区域显示"Discussions 未启用"提示信息
- **AND** 提示用户如何启用 Discussions

### Requirement: 评论配置参数

系统 SHALL 支持以下 Giscus 配置参数：
- `repo`: GitHub 仓库（格式：owner/repo）
- `repoId`: 仓库 ID（从 Giscus 配置页面获取）
- `categoryId`: Discussion 分类 ID
- `mapping`: 关联方式（pathname/title/number）
- `lang`: 语言设置（zh-CN）

#### Scenario: 使用 pathname mapping
- **WHEN** 配置 mapping 为 "pathname"
- **THEN** 评论基于文章 URL 路径自动创建 Discussion
- **AND** 相同路径的文章共享同一 Discussion

### Requirement: 主题适配

系统 SHALL 使评论组件主题与站点主题一致。

#### Scenario: 浅色主题
- **WHEN** 站点使用浅色主题
- **THEN** 评论组件使用浅色主题样式

#### Scenario: 深色主题
- **WHEN** 站点使用深色主题
- **THEN** 评论组件自动切换为深色主题样式

### Requirement: 错误处理 UI

系统 SHALL 在评论加载失败时显示友好的错误提示。

#### Scenario: 网络加载失败
- **WHEN** Giscus 脚本加载失败（网络问题或 GitHub 不可用）
- **THEN** 显示"评论加载失败"提示
- **AND** 提供刷新按钮重试加载

#### Scenario: 配置错误
- **WHEN** Giscus 配置参数无效（repo/repoId/categoryId 错误）
- **THEN** 显示"评论配置错误"提示
- **AND** 提示管理员检查配置