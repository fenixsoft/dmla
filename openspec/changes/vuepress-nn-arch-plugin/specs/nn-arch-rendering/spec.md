## ADDED Requirements

### Requirement: SVG 动态渲染

系统 SHALL 在页面加载后调用 nn-arch API，将 YAML 网络定义转换为 SVG 架构图并嵌入文档。

#### Scenario: 成功渲染
- **WHEN** 页面包含有效的 nn-arch 代码块
- **THEN** 系统 SHALL 加载 nn-arch npm 包
- **AND** 调用 `NNArch.generateFromYaml(yaml)` 生成 SVG
- **AND** 将 SVG 嵌入代码块位置

#### Scenario: 渲染失败处理
- **WHEN** YAML 内容无效或 nn-arch 加载失败
- **THEN** 系统 SHALL 显示错误提示信息
- **AND** 不 SHALL 导致页面崩溃

### Requirement: 尺寸控制

系统 SHALL 根据解析的尺寸参数控制 SVG 显示尺寸。

#### Scenario: 应用指定尺寸
- **WHEN** 代码块指定了 width 或 height 参数
- **THEN** SVG SHALL 以指定尺寸显示
- **AND** SVG SHALL 保持原始比例（未指定的维度自动计算）

#### Scenario: 默认尺寸
- **WHEN** 代码块未指定尺寸参数
- **THEN** SVG SHALL 以原始尺寸显示（100% 宽度，自适应高度）

### Requirement: 路由切换支持

系统 SHALL 在 Vue Router 路由切换后重新渲染 nn-arch 图表。

#### Scenario: 页面导航
- **WHEN** 用户从包含 nn-arch 的页面导航到另一个包含 nn-arch 的页面
- **THEN** 系统 SHALL 重新检测并渲染新的 nn-arch 代码块

### Requirement: 样式一致性

渲染的 SVG SHALL 与网站整体风格保持一致。

#### Scenario: SVG 居中显示
- **WHEN** SVG 渲染完成
- **THEN** SVG SHALL 居中显示在文档内容区域
- **AND** 周围 SHALL 有适当边距