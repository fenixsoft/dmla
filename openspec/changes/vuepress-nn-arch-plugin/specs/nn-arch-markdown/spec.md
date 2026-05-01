## ADDED Requirements

### Requirement: Markdown 代码块语法解析

系统 SHALL 识别 Markdown 中以 `nn-arch` 为语言标识符的代码块，并提取其中的 YAML 内容作为神经网络架构定义。

#### Scenario: 基础代码块识别
- **WHEN** Markdown 文档包含 ` ```nn-arch ` 代码块
- **THEN** 系统 SHALL 提取代码块内容作为 YAML 网络定义

#### Scenario: 带尺寸参数的代码块
- **WHEN** Markdown 文档包含 ` ```nn-arch width=800 height=400 ` 代码块
- **THEN** 系统 SHALL 解析 width 和 height 参数
- **AND** 参数值 SHALL 为正整数
- **AND** 未指定参数时 SHALL 使用默认值

### Requirement: 尺寸参数格式

系统 SHALL 支持以下尺寸参数格式：
- `width=N`: 设置图片宽度（像素）
- `height=N`: 设置图片高度（像素）
- 参数顺序可任意组合

#### Scenario: 仅指定宽度
- **WHEN** 代码块标识符为 `nn-arch width=600`
- **THEN** 系统 SHALL 设置宽度为 600px
- **AND** 高度 SHALL 自动根据 SVG 内容比例计算

#### Scenario: 仅指定高度
- **WHEN** 代码块标识符为 `nn-arch height=300`
- **THEN** 系统 SHALL 设置高度为 300px
- **AND** 宽度 SHALL 自动根据 SVG 内容比例计算

#### Scenario: 同时指定宽高
- **WHEN** 代码块标识符为 `nn-arch width=800 height=400`
- **THEN** 系统 SHALL 设置宽度为 800px，高度为 400px

#### Scenario: 无效参数处理
- **WHEN** 代码块标识符包含无效参数（如 `nn-arch width=abc`）
- **THEN** 系统 SHALL 忽略无效参数
- **AND** 使用默认尺寸渲染