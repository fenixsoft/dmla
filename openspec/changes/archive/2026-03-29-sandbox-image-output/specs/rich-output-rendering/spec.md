## ADDED Requirements

### Requirement: 多类型输出渲染

前端 SHALL 支持渲染多种输出类型，包括文本流、图片和错误信息。

#### Scenario: 文本输出渲染
- **WHEN** 后端返回 `{ type: 'stream', name: 'stdout', text: 'Hello' }`
- **THEN** 前端在 `<pre>` 元素中显示文本内容
- **AND** 应用等宽字体样式

#### Scenario: stderr 渲染
- **WHEN** 后端返回 `{ type: 'stream', name: 'stderr', text: 'Error' }`
- **THEN** 前端在 `<pre class="stderr">` 元素中显示文本
- **AND** 应用红色错误样式

### Requirement: 图片渲染

前端 SHALL 将 base64 编码的图片渲染为可见的 `<img>` 元素。

#### Scenario: 单图渲染
- **WHEN** 后端返回 `{ type: 'display_data', data: { 'image/png': '<base64>' } }`
- **THEN** 前端渲染 `<img src="data:image/png;base64,<base64>">`

#### Scenario: 图片样式
- **WHEN** 前端渲染图片输出
- **THEN** 图片应用与 Markdown 图片相同的样式（max-width: 100%, border-radius, box-shadow）
- **AND** 图片可在输出区域内水平居中显示

### Requirement: 图片点击放大

前端 SHALL 支持点击图片后显示全尺寸图片的模态框。

#### Scenario: 打开图片模态框
- **WHEN** 用户点击输出区域中的图片
- **THEN** 系统显示全屏模态框
- **AND** 模态框中显示原图的完整尺寸版本

#### Scenario: 关闭图片模态框
- **WHEN** 用户点击模态框背景或按 ESC 键
- **THEN** 系统关闭模态框

### Requirement: 错误输出渲染

前端 SHALL 以易读的格式渲染 Python 错误信息。

#### Scenario: 错误信息显示
- **WHEN** 后端返回 `{ type: 'error', ename: 'NameError', evalue: '...', traceback: [...] }`
- **THEN** 前端显示错误名称和错误值
- **AND** 显示格式化的 traceback 信息
- **AND** 应用红色错误样式

### Requirement: 混合输出渲染

前端 SHALL 按顺序渲染所有输出项，保持输出的时序性。

#### Scenario: 文本和图片混合输出
- **WHEN** 后端返回 `[stream, display_data, stream]` 序列
- **THEN** 前端按顺序渲染每个输出项

#### Scenario: 执行时间显示
- **WHEN** 代码执行完成
- **THEN** 前端在输出末尾显示执行时间