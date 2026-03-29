## ADDED Requirements

### Requirement: IPython Kernel 代码执行

系统 SHALL 使用 IPython Kernel 执行用户提交的 Python 代码，支持 Jupyter 消息协议的富输出格式。

#### Scenario: 成功执行简单代码
- **WHEN** 用户提交代码 `print("Hello, World!")`
- **THEN** 系统返回 `{ type: 'stream', name: 'stdout', text: 'Hello, World!\n' }`

#### Scenario: 执行带返回值的表达式
- **WHEN** 用户提交代码 `1 + 1`
- **THEN** 系统返回 `{ type: 'execute_result', data: { 'text/plain': '2' }, execution_count: 1 }`

#### Scenario: 执行错误代码
- **WHEN** 用户提交代码 `1/0`
- **THEN** 系统返回 `{ type: 'error', ename: 'ZeroDivisionError', evalue: 'division by zero', traceback: [...] }`

### Requirement: Kernel 生命周期管理

系统 SHALL 为每次执行请求创建独立的 IPython Kernel，执行完成后立即关闭。

#### Scenario: Kernel 创建和关闭
- **WHEN** 用户提交代码执行请求
- **THEN** 系统创建新的 Kernel 进程
- **AND** 执行完成后关闭 Kernel 进程

#### Scenario: Kernel 超时处理
- **WHEN** 代码执行超过 60 秒
- **THEN** 系统强制终止 Kernel
- **AND** 返回超时错误信息

### Requirement: matplotlib 图片输出

系统 SHALL 支持 matplotlib 的 inline 后端，使 `plt.show()` 自动生成图片输出。

#### Scenario: 单图输出
- **WHEN** 用户提交代码 `import matplotlib.pyplot as plt; plt.plot([1,2,3]); plt.show()`
- **THEN** 系统返回 `{ type: 'display_data', data: { 'image/png': '<base64>' } }`

#### Scenario: 多图输出
- **WHEN** 用户提交代码调用多次 `plt.show()`
- **THEN** 系统为每次 `plt.show()` 返回独立的 `display_data` 输出

#### Scenario: 图片尺寸信息
- **WHEN** 系统返回图片输出
- **THEN** 输出包含 metadata 字段，提供图片宽度和高度信息

### Requirement: 结构化错误输出

系统 SHALL 返回结构化的错误信息，包含异常名称、异常值和完整的 Python traceback。

#### Scenario: 语法错误
- **WHEN** 用户提交代码 `print("missing quote`
- **THEN** 系统返回 `{ type: 'error', ename: 'SyntaxError', evalue: '...', traceback: [...] }`

#### Scenario: 运行时错误
- **WHEN** 用户提交代码 `undefined_variable`
- **THEN** 系统返回 `{ type: 'error', ename: 'NameError', evalue: "name 'undefined_variable' is not defined", traceback: [...] }`

### Requirement: GPU 支持

系统 SHALL 支持在 GPU 镜像中启动 IPython Kernel，保持与现有 GPU 执行能力兼容。

#### Scenario: GPU 镜像执行
- **WHEN** 用户请求使用 GPU 执行
- **THEN** 系统在 GPU 版本的 Docker 镜像中启动 Kernel
- **AND** 正确处理 GPU 相关的输出