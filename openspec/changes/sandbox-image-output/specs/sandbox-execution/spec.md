## MODIFIED Requirements

### Requirement: 沙箱代码执行

系统 SHALL 通过 IPython Kernel 执行 Python 代码，返回结构化的输出结果。

**变更说明**: 执行方式从 `python3 -c` 改为 IPython Kernel，输出格式从纯文本变为结构化输出数组。

#### Scenario: 成功执行返回结构化输出
- **WHEN** 用户通过 API 提交代码 `print("Hello")`
- **THEN** 系统返回 `{ success: true, outputs: [...], executionTime: <number> }`
- **AND** outputs 数组包含 Jupyter 消息协议格式的输出项

#### Scenario: 执行失败返回错误信息
- **WHEN** 用户提交的代码执行失败
- **THEN** 系统返回 `{ success: true, outputs: [{ type: 'error', ... }] }`
- **AND** error 输出项包含完整的 traceback 信息

#### Scenario: GPU 执行请求
- **WHEN** 用户请求使用 GPU 执行代码
- **THEN** 系统在 GPU 版本的 Docker 镜像中启动 IPython Kernel
- **AND** 正确透传 GPU 资源

### Requirement: API 响应格式

API 响应 SHALL 使用新的结构化输出格式。

**变更说明**: 响应格式从 `{ output, error }` 变为 `{ outputs }`。

#### Scenario: 新响应格式
- **WHEN** API 返回执行结果
- **THEN** 响应包含 `outputs` 数组字段
- **AND** 每个 output 包含 `type` 字段标识输出类型

#### Scenario: 向后兼容（可选）
- **WHEN** API 返回执行结果
- **THEN** 响应可包含 `output` 字段作为所有 stream 文本的合并（向后兼容过渡期）