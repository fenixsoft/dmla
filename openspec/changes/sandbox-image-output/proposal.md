## Why

当前沙箱仅支持文本输出，用户执行 matplotlib 代码后调用 `plt.show()` 无法看到图片。这在数据科学和机器学习教学场景中是核心痛点，因为可视化是理解数据和模型的关键手段。

采用 IPython Kernel 方案，不仅能解决 matplotlib 图片输出问题，还为未来支持其他富输出（pandas DataFrame 表格、sympy LaTeX 公式、HTML 等）打下基础。

## What Changes

- **新增**: IPython Kernel 集成，支持 Jupyter 消息协议
- **新增**: 图片输出渲染，支持 matplotlib 多图显示和点击放大
- **新增**: 结构化错误输出，显示完整的 Python traceback
- **修改**: 后端执行方式从 `python3 -c` 改为 kernel_runner.py
- **修改**: 前端输出区域支持多种输出类型渲染
- **修改**: Docker 镜像添加 ipykernel、jupyter_client 依赖

## Capabilities

### New Capabilities

- `ipython-kernel-execution`: IPython Kernel 代码执行能力，支持 Jupyter 消息协议的富输出
- `rich-output-rendering`: 前端富输出渲染能力，支持图片、文本、错误等多种输出类型

### Modified Capabilities

- `sandbox-execution`: 执行方式从直接调用 Python 改为通过 IPython Kernel 执行，输出格式从纯文本变为结构化输出

## Impact

### 后端影响

| 文件 | 变更 |
|------|------|
| `local-server/Dockerfile.sandbox` | 添加 ipykernel, jupyter_client |
| `local-server/Dockerfile.sandbox.cpu` | 同上 |
| `local-server/src/sandbox.js` | 调用 kernel_runner.py 替代 `python3 -c` |
| `local-server/src/kernel_runner.py` | 新增：Kernel 执行器 |

### 前端影响

| 文件 | 变更 |
|------|------|
| `RunnableCode.vue` | 解析结构化输出，渲染图片、文本、错误 |
| `index.scss` | 添加图片输出样式、模态框样式 |

### API 变更

响应格式从：
```json
{
  "success": true,
  "output": "文本内容",
  "error": null,
  "executionTime": 0.5
}
```

变为：
```json
{
  "success": true,
  "outputs": [
    { "type": "stream", "name": "stdout", "text": "文本内容" },
    { "type": "display_data", "data": { "image/png": "base64..." } }
  ],
  "executionTime": 0.5
}
```

**BREAKING**: API 响应格式变更，但前端同步更新，无兼容性问题。