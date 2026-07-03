# Task 1: 创建 FC HTTP Handler

## Status

DONE

## Commits

- `215060d` feat: 创建 FC Function Compute HTTP Handler 入口文件

## 文件创建

- `local-server/src/fc_handler.py` — FC 函数计算的 HTTP 入口文件

## 实现功能

1. **SandboxHandler.do_GET** — 健康检查，返回 `{"status": "ok", "mode": "fc", "timestamp": ...}`
2. **SandboxHandler.do_POST** — 解析 JSON body `{code, timeout?, stream?}`，调用 `run_code()`，返回 JSON 结果
3. 端口从 `FC_SERVER_PORT` 环境变量读取，默认 9000
4. 错误处理：空 body、无效 JSON、缺少 code 字段、run_code 异常
5. `_send_json()` 辅助方法统一发送 JSON 响应
6. 日志输出到 stderr

## 验证结果

| 验证项 | 结果 |
|--------|------|
| Python 语法编译检查 | 通过 |
| `kernel_runner.run_code` 导入验证 | 通过 |

## 注意事项

- `/workspace` 已加入 sys.path 以适配 FC 运行时目录结构
- `DEFAULT_TIMEOUT` 保持 60 秒，与 kernel_runner.py 一致
- 流式响应（stream 模式）仅发送 200 状态码，不关闭连接，由调用方控制后续数据接收
