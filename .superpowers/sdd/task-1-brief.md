# Task 1: 创建 FC HTTP Handler

## 目标

创建 `local-server/src/fc_handler.py` —— FC 函数计算的 HTTP 入口文件，接收代码执行请求并调用 `kernel_runner.run_code()`。

## 文件

- Create: `local-server/src/fc_handler.py`

## 接口

- Consumes: `kernel_runner.run_code(code: str, timeout: int, stream: bool) -> dict`
- Produces: HTTP `GET /api/sandbox/health`, `POST /api/sandbox/run`, `POST /api/sandbox/stream`

## 实现要求

编写一个基于 Python 内置 `http.server` 模块的 HTTP 服务，包含 `SandboxHandler` 类：

1. **do_GET** → 健康检查，返回 `{"status": "ok", "mode": "fc", "timestamp": time.time()}`
2. **do_POST** → 解析 JSON body `{code, timeout?, stream?}`，调用 `run_code(code, timeout, stream)`，返回 JSON 结果
3. 端口从 `FC_SERVER_PORT` 环境变量读取，默认 9000
4. 错误处理：空 body、无效 JSON、缺少 code 字段、run_code 异常
5. 使用 `BaseHTTPRequestHandler._send_json()` 辅助方法统一发送 JSON 响应
6. 日志输出到 stderr

## 完整代码（从计划中复制）

```python
#!/usr/bin/env python3
"""
FC (Function Compute) HTTP Handler
轻量 HTTP 入口，接收代码执行请求并调用 kernel_runner.py
"""

import json
import os
import sys
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

# 将 /workspace 加入路径，确保可以 import kernel_runner
sys.path.insert(0, '/workspace')

# 延迟导入 kernel_runner（内部 import matplotlib，首次有开销）
from kernel_runner import run_code

FC_SERVER_PORT = int(os.environ.get('FC_SERVER_PORT', 9000))
DEFAULT_TIMEOUT = 60


class SandboxHandler(BaseHTTPRequestHandler):
    """FC 沙箱 HTTP 请求处理器"""

    def log_message(self, format, *args):
        """重写日志方法，输出到 stderr（FC 日志采集）"""
        print(f"[fc_handler] {format % args}", file=sys.stderr)

    def _send_json(self, status_code, data):
        """发送 JSON 响应"""
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """健康检查（Settings 测试连接使用）"""
        self._send_json(200, {
            'status': 'ok',
            'mode': 'fc',
            'timestamp': __import__('time').time()
        })

    def do_POST(self):
        """代码执行"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_json(400, {
                    'success': False,
                    'outputs': [{
                        'type': 'error',
                        'ename': 'EmptyRequest',
                        'evalue': 'Request body is empty',
                        'traceback': []
                    }],
                    'executionTime': 0
                })
                return

            raw_body = self.rfile.read(content_length)
            data = json.loads(raw_body)

            code = data.get('code', '')
            timeout = data.get('timeout', DEFAULT_TIMEOUT)
            stream = data.get('stream', False)

            if not code:
                self._send_json(400, {
                    'success': False,
                    'outputs': [{
                        'type': 'error',
                        'ename': 'MissingCode',
                        'evalue': 'No code provided in request',
                        'traceback': []
                    }],
                    'executionTime': 0
                })
                return

            result = run_code(code, timeout=timeout, stream=stream)

            if stream:
                self.send_response(200)
                self.end_headers()
            else:
                self._send_json(200, result)

        except json.JSONDecodeError:
            self._send_json(400, {
                'success': False,
                'outputs': [{
                    'type': 'error',
                    'ename': 'InvalidJSON',
                    'evalue': 'Request body is not valid JSON',
                    'traceback': []
                }],
                'executionTime': 0
            })
        except Exception as e:
            print(f"[fc_handler] Unexpected error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self._send_json(500, {
                'success': False,
                'outputs': [{
                    'type': 'error',
                    'ename': type(e).__name__,
                    'evalue': str(e),
                    'traceback': traceback.format_exc().split('\n')
                }],
                'executionTime': 0
            })


def main():
    server = HTTPServer(('0.0.0.0', FC_SERVER_PORT), SandboxHandler)
    print(f"[fc_handler] Listening on port {FC_SERVER_PORT}", file=sys.stderr)
    server.serve_forever()


if __name__ == '__main__':
    main()
```

## 验证步骤

1. 语法检查：`python3 -c "import py_compile; py_compile.compile('local-server/src/fc_handler.py', doraise=True)"`
2. 验证 kernel_runner 可导入：`python3 -c "import sys; sys.path.insert(0, 'local-server/src'); from kernel_runner import run_code; print('run_code imported OK')"`
3. Commit

## 全局约束（来自设计文档）

- `kernel_runner.py` 的 `DEFAULT_TIMEOUT` 保持 60 秒不变
- FC handler 端口：读取 `FC_SERVER_PORT` 环境变量，默认 9000
