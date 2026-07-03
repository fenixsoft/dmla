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

    def _set_cors_headers(self):
        """设置 CORS 响应头，允许跨域访问"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _send_json(self, status_code, data):
        """发送 JSON 响应"""
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status_code)
        self._set_cors_headers()
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """CORS 预检请求"""
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

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
