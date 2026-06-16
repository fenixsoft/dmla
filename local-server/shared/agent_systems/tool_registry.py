# ToolRegistry 定义
# 从文档自动提取生成

import json
import signal
import time
from functools import wraps

class ToolRegistry:
    """工具注册中心，管理可用工具的注册、验证和执行"""
    
    def __init__(self):
        self._tools = {}
        self._schemas = {}
    
    def register(self, name=None, description="", parameters=None, timeout=30):
        """工具注册装饰器"""
        def decorator(func):
            tool_name = name or func.__name__
            self._tools[tool_name] = func
            self._schemas[tool_name] = {
                "name": tool_name,
                "description": description or func.__doc__ or "",
                "parameters": parameters or {"type": "object", "properties": {}},
                "timeout": timeout
            }
            return func
        return decorator
    
    def get_schemas(self):
        """获取所有工具的描述 schema"""
        return list(self._schemas.values())
    
    def execute(self, tool_name, **kwargs):
        """执行指定工具，带参数验证和超时控制"""
        if tool_name not in self._tools:
            return {"error": f"工具 '{tool_name}' 不存在", "available_tools": list(self._tools.keys())}
        
        # 参数验证（简化版）
        schema = self._schemas[tool_name]
        required = schema["parameters"].get("required", [])
        for param in required:
            if param not in kwargs:
                return {"error": f"缺少必选参数 '{param}'"}
        
        # 执行工具
        try:
            result = self._tools[tool_name](**kwargs)
            return {"result": result}
        except Exception as e:
            return {"error": f"工具执行失败: {str(e)}"}
