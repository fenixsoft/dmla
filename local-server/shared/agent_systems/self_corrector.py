# SelfCorrector 定义
# 从文档自动提取生成

import json

# 自我修正模块：检测错误并根据类型选择修正策略
class SelfCorrector:
    """自我修正模块，根据错误类型执行对应的恢复策略"""

    MAX_RETRIES = 3

    def __init__(self, tool_registry):
        self.registry = tool_registry
        self.error_history = []

    def correct(self, tool_name, params, error_message):
        """分析错误类型，尝试逐步升级的修正策略"""
        self.error_history.append({
            "tool": tool_name, "params": params, "error": error_message
        })

        # 策略 1：参数修正（针对格式类错误）
        if self._is_format_error(error_message):
            fixed = self._fix_params(params, error_message)
            if fixed != params:
                return self._retry(tool_name, fixed)

        # 策略 2：简化参数重试（针对内容类错误）
        simplified = self._simplify_params(params)
        if simplified != params:
            result = self._retry(tool_name, simplified)
            if result.get("success"):
                return result

        # 策略 3：换用备选工具（当前工具不可用时）
        alt = self._find_alternative(tool_name)
        if alt:
            result = self._retry(alt, params)
            if result.get("success"):
                return result

        return {"success": False, "error": "所有修正策略已耗尽", "history": self.error_history[-self.MAX_RETRIES:]}

    def _is_format_error(self, error):
        fmt_keywords = ["json", "parse", "参数", "格式", "缺少", "required", "类型", "type"]
        return any(kw in str(error).lower() for kw in fmt_keywords)

    def _fix_params(self, params, error):
        """尝试修复参数（简化实现：传递原始参数让 LLM 决定如何调整）"""
        return params

    def _simplify_params(self, params):
        """简化参数：去除可能引起问题的可选字段"""
        return {k: v for k, v in params.items() if v is not None}

    def _find_alternative(self, tool_name):
        """查找功能相近的备选工具"""
        alternatives = {
            "search": ["read_file"],
            "execute_code": [],
        }
        return alternatives.get(tool_name, [None])[0]

    def _retry(self, tool_name, params):
        """执行重试并返回结果"""
        result = self.registry.execute(tool_name, **params)
        success = "error" not in result
        return {"success": success, "result": result}
