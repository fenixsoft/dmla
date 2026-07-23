# FaultHandler 定义
# 从文档自动提取生成

import re

# 基础容错模块：超时保护与断路器
class FaultHandler:
    """容错处理器，提供超时和断路器两种基础保护机制"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold=3, recovery_timeout=30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = {}
        self._states = {}
        self._last_failure = {}

    def can_execute(self, agent_id):
        """检查 Agent 是否可用（断路保护）"""
        state = self._states.get(agent_id, self.CLOSED)

        if state == self.OPEN:
            elapsed = time.time() - self._last_failure.get(agent_id, 0)
            if elapsed >= self.recovery_timeout:
                self._states[agent_id] = self.HALF_OPEN
                return True
            return False

        return True

    def record_success(self, agent_id):
        """记录成功执行，重置断路器"""
        self._states[agent_id] = self.CLOSED
        self._failures[agent_id] = 0

    def record_failure(self, agent_id):
        """记录执行失败，达到阈值时打开断路器"""
        self._failures[agent_id] = self._failures.get(agent_id, 0) + 1
        self._last_failure[agent_id] = time.time()

        if self._failures[agent_id] >= self.failure_threshold:
            self._states[agent_id] = self.OPEN
            return True
        return False
