# CircuitBreaker 定义
# 从文档自动提取生成

# 断路器实现
class CircuitBreaker:
    """断路器：在 Agent 持续失败时暂停向其分配任务"""
    
    CLOSED = "closed"      # 正常工作
    OPEN = "open"          # 暂停分配
    HALF_OPEN = "half_open"  # 试探性恢复
    
    def __init__(self, failure_threshold=3, recovery_timeout=60, half_open_max=1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self._states = {}  # agent_id -> state
        self._failure_counts = {}
        self._last_failure_time = {}
        self._half_open_counts = {}
    
    def can_execute(self, agent_id):
        """检查是否可以向该 Agent 分配任务"""
        state = self._states.get(agent_id, self.CLOSED)
        
        if state == self.CLOSED:
            return True
        
        if state == self.OPEN:
            # 检查是否到达恢复时间
            elapsed = time.time() - self._last_failure_time.get(agent_id, 0)
            if elapsed >= self.recovery_timeout:
                self._states[agent_id] = self.HALF_OPEN
                self._half_open_counts[agent_id] = 0
                return True
            return False
        
        if state == self.HALF_OPEN:
            # 限制试探性请求的数量
            return self._half_open_counts.get(agent_id, 0) < self.half_open_max
        
        return False
    
    def record_success(self, agent_id):
        """记录成功执行"""
        self._states[agent_id] = self.CLOSED
        self._failure_counts[agent_id] = 0
    
    def record_failure(self, agent_id):
        """记录执行失败"""
        self._failure_counts[agent_id] = self._failure_counts.get(agent_id, 0) + 1
        self._last_failure_time[agent_id] = time.time()
        
        if self._failure_counts[agent_id] >= self.failure_threshold:
            self._states[agent_id] = self.OPEN
