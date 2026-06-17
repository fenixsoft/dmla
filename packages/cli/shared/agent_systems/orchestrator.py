# Orchestrator 定义
# 从文档自动提取生成

class Orchestrator:
    """集中式编排器，负责任务分解、分配和结果整合"""
    
    def __init__(self, message_bus, agents):
        self.bus = message_bus
        self.agents = {agent.agent_id: agent for agent in agents}
        self.task_results = {}
        self.workflow = None
    
    def decompose_task(self, goal):
        """将高层目标分解为子任务列表"""
        # 调用 LLM 进行任务分解
        # 简化实现：基于规则的分解
        subtasks = self._rule_based_decompose(goal)
        return subtasks
    
    def assign_task(self, task, agent_id):
        """将子任务分配给指定 Agent"""
        msg = AgentMessage(
            msg_type=MessageType.TASK_ASSIGN,
            sender="orchestrator",
            receiver=agent_id,
            payload=task
        )
        self.bus.send(msg)
    
    def collect_results(self, task_ids, timeout=60):
        """收集指定任务的执行结果"""
        results = {}
        deadline = time.time() + timeout
        
        while time.time() < deadline and len(results) < len(task_ids):
            msg = self.bus.receive("orchestrator")
            if msg and msg.type == MessageType.RESULT_SUBMIT:
                results[msg.correlation_id] = msg.payload
            elif msg and msg.type == MessageType.ERROR_REPORT:
                results[msg.correlation_id] = {"status": "error", "error": msg.payload}
        
        return results
    
    def _rule_based_decompose(self, goal):
        """基于规则的任务分解（简化实现）"""
        # 实际实现中应调用 LLM 进行智能分解
        return [{"id": "task_1", "description": goal, "type": "general"}]
