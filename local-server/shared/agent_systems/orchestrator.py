# Orchestrator 定义
# 从文档自动提取生成

from shared.agent_systems.specialized_agent import AgentMessage

# 编排器：任务分解、Agent 分配和结果整合
class Orchestrator:
    """集中式编排器，负责任务分解、分配和结果整合"""

    def __init__(self, bus, agents, planner, fault_handler=None):
        self.bus = bus
        self.agents = {a.agent_id: a for a in agents}
        self.planner = planner
        self.fault = fault_handler
        self.results = {}

    def execute(self, goal):
        """执行完整工作流：分解→分配→收集→整合"""
        tasks = self.planner.decompose(goal)
        report_parts = []

        for task in tasks:
            agent_id = self._select_agent(task["action"])
            if agent_id is None:
                continue

            if self.fault and not self.fault.can_execute(agent_id):
                print(f"断路器已断开，跳过 Agent: {agent_id}")
                report_parts.append({"step": task["description"], "agent": agent_id, "result": {"status": "skipped"}})
                continue

            self._assign(task, agent_id)
            result = self._collect(timeout=120)

            if result:
                if self.fault:
                    self.fault.record_success(agent_id)
                report_parts.append({"step": task["description"], "agent": agent_id, "result": result})
            else:
                if self.fault:
                    triggered = self.fault.record_failure(agent_id)
                    if triggered:
                        print(f"Agent {agent_id} 连续失败，断路器已打开")
                report_parts.append({"step": task["description"], "agent": agent_id, "result": {"status": "timeout"}})

        return self._compile_report(goal, report_parts)

    def _select_agent(self, action_type):
        """根据任务类型选择最合适的 Agent"""
        role_map = {
            "research": "researcher",
            "filter": "researcher",
            "implement": "coder",
            "test": "coder",
            "benchmark": "coder",
            "report": "researcher"
        }
        target_role = role_map.get(action_type)
        for agent in self.agents.values():
            if target_role and target_role in agent.role.lower():
                return agent.agent_id
        return list(self.agents.keys())[0] if self.agents else None

    def _assign(self, task, agent_id):
        """将子任务分配给指定 Agent"""
        from shared.agent_systems.specialized_agent import AgentMessage
        msg = AgentMessage(
            msg_type="task_assign",
            sender="orchestrator",
            receiver=agent_id,
            payload=task,
            correlation_id=task["id"]
        )
        self.bus.send(msg)

    def _collect(self, timeout=120):
        """等待并收集 Agent 的执行结果"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.bus.receive("orchestrator")
            if msg:
                if msg.type == "result_submit":
                    self.results[msg.correlation_id] = msg.payload
                    return msg.payload
                elif msg.type == "error_report":
                    self.results[msg.correlation_id] = msg.payload
                    return msg.payload
            time.sleep(0.1)
        return None

    def _compile_report(self, goal, parts):
        """整合各阶段的产物为最终报告"""
        sections = []
        for p in parts:
            sections.append(f"## {p['step']}\n（由 {p['agent']} 完成）\n\n{p['result']}")
        return {
            "title": f"技术调研报告: {goal[:50]}",
            "sections": sections,
            "metadata": {"steps": len(parts), "completed": sum(1 for p in parts if p['result'])}
        }
