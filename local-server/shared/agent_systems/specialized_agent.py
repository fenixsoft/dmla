# SpecializedAgent, AgentMessage, MessageBus 定义
# 从文档自动提取生成

import time

class SpecializedAgent:
    """专业化 Agent 基类，封装角色定义和消息处理循环"""

    def __init__(self, agent_id, role, description, tools, bus):
        self.agent_id = agent_id
        self.role = role
        self.description = description
        self.tools = tools
        self.bus = bus
        self.status = "idle"

    def get_system_prompt(self):
        """根据角色生成系统提示词"""
        tool_list = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools.get_schemas()])
        return "\n".join([
            f"你是{self.role}。{self.description}",
            "",
            "可用工具：",
            tool_list,
            "",
            "行为规范：",
            "1. 只处理与你的角色专长相关的任务",
            "2. 使用可用工具完成分配的任务",
            "3. 任务完成后通过 RESULT_SUBMIT 消息提交结构化结果",
            "4. 遇到无法处理的问题时通过 ERROR_REPORT 消息说明具体原因",
        ])

    def process(self, message):
        """处理接收到的消息"""
        if message.type == "task_assign":
            self.status = "working"
            result = self._execute(message.payload)
            reply = AgentMessage(
                msg_type="result_submit" if "error" not in result else "error_report",
                sender=self.agent_id,
                receiver=message.sender,
                payload=result,
                correlation_id=message.correlation_id
            )
            self.bus.send(reply)
            self.status = "idle"

    def _execute(self, task):
        """执行具体任务（子类覆盖以提供领域专长）"""
        return {"status": "completed", "summary": f"{self.role} 完成任务: {task.get('description', '')}"}


class AgentMessage:
    """Agent 间通信的结构化消息"""

    def __init__(self, msg_type, sender, receiver, payload, correlation_id=None):
        self.type = msg_type
        self.sender = sender
        self.receiver = receiver
        self.payload = payload
        self.correlation_id = correlation_id
        self.timestamp = time.time()


class MessageBus:
    """消息总线：支持点对点消息传递"""

    def __init__(self):
        self._queues = {}

    def send(self, message):
        """向指定接收者发送消息"""
        if message.receiver not in self._queues:
            self._queues[message.receiver] = []
        self._queues[message.receiver].append(message)

    def receive(self, agent_id):
        """接收下一条消息（FIFO 顺序）"""
        queue = self._queues.get(agent_id, [])
        if queue:
            return queue.pop(0)
        return None
