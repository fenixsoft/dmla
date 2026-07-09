# SpecializedAgent 定义
# 从文档自动提取生成

# 专业化 Agent 基类
class SpecializedAgent:
    """专业化 Agent 基类，每个子类实现特定领域的逻辑"""
    
    def __init__(self, agent_id, role, description, tools, message_bus):
        self.agent_id = agent_id
        self.role = role
        self.description = description
        self.tools = tools
        self.bus = message_bus
        self.status = "idle"
        self.current_task = None
    
    def get_system_prompt(self):
        """生成 Agent 的系统提示词"""
        tool_descriptions = "\n".join([
            f"- {t['name']}: {t['description']}" for t in self.tools.get_schemas()
        ])
        return f"""你是一个{self.role}。
