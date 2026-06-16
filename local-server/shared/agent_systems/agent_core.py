# AgentCore 定义
# 从文档自动提取生成

import json

class AgentCore:
    """基于 ReAct 模式的 Agent 核心"""
    
    def __init__(self, tool_registry, memory_manager, max_iterations=10):
        self.tools = tool_registry
        self.memory = memory_manager
        self.max_iterations = max_iterations
    
    def run(self, goal):
        """执行 Agent 主循环"""
        self.memory.add("user", goal)
        
        for i in range(self.max_iterations):
            # 构建提示词：系统指令 + 记忆上下文 + 工具描述
            prompt = self._build_prompt()
            
            # 调用 LLM 生成思考和行动
            response = self._call_llm(prompt)
            
            # 解析响应：提取思考、行动和最终答案
            thought, action, final_answer = self._parse_response(response)
            
            if thought:
                self.memory.add("thought", thought)
            
            if final_answer:
                return final_answer
            
            if action:
                # 执行工具调用
                observation = self.tools.execute(
                    action["tool"], **action.get("parameters", {})
                )
                self.memory.add("observation", json.dumps(observation, ensure_ascii=False))
        
        return "达到最大迭代次数，任务未完成"
