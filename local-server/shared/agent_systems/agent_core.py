# AgentCore 定义
# 从文档自动提取生成

import json

class AgentCore:
    """基于 ReAct 模式的 Agent 核心，管理思考-行动-观察循环"""

    def __init__(self, tool_registry, memory_manager, max_iterations=10):
        self.tools = tool_registry
        self.memory = memory_manager
        self.max_iterations = max_iterations

    def run(self, goal):
        """执行主循环直到任务完成或达到最大迭代次数"""
        self.memory.add("user", goal)

        for iteration in range(self.max_iterations):
            prompt = self._build_prompt()
            response = self._call_llm(prompt)
            thought, action, final_answer = self._parse_response(response)

            if thought:
                self.memory.add("thought", thought)

            if final_answer:
                self.memory.add("answer", final_answer)
                return final_answer

            if action:
                tool_name = action.get("tool", "")
                params = action.get("parameters", {})
                observation = self.tools.execute(tool_name, **params)
                self.memory.add("observation", json.dumps(observation, ensure_ascii=False))
            else:
                self.memory.add("observation", "[警告] 未能解析出有效的行动指令")

        return "已达到最大迭代次数，任务未完成。"

    def _build_prompt(self):
        """构建发送给 LLM 的完整提示词"""
        tools_desc = json.dumps(self.tools.get_schemas(), ensure_ascii=False, indent=2)
        context = self.memory.get_context()
        history = "\n".join([f"[{m['role']}] {m['content']}" for m in context])

        return "\n".join([
            "你是一个自主技术调研 Agent。你的任务是完成用户指定的调研目标。",
            "",
            "可用工具（JSON 格式）：",
            tools_desc,
            "",
            "执行流程：",
            "1. 每次只调用一个工具",
            "2. 观察工具返回的结果",
            "3. 根据观察决定下一步",
            "4. 当任务完成时，输出最终答案",
            "",
            "输出格式：",
            '[Thought] 当前状态分析和下一步策略',
            '[Action] {"tool": "工具名", "parameters": {"参数名": "参数值"}}',
            "",
            "任务完成时：",
            "[Thought] 确认任务已完成",
            "[FinalAnswer] 完整的研究报告",
            "",
            "对话历史：",
            history,
        ])

    def _call_llm(self, prompt):
        """调用 LLM API 获取响应（简化实现，实际应接入真实 API）"""
        return f"[Thought] 处理用户请求\n[Action] {{\"tool\": \"search\", \"parameters\": {{\"query\": \"示例搜索\"}}}}"

    def _parse_response(self, response):
        """解析 LLM 响应，提取思考、行动和最终答案"""
        thought = None
        action = None
        final_answer = None

        if "[Thought]" in response:
            parts = response.split("[Thought]", 1)
            if len(parts) > 1:
                thought_part = parts[1]
                if "[Action]" in thought_part:
                    thought = thought_part.split("[Action]")[0].strip()
                elif "[FinalAnswer]" in thought_part:
                    thought = thought_part.split("[FinalAnswer]")[0].strip()
                else:
                    thought = thought_part.strip()

        if "[Action]" in response:
            action_part = response.split("[Action]")[1]
            if "[FinalAnswer]" in action_part:
                action_part = action_part.split("[FinalAnswer]")[0]
            action_part = action_part.strip()
            try:
                action = json.loads(action_part)
            except json.JSONDecodeError:
                action = None

        if "[FinalAnswer]" in response:
            final_answer = response.split("[FinalAnswer]")[1].strip()

        return thought, action, final_answer
