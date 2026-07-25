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
        self._search_count = 0

        for iteration in range(self.max_iterations):
            print(f"\n{'─' * 40}")
            print(f"第 {iteration + 1}/{self.max_iterations} 轮")

            prompt = self._build_prompt(iteration)
            response = self._call_llm(prompt)
            thought, action, final_answer = self._parse_response(response)

            if thought:
                print(f"[思考] {thought[:200]}{'...' if len(thought) > 200 else ''}")
                self.memory.add("thought", thought)

            if final_answer:
                final_answer = self._unescape_newlines(final_answer)
                print(f"[完成] 模型判定任务已完成，生成最终报告")
                self.memory.add("answer", final_answer)
                return final_answer

            if action:
                tool_name = action.get("tool", "")
                params = action.get("parameters", {})
                param_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in params.items())
                print(f"[调用] {tool_name}({param_str})")
                observation = self.tools.execute(tool_name, **params)
                if tool_name == "search":
                    self._search_count += 1
                # 输出执行结果摘要
                if "error" in observation:
                    print(f"[结果] 错误: {str(observation['error'])[:200]}")
                elif "result" in observation:
                    print(f"[结果] 成功: {str(observation['result'])[:200]}")
                else:
                    obs_str = json.dumps(observation, ensure_ascii=False)
                    print(f"[结果] {obs_str[:200]}")
                self.memory.add("observation", json.dumps(observation, ensure_ascii=False))
            else:
                # 无有效行动时，检查模型是否表达了"可以生成报告"的意图
                done_keywords = ["已完成", "信息足够", "可以完成", "任务完成", "报告如下", "现在输出报告"]
                if thought and any(kw in thought for kw in done_keywords):
                    print(f"[意图] 模型判定可以生成报告，立即尝试输出...")
                    fallback_prompt = self._build_fallback_prompt()
                    if fallback_prompt:
                        response = self._call_llm(fallback_prompt)
                        _, _, final_answer = self._parse_response(response)
                        if final_answer:
                            final_answer = self._unescape_newlines(final_answer)
                            print(f"[完成] 回退生成最终报告")
                            self.memory.add("answer", final_answer)
                            return final_answer
                    print(f"[提示] 回退未能生成报告，继续下一轮")
                else:
                    print(f"[提示] 未解析出有效行动，请务必在下轮输出 [Action] 或 [FinalAnswer]")
                self.memory.add("observation", "[提示] 上一步未生成有效行动，请根据当前已收集的信息直接生成 [FinalAnswer]")

        # 达到最大迭代次数时，尝试强制生成部分报告
        print(f"\n{'─' * 40}")
        print(f"已达最大迭代次数 {self.max_iterations}，尝试回退生成报告...")
        fallback_prompt = self._build_fallback_prompt()
        if fallback_prompt:
            response = self._call_llm(fallback_prompt)
            _, _, final_answer = self._parse_response(response)
            if final_answer:
                final_answer = self._unescape_newlines(final_answer)
                print("[回退] 成功生成部分报告")
                return final_answer
            else:
                print("[回退] 未能生成报告")
        return "已达到最大迭代次数，任务未完成。"

    def _build_fallback_prompt(self):
        """构建回退提示词：强制要求模型基于已有信息生成报告"""
        context = self.memory.get_context()
        history = "\n".join(f"[{m['role']}] {m['content']}" for m in context)
        return "\n".join([
            "你已完成多轮搜索和代码执行，现在必须基于以下对话历史中的信息，",
            "直接生成最终调研报告。不要输出 [Thought] 或 [Action]，只输出 [FinalAnswer]。",
            "",
            "对话历史：",
            history,
            "",
            "[FinalAnswer]",
        ])

    def _build_prompt(self, iteration=0):
        """构建发送给 LLM 的完整提示词"""
        context = self.memory.get_context()
        history = "\n".join(
            f"[{m['role']}] {self._truncate(m['content'], 400 if m['role'] != 'thought' else 80)}"
            for m in context
        )

        # 根据当前轮次选择可用的工具、阶段指引和输出格式
        remaining = self.max_iterations - iteration
        if iteration < 3:
            # 搜索阶段：不允许 [FinalAnswer]，必须先搜索
            tools_desc = json.dumps(self.tools.get_schemas(), ensure_ascii=False, indent=2)
            phase_hint = "搜索阶段：必须先搜索！用英文关键词搜索算法资料。"
            action_hint = '[Action] {"tool":"search","parameters":{"query":"关键词"}}'
            final_hint = None
        elif iteration < 7:
            # 编码阶段：移除 search 工具
            coding_schemas = [s for s in self.tools.get_schemas() if s["name"] != "search"]
            tools_desc = json.dumps(coding_schemas, ensure_ascii=False, indent=2)
            phase_hint = (
                "编码阶段：搜索已禁用。必须用 execute_code 编写并运行 Python 代码。"
                "代码应包含：\n"
                "1. quick_sort 和 merge_sort 两个函数\n"
                "2. 对随机数组排序的测试代码\n"
                "3. 用 time 模块测量两种算法的运行时间\n"
                "现在立即调用 execute_code 执行上述代码！"
            )
            action_hint = '[Action] {"tool":"execute_code","parameters":{"code":"import time,random\\narr=[random.randint(0,1000) for _ in range(100)]\\nt=time.time();print(sorted(arr)[:10]);print(time.time()-t)"}}'
            final_hint = (
                "[FinalAnswer] # 调研报告\n\n"
                "## 算法原理\n（说明快速排序和归并排序的原理，列出时间复杂度）\n\n"
                "## 代码实现\n```python\n（两种算法的 Python 实现代码）\n```\n\n"
                "## 性能对比\n| 算法 | 时间复杂度 | 运行时间 |\n|------|-----------|----------|\n| ... | ... | ... |"
            )
        else:
            # 报告阶段：移除所有工具，只允许 [FinalAnswer]
            phase_hint = f"报告阶段：只剩 {remaining} 轮。必须立即输出 [FinalAnswer] 完整报告。"
            tools_desc = "（所有工具已禁用）"
            action_hint = None
            final_hint = (
                "[FinalAnswer] # 调研报告\n\n"
                "## 算法原理\n（说明快速排序和归并排序的原理，列出时间复杂度）\n\n"
                "## 代码实现\n```python\n（两种算法的 Python 实现代码）\n```\n\n"
                "## 性能对比\n| 算法 | 时间复杂度 | 运行时间 |\n|------|-----------|----------|\n| ... | ... | ... |"
            )

        lines = [
            "你是技术调研助手。完成调研目标，输出包含算法说明、代码实现和性能对比的报告。",
            "",
            f"【阶段指引】{phase_hint}",
            "",
            "可用工具：",
            tools_desc,
            "",
            "输出格式（严格按此格式）：",
            "[Thought] 简短说明当前需要什么",
        ]
        if action_hint:
            lines.append(action_hint)
        if final_hint:
            lines.extend(["", "任务完成时：", "[Thought] 已完成", final_hint])
        lines.extend(["", "执行历史：", history])

        return "\n".join(lines)

    @staticmethod
    def _unescape_newlines(text):
        """将 LLM 输出中的字面 \\n 替换为真正的换行符"""
        return text.replace("\\n", "\n")

    @staticmethod
    def _truncate(text, max_len):
        """截断文本到指定长度，超出部分用省略号标记"""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "……"

    def _call_llm(self, prompt):
        """使用本地 Qwen3.5-0.8B-Instruct 模型生成响应"""
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        data_dir = os.environ.get('DMLA_DATA_PATH', '/data')
        model_path = os.path.join(data_dir, 'models', 'llm', 'qwen3.5-0.8b-instruct')

        if not hasattr(self, '_model'):

            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16 if self._device.type == 'cuda' else torch.float32,
                device_map="auto" if self._device.type == 'cuda' else None,
                local_files_only=True,
            )
            if self._device.type == 'cpu':
                self._model = self._model.to(self._device)
            self._model.eval()

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=4096).to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        return response

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
            # 用栈匹配提取 JSON 对象，正确处理嵌套大括号
            action = self._extract_json(action_part)

        if "[FinalAnswer]" in response:
            final_answer = response.split("[FinalAnswer]")[1].strip()

        return thought, action, final_answer

    @staticmethod
    def _extract_json(text):
        """用栈匹配从文本中提取第一个完整 JSON 对象，处理嵌套大括号"""
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None
