# 工程实训：技术调研 Agent 协作系统

本次工程实训中，笔者将与你一同构建一个能自动完成技术调研任务的 Agent 系统。输入一个调研需求（如"对比三种排序算法的性能"），系统会自主完成资料搜索、代码编写、基准测试和报告生成。我们将从零开始，先搭建一个能调用工具的简单 Agent，再逐步加入规划、记忆和自我修正能力，最后将任务按专业领域拆解给多个 Agent 协作完成。在此过程中，你可以体会到 Agent 系统中每个组件的引入不是来自预先的架构设计，而是来自任务本身提出的切实需求。

## 实验准备

本实验按任务的演进顺序组织为五个阶段：
- **第一阶段**：从分析调研任务的具体需求出发，确定 Agent 需要哪些工具，构建一个能完成简单搜索和总结的单 Agent，理解工具调用、ReAct 循环和提示词工程在实际任务中的运作方式。
- **第二阶段**：引入规划和记忆，让 Agent 能处理"搜索多个子主题→筛选信息→编码→测试→生成报告"这类多步骤复杂任务，让 Agent 能分解复杂目标并记住中间产物，避免在长任务中丢失上下文。
- **第三阶段**：加入自我修正，应对搜索不相关、代码报错等真实执行中的异常，让 Agent 在搜索、编码和报告生成各阶段遇到错误时能自动调整策略。
- **第四阶段**：分析单 Agent 在调研-编码-审查三个专业门槛上的瓶颈，将系统拆分为 Researcher、Coder、Reviewer 三个专业化 Agent。
- **第五阶段**：为多 Agent 系统加入编排和容错能力，让协作可靠运行。最后用真实的调研任务对单 Agent 和多 Agent 两种方案进行端到端测试和对比。

在开始实验之前，请确保已完成以下准备工作：

1. 已下载  [Qwen3.5-0.8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B) 语言模型。

```bash
# 选择 "下载模型" -> 选择 "Qwen3.5-0.8B-Instruct"
dmla model
```

2. （可选但推荐）注册 **AnySearch API Key**。搜索工具通过 [AnySearch](https://anysearch.com) API 获取互联网搜索结果，未设置 API Key 时以匿名模式运行（频率限制较低，但不影响基本使用）。如需更高的搜索频率，请访问 [AnySearch 控制台](https://anysearch.com/console/api-keys) 创建免费 API Key，然后在启动实验前设置环境变量：

```bash
export ANYSEARCH_API_KEY="你的API密钥"
```

实验的依赖项已在 DMLA 沙箱镜像中预装。可通过以下代码验证 GPU 环境、模型是否正常：

```python gpuonly runnable
# 验证实验环境与模型下载状态
import os
import sys

print(f"Python 版本: {sys.version}")

# 检查 jsonschema 包
try:
    import jsonschema
    print(f"jsonschema: 已安装")
except ImportError:
    print("jsonschema: 未安装，请通过 pip install jsonschema 安装")

# 检查 Qwen3.5-0.8B-Instruct 模型是否已下载
MODEL_PATH = os.path.join(DATA_DIR, "models", "llm", "qwen3.5-0.8b-instruct")
if os.path.isdir(MODEL_PATH):
    has_config = os.path.exists(os.path.join(MODEL_PATH, "config.json"))
    has_model = (os.path.exists(os.path.join(MODEL_PATH, "model.safetensors")) or
                 os.path.exists(os.path.join(MODEL_PATH, "pytorch_model.bin")))
    if has_config and has_model:
        size_mb = sum(
            os.path.getsize(os.path.join(MODEL_PATH, f))
            for f in os.listdir(MODEL_PATH)
            if os.path.isfile(os.path.join(MODEL_PATH, f))
        ) / (1024 * 1024)
        print(f"Qwen3.5-0.8B-Instruct: 已下载 ({size_mb:.0f} MB)")
    else:
        print("Qwen3.5-0.8B-Instruct: 模型文件不完整，请通过 dmla model 重新下载")
else:
    print("Qwen3.5-0.8B-Instruct: 未找到，请运行 dmla model 选择 Qwen3.5-0.8B 下载")
```

## 第一阶段：构建基础 Agent

技术调研任务要求 Agent 依次完成理解调研主题、搜索相关资料、筛选和整理信息、编写示例代码、运行代码验证、将结果整合为报告。Agent 至少需要获取外部信息（搜索）、执行代码并观察结果、读写文件以保存中间产物三种能力。我们先实现这些工具，再围绕它们构建 Agent 的决策循环。

### 工具注册中心

工具是 Agent 接触外部世界的接口。`ToolRegistry` 提供了一套统一的机制来管理工具的注册、描述和调用。每个工具注册时需要提供名称、功能描述和参数的 JSON Schema，LLM 通过读取这些 Schema 来理解何时以及如何使用每个工具。调用时，注册中心自动验证必选参数并捕获执行异常，确保单个工具的失败不会直接导致 Agent 崩溃。

```python runnable gpuonly extract-class="ToolRegistry"
# 工具注册中心：管理工具的描述、注册和执行
from functools import wraps

class ToolRegistry:
    """工具注册中心，管理可用工具的注册、schema 查询和执行"""

    def __init__(self):
        self._tools = {}
        self._schemas = {}

    def register(self, name=None, description="", parameters=None):
        """工具注册装饰器"""
        def decorator(func):
            tool_name = name or func.__name__
            self._tools[tool_name] = func
            self._schemas[tool_name] = {
                "name": tool_name,
                "description": description or (func.__doc__ or "").strip(),
                "parameters": parameters or {"type": "object", "properties": {}}
            }
            return func
        return decorator

    def get_schemas(self):
        """获取所有已注册工具的描述 schema，供 LLM 理解可用工具"""
        return list(self._schemas.values())

    def execute(self, tool_name, **kwargs):
        """执行指定工具，自动验证必选参数并捕获异常"""
        if tool_name not in self._tools:
            return {"error": f"工具 '{tool_name}' 不存在", "available": list(self._tools.keys())}

        schema = self._schemas[tool_name]
        required = schema["parameters"].get("required", [])
        for param in required:
            if param not in kwargs:
                return {"error": f"缺少必选参数 '{param}'"}

        try:
            result = self._tools[tool_name](**kwargs)
            return {"result": result}
        except Exception as e:
            return {"error": f"工具执行异常: {str(e)}"}
```

### 为调研任务注册工具

接下来为我们的调研任务注册四个工具。搜索工具负责获取外部资料，代码执行工具负责运行 Python 代码并收集输出，文件读写工具负责保存中间产物和最终报告。注意每个工具的 `description` 是写给 LLM 看的，它决定了 LLM 选择工具的准确率。描述要准确说明工具的用途和适用场景，不能太笼统也不能太细节。

```python runnable
# 注册调研任务所需的工具
import subprocess
import os
from shared.agent_systems.tool_registry import ToolRegistry

registry = ToolRegistry()

@registry.register(
    name="search",
    description="搜索互联网获取技术资料。适用于查找算法原理、技术文档、学术论文等信息。返回搜索结果摘要列表。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "description": "返回结果数量上限", "default": 5}
        },
        "required": ["query"]
    }
)
def search_tool(query, max_results=5):
    """通过 AnySearch API 搜索互联网，支持匿名访问和 API Key 两种模式"""
    import requests
    import os

    api_key = os.environ.get("ANYSEARCH_API_KEY", "")
    headers = {
        "Content-Type": "application/json",
        "X-Anysearch-Client": "dmla-agent/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search",
            "arguments": {
                "query": query,
                "max_results": min(max_results, 10)
            }
        }
    }

    try:
        resp = requests.post(
            "https://api.anysearch.com/mcp",
            json=payload,
            headers=headers,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            return {
                "query": query,
                "error": data["error"].get("message", str(data["error"])),
                "results": ""
            }

        result = data.get("result", {})
        content = result.get("content", [])
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        results_text = "\n".join(text_parts) if text_parts else str(result)
        return {
            "query": query,
            "results": results_text[:5000],
            "count": len(text_parts)
        }

    except requests.exceptions.ConnectionError:
        return {"query": query, "error": "无法连接到搜索服务，请检查网络连接", "results": ""}
    except requests.exceptions.Timeout:
        return {"query": query, "error": "搜索请求超时", "results": ""}
    except Exception as e:
        return {"query": query, "error": f"搜索异常: {str(e)}", "results": ""}

@registry.register(
    name="execute_code",
    description="执行 Python 代码并返回标准输出。适用于验证算法实现、运行基准测试、检查代码正确性。每次调用是独立的。",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "待执行的 Python 代码"}
        },
        "required": ["code"]
    }
)
def execute_code_tool(code):
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return {"output": output, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"error": "代码执行超时（30 秒）"}

@registry.register(
    name="write_file",
    description="将内容写入文件。适用于保存调研笔记、代码草稿和最终报告。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "要写入的内容"}
        },
        "required": ["path", "content"]
    }
)
def write_file_tool(path, content):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"written": path, "size": len(content)}

@registry.register(
    name="read_file",
    description="读取文件内容。适用于查看之前保存的笔记、代码或报告草稿。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"}
        },
        "required": ["path"]
    }
)
def read_file_tool(path):
    if not os.path.exists(path):
        return {"error": f"文件不存在: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content, "size": len(content)}

print(f"已注册工具: {[s['name'] for s in registry.get_schemas()]}")
```

### Agent 核心循环

有了工具，Agent 还需要一个决策中枢来决定什么时候调用哪个工具。我们采用 ReAct 模式，每轮循环中，Agent 先思考当前处境和下一步策略，然后选择工具执行，观察执行结果，再根据观察更新思考，如此循环直到任务完成。这个"行动→观察→调整"的闭环正是 Agent 区别于单次问答的关键，它能根据实际执行反馈动态调整策略，而不是一条路走到黑。

```python runnable gpuonly extract-class="AgentCore"
# Agent 核心循环：实现 ReAct 模式的思考-行动-观察循环
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
```

## 第二阶段：规划与记忆

第一阶段的 Agent 能完成搜索一个主题并总结这样的简单任务。但真实的技术调研需要搜索多个子主题、筛选对比来源、编写和测试代码、将分散的发现组织成结构化报告。这些子任务之间存在依赖关系（代码实现依赖于对算法的理解，基准测试依赖于代码已调通），如果 Agent 想到哪做到哪，很容易遗漏关键步骤。此外，多轮对话产生的大量中间信息（搜索结果、代码片段、测试数据）会迅速超出 LLM 的上下文窗口，早期的重要信息一旦被裁剪就永久丢失。

### 任务规划器

规划器的职责是将高层目标分解为结构化的子任务序列。下面的 `Planner` 使用基于规则的分解策略，根据目标中的关键词判断需要哪些子任务，生成一个有序的任务列表。任务之间形成有向无环的依赖关系，每个步骤的产出是下一步骤的输入，确保不遗漏关键环节。

```python runnable gpuonly extract-class="Planner"
# 任务规划器：将高层目标分解为子任务序列
class Planner:
    """任务规划器，负责目标分解和进度跟踪"""

    def __init__(self):
        self.plan = []
        self.current_step = 0

    def decompose(self, goal):
        """根据目标类型选择分解策略，生成子任务列表"""
        keywords = goal.lower()
        tasks = []

        tasks.append({"id": "step_1", "action": "research", "description": "搜索并整理核心概念和原理"})
        tasks.append({"id": "step_2", "action": "filter", "description": "筛选可靠来源，提取关键信息"})

        if "代码" in goal or "实现" in goal or "code" in keywords or "implement" in keywords:
            tasks.append({"id": "step_3", "action": "implement", "description": "根据调研结果编写实现代码"})
            tasks.append({"id": "step_4", "action": "test", "description": "运行测试验证代码正确性"})

        if "对比" in goal or "比较" in goal or "benchmark" in keywords or "compare" in keywords:
            tasks.append({"id": "step_bench", "action": "benchmark", "description": "设计并运行对比实验，收集性能数据"})

        tasks.append({"id": "step_final", "action": "report", "description": "整合所有发现和代码，生成最终报告"})

        self.plan = tasks
        self.current_step = 0
        return tasks

    def next_task(self):
        """返回下一个待执行的子任务"""
        if self.current_step < len(self.plan):
            task = self.plan[self.current_step]
            self.current_step += 1
            return task
        return None

    def progress(self):
        """返回当前执行进度"""
        total = len(self.plan)
        done = self.current_step
        return {"completed": done, "total": total, "percent": int(done / total * 100) if total > 0 else 0}
```

### 记忆管理器

记忆管理器维护两类信息。对话历史是 Agent 推理的直接上下文，记录每轮的用户输入、Agent 思考、工具调用和观察结果。当历史长度超过阈值时，早期记录被压缩为摘要以控制上下文窗口的占用。关键事实是从对话中提取的持久信息（搜索到的技术细节、已完成的代码片段、测试收集的性能数据），它们独立于对话轮次而存在，在后续阶段中可被随时检索。

```python runnable gpuonly extract-class="MemoryManager"
# 记忆管理器：维护对话历史与关键事实
class MemoryManager:
    """记忆管理器，维护短期对话历史和长期关键信息"""

    def __init__(self, max_history=20):
        self.history = []
        self.key_facts = []
        self.max_history = max_history

    def add(self, role, content):
        """添加一条记录到对话历史"""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self._compress()

    def remember(self, fact):
        """将关键信息存入长期记忆"""
        if fact not in self.key_facts:
            self.key_facts.append(fact)

    def get_context(self):
        """获取当前完整上下文"""
        context = []
        if self.key_facts:
            context.append({"role": "system", "content": "[长期记忆]\n" + "\n".join(f"- {f}" for f in self.key_facts)})
        context.extend(self.history)
        return context

    def _compress(self):
        """压缩早期历史：保留最近 2/3，将前 1/3 替换为摘要"""
        split = len(self.history) // 3
        old = self.history[:split]
        self.history = self.history[split:]
        summary = f"[历史摘要: 前 {len(old)} 轮对话已压缩]"
        self.history.insert(0, {"role": "system", "content": summary})
```

## 第三阶段：自我修正

前两阶段构建的 Agent 假设工具调用总是顺利的。但在实际执行中，搜索可能返回不相关内容，代码执行可能因为语法错误失败，Agent 的推理可能走到死胡同。这些问题在单次调用中看似小概率，但在需要多轮交互的调研任务中，每一步都完美的概率是每步成功率的乘积，很快就会变得很低。自我修正不是锦上添花，而是让 Agent 在真实环境中能持续推进任务的必要机制。

Agent 执行中的错误可以分为三类，需要不同的处理策略。参数格式错误（JSON 解析失败、缺少必选字段）修正成本极低，修正参数后重试通常就能解决。工具执行失败（代码运行报错、文件不存在）需要 Agent 根据错误信息调整输入内容。逻辑错误（搜索方向跑偏、代码不报错但结果不对）最隐蔽也最难检测，往往需要事实交叉验证。下面的 `SelfCorrector` 按照代价递增原则组织修正策略，先尝试代价最低的参数修正，不行再升级到简化参数和切换工具。

```python runnable gpuonly extract-class="SelfCorrector"
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
```

## 第四阶段：多 Agent 协作

前三阶段构建的单 Agent 承担了搜索资料、编写代码、验证正确性、撰写报告所有职责。当调研任务的复杂度上升。譬如需要对比五个算法而不是三个，需要在多个数据规模上进行基准测试，需要引用学术文献并标注来源——单 Agent 的弱点就暴露出来了。它在搜索时的深度不如专门的检索系统，在编码时不如专注代码质量的工具，在审查时容易漏过自己生成的错误。多 Agent 协作的思路是按任务的专业门槛将职责拆分，让各有所长的 Agent 各司其职，而不是让一个 Agent 面面俱到。

从调研任务的结构可以直接推导出三个角色。调研阶段需要广泛搜集资料、判断来源可信度、提取技术要点，这需要信息检索和分析能力强的 Researcher。编码阶段需要将算法描述转化为正确可运行的代码并执行测试，这需要编程能力强的 Coder。审查阶段需要交叉验证报告中的数据、检查代码逻辑、确认结论与实验数据一致，这需要细心且持怀疑态度的 Reviewer。三个 Agent 通过消息总线进行通信，消息总线提供点对点消息传递，每条消息携带关联 ID 用于请求和响应的匹配。

```python runnable gpuonly extract-class="SpecializedAgent, AgentMessage, MessageBus"
# 专业化 Agent 及消息总线
import time

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
```

每个专业化 Agent 的核心差异体现在三个地方：系统提示词中的角色定义决定了 LLM 的行为倾向，可用工具集决定了 Agent 能做什么，`_execute` 方法中的领域逻辑决定了如何处理任务。这三个差异点让 Researcher 在搜索和整理信息上更专注，让 Coder 在编写和测试代码上更可靠，让 Reviewer 在发现问题和验证事实上更挑剔。

## 第五阶段：编排与容错

有了三个各有所长的 Agent，还需要一个编排器来协调它们的工作。编排器负责将调研目标分解为结构化的子任务、根据任务类型分配给合适的 Agent、收集整合各 Agent 的执行结果。它本身不执行具体工作，而是确保整体流程按合理的顺序推进。调研任务的各阶段存在线性依赖（必须在理解算法后才能写代码，必须在代码调通后才能审查），这种"A 的输出是 B 的输入"的结构最适合用管道编排模式。

```python runnable gpuonly extract-class="Orchestrator"
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
```

多 Agent 场景中，任何一个 Agent 都可能因为 LLM API 暂时不可用、工具调用超时或推理陷入循环而失败。管道编排的线性依赖意味着上游故障会阻塞所有下游任务。`FaultHandler` 提供断路器和超时两种基础保护。断路器在 Agent 连续失败达到阈值时自动切断任务分配，给故障组件留出恢复时间，避免在已知会失败的操作上浪费资源。

```python runnable gpuonly extract-class="FaultHandler"
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
```

## 集成测试

用真实的调研任务对单 Agent 和多 Agent 两种方案进行端到端测试。任务是对比快速排序和归并排序的性能，提供 Python 实现和基准测试，生成技术报告，覆盖搜索、编码、测试和报告生成四个阶段。测试会真正启动 Qwen3.5-0.8B-Instruct 模型进行推理，并通过 AnySearch API 执行真实的互联网搜索。

### 单 Agent 端到端测试

下面的测试注册了完整的调研工具集（互联网搜索、代码执行、文件读写），然后将它们交给 `AgentCore`，让 LLM 自主完成从搜索资料到生成报告的完整调研流程。这是对 Agent 核心循环、工具调用、记忆管理和提示词工程的一次全面验证。

```python runnable gpuonly
# 单 Agent 端到端测试：LLM + 真实工具 + 报告生成
from shared.agent_systems.tool_registry import ToolRegistry
from shared.agent_systems.memory_manager import MemoryManager
from shared.agent_systems.agent_core import AgentCore
import os, json, subprocess, requests, time

# ---- 注册调研工具 ----
tools = ToolRegistry()

@tools.register(
    name="search",
    description="搜索互联网获取技术资料。适用于查找算法原理、技术文档等信息。返回搜索结果摘要。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "description": "返回结果数量上限", "default": 5}
        },
        "required": ["query"]
    }
)
def search_tool(query, max_results=5):
    api_key = os.environ.get("ANYSEARCH_API_KEY", "")
    headers = {"Content-Type": "application/json", "X-Anysearch-Client": "dmla-agent/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": query, "max_results": min(max_results, 10)}}
    }
    try:
        resp = requests.post("https://api.anysearch.com/mcp", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return {"query": query, "error": data["error"].get("message", str(data["error"])), "results": ""}
        content = data.get("result", {}).get("content", [])
        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        results_text = "\n".join(text_parts) if text_parts else str(data.get("result", ""))
        return {"query": query, "results": results_text[:5000], "count": len(text_parts)}
    except requests.exceptions.ConnectionError:
        return {"query": query, "error": "无法连接到搜索服务", "results": ""}
    except requests.exceptions.Timeout:
        return {"query": query, "error": "搜索请求超时", "results": ""}
    except Exception as e:
        return {"query": query, "error": f"搜索异常: {str(e)}", "results": ""}

@tools.register(
    name="execute_code",
    description="执行 Python 代码并返回标准输出。适用于验证算法实现、运行基准测试、检查代码正确性。每次调用是独立的。",
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "待执行的 Python 代码"}},
        "required": ["code"]
    }
)
def execute_code_tool(code):
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return {"output": output, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"error": "代码执行超时（30 秒）"}

@tools.register(
    name="write_file",
    description="将内容写入文件。适用于保存调研笔记、代码草稿和最终报告。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "要写入的内容"}
        },
        "required": ["path", "content"]
    }
)
def write_file_tool(path, content):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"written": path, "size": len(content)}

@tools.register(
    name="read_file",
    description="读取文件内容。适用于查看之前保存的笔记、代码或报告草稿。",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "文件路径"}},
        "required": ["path"]
    }
)
def read_file_tool(path):
    if not os.path.exists(path):
        return {"error": f"文件不存在: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content, "size": len(content)}

print(f"已注册工具: {[s['name'] for s in tools.get_schemas()]}")

# ---- 执行调研任务 ----
print("\n" + "=" * 60)
print("单 Agent 端到端调研")
print("=" * 60)

memory = MemoryManager(max_history=20)
agent = AgentCore(tools, memory, max_iterations=10)

goal = "对比快速排序和归并排序的性能，用 Python 实现两种算法并运行基准测试，生成技术报告"
print(f"调研目标: {goal}")
print("正在加载 Qwen3.5-0.8B-Instruct 模型...")

start_time = time.time()
report = agent.run(goal)
elapsed = time.time() - start_time

print(f"\n调研完成，耗时 {elapsed:.0f} 秒，共使用 {len(memory.history)} 条对话记录")
print("\n" + "-" * 40)
print("调研报告")
print("-" * 40)
print(report)

# 保存报告
report_path = os.path.join(DATA_DIR, "outputs", "single_agent_report.md")
tools.execute("write_file", path=report_path, content=report)
print(f"\n报告已保存至 {report_path}")
```

### 多 Agent 端到端测试

下面的测试将同一个调研任务交给多 Agent 协作系统。`Planner` 将目标分解为搜索、编码、测试、报告四个子任务，`Orchestrator` 按任务类型分配给 Researcher 和 Coder 两个专业化 Agent。每个 Agent 内部使用独立的 `AgentCore` 实例驱动 LLM 推理，通过 `MessageBus` 与编排器通信。`FaultHandler` 监控各 Agent 的执行状态，在连续失败时触发断路保护。

```python runnable gpuonly
# 多 Agent 端到端测试：编排器 + 专业化 Agent + LLM + 报告生成
from shared.agent_systems.tool_registry import ToolRegistry
from shared.agent_systems.memory_manager import MemoryManager
from shared.agent_systems.agent_core import AgentCore
from shared.agent_systems.specialized_agent import SpecializedAgent, MessageBus, AgentMessage
from shared.agent_systems.planner import Planner
from shared.agent_systems.fault_handler import FaultHandler
from shared.agent_systems.orchestrator import Orchestrator
import os, json, subprocess, requests, time

# ---- 注册调研工具（与单 Agent 测试相同的工具集）----
tools = ToolRegistry()

@tools.register(
    name="search",
    description="搜索互联网获取技术资料。返回搜索结果摘要。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "description": "返回结果数量上限", "default": 5}
        },
        "required": ["query"]
    }
)
def search_tool(query, max_results=5):
    api_key = os.environ.get("ANYSEARCH_API_KEY", "")
    headers = {"Content-Type": "application/json", "X-Anysearch-Client": "dmla-agent/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": query, "max_results": min(max_results, 10)}}
    }
    try:
        resp = requests.post("https://api.anysearch.com/mcp", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return {"query": query, "error": data["error"].get("message", str(data["error"])), "results": ""}
        content = data.get("result", {}).get("content", [])
        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        results_text = "\n".join(text_parts) if text_parts else str(data.get("result", ""))
        return {"query": query, "results": results_text[:5000], "count": len(text_parts)}
    except requests.exceptions.ConnectionError:
        return {"query": query, "error": "无法连接到搜索服务", "results": ""}
    except requests.exceptions.Timeout:
        return {"query": query, "error": "搜索请求超时", "results": ""}
    except Exception as e:
        return {"query": query, "error": f"搜索异常: {str(e)}", "results": ""}

@tools.register(
    name="execute_code",
    description="执行 Python 代码并返回标准输出。每次调用是独立的。",
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "待执行的 Python 代码"}},
        "required": ["code"]
    }
)
def execute_code_tool(code):
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return {"output": output, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"error": "代码执行超时（30 秒）"}

@tools.register(
    name="write_file",
    description="将内容写入文件。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "要写入的内容"}
        },
        "required": ["path", "content"]
    }
)
def write_file_tool(path, content):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"written": path, "size": len(content)}

@tools.register(
    name="read_file",
    description="读取文件内容。",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "文件路径"}},
        "required": ["path"]
    }
)
def read_file_tool(path):
    if not os.path.exists(path):
        return {"error": f"文件不存在: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content, "size": len(content)}

print(f"已注册工具: {[s['name'] for s in tools.get_schemas()]}")

# ---- 创建多 Agent 协作系统 ----
print("\n" + "=" * 60)
print("多 Agent 端到端调研")
print("=" * 60)

bus = MessageBus()
planner = Planner()
fault_handler = FaultHandler(failure_threshold=3, recovery_timeout=30)

# 创建专业化 Agent
researcher = SpecializedAgent("researcher", "研究员",
    "负责搜索技术资料、筛选可靠来源、提取关键信息。使用 search 工具查找资料，用 write_file 保存中间产物。",
    tools, bus)
coder = SpecializedAgent("coder", "工程师",
    "负责将算法描述转化为可运行代码、执行基准测试、验证结果正确性。使用 execute_code 工具运行代码。",
    tools, bus)

# 为每个 Agent 配备独立的 AgentCore 以驱动 LLM 推理
print("正在为各 Agent 加载 Qwen3.5-0.8B-Instruct 模型...")

researcher_core = AgentCore(tools, MemoryManager(max_history=15), max_iterations=8)
coder_core = AgentCore(tools, MemoryManager(max_history=15), max_iterations=8)

# 将 AgentCore 绑定到专业化 Agent 的 _execute 方法
# 为每个角色提供不同的系统提示词，引导 LLM 专注于各自的专业领域
def make_execute(core, role_prompt):
    """创建使用 AgentCore（LLM）处理任务的 _execute 方法"""
    def _execute(task):
        task_desc = task.get("description", str(task))
        # 为每个子任务创建独立的记忆上下文，避免跨任务的信息污染
        memory = MemoryManager(max_history=15)
        memory.add("system", role_prompt)
        core.memory = memory
        return core.run(task_desc)
    return _execute

researcher._execute = make_execute(researcher_core,
    "你是技术研究员。使用 search 工具查找资料，整理关键信息，将发现保存到文件后返回。"
    "每找到一个重要信息就记录到关键事实中。任务完成后，返回结构化的调研笔记。")
coder._execute = make_execute(coder_core,
    "你是算法工程师。使用 execute_code 工具编写和运行代码，确保输出正确。"
    "先写实现代码，再写基准测试，最后汇总性能对比数据。任务完成后返回完整的代码和测试结果。")

# 创建编排器
orchestrator = Orchestrator(bus, [researcher, coder], planner, fault_handler)

# 修复单线程环境下的消息分发：编排器发送消息后需要手动触发 Agent 处理
agents = {"researcher": researcher, "coder": coder}
original_assign = orchestrator._assign
def assign_with_dispatch(task, agent_id):
    original_assign(task, agent_id)
    msg = bus.receive(agent_id)
    if msg:
        agents[agent_id].process(msg)
orchestrator._assign = assign_with_dispatch

# ---- 执行多 Agent 调研 ----
goal = "对比快速排序和归并排序的性能，用 Python 实现两种算法并运行基准测试，生成技术报告"
print(f"调研目标: {goal}")

start_time = time.time()
multi_result = orchestrator.execute(goal)
elapsed = time.time() - start_time

# ---- 输出结果 ----
print(f"\n调研完成，耗时 {elapsed:.0f} 秒")
print(f"完成情况: {multi_result['metadata']['completed']}/{multi_result['metadata']['steps']} 个子任务")

print("\n" + "-" * 40)
print("多 Agent 调研报告")
print("-" * 40)
print(f"标题: {multi_result['title']}")
for section in multi_result["sections"]:
    print(section)

# 保存报告
report_text = f"# {multi_result['title']}\n\n"
for section in multi_result["sections"]:
    report_text += section + "\n\n"

report_path = os.path.join(DATA_DIR, "outputs", "multi_agent_report.md")
tools.execute("write_file", path=report_path, content=report_text)
print(f"\n报告已保存至 {report_path}")
```

## 实验总结

在同一个调研任务上对比两种方案，各自有其适用边界。单 Agent 的优势在于结构简单、没有通信延迟和编排开销。搜索快速排序原理并总结这种小任务用单 Agent 效率最高。当任务涉及多个不同专业领域时（既要懂算法理论又要能写出正确代码还要会审查质量），单 Agent 的博而不精的问题就开始显现。多 Agent 的优势在于专业化深度，每个 Agent 只需在自己的领域内做到最好。但这种专业化也带来了成本，如 Agent 间的消息传递延迟、编排器的协调负担、某个 Agent 失败时的级联影响。因此，选择哪种方案的判断标准不是看 Agent 数量多少，而是任务的复杂度是否超过了单个 Agent 的专业能力范围。如果一个任务的多个阶段需要的知识和技能没有显著差异，强行拆分反而增加不必要的复杂度。

当前实现有几个值得注意的局限。编排器使用基于规则的固定分解策略，面对超出预设规则的任务类型时缺乏灵活性，改进方向是引入 LLM 驱动的动态任务分解。Agent 之间的通信是同步的点对点模式，限制了并行执行的能力；引入异步消息和扇出-扇入编排可以让无依赖的子任务同时执行。容错机制目前只覆盖了超时和断路，缺失检查点恢复，这意味着如果系统在任务中途崩溃，所有进度都会丢失。此外，代码执行在本地环境中直接运行，生产环境需要隔离在沙箱中以防范安全风险。