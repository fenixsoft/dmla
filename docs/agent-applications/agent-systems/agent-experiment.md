# 工程实训：技术调研 Agent 协作系统

本次工程实训中，笔者将与你一同构建一个能自动完成技术调研任务的 Agent 系统。输入一个调研需求（如"对比三种排序算法的性能"），系统会自主完成资料搜索、代码编写、基准测试和报告生成。我们将从零开始，先搭建一个能调用工具的简单 Agent，再逐步加入规划、记忆和自我修正能力，最后将任务按专业领域拆解给多个 Agent 协作完成。在此过程中，你可以体会到 Agent 系统中每个组件的引入不是来自预先的架构设计，而是来自任务本身提出的切实需求。

## 实验目标

- 构建一个能完成多步骤技术调研任务的自主 Agent，理解工具调用、ReAct 循环和提示词工程在实际任务中的运作方式
- 实现任务规划与记忆管理，让 Agent 能分解复杂目标并记住中间产物，避免在长任务中丢失上下文
- 实现自我修正机制，让 Agent 在搜索、编码和报告生成各阶段遇到错误时能自动调整策略
- 构建多 Agent 协作系统，按调研任务的三个专业领域（信息检索、代码实现、质量审查）拆分为三个专业化 Agent
- 为多 Agent 系统加入编排和基础容错能力，确保协作流程在部分 Agent 故障时仍能推进

## 实验准备

### 环境与依赖

- Python 3.10+ 运行环境
- 可访问的 LLM API（支持函数调用功能）
- 实验所需的 Python 包：`openai`（或兼容 API 客户端）、`jsonschema`（参数验证）

```python runnable
# 验证实验环境
import sys
print(f"Python 版本: {sys.version}")

try:
    import jsonschema
    try:
        from importlib.metadata import version
        print(f"jsonschema: {version('jsonschema')}")
    except ImportError:
        print(f"jsonschema: {jsonschema.__version__}")
except ImportError:
    print("jsonschema: 未安装，本实验中的参数验证将使用简化实现")

try:
    import openai
    print(f"openai: {openai.__version__}")
except ImportError:
    print("openai: 未安装，请通过 pip install openai 安装（或使用兼容的 API 客户端）")
```

### 实验架构概览

本实验按任务的演进顺序组织为五个阶段。第一阶段从分析调研任务的具体需求出发，确定 Agent 需要哪些工具，构建一个能完成简单搜索和总结的单 Agent。第二阶段引入规划和记忆，让 Agent 能处理"搜索多个子主题→筛选信息→编码→测试→生成报告"这类多步骤复杂任务。第三阶段加入自我修正，应对搜索不相关、代码报错等真实执行中的异常。第四阶段分析单 Agent 在调研-编码-审查三个专业门槛上的瓶颈，将系统拆分为 Researcher、Coder、Reviewer 三个专业化 Agent。第五阶段为多 Agent 系统加入编排和容错能力，让协作可靠运行。最后用真实的调研任务对单 Agent 和多 Agent 两种方案进行端到端测试和对比。

## 第一阶段：构建基础 Agent

技术调研任务要求 Agent 依次完成：理解调研主题、搜索相关资料、筛选和整理信息、编写示例代码、运行代码验证、将结果整合为报告。Agent 至少需要三种能力：获取外部信息（搜索）、执行代码并观察结果、读写文件以保存中间产物。我们先实现这些工具，再围绕它们构建 Agent 的决策循环。

### 工具注册中心

工具是 Agent 接触外部世界的接口。`ToolRegistry` 提供了一套统一的机制来管理工具的注册、描述和调用。每个工具注册时需要提供名称、功能描述和参数的 JSON Schema，LLM 通过读取这些 Schema 来理解何时以及如何使用每个工具。调用时，注册中心自动验证必选参数并捕获执行异常，确保单个工具的失败不会直接导致 Agent 崩溃。

```python runnable extract-class="ToolRegistry"
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

接下来为我们的调研任务注册四个工具。搜索工具负责获取外部资料，代码执行工具负责运行 Python 代码并收集输出，文件读写工具负责保存中间产物和最终报告。注意每个工具的 `description` 是写给 LLM 看的，它决定了 LLM 选择工具的准确率——描述要准确说明工具的用途和适用场景，不能太笼统也不能太细节。

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
    return {"query": query, "results": f"已搜索 '{query}'，获得 {max_results} 条结果（生产环境应接入真实搜索 API）", "count": max_results}

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

有了工具，Agent 还需要一个决策中枢来决定什么时候调用哪个工具。我们采用 ReAct 模式：每轮循环中，Agent 先思考当前处境和下一步策略，然后选择工具执行，观察执行结果，再根据观察更新思考，如此循环直到任务完成。这个"行动→观察→调整"的闭环正是 Agent 区别于单次问答的关键——它能根据实际执行反馈动态调整策略，而不是一条路走到黑。

```python runnable extract-class="AgentCore"
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
```

## 第二阶段：规划与记忆

第一阶段的 Agent 能完成"搜索一个主题并总结"这样的简单任务。但真实的技术调研需要搜索多个子主题、筛选对比来源、编写和测试代码、将分散的发现组织成结构化报告。这些子任务之间存在依赖关系（代码实现依赖于对算法的理解，基准测试依赖于代码已调通），如果 Agent 想到哪做到哪，很容易遗漏关键步骤。此外，多轮对话产生的大量中间信息（搜索结果、代码片段、测试数据）会迅速超出 LLM 的上下文窗口，早期的重要信息一旦被裁剪就永久丢失。

### 任务规划器

规划器的职责是将高层目标分解为结构化的子任务序列。下面的 `Planner` 使用基于规则的分解策略，根据目标中的关键词判断需要哪些子任务，生成一个有序的任务列表。任务之间形成有向无环的依赖关系，每个步骤的产出是下一步骤的输入，确保不遗漏关键环节。

```python runnable extract-class="Planner"
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

```python runnable extract-class="MemoryManager"
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

### 错误分类与修正策略

Agent 执行中的错误可以分为三类，需要不同的处理策略。参数格式错误（JSON 解析失败、缺少必选字段）修正成本极低，修正参数后重试通常就能解决。工具执行失败（代码运行报错、文件不存在）需要 Agent 根据错误信息调整输入内容。逻辑错误（搜索方向跑偏、代码不报错但结果不对）最隐蔽也最难检测，往往需要事实交叉验证。下面的 `SelfCorrector` 按照"代价递增"原则组织修正策略：先尝试代价最低的参数修正，不行再升级到简化参数和切换工具。

```python runnable extract-class="SelfCorrector"
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

前三阶段构建的单 Agent 承担了搜索资料、编写代码、验证正确性、撰写报告所有职责。当调研任务的复杂度上升——比如需要对比五个算法而不是三个，需要在多个数据规模上进行基准测试，需要引用学术文献并标注来源——单 Agent 的弱点就暴露出来了。它在搜索时的深度不如专门的检索系统，在编码时不如专注代码质量的工具，在审查时容易漏过自己生成的错误。多 Agent 协作的思路是按任务的专业门槛将职责拆分，让各有所长的 Agent 各司其职，而不是让一个 Agent 面面俱到。

### 专业化 Agent 与通信协议

从调研任务的结构可以直接推导出三个角色。调研阶段需要广泛搜集资料、判断来源可信度、提取技术要点，这需要信息检索和分析能力强的 Researcher。编码阶段需要将算法描述转化为正确可运行的代码并执行测试，这需要编程能力强的 Coder。审查阶段需要交叉验证报告中的数据、检查代码逻辑、确认结论与实验数据一致，这需要细心且持怀疑态度的 Reviewer。三个 Agent 通过消息总线进行通信，消息总线提供点对点消息传递，每条消息携带关联 ID 用于请求和响应的匹配。

```python runnable extract-class="SpecializedAgent, AgentMessage, MessageBus"
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

```python runnable extract-class="Orchestrator"
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

### 基础容错

多 Agent 场景中，任何一个 Agent 都可能因为 LLM API 暂时不可用、工具调用超时或推理陷入循环而失败。管道编排的线性依赖意味着上游故障会阻塞所有下游任务。`FaultHandler` 提供断路器和超时两种基础保护。断路器在 Agent 连续失败达到阈值时自动切断任务分配，给故障组件留出恢复时间，避免在已知会失败的操作上浪费资源。

```python runnable extract-class="FaultHandler"
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

用真实的调研任务对单 Agent 和多 Agent 两种方案进行端到端测试。任务是"对比快速排序和归并排序的性能，提供 Python 实现和基准测试，生成技术报告"，覆盖搜索、编码、测试和报告生成四个阶段。

```python runnable
# 集成测试：验证各组件的协同工作
from shared.agent_systems.tool_registry import ToolRegistry
from shared.agent_systems.memory_manager import MemoryManager
from shared.agent_systems.agent_core import AgentCore
from shared.agent_systems.specialized_agent import SpecializedAgent, MessageBus, AgentMessage
from shared.agent_systems.planner import Planner
from shared.agent_systems.fault_handler import FaultHandler
from shared.agent_systems.orchestrator import Orchestrator
from shared.agent_systems.self_corrector import SelfCorrector
import time

# 1. 验证工具注册与执行
print("=== 测试 1: 工具注册与执行 ===")
tools = ToolRegistry()

@tools.register(name="greet", description="返回问候语",
    parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
def greet(name):
    return f"你好，{name}！"

result = tools.execute("greet", name="DMLA")
assert "error" not in result, f"工具执行失败: {result}"
print(f"工具调用结果: {result['result']}")

# 2. 验证任务规划
print("\n=== 测试 2: 任务规划 ===")
planner = Planner()
tasks = planner.decompose("对比快速排序和归并排序的性能，提供 Python 实现和基准测试")
print(f"规划结果: {len(tasks)} 个子任务")
for t in tasks:
    print(f"  - [{t['action']}] {t['description']}")

# 3. 验证记忆管理
print("\n=== 测试 3: 记忆管理 ===")
memory = MemoryManager(max_history=5)
memory.add("user", "调研排序算法")
memory.add("observation", "快速排序平均复杂度 O(n log n)")
memory.remember("快速排序：分治策略，平均 O(n log n)，最坏 O(n²)")
context = memory.get_context()
print(f"上下文条目数: {len(context)}")
print(f"关键事实数: {len(memory.key_facts)}")

# 4. 验证 Agent 消息通信
print("\n=== 测试 4: Agent 消息通信 ===")
bus = MessageBus()
researcher = SpecializedAgent("researcher", "研究员",
    "负责搜索技术资料。", tools, bus)
coder = SpecializedAgent("coder", "工程师",
    "负责编写代码。", tools, bus)

# 直接触发 Agent 处理任务消息
task_msg = AgentMessage(
    msg_type="task_assign",
    sender="orchestrator",
    receiver="researcher",
    payload={"description": "搜索快速排序的资料"},
    correlation_id="test_001"
)
bus.send(task_msg)
msg = bus.receive("researcher")
if msg:
    researcher.process(msg)
print(f"Researcher 状态: {researcher.status}")

# 5. 验证编排器
print("\n=== 测试 5: 编排器 ===")
orchestrator = Orchestrator(bus, [researcher, coder], planner, FaultHandler())
result = orchestrator._select_agent("research")
print(f"research 任务分配给: {result}")
assert result == "researcher", f"分配错误: {result}"

# 6. 验证自我修正
print("\n=== 测试 6: 自我修正 ===")
corrector = SelfCorrector(tools)
result = corrector.correct("greet", {"wrong_param": "test"}, "缺少必选参数 'name'")
print(f"修正结果: {result['success']}")

print("\n=== 全部测试通过 ===")
```

### 单 Agent 与多 Agent 的对比

在同一个调研任务上对比两种方案，各自有其适用边界。单 Agent 的优势在于结构简单、没有通信延迟和编排开销。"搜索快速排序原理并总结"这种小任务用单 Agent 效率最高。当任务涉及多个不同专业领域时（既要懂算法理论又要能写出正确代码还要会审查质量），单 Agent 的"博而不精"就开始拖后腿。多 Agent 的优势在于专业化深度，每个 Agent 只需在自己的领域内做到最好。但这种专业化也带来了成本：Agent 间的消息传递延迟、编排器的协调负担、某个 Agent 失败时的级联影响。

选择哪种方案的判断标准不是"多 Agent 更先进"，而是任务的复杂度是否超过了单个 Agent 的专业能力范围。如果一个任务的多个阶段需要的知识和技能没有显著差异，强行拆分反而增加不必要的复杂度。

### 局限性与改进方向

当前实现有几个值得注意的局限。编排器使用基于规则的固定分解策略，面对超出预设规则的任务类型时缺乏灵活性，改进方向是引入 LLM 驱动的动态任务分解。Agent 之间的通信是同步的点对点模式，限制了并行执行的能力；引入异步消息和扇出-扇入编排可以让无依赖的子任务同时执行。容错机制目前只覆盖了超时和断路，缺失检查点恢复，这意味着如果系统在任务中途崩溃，所有进度都会丢失。此外，代码执行在本地环境中直接运行，生产环境需要隔离在沙箱中以防范安全风险。