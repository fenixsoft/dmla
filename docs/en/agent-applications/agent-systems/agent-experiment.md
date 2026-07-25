# Hands-on Lab: Technical Research Agent Collaboration System

In this hands-on lab, we will build an Agent system capable of automatically completing technical research tasks together. Given a research requirement (such as "compare the performance of three sorting algorithms"), the system will autonomously handle information retrieval, code writing, benchmark testing, and report generation. We will start from scratch, first building a simple tool-using Agent, then gradually adding planning, memory, and self-correction capabilities, and finally decomposing the task by domain expertise for collaborative work among multiple Agents. Throughout this process, you will see that each component in the Agent system is introduced not from a pre-designed architecture, but from the practical demands of the task itself.

## Prerequisites

This lab is organized into five phases following the evolutionary order of the tasks:
- **Phase 1**: Starting from analyzing the concrete requirements of the research task, determine what tools the Agent needs, build a single Agent capable of simple search and summarization, and understand how tool calling, the ReAct loop, and prompt engineering work in practice.
- **Phase 2**: Introduce planning and memory, enabling the Agent to handle multi-step complex tasks such as "search multiple subtopics → filter information → code → test → generate report," allowing the Agent to decompose complex goals and remember intermediate artifacts, avoiding context loss in long tasks.
- **Phase 3**: Add self-correction to handle real-world execution anomalies such as irrelevant search results and code errors, enabling the Agent to automatically adjust strategies when encountering errors during search, coding, and report generation.
- **Phase 4**: Analyze the bottlenecks of a single Agent across the three professional thresholds of research, coding, and review, splitting the system into three specialized Agents: Researcher, Coder, and Reviewer.
- **Phase 5**: Add orchestration and fault tolerance to the multi-Agent system, ensuring reliable collaborative operation. Finally, run end-to-end tests comparing the single-agent and multi-agent approaches on a real research task.

Before starting the lab, please ensure you have completed the following preparations:

1. Download the [Qwen3.5-0.8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B-Instruct) language model.

```bash
# Select "Download Model" -> Select "Qwen3.5-0.8B-Instruct"
dmla model
```

2. (Optional but recommended) Register an **AnySearch API Key**. The search tool accesses internet search results via the [AnySearch](https://anysearch.com) API. If no API Key is set, it runs in anonymous mode (with rate limiting, though basic usage is not affected). For higher search frequency, please visit the [AnySearch Console](https://anysearch.com/console/api-keys) to create a free API Key, then set the environment variable before starting the lab:

```bash
export ANYSEARCH_API_KEY="your_API_key"
```

The lab dependencies are pre-installed in the DMLA sandbox image. You can verify the GPU environment and model status with the following code:

```python gpuonly runnable
# Verify experiment environment and model download status
import os
import sys

print(f"Python version: {sys.version}")

# Check jsonschema package
try:
    import jsonschema
    print(f"jsonschema: installed")
except ImportError:
    print("jsonschema: not installed, please install via pip install jsonschema")

# Check if Qwen3.5-0.8B-Instruct model is downloaded
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
        print(f"Qwen3.5-0.8B-Instruct: downloaded ({size_mb:.0f} MB)")
    else:
        print("Qwen3.5-0.8B-Instruct: model files incomplete, please re-download via dmla model")
else:
    print("Qwen3.5-0.8B-Instruct: not found, please run dmla model and select Qwen3.5-0.8B-Instruct to download")
```

## Phase 1: Building a Basic Agent

A technical research task requires the Agent to sequentially understand the research topic, search for relevant materials, filter and organize information, write example code, run and verify the code, and synthesize the results into a report. The Agent needs at least three capabilities: obtaining external information (search), executing code and observing results, and reading/writing files to save intermediate artifacts. We will first implement these tools, then build the Agent's decision loop around them.

### Tool Registry

Tools are the Agent's interface to the external world. `ToolRegistry` provides a unified mechanism for managing tool registration, description, and invocation. Each tool, when registered, must provide a name, a functional description, and a JSON Schema for its parameters. The LLM reads these Schemas to understand when and how to use each tool. During invocation, the registry automatically validates required parameters and catches execution exceptions, ensuring that a single tool failure does not directly crash the Agent.

```python runnable gpuonly extract-class="ToolRegistry"
# Tool Registry: manages tool descriptions, registration, and execution
from functools import wraps

class ToolRegistry:
    """Tool Registry, manages available tools' registration, schema querying, and execution"""

    def __init__(self):
        self._tools = {}
        self._schemas = {}

    def register(self, name=None, description="", parameters=None):
        """Tool registration decorator"""
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
        """Get schemas for all registered tools, for the LLM to understand available tools"""
        return list(self._schemas.values())

    def execute(self, tool_name, **kwargs):
        """Execute the specified tool, automatically validate required params and catch exceptions"""
        if tool_name not in self._tools:
            return {"error": f"Tool '{tool_name}' does not exist", "available": list(self._tools.keys())}

        schema = self._schemas[tool_name]
        required = schema["parameters"].get("required", [])
        for param in required:
            if param not in kwargs:
                return {"error": f"Missing required parameter '{param}'"}

        try:
            result = self._tools[tool_name](**kwargs)
            return {"result": result}
        except Exception as e:
            return {"error": f"Tool execution exception: {str(e)}"}
```

### Registering Tools for the Research Task

Next, we register four tools for our research task. The search tool retrieves external materials, the code execution tool runs Python code and collects output, and the file read/write tools handle saving intermediate artifacts and the final report. Note that each tool's `description` is written for the LLM -- it determines the accuracy of the LLM's tool selection. The description should accurately state the tool's purpose and applicable scenarios, neither too vague nor too detailed.

```python runnable
# Register tools needed for the research task
import subprocess
import os
from shared.agent_systems.tool_registry import ToolRegistry

registry = ToolRegistry()

@registry.register(
    name="search",
    description="Search the internet for technical information. Suitable for finding algorithm principles, technical documentation, academic papers, etc. Returns a list of search result summaries.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keywords"},
            "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 5}
        },
        "required": ["query"]
    }
)
def search_tool(query, max_results=5):
    """Search the internet via AnySearch API, supporting both anonymous and API Key modes"""
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
        return {"query": query, "error": "Unable to connect to the search service, please check your network connection", "results": ""}
    except requests.exceptions.Timeout:
        return {"query": query, "error": "Search request timed out", "results": ""}
    except Exception as e:
        return {"query": query, "error": f"Search exception: {str(e)}", "results": ""}

@registry.register(
    name="execute_code",
    description="Execute Python code and return standard output. Suitable for verifying algorithm implementations, running benchmarks, and checking code correctness. Each call is independent.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"}
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
        return {"error": "Code execution timed out (30 seconds)"}

@registry.register(
    name="write_file",
    description="Write content to a file. Suitable for saving research notes, code drafts, and the final report.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "Content to write"}
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
    description="Read file content. Suitable for viewing previously saved notes, code, or report drafts.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"}
        },
        "required": ["path"]
    }
)
def read_file_tool(path):
    if not os.path.exists(path):
        return {"error": f"File does not exist: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content, "size": len(content)}

print(f"Registered tools: {[s['name'] for s in registry.get_schemas()]}")
```

### Agent Core Loop

With tools in place, the Agent needs a decision-making center to determine when to call which tool. We adopt the ReAct pattern. In each cycle, the Agent first thinks about its current situation and next strategy, selects a tool to execute, observes the result, then updates its thinking based on the observation, looping until the task is complete. This "act, observe, adjust" closed loop is what distinguishes an Agent from a single-turn Q&A system -- it dynamically adjusts its strategy based on actual execution feedback, rather than blindly following a fixed path.

```python runnable gpuonly extract-class="AgentCore"
# Agent Core Loop: implements the ReAct pattern's think-act-observe loop
import json

class AgentCore:
    """ReAct-based Agent core, managing the think-act-observe loop"""

    def __init__(self, tool_registry, memory_manager, max_iterations=10):
        self.tools = tool_registry
        self.memory = memory_manager
        self.max_iterations = max_iterations

    def run(self, goal):
        """Execute the main loop until task completion or max iterations reached"""
        self.memory.add("user", goal)
        self._search_count = 0

        for iteration in range(self.max_iterations):
            print(f"\n{'─' * 40}")
            print(f"Round {iteration + 1}/{self.max_iterations}")

            prompt = self._build_prompt(iteration)
            response = self._call_llm(prompt)
            thought, action, final_answer = self._parse_response(response)

            if thought:
                print(f"[Thought] {thought[:200]}{'...' if len(thought) > 200 else ''}")
                self.memory.add("thought", thought)

            if final_answer:
                final_answer = self._unescape_newlines(final_answer)
                print(f"[Done] Model determined task complete, generating final report")
                self.memory.add("answer", final_answer)
                return final_answer

            if action:
                tool_name = action.get("tool", "")
                params = action.get("parameters", {})
                param_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in params.items())
                print(f"[Call] {tool_name}({param_str})")
                observation = self.tools.execute(tool_name, **params)
                if tool_name == "search":
                    self._search_count += 1
                # Print execution result summary
                if "error" in observation:
                    print(f"[Result] Error: {str(observation['error'])[:200]}")
                elif "result" in observation:
                    print(f"[Result] Success: {str(observation['result'])[:200]}")
                else:
                    obs_str = json.dumps(observation, ensure_ascii=False)
                    print(f"[Result] {obs_str[:200]}")
                self.memory.add("observation", json.dumps(observation, ensure_ascii=False))
            else:
                # When no valid action is produced, check if the model expressed intent to generate a report
                done_keywords = ["completed", "information sufficient", "can complete", "task complete", "report below", "output report now"]
                if thought and any(kw in thought.lower() for kw in done_keywords):
                    print(f"[Intent] Model determined report is ready, attempting output...")
                    fallback_prompt = self._build_fallback_prompt()
                    if fallback_prompt:
                        response = self._call_llm(fallback_prompt)
                        _, _, final_answer = self._parse_response(response)
                        if final_answer:
                            final_answer = self._unescape_newlines(final_answer)
                            print(f"[Done] Fallback generated final report")
                            self.memory.add("answer", final_answer)
                            return final_answer
                    print(f"[Hint] Fallback failed to generate report, continuing to next round")
                else:
                    print(f"[Hint] No valid action parsed. Please output [Action] or [FinalAnswer] in the next round")
                self.memory.add("observation", "[Hint] No valid action generated in the previous step. Please generate [FinalAnswer] directly based on the information collected so far")

        # Reached max iterations, attempt forced report generation
        print(f"\n{'─' * 40}")
        print(f"Reached max iterations {self.max_iterations}, attempting fallback report generation...")
        fallback_prompt = self._build_fallback_prompt()
        if fallback_prompt:
            response = self._call_llm(fallback_prompt)
            _, _, final_answer = self._parse_response(response)
            if final_answer:
                final_answer = self._unescape_newlines(final_answer)
                print("[Fallback] Successfully generated partial report")
                return final_answer
            else:
                print("[Fallback] Failed to generate report")
        return "Reached maximum iterations. Task incomplete."

    def _build_fallback_prompt(self):
        """Build fallback prompt: force the model to generate a report based on existing information"""
        context = self.memory.get_context()
        history = "\n".join(f"[{m['role']}] {m['content']}" for m in context)
        return "\n".join([
            "You have completed multiple rounds of search and code execution. Now you must",
            "directly generate the final research report based on the information in the conversation history below.",
            "Do not output [Thought] or [Action], only output [FinalAnswer].",
            "",
            "Conversation history:",
            history,
            "",
            "[FinalAnswer]",
        ])

    def _build_prompt(self, iteration=0):
        """Build the complete prompt to send to the LLM"""
        context = self.memory.get_context()
        history = "\n".join(
            f"[{m['role']}] {self._truncate(m['content'], 400 if m['role'] != 'thought' else 80)}"
            for m in context
        )

        # Select available tools, phase guidance, and output format based on current round
        remaining = self.max_iterations - iteration
        if iteration < 3:
            # Search phase: [FinalAnswer] not allowed, must search first
            tools_desc = json.dumps(self.tools.get_schemas(), ensure_ascii=False, indent=2)
            phase_hint = "Search phase: You must search first! Use English keywords to search for algorithm materials."
            action_hint = '[Action] {"tool":"search","parameters":{"query":"keywords"}}'
            final_hint = None
        elif iteration < 7:
            # Coding phase: remove search tool
            coding_schemas = [s for s in self.tools.get_schemas() if s["name"] != "search"]
            tools_desc = json.dumps(coding_schemas, ensure_ascii=False, indent=2)
            phase_hint = (
                "Coding phase: Search is disabled. You must use execute_code to write and run Python code."
                "The code should include:\n"
                "1. quick_sort and merge_sort functions\n"
                "2. Test code that sorts a random array\n"
                "3. Use the time module to measure the runtime of both algorithms\n"
                "Now immediately call execute_code to run the above code!"
            )
            action_hint = '[Action] {"tool":"execute_code","parameters":{"code":"import time,random\\narr=[random.randint(0,1000) for _ in range(100)]\\nt=time.time();print(sorted(arr)[:10]);print(time.time()-t)"}}'
            final_hint = (
                "[FinalAnswer] # Research Report\n\n"
                "## Algorithm Principles\n(Explain the principles of quicksort and merge sort, list time complexities)\n\n"
                "## Code Implementation\n```python\n(Python implementations of both algorithms)\n```\n\n"
                "## Performance Comparison\n| Algorithm | Time Complexity | Runtime |\n|-----------|-----------------|---------|\n| ... | ... | ... |"
            )
        else:
            # Report phase: remove all tools, only allow [FinalAnswer]
            phase_hint = f"Report phase: Only {remaining} rounds left. You must output [FinalAnswer] with the complete report immediately."
            tools_desc = "(All tools disabled)"
            action_hint = None
            final_hint = (
                "[FinalAnswer] # Research Report\n\n"
                "## Algorithm Principles\n(Explain the principles of quicksort and merge sort, list time complexities)\n\n"
                "## Code Implementation\n```python\n(Python implementations of both algorithms)\n```\n\n"
                "## Performance Comparison\n| Algorithm | Time Complexity | Runtime |\n|-----------|-----------------|---------|\n| ... | ... | ... |"
            )

        lines = [
            "You are a technical research assistant. Complete the research goal and output a report containing algorithm descriptions, code implementations, and performance comparisons.",
            "",
            f"[Phase Guidance] {phase_hint}",
            "",
            "Available tools:",
            tools_desc,
            "",
            "Output format (strictly follow this format):",
            "[Thought] Brief explanation of what is needed now",
        ]
        if action_hint:
            lines.append(action_hint)
        if final_hint:
            lines.extend(["", "When task is complete:", "[Thought] Task complete", final_hint])
        lines.extend(["", "Execution history:", history])

        return "\n".join(lines)

    @staticmethod
    def _unescape_newlines(text):
        """Replace literal \\n in LLM output with actual newlines"""
        return text.replace("\\n", "\n")

    @staticmethod
    def _truncate(text, max_len):
        """Truncate text to specified length, mark overflow with ellipsis"""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _call_llm(self, prompt):
        """Generate response using the local Qwen3.5-0.8B-Instruct model"""
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
        """Parse LLM response, extracting thought, action, and final answer"""
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
            # Use bracket matching to extract JSON object, correctly handling nested braces
            action = self._extract_json(action_part)

        if "[FinalAnswer]" in response:
            final_answer = response.split("[FinalAnswer]")[1].strip()

        return thought, action, final_answer

    @staticmethod
    def _extract_json(text):
        """Use bracket matching to extract the first complete JSON object from text, handling nested braces"""
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

## Phase 2: Planning and Memory

The Agent built in Phase 1 can handle simple tasks like searching for a single topic and summarizing it. But real technical research requires searching multiple subtopics, filtering and comparing sources, writing and testing code, and organizing scattered findings into a structured report. These subtasks have dependencies (code implementation depends on understanding the algorithm, benchmarking depends on working code). If the Agent acts haphazardly, it can easily miss critical steps. Moreover, the large volume of intermediate information generated across multiple rounds (search results, code snippets, test data) can quickly exceed the LLM's context window, permanently losing early important information once it is truncated.

### Task Planner

The planner's responsibility is to decompose high-level goals into a structured sequence of subtasks. The `Planner` below uses a rule-based decomposition strategy, determining which subtasks are needed based on keywords in the goal and generating an ordered task list. The tasks form a directed acyclic dependency graph, where the output of each step serves as the input for the next, ensuring no critical step is missed.

```python runnable gpuonly extract-class="Planner"
# Task Planner: decomposes high-level goals into a sequence of subtasks
class Planner:
    """Task Planner, responsible for goal decomposition and progress tracking"""

    def __init__(self):
        self.plan = []
        self.current_step = 0

    def decompose(self, goal):
        """Select decomposition strategy based on goal type, generating a list of subtasks"""
        keywords = goal.lower()
        tasks = []

        tasks.append({"id": "step_1", "action": "research", "description": "Search and organize core concepts and principles"})
        tasks.append({"id": "step_2", "action": "filter", "description": "Filter reliable sources and extract key information"})

        if "code" in keywords or "implement" in keywords:
            tasks.append({"id": "step_3", "action": "implement", "description": "Write implementation code based on research findings"})
            tasks.append({"id": "step_4", "action": "test", "description": "Run tests to verify code correctness"})

        if "benchmark" in keywords or "compare" in keywords:
            tasks.append({"id": "step_bench", "action": "benchmark", "description": "Design and run comparison experiments, collect performance data"})

        tasks.append({"id": "step_final", "action": "report", "description": "Synthesize all findings and code into the final report"})

        self.plan = tasks
        self.current_step = 0
        return tasks

    def next_task(self):
        """Return the next subtask to execute"""
        if self.current_step < len(self.plan):
            task = self.plan[self.current_step]
            self.current_step += 1
            return task
        return None

    def progress(self):
        """Return current execution progress"""
        total = len(self.plan)
        done = self.current_step
        return {"completed": done, "total": total, "percent": int(done / total * 100) if total > 0 else 0}
```

### Memory Manager

The memory manager maintains two types of information. Conversation history serves as the direct context for Agent reasoning, recording each round's user input, Agent thoughts, tool calls, and observations. When the history length exceeds a threshold, early records are compressed into a summary to control context window usage. Key facts are persistent information extracted from the conversation (technical details found, code snippets completed, performance data collected from tests) that exist independently of conversation rounds and can be retrieved at any time during later phases.

```python runnable gpuonly extract-class="MemoryManager"
# Memory Manager: maintains conversation history and key facts
class MemoryManager:
    """Memory Manager, maintains short-term conversation history and long-term key information"""

    def __init__(self, max_history=20):
        self.history = []
        self.key_facts = []
        self.max_history = max_history

    def add(self, role, content):
        """Add a record to the conversation history"""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self._compress()

    def remember(self, fact):
        """Store key information in long-term memory"""
        if fact not in self.key_facts:
            self.key_facts.append(fact)

    def get_context(self):
        """Get the complete current context"""
        context = []
        if self.key_facts:
            context.append({"role": "system", "content": "[Long-term Memory]\n" + "\n".join(f"- {f}" for f in self.key_facts)})
        context.extend(self.history)
        return context

    def _compress(self):
        """Compress early history: keep the most recent 2/3, replace the first 1/3 with a summary"""
        split = len(self.history) // 3
        old = self.history[:split]
        self.history = self.history[split:]
        summary = f"[History Summary: Previous {len(old)} rounds of conversation compressed]"
        self.history.insert(0, {"role": "system", "content": summary})
```

## Phase 3: Self-Correction

The Agent built in the first two phases assumes tool calls always succeed. But in practice, search may return irrelevant content, code execution may fail due to syntax errors, and the Agent's reasoning may hit dead ends. These issues may seem low-probability in a single call, but in a research task requiring multiple rounds of interaction, the probability of each step being perfect is the product of individual step success rates, quickly becoming very low. Self-correction is not a nice-to-have feature; it is a necessary mechanism for the Agent to keep progressing in real-world environments.

Errors during Agent execution can be divided into three types, each requiring a different handling strategy. Parameter format errors (JSON parsing failure, missing required fields) have very low correction cost -- simply fixing the parameters and retrying usually resolves them. Tool execution failures (code runtime errors, file not found) require the Agent to adjust its input based on the error message. Logic errors (research going off-track, code runs but produces wrong results) are the most subtle and hardest to detect, often requiring factual cross-verification. The `SelfCorrector` below organizes correction strategies by increasing cost: first try the cheapest parameter fix, then escalate to parameter simplification and tool switching if needed.

```python runnable gpuonly extract-class="SelfCorrector"
# Self-Correction Module: detects errors and selects correction strategy based on error type
class SelfCorrector:
    """Self-Correction Module, executes recovery strategies based on error type"""

    MAX_RETRIES = 3

    def __init__(self, tool_registry):
        self.registry = tool_registry
        self.error_history = []

    def correct(self, tool_name, params, error_message):
        """Analyze error type and attempt progressively escalating correction strategies"""
        self.error_history.append({
            "tool": tool_name, "params": params, "error": error_message
        })

        # Strategy 1: Parameter fix (for format errors)
        if self._is_format_error(error_message):
            fixed = self._fix_params(params, error_message)
            if fixed != params:
                return self._retry(tool_name, fixed)

        # Strategy 2: Simplify parameters and retry (for content errors)
        simplified = self._simplify_params(params)
        if simplified != params:
            result = self._retry(tool_name, simplified)
            if result.get("success"):
                return result

        # Strategy 3: Use alternative tool (when current tool is unavailable)
        alt = self._find_alternative(tool_name)
        if alt:
            result = self._retry(alt, params)
            if result.get("success"):
                return result

        return {"success": False, "error": "All correction strategies exhausted", "history": self.error_history[-self.MAX_RETRIES:]}

    def _is_format_error(self, error):
        fmt_keywords = ["json", "parse", "parameter", "format", "missing", "required", "type"]
        return any(kw in str(error).lower() for kw in fmt_keywords)

    def _fix_params(self, params, error):
        """Attempt to fix parameters (simplified implementation: pass original params and let the LLM decide how to adjust)"""
        return params

    def _simplify_params(self, params):
        """Simplify parameters: remove optional fields that may cause issues"""
        return {k: v for k, v in params.items() if v is not None}

    def _find_alternative(self, tool_name):
        """Find a functionally similar alternative tool"""
        alternatives = {
            "search": ["read_file"],
            "execute_code": [],
        }
        return alternatives.get(tool_name, [None])[0]

    def _retry(self, tool_name, params):
        """Execute retry and return the result"""
        result = self.registry.execute(tool_name, **params)
        success = "error" not in result
        return {"success": success, "result": result}
```

## Phase 4: Multi-Agent Collaboration

The single Agent built in the first three phases handles all responsibilities: searching materials, writing code, verifying correctness, and composing reports. As the research task grows in complexity -- for instance, comparing five algorithms instead of three, running benchmarks across multiple data scales, citing academic literature with proper attribution -- the weaknesses of a single Agent become apparent. Its search depth cannot match a dedicated retrieval system, its coding quality lags behind code-focused tools, and its review tends to overlook its own errors. The idea behind multi-agent collaboration is to split responsibilities according to the professional demands of each task, letting each specialized Agent focus on what it does best, rather than asking one Agent to be a jack-of-all-trades.

Three roles emerge naturally from the structure of the research task. The research phase requires extensive information gathering, source credibility assessment, and technical point extraction -- this calls for a Researcher with strong information retrieval and analysis capabilities. The coding phase requires translating algorithm descriptions into correct, runnable code and executing tests -- this calls for a Coder with strong programming skills. The review phase requires cross-validating data in the report, checking code logic, and confirming that conclusions align with experimental data -- this calls for a Reviewer who is meticulous and skeptical. The three Agents communicate through a message bus that provides point-to-point message passing, with each message carrying a correlation ID to match requests and responses.

```python runnable gpuonly extract-class="SpecializedAgent, AgentMessage, MessageBus"
# Specialized Agent and Message Bus
import time

class AgentMessage:
    """Structured message for inter-Agent communication"""

    def __init__(self, msg_type, sender, receiver, payload, correlation_id=None):
        self.type = msg_type
        self.sender = sender
        self.receiver = receiver
        self.payload = payload
        self.correlation_id = correlation_id
        self.timestamp = time.time()

class MessageBus:
    """Message Bus: supports point-to-point message passing"""

    def __init__(self):
        self._queues = {}

    def send(self, message):
        """Send a message to the specified recipient"""
        if message.receiver not in self._queues:
            self._queues[message.receiver] = []
        self._queues[message.receiver].append(message)

    def receive(self, agent_id):
        """Receive the next message (FIFO order)"""
        queue = self._queues.get(agent_id, [])
        if queue:
            return queue.pop(0)
        return None

class SpecializedAgent:
    """Base class for specialized Agents, encapsulating role definition and message processing loop"""

    def __init__(self, agent_id, role, description, tools, bus):
        self.agent_id = agent_id
        self.role = role
        self.description = description
        self.tools = tools
        self.bus = bus
        self.status = "idle"

    def get_system_prompt(self):
        """Generate system prompt based on role"""
        tool_list = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools.get_schemas()])
        return "\n".join([
            f"You are the {self.role}. {self.description}",
            "",
            "Available tools:",
            tool_list,
            "",
            "Behavior guidelines:",
            "1. Only handle tasks related to your role's expertise",
            "2. Use available tools to complete assigned tasks",
            "3. Submit structured results via RESULT_SUBMIT message upon task completion",
            "4. If you encounter an issue you cannot handle, explain the specific reason via ERROR_REPORT message",
        ])

    def process(self, message):
        """Process a received message"""
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
        """Execute the specific task (subclasses override to provide domain expertise)"""
        return {"status": "completed", "summary": f"{self.role} completed task: {task.get('description', '')}"}
```

The core differences between each specialized Agent manifest in three aspects: the role definition in the system prompt determines the LLM's behavioral tendency, the available toolset determines what the Agent can do, and the domain logic in the `_execute` method determines how tasks are processed. These three differentiators make the Researcher more focused on searching and organizing information, the Coder more reliable at writing and testing code, and the Reviewer more critical at finding issues and verifying facts.

## Phase 5: Orchestration and Fault Tolerance

With three specialized Agents in place, we need an orchestrator to coordinate their work. The orchestrator is responsible for decomposing the research goal into structured subtasks, assigning each subtask to the appropriate Agent based on its type, and collecting and integrating the execution results from all Agents. It does not perform the actual work itself but ensures the overall process proceeds in a logical order. The phases of a research task exhibit linear dependencies (code can only be written after the algorithm is understood, and review can only happen after the code is working). This "A's output is B's input" structure is best suited to a pipeline orchestration pattern.

```python runnable gpuonly extract-class="Orchestrator"
# Orchestrator: task decomposition, Agent assignment, and result integration
class Orchestrator:
    """Centralized orchestrator, responsible for task decomposition, assignment, and result integration"""

    def __init__(self, bus, agents, planner, fault_handler=None):
        self.bus = bus
        self.agents = {a.agent_id: a for a in agents}
        self.planner = planner
        self.fault = fault_handler
        self.results = {}

    def execute(self, goal):
        """Execute the complete workflow: decompose, assign, collect, integrate"""
        tasks = self.planner.decompose(goal)
        report_parts = []

        for task in tasks:
            agent_id = self._select_agent(task["action"])
            if agent_id is None:
                continue

            if self.fault and not self.fault.can_execute(agent_id):
                print(f"Circuit breaker open, skipping Agent: {agent_id}")
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
                        print(f"Agent {agent_id} consecutive failures, circuit breaker opened")
                report_parts.append({"step": task["description"], "agent": agent_id, "result": {"status": "timeout"}})

        return self._compile_report(goal, report_parts)

    def _select_agent(self, action_type):
        """Select the most suitable Agent based on task type"""
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
        """Assign a subtask to the specified Agent"""
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
        """Wait for and collect the Agent's execution result"""
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
        """Synthesize the outputs from all phases into the final report"""
        sections = []
        for p in parts:
            sections.append(f"## {p['step']}\n(Completed by {p['agent']})\n\n{p['result']}")
        return {
            "title": f"Technical Research Report: {goal[:50]}",
            "sections": sections,
            "metadata": {"steps": len(parts), "completed": sum(1 for p in parts if p['result'])}
        }
```

In a multi-agent scenario, any individual Agent may fail due to a temporarily unavailable LLM API, tool call timeout, or reasoning loops. The linear dependencies in pipeline orchestration mean that an upstream failure blocks all downstream tasks. The `FaultHandler` provides two basic protections: circuit breaker and timeout. The circuit breaker automatically cuts off task assignment when an Agent fails consecutively beyond a threshold, giving the faulty component time to recover and avoiding wasted resources on operations known to fail.

```python runnable gpuonly extract-class="FaultHandler"
# Basic Fault Tolerance Module: timeout protection and circuit breaker
class FaultHandler:
    """Fault handler, providing timeout and circuit breaker as basic protection mechanisms"""

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
        """Check if the Agent is available (circuit breaker protection)"""
        state = self._states.get(agent_id, self.CLOSED)

        if state == self.OPEN:
            elapsed = time.time() - self._last_failure.get(agent_id, 0)
            if elapsed >= self.recovery_timeout:
                self._states[agent_id] = self.HALF_OPEN
                return True
            return False

        return True

    def record_success(self, agent_id):
        """Record a successful execution, reset the circuit breaker"""
        self._states[agent_id] = self.CLOSED
        self._failures[agent_id] = 0

    def record_failure(self, agent_id):
        """Record an execution failure, open the circuit breaker when threshold is reached"""
        self._failures[agent_id] = self._failures.get(agent_id, 0) + 1
        self._last_failure[agent_id] = time.time()

        if self._failures[agent_id] >= self.failure_threshold:
            self._states[agent_id] = self.OPEN
            return True
        return False
```

## Integration Testing

We conduct end-to-end tests comparing the single-agent and multi-agent approaches on a real research task. The task is to compare the performance of quicksort and merge sort, provide Python implementations and benchmarks, and generate a technical report, covering the four phases of search, coding, testing, and report generation. The test will actually start the Qwen3.5-0.8B-Instruct model for inference and execute real internet searches via the AnySearch API.

### Single Agent End-to-End Test

The test below registers the complete research toolset (internet search, code execution, file read/write), then hands them over to `AgentCore`, letting the LLM autonomously complete the full research workflow from searching materials to generating the report. This is a comprehensive verification of the Agent core loop, tool calling, memory management, and prompt engineering.

```python runnable gpuonly
# Single Agent End-to-End Test: LLM + real tools + report generation
from shared.agent_systems.tool_registry import ToolRegistry
from shared.agent_systems.memory_manager import MemoryManager
from shared.agent_systems.agent_core import AgentCore
import os, json, subprocess, requests, time

# ---- Register research tools ----
tools = ToolRegistry()

@tools.register(
    name="search",
    description="Search the internet for technical information. Suitable for finding algorithm principles, technical documentation, etc. Returns search result summaries.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keywords"},
            "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 5}
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
        return {"query": query, "error": "Unable to connect to the search service", "results": ""}
    except requests.exceptions.Timeout:
        return {"query": query, "error": "Search request timed out", "results": ""}
    except Exception as e:
        return {"query": query, "error": f"Search exception: {str(e)}", "results": ""}

@tools.register(
    name="execute_code",
    description="Execute Python code and return standard output. Suitable for verifying algorithm implementations, running benchmarks, and checking code correctness. Each call is independent.",
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python code to execute"}},
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
        return {"error": "Code execution timed out (30 seconds)"}

@tools.register(
    name="write_file",
    description="Write content to a file. Suitable for saving research notes, code drafts, and the final report.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "Content to write"}
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
    description="Read file content. Suitable for viewing previously saved notes, code, or report drafts.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path"}},
        "required": ["path"]
    }
)
def read_file_tool(path):
    if not os.path.exists(path):
        return {"error": f"File does not exist: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content, "size": len(content)}

print(f"Registered tools: {[s['name'] for s in tools.get_schemas()]}")

# ---- Execute research task ----
print("\n" + "=" * 60)
print("Single Agent End-to-End Research")
print("=" * 60)

memory = MemoryManager(max_history=20)
agent = AgentCore(tools, memory, max_iterations=10)

goal = "Compare the performance of quicksort and merge sort, implement both algorithms in Python and run benchmarks, generate a technical report"
print(f"Research goal: {goal}")
print("Loading Qwen3.5-0.8B-Instruct model...")

start_time = time.time()
report = agent.run(goal)
elapsed = time.time() - start_time

print(f"\nResearch complete, took {elapsed:.0f} seconds, used {len(memory.history)} conversation records")
print("\n" + "-" * 40)
print("Research Report")
print("-" * 40)
print(report)

# Save report
report_path = os.path.join(DATA_DIR, "outputs", "single_agent_report.md")
tools.execute("write_file", path=report_path, content=report)
print(f"\nReport saved to {report_path}")
```

### Multi-Agent End-to-End Test

The test below assigns the same research task to the multi-agent collaboration system. The `Planner` decomposes the goal into four subtasks: search, coding, testing, and report generation. The `Orchestrator` assigns each subtask to the appropriate specialized Agent (Researcher and Coder) based on task type. Each Agent uses an independent `AgentCore` instance to drive LLM inference and communicates with the orchestrator via the `MessageBus`. The `FaultHandler` monitors the execution status of each Agent and triggers circuit breaker protection on consecutive failures.

```python runnable gpuonly
# Multi-Agent End-to-End Test: Orchestrator + specialized Agents + LLM + report generation
from shared.agent_systems.tool_registry import ToolRegistry
from shared.agent_systems.memory_manager import MemoryManager
from shared.agent_systems.agent_core import AgentCore
from shared.agent_systems.specialized_agent import SpecializedAgent, MessageBus, AgentMessage
from shared.agent_systems.planner import Planner
from shared.agent_systems.fault_handler import FaultHandler
from shared.agent_systems.orchestrator import Orchestrator
import os, json, subprocess, requests, time

# ---- Register research tools (same toolset as single-agent test) ----
tools = ToolRegistry()

@tools.register(
    name="search",
    description="Search the internet for technical information. Returns search result summaries.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keywords"},
            "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 5}
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
        return {"query": query, "error": "Unable to connect to the search service", "results": ""}
    except requests.exceptions.Timeout:
        return {"query": query, "error": "Search request timed out", "results": ""}
    except Exception as e:
        return {"query": query, "error": f"Search exception: {str(e)}", "results": ""}

@tools.register(
    name="execute_code",
    description="Execute Python code and return standard output. Each call is independent.",
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python code to execute"}},
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
        return {"error": "Code execution timed out (30 seconds)"}

@tools.register(
    name="write_file",
    description="Write content to a file.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "Content to write"}
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
    description="Read file content.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path"}},
        "required": ["path"]
    }
)
def read_file_tool(path):
    if not os.path.exists(path):
        return {"error": f"File does not exist: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content, "size": len(content)}

print(f"Registered tools: {[s['name'] for s in tools.get_schemas()]}")

# ---- Create multi-agent collaboration system ----
print("\n" + "=" * 60)
print("Multi-Agent End-to-End Research")
print("=" * 60)

bus = MessageBus()
planner = Planner()
fault_handler = FaultHandler(failure_threshold=3, recovery_timeout=30)

# Create specialized Agents
researcher = SpecializedAgent("researcher", "Researcher",
    "Responsible for searching technical materials, filtering reliable sources, and extracting key information. Use the search tool to find materials and write_file to save intermediate artifacts.",
    tools, bus)
coder = SpecializedAgent("coder", "Engineer",
    "Responsible for translating algorithm descriptions into runnable code, executing benchmarks, and verifying result correctness. Use the execute_code tool to run code.",
    tools, bus)

# Equip each Agent with an independent AgentCore for LLM inference
print("Loading Qwen3.5-0.8B-Instruct model for each Agent...")

researcher_core = AgentCore(tools, MemoryManager(max_history=15), max_iterations=8)
coder_core = AgentCore(tools, MemoryManager(max_history=15), max_iterations=8)

# Bind AgentCore to the specialized Agent's _execute method
# Provide different system prompts for each role, guiding the LLM to focus on its respective domain
def make_execute(core, role_prompt):
    """Create an _execute method that uses AgentCore (LLM) to process tasks"""
    def _execute(task):
        task_desc = task.get("description", str(task))
        # Create independent memory context for each subtask to avoid cross-task information contamination
        memory = MemoryManager(max_history=15)
        memory.add("system", role_prompt)
        core.memory = memory
        return core.run(task_desc)
    return _execute

researcher._execute = make_execute(researcher_core,
    "You are a technical researcher. Use the search tool to find materials, organize key information, save findings to a file and return. "
    "Record every important finding as a key fact. Return structured research notes upon completion.")
coder._execute = make_execute(coder_core,
    "You are an algorithm engineer. Use the execute_code tool to write and run code, ensuring correct output. "
    "Write implementation code first, then run benchmarks, and finally summarize performance comparison data. Return complete code and test results upon completion.")

# Create orchestrator
orchestrator = Orchestrator(bus, [researcher, coder], planner, fault_handler)

# Fix message dispatch in single-threaded environment: manually trigger Agent processing after the orchestrator sends a message
agents = {"researcher": researcher, "coder": coder}
original_assign = orchestrator._assign
def assign_with_dispatch(task, agent_id):
    original_assign(task, agent_id)
    msg = bus.receive(agent_id)
    if msg:
        agents[agent_id].process(msg)
orchestrator._assign = assign_with_dispatch

# ---- Execute multi-agent research ----
goal = "Compare the performance of quicksort and merge sort, implement both algorithms in Python and run benchmarks, generate a technical report"
print(f"Research goal: {goal}")

start_time = time.time()
multi_result = orchestrator.execute(goal)
elapsed = time.time() - start_time

# ---- Output results ----
print(f"\nResearch complete, took {elapsed:.0f} seconds")
print(f"Completion: {multi_result['metadata']['completed']}/{multi_result['metadata']['steps']} subtasks")

print("\n" + "-" * 40)
print("Multi-Agent Research Report")
print("-" * 40)
print(f"Title: {multi_result['title']}")
for section in multi_result["sections"]:
    print(section)

# Save report
report_text = f"# {multi_result['title']}\n\n"
for section in multi_result["sections"]:
    report_text += section + "\n\n"

report_path = os.path.join(DATA_DIR, "outputs", "multi_agent_report.md")
tools.execute("write_file", path=report_path, content=report_text)
print(f"\nReport saved to {report_path}")
```

## Summary

Comparing the two approaches on the same research task reveals their respective applicability boundaries. The single-agent approach has the advantage of structural simplicity, no communication latency, and no orchestration overhead. For small tasks like searching for quicksort principles and summarizing them, a single Agent is most efficient. When the task spans multiple professional domains (requiring both algorithm theory knowledge and the ability to write correct code and review quality), the single Agent's breadth-over-depth problem begins to show. The multi-agent approach offers the advantage of specialized depth -- each Agent only needs to excel in its own domain. However, this specialization also comes with costs, such as message-passing latency between Agents, the orchestrator's coordination burden, and the cascading impact when an Agent fails. Therefore, the criterion for choosing between the two approaches is not the number of Agents, but whether the task complexity exceeds the professional capacity of a single Agent. If the multiple phases of a task do not require significantly different knowledge and skills, forcibly splitting them only adds unnecessary complexity.

The current implementation has several notable limitations. The orchestrator uses a rule-based fixed decomposition strategy that lacks flexibility when encountering task types beyond the preset rules; a future improvement would be to introduce LLM-driven dynamic task decomposition. Inter-Agent communication uses a synchronous point-to-point pattern that limits parallel execution capability; introducing asynchronous messaging and fan-out/fan-in orchestration would allow independent subtasks to execute concurrently. The fault tolerance mechanism currently only covers timeouts and circuit breakers, lacking checkpoint recovery, which means all progress is lost if the system crashes mid-task. Additionally, code execution runs directly in the local environment; a production deployment would need sandbox isolation to mitigate security risks.
