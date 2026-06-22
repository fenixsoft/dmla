# 工程实训：构建自主 Agent

前面的章节从原理层面讨论了 Agent 的架构、工具调用、规划推理和记忆系统。本次工程实训将把这些概念落地为一个可运行的自主 Agent，从零开始实现一个能够理解目标、规划步骤、调用工具、观察结果并自我修正的完整系统。通过动手实践，你将理解 Agent 系统中每个组件的设计权衡和实现细节。

## 实验目标

- 构建一个基于 ReAct 模式的自主 Agent，能够完成多步骤的复杂任务
- 实现工具注册与调用机制，支持多种工具的动态注册和参数验证
- 实现简单的记忆系统，支持对话历史管理和关键信息提取
- 实现自我修正机制，在工具调用失败时自动调整策略
- 通过实际任务验证 Agent 的自主性和可靠性

## 实验准备

### 环境与依赖

- Python 3.10+ 运行环境
- 可访问的 LLM API（支持函数调用功能）
- 实验所需的 Python 包：`openai`（或兼容 API 客户端）、`jsonschema`（参数验证）、`faiss-cpu`（向量检索）

```python runnable
# 验证实验环境
import sys
print(f"Python 版本: {sys.version}")

# 检查关键依赖
try:
    import jsonschema
    print(f"jsonschema: {jsonschema.__version__}")
except ImportError:
    print("jsonschema: 未安装，将使用简化验证")

try:
    import faiss
    print(f"faiss-cpu: 已安装")
except ImportError:
    print("faiss-cpu: 未安装，将使用纯 Python 向量检索")
```

### 实验架构概览

本次实验构建的 Agent 系统包含以下核心组件：

1. **AgentCore**：Agent 的核心循环，实现 ReAct 模式的思考-行动-观察循环
2. **ToolRegistry**：工具注册中心，管理可用工具的描述、参数 schema 和执行函数
3. **MemoryManager**：记忆管理器，维护对话历史和提取的关键信息
4. **Planner**：简单规划器，将复杂目标分解为子任务序列
5. **SelfCorrector**：自我修正模块，检测执行异常并调整策略

## 第一阶段：工具注册与调用

### 工具描述 schema 设计

- 定义统一的工具描述格式，包含名称、描述、参数 schema、返回值格式
- 参数 schema 使用 JSON Schema 表达，支持类型约束、必选/可选、枚举值
- 工具描述的质量直接影响 LLM 选择工具和填写参数的准确率

```python runnable
# 工具描述 schema 示例
TOOL_SCHEMAS = {
    "calculator": {
        "name": "calculator",
        "description": "执行数学计算，支持加减乘除和幂运算",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2 + 3 * 4'"
                }
            },
            "required": ["expression"]
        }
    },
    "web_search": {
        "name": "web_search",
        "description": "搜索互联网获取信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回结果数量上限",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}
```

### 工具注册中心实现

- 实现工具的动态注册：开发者通过装饰器或注册函数将工具添加到注册中心
- 实现参数验证：调用前根据 schema 验证参数类型和完整性
- 实现执行隔离：每个工具在 try-except 中执行，捕获异常并返回结构化错误信息
- 实现超时控制：为每个工具设置默认超时，防止工具永远不返回

```python runnable extract-class="ToolRegistry"
# 工具注册中心实现
import json
import time
import signal
from functools import wraps

class ToolRegistry:
    """工具注册中心，管理可用工具的注册、验证和执行"""
    
    def __init__(self):
        self._tools = {}
        self._schemas = {}
    
    def register(self, name=None, description="", parameters=None, timeout=30):
        """工具注册装饰器"""
        def decorator(func):
            tool_name = name or func.__name__
            self._tools[tool_name] = func
            self._schemas[tool_name] = {
                "name": tool_name,
                "description": description or func.__doc__ or "",
                "parameters": parameters or {"type": "object", "properties": {}},
                "timeout": timeout
            }
            return func
        return decorator
    
    def get_schemas(self):
        """获取所有工具的描述 schema"""
        return list(self._schemas.values())
    
    def execute(self, tool_name, **kwargs):
        """执行指定工具，带参数验证和超时控制"""
        if tool_name not in self._tools:
            return {"error": f"工具 '{tool_name}' 不存在", "available_tools": list(self._tools.keys())}
        
        # 参数验证（简化版）
        schema = self._schemas[tool_name]
        required = schema["parameters"].get("required", [])
        for param in required:
            if param not in kwargs:
                return {"error": f"缺少必选参数 '{param}'"}
        
        # 执行工具
        try:
            result = self._tools[tool_name](**kwargs)
            return {"result": result}
        except Exception as e:
            return {"error": f"工具执行失败: {str(e)}"}
```

### 内置工具实现

- 实现计算器工具：解析和执行数学表达式
- 实现文件操作工具：读取和写入本地文件
- 实现列表操作工具：对列表进行排序、过滤、统计
- 每个工具都包含完整的错误处理和输入验证

## 第二阶段：Agent 核心循环

### ReAct 循环实现

- 实现 Agent 的主循环：接收用户目标 → 生成思考 → 选择行动 → 执行工具 → 观察结果 → 继续思考或输出答案
- 循环的终止条件：Agent 输出最终答案、达到最大循环次数、遇到不可恢复的错误
- 思考（Thought）的格式：让模型在行动前显式推理当前状态和下一步策略
- 观察（Observation）的格式：将工具返回结果格式化后注入上下文

```python runnable extract-class="AgentCore"
# Agent 核心循环实现
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
```

### 提示词工程

- 系统提示词的设计：定义 Agent 的角色、行为规范、输出格式
- 工具描述的注入方式：将工具 schema 格式化后注入系统提示词
- 思考-行动-观察的格式约束：使用固定标记（如 `[Thought]`、`[Action]`、`[Observation]`）分隔各部分
- 上下文窗口管理：当对话历史过长时，压缩早期内容

### 响应解析

- 从 LLM 的文本输出中提取结构化的行动指令：工具名和参数
- 处理解析失败：当模型输出不符合预期格式时，尝试修复或请求重新生成
- 支持多种输出格式：纯文本标记格式、JSON 格式、函数调用格式

## 第三阶段：记忆系统

### 对话历史管理

- 维护完整的对话历史，包括用户输入、Agent 思考、工具调用和观察结果
- 实现滑动窗口策略：保留最近 N 轮对话，更早的对话压缩为摘要
- 对话历史的格式化：将历史记录格式化为 LLM 可理解的上下文

```python runnable extract-class="MemoryManager"
# 记忆管理器实现
class MemoryManager:
    """简单的记忆管理器，维护对话历史和关键信息"""
    
    def __init__(self, max_history=20):
        self.history = []
        self.key_facts = []
        self.max_history = max_history
    
    def add(self, role, content):
        """添加一条记录到对话历史"""
        self.history.append({"role": role, "content": content})
        
        # 超出限制时压缩早期历史
        if len(self.history) > self.max_history:
            self._compress_history()
    
    def get_context(self):
        """获取当前上下文（格式化的对话历史）"""
        return self.history.copy()
    
    def extract_fact(self, fact):
        """提取关键事实到长期记忆"""
        self.key_facts.append(fact)
    
    def _compress_history(self):
        """压缩早期历史：保留最近记录，将早期记录摘要"""
        # 保留最近 2/3 的记录，将前 1/3 压缩为摘要
        compress_count = len(self.history) // 3
        old_records = self.history[:compress_count]
        self.history = self.history[compress_count:]
        
        # 生成摘要（简化实现：提取关键信息）
        summary = f"[历史摘要: 共 {len(old_records)} 轮对话已压缩]"
        self.history.insert(0, {"role": "system", "content": summary})
```

### 关键信息提取

- 在对话过程中自动提取关键实体和关系（如用户提到的文件名、配置参数、错误信息）
- 提取策略：基于模式匹配（正则表达式提取特定格式信息）和基于 LLM（让模型判断哪些信息值得记住）
- 提取的信息存储在长期记忆中，在后续对话中主动注入上下文

## 第四阶段：自我修正

### 错误检测

- 检测工具调用失败：工具返回错误信息时，触发修正流程
- 检测推理矛盾：Agent 的思考中出现逻辑矛盾时，触发重新推理
- 检测目标偏离：Agent 的行动与原始目标无关时，触发方向修正

### 修正策略

- 重试策略：同一工具调用失败时，修正参数后重试（如修正 JSON 格式错误）
- 换路策略：当前工具无法完成任务时，切换到备选工具
- 回溯策略：当前路径明显错误时，回到之前的决策点重新规划
- 降级策略：所有自动修正都失败时，简化任务目标或请求人类协助

```python runnable
# 自我修正示例
def self_correct(tool_name, parameters, error, registry, max_retries=3):
    """自我修正：根据错误类型选择修正策略"""
    for attempt in range(max_retries):
        # 策略1：参数格式修正（如 JSON 解析错误）
        if "json" in error.lower() or "parse" in error.lower():
            # 尝试修复参数格式
            fixed_params = try_fix_parameters(parameters, error)
            if fixed_params != parameters:
                result = registry.execute(tool_name, **fixed_params)
                if "error" not in result:
                    return result
        
        # 策略2：换用备选工具
        alternatives = find_alternative_tools(tool_name, registry)
        for alt_tool in alternatives:
            result = registry.execute(alt_tool, **parameters)
            if "error" not in result:
                return result
        
        # 策略3：简化参数重试
        simplified = simplify_parameters(parameters)
        result = registry.execute(tool_name, **simplified)
        if "error" not in result:
            return result
    
    return {"error": f"自我修正失败，已尝试 {max_retries} 次"}
```

## 第五阶段：集成测试

### 测试任务设计

- 任务一：数学推理（"计算 1 到 100 的所有素数之和"）——测试工具调用和推理能力
- 任务二：信息检索与整合（"查找 Python 3.12 的新特性并总结"）——测试搜索工具和信息整合能力
- 任务三：多步骤操作（"读取配置文件，验证参数，生成报告"）——测试规划和自我修正能力
- 任务四：错误恢复（故意提供无效参数，测试自我修正机制）——测试容错能力

### 性能评估

- 任务完成率：Agent 成功完成任务的比例
- 平均迭代次数：完成任务所需的思考-行动-观察循环次数
- 工具调用准确率：工具选择和参数填写的正确率
- 自我修正成功率：遇到错误后成功修正的比例

### 局限性分析

- 当前实现的局限：依赖 LLM 的推理能力、工具数量有限、记忆系统简单
- 与生产级 Agent 的差距：缺少持久化存储、缺少并发支持、缺少安全沙箱
- 改进方向：引入向量检索增强记忆、支持流式输出、增加人类确认机制

## 可视化建议

1. **Agent 架构总览图**：展示 AgentCore、ToolRegistry、MemoryManager、SelfCorrector 的关系
2. **ReAct 循环时序图**：展示一次完整任务执行中的思考-行动-观察循环
3. **自我修正决策树**：展示不同错误类型对应的修正策略选择
4. **记忆管理流程图**：展示对话历史的写入、压缩和检索流程

## 代码示例建议

本实训的所有代码示例均为 runnable，按阶段组织：
1. 工具注册中心完整实现（含装饰器、参数验证、超时控制）
2. Agent 核心循环完整实现（含 ReAct 模式、响应解析、终止条件）
3. 记忆管理器完整实现（含对话历史管理、压缩、关键信息提取）
4. 自我修正模块完整实现（含错误检测、修正策略、降级处理）
5. 集成测试脚本（运行多个测试任务，输出评估指标）
