# 工程实训：构建多智能体协作系统

在[上一章的实训](agent-experiment.md)中，我们构建了一个单 Agent 系统，它能够自主规划、调用工具和自我修正。但单个 Agent 的能力受限于其知识范围和工具集。本次工程实训将构建一个多智能体协作系统，让多个专业化 Agent 通过协调器分工协作，完成单个 Agent 无法胜任的复杂任务。通过实践，你将理解协作架构的设计、通信协议的实现、编排策略的选择和容错机制的部署。

## 实验目标

- 构建一个基于集中式架构的多智能体协作系统，包含协调器和多个专业化 Agent
- 实现 Agent 间的通信协议，支持结构化消息传递和状态同步
- 实现工作流编排器，支持管道、扇出-扇入和条件路由等编排模式
- 实现容错机制，包括超时重试、断路器和检查点恢复
- 通过实际协作任务验证系统的协调能力和容错能力

## 实验准备

### 环境与依赖

- Python 3.10+ 运行环境
- 可访问的 LLM API
- 实验所需的 Python 包：`openai`（或兼容 API 客户端）、`jsonschema`（消息验证）、`asyncio`（异步执行）

```python runnable
# 验证实验环境
import sys
import asyncio
print(f"Python 版本: {sys.version}")
print(f"asyncio: 可用")

# 检查依赖
try:
    import jsonschema
    print(f"jsonschema: {jsonschema.__version__}")
except ImportError:
    print("jsonschema: 未安装，将使用简化验证")
```

### 实验架构概览

本次实验构建的多智能体系统包含以下核心组件：

1. **MessageBus**：消息总线，实现 Agent 间的结构化通信
2. **Orchestrator**：编排器，负责任务分解、Agent 分配和流程控制
3. **SpecializedAgent**：专业化 Agent 基类，每个 Agent 有独立的角色定义和工具集
4. **WorkflowEngine**：工作流引擎，执行预定义的工作流 DAG
5. **FaultTolerance**：容错模块，提供超时重试、断路器和检查点恢复

## 第一阶段：通信协议

### 消息格式定义

- 定义统一的消息格式：消息类型、发送者、接收者、时间戳、负载、关联标识
- 定义消息类型枚举：TASK_ASSIGN（任务分配）、STATUS_UPDATE（状态更新）、RESULT_SUBMIT（结果提交）、ERROR_REPORT（错误报告）、CONTROL（控制指令）
- 实现消息的序列化和反序列化

```python runnable extract-class="AgentMessage"
# Agent 间通信消息定义
import json
import time
import uuid
from enum import Enum

class MessageType(Enum):
    TASK_ASSIGN = "task_assign"
    STATUS_UPDATE = "status_update"
    RESULT_SUBMIT = "result_submit"
    ERROR_REPORT = "error_report"
    CONTROL = "control"

class AgentMessage:
    """Agent 间通信的结构化消息"""
    
    def __init__(self, msg_type, sender, receiver, payload, 
                 correlation_id=None, priority=0, ttl=None):
        self.id = str(uuid.uuid4())
        self.type = MessageType(msg_type) if isinstance(msg_type, str) else msg_type
        self.sender = sender
        self.receiver = receiver
        self.payload = payload
        self.correlation_id = correlation_id or self.id
        self.priority = priority
        self.ttl = ttl
        self.timestamp = time.time()
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data):
        msg = cls(
            msg_type=data["type"],
            sender=data["sender"],
            receiver=data["receiver"],
            payload=data["payload"],
            correlation_id=data.get("correlation_id"),
            priority=data.get("priority", 0)
        )
        msg.id = data["id"]
        msg.timestamp = data.get("timestamp", time.time())
        return msg
    
    def is_expired(self):
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
```

### 消息总线实现

- 实现发布-订阅模式的消息总线：Agent 可以订阅特定类型的消息
- 实现点对点消息传递：直接向指定 Agent 发送消息
- 实现消息队列：每个 Agent 有自己的消息队列，支持按优先级排序
- 实现消息确认机制：接收方确认消息后，发送方才视为消息已送达

```python runnable extract-class="MessageBus"
# 消息总线实现
from collections import defaultdict
import heapq

class MessageBus:
    """Agent 间的消息总线，支持发布-订阅和点对点通信"""
    
    def __init__(self):
        self._subscribers = defaultdict(list)  # topic -> [agent_ids]
        self._queues = defaultdict(list)        # agent_id -> [(priority, message)]
        self._ack_pending = {}                   # msg_id -> (sender, callback)
    
    def subscribe(self, agent_id, topic):
        """订阅特定主题的消息"""
        if agent_id not in self._subscribers[topic]:
            self._subscribers[topic].append(agent_id)
    
    def publish(self, message):
        """发布消息到主题，所有订阅者都会收到"""
        topic = message.type.value
        for agent_id in self._subscribers.get(topic, []):
            self._enqueue(agent_id, message)
    
    def send(self, message):
        """点对点发送消息给指定 Agent"""
        self._enqueue(message.receiver, message)
    
    def receive(self, agent_id):
        """接收 Agent 的下一条消息（按优先级排序）"""
        if not self._queues[agent_id]:
            return None
        _, message = heapq.heappop(self._queues[agent_id])
        if message.is_expired():
            return self.receive(agent_id)  # 跳过过期消息
        return message
    
    def _enqueue(self, agent_id, message):
        """将消息加入 Agent 的消息队列（按优先级排序）"""
        heapq.heappush(self._queues[agent_id], (-message.priority, message))
```

## 第二阶段：专业化 Agent

### Agent 基类设计

- 定义专业化 Agent 的基类，包含角色定义、工具集、消息处理循环
- 每个 Agent 有独立的系统提示词，定义其角色、能力和行为约束
- Agent 的消息处理循环：从消息总线接收消息 → 根据消息类型处理 → 发送响应

```python runnable extract-class="SpecializedAgent"
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
{self.description}

可用工具:
{tool_descriptions}

行为规范:
1. 只处理与你角色相关的任务
2. 使用可用工具完成任务
3. 完成后通过 RESULT_SUBMIT 消息提交结果
4. 遇到无法处理的问题通过 ERROR_REPORT 消息上报"""
    
    def process_message(self, message):
        """处理接收到的消息"""
        if message.type == MessageType.TASK_ASSIGN:
            self.status = "working"
            self.current_task = message.payload
            result = self._execute_task(message.payload)
            # 提交结果
            result_msg = AgentMessage(
                msg_type=MessageType.RESULT_SUBMIT,
                sender=self.agent_id,
                receiver=message.sender,
                payload=result,
                correlation_id=message.correlation_id
            )
            self.bus.send(result_msg)
            self.status = "idle"
            self.current_task = None
        
        elif message.type == MessageType.CONTROL:
            action = message.payload.get("action")
            if action == "cancel":
                self.status = "idle"
                self.current_task = None
    
    def _execute_task(self, task):
        """执行具体任务（子类可覆盖）"""
        # 调用 LLM 进行推理和工具调用
        # 这里是简化实现
        return {"status": "completed", "task": task.get("description", "")}
```

### 实现 Researcher Agent

- 角色：信息研究员，负责搜索和整理信息
- 工具集：Web 搜索、文档读取、信息摘要
- 专精能力：从多个来源收集信息，去重整合，生成结构化的研究报告

### 实现 Coder Agent

- 角色：代码工程师，负责编写和调试代码
- 工具集：代码执行、文件读写、语法检查
- 专精能力：根据需求编写代码，执行测试，修复错误

### 实现 Reviewer Agent

- 角色：质量审查员，负责审查和验证其他 Agent 的输出
- 工具集：代码审查、文本对比、规则检查
- 专精能力：发现代码缺陷、逻辑错误、格式问题，提出修改建议

## 第三阶段：编排器

### 任务分解与分配

- 编排器接收高层目标，将其分解为子任务，并根据 Agent 的角色和能力分配
- 分配策略：基于角色匹配（将代码任务分配给 Coder Agent）、基于负载均衡（将任务分配给当前空闲的 Agent）、基于能力评分（根据 Agent 历史表现选择最合适的 Agent）

```python runnable extract-class="Orchestrator"
# 编排器实现
class Orchestrator:
    """集中式编排器，负责任务分解、分配和结果整合"""
    
    def __init__(self, message_bus, agents):
        self.bus = message_bus
        self.agents = {agent.agent_id: agent for agent in agents}
        self.task_results = {}
        self.workflow = None
    
    def decompose_task(self, goal):
        """将高层目标分解为子任务列表"""
        # 调用 LLM 进行任务分解
        # 简化实现：基于规则的分解
        subtasks = self._rule_based_decompose(goal)
        return subtasks
    
    def assign_task(self, task, agent_id):
        """将子任务分配给指定 Agent"""
        msg = AgentMessage(
            msg_type=MessageType.TASK_ASSIGN,
            sender="orchestrator",
            receiver=agent_id,
            payload=task
        )
        self.bus.send(msg)
    
    def collect_results(self, task_ids, timeout=60):
        """收集指定任务的执行结果"""
        results = {}
        deadline = time.time() + timeout
        
        while time.time() < deadline and len(results) < len(task_ids):
            msg = self.bus.receive("orchestrator")
            if msg and msg.type == MessageType.RESULT_SUBMIT:
                results[msg.correlation_id] = msg.payload
            elif msg and msg.type == MessageType.ERROR_REPORT:
                results[msg.correlation_id] = {"status": "error", "error": msg.payload}
        
        return results
    
    def _rule_based_decompose(self, goal):
        """基于规则的任务分解（简化实现）"""
        # 实际实现中应调用 LLM 进行智能分解
        return [{"id": "task_1", "description": goal, "type": "general"}]
```

### 工作流引擎

- 实现 DAG 形式的工作流定义：节点是子任务，边是依赖关系
- 实现工作流执行引擎：按拓扑排序执行，无依赖的节点并行执行
- 支持条件路由：根据前一步的结果决定后续路径

```python runnable extract-class="WorkflowEngine"
# 工作流引擎实现
from collections import deque

class WorkflowEngine:
    """基于 DAG 的工作流执行引擎"""
    
    def __init__(self, message_bus, orchestrator):
        self.bus = message_bus
        self.orchestrator = orchestrator
        self.checkpoints = {}
    
    def execute(self, workflow):
        """执行工作流 DAG"""
        # 拓扑排序确定执行顺序
        execution_order = self._topological_sort(workflow)
        
        # 按层级执行（同层节点可并行）
        for level in execution_order:
            # 并行分配同层任务
            for node_id in level:
                node = workflow["nodes"][node_id]
                agent_id = node.get("agent_id")
                if agent_id:
                    self.orchestrator.assign_task(node["task"], agent_id)
            
            # 等待同层任务完成
            results = self.orchestrator.collect_results(level)
            
            # 保存检查点
            self.checkpoints[level] = results
            
            # 检查是否有失败的任务
            for task_id, result in results.items():
                if result.get("status") == "error":
                    # 触发容错处理
                    self._handle_failure(task_id, result, workflow)
        
        return self.checkpoints
    
    def _topological_sort(self, workflow):
        """拓扑排序，返回按层级组织的执行顺序"""
        nodes = workflow["nodes"]
        edges = workflow.get("edges", [])
        
        # 计算入度
        in_degree = {nid: 0 for nid in nodes}
        children = {nid: [] for nid in nodes}
        for edge in edges:
            children[edge["from"]].append(edge["to"])
            in_degree[edge["to"]] += 1
        
        # BFS 分层
        levels = []
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        while queue:
            level = []
            for _ in range(len(queue)):
                nid = queue.popleft()
                level.append(nid)
                for child in children[nid]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
            levels.append(level)
        
        return levels
```

## 第四阶段：容错机制

### 超时与重试

- 为每个 Agent 任务设置超时时间，超时后视为失败
- 实现指数退避重试：首次重试间隔 1 秒，每次翻倍，最多重试 3 次
- 重试时可以选择同一 Agent（假设失败是暂时性的）或切换到备选 Agent

### 断路器实现

- 为每个 Agent 维护断路器状态：连续失败 N 次后打开断路器，不再向该 Agent 分配任务
- 断路器打开后，经过冷却期进入半开状态，允许试探性任务
- 试探成功则关闭断路器，失败则重新打开

```python runnable extract-class="CircuitBreaker"
# 断路器实现
class CircuitBreaker:
    """断路器：在 Agent 持续失败时暂停向其分配任务"""
    
    CLOSED = "closed"      # 正常工作
    OPEN = "open"          # 暂停分配
    HALF_OPEN = "half_open"  # 试探性恢复
    
    def __init__(self, failure_threshold=3, recovery_timeout=60, half_open_max=1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self._states = {}  # agent_id -> state
        self._failure_counts = {}
        self._last_failure_time = {}
        self._half_open_counts = {}
    
    def can_execute(self, agent_id):
        """检查是否可以向该 Agent 分配任务"""
        state = self._states.get(agent_id, self.CLOSED)
        
        if state == self.CLOSED:
            return True
        
        if state == self.OPEN:
            # 检查是否到达恢复时间
            elapsed = time.time() - self._last_failure_time.get(agent_id, 0)
            if elapsed >= self.recovery_timeout:
                self._states[agent_id] = self.HALF_OPEN
                self._half_open_counts[agent_id] = 0
                return True
            return False
        
        if state == self.HALF_OPEN:
            # 限制试探性请求的数量
            return self._half_open_counts.get(agent_id, 0) < self.half_open_max
        
        return False
    
    def record_success(self, agent_id):
        """记录成功执行"""
        self._states[agent_id] = self.CLOSED
        self._failure_counts[agent_id] = 0
    
    def record_failure(self, agent_id):
        """记录执行失败"""
        self._failure_counts[agent_id] = self._failure_counts.get(agent_id, 0) + 1
        self._last_failure_time[agent_id] = time.time()
        
        if self._failure_counts[agent_id] >= self.failure_threshold:
            self._states[agent_id] = self.OPEN
```

### 检查点与恢复

- 在工作流的关键节点保存检查点：已完成的任务及其结果、当前工作流状态
- 故障后从最近的检查点恢复：跳过已完成的任务，从失败点重新执行
- 检查点的持久化：将检查点数据序列化到文件，支持跨会话恢复

## 第五阶段：集成测试

### 协作任务设计

- 任务一：研究报告生成（Researcher 搜索信息 → Coder 生成数据可视化 → Reviewer 审查质量）——测试管道编排
- 任务二：多方案评估（多个 Coder Agent 各自实现方案 → Reviewer 评估 → 选择最优）——测试扇出-扇入编排
- 任务三：条件处理（Researcher 分析问题 → 根据复杂度选择 Coder 或直接回答 → Reviewer 验证）——测试条件路由
- 任务四：故障恢复（模拟 Agent 超时和工具失败，测试断路器和重试机制）——测试容错能力

### 协作效率评估

- 任务完成率：协作系统成功完成任务的比例
- 通信开销：Agent 间消息传递的数量和延迟
- 并行效率：并行执行的任务数与总任务数的比率
- 容错恢复率：故障后成功恢复并完成任务的比例

### 与单 Agent 的对比

- 在相同任务上对比单 Agent 和多 Agent 系统的表现
- 分析多 Agent 系统在哪些场景下优于单 Agent，在哪些场景下不如单 Agent
- 讨论通信开销和协调成本对整体效率的影响

### 局限性与改进方向

- 当前实现的局限：集中式架构的单点瓶颈、通信延迟、Agent 数量有限
- 改进方向：引入层级式架构支持更大规模、实现去中心化协调、增加人类协作接口
- 与生产级多 Agent 系统的差距：缺少持久化部署、缺少安全隔离、缺少监控仪表盘

## 可视化建议

1. **多智能体系统架构图**：展示 Orchestrator、MessageBus、多个 SpecializedAgent 的关系
2. **工作流 DAG 示例图**：以研究报告生成任务为例，展示任务节点和依赖关系
3. **断路器状态转移图**：展示 Closed-Open-Half_Open 三个状态的转移条件
4. **协作时序图**：展示一次完整协作任务中各 Agent 的消息交互时序
5. **单 Agent vs 多 Agent 对比图**：在任务完成率、执行时间等维度上的对比

## 代码示例建议

本实训的所有代码示例均为 runnable，按阶段组织：
1. 消息格式与消息总线完整实现（含发布-订阅、点对点、优先级队列）
2. 专业化 Agent 基类及三个具体 Agent 实现（Researcher、Coder、Reviewer）
3. 编排器与工作流引擎完整实现（含任务分解、DAG 执行、条件路由）
4. 容错模块完整实现（含超时重试、断路器、检查点恢复）
5. 集成测试脚本（运行多个协作任务，输出评估指标和对比分析）
