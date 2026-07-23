# 规划与推理

部署一个应用需要规划步骤，调试一段代码需要分析因果，撰写一份报告需要组织素材。这些任务的共同特征是依赖多步推理和前瞻性规划。在[第一篇文章](llm-to-agent.md)中，我们讨论了 Agent 如何从被动的语言模型演变为能自主行动的系统，介绍了 ReAct 和 Planner-Executor 等设计模式。那篇文章回答的是 Agent 怎样组织推理与行动的循环，而本文要讨论的是一个更底层的问题 —— Agent 如何将复杂目标分解为可执行的步骤序列。

自动规划（Automated Planning）是人工智能领域历史最悠久的子领域之一。早在语言模型出现之前，研究者们就在探索如何让机器自主生成行动计划。1971 年，斯坦福研究所的理查德·菲克斯（Richard Fikes）和尼尔斯·尼尔森（Nils Nilsson）发明了 STRIPS（Stanford Research Institute Problem Solver）规划系统，这是世界上第一个将规划问题形式化为状态空间搜索的系统。STRIPS 让一个名叫 Shakey 的机器人能够在不同房间之间移动，推开障碍物，完成简单的物品搬运任务。为此，菲克斯和尼尔森定义了规划的三种基本要素，即初始状态、目标状态以及一组带有前置条件和效果的、可执行的操作。这个框架奠定了此后半个世纪自动规划研究的理论基础。

从 STRIPS 到今天的 LLM Agent，规划技术经历了多次范式变迁。1990 年代的图规划（Graphplan）将规划问题转化为图搜索，大幅提升了求解速度。2000 年代启发式搜索规划的兴起，让规划器能够处理数百个操作的大规模问题。最近的变革来自大语言模型。LLM 的常识推理能力让规划不再依赖于人工编码的操作模型，Agent 可以用自然语言描述子任务，推理步骤间的因果关系，甚至在没有完整信息的情况下做出合理的推测。不过，经典规划中积累的智慧仍然是 LLM Agent 规划的根基。本章就从这些根基讲起。

## 规划的基础

规划的本质是把一个过于复杂而无法直接求解的问题，转换为一系列足够简单而可以直接执行的步骤。要完成这个转换，需要依次回答三个层层递进的问题，怎样把大任务拆成小任务、拆完之后用什么形式来表达这个计划、以及当子任务之间存在依赖时如何安排执行顺序。

### 任务分解

想象你接到了一个需求，要为一个 Python 机器学习项目搭建 CI/CD 流水线，让它能够在每次代码提交后自动运行测试、构建镜像并部署到测试环境。面对这样一个笼统的目标，有经验的开发者不会立刻打开终端敲命令，而是先在脑海中勾勒出任务的骨架，首先是环境的准备（安装依赖、配置 Secrets），然后是流程的定义（编写测试阶段、构建阶段、部署阶段的脚本），最后是验证和排错。这种将宏观目标逐层展开为可操作步骤的过程，称为**任务分解**（Task Decomposition）。

任务分解既不能过粗也不宜过细。如果分解得太粗，譬如把整个 CI/CD 搭建概括为写一个流水线配置文件，那么单步操作依然复杂到难以执行，丧失了分解的价值。反过来，如果把安装 Python 依赖进一步拆成安装 NumPy、安装 Pandas、安装 SciKit-Learn……，虽然每一步都简单了，但协调几十个微小步骤的开销会急剧膨胀，丧失了从整体上把握任务结构的能力。一个经验法则是让每个子任务恰好对应于一个可以被单次工具调用完成的操作，譬如一个子任务是"编写 Dockerfile"，另一个子任务是"配置 GitHub Actions 的 Secrets"，每个子任务都有明确的输入、输出和完成标准。

从结构上看，任务分解的策略可以归为三种基本类型。按时间顺序分解（Sequential Decomposition）适用于流程明确的任务，就像执行一份操作手册，步骤之间有先后关系，前一步的输出是后一步的输入。按依赖关系分解（Dependency-Based Decomposition）适用于子任务可以独立进行的场景，譬如同时准备前端和后端的 Dockerfile，两者互不依赖，可以并行推进。按功能模块分解（Hierarchical Decomposition）适用于大型任务，先在顶层划分出"测试"、"构建"、"部署"三个模块，再对每个模块进行下一层级的分解，形成树状结构。实际任务很少只属于一种类型，更常见的情况是在顶层使用功能模块分解，在中层使用依赖关系分解来识别并行机会，在底层使用顺序分解来细化单个执行流程。

好的分解通常满足 **MECE 原则**（Mutually Exclusive, Collectively Exhaustive），这个原则源自管理咨询领域，但在任务分解中同样适用。**互斥**（Mutually Exclusive）要求子任务之间边界清晰，不重叠。譬如不会出现"配置环境"和"安装 Python"两个子任务都涉及 `pip install` 的情况。**穷尽**（Collectively Exhaustive）要求所有子任务合在一起恰好覆盖了原目标的全部范围，不遗漏任何环节。实际的 Agent 系统中，分解逻辑通常由 LLM 来完成。那么前面讨论的 MECE 原则、分解粒度控制、分解策略这些内容要通过提示词和少样本实例准确传递给 LLM，并且在分解结束后让模型自检，以审查者的角色对自己的输出做一次 MECE 检查。

### 计划表示

把任务分解为子任务之后，这些子任务需要以一种结构化的方式组织起来，形成 Agent 可以按图索骥执行的操作序列。这种对任务步骤及其相互关系的结构化描述，称为**计划表示**（Plan Representation）。计划表示的选择影响执行效率、可读性和重新规划的便利程度。常见的表示形式有如下几种：

- 线性计划是最简单的形式，本质上是一个有序步骤列表 `[step1, step2, ..., stepN]`，适合流程明确、无分支的任务。譬如部署流程可以表示为 `[拉取最新代码, 运行测试套件, 构建 Docker 镜像, 推送到镜像仓库, 重启服务]`。它的执行逻辑一目了然，但中间步骤一旦失败，执行器缺少应对策略。
- 条件计划增加了分支判断，通常表示为带条件节点的有向图，每个节点携带操作和根据条件选择的后继节点。譬如"运行测试套件"节点可以附加规则：全部通过则构建镜像，有失败用例则通知开发者并终止。条件计划提升了环境适应性，代价是需要预判所有分支走向。
- 层级计划采用树状结构，与任务分解的层级关系对应。根节点是最高层目标，叶子节点是可执行的原子操作。它支持不同抽象等级的查看，执行器在失败时可以回溯到上层节点另选路径，而不必推翻整棵树。

选择哪种表示取决于任务的确定性和复杂度。步骤确定、无分支的任务，线性计划就够用。存在明确条件分叉的任务，条件计划更自然。具有多层抽象结构且需要局部回溯的任务，层级计划提供更灵活的控制。实践中这三种表示常嵌套使用。层级计划的叶子节点可能是线性步骤列表，线性步骤中的某个步骤又可能触发条件分支。

### 依赖分析与调度

子任务拆解好、计划结构也确定之后，第三件事情是安排执行顺序，梳理子任务之间的依赖关系，明确 A 的输出是不是 B 的输入，C 和 D 是不是真的可以同时进行。依赖关系的形式化工具是依赖图（Dependency Graph），它是一个有向无环图（DAG），图中每个节点代表一个子任务，每条有向边 $A \rightarrow B$ 表示 B 依赖 A 的输出，因此 A 必须在 B 之前完成。DAG 确保了没有循环依赖（如果 A 依赖 B 而 B 又依赖 A，两个任务就永远无法启动），这个性质本身也是对任务分解质量的一次检验，如果你画出的依赖图中出现了环，说明分解中一定存在逻辑错误。

有了依赖图之后，调度就转化为一个拓扑排序问题，找到一种节点的线性排列，使得每条边的起点都排在终点之前。拓扑排序的结果给出了一个合法的执行顺序。一个图中可能有多种拓扑排序结果，这意味着存在多个正确的执行方案，可以从中选择并行度最高的那个。拓扑排序过程中同一批入度为零的节点之间不存在依赖关系，它们可以被并行派发执行。下面的代码演示了依赖图的构建、拓扑排序以及并行组的识别。

```python runnable extract-class="DependencyScheduler, TaskNode"
from collections import deque
from dataclasses import dataclass, field


@dataclass
class TaskNode:
    """依赖图中的一个任务节点"""
    name: str
    description: str = ""
    dependencies: list["TaskNode"] = field(default_factory=list)
    dependents: list["TaskNode"] = field(default_factory=list)


class DependencyScheduler:
    """
    基于 DAG 的任务调度器

    通过拓扑排序确定执行顺序，并识别可并行执行的任务组
    """

    def __init__(self):
        self.nodes: dict[str, TaskNode] = {}

    def add_task(self, name: str, description: str = "") -> TaskNode:
        """添加一个任务节点"""
        node = TaskNode(name=name, description=description)
        self.nodes[name] = node
        return node

    def add_dependency(self, task: TaskNode, depends_on: TaskNode):
        """
        声明依赖关系：task 依赖于 depends_on
        
        即 depends_on 必须在 task 之前完成
        """
        task.dependencies.append(depends_on)
        depends_on.dependents.append(task)

    def has_cycle(self) -> bool:
        """
        检测依赖图中是否存在环路
        
        使用 DFS 三色标记法：白色=未访问，灰色=访问中，黑色=已完成
        在 DFS 过程中遇到灰色节点即表示存在环路
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self.nodes}

        def dfs(node_name: str) -> bool:
            color[node_name] = GRAY
            node = self.nodes[node_name]
            for dep in node.dependencies:
                if color[dep.name] == GRAY:
                    return True  # 发现环
                if color[dep.name] == WHITE:
                    if dfs(dep.name):
                        return True
            color[node_name] = BLACK
            return False

        for name in self.nodes:
            if color[name] == WHITE:
                if dfs(name):
                    return True
        return False

    def topological_groups(self) -> list[list[str]]:
        """
        拓扑排序并按并行组返回

        每一组内的任务没有相互依赖，可以并行执行
        组与组之间按依赖顺序排列
        """
        if self.has_cycle():
            raise ValueError("依赖图中存在环路，无法进行拓扑排序")

        in_degree = {name: len(node.dependencies) for name, node in self.nodes.items()}
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        groups = []

        while queue:
            # 当前批次：所有入度为 0 的节点可以并行执行
            current_group = list(queue)
            groups.append(current_group)
            queue.clear()

            for name in current_group:
                node = self.nodes[name]
                for dependent in node.dependents:
                    in_degree[dependent.name] -= 1
                    if in_degree[dependent.name] == 0:
                        queue.append(dependent.name)

        return groups


# 构建 CI/CD 流水线的依赖图并调度
scheduler = DependencyScheduler()

# 定义所有任务
setup_env = scheduler.add_task("配置环境", "安装 Docker 和依赖")
write_dockerfile = scheduler.add_task("编写 Dockerfile", "定义应用运行环境")
config_tests = scheduler.add_task("配置测试", "编写 pytest 配置和测试脚本")
config_ci = scheduler.add_task("配置 CI 流水线", "编写 GitHub Actions 工作流")
config_cd = scheduler.add_task("配置 CD 流水线", "编写自动部署脚本")
test_locally = scheduler.add_task("本地测试", "在本地运行测试确保通过")
push_code = scheduler.add_task("推送代码", "将代码推送到 GitHub 仓库")
verify_deploy = scheduler.add_task("验证部署", "确认测试环境运行正常")

# 声明依赖关系
scheduler.add_dependency(write_dockerfile, setup_env)
scheduler.add_dependency(config_tests, setup_env)
scheduler.add_dependency(config_ci, write_dockerfile)
scheduler.add_dependency(config_ci, config_tests)
scheduler.add_dependency(config_cd, write_dockerfile)
scheduler.add_dependency(test_locally, config_tests)
scheduler.add_dependency(test_locally, write_dockerfile)
scheduler.add_dependency(push_code, test_locally)
scheduler.add_dependency(push_code, config_ci)
scheduler.add_dependency(verify_deploy, push_code)
scheduler.add_dependency(verify_deploy, config_cd)

# 执行调度
groups = scheduler.topological_groups()
print("依赖图调度结果（同一组内的任务可以并行执行）：")
for i, group in enumerate(groups):
    print(f"  第 {i + 1} 组（并行）：{', '.join(group)}")
```

上面的输出清晰展示了 CI/CD 流水线中哪些步骤可以并行推进。"配置环境"完成后，"编写 Dockerfile"和"配置测试"两个任务因为不存在相互依赖，被归入了同一并行组，可以同时启动。推向生产环境之前，整个流水线的工程实践会相当依赖这种调度能力。规划器生成依赖图，调度器计算并行组，执行器按照组序依次派发任务，每一组内部并发执行。这就是 Planner-Executor 模式在执行层面的技术实现。

隐式依赖（Implicit Dependency）是调度中隐蔽的陷阱。两个任务在依赖图上明明没有任何边相连，表面上完全可以并行执行，但它们可能共享一个可变的资源，譬如都往同一个临时文件里写数据、都占用同一个端口、都依赖同一把互斥锁。隐式依赖的识别远比显式依赖困难，因为它不是结构层面的问题，而是运行时状态层面的问题。目前处理隐式依赖的策略有两种。悲观策略是默认假设所有同层任务都存在隐式资源竞争，全部串行执行，安全但牺牲了并行度。乐观策略是先并行执行，在执行层通过锁机制或事务机制捕获冲突后回退重试。实际生产中通常按任务的风险等级混合使用这两种策略，如对文件系统操作采用乐观并行加冲突重试，对数据库写操作采用保守串行加显式锁。

## 推理策略

计划的结构和执行顺序都是服务于推理的。面对同一个目标，从已知条件往前推出下一步操作和从目标往回推需要什么前置条件是两条截然不同的思路。这两条思路分别对应着前向推理与目标回推，它们各自适合不同类型的任务场景。

### 前向推理

**前向推理**（Forward Reasoning）从当前已知的所有信息出发，寻找一个能让我们朝目标靠近的行动，然后执行它，观察结果，再在新的状态下重复这个过程。这种策略对应着 [ReAct 模式](llm-to-agent.md#react-模式)，每一步行动之前先评估当前状态，选择最有希望的下一步，执行后根据观察结果再评估，如此循环往复。举个例子。假设 Agent 面对的任务是搞清楚为什么生产环境中某个微服务的响应延迟突然从 50ms 飙升到了 2s。前向推理的 Agent 不会一开始就尝试猜测根因，而是先拉取最近 10 分钟的监控数据，观察 CPU、内存、网络 IO 等指标。如果发现数据库查询延迟在相同时段也出现了尖峰，下一步就去检查慢查询日志。从慢查询日志中发现某条 SQL 的执行计划在昨天发生了变更，于是去查部署记录确认昨天是否有索引变更。这条推理链路完全由每一步观察到的信息驱动，每步推理都建立在已验证的事实之上，因此链路的可靠性较高。

前向推理的可靠性来自它的每一步都有充分依据，但它缺乏目标导向，推理过程可能沿着有趣但与解决问题无关的路径越走越远。譬如上面的例子中，Agent 发现 CPU 使用率上升时，可能被带偏去分析 CPU 调度的细节，而忽略了真正的问题在数据库索引上。另外，前向推理在状态空间太大时搜索效率急剧下降。如果每个状态下有数十种可能的下一步行动，盲目前向搜索很快就会遇到组合爆炸。缓解这两个局限的常见手段是引入启发式搜索（Heuristic Search），在每个决策点用评估函数给各候选行动打分，优先探索得分高的方向。此时模型扮演的正是启发函数的角色，在[思维树](../../language-models/reasoning/test-time-compute.md#树搜索)（Tree of Thoughts）等方法中，LLM 被用来对多个候选推理路径进行打分和剪枝，这实际上是前向推理与启发式搜索的结合。

### 目标回推

**目标回推**（Backward Reasoning）走的是与前向推理相反的方向。它从最终目标出发，在每一步反问要达到这个状态，需要什么前置条件，把前置条件当作新的子目标，继续追问它们的前置条件，直到所有子目标都可以用当前已有的信息或能力直接满足。

还是用定位延迟问题作为例子。目标回推的 Agent 会这样思考：最终目标是将延迟恢复到正常水平，这意味着需要先定位延迟的根因。要定位根因，需要同时拿到异常时段的指标数据和该时段内的变更记录。要拿到指标数据，需要先确定异常时段的时间范围。在这个推理链条中，每一步都在为目标的上一步铺路，Agent 先确定了时间范围，再去拉取指标和变更记录，最后在两者的交集中锁定根因。可以看出，目标回推天然适合需要前置条件和后置条件推理的任务，这与 [Planner-Executor 模式](llm-to-agent.md#planner-executor-模式)先做全局规划再执行的思路高度一致。规划器使用目标回推生成依赖树，执行器按照从叶子到根的逆序逐步推进。

目标回推同样有自己的局限。当目标本身比较模糊时（譬如"优化系统性能"），什么是"优化"、以什么指标衡量、优化到什么程度算完成，这些问题本身就不明确，目标回推找不到一个清晰的起点。目标回推另一个问题是前置条件的不唯一性，要达到某个子目标，可能有多条不同的前置条件路径，而这些路径彼此之间存在复杂的交互。目标回推在条件不唯一时需要搜索一个可能很大的解空间，如果不加启发式引导，同样会遭遇组合爆炸。

### 混合推理

现实中的 Agent 任务很少能纯粹用前向推理或目标回推单独完成。更常见的模式是把两种策略结合起来使用。目标回推负责确定方向，在任务开始时进行高层规划，产生一棵子目标树，明确先做什么、后做什么、哪些事情可以并行。前向推理负责填充细节，在执行每个子目标时，从当前状态出发，逐步展开具体操作，根据中间结果动态调整。

混合推理中最微妙的环节是判断何时切换推理方向。这个判断本身是一个元推理（Meta-Reasoning）问题。一个比较实用的经验法则是当信息充分、目标明确时使用目标回推做规划；当信息不完整、需要探索时切换到前向推理收集信息；在积累足够的信息后再回到目标回推更新规划。这与 Planner-Executor 模式中执行器遇到异常时上报给规划器请求重新规划的机制是完全对应的。在实践中，这种方向切换通常不需要设计复杂的切换逻辑，而是由 LLM 的上下文推理自然完成。当 Prompt 中包含完整的历史信息和当前状态时，模型能够自主判断下一步应该"规划"还是"行动"。

## 本章小结

规划是 Agent 将复杂目标转化为可执行步骤的核心能力。本章从任务分解讲起，介绍了 MECE 原则、计划表示的三种形式（线性、条件、层级）以及基于 DAG 的依赖分析与拓扑调度。推理策略部分对比了前向推理与目标回推，前者从已知信息出发逐步推进，后者从目标反推前置条件。实际系统中通常采用混合推理，以目标回推制定高层规划，以前向推理填充执行细节，两者在信息充分性变化时动态切换。


## 练习题

1. 对于一个"为新项目搭建 CI/CD 流水线"的任务，请画出任务分解树（至少两层）和对应的依赖 DAG 图，标注哪些子任务可以并行执行。

   <details>
   <summary>参考答案</summary>

   任务分解树（层级结构）
   - 根节点是搭建 CI/CD 流水线
     - 环境准备
       - 创建 GitHub 仓库
       - 配置 Secrets（Docker Hub 凭据）
     - 应用容器化
       - 编写 Dockerfile
       - 编写 .dockerignore
     - 自动化流水线
       - 测试阶段（pytest + lint）
       - 构建阶段（docker build + push）
       - 部署阶段（ssh 到服务器 + docker compose up）

   依赖 DAG 中，环境准备中的两个子任务可以并行；"编写 Dockerfile"是构建阶段的依赖，但与"测试阶段"的脚本编写可并行。

   </details>

2. 目标回推与 Planner-Executor 模式有什么内在联系？在什么情况下，目标回推生成的计划会在执行中频繁触发重新规划？

   <details>
   <summary>参考答案</summary>

   目标回推与 Planner-Executor 的设计哲学一致，都是先全局规划再执行。规划器使用目标回推从目标反推前置条件，生成子目标依赖树，执行器从叶子节点（前置条件全部满足的子目标）开始执行。

   频繁重新规划通常发生于规划阶段掌握的信息严重不足，导致初始计划在多个步骤上基于错误假设，环境变化速度快于执行速度，任务的前置条件之间隐含了未被识别的依赖关系。在这些情况下，提高初始信息的充分性（如在规划前先执行一轮信息收集）比频繁重规划更有效。

   </details>

