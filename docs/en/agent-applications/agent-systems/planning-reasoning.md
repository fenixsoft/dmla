# Planning and Reasoning

Deploying an application requires planning steps, debugging code requires analyzing causality, and writing a report requires organizing material. What these tasks have in common is that they rely on multi-step reasoning and forward-looking planning. In the [first article](llm-to-agent.md), we discussed how agents evolved from passive language models into systems capable of autonomous action, introducing design patterns such as ReAct and Planner-Executor. That article addressed how agents organize the loop between reasoning and action, while this article tackles a more fundamental question -- how agents decompose complex goals into executable sequences of steps.

Automated Planning is one of the oldest subfields of artificial intelligence. Long before the advent of language models, researchers were exploring how to make machines autonomously generate action plans. In 1971, Richard Fikes and Nils Nilsson at the Stanford Research Institute invented the STRIPS (Stanford Research Institute Problem Solver) planning system, the world's first system to formalize planning as state-space search. STRIPS enabled a robot named Shakey to move between rooms, push obstacles aside, and complete simple object-transport tasks. To achieve this, Fikes and Nilsson defined three essential elements of planning: an initial state, a goal state, and a set of executable operations each with preconditions and effects. This framework laid the theoretical foundation for half a century of automated planning research.

From STRIPS to today's LLM-based agents, planning technology has undergone multiple paradigm shifts. In the 1990s, Graphplan transformed planning problems into graph search, dramatically improving solution speed. In the 2000s, the rise of heuristic search planning allowed planners to handle large-scale problems involving hundreds of operations. The most recent transformation comes from large language models. The commonsense reasoning ability of LLMs means that planning no longer depends on manually encoded action models; agents can describe subtasks in natural language, reason about causal relationships between steps, and even make reasonable guesses without complete information. Nevertheless, the accumulated wisdom of classical planning remains the foundation for LLM agent planning. This chapter begins with that foundation.

## Foundations of Planning

The essence of planning is to transform a problem too complex to solve directly into a sequence of steps simple enough to execute directly. To accomplish this transformation, three progressively deeper questions must be answered: how to break a large task into smaller ones, what form to use to represent the plan once decomposition is done, and how to arrange execution order when dependencies exist between subtasks.

### Task Decomposition

Imagine you receive a requirement to set up a CI/CD pipeline for a Python machine learning project, so that it can automatically run tests, build images, and deploy to a staging environment after every code commit. Faced with such a broad goal, an experienced developer would not immediately open a terminal and start typing commands. Instead, they would first sketch the skeleton of the task in their mind: first, environment preparation (installing dependencies, configuring secrets); then, workflow definition (writing scripts for the test phase, build phase, and deploy phase); and finally, validation and troubleshooting. This process of progressively unfolding a macro-level goal into actionable steps is called **task decomposition**.

Task decomposition should neither be too coarse nor too fine-grained. If decomposition is too coarse -- for example, summarizing the entire CI/CD setup as "write a pipeline configuration file" -- each individual step remains too complex to execute, defeating the purpose of decomposition. Conversely, if installing Python dependencies is further broken down into installing NumPy, installing Pandas, installing scikit-learn, and so on, each step becomes simpler, but the overhead of coordinating dozens of tiny steps balloons, making it difficult to maintain an overall grasp of the task structure. A useful rule of thumb is to let each subtask correspond exactly to a single tool-callable operation. For instance, one subtask might be "write a Dockerfile," another "configure GitHub Actions secrets." Each subtask has a clear input, output, and completion criterion.

Structurally, task decomposition strategies can be grouped into three basic types. **Sequential decomposition** applies to tasks with a clear procedure, like following an operation manual -- steps have a sequential relationship, and the output of one step is the input of the next. **Dependency-based decomposition** applies to scenarios where subtasks can proceed independently -- for example, preparing frontend and backend Dockerfiles simultaneously, where neither depends on the other and they can progress in parallel. **Hierarchical decomposition** applies to large tasks: first, divide the top level into modules such as "testing," "building," and "deployment," then further decompose each module at the next level, forming a tree structure. Real-world tasks rarely fall into a single type; a more common approach uses hierarchical decomposition at the top level, dependency-based decomposition at the middle level to identify parallelization opportunities, and sequential decomposition at the bottom level to refine individual execution flows.

A good decomposition typically satisfies the **MECE principle** (Mutually Exclusive, Collectively Exhaustive), which originated in management consulting but applies equally well to task decomposition. **Mutually Exclusive** means subtask boundaries are clear and non-overlapping; for example, it should not happen that both "configure environment" and "install Python" involve `pip install`. **Collectively Exhaustive** means that all subtasks together exactly cover the full scope of the original goal, with no gaps. In practical agent systems, the decomposition logic is usually performed by an LLM. The MECE principle, granularity control, and decomposition strategies discussed above must be accurately conveyed to the LLM through prompts and few-shot examples, and after the decomposition is complete, the model should self-check by reviewing its own output as an auditor using the MECE criteria.

### Plan Representation

Once tasks are decomposed into subtasks, these subtasks need to be organized in a structured way, forming a sequence of operations the agent can follow like a map. This structured description of task steps and their interrelationships is called **plan representation**. The choice of plan representation affects execution efficiency, readability, and the ease of replanning. Common representation forms include the following:

- **Linear plans** are the simplest form, essentially an ordered list of steps `[step1, step2, ..., stepN]`, suitable for tasks with clear procedures and no branching. For example, a deployment process could be represented as `[pull latest code, run test suite, build Docker image, push to image registry, restart service]`. The execution logic is immediately clear, but if any intermediate step fails, the executor lacks a fallback strategy.
- **Conditional plans** add branching logic, typically represented as a directed graph with conditional nodes. Each node carries an operation and successor nodes chosen based on conditions. For example, the "run test suite" node could attach rules: if all tests pass, build the image; if any tests fail, notify the developer and terminate. Conditional plans improve environmental adaptability at the cost of needing to anticipate all branch outcomes.
- **Hierarchical plans** adopt a tree structure aligned with the hierarchy of task decomposition. The root node is the highest-level goal, and leaf nodes are executable atomic operations. They support viewing at different levels of abstraction, and the executor can backtrack to an upper-level node to choose an alternative path upon failure, without having to discard the entire tree.

Which representation to choose depends on the certainty and complexity of the task. For tasks with deterministic steps and no branches, linear plans suffice. For tasks with clear conditional branches, conditional plans are more natural. For tasks with multi-level abstraction structures requiring local backtracking, hierarchical plans offer more flexible control. In practice, these three representations are often nested. The leaf nodes of a hierarchical plan may be a linear step list, and a particular step within a linear sequence may trigger a conditional branch.

### Dependency Analysis and Scheduling

Once subtasks are decomposed and the plan structure is determined, the third task is to arrange the execution order: untangle the dependency relationships between subtasks, identify whether A's output is B's input, and determine whether C and D can genuinely proceed simultaneously. The formal tool for representing dependencies is the **dependency graph**, a directed acyclic graph (DAG) where each node represents a subtask and each directed edge $A \rightarrow B$ indicates that B depends on A's output, so A must complete before B. The DAG guarantees no circular dependencies (if A depends on B and B depends on A, neither task can ever start). This property itself serves as a quality check on task decomposition: if your dependency graph contains a cycle, there must be a logical error in the decomposition.

With the dependency graph in hand, scheduling becomes a **topological sorting** problem: finding a linear ordering of nodes such that the starting point of every edge comes before its endpoint. The result of topological sorting provides a valid execution order. A given graph may have multiple valid topological orderings, which means multiple correct execution plans exist, and the one with the highest degree of parallelism can be selected. During topological sorting, nodes whose in-degree reaches zero at the same time have no dependencies on each other and can be dispatched for parallel execution. The following code demonstrates dependency graph construction, topological sorting, and the identification of parallel groups.

```python runnable extract-class="DependencyScheduler, TaskNode"
from collections import deque
from dataclasses import dataclass, field


@dataclass
class TaskNode:
    """A task node in the dependency graph"""
    name: str
    description: str = ""
    dependencies: list["TaskNode"] = field(default_factory=list)
    dependents: list["TaskNode"] = field(default_factory=list)


class DependencyScheduler:
    """
    A DAG-based task scheduler

    Determines execution order via topological sorting and identifies
    groups of tasks that can be executed in parallel
    """

    def __init__(self):
        self.nodes: dict[str, TaskNode] = {}

    def add_task(self, name: str, description: str = "") -> TaskNode:
        """Add a task node"""
        node = TaskNode(name=name, description=description)
        self.nodes[name] = node
        return node

    def add_dependency(self, task: TaskNode, depends_on: TaskNode):
        """
        Declare that task depends on depends_on
        
        That is, depends_on must complete before task can start
        """
        task.dependencies.append(depends_on)
        depends_on.dependents.append(task)

    def has_cycle(self) -> bool:
        """
        Detect whether the dependency graph contains a cycle
        
        Uses the DFS three-color marking method: WHITE=unvisited, GRAY=visiting, BLACK=finished
        Encountering a gray node during DFS indicates a cycle
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self.nodes}

        def dfs(node_name: str) -> bool:
            color[node_name] = GRAY
            node = self.nodes[node_name]
            for dep in node.dependencies:
                if color[dep.name] == GRAY:
                    return True  # cycle detected
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
        Topologically sort and return results grouped by parallel stage

        Tasks within the same group have no mutual dependencies and can be executed in parallel
        Groups are ordered by dependency sequence
        """
        if self.has_cycle():
            raise ValueError("The dependency graph contains a cycle; topological sorting cannot proceed")

        in_degree = {name: len(node.dependencies) for name, node in self.nodes.items()}
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        groups = []

        while queue:
            # Current batch: all nodes with in-degree 0 can be executed in parallel
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


# Build the dependency graph for a CI/CD pipeline and schedule it
scheduler = DependencyScheduler()

# Define all tasks
setup_env = scheduler.add_task("Configure Environment", "Install Docker and dependencies")
write_dockerfile = scheduler.add_task("Write Dockerfile", "Define the application runtime environment")
config_tests = scheduler.add_task("Configure Tests", "Write pytest configuration and test scripts")
config_ci = scheduler.add_task("Configure CI Pipeline", "Write GitHub Actions workflow")
config_cd = scheduler.add_task("Configure CD Pipeline", "Write automated deployment script")
test_locally = scheduler.add_task("Test Locally", "Run tests locally to ensure they pass")
push_code = scheduler.add_task("Push Code", "Push code to GitHub repository")
verify_deploy = scheduler.add_task("Verify Deployment", "Confirm staging environment is running correctly")

# Declare dependencies
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

# Execute scheduling
groups = scheduler.topological_groups()
print("Dependency graph scheduling result (tasks in the same group can be executed in parallel):")
for i, group in enumerate(groups):
    print(f"  Group {i + 1} (parallel): {', '.join(group)}")
```

The output above clearly shows which steps in the CI/CD pipeline can proceed in parallel. Once "Configure Environment" is complete, "Write Dockerfile" and "Configure Tests" have no mutual dependencies and are placed in the same parallel group, so they can start simultaneously. Before pushing to production, the engineering practice of a pipeline heavily relies on this scheduling capability. The planner generates a dependency graph, the scheduler computes parallel groups, and the executor dispatches tasks group by group, executing within each group concurrently. This is the technical realization of the Planner-Executor pattern at the execution level.

**Implicit dependencies** are a subtle trap in scheduling. Two tasks may have no edges connecting them in the dependency graph, appearing perfectly parallelizable on the surface, but they might share a mutable resource -- for example, both writing to the same temporary file, both occupying the same port, or both contending for the same mutex lock. Identifying implicit dependencies is far more difficult than identifying explicit ones, because the issue is not structural but a matter of runtime state. Two strategies currently exist for handling implicit dependencies. The pessimistic strategy assumes that all sibling tasks have implicit resource contention and executes them serially -- safe, but at the expense of parallelism. The optimistic strategy executes them in parallel first and relies on locking or transactional mechanisms at the execution layer to catch conflicts and retry. In practice, production systems mix these two strategies according to task risk level: for example, optimistic parallelism with conflict retry for filesystem operations, and conservative serialization with explicit locks for database writes.

## Reasoning Strategies

The structure of the plan and the execution order both serve reasoning. Given the same goal, working forward from known conditions to determine the next action and working backward from the goal to determine what preconditions are needed are two fundamentally different approaches. These correspond to forward reasoning and backward reasoning, each suited to different types of tasks.

### Forward Reasoning

**Forward reasoning** starts from all currently known information, finds an action that moves us closer to the goal, executes it, observes the result, and repeats the process from the new state. This strategy corresponds to the [ReAct pattern](llm-to-agent.md#the-react-pattern), where before each action, the agent assesses the current state, selects the most promising next step, executes it, and reassesses based on the observation -- and so on in a cycle. For example, suppose an agent faces the task of figuring out why the response latency of a microservice in production suddenly spiked from 50 ms to 2 s. A forward-reasoning agent would not start by guessing the root cause. Instead, it would first pull monitoring data from the last 10 minutes, observe metrics such as CPU, memory, and network I/O. If it finds that database query latency also spiked during the same period, it would next check the slow query log. From the slow query log, it discovers that the execution plan for a particular SQL statement changed yesterday, so it checks the deployment records to confirm whether an index change was made. This reasoning chain is entirely driven by information observed at each step, and every inference is built on verified facts, giving the chain high reliability.

The reliability of forward reasoning stems from every step having a solid basis, but it lacks goal orientation, and the reasoning process may wander down interesting yet irrelevant paths. For instance, in the example above, upon noticing a CPU usage increase, the agent might be led astray into analyzing CPU scheduling details, missing the real issue with the database index. Additionally, forward reasoning suffers from sharply diminishing search efficiency when the state space is large. If there are dozens of possible next actions from each state, blind forward search quickly runs into combinatorial explosion. A common remedy for both limitations is to introduce heuristic search: at each decision point, an evaluation function scores each candidate action, and the direction with the highest score is explored first. The model plays the role of the heuristic function here. In methods such as [Tree of Thoughts](../../language-models/reasoning/test-time-compute.md#tree-search), the LLM is used to score and prune multiple candidate reasoning paths -- this is essentially a combination of forward reasoning and heuristic search.

### Backward Reasoning

**Backward reasoning** proceeds in the opposite direction from forward reasoning. Starting from the final goal, at each step it asks: to reach this state, what preconditions are needed? It then treats those preconditions as new subgoals and continues asking for their preconditions, until all subgoals can be satisfied directly by currently available information or capabilities.

Take the latency troubleshooting example again. A backward-reasoning agent would think as follows: the ultimate goal is to restore latency to normal levels, which means the root cause of the latency must first be identified. To identify the root cause, both the metric data from the anomalous period and the change records from that same period must be obtained. To obtain the metric data, the time range of the anomalous period must first be determined. In this reasoning chain, each step paves the way for the step before it. The agent first determines the time range, then fetches metrics and change records, and finally pinpoints the root cause at their intersection. As can be seen, backward reasoning is naturally suited to tasks requiring precondition and postcondition reasoning, which aligns closely with the [Planner-Executor pattern](llm-to-agent.md#the-planner-executor-pattern), where a global plan is made before execution. The planner uses backward reasoning to generate a dependency tree, and the executor proceeds step by step in reverse order, from leaves to root.

Backward reasoning also has its limitations. When the goal itself is vague (e.g., "optimize system performance"), what constitutes "optimization," what metrics to use, and how much improvement is sufficient are themselves unclear -- backward reasoning cannot find a clear starting point. Another issue is the non-uniqueness of preconditions. Achieving a subgoal may have multiple different precondition paths, and these paths may interact with each other in complex ways. When preconditions are not unique, backward reasoning must search through a potentially large solution space; without heuristic guidance, it too runs into combinatorial explosion.

### Hybrid Reasoning

Real-world agent tasks can rarely be completed using purely forward or purely backward reasoning. The more common pattern is to combine both strategies. Backward reasoning handles direction: at the start of the task, it performs high-level planning, producing a tree of subgoals that clarifies what to do first, what to do next, and what can be done in parallel. Forward reasoning handles the details: when executing each subgoal, it starts from the current state, incrementally unfolds concrete operations, and adjusts dynamically based on intermediate results.

The most subtle aspect of hybrid reasoning is deciding when to switch reasoning direction. This decision itself is a meta-reasoning problem. A practical rule of thumb is to use backward reasoning for planning when information is sufficient and the goal is clear; switch to forward reasoning to gather information when information is incomplete and exploration is needed; then return to backward reasoning to update the plan once enough information has been accumulated. This directly corresponds to the mechanism in the Planner-Executor pattern where the executor, upon encountering an exception, reports it to the planner and requests replanning. In practice, such directional switching typically does not require designing complex switching logic; rather, it is naturally accomplished by the LLM's contextual reasoning. When the prompt contains complete historical information and the current state, the model can autonomously determine whether the next step should be "planning" or "action."

## Summary

Planning is the core capability that enables agents to transform complex goals into executable sequences of steps. This chapter began with task decomposition, introducing the MECE principle, three forms of plan representation (linear, conditional, and hierarchical), and DAG-based dependency analysis with topological scheduling. The reasoning strategies section compared forward reasoning and backward reasoning: the former incrementally advances from known information, while the latter derives preconditions by working backward from the goal. Practical systems typically employ hybrid reasoning, using backward reasoning for high-level planning and forward reasoning to fill in execution details, dynamically switching between the two as information sufficiency changes.


## Exercises

1. For a task titled "Set up a CI/CD pipeline for a new project," draw a task decomposition tree (at least two levels) and the corresponding dependency DAG, marking which subtasks can be executed in parallel.

   <details>
   <summary>Reference Answer</summary>

   Task decomposition tree (hierarchical structure)
   - Root: Set up CI/CD pipeline
     - Environment Preparation
       - Create GitHub repository
       - Configure Secrets (Docker Hub credentials)
     - Application Containerization
       - Write Dockerfile
       - Write .dockerignore
     - Automation Pipeline
       - Testing phase (pytest + lint)
       - Build phase (docker build + push)
       - Deploy phase (ssh to server + docker compose up)

   In the dependency DAG, the two subtasks under "Environment Preparation" can run in parallel; "Write Dockerfile" is a dependency of the build phase but can proceed in parallel with writing the testing phase scripts.

   </details>

2. What is the intrinsic connection between backward reasoning and the Planner-Executor pattern? Under what circumstances would a plan generated by backward reasoning frequently trigger replanning during execution?

   <details>
   <summary>Reference Answer</summary>

   Backward reasoning shares the same design philosophy as Planner-Executor: both emphasize making a global plan before execution. The planner uses backward reasoning to derive preconditions from the goal, generating a subgoal dependency tree, and the executor starts executing from the leaf nodes (subgoals whose preconditions are all satisfied).

   Frequent replanning typically occurs when the planning phase has severely insufficient information, causing the initial plan to be based on incorrect assumptions across multiple steps; when the environment changes faster than execution; or when there are unrecognized implicit dependencies between the task's preconditions. In these cases, improving the adequacy of initial information (e.g., running a round of information gathering before planning) is more effective than frequent replanning.

   </details>
