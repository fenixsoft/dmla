# DependencyScheduler, TaskNode 定义
# 从文档自动提取生成

from collections import Counter, defaultdict, deque
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
