# WorkflowEngine 定义
# 从文档自动提取生成

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
