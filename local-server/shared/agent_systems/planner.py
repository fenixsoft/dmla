# Planner 定义
# 从文档自动提取生成

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
