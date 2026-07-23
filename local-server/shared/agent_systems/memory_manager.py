# MemoryManager 定义
# 从文档自动提取生成

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
