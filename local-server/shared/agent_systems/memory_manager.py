# MemoryManager 定义
# 从文档自动提取生成

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
