# MessageBus 定义
# 从文档自动提取生成

import heapq
from collections import defaultdict

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
