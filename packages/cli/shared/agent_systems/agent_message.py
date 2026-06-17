# AgentMessage 定义
# 从文档自动提取生成

import json
import time
import uuid
from enum import Enum

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
