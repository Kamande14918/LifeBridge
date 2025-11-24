from typing import Dict, List
from langchain.memory import ConversationBufferMemory
import redis, json
from .config import settings
from .logging import logger

class InMemoryChatStore:
    def __init__(self):
        self.sessions: Dict[str, ConversationBufferMemory] = {}

    def get(self, chat_id: str) -> ConversationBufferMemory:
        if chat_id not in self.sessions:
            self.sessions[chat_id] = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        return self.sessions[chat_id]

class RedisChatStore:
    def __init__(self):
        logger.info(f"Connecting Redis memory: {settings.REDIS_URL}")
        self.r = redis.from_url(settings.REDIS_URL, decode_responses=True)

    def get(self, chat_id: str) -> ConversationBufferMemory:
        mem = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        key = f"mem:{chat_id}"
        raw = self.r.get(key)
        if raw:
            try:
                msgs = json.loads(raw)
                mem.chat_memory.messages = msgs  # list of dicts via LangChain schema serialization
            except Exception:
                pass
        return mem

    def set(self, chat_id: str, mem: ConversationBufferMemory):
        key = f"mem:{chat_id}"
        msgs = [m.dict() for m in mem.chat_memory.messages]
        self.r.setex(key, settings.MEMORY_EXPIRY, json.dumps(msgs))

def get_store():
    return RedisChatStore() if settings.USE_REDIS_MEMORY else InMemoryChatStore()
