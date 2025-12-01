from typing import Dict, List, Any
import redis, json
from config import settings
from life_bridge_logger import logger

# ConversationBufferMemory is provided by langchain. Different LangChain
# releases expose the class in different modules; attempt multiple import
# locations and fall back to a helpful error message.
ConversationBufferMemory = None
import importlib
_try_paths = [
    "langchain.memory",
    "langchain.memory.buffer",
    "langchain.memory.chat_memory",
    "langchain.memory.conversation_buffer",
]
for _p in _try_paths:
    try:
        mod = importlib.import_module(_p)
        if hasattr(mod, "ConversationBufferMemory"):
            ConversationBufferMemory = getattr(mod, "ConversationBufferMemory")
            break
    except Exception:
        continue

if ConversationBufferMemory is None:
    # If LangChain's ConversationBufferMemory isn't available (different
    # langchain versions expose memory utilities in different modules),
    # provide a small, local fallback so the app can run. This fallback
    # implements the minimal interface used by this project: a
    # `.chat_memory.messages` list, `load_memory_variables(...)`, and
    # `save_context(...)`. It is intentionally simple and should be
    # replaced by the real LangChain class in production for full
    # compatibility.
    logger.warning(
        "ConversationBufferMemory not found in langchain; using lightweight fallback. "
        "Install a compatible langchain version for full features: `pip install -r requirements.txt`."
    )

    class _SimpleChatMemory:
        def __init__(self):
            self.messages = []

    class _SimpleConversationBufferMemory:
        def __init__(self, return_messages: bool = True, memory_key: str = "chat_history"):
            self.return_messages = return_messages
            self.memory_key = memory_key
            self.chat_memory = _SimpleChatMemory()

        def load_memory_variables(self, inputs: dict):
            # Return memory in the shape expected by callers
            return {self.memory_key: self.chat_memory.messages}

        def save_context(self, inputs: dict, outputs: dict):
            # Append a simple record; callers may expect dict-like objects
            try:
                self.chat_memory.messages.append({"input": inputs, "output": outputs})
            except Exception:
                # keep the fallback robust
                pass

    ConversationBufferMemory = _SimpleConversationBufferMemory

class InMemoryChatStore:
    def __init__(self):
        self.sessions: Dict[str, Any] = {}

    def get(self, chat_id: str) -> Any:
        if chat_id not in self.sessions:
            self.sessions[chat_id] = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        return self.sessions[chat_id]

class RedisChatStore:
    def __init__(self):
        logger.info(f"Connecting Redis memory: {settings.REDIS_URL}")
        self.r = redis.from_url(settings.REDIS_URL, decode_responses=True)

    def get(self, chat_id: str) -> Any:
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

    def set(self, chat_id: str, mem: Any):
        key = f"mem:{chat_id}"
        msgs = [m.dict() for m in mem.chat_memory.messages]
        self.r.setex(key, settings.MEMORY_EXPIRY, json.dumps(msgs))

def get_store():
    return RedisChatStore() if settings.USE_REDIS_MEMORY else InMemoryChatStore()
