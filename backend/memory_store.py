from typing import Dict, List
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
    # Provide an actionable error message with guidance.
    raise ImportError(
        "Could not find `ConversationBufferMemory` in the installed langchain package.\n"
        "This project expects LangChain to provide `ConversationBufferMemory`.\n"
        "Please install the project's dependencies into your virtualenv and ensure you run Python from the same environment:\n"
        "  .\\venv\\Scripts\\Activate.ps1   # PowerShell activate (Windows)\n"
        "  pip install -r requirements.txt\n"
        "If you have a custom or very new/old LangChain version, you may need to install a compatible version, for example: `pip install 'langchain==1.1.0'`.\n"
    )

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
