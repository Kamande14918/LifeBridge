import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    ENV = os.getenv("ENV", "production")
    PORT = i/nt(os.getenv("PORT", "8000"))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    # Models
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    GENERATION_MODEL = os.getenv("GENERATION_MODEL", "google/flan-t5-base")
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
    TOP_K = int(os.getenv("TOP_K", "5"))

    # Data
    DATA_DIR = os.getenv("DATA_DIR", "app/data")
    INDEX_DIR = os.getenv("INDEX_DIR", "app/data/faiss_index")

    # Memory
    USE_REDIS_MEMORY = os.getenv("USE_REDIS_MEMORY", "false").lower() == "true"
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    MEMORY_EXPIRY = int(os.getenv("MEMORY_EXPIRY", "3600"))  # seconds

    # TTS
    ENABLE_SERVER_TTS = os.getenv("ENABLE_SERVER_TTS", "false").lower() == "true"
    TTS_OUTPUT_DIR = os.getenv("TTS_OUTPUT_DIR", "app/data/tts")

settings = Settings()
