# app/server.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from config import settings
from life_bridge_logger import logger
from memory_store import get_store, RedisChatStore
from rag_chains import load_vectorstore, build_llm, chain_with_memory, parse_answer, compute_confidence
import os
import threading

# tts_service may not exist in all setups; provide a harmless fallback so
# server can start when TTS is not implemented.
try:
    from tts_service import TTSServer
except Exception:
    class TTSServer:
        def __init__(self):
            pass
        def synth(self, text):
            return None

app = Flask(__name__)
CORS(app, origins=[settings.CORS_ORIGINS] if settings.CORS_ORIGINS != "*" else "*")

services_lock = threading.Lock()
services_ready = False
# Static for TTS files
if settings.ENABLE_SERVER_TTS:
    app.add_url_rule("/static/tts/<path:filename>", "static_tts", lambda filename: send_from_directory(settings.TTS_OUTPUT_DIR, filename))

# Serve the web UI (single-file) so testers can open the chatbot in a browser
@app.get('/')
@app.get('/ui')
def serve_ui():
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    return send_from_directory(static_dir, 'chat_ui.html')

# Defer heavy initialization (models, FAISS, Redis) until first request so
# importing this module (for tests or tooling) doesn't attempt to download
# models or connect to external services.
store = None
vectorstore = None
llm = None
tts = None

def ensure_services():
    global store, vectorstore, llm, tts, services_ready
    # Fast path: already ready
    if services_ready:
        return

    with services_lock:
        if services_ready:
            return

        logger.info("Initializing services: store, vectorstore, llm")

        # Initialize store
        try:
            store = get_store()
        except Exception as e:
            logger.warning(f"Failed to initialize store: {e}")
            store = None

        # Load vectorstore (FAISS)
        try:
            vectorstore = load_vectorstore()
        except Exception as e:
            logger.warning(f"Failed to load vectorstore: {e}")
            vectorstore = None

        # Build LLM / generator
        try:
            llm = build_llm()
        except Exception as e:
            logger.warning(f"Failed to build llm: {e}")
            llm = None

        # Initialize TTS if available
        try:
            tts = TTSServer()
        except Exception as e:
            logger.warning(f"Failed to initialize TTS server: {e}")
            tts = None

        # Mark ready only if vectorstore and llm are available
        services_ready = (vectorstore is not None and llm is not None)
        logger.info(f"Services ready: {services_ready}")


def _start_background_init():
    t = threading.Thread(target=ensure_services, daemon=True)
    t.start()


# Kick off initialization at import time (background)
_start_background_init()

@app.get("/api/health")
def health():
    # ensure services are attempted to be initialized for diagnostics
    try:
        ensure_services()
    except Exception:
        pass
    return jsonify({"status": "ok", "env": settings.ENV})

@app.post("/api/chat")
def chat():
    """
    Body:
      { "chat_id": "uuid-or-string", "message": "text", "k": 5 }
    Returns structured guidance + optional audio_url
    """
    data = request.get_json(force=True)
    chat_id = data.get("chat_id", "default")
    message = data.get("message", "").strip()
    k = int(data.get("k", settings.TOP_K))
    if not message:
        return jsonify({"error":"message required"}), 400

    # Ensure services are initialized and available
    ensure_services()
    if store is None or vectorstore is None or llm is None:
        return jsonify({"error":"server not fully initialized"}), 503

    # Load memory for chat_id
    memory = store.get(chat_id)

    # Build chain
    chain = chain_with_memory(vectorstore, llm, memory)
    # Invoke chain (LangChain CRChain expects {"question": "..."} )
    result = chain.invoke({"question": message})
    answer = result["answer"] if "answer" in result else result.get("result","")
    source_docs = result.get("source_documents", [])

    parsed = parse_answer(answer)
    confidence = compute_confidence(source_docs)

    # Safety fallback
    if confidence < 0.3:
        parsed["steps"] = [
            "Ensure scene safety and avoid hazards.",
            "Call emergency services immediately.",
            "Provide reassurance and monitor breathing until help arrives."
        ]
        parsed["warnings"] = ["Do not administer medications or perform invasive procedures."]
        parsed["escalate"] = ["If unresponsive or not breathing, begin CPR until help arrives."]

    # Persist memory if using Redis
    if isinstance(store, RedisChatStore):
        store.set(chat_id, memory)

    audio_url = None
    if settings.ENABLE_SERVER_TTS and tts is not None:
        speech_text = "\n".join(parsed["steps"])
        try:
            audio_url = tts.synth(speech_text)
        except Exception:
            audio_url = None

    response = {
        "steps": parsed["steps"],
        "warnings": parsed["warnings"],
        "escalate": parsed["escalate"],
        "contacts": parsed["contacts"],
        "visual_guide": parsed["visual_guide"],
        "sources": [doc.metadata.get("title","Untitled") for doc in source_docs],
        "confidence": confidence,
        "audio_url": audio_url
    }
    return jsonify(response)

@app.get("/api/contacts")
def contacts():
    """
    Query param: ?q=snake bite
    Returns nearest emergency contacts from top retrieved doc
    """
    ensure_services()
    if vectorstore is None:
        return jsonify({"contacts":[], "title": None})
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error":"q required"}), 400
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents(q)
    if not docs:
        return jsonify({"contacts":[],"title":None})
    md = docs[0].metadata
    contacts = md.get("contacts", "")
    return jsonify({"contacts": contacts.split("; "), "title": md.get("title"), "visual": md.get("visual")})

@app.post("/api/tts")
def tts_api():
    """
    Body: { "text": "Speak these steps..." }
    """
    if not settings.ENABLE_SERVER_TTS or tts is None:
        return jsonify({"audio_url": None, "enabled": False})
    data = request.get_json(force=True)
    text = data.get("text","").strip()
    if not text:
        return jsonify({"error":"text required"}), 400
    url = tts.synth(text)
    return jsonify({"audio_url": url, "enabled": True})

@app.post("/api/stt")
def stt_api():
    # Placeholder for server-side STT; recommend client-side Web Speech API
    return jsonify({"text":"", "confidence":0.0})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.PORT)
