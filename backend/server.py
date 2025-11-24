# app/server.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from .config import settings
from .logging import logger
from .memory_store import get_store, RedisChatStore
from .rag_chain import load_vectorstore, build_llm, chain_with_memory, parse_answer, compute_confidence
from .tts_service import TTSServer
import os

app = Flask(__name__)
CORS(app, origins=[settings.CORS_ORIGINS] if settings.CORS_ORIGINS != "*" else "*")

# Static for TTS files
if settings.ENABLE_SERVER_TTS:
    app.add_url_rule("/static/tts/<path:filename>", "static_tts", lambda filename: send_from_directory(settings.TTS_OUTPUT_DIR, filename))

store = get_store()
vectorstore = load_vectorstore()
llm = build_llm()
tts = TTSServer()

@app.get("/api/health")
def health():
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
    if settings.ENABLE_SERVER_TTS:
        speech_text = "\n".join(parsed["steps"])
        audio_url = tts.synth(speech_text)

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
    if not settings.ENABLE_SERVER_TTS:
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
