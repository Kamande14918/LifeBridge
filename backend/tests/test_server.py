import json
import pytest

import server


class DummyStore:
    def __init__(self):
        self.data = {}

    def get(self, chat_id):
        # return a simple mutable object as memory
        return {"chat_history": []}

    def set(self, chat_id, mem):
        self.data[chat_id] = mem


class DummyChain:
    def __init__(self, answer="OK", docs=None):
        self._answer = answer
        self._docs = docs or []

    def invoke(self, payload):
        return {"answer": self._answer, "source_documents": self._docs}


class DummyDoc:
    def __init__(self, title="T", contacts="911", visual=None):
        self.metadata = {"title": title, "contacts": contacts, "visual": visual}


class DummyVectorStore:
    def __init__(self, docs=None):
        self._docs = docs or []

    def as_retriever(self, search_kwargs=None):
        class Retriever:
            def __init__(self, docs):
                self.docs = docs

            def get_relevant_documents(self, q):
                return self.docs

        return Retriever(self._docs)


class DummyTTS:
    def synth(self, text):
        return "http://audio.example/test.mp3"


@pytest.fixture(autouse=True)
def isolate_server_module(monkeypatch):
    # Prevent ensure_services from attempting heavy init
    monkeypatch.setattr(server, "ensure_services", lambda: None)
    # Provide lightweight defaults that endpoints expect
    server.store = DummyStore()
    server.vectorstore = DummyVectorStore()
    server.llm = object()
    server.tts = DummyTTS()
    yield
    # cleanup
    server.store = None
    server.vectorstore = None
    server.llm = None
    server.tts = None


def test_health():
    c = server.app.test_client()
    r = c.get("/api/health")
    assert r.status_code == 200
    j = r.get_json()
    assert j["status"] == "ok"


def test_contacts_no_query():
    c = server.app.test_client()
    r = c.get("/api/contacts")
    assert r.status_code == 400


def test_contacts_with_results(monkeypatch):
    # make vectorstore return one document with metadata
    server.vectorstore = DummyVectorStore(docs=[DummyDoc(title="Snake Bite", contacts="Ambulance; 112")])
    c = server.app.test_client()
    r = c.get("/api/contacts?q=snake")
    assert r.status_code == 200
    j = r.get_json()
    assert "contacts" in j and isinstance(j["contacts"], list)


def test_chat_success(monkeypatch):
    # Patch chain_with_memory to return a dummy chain
    monkeypatch.setattr(server, "chain_with_memory", lambda vs, llm, mem: DummyChain(answer="1) Do something\n2) Do not do that", docs=[]))
    c = server.app.test_client()
    payload = {"chat_id": "abc", "message": "help me"}
    r = c.post("/api/chat", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == 200
    j = r.get_json()
    assert "steps" in j and isinstance(j["steps"], list)


def test_tts_api_disabled(monkeypatch):
    # If TTS disabled in settings, should return enabled:false
    monkeypatch.setattr(server.settings, "ENABLE_SERVER_TTS", False)
    c = server.app.test_client()
    r = c.post("/api/tts", data=json.dumps({"text": "Hi"}), content_type="application/json")
    assert r.status_code == 200
    j = r.get_json()
    assert j["enabled"] is False


def test_tts_api_enabled(monkeypatch):
    monkeypatch.setattr(server.settings, "ENABLE_SERVER_TTS", True)
    # ensure tts present
    server.tts = DummyTTS()
    c = server.app.test_client()
    r = c.post("/api/tts", data=json.dumps({"text": "Speak this"}), content_type="application/json")
    assert r.status_code == 200
    j = r.get_json()
    assert j["enabled"] is True
    assert j["audio_url"] is not None


if __name__ == "__main__":
    # Allow running tests directly with `python test_server.py` — this will
    # invoke pytest programmatically so fixtures and test collection work.
    import sys
    try:
        import pytest
    except Exception as e:
        print("pytest is not installed in this environment. Install it with:\n  pip install pytest", file=sys.stderr)
        raise
    sys.exit(pytest.main([__file__]))
