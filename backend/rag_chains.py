from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from config import settings
from life_bridge_logger import logger

def load_vectorstore() -> FAISS:
    logger.info(f"Loading FAISS index from {settings.INDEX_DIR}")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    vs = FAISS.load_local(settings.INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vs

def build_llm():
    logger.info(f"Loading generator: {settings.GENERATION_MODEL}")
    # Use low temperature and deterministic decoding to reduce hallucinations.
    # `do_sample=False` ensures greedy decoding where possible.
    gen_pipe = pipeline(
        "text2text-generation",
        model=settings.GENERATION_MODEL,
        max_new_tokens=settings.MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)

def build_prompt():
    # Few-shot style instructions tuned for structured first-aid answers.
    # Strong grounding instructions: insist on using ONLY the provided
    # context and include inline citations. If the answer cannot be found
    # in the provided context, respond with a short safe fallback and the
    # string "INSUFFICIENT_CONTEXT" so callers can detect low-coverage
    # answers.
    template = """You are a First Aid assistant for LifeBridge First Aid in Kenya.
Use ONLY the provided context (do NOT use prior model knowledge). If the
context does not contain an answer, do NOT guess — instead return the
phrase: INSUFFICIENT_CONTEXT and then a short emergency safety fallback.

Return four sections, separated clearly:
1) Numbered steps (short imperative sentences). After any step that
   depends on a retrieved source, include the source title in square
   brackets, e.g. "Apply pressure to the wound. [First Aid Manual]".
2) 'Do not' warnings.
3) When to escalate (call emergency services).
4) Emergency contacts (from context).

If visual guide URLs exist in the context, add: Visual guide: <URL>

Strict rules:
- Use ONLY the provided context. Do not invent facts or sources.
- If you cannot answer from the context, output exactly the word
  INSUFFICIENT_CONTEXT on its own line, then a brief safety fallback.
- Do not repeat or pad steps with filler or repeated phrases.

Context:
{context}

Chat history:
{chat_history}

User question:
{question}
"""
    return PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])

def chain_with_memory(vectorstore, llm, memory):
    prompt = build_prompt()
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K})
    # ConversationalRetrievalChain injects memory chat history and retrieved docs into prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        # Ensure memory knows which output to persist when the chain returns
        # multiple keys (e.g. 'answer' and 'source_documents'). Setting
        # output_key to 'answer' tells LangChain to store the textual
        # answer into the conversation memory.
        output_key="answer",
        return_source_documents=True,
        verbose=False,
    )
    # LangChain sometimes attempts to save to memory automatically and may
    # raise if multiple output keys are present. To keep full memory
    # functionality while avoiding ambiguous automatic saves, wrap the
    # chain so we manually control what gets written to memory.

    class _MemoryProxy:
        """A thin proxy that exposes the read methods used by chains but
        no-op's the save_context call to prevent automatic ambiguous writes.
        """
        def __init__(self, real):
            self._real = real

        def load_memory_variables(self, inputs: Dict[str, Any]):
            return self._real.load_memory_variables(inputs)

        def __getattr__(self, name):
            # Delegate other attributes to the real memory (e.g., chat_memory)
            return getattr(self._real, name)

        def save_context(self, *args, **kwargs):
            # Intentionally no-op: we'll save explicitly after invoke
            return None

    class _WrappedChain:
        def __init__(self, inner_chain, real_memory):
            self._chain = inner_chain
            self._real_memory = real_memory
            # Replace the chain's memory with the proxy to prevent auto-save
            if real_memory is not None:
                self._chain.memory = _MemoryProxy(real_memory)
                # Also monkeypatch the real memory's save_context so that if any
                # internal chain tries to call it directly, it will only store
                # the 'answer' field (avoids ambiguous multiple output keys).
                if hasattr(real_memory, 'save_context'):
                    original_save = real_memory.save_context

                    def _safe_save(inputs, outputs):
                        try:
                            # If outputs contains multiple keys, pick 'answer'
                            if isinstance(outputs, dict) and 'answer' in outputs:
                                original_save(inputs, {'answer': outputs.get('answer')})
                            else:
                                # If single key or not a dict, pass through
                                original_save(inputs, outputs)
                        except Exception as e:
                            logger.warning(f"Wrapped memory.save_context failed: {e}")

                    # Replace method
                    try:
                        real_memory.save_context = _safe_save
                    except Exception:
                        # Some memory implementations may not allow assignment; ignore
                        pass

        def invoke(self, inputs: Dict[str, Any]):
            # Call the underlying chain
            result = self._chain.invoke(inputs)
            # After successful invoke, persist only the 'answer' into memory if
            # the real memory supports save_context and wasn't already called.
            try:
                if self._real_memory is not None and hasattr(self._real_memory, 'save_context'):
                    # Call with a single-key output to avoid ambiguity
                    self._real_memory.save_context(inputs, {'answer': result.get('answer')})
            except Exception:
                logger.warning("Failed to save chat memory (non-fatal)")
            return result

    return _WrappedChain(chain, memory)

def parse_answer(text: str):
    # Parse sections robustly
    steps, warnings, escalate, contacts, visual = [], [], [], [], None
    section = None
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        # Detect the explicit INSUFFICIENT_CONTEXT marker and return fallback
        if ln == "INSUFFICIENT_CONTEXT":
            return {
                "steps": [],
                "warnings": [],
                "escalate": ["INSUFFICIENT_CONTEXT"],
                "contacts": [],
                "visual_guide": None,
            }
        low = ln.lower()
        if low.startswith("1)") or "steps" in low:
            section = "steps"; continue
        if low.startswith("2)") or "do not" in low:
            section = "warnings"; continue
        if low.startswith("3)") or "escalate" in low or "call emergency" in low:
            section = "escalate"; continue
        if low.startswith("4)") or "contacts" in low:
            section = "contacts"; continue
        if "visual guide:" in low and "http" in low:
            visual = ln.split(":", 1)[1].strip()
            continue
        if section == "steps": steps.append(ln)
        elif section == "warnings": warnings.append(ln)
        elif section == "escalate": escalate.append(ln)
        elif section == "contacts": contacts.append(ln)

    return {
        "steps": steps or [text],
        "warnings": warnings,
        "escalate": escalate,
        "contacts": contacts,
        "visual_guide": visual,
    }

def compute_confidence(source_docs: List[Document]) -> float:
    if not source_docs:
        return 0.0
    # Heuristic: more non-empty sources -> higher confidence. If source
    # titles are empty (or all blank), reduce confidence because the chain
    # may be hallucinating sources.
    titles = [getattr(d.metadata, 'title', None) if hasattr(d, 'metadata') else d.metadata.get('title') if isinstance(d.metadata, dict) else None for d in source_docs]
    non_empty = sum(1 for t in titles if t)
    base = min(non_empty / max(settings.TOP_K, 1), 1.0)
    # Scale between 0.2 and 1.0 to avoid presenting overly confident values
    return round(0.2 + 0.8 * base, 3)
