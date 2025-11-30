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
    gen_pipe = pipeline("text2text-generation", model=settings.GENERATION_MODEL, max_new_tokens=settings.MAX_NEW_TOKENS)
    return HuggingFacePipeline(pipeline=gen_pipe)

def build_prompt():
    # Few-shot style instructions tuned for structured first-aid answers
    template = """You are a First Aid assistant for LifeBridge First Aid in  Kenya. Use ONLY the provided context from trusted sources.
Return four sections:
1) Numbered steps (short imperative sentences).
2) 'Do not' warnings.
3) When to escalate (call emergency services).
4) Emergency contacts (from context).

If visual guide URLs exist, add: Visual guide: <URL>

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
    return chain

def parse_answer(text: str):
    # Parse sections robustly
    steps, warnings, escalate, contacts, visual = [], [], [], [], None
    section = None
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
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
    # Heuristic: more sources -> higher confidence
    base = min(len(source_docs) / settings.TOP_K, 1.0)
    return round(0.4 + 0.6 * base, 3)
