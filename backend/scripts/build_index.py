"""Build and save a FAISS vectorstore from the CSV data.

Run this once (or whenever your data changes) to precompute embeddings and
save a local FAISS index for fast loading by the server.

Usage:
  python scripts/build_index.py

Requirements:
  - sentence-transformers or the embedding model available
  - faiss or faiss-cpu
  - langchain_community (optional for compatibility)

This script tries to use the same embedding class as the server to remain
compatible with `rag_chains.load_vectorstore()` which calls
`FAISS.load_local(settings.INDEX_DIR, embeddings)`.
"""
import os
import sys
import csv
from pathlib import Path

# Ensure the project root (backend/) is on sys.path so imports like
# `from config import settings` work when running this script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings

def load_rows(csv_path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def build_index():
    data_dir = Path(settings.DATA_DIR)
    csv_path = data_dir / 'first_aid_sample.csv'
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    print(f"Loading CSV from {csv_path}")
    rows = load_rows(csv_path)
    texts = []
    metadatas = []
    for r in rows:
        text = (
            f"Scenario: {r.get('Scenario','')}\n"
            f"Question: {r.get('Question','')}\n"
            f"Primary Action: {r.get('Primary Action','')}\n"
            f"Emergency Contact: {r.get('Emergency Contact','')}\n"
            f"VisualGuideURL: {r.get('VisualGuideURL','')}\n"
        )
        texts.append(text)
        metadatas.append({
            'title': r.get('Scenario',''),
            'contacts': r.get('Emergency Contact',''),
            'visual': r.get('VisualGuideURL','')
        })

    print(f"Preparing embeddings using model: {settings.EMBEDDING_MODEL}")
    try:
        # Prefer the same embedding class used by the app
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        print("Computing embeddings (this may take a while)...")
        # langchain FAISS.from_documents expects Document objects (with
        # a `page_content` attribute). Construct proper Documents here.
        from langchain.schema import Document
        docs = []
        for t, m in zip(texts, metadatas):
            docs.append(Document(page_content=t, metadata=m))

        # Build FAISS index
        vs = FAISS.from_documents(docs, embeddings)
        index_dir = Path(settings.INDEX_DIR)
        index_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving index to {index_dir}")
        vs.save_local(str(index_dir))
        print("Index build complete.")
    except Exception as e:
        print("Failed to build index:", e)
        raise

if __name__ == '__main__':
    build_index()
