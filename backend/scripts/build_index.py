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
        # Robust CSV reading: skip leading blank lines and ensure header
        # is read correctly even if the file starts with an empty line.
        reader = csv.reader(f)
        header = None
        for row in reader:
            # skip empty rows
            if not row or all((not c or str(c).strip() == '') for c in row):
                continue
            # first non-empty row is the header
            header = [h.strip() for h in row]
            break
        if header is None:
            return rows
        # Now read remaining rows and map to header
        for row in reader:
            if not row or all((not c or str(c).strip() == '') for c in row):
                continue
            # If row length mismatches header, pad with empty strings
            if len(row) < len(header):
                row = row + [''] * (len(header) - len(row))
            mapped = {h: v for h, v in zip(header, row)}
            rows.append(mapped)
    return rows

def build_index():
    # Resolve DATA_DIR relative to the project root when a relative path
    # is used in settings. Also try fallback locations commonly used in
    # this repo so the script works when run from `backend/scripts`.
    candidates = []
    configured = Path(settings.DATA_DIR)
    if configured.is_absolute():
        candidates.append(configured)
    else:
        candidates.append(ROOT / configured)
        candidates.append(ROOT / 'data')
        candidates.append(ROOT / 'app' / 'data')

    csv_path = None
    for d in candidates:
        p = Path(d) / 'first_aid_sample.csv'
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        tried = ', '.join(str((Path(d) / 'first_aid_sample.csv')) for d in candidates)
        raise RuntimeError(f"CSV not found; checked: {tried}")

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
            'visual': r.get('VisualGuideURL',''),
            'primary_action': r.get('Primary Action',''),
            'question': r.get('Question','')
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
        # The `Document` import path changed across LangChain versions.
        # Try the common locations, and fall back to `FAISS.from_texts`
        # when a Document class is not available.
        try:
            from langchain.schema import Document  # new path
        except Exception:
            try:
                from langchain.docstore.document import Document  # older path
            except Exception:
                Document = None

        if Document is not None:
            docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
            # Build FAISS index from Document objects
            vs = FAISS.from_documents(docs, embeddings)
        else:
            # Try to build directly from texts (some FAISS wrappers support this)
            try:
                vs = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            except Exception as e_texts:
                raise RuntimeError(
                    "Could not construct Document objects nor use FAISS.from_texts. "
                    "Install a compatible `langchain` (or `langchain-huggingface`) and retry. "
                    f"Underlying error: {e_texts}"
                )
        # Resolve index directory similarly so save/load use the same path
        configured_index = Path(settings.INDEX_DIR)
        if configured_index.is_absolute():
            index_dir = configured_index
        else:
            index_dir = ROOT / configured_index

        index_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving index to {index_dir}")
        vs.save_local(str(index_dir))
        print("Index build complete.")
    except Exception as e:
        print("Failed to build index:", e)
        raise

if __name__ == '__main__':
    build_index()
