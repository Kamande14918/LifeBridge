import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag_chains import load_vectorstore, _extract_primary_actions_from_doc
from config import settings

if __name__ == '__main__':
    vs = load_vectorstore()
    q = "How do I control severe bleeding from a traffic accident?"
    print('Query:', q)
    try:
        if hasattr(vs, 'as_retriever'):
            retr = vs.as_retriever(search_kwargs={'k': 3})
            docs = retr.get_relevant_documents(q)
        else:
            docs = vs.similarity_search(q, k=3)
    except Exception as e:
        print('Retrieval failed:', e)
        docs = []
    print('Retrieved', len(docs), 'docs')
    for i, d in enumerate(docs):
        print('--- DOC', i)
        print('page_content preview:', repr(getattr(d, 'page_content', '')[:300]))
        print('title:', d.metadata if hasattr(d, 'metadata') else None)
        steps = _extract_primary_actions_from_doc(d)
        print('extracted steps:', steps)
