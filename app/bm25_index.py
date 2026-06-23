from rank_bm25 import BM25Okapi
from app.vector_store import get_all_documents

# One BM25 index per corpus, built lazily on first query and cached for the server lifetime.
# Built by pulling all documents from the active vector store — so ingest must run before the first query.
_indexes: dict[str, tuple] = {}


def get_bm25_index(corpus: str) -> tuple:
    if corpus not in _indexes:
        ids, docs, metadatas = get_all_documents(corpus)
        tokenized = [doc.lower().split() for doc in docs]
        _indexes[corpus] = (BM25Okapi(tokenized), ids, docs, metadatas)
    return _indexes[corpus]


def bm25_search(corpus: str, query: str, n: int) -> list[tuple]:
    """Return top-n results as (id, doc, metadata, bm25_score) tuples."""
    bm25, ids, docs, metadatas = get_bm25_index(corpus)
    scores = bm25.get_scores(query.lower().split())
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [(ids[i], docs[i], metadatas[i], float(scores[i])) for i in top_n]
