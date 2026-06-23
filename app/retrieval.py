import json
from pathlib import Path

from app.models import Source
from app.vector_store import embed, vector_search
from app.bm25_index import bm25_search
from app.reranker import rerank

_POOL = 20
_DATA_DIR = Path("data/processed")
_WINDOW = 2  # paragraphs of context added before and after the matched child chunk


def _expand_to_window(doc_id: str, para_start: int, para_end: int, fallback: str) -> str:
    """Load the source document and return the matched paragraphs plus surrounding context."""
    try:
        corpus = "wiki" if doc_id.startswith("wiki") else "lotr"
        text = json.loads((_DATA_DIR / corpus / f"{doc_id}.json").read_text())["text"]
        delimiter = "\n\n" if "\n\n" in text else "\n"
        paragraphs = [p.strip() for p in text.split(delimiter) if p.strip()]
        start = max(0, para_start - _WINDOW)
        end = min(len(paragraphs), para_end + _WINDOW + 1)
        return " ".join(paragraphs[start:end])
    except Exception:
        return fallback


def _vector_search(corpus: str, query_vector: list[float], n: int) -> list[tuple]:
    """Return top-n results as (id, doc, metadata) tuples."""
    return vector_search(corpus, query_vector, n)


def _rrf_merge(vector: list[tuple], bm25: list[tuple], k: int = 60) -> list[tuple]:
    """
    Reciprocal Rank Fusion — merge two ranked lists without needing comparable scores.

    For each list, every item at rank r contributes 1/(k + r + 1) to its total score.
    Items appearing in both lists accumulate score from both — they float to the top.
    k=60 is the standard constant that dampens the influence of very high ranks.
    """
    scores: dict[str, float] = {}
    lookup: dict[str, tuple] = {}

    for rank, (id_, doc, meta) in enumerate(vector):
        scores[id_] = scores.get(id_, 0) + 1 / (k + rank + 1)
        lookup[id_] = (doc, meta)

    for rank, (id_, doc, meta, _) in enumerate(bm25):
        scores[id_] = scores.get(id_, 0) + 1 / (k + rank + 1)
        lookup[id_] = (doc, meta)

    merged = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [(id_, *lookup[id_]) for id_ in merged]


def retrieve(query: str, corpus: str, top_k: int) -> list[Source]:
    n = max(_POOL, top_k * 4)

    query_vector = embed([query])[0]

    corpora = ["lotr", "wiki"] if corpus == "both" else [corpus]

    vector_results = []
    bm25_results = []
    for c in corpora:
        vector_results += _vector_search(c, query_vector, n)
        bm25_results += bm25_search(c, query, n)

    candidates = _rrf_merge(vector_results, bm25_results)
    ranked = rerank(query, candidates)

    # Deduplicate by doc_id — multiple chunks from the same document produce
    # overlapping windows, so keep only the top-ranked chunk per document.
    seen: set[str] = set()
    deduped = []
    for row in ranked:
        doc_id = row[2].get("doc_id", row[0])
        if doc_id not in seen:
            seen.add(doc_id)
            deduped.append(row)
        if len(deduped) >= top_k:
            break

    return [
        Source(
            text=_expand_to_window(
                meta.get("doc_id", ""),
                meta.get("para_start", 0),
                meta.get("para_end", 0),
                fallback=doc,
            ),
            source=meta.get("source", ""),
            chapter_or_page=meta.get("chapter_or_page", ""),
            score=score,
        )
        for _, doc, meta, score in deduped
    ]
