import math
from sentence_transformers import CrossEncoder

# Lightweight cross-encoder trained on MS MARCO passage ranking.
# Sees (query, chunk) together — more accurate than bi-encoder but too slow
# to run against all chunks, so we only use it on the ~20 RRF candidates.
_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_MODEL)
    return _reranker


def rerank(query: str, candidates: list[tuple]) -> list[tuple]:
    """
    Score each (id, doc, metadata) candidate against the query.
    Returns candidates sorted by relevance, with a 0-1 score appended.

    The cross-encoder outputs raw logits (unbounded). We apply sigmoid
    to normalise them to 0-1 so the score field stays interpretable.
    """
    if not candidates:
        return []
    reranker = get_reranker()
    pairs = [(query, doc) for _, doc, _ in candidates]
    logits = reranker.predict(pairs)
    scored = [
        (id_, doc, meta, round(1 / (1 + math.exp(-float(logit))), 4))
        for (id_, doc, meta), logit in zip(candidates, logits)
    ]
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored
