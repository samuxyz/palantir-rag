import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from app.models import IngestRequest, IngestResponse
from app.chunking import chunk_by_paragraph
from app.vector_store import upsert_chunks, delete_corpus

router = APIRouter(prefix="/ingest", tags=["ingest"])

DATA_DIR = Path("data/processed")
MIN_WORDS = 30
MAX_WORDS = 120


@router.post("", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    corpus_dir = DATA_DIR / request.corpus

    if not corpus_dir.exists():
        raise HTTPException(status_code=404, detail=f"No data found for corpus '{request.corpus}'. Run the ingestion script first.")

    if request.reset:
        delete_corpus(request.corpus)

    docs = [json.loads(f.read_text()) for f in sorted(corpus_dir.glob("*.json"))]

    all_chunks = []
    for doc in docs:
        chunks = chunk_by_paragraph(
            text=doc["text"],
            min_words=MIN_WORDS,
            max_words=MAX_WORDS,
            doc_id=doc["id"],
            metadata={
                "source": doc.get("source", ""),
                "volume": doc.get("volume", ""),
                "book": doc.get("book", ""),
                "chapter_num": doc.get("chapter_num", ""),
                "title": doc.get("title", ""),
                "chapter_or_page": doc.get("chapter_or_page", ""),
            },
        )
        all_chunks.extend(chunks)

    upsert_chunks(all_chunks, corpus=request.corpus)

    return IngestResponse(
        corpus=request.corpus,
        chunks_indexed=len(all_chunks),
        message=f"Indexed {len(docs)} documents into {len(all_chunks)} chunks.",
    )
