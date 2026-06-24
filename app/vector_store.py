from abc import ABC, abstractmethod
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.chunking import Chunk


# ── Embedding (shared by both backends) ──────────────────────────────────────

_embedding_model: SentenceTransformer | None = None
_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def embed(texts: list[str]) -> list[list[float]]:
    if settings.embedding_model.startswith("text-embedding-"):
        client = _get_openai_client()
        batch_size = 500
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(input=batch, model=settings.embedding_model)
            results.extend(item.embedding for item in response.data)
        return results
    else:
        model = get_embedding_model()
        return model.encode(texts, show_progress_bar=True).tolist()


# ── Abstract interface ────────────────────────────────────────────────────────

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], corpus: str) -> int:
        """Embed and store chunks. Returns number of chunks upserted."""
        ...

    @abstractmethod
    def query(self, corpus: str, query_vector: list[float], n: int) -> list[tuple]:
        """Return top-n results as (id, doc, metadata) tuples."""
        ...

    @abstractmethod
    def get_all(self, corpus: str) -> tuple[list[str], list[str], list[dict]]:
        """Return (ids, documents, metadatas) for all chunks in the corpus — used by BM25."""
        ...

    @abstractmethod
    def delete(self, corpus: str) -> None:
        """Drop the entire corpus collection so re-ingest starts clean."""
        ...


# ── ChromaDB implementation (local) ──────────────────────────────────────────

class ChromaVectorStore(VectorStore):
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_path)

    def _collection(self, corpus: str):
        return self._client.get_or_create_collection(
            name=corpus,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], corpus: str) -> int:
        collection = self._collection(corpus)
        embeddings = embed([c.text for c in chunks])

        ids = [f"{c.doc_id}_chunk_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [_chunk_metadata(c) for c in chunks]

        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        return len(chunks)

    def query(self, corpus: str, query_vector: list[float], n: int) -> list[tuple]:
        results = self._collection(corpus).query(
            query_embeddings=[query_vector], n_results=n
        )
        return list(zip(results["ids"][0], results["documents"][0], results["metadatas"][0]))

    def get_all(self, corpus: str) -> tuple[list[str], list[str], list[dict]]:
        result = self._collection(corpus).get(include=["documents", "metadatas"])
        return result["ids"], result["documents"], result["metadatas"]

    def delete(self, corpus: str) -> None:
        try:
            self._client.delete_collection(corpus)
        except Exception:
            pass


# ── Qdrant implementation (production) ───────────────────────────────────────

class QdrantVectorStore(VectorStore):
    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._Distance = Distance
        self._VectorParams = VectorParams

    def _ensure_collection(self, corpus: str) -> None:
        from qdrant_client.models import Distance, VectorParams
        existing = [c.name for c in self._client.get_collections().collections]
        if corpus not in existing:
            self._client.create_collection(
                collection_name=corpus,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, chunks: list[Chunk], corpus: str) -> int:
        from qdrant_client.models import PointStruct

        self._ensure_collection(corpus)
        embeddings = embed([c.text for c in chunks])

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c.doc_id}_chunk_{c.chunk_index}")),
                vector=embeddings[i],
                payload={**_chunk_metadata(c), "document": c.text},
            )
            for i, c in enumerate(chunks)
        ]

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=corpus,
                points=points[i : i + batch_size],
            )
        return len(chunks)

    def query(self, corpus: str, query_vector: list[float], n: int) -> list[tuple]:
        response = self._client.query_points(
            collection_name=corpus,
            query=query_vector,
            limit=n,
            with_payload=True,
        )
        return [
            (
                str(r.id),
                r.payload.get("document", ""),
                {k: v for k, v in r.payload.items() if k != "document"},
            )
            for r in response.points
        ]

    def get_all(self, corpus: str) -> tuple[list[str], list[str], list[dict]]:
        ids, docs, metadatas = [], [], []
        offset = None
        while True:
            results, offset = self._client.scroll(
                collection_name=corpus,
                with_payload=True,
                limit=1000,
                offset=offset,
            )
            for r in results:
                ids.append(str(r.id))
                docs.append(r.payload.get("document", ""))
                metadatas.append({k: v for k, v in r.payload.items() if k != "document"})
            if offset is None:
                break
        return ids, docs, metadatas

    def delete(self, corpus: str) -> None:
        try:
            self._client.delete_collection(corpus)
        except Exception:
            pass


# ── Factory ───────────────────────────────────────────────────────────────────

_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        if settings.vector_store == "qdrant":
            _store = QdrantVectorStore()
        else:
            _store = ChromaVectorStore()
    return _store


# ── Module-level wrappers (keep ingest.py and retrieval.py call sites clean) ─

def upsert_chunks(chunks: list[Chunk], corpus: str) -> int:
    return get_vector_store().upsert(chunks, corpus)


def vector_search(corpus: str, query_vector: list[float], n: int) -> list[tuple]:
    return get_vector_store().query(corpus, query_vector, n)


def get_all_documents(corpus: str) -> tuple[list[str], list[str], list[dict]]:
    return get_vector_store().get_all(corpus)


def delete_corpus(corpus: str) -> None:
    get_vector_store().delete(corpus)


# ── Shared metadata helper ────────────────────────────────────────────────────

def _chunk_metadata(chunk: Chunk) -> dict:
    return {
        "source": chunk.metadata.get("source", ""),
        "volume": chunk.metadata.get("volume", ""),
        "book": chunk.metadata.get("book", ""),
        "chapter_num": chunk.metadata.get("chapter_num", ""),
        "title": chunk.metadata.get("title", ""),
        "chapter_or_page": chunk.metadata.get("chapter_or_page", ""),
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "para_start": chunk.metadata.get("para_start", 0),
        "para_end": chunk.metadata.get("para_end", 0),
    }
