# Palantír RAG — API

FastAPI backend for the Palantír oracle. Answers questions about Middle-earth using a hybrid RAG pipeline (BM25 + vector search + reranking) over the LOTR corpus and Wikipedia lore pages, with Claude as the generation model.

## Stack

- **FastAPI** — REST API
- **ChromaDB** (local) / **Qdrant** (production) — vector store
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **rank-bm25** — keyword search
- **cross-encoder** — reranking
- **Claude** (`claude-opus-4-8`) — generation

## Prerequisites

- Python 3.11+
- An Anthropic API key

## Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd palantir-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

`.env.example`:
```
ANTHROPIC_API_KEY=your-key-here
VECTOR_STORE=chroma        # or "qdrant" for production
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Ingest

Before querying, you need to build the index. Processed corpus JSON files are under `data/processed/lotr/` and `data/processed/wiki/`.

```bash
# Ingest LOTR corpus
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"corpus": "lotr"}'

# Ingest Wikipedia corpus
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"corpus": "wiki"}'
```

Pass `"reset": true` to drop and recreate the collection before ingesting. Required when the corpus files have changed, since upsert alone won't remove stale chunks:

```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"corpus": "wiki", "reset": true}'
```

## Run locally

```bash
source .venv/bin/activate && uvicorn app.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`.

## Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is Aragorn?", "corpus": "both", "top_k": 5}'
```

## Production (Qdrant)

Set these additional env vars on Railway:

```
VECTOR_STORE=qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key
EMBEDDING_DIM=384
```

Then re-run ingest once to populate the Qdrant collections.
