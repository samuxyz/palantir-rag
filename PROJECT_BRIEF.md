# Project Brief: RAG Learning Project (FastAPI + LlamaIndex/raw Python)

## Goal
This is a **learning project**, not a "build it for me" project. The purpose is to
understand RAG mechanics (chunking, embedding, retrieval, reranking, generation) deeply
by implementing them, not to get a finished app as fast as possible.

**Important constraint for you (Claude Code): scaffold and assist, don't author the core
RAG logic.** Specifically:
- DO: set up project structure, dependencies, boilerplate FastAPI routes, Pydantic
  models, config/env handling, Dockerfile, test harness scaffolding, Wikipedia API
  fetch script.
- DO NOT: write the chunking algorithm, retrieval logic, or prompt-assembly logic for
  me wholesale. If asked to help with these, explain the approach, show small
  illustrative snippets or pseudocode, and let me write the real implementation. Always
  explain *why*, not just produce working code.
- If I ask you to "just implement" a core RAG piece, push back once and ask if I want
  to write it myself first with your guidance instead.

## Architecture

```
Next.js (UI + API routes) --> FastAPI (RAG service, internal-only) --> Chroma + Claude API
```

- Phase 1 (current): FastAPI service built and tested **standalone** — no Next.js, no
  auth, tested via curl / Postman / FastAPI's built-in Swagger docs (`/docs`).
- Phase 2 (later, not now): Next.js frontend + API routes calling FastAPI.
- Express is NOT part of this stack — deliberately dropped in favor of Next.js API
  routes talking directly to FastAPI.

## Stack
- **API**: FastAPI
- **RAG approach**: Build raw/manual first (explicit chunking, explicit vector DB
  calls, explicit prompt assembly) to learn the mechanics. Re-implement with
  LlamaIndex as a second pass later, specifically to compare what the framework
  automates.
- **Vector DB**: Chroma (local/embedded, no infra needed)
- **Embeddings**: start with local `sentence-transformers` (e.g. `all-MiniLM-L6-v2`)
  to avoid burning API calls while iterating; swap to Claude/OpenAI embeddings later
  to compare quality.
- **LLM for generation**: Claude via Anthropic API.

## Corpora (two, for comparison)
1. **LOTR epub** — I own this legally. I will process it entirely on my own machine.
   Source text itself should never be pasted into chat with you or any LLM session for
   "processing" — only code that operates on it locally.
   - Parse with `ebooklib` (epub structure) + `BeautifulSoup` (HTML -> text)
   - Inspect chapter/section boundaries before deciding chunking unit
2. **Wikipedia Middle-earth / Tolkien category pages** — pulled via the MediaWiki API
   (or `wikipedia-api` package), not scraped HTML.
   - Decide category scope, page count, and size cutoffs deliberately.

Before any chunking work, the goal is to produce for each corpus:
- Document count
- Size distribution (min/max/median word count per doc/chapter)
- A clear definition of "what is one document" (e.g., LOTR = per chapter? per book?
  Wikipedia = per page, with long-page splitting rules?)
- Available metadata fields (chapter/book title, Wikipedia page title/URL, category)
  — this feeds the `sources` field in API responses.

## API contract (design now, even before logic is built)
- `POST /ingest` — builds/updates the index for a given corpus
- `POST /query` — request: `{ query: str, corpus: "lotr" | "wiki" | "both", top_k: int }`
  response: `{ answer: str, sources: [{ text, source, chapter_or_page, score }] }`
- `GET /health`

Returning structured sources (not just a raw answer string) is intentional — it's
central to evaluating whether retrieval actually worked, and avoids retrofitting
citations into the UI later.

## Build order (please scaffold in this order, leaving clear TODOs for me to fill in
the core logic at each step)
1. Project skeleton: FastAPI app, folder structure, `requirements.txt`/`pyproject.toml`,
   `.env` handling for API keys (Anthropic key, etc.), basic `/health` endpoint.
2. Ingestion scripts: epub parsing/inspection script (LOTR), Wikipedia API fetch script
   — both should output structured intermediate data (e.g. JSON per doc/chapter) for me
   to inspect before any chunking happens.
3. Chunking module — STUB ONLY with clear function signatures and docstrings
   describing what I need to implement (e.g. fixed-size+overlap, then later
   paragraph/structure-aware, then semantic). Leave the actual logic as TODO for me.
4. Embedding + Chroma storage wiring — boilerplate client setup is fine to generate;
   leave decisions about metadata schema to me.
5. Retrieval — stub endpoint, boilerplate FastAPI wiring is fine; core similarity/
   filtering/hybrid-search logic should be left for me to implement with your guidance.
6. Generation — Claude API call wiring (boilerplate client setup ok), prompt template
   itself should be something we iterate on together, not auto-generated once and
   forgotten.
7. Eval harness scaffold — a place for me to write ~15-20 hand-written Q&A pairs per
   corpus (with expected source chapter/page) and a script that runs them against
   `/query` and reports basic hit/miss metrics. Scaffold the runner; I'll write the
   eval set.
8. Hybrid search (BM25 + vector) and reranking (cross-encoder) — add as a later
   iteration once basic retrieval works and I have eval numbers to compare against.

## Working style
- Prefer explaining tradeoffs over picking silently on my behalf for anything
  RAG-conceptual (chunking strategy, retrieval approach, reranking).
- Mechanical/boilerplate decisions (folder layout, dependency choices, FastAPI
  conventions) — fine to just make sensible defaults and move on.
- Keep corpus text out of chat/context where avoidable — operate on it via scripts I
  run locally, not by having full chapters pasted into a conversation.
