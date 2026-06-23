from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import ingest, query

app = FastAPI(
    title=settings.app_name,
    description="RAG service for LOTR and Wikipedia Middle-earth corpora",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "app": settings.app_name}
