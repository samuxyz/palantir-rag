from fastapi import APIRouter, HTTPException
from app.models import QueryRequest, QueryResponse
from app.retrieval import retrieve
from app.generation import generate

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    sources = retrieve(
        query=request.query,
        corpus=request.corpus,
        top_k=request.top_k,
    )
    answer = generate(query=request.query, sources=sources)
    return QueryResponse(answer=answer, sources=sources)
