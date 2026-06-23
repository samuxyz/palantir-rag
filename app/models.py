from typing import Literal
from pydantic import BaseModel


class IngestRequest(BaseModel):
    corpus: Literal["lotr", "wiki"]


class IngestResponse(BaseModel):
    corpus: str
    chunks_indexed: int
    message: str


class Source(BaseModel):
    text: str
    source: str
    chapter_or_page: str
    score: float


class QueryRequest(BaseModel):
    query: str
    corpus: Literal["lotr", "wiki", "both"]
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
