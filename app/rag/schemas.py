from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    k: int = Field(default=4, ge=1, le=20)


class Source(BaseModel):
    title: str
    url: str
    source: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieved_chunks: int
