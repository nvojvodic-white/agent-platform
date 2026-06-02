from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    k: int = Field(default=4, ge=1, le=20)


class StreamQueryRequest(BaseModel):
    """Request shape for /agent_query_stream_v2. session_id is optional; when
    absent the endpoint behaves like /agent_query_stream (no memory)."""

    question: str = Field(..., min_length=1, max_length=500)
    session_id: str | None = Field(default=None, max_length=128)
    k: int = Field(default=4, ge=1, le=20)


class HistoryTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=10000)


class RouteRequest(BaseModel):
    """Request shape for /route_question. Optional `history` carries the last
    few turns so the meta-classifier can disambiguate pronoun-y follow-ups."""

    question: str = Field(..., min_length=1, max_length=500)
    history: list[HistoryTurn] | None = Field(default=None, max_length=20)


class Source(BaseModel):
    title: str
    url: str
    source: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieved_chunks: int
    # Set true if the response was served from the semantic cache rather than
    # a fresh pipeline run. Always false unless SEMANTIC_CACHE_ENABLED=1.
    from_cache: bool = False
    # Similarity to the closest cached question (0.0 if cache disabled or empty).
    # Only meaningful when from_cache=true; otherwise it is the best near-miss.
    cache_similarity: float | None = None
