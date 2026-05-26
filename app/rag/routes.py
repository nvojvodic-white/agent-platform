from fastapi import APIRouter
from pydantic import BaseModel

from app.rag.llm import chain

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    answer = chain.invoke({"question": req.question})
    return QueryResponse(answer=answer, sources=[])
