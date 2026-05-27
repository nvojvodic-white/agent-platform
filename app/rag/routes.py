from fastapi import APIRouter

from app.rag.chain.rag_chain import build_chain
from app.rag.schemas import QueryRequest, QueryResponse, Source

router = APIRouter()

SNIPPET_CHARS = 200


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    chain = build_chain(k=req.k)
    result = chain.invoke(req.question)

    sources = [
        Source(
            title=doc.metadata.get("title", "Unknown"),
            url=doc.metadata.get("url", ""),
            source=doc.metadata.get("source", "unknown"),
            snippet=(
                doc.page_content[:SNIPPET_CHARS]
                + ("..." if len(doc.page_content) > SNIPPET_CHARS else "")
            ),
        )
        for doc in result["docs"]
    ]
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        retrieved_chunks=len(result["docs"]),
    )
