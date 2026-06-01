from fastapi import APIRouter

from app.rag.agent.graph import get_agent
from app.rag.chain.rag_chain import build_chain
from app.rag.schemas import QueryRequest, QueryResponse, Source

router = APIRouter()

SNIPPET_CHARS = 200


def _snippet(text: str) -> str:
    return text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else "")


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


@router.post("/agent_query", response_model=QueryResponse)
def agent_query(req: QueryRequest) -> QueryResponse:
    """Routing RAG agent (LangGraph): classify -> retrieve -> grade -> [rewrite]
    -> synthesize. Same response shape as /query so callers can A/B."""
    agent = get_agent()
    result = agent.invoke({"question": req.question})
    docs = result.get("documents", [])
    sources = [
        Source(
            title=d.metadata.get("title", "Unknown"),
            url=d.metadata.get("url", ""),
            source=d.metadata.get("source", "unknown"),
            snippet=_snippet(d.page_content),
        )
        for d in docs
    ]
    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        retrieved_chunks=len(docs),
    )


@router.post("/agent_query_debug")
def agent_query_debug(req: QueryRequest) -> dict:
    """Dev endpoint: agent_query + full agent state (route, grade, trace).
    Same generation; surfaces the classifier and grader decisions for debugging."""
    agent = get_agent()
    result = agent.invoke({"question": req.question})
    return {
        "question": req.question,
        "route": result.get("route"),
        "grade": result.get("grade"),
        "attempt": result.get("attempt"),
        "trace": result.get("trace", []),
        "answer": result.get("answer"),
        "source_titles": [
            d.metadata.get("title") for d in result.get("documents", [])
        ],
    }
