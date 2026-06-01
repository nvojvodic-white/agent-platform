import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.rag.agent.graph import get_agent
from app.rag.agent.graph_streaming import (
    get_streaming_agent,
    synthesize_streaming,
)
from app.rag.chain.rag_chain import build_chain
from app.rag.schemas import QueryRequest, QueryResponse, Source

router = APIRouter()

SNIPPET_CHARS = 200


def _sse(payload: dict) -> str:
    """Format a payload as a Server-Sent Events frame.

    The trailing double-newline is required by the SSE protocol; without it
    browsers / clients will not flush the event.
    """
    return f"data: {json.dumps(payload)}\n\n"


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


@router.post("/agent_query_stream")
async def agent_query_stream(req: QueryRequest) -> StreamingResponse:
    """Streaming variant of /agent_query.

    Two-phase: (1) run the streaming graph to completion (classify, retrieve,
    grade, optionally rewrite + retry once) and emit a single metadata frame
    with the route, grade, trace, and sources; (2) stream synthesis tokens as
    they arrive from Claude. Closes with a `done` frame.

    Frames are Server-Sent Events: `data: {json}\\n\\n`. Use with curl -N or a
    browser EventSource / fetch+ReadableStream client.

    NOTE: the non-streaming /agent_query is preserved byte-identical so RAGAS,
    the Day-11 agent probes, and any A/B caller continue to work.
    """
    streaming_agent = get_streaming_agent()

    async def event_stream():
        try:
            state = await streaming_agent.ainvoke({"question": req.question})
        except Exception as e:
            yield _sse({"type": "error", "message": f"agent failed: {e}"})
            yield _sse({"type": "done"})
            return

        docs = state.get("documents", [])
        yield _sse(
            {
                "type": "metadata",
                "route": state.get("route"),
                "grade": state.get("grade"),
                "attempt": state.get("attempt"),
                "trace": state.get("trace", []),
                "sources": [
                    {
                        "title": d.metadata.get("title", "Unknown"),
                        "url": d.metadata.get("url", ""),
                        "source": d.metadata.get("source", "unknown"),
                        "snippet": _snippet(d.page_content),
                    }
                    for d in docs
                ],
                "retrieved_chunks": len(docs),
            }
        )

        async for event in synthesize_streaming(state):
            yield _sse(event)

        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
