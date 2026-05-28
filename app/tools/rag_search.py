"""RAG search tool — calls the RAG query endpoint over HTTP.

Synchronous to match the agent runner (app/agent/runner.py), which uses the
sync Anthropic client, and to match the other tools in app/tools/executor.py
which return plain strings. execute_tool feeds this return value straight into
a tool_result block and calls len() on it, so it must return a str. Failures
are returned as text, never raised, so the agent loop can recover instead of
crashing.
"""
import os

import httpx

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
TIMEOUT = 30.0
MAX_K = 10


def search_middle_earth(question: str, k: int = 4) -> str:
    """Search the Middle-earth lore corpus and return a grounded answer."""
    k = max(1, min(int(k), MAX_K))
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(
                f"{RAG_SERVICE_URL}/api/v1/rag/query",
                json={"question": question, "k": k},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return (
            f"Lore search unavailable: the RAG service returned status "
            f"{e.response.status_code}. The user should be told the search "
            f"could not be completed."
        )
    except httpx.RequestError as e:
        return (
            f"Lore search unavailable: the RAG service could not be reached "
            f"({type(e).__name__}). The user should be told the search "
            f"could not be completed."
        )

    sources = data.get("sources", [])
    sources_text = "\n".join(
        f"[{i}] {s.get('title', 'Unknown')} ({s.get('source', '?')}) — "
        f"{s.get('url', '')}"
        for i, s in enumerate(sources, 1)
    )
    return (
        f"Answer from the Middle-earth corpus:\n{data.get('answer', '')}\n\n"
        f"Sources:\n{sources_text}\n\n"
        f"(Retrieved {data.get('retrieved_chunks', 0)} chunks.)"
    )
