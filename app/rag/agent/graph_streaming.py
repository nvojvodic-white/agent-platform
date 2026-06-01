"""Streaming-path variant of the routing agent.

Day 13 streaming addendum to graph.py. Per the plan:
  - Keep the non-streaming graph (graph.build_agent) untouched so RAGAS, the
    Day-11 agent_v2 probes, the /agent_query route, and /agent_query_debug all
    keep working byte-identical.
  - Build a parallel async graph that runs classify -> retrieve -> grade ->
    [rewrite -> retrieve] and exits when retrieval is finalised. Synthesis is
    NOT a graph node here; the endpoint pipes the resulting state into
    synthesize_streaming() and yields tokens as they arrive.

Why split rather than make synthesize a graph node: LangGraph nodes return
state, not async iterators. Driving SSE through astream_events would work but
adds protocol surface for no UX gain over just running the streaming synth
after the graph finishes.

All node functions are async + use ainvoke / asimilarity_search so the event
loop is never blocked. The LRU retrieval cache from graph.py is intentionally
NOT used here: it's keyed on sync retriever calls; correctness over throughput
on the streaming path. (Streaming is for UX, not for cheaper repeats.)

Public surface:
  build_streaming_agent() -> compiled graph, callable via .ainvoke(state)
  synthesize_streaming(state) -> async generator of {"type": ..., "content": ...}
                                 frames (token | answer_complete | error).
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langgraph.graph import END, START, StateGraph

from app.rag.agent.graph import (
    AgentState,
    Classification,
    GradeOutput,
    ROUTE_TO_RETRIEVER,
    classifier_llm,
    classify_prompt,
    decide_next,
    grade_llm,
    grade_prompt,
    rag_prompt,
    rewrite_llm,
    rewrite_prompt,
    synth_llm,
)
from app.rag.retrieval.vectorstore import get_retriever


# --- async nodes ------------------------------------------------------------


async def aclassify_query(state: AgentState) -> dict:
    parser = PydanticOutputParser(pydantic_object=Classification)
    chain = classify_prompt | classifier_llm | parser
    try:
        result = await chain.ainvoke({"question": state["question"]})
        return {
            "route": result.route,
            "attempt": 0,
            "trace": [f"classified as {result.route}: {result.reasoning}"],
        }
    except Exception as e:
        return {
            "route": "general",
            "attempt": 0,
            "trace": [f"classification failed ({e}); defaulting to general"],
        }


async def aretrieve(state: AgentState) -> dict:
    route = state["route"]
    question = state.get("rewritten_question") or state["question"]
    retriever_kind = ROUTE_TO_RETRIEVER[route]
    retriever = get_retriever(k=4, kind=retriever_kind)
    docs = await retriever.ainvoke(question)
    return {
        "documents": docs,
        "trace": [
            f"retrieved {len(docs)} docs via {retriever_kind} "
            f"(attempt {state.get('attempt', 0) + 1})"
        ],
    }


async def agrade_documents(state: AgentState) -> dict:
    parser = PydanticOutputParser(pydantic_object=GradeOutput)
    chain = grade_prompt | grade_llm | parser
    passages = "\n\n---\n\n".join(
        f"[{i}] {d.metadata.get('title', '?')}\n{d.page_content[:500]}"
        for i, d in enumerate(state["documents"], 1)
    )
    try:
        result = await chain.ainvoke(
            {"question": state["question"], "passages": passages}
        )
        return {
            "grade": result.grade,
            "trace": [f"graded {result.grade}: {result.reasoning}"],
        }
    except Exception as e:
        return {
            "grade": "partial",
            "trace": [f"grading failed ({e}); defaulted to partial"],
        }


async def arewrite_query(state: AgentState) -> dict:
    chain = rewrite_prompt | rewrite_llm | StrOutputParser()
    rewritten = (await chain.ainvoke({"question": state["question"]})).strip()
    return {
        "rewritten_question": rewritten,
        "attempt": state.get("attempt", 0) + 1,
        "trace": [f"rewrote to: {rewritten}"],
    }


# --- streaming synthesis (NOT a graph node) ---------------------------------


async def synthesize_streaming(state: AgentState) -> AsyncIterator[dict]:
    """Async generator: yields a stream of {type, content} frames.

    Frame types:
      - {"type": "token", "content": "<chunk>"}: each LLM token chunk.
      - {"type": "answer_complete", "content": "<full answer>"}: emitted once
        at the end so downstream (logs, caches, eval) have the full string.
      - {"type": "error", "message": "<...>"}: emitted if the LLM call fails
        partway through (the HTTP 200 was already sent, so error must travel
        in-band).
    """
    context = "\n\n---\n\n".join(
        f"[{i}] {d.metadata.get('title', '?')}\n{d.page_content}"
        for i, d in enumerate(state.get("documents", []), 1)
    )
    chain = rag_prompt | synth_llm | StrOutputParser()
    parts: list[str] = []
    try:
        async for chunk in chain.astream(
            {"context": context, "question": state["question"]}
        ):
            parts.append(chunk)
            yield {"type": "token", "content": chunk}
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return
    yield {"type": "answer_complete", "content": "".join(parts)}


# --- graph ------------------------------------------------------------------


def build_streaming_agent():
    """Async graph that ends at synthesize (i.e. graph exits with retrieval
    finalised; the endpoint streams synthesis after).

    Same nodes / same conditional edge as the non-streaming agent, except:
      - all nodes are async (so .ainvoke never blocks the event loop)
      - decide_next's "synthesize" verdict routes to END (the endpoint will
        stream synthesis OUTSIDE the graph)
      - retrieve uses retriever.ainvoke directly (no LRU cache on this path)
    """
    g = StateGraph(AgentState)
    g.add_node("classify_query", aclassify_query)
    g.add_node("retrieve", aretrieve)
    g.add_node("grade_documents", agrade_documents)
    g.add_node("rewrite_query", arewrite_query)
    # NOTE: no synthesize node. Synthesis runs as a streaming generator OUTSIDE
    # the graph; the endpoint pipes the final state through synthesize_streaming.

    g.add_edge(START, "classify_query")
    g.add_edge("classify_query", "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        decide_next,
        # decide_next returns "synthesize" or "rewrite_query"; route "synthesize"
        # to END so the endpoint can stream synthesis itself.
        {"synthesize": END, "rewrite_query": "rewrite_query"},
    )
    g.add_edge("rewrite_query", "retrieve")
    return g.compile()


_streaming_agent = None


def get_streaming_agent():
    global _streaming_agent
    if _streaming_agent is None:
        _streaming_agent = build_streaming_agent()
    return _streaming_agent
