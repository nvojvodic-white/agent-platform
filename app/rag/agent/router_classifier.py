"""Meta-classifier: decide whether a user's question should go to the RAG
service (Middle-earth lore) or to the general Claude agent (everything else,
including web search via Tavily, code execution, file I/O).

This sits ABOVE the RAG agent's own internal classify_query node, which only
picks between RAG retrieval routes (definitional / multi_hop / general). This
meta-classifier picks BACKENDS.

Returns reasoning along with the route so the UI can show why a route was
picked and the user can override silently bad routes.
"""
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

MetaRoute = Literal["rag", "agent"]


class MetaClassification(BaseModel):
    route: MetaRoute = Field(
        description="'rag' for Middle-earth / Tolkien lore questions; 'agent' "
        "for anything else (current events, math, code, file lookup, web search)."
    )
    reasoning: str = Field(
        description="One short sentence explaining the choice. The user sees this."
    )


_meta_llm = ChatAnthropic(
    model="claude-sonnet-4-5", max_tokens=200, max_retries=5
)


_META_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Route the user's question to one of two backends:\n\n"
            "- `rag`: a curated Middle-earth lore RAG (Fandom LotR wiki + "
            "Wikipedia Tolkien articles). Pick this whenever the question is "
            "about Tolkien's legendarium: characters (Gandalf, Smaug, "
            "Bombadil), places (the Shire, Mordor), events (the Battle of "
            "Five Armies), artifacts (the One Ring, mithril), races (Hobbits, "
            "Elves), languages (Sindarin), or any aspect of The Hobbit, "
            "The Lord of the Rings, or The Silmarillion.\n"
            "- `agent`: a general Claude agent with tools (web search, code "
            "execution, file read). Pick this for everything else: current "
            "events, math, code, weather, recent news, anything that isn't "
            "Middle-earth lore.\n\n"
            "Borderline cases: if a question mentions Middle-earth content but "
            "asks something the RAG corpus would not know (the actor who "
            "played Gandalf, real-world Tolkien biography post-1973, comparisons "
            "to Game of Thrones), route to `agent`. The RAG corpus is curated "
            "lore, not general Tolkien meta-knowledge.\n\n"
            "Reply with JSON: {{\"route\": \"rag\"|\"agent\", \"reasoning\": "
            "\"<one sentence>\"}}",
        ),
        ("human", "{question}"),
    ]
)


async def aclassify_route(question: str) -> MetaClassification:
    """Async classify. Falls back to 'agent' on parse failure (the agent path
    can always handle anything, so defaulting there is the safer side)."""
    parser = PydanticOutputParser(pydantic_object=MetaClassification)
    chain = _META_PROMPT | _meta_llm | parser
    try:
        return await chain.ainvoke({"question": question})
    except Exception as e:
        return MetaClassification(
            route="agent",
            reasoning=f"meta-classifier failed ({type(e).__name__}); "
            "defaulting to agent (handles arbitrary inputs)",
        )
