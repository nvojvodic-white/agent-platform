"""Prompt-variant experiments for Day 5.

Three experiments, selected by argv[1]:
  faithfulness  (Exp 1) baseline vs strict on adversarial probes. Centerpiece is
                the restricted-context probes: the answer IS in the corpus but we
                deliberately retrieve an UNRELATED query's chunks, so the answer
                is provably absent and a correct answer can only come from the
                model's priors. Real corpus gaps are run as corroboration.
  frontier      (Exp 2) permissive/baseline/strict/strict_hedge on the in_corpus
                probe mix plus the real gaps. The faithfulness/helpfulness frontier.
  routing       (Exp 3) bare model (NO retrieval) on the 'wisest character'
                question with neutral/diversity/anti-tolkien prompts: can a prompt
                rebalance content priors, or only suppress them?

Retrieval is held fixed (same k, deterministic) so the system prompt is the only
variable. Prints grouped by probe so variants are compared on the same question.
"""
import sys
from functools import lru_cache

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.rag.eval import probe_loader
from app.rag.retrieval.vectorstore import get_retriever

load_dotenv()

K = 4

BASELINE = """You are a Middle-earth lore expert answering questions using only the provided context.

Rules:
1. Answer ONLY from the context below. If the context does not contain enough information to answer, say so explicitly. Do not guess and do not use outside knowledge about Tolkien even if you have it.
2. If you do use outside knowledge (you should not), prefix that sentence with "Outside context:" so the reader can tell.
3. Cite sources inline using bracketed numbers like [1], [2] that correspond to the numbered context entries.
4. Be concise. One to three short paragraphs maximum.
5. If sources conflict, note the conflict rather than picking arbitrarily.
6. Use names and spellings exactly as they appear in the context (e.g., "Lúthien", not "Luthien"; "Eärendil", not "Earendil").
"""

STRICT = BASELINE + """
7. CRITICAL: If a fact seems obviously true but does not appear in the context above, you must still treat it as unknown. State that the context does not contain it. Do not supplement context with prior knowledge under any circumstances, even for facts you are certain about.
"""

PERMISSIVE = """You are a Middle-earth lore expert. Answer the question using the provided context.

Rules:
1. Prefer the context, but you may supplement with general knowledge if it helps the user. Clearly mark any supplemented fact with the prefix "Beyond the corpus:".
2. Cite context sources inline using bracketed numbers like [1], [2].
3. Be concise. One to three short paragraphs maximum.
4. Use names and spellings exactly as they appear in the context.
"""

STRICT_HEDGE = BASELINE + """
7. CRITICAL: If a fact seems obviously true but does not appear in the context above, treat it as unknown and do not supply it from prior knowledge.
8. If the context only partially answers the question, answer what you CAN from the context and then explicitly list what the context does not cover, rather than refusing entirely. A partial grounded answer plus a clear note of the gaps is better than a flat refusal.
"""

# Exp 3 bare-model variants (no retrieval / no context)
ROUTING_NEUTRAL = "You are a helpful assistant."
ROUTING_DIVERSITY = (
    "You are a helpful assistant. When listing examples, draw deliberately from "
    "diverse literary traditions and authors. Do not over-represent any single "
    "fictional universe."
)
ROUTING_ANTI_TOLKIEN = (
    "You are a helpful assistant. Do not mention Tolkien or Middle-earth "
    "characters unless specifically asked."
)


@lru_cache(maxsize=1)
def _llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-5", max_tokens=1024)


def format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] {d.metadata.get('title', '?')} ({d.metadata.get('source', '?')})\n"
        f"{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def answer_with_context(system_prompt: str, context: str, question: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ]
    )
    chain = prompt | _llm() | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def answer_bare(system_prompt: str, question: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{question}")]
    )
    chain = prompt | _llm() | StrOutputParser()
    return chain.invoke({"question": question})


def exp_faithfulness() -> None:
    retriever = get_retriever(k=K)
    variants = [("baseline", BASELINE), ("strict", STRICT)]

    print("\n" + "#" * 78)
    print("# EXP 1: FAITHFULNESS UNDER PRESSURE")
    print("# Centerpiece: restricted-context (forced off-topic retrieval)")
    print("#" * 78)

    print("\n\n=== RESTRICTED-CONTEXT PROBES (answer absent by construction) ===")
    for p in probe_loader.load("adversarial_restricted"):
        q = p["query"]
        forced = p["force_context_query"]
        docs = retriever.invoke(forced)
        context = format_docs(docs)
        print(f"\n{'=' * 70}\nQ: {q}")
        print(f"prior_answer (must NOT appear): {p['prior_answer']}")
        print(f"forced off-topic context from: {forced!r}")
        print(f"context titles: {[d.metadata.get('title', '?') for d in docs]}")
        for name, sp in variants:
            print(f"\n--- [{name}] ---")
            print(answer_with_context(sp, context, q))

    print("\n\n=== REAL CORPUS GAPS (answer genuinely absent from on-topic context) ===")
    for p in probe_loader.load("adversarial_real"):
        q = p["query"]
        docs = retriever.invoke(q)
        context = format_docs(docs)
        print(f"\n{'=' * 70}\nQ: {q}")
        print(f"prior_answer (must NOT appear): {p['prior_answer']}")
        print(f"context titles: {[d.metadata.get('title', '?') for d in docs]}")
        for name, sp in variants:
            print(f"\n--- [{name}] ---")
            print(answer_with_context(sp, context, q))


def exp_frontier() -> None:
    retriever = get_retriever(k=K)
    variants = [
        ("permissive", PERMISSIVE),
        ("baseline", BASELINE),
        ("strict", STRICT),
        ("strict_hedge", STRICT_HEDGE),
    ]
    print("\n" + "#" * 78)
    print("# EXP 2: FAITHFULNESS / HELPFULNESS FRONTIER")
    print("#" * 78)
    for p in probe_loader.load("all"):
        q = p["query"]
        docs = retriever.invoke(q)
        context = format_docs(docs)
        print(f"\n{'=' * 70}\nQ: {q}")
        print(f"context titles: {[d.metadata.get('title', '?') for d in docs]}")
        for name, sp in variants:
            print(f"\n--- [{name}] ---")
            print(answer_with_context(sp, context, q))


def exp_routing() -> None:
    question = "Who is the wisest character in fantasy literature?"
    variants = [
        ("neutral", ROUTING_NEUTRAL),
        ("diversity", ROUTING_DIVERSITY),
        ("anti_tolkien", ROUTING_ANTI_TOLKIEN),
    ]
    print("\n" + "#" * 78)
    print("# EXP 3: ROUTING-VS-RANKING (bare model, no retrieval)")
    print(f"# Q: {question}")
    print("#" * 78)
    for name, sp in variants:
        print(f"\n--- [{name}] ---")
        print(answer_bare(sp, question))


EXPERIMENTS = {
    "faithfulness": exp_faithfulness,
    "frontier": exp_frontier,
    "routing": exp_routing,
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        exp_faithfulness()
        exp_frontier()
        exp_routing()
    else:
        EXPERIMENTS[which]()


if __name__ == "__main__":
    main()
