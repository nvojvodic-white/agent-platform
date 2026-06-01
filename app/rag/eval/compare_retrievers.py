"""Compare dense / sparse / hybrid retrieval on the in_corpus probe set.

Embeddings + BM25 only (no LLM generation), so this runs without Anthropic
credits. Prints retrieved titles per retriever plus Hit@1 / Hit@k against each
probe's expected_top_source_titles, so regressions on already-clean probes are
as visible as any Bombadil rescue.
"""
import json
from pathlib import Path

from dotenv import load_dotenv

from app.rag.retrieval.vectorstore import (
    get_dense_retriever,
    get_hybrid_retriever,
    get_sparse_retriever,
)

load_dotenv()

PROBES_PATH = Path(__file__).parent / "probe_queries.json"


def titles(docs) -> list[str]:
    return [d.metadata.get("title", "?") for d in docs]


def hit_at_1(expected: list[str], got: list[str]) -> bool:
    return bool(expected) and bool(got) and got[0] in expected


def hit_at_k(expected: list[str], got: list[str]) -> bool:
    return any(t in expected for t in got)


def main() -> None:
    probes = json.loads(PROBES_PATH.read_text())["in_corpus"]
    retrievers = {
        "dense": get_dense_retriever(k=4),
        "sparse": get_sparse_retriever(k=4),
        "hybrid_50_50": get_hybrid_retriever(k=4, dense_weight=0.5, sparse_weight=0.5),
        "hybrid_40_60": get_hybrid_retriever(k=4, dense_weight=0.4, sparse_weight=0.6),
    }

    for probe in probes:
        q = probe["query"]
        expected = probe.get("expected_top_source_titles", [])
        print(f"\n{'=' * 78}\nQ: {q}")
        print(f"Expected top titles: {expected}")
        for name, r in retrievers.items():
            got = titles(r.invoke(q))
            h1 = "H@1" if hit_at_1(expected, got) else "   "
            hk = "H@k" if hit_at_k(expected, got) else "   "
            print(f"  [{name:13}] {h1} {hk}  {got}")


if __name__ == "__main__":
    main()
