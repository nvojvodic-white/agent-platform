"""Hit the live RAG service with the probe query set, print everything for review.

Run after `uvicorn app.main:app --port 8000` is up and the Chroma index exists.
Reads probe_queries.json and runs both in_corpus and out_of_corpus slices.
"""
import json
import os
from pathlib import Path

import requests

PROBES_PATH = Path(__file__).parent / "probe_queries.json"
URL = os.getenv("RAG_URL", "http://localhost:8000/api/v1/rag/query")


def _hit(question: str) -> dict | None:
    try:
        r = requests.post(URL, json={"question": question, "k": 4}, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"REQUEST FAILED: {e}")
        return None


def run_in_corpus(probes: list[dict]) -> None:
    for probe in probes:
        print(f"\n{'=' * 70}")
        print(f"Q: {probe['query']}")
        print(f"Expected top titles: {probe['expected_top_source_titles']}")
        print(
            f"Retrieval verdict: {probe['verdict']} / {probe['failure_mode']}"
        )
        data = _hit(probe["query"])
        if data is None:
            continue
        print(f"\nAnswer:\n{data['answer']}\n")
        print(f"Retrieved {data['retrieved_chunks']} chunks:")
        for i, s in enumerate(data["sources"], 1):
            print(f"  [{i}] {s['source']} :: {s['title']}")
        print(
            f"\nSynthesis finding: "
            f"{probe.get('synthesis_finding', '(none recorded)')}"
        )


def run_out_of_corpus(probes: list[dict]) -> None:
    print(f"\n\n{'#' * 70}\n# OUT-OF-CORPUS PROBES (refusal behaviour)\n{'#' * 70}")
    for probe in probes:
        print(f"\n{'=' * 70}")
        print(f"Q: {probe['query']}")
        print(f"Expected: {probe['expected_behavior']}")
        data = _hit(probe["query"])
        if data is None:
            continue
        print(f"\nAnswer:\n{data['answer']}\n")
        print(f"Retrieved {data['retrieved_chunks']} chunks:")
        for i, s in enumerate(data["sources"], 1):
            print(f"  [{i}] {s['source']} :: {s['title']}")
        print(f"\nNotes: {probe['notes']}")


def main() -> None:
    probes = json.loads(PROBES_PATH.read_text())
    run_in_corpus(probes["in_corpus"])
    run_out_of_corpus(probes["out_of_corpus"])


if __name__ == "__main__":
    main()
