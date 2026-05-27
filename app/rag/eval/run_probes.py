"""Hit the live RAG service with the probe query set, print everything for review.

Run after `uvicorn app.main:app --port 8000` is up and the Chroma index exists.
"""
import json
import os
from pathlib import Path

import requests

PROBES_PATH = Path(__file__).parent / "probe_queries.json"
URL = os.getenv("RAG_URL", "http://localhost:8000/api/v1/rag/query")


def main() -> None:
    probes = json.loads(PROBES_PATH.read_text())["queries"]
    for probe in probes:
        print(f"\n{'=' * 70}")
        print(f"Q: {probe['query']}")
        print(f"Expected top titles: {probe['expected_top_source_titles']}")
        print(f"Day 2 verdict: {probe['verdict']}"
              f" / failure_mode: {probe['failure_mode']}")

        try:
            r = requests.post(
                URL,
                json={"question": probe["query"], "k": 4},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"REQUEST FAILED: {e}")
            continue

        print(f"\nAnswer:\n{data['answer']}\n")
        print(f"Retrieved {data['retrieved_chunks']} chunks:")
        for i, s in enumerate(data["sources"], 1):
            print(f"  [{i}] {s['source']} :: {s['title']}")
        print(f"\nNotes: {probe['notes']}")


if __name__ == "__main__":
    main()
