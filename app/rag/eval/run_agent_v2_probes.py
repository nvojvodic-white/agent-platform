"""Sweep all 7 in-corpus probes against the LangGraph routing agent.

Measures classifier accuracy (vs the Day 11 expected routes from the plan) and
captures grade + retry behavior per probe. The expected routes are a HYPOTHESIS
from the Day 8-9 evidence, not ground truth - the plan flags this explicitly.
A classifier that disagrees with the expectation but still produces a correct
answer is itself a finding (suggests the routing distinction matters less than
predicted on that probe).

Override URL: AGENT_DEBUG_URL=http://localhost:8140/... python -m ...
"""
import json
import os
from pathlib import Path

import requests

URL = os.getenv(
    "AGENT_DEBUG_URL", "http://localhost:8000/api/v1/rag/agent_query_debug"
)
PROBES_PATH = Path(__file__).parent / "probe_queries.json"

# Day 11 plan's hypothesis of correct routes, from Day 8-9 evidence.
EXPECTED_ROUTES = {
    "Who is Gandalf?": "definitional",
    "Who killed Smaug?": "multi_hop",
    "What rings did the Dwarves get?": "multi_hop",
    "Tell me about the Battle of Five Armies": "general",
    "Beren and Luthien": "general",
    "What is mithril?": "definitional",
    "Tom Bombadil": "definitional",
}


def main() -> None:
    probes = json.loads(PROBES_PATH.read_text())["in_corpus"]
    n_match = 0
    grade_counts: dict[str, int] = {}
    retried: list[str] = []
    rows: list[dict] = []

    for probe in probes:
        q = probe["query"]
        r = requests.post(URL, json={"question": q}, timeout=180)
        r.raise_for_status()
        d = r.json()
        route = d.get("route")
        grade = d.get("grade")
        attempt = d.get("attempt") or 0
        expected = EXPECTED_ROUTES.get(q, "?")
        match = route == expected
        if match:
            n_match += 1
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        if attempt > 0:
            retried.append(q)

        marker = "OK" if match else "!!"
        print("=" * 70)
        print(f"[{marker}] {q!r}")
        print(
            f"  expected={expected}  got={route}  grade={grade}  "
            f"attempts={attempt + 1}"
        )
        print(f"  source_titles={d.get('source_titles')}")
        for line in d.get("trace", []):
            print(f"    - {line}")
        rows.append(
            {
                "query": q,
                "expected": expected,
                "route": route,
                "grade": grade,
                "attempts": attempt + 1,
                "match": match,
                "source_titles": d.get("source_titles"),
            }
        )

    print()
    print("=" * 70)
    print(
        f"SUMMARY: classifier match {n_match}/{len(probes)} "
        f"({100 * n_match / len(probes):.0f}%)"
    )
    print(f"grades: {grade_counts}")
    print(f"retried: {retried or '(none)'}")
    print()
    print("Per-probe table (markdown):")
    print()
    print("| probe | expected | got | match | grade | attempts |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        marker = "✓" if row["match"] else "✗"
        print(
            f"| {row['query']} | {row['expected']} | {row['route']} | "
            f"{marker} | {row['grade']} | {row['attempts']} |"
        )


if __name__ == "__main__":
    main()
