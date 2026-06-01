"""Smoke-test the LangGraph routing agent via /agent_query_debug.

Hits the three plan probes (mithril, Smaug, Battle of Five Armies) and prints
each one's route, grade, attempt count, retrieved-doc titles, full trace, and
the first 180 chars of the answer. The trace is the diagnostic - it shows the
classifier's reasoning so misroutes are obvious.

Override the URL with AGENT_DEBUG_URL if the service is on a non-default port:
  AGENT_DEBUG_URL=http://localhost:8140/api/v1/rag/agent_query_debug \\
    python -m app.rag.eval.run_agent_debug
"""
import os

import requests

URL = os.getenv(
    "AGENT_DEBUG_URL", "http://localhost:8000/api/v1/rag/agent_query_debug"
)

PROBES = [
    "What is mithril?",
    "Who killed Smaug?",
    "Tell me about the Battle of Five Armies",
]


def main() -> None:
    for q in PROBES:
        print("=" * 70)
        print(f"Q: {q}")
        r = requests.post(URL, json={"question": q}, timeout=180)
        r.raise_for_status()
        d = r.json()
        print(f"route   : {d.get('route')}")
        print(f"grade   : {d.get('grade')}")
        print(f"attempt : {d.get('attempt')}")
        print(f"titles  : {d.get('source_titles')}")
        print("trace:")
        for line in d.get("trace", []):
            print(f"  - {line}")
        ans = (d.get("answer") or "").replace("\n", " ")[:180]
        print(f"answer[:180]: {ans}")


if __name__ == "__main__":
    main()
