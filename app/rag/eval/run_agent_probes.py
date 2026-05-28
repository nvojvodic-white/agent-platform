"""Agent-level probe sweep: hit the agent endpoint, observe tool routing.

Distinct from run_probes.py (which tests the RAG service directly). This tests
the agent's decision of WHICH tool to call and whether the final answer keeps
citations. Each probe declares the tool we expect the agent to route to (or
none for general-knowledge questions).

Creates a session (POST /api/v1/sessions), polls GET until terminal status,
then inspects session.tool_calls (the runner records {tool, input, result} per
call) and session.result (the final answer text).

Run after the service is up (uvicorn app.main:app) with the Chroma index built.
"""
import os
import time

import requests

BASE = os.getenv("AGENT_BASE_URL", "http://localhost:8000/api/v1")
POLL_INTERVAL = 1.0
POLL_TIMEOUT = 120.0

PROBES = [
    {
        "task": "Who killed Smaug?",
        "expected_tool": "search_middle_earth",
        "category": "in-corpus, should call RAG",
    },
    {
        "task": "Tell me about Tom Bombadil.",
        "expected_tool": "search_middle_earth",
        "category": "in-corpus (candidate-set-miss), should still call RAG",
    },
    {
        "task": "Who is Sauron's accountant?",
        "expected_tool": "search_middle_earth",
        "category": "out-of-corpus Middle-earth, should call RAG + pass through refusal",
    },
    {
        "task": "What's the capital of Mongolia?",
        "expected_tool": None,
        "category": "not Middle-earth, should NOT call search_middle_earth",
    },
    {
        "task": "Search the web for today's Bitcoin price.",
        "expected_tool": "web_search",
        "category": "edge: existing tool still routes correctly",
    },
    {
        "task": "Who is the wisest character in fantasy literature?",
        "expected_tool": "ambiguous",
        "category": "edge: ambiguous, either choice defensible",
    },
]


def run_probe(probe: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"Task: {probe['task']}")
    print(f"Category: {probe['category']}")
    print(f"Expected tool: {probe['expected_tool']}")

    resp = requests.post(f"{BASE}/sessions", json={"task": probe["task"]}, timeout=30)
    resp.raise_for_status()
    session_id = resp.json()["session_id"]

    deadline = time.time() + POLL_TIMEOUT
    session = None
    while time.time() < deadline:
        s = requests.get(f"{BASE}/sessions/{session_id}", timeout=30).json()
        if s.get("status") in {"completed", "failed"}:
            session = s
            break
        time.sleep(POLL_INTERVAL)

    if session is None:
        print("TIMED OUT waiting for terminal status")
        return

    tools_called = [tc["tool"] for tc in session.get("tool_calls", [])]
    print(f"Status: {session['status']}")
    print(f"Tools called: {tools_called or '(none)'}")

    expected = probe["expected_tool"]
    if expected == "ambiguous":
        verdict = "OBSERVE"
    elif expected is None:
        verdict = "PASS" if "search_middle_earth" not in tools_called else "FAIL (called RAG)"
    else:
        verdict = "PASS" if expected in tools_called else f"FAIL (expected {expected})"
    print(f"Routing verdict: {verdict}")

    result = session.get("result") or ""
    if "search_middle_earth" in tools_called:
        has_citation = "[1]" in result or "[2]" in result
        print(f"Final answer keeps [n] citations: {has_citation}")
    print(f"Final answer:\n{result[:500]}")


def main() -> None:
    for probe in PROBES:
        run_probe(probe)


if __name__ == "__main__":
    main()
