"""Multi-turn smoke probes: a fixed conversation sequence against
/agent_query_stream_v2 that exercises coref + history-aware synthesis.

For each turn:
  - posts the question with the shared session_id
  - parses SSE frames, prints route / grade / resolved_question / source titles
  - shows the first ~180 chars of the assembled answer

Reads sequences from app/rag/eval/probes_multiturn.json (default) or any file
given via --probes. Override the endpoint with AGENT_STREAM_URL_V2.
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path

import requests

URL = os.getenv(
    "AGENT_STREAM_URL_V2",
    "http://localhost:8000/api/v1/rag/agent_query_stream_v2",
)
DEFAULT_PROBES = Path(__file__).parent / "probes_multiturn.json"


def stream(session_id: str, question: str) -> dict:
    """Post and accumulate frames. Returns {metadata, answer, frames}."""
    metadata: dict | None = None
    parts: list[str] = []
    frames_seen = 0
    with requests.post(
        URL,
        json={"session_id": session_id, "question": question},
        stream=True,
        timeout=180,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            frames_seen += 1
            t = payload.get("type")
            if t == "metadata":
                metadata = payload
            elif t == "token":
                parts.append(payload.get("content", ""))
            elif t == "error":
                print(f"  ERROR FRAME: {payload.get('message')}")
            # ignore answer_complete + done; we accumulate from tokens
    return {
        "metadata": metadata or {},
        "answer": "".join(parts),
        "frames_seen": frames_seen,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    ap.add_argument(
        "--session-id",
        default=f"smoke-{uuid.uuid4().hex[:8]}",
        help="Reuse a session id to test loaded history; default = fresh per run",
    )
    args = ap.parse_args()

    if not args.probes.exists():
        raise SystemExit(f"probes file not found: {args.probes}")

    data = json.loads(args.probes.read_text())
    sequences = data.get("sequences", [data])  # accept a single sequence too

    for seq_idx, seq in enumerate(sequences):
        sid = (
            args.session_id
            if len(sequences) == 1
            else f"{args.session_id}-seq{seq_idx}"
        )
        print(f"\n{'#' * 70}")
        print(f"# sequence {seq_idx}: {seq.get('name', '(unnamed)')} (session_id={sid})")
        print(f"{'#' * 70}")
        for turn_idx, turn in enumerate(seq["turns"]):
            q = turn["question"]
            expected = turn.get("expected_resolved_question")
            print(f"\n--- turn {turn_idx}: {q!r} ---")
            if expected:
                print(f"    expected resolved: {expected!r}")
            t0 = time.time()
            out = stream(sid, q)
            dt = time.time() - t0
            m = out["metadata"]
            resolved = m.get("resolved_question")
            print(f"    resolved        : {resolved!r}")
            print(f"    route / grade   : {m.get('route')} / {m.get('grade')}")
            print(f"    history loaded  : {m.get('history_turns_loaded')}")
            print(
                f"    source titles   : "
                f"{[s.get('title') for s in m.get('sources', [])]}"
            )
            ans = out["answer"].replace("\n", " ")[:180]
            print(f"    answer[:180]    : {ans}")
            print(
                f"    frames={out['frames_seen']} elapsed={dt:.1f}s"
            )


if __name__ == "__main__":
    main()
