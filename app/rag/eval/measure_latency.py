"""Measure per-node latency by hitting /agent_query_debug and parsing the
[timing] lines the @timed decorator appends to the trace.

Two passes: first cold, second warm (to demonstrate the retrieval cache).
Prints a per-probe table and a per-node-per-route aggregate table.

Override AGENT_DEBUG_URL to point at a non-default port:
  AGENT_DEBUG_URL=http://localhost:8142/api/v1/rag/agent_query_debug \\
    python -m app.rag.eval.measure_latency
"""
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

import requests

URL = os.getenv(
    "AGENT_DEBUG_URL", "http://localhost:8000/api/v1/rag/agent_query_debug"
)
PROBES_PATH = Path(__file__).parent / "probe_queries.json"
TIMING_RE = re.compile(r"^\[timing\] (\w+): (\d+)ms$")


def hit(q: str) -> dict:
    r = requests.post(URL, json={"question": q}, timeout=180)
    r.raise_for_status()
    return r.json()


def parse_timings(trace: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in trace:
        m = TIMING_RE.match(line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def run_pass(label: str, probes: list[dict]) -> list[dict]:
    rows: list[dict] = []
    print(f"\n=== pass: {label} ===")
    print(f"{'probe':<45} route          " + " ".join(f"{n:>9}" for n in ("classify","retrieve","grade","synth","total")))
    for p in probes:
        q = p["query"]
        d = hit(q)
        timings = parse_timings(d.get("trace", []))
        total = sum(timings.values())
        rows.append(
            {
                "query": q,
                "route": d.get("route"),
                "cache_hit": "cache hit" in " ".join(d.get("trace", [])),
                **timings,
                "total": total,
            }
        )
        print(
            f"{q[:43]:<45} {d.get('route'):<14} "
            f"{timings.get('classify',0):>9} {timings.get('retrieve',0):>9} "
            f"{timings.get('grade',0):>9} {timings.get('synthesize',0):>9} "
            f"{total:>9}"
        )
    return rows


def per_node_per_route(rows: list[dict]) -> None:
    by_route_node: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        for node in ("classify", "retrieve", "grade", "synthesize"):
            if node in r:
                by_route_node[(r["route"], node)].append(r[node])
    print(f"\n=== per-node-per-route (mean ms, n=samples) ===")
    print(f"{'route':<14} {'node':<12} {'mean':>7} {'p50':>7} {'p90':>7} {'n':>3}")
    for (route, node), vals in sorted(by_route_node.items()):
        vals_sorted = sorted(vals)
        mean = sum(vals) / len(vals)
        p50 = vals_sorted[len(vals_sorted) // 2]
        p90 = vals_sorted[int(len(vals_sorted) * 0.9)] if len(vals_sorted) > 1 else vals_sorted[0]
        print(f"{route:<14} {node:<12} {mean:>7.0f} {p50:>7} {p90:>7} {len(vals):>3}")


def main() -> None:
    probes = json.loads(PROBES_PATH.read_text())["in_corpus"]
    cold = run_pass("cold (first hit, no cache)", probes)
    warm = run_pass("warm (re-hit, retrieval cached)", probes)

    print("\n=== cache savings ===")
    print(f"{'probe':<45} {'cold ret':>9} {'warm ret':>9} {'cold total':>11} {'warm total':>11}")
    for c, w in zip(cold, warm):
        print(
            f"{c['query'][:43]:<45} {c.get('retrieve',0):>9} {w.get('retrieve',0):>9} "
            f"{c.get('total',0):>11} {w.get('total',0):>11}"
        )

    per_node_per_route(cold)


if __name__ == "__main__":
    main()
