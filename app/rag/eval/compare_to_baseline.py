"""Compare the most recent RAGAS history record to a baseline.

Reads JSON history files written by ragas_eval.py (--n-runs) and prints a
delta table. Flags metrics that moved more than the run-to-run noise floor.

Usage:
  # compare latest two runs for the same retriever
  python -m app.rag.eval.compare_to_baseline --retriever dense

  # compare two specific files
  python -m app.rag.eval.compare_to_baseline --baseline path/to/base.json \\
      --current path/to/curr.json

  # compare across retrievers (latest each)
  python -m app.rag.eval.compare_to_baseline --baseline-retriever dense \\
      --retriever hyde

Significance: a delta is flagged "significant" if abs(delta) > max(0.02,
2 * baseline_run_std). 0.02 is a small absolute floor for the deterministic
context metrics; the 2x-std term sets the bar for the noisy faith/relevancy
metrics using THIS retriever's own measured run-to-run noise.
"""
import argparse
import json
from pathlib import Path

HISTORY_DIR = Path(__file__).parent / "ragas_history"
METRIC_ORDER = (
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
)
ABS_NOISE_FLOOR = 0.02


def _latest(retriever: str | None, k: int | None) -> Path | None:
    files = sorted(HISTORY_DIR.glob("*.json"))
    if retriever:
        files = [f for f in files if f"_{retriever}_" in f.name]
    if k is not None:
        files = [f for f in files if f.name.endswith(f"_k{k}.json")]
    return files[-1] if files else None


def _second_latest(retriever: str | None, k: int | None) -> Path | None:
    files = sorted(HISTORY_DIR.glob("*.json"))
    if retriever:
        files = [f for f in files if f"_{retriever}_" in f.name]
    if k is not None:
        files = [f for f in files if f.name.endswith(f"_k{k}.json")]
    return files[-2] if len(files) >= 2 else None


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, help="Path to baseline history JSON")
    ap.add_argument("--current", type=Path, help="Path to current history JSON")
    ap.add_argument("--retriever", help="Use latest history for this retriever as current")
    ap.add_argument("--baseline-retriever", help="Use latest history for this retriever as baseline")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    if args.current:
        current = _load(args.current)
        current_path = args.current
    elif args.retriever:
        p = _latest(args.retriever, args.k)
        if not p:
            raise SystemExit(f"No history for retriever={args.retriever} k={args.k}")
        current = _load(p)
        current_path = p
    else:
        raise SystemExit("Provide --current or --retriever")

    if args.baseline:
        baseline = _load(args.baseline)
        baseline_path = args.baseline
    elif args.baseline_retriever:
        p = _latest(args.baseline_retriever, args.k)
        if not p:
            raise SystemExit(
                f"No history for baseline retriever={args.baseline_retriever}"
            )
        baseline = _load(p)
        baseline_path = p
    else:
        # Default: previous run of the SAME retriever.
        p = _second_latest(current.get("retriever"), args.k)
        if not p:
            raise SystemExit(
                f"No prior history for retriever={current.get('retriever')} to compare against"
            )
        baseline = _load(p)
        baseline_path = p

    print(f"baseline: {baseline_path.name}  ({baseline['retriever']}, "
          f"n_runs={baseline['n_runs']}, git={baseline['git_sha']})")
    print(f"current : {current_path.name}  ({current['retriever']}, "
          f"n_runs={current['n_runs']}, git={current['git_sha']})")
    print()

    rows = []
    for m in METRIC_ORDER:
        b = baseline["metrics"].get(m)
        c = current["metrics"].get(m)
        if b is None or c is None:
            continue
        delta = c["mean"] - b["mean"]
        threshold = max(ABS_NOISE_FLOOR, 2 * b["std_runs"])
        if abs(delta) > threshold:
            tag = "↑ improved" if delta > 0 else "↓ regressed"
        else:
            tag = "≈ within noise"
        rows.append((m, b["mean"], c["mean"], delta, threshold, tag))

    name_w = max(len(r[0]) for r in rows) if rows else 30
    print(
        f"{'metric':<{name_w}}  {'baseline':>9}  {'current':>9}  "
        f"{'delta':>8}  {'threshold':>9}  status"
    )
    print("-" * (name_w + 60))
    for name, b, c, d, t, tag in rows:
        sign = "+" if d >= 0 else ""
        print(
            f"{name:<{name_w}}  {b:9.3f}  {c:9.3f}  "
            f"{sign}{d:7.3f}  {t:9.3f}  {tag}"
        )


if __name__ == "__main__":
    main()
