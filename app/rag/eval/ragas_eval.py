"""Run RAGAS metrics against the in-corpus probe set.

Written for ragas 0.4.0 (the plan's 0.2.x snippet is incompatible with our
langchain 1.x stack). 0.4.0 uses EvaluationDataset[SingleTurnSample] with field
names user_input / response / retrieved_contexts / reference, and metric
instances that take the judge LLM at evaluate() time.

Judge = Claude (conservative on faithfulness); embeddings = OpenAI (for
answer-relevancy's question-regeneration similarity).

Day 10: --n-runs aggregation. The judge is non-deterministic, so faithfulness
and answer_relevancy drift ~0.04 between runs. Running n times and reporting
mean + stdev makes deltas attributable. Writes the per-run-averaged scores to
the CSV (compatible with Day 7-9 files) AND a richer JSON history record
(per-run scores, mean, std, timestamp, retriever, k, git_sha) for
compare_to_baseline.py.

Usage:
  python -m app.rag.eval.ragas_eval --retriever dense
  python -m app.rag.eval.ragas_eval --retriever hyde --n-runs 3
  python -m app.rag.eval.ragas_eval --retriever pdr --k 4 --n-runs 3
"""
import argparse
import datetime as dt
import json
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

# Must run before importing ragas: shims the langchain_community.chat_models
# .vertexai module that ragas 0.4.0 imports but community >= 0.4.2 removed.
from app.rag.eval import _ragas_compat  # noqa: F401
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from app.rag.chain.rag_chain import build_chain

load_dotenv()

PROBES_PATH = Path(__file__).parent / "probe_queries.json"
RESULTS_DIR = Path(__file__).parent / "ragas_results"
HISTORY_DIR = Path(__file__).parent / "ragas_history"

METRIC_COLS = (
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
)


def build_eval_dataset(
    retriever_kind: str, k: int, agent: bool = False
) -> EvaluationDataset:
    """Build a RAGAS dataset by running each probe through either a single
    retriever's chain (default) or the full routing agent (--agent).

    Agent mode uses the in-process get_agent() (the non-streaming LangGraph
    powering /agent_query): classify_query -> retrieve via the chosen route ->
    grade -> [rewrite -> retrieve] -> synthesize. Same code path as the
    production endpoint; just invoked directly to skip the HTTP layer for the
    eval (cleaner, same as how the other rows are measured)."""
    probes = json.loads(PROBES_PATH.read_text())["in_corpus"]

    if agent:
        from app.rag.agent.graph import get_agent

        _agent = get_agent()

        def run_one(q: str) -> tuple[str, list]:
            state = _agent.invoke({"question": q})
            return state.get("answer", ""), state.get("documents", [])

    else:
        chain = build_chain(k=k, retriever_kind=retriever_kind)

        def run_one(q: str) -> tuple[str, list]:
            result = chain.invoke(q)
            return result["answer"], result["docs"]

    samples = []
    for p in probes:
        if not p.get("ground_truth"):
            continue
        q = p["query"]
        answer, docs = run_one(q)
        samples.append(
            SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=[d.page_content for d in docs],
                reference=p["ground_truth"],
            )
        )
    return EvaluationDataset(samples=samples)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def run_once(ds, judge, embed) -> pd.DataFrame:
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=judge,
        embeddings=embed,
        run_config=RunConfig(max_workers=4),
    )
    return result.to_pandas()


def main(retriever_kind: str, k: int, n_runs: int, agent: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    # When --agent is set, output files are named "agent" regardless of
    # retriever_kind: the routing happens inside the graph, not via the
    # caller's choice, so retriever_kind is meaningless in that mode.
    label = "agent" if agent else retriever_kind
    print(
        f"Building eval dataset "
        f"({'agent (routing)' if agent else f'retriever={retriever_kind}'}, "
        f"k={k})..."
    )
    ds = build_eval_dataset(retriever_kind, k, agent=agent)
    print(f"Dataset: {len(ds.samples)} probes with ground truth")

    judge = LangchainLLMWrapper(
        ChatAnthropic(model="claude-sonnet-4-5", max_tokens=4096, max_retries=5)
    )
    embed = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    per_run_dfs: list[pd.DataFrame] = []
    for i in range(n_runs):
        if n_runs > 1:
            print(f"\n--- run {i + 1}/{n_runs} ---")
        per_run_dfs.append(run_once(ds, judge, embed))

    metric_cols = [c for c in METRIC_COLS if c in per_run_dfs[0].columns]

    # Per-probe per-metric stack across runs: shape (n_runs, n_probes)
    stacks = {c: np.array([df[c].values for df in per_run_dfs]) for c in metric_cols}
    mean_per_probe = {c: stacks[c].mean(axis=0) for c in metric_cols}
    std_per_probe = {c: stacks[c].std(axis=0, ddof=0) for c in metric_cols}

    # CSV: per-probe mean scores across runs (compatible with Day 7-9 schema).
    base = per_run_dfs[0].copy()
    for c in metric_cols:
        base[c] = mean_per_probe[c]
    out_csv = RESULTS_DIR / f"ragas_{label}_k{k}.csv"
    base.to_csv(out_csv, index=False)

    print("\nMean scores (over runs):")
    for c in metric_cols:
        mean_overall = float(np.mean(mean_per_probe[c]))
        # std of per-run-mean across runs - captures judge noise on this metric.
        per_run_means = stacks[c].mean(axis=1)
        std_overall = float(per_run_means.std(ddof=0))
        if n_runs > 1:
            print(f"  {c:40}: {mean_overall:.3f}  (run-to-run ±{std_overall:.3f})")
        else:
            print(f"  {c:40}: {mean_overall:.3f}")
    print(f"\nFull CSV: {out_csv}")
    print(base[["user_input", *metric_cols]].to_string())

    # JSON history: rich record per invocation for compare_to_baseline.py.
    history = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "retriever": label,
        "agent": agent,
        "k": k,
        "n_runs": n_runs,
        "judge_model": "claude-sonnet-4-5",
        "embed_model": "text-embedding-3-small",
        "n_probes": len(ds.samples),
        "metrics": {
            c: {
                "mean": float(np.mean(mean_per_probe[c])),
                "std_runs": float(stacks[c].mean(axis=1).std(ddof=0)),
                "per_probe_mean": mean_per_probe[c].tolist(),
                "per_probe_std": std_per_probe[c].tolist(),
            }
            for c in metric_cols
        },
        "queries": [s.user_input for s in ds.samples],
    }
    ts_safe = history["timestamp"].replace(":", "-")
    out_json = HISTORY_DIR / f"{ts_safe}_{label}_k{k}.json"
    out_json.write_text(json.dumps(history, indent=2))
    print(f"History: {out_json}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--retriever",
        default="dense",
        help="dense | sparse | hybrid | hybrid_40_60 | hyde | multi_query | pdr | semantic",
    )
    p.add_argument("--k", type=int, default=4)
    p.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of full eval runs to average (smooths judge noise on faith/relevancy)",
    )
    p.add_argument(
        "--agent",
        action="store_true",
        help="Evaluate the full routing agent (classify -> route -> grade -> "
        "[retry] -> synthesize) instead of a single retriever. --retriever is "
        "ignored in this mode.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        retriever_kind=args.retriever,
        k=args.k,
        n_runs=args.n_runs,
        agent=args.agent,
    )
