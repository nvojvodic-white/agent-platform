"""Run RAGAS metrics against the in-corpus probe set.

Written for ragas 0.4.0 (the plan's 0.2.x snippet is incompatible with our
langchain 1.x stack). 0.4.0 uses EvaluationDataset[SingleTurnSample] with field
names user_input / response / retrieved_contexts / reference, and metric
instances that take the judge LLM at evaluate() time.

Judge = Claude (conservative on faithfulness); embeddings = OpenAI (for
answer-relevancy's question-regeneration similarity).

Usage:
  python -m app.rag.eval.ragas_eval [dense|sparse|hybrid] [k]
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import json

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
from app.rag.retrieval.vectorstore import get_retriever

load_dotenv()

PROBES_PATH = Path(__file__).parent / "probe_queries.json"
RESULTS_DIR = Path(__file__).parent / "ragas_results"


def build_eval_dataset(retriever_kind: str, k: int) -> EvaluationDataset:
    probes = json.loads(PROBES_PATH.read_text())["in_corpus"]
    chain = build_chain(k=k, retriever_kind=retriever_kind)
    retriever = get_retriever(k=k, kind=retriever_kind)

    samples = []
    for p in probes:
        if not p.get("ground_truth"):
            continue
        q = p["query"]
        result = chain.invoke(q)  # {docs, question, context, answer}
        samples.append(
            SingleTurnSample(
                user_input=q,
                response=result["answer"],
                retrieved_contexts=[d.page_content for d in result["docs"]],
                reference=p["ground_truth"],
            )
        )
    return EvaluationDataset(samples=samples)


def main(retriever_kind: str = "dense", k: int = 4) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building eval dataset (retriever={retriever_kind}, k={k})...")
    ds = build_eval_dataset(retriever_kind, k)
    print(f"Dataset: {len(ds.samples)} probes with ground truth")

    judge = LangchainLLMWrapper(
        ChatAnthropic(model="claude-sonnet-4-5", max_tokens=4096, max_retries=5)
    )
    embed = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

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

    df = result.to_pandas()
    out_path = RESULTS_DIR / f"ragas_{retriever_kind}_k{k}.csv"
    df.to_csv(out_path, index=False)

    metric_cols = [
        c
        for c in (
            "faithfulness",
            "answer_relevancy",
            "llm_context_precision_with_reference",
            "context_recall",
        )
        if c in df.columns
    ]
    print("\nMean scores:")
    for col in metric_cols:
        print(f"  {col:40}: {df[col].mean():.3f}")
    print(f"\nFull results: {out_path}")
    print(df[["user_input", *metric_cols]].to_string())


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "dense"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    main(retriever_kind=kind, k=k)
