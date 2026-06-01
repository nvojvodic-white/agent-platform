# RAG retrieval scorecard

End-of-project summary of every measured retriever variant against the same probe set.
Fill in at the end of the build from the latest `ragas_history/*.json` records.

- **Probe set:** `app/rag/eval/probe_queries.json` (7 in-corpus probes with hand-written
  ground truths; see [findings.md](findings.md) for per-probe diagnostics)
- **Judge:** Claude `claude-sonnet-4-5`, `max_tokens=4096`, `max_retries=5`
- **Embeddings:** OpenAI `text-embedding-3-small`
- **k:** 4 unless noted
- **Run-to-run noise floor (judge):** faithfulness/answer_relevancy ~0.04;
  context_precision/context_recall ~0 (deterministic given fixed retrieval)

## Headline retrievers

| retriever | faithfulness | answer_relevancy | context_precision | context_recall | n_runs | day | notes |
|---|---|---|---|---|---|---|---|
| dense (baseline) | _ | _ | _ | _ | _ | 7 | recursive 800/120, fixed default |
| sparse | _ | _ | _ | _ | _ | 7 | BM25 with custom preprocessor |
| hybrid (50/50) | _ | _ | _ | _ | _ | 8 | deferred from Day 7, kind=hybrid |
| hybrid_40_60 | _ | _ | _ | _ | _ | 8 | deferred from Day 7 |
| hyde | _ | _ | _ | _ | _ | 8 | hypothetical-answer embedding |
| multi_query | _ | _ | _ | _ | _ | 8 | re-ranked union, k=4 truncated |
| pdr | _ | _ | _ | _ | _ | 9 | 400-char children, 2000-char parents |
| semantic | _ | _ | _ | _ | _ | 9 | SemanticChunker topic boundaries |

## Per-probe diagnostic table (dense baseline)

The probes that drive variance across retrievers. Fill from the dense history record.

| probe | precision | recall | faith | failure mode (if any) |
|---|---|---|---|---|
| Who is Gandalf? | _ | _ | _ | clean control |
| Who killed Smaug? | _ | _ | _ | multi-hop / synthesis-recovered |
| What rings did the Dwarves get? | _ | _ | _ | _ |
| Battle of Five Armies | _ | _ | _ | _ |
| Beren and Lúthien | _ | _ | _ | corpus-coverage limit (no Beren article) |
| What is mithril? | _ | _ | _ | low recall on dense → fixed by semantic |
| Tom Bombadil | _ | _ | _ | low recall on dense → fixed by semantic; corpus gap closed Day 6.5 |

## Headline decisions

- **Default retriever:** `_` (justification from the table)
- **Available alternates:** `_` (which `kind=` values stay wired, and when to use each)
- **Known recall gaps that survived every intervention:** `_`
- **Most surprising falsified hypothesis:** `_`

## How to regenerate this scorecard

```bash
make eval-all                                # run all retrievers --n-runs=3
make eval-compare RETRIEVER=hyde             # diff vs baseline
# then update the tables above from app/rag/eval/ragas_history/*.json
```
