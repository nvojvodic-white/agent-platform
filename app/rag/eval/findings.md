# RAG findings

## Day 7 — RAGAS automated eval (replaces Day 6 hand-scoring)

Built `app/rag/eval/ragas_eval.py` (ragas 0.4.0; the plan's 0.2.6 pin conflicts with
our langchain 1.x stack, so the harness uses the 0.4.0 `EvaluationDataset` /
`SingleTurnSample` API). Judge = Claude (`claude-sonnet-4-5`, max_tokens 4096);
embeddings = OpenAI `text-embedding-3-small`. Ground truths for all 7 in-corpus probes
were written by hand against what the corpus actually supports (not the platonic Tolkien
answer). Per-probe CSVs in `app/rag/eval/ragas_results/`.

Mean scores (7 probes, k=4):

| retriever | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| dense  | 0.913 | 0.803 | 0.881 | 0.821 |
| sparse | 0.832 | 0.691 | 0.440 | 0.179 |
| hybrid 50/50  | (deferred) | | | |
| hybrid 40/60  | (deferred) | | | |

**Quantified the Day 6 verdict:** dense beats sparse on every metric, decisively on the
two retrieval metrics — context_precision 0.88 vs 0.44, context_recall 0.82 vs 0.18.
Sparse's BM25 literalism (the Ori / Farmer Maggot noise seen by hand on Day 6) shows up
as a precision/recall collapse; "What rings did the Dwarves get?" even scored
answer_relevancy 0.0 under sparse because the retrieved chunks were too off-topic to
support a relevant answer. This is the hand-scored Day 6 finding, now with numbers.

**Faithfulness is high but not uniform.** Dense 0.91 — the strong prompt does most of the
grounding work, a partial answer to the deferred Day 5 question. But it is not a flat
1.0: the "Battle of Five Armies" probe scored 0.61 faithfulness under dense (the answer
made battle-detail claims the retrieved chunks only partly support). So retrieval quality
*does* interact with groundedness — subtler than the clean two-layer model.

**The two context metrics surfaced real corpus-quality signals the eye missed:**
- Mithril: context_recall 0.33 under dense. The answer is faithful, but the retrieved WP
  Mithril chunks lean definitional (etymology/properties) and miss the Moria / Bilbo's-mail
  facts in the ground truth. A recall gap on an otherwise "clean" probe.
- Tom Bombadil (post Day-6.5 fix): recall 0.67, precision 0.92 under dense. The new
  Wikipedia article supports most but not all canonical facts — the article isn't
  maximally dense, exactly the test Day 6.5 set up.

**Method caveats worth keeping:** (1) RAGAS judge scores are non-deterministic —
faithfulness on the same dense run moved 0.938 → 0.913 across two runs, so treat sub-0.05
differences as noise. (2) The judge initially truncated (NaN) at max_tokens 2048 on the
Battle probe's claim decomposition; raised to 4096. (3) The hybrid 50/50 and 40/60 rows
are DEFERRED: Anthropic was returning intermittent 500s (~1 in 3 calls) during this
session, which killed both hybrid runs mid-evaluation. dense + sparse completed cleanly
before the instability. Hybrid rows to be filled in when the API stabilizes; Day 6's
hand-scored hybrid evidence (neutral-to-slightly-worse than dense) stands in the interim.

**Net:** dense remains the default, now with a measured baseline that can be re-run in
~2 min per retriever whenever anything downstream changes.

## Day 6 — Hybrid retrieval (BM25 + dense)

Built a swappable retriever (`get_retriever(kind=dense|sparse|hybrid)`, EnsembleRetriever
RRF fusion) and compared dense / sparse / hybrid_50_50 / hybrid_40_60 across the 7
in-corpus probes. **Kept dense as the default** — the measurement did not support
promoting hybrid on this corpus.

### The Bombadil thesis was falsified

The day's central hypothesis ("dense routes 'Tom Bombadil' to the generic Middle-earth
peoples article; BM25 sees 'Bombadil' as a rare token and rescues it; hybrid fixes it")
was **wrong**. No retriever — including pure BM25, which should trivially match the
literal token "Bombadil" — surfaces a "Tom Bombadil" article, because **the corpus does
not contain one** (the closest page is "The Adventures of Tom Bombadil", a poetry
collection). This was never a dense-vs-sparse problem; it is a corpus-coverage gap.
Exactly the failure mode the plan flagged under "the prediction might not pan out". The
real fix is ingesting Bombadil's article, not any retriever change.

### Dense was already the best retriever here

| Probe | dense | sparse | hybrid_50_50 | hybrid_40_60 |
|---|---|---|---|---|
| Who is Gandalf? | H@1 | H@1 | H@1 | H@1 |
| Who killed Smaug? | H@k | miss | H@k | miss |
| Dwarves' rings | H@1 | miss | H@1 | miss |
| Battle of Five Armies | H@1 | H@k | H@1 | H@1 |
| Beren and Lúthien | H@1 | H@k | H@1 | H@1 |
| What is mithril? | H@1 | H@1 | H@1 | H@1 |
| Tom Bombadil | miss | miss | miss | miss |

Dense: H@1 on 5/7, H@k on Smaug, only Bombadil missed. Pure sparse is clearly worse
(BM25 literalism pulls token-co-occurrence noise like Ori / Ufedhin / Farmer Maggot).
**hybrid_40_60 actively regresses two probes** (Smaug and Dwarves'-rings) by
over-weighting sparse. hybrid_50_50 matches dense everywhere except a rank-ordering
wobble on Smaug — neutral at best. So the Tolkien-corpus assumption "lots of rare named
entities → favor BM25" did not hold; dense embeddings already capture the entities well.

### A real BM25 bug, fixed

The first comparison run had sparse returning nonsense ("Who is Gandalf?" → Figwit,
Radagast, Saruman). Cause: BM25Retriever's default preprocessor is `text.split()` — no
lowercasing, no punctuation stripping, no stopword removal. So "Gandalf?" != "Gandalf"
and common words ("who", "is") dominated scoring. Added a preprocessor (lowercase, regex
tokenize, drop a small English stopword set); after that, sparse nails Gandalf and
Mithril. The fix is worth keeping regardless of the promotion decision — it's a generic
correctness fix for the sparse path.

### Also fixed: EnsembleRetriever returned too many docs

EnsembleRetriever's RRF returns the deduped union of all sub-retriever candidates, so
feeding it 2k from each retriever yielded ~11 docs for k=4. Wrapped it in a
`_TopKEnsembleRetriever` that truncates in `rank_fusion` (the method `invoke` actually
routes through) to return exactly top-k.

### Net

Hybrid is built, correct, and A/B-swappable, but does not earn promotion on this corpus:
it is neutral at best and regresses two queries at 40/60. Dense stays the default. The
one genuine candidate-set-miss (Bombadil) is a corpus gap, addressable only by expanding
the corpus, not by retrieval tuning.

**Deferred:** the Day 4 agent-probe re-run — the change kept dense as the live default,
so agent behavior is unchanged from Day 4 and a re-run is not strictly required; will run
it if/when convenient with Anthropic credits to confirm.
