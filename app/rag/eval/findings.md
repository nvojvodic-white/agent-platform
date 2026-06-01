# RAG findings

## Day 12 (Hour 1) — Latency stack + targeted retrieval cache

Added `@timed` decorators to every LangGraph node and an `lru_cache(maxsize=256)`
on retrieval, keyed by `(question, route, k)`. Ran the 7-probe sweep twice via
`/agent_query_debug` — cold and warm — and parsed the `[timing] node: ms` lines
the decorator now appends to the trace.

**Cold-pass per-node latency (ms, n=7):**

| node | mean cold ms | per-route detail |
|---|---|---|
| classify | ~2500 | LLM call, small prompt |
| retrieve (dense) | ~260 | OpenAI embed + Chroma |
| retrieve (semantic) | ~1000 (213-3080) | same path, larger semantic chunks |
| retrieve (hyde) | ~6100 | **adds a hypothetical-generation LLM call** |
| grade | ~3000 | LLM call, larger prompt |
| synthesize | ~7000 | LLM call, ~1KB output |
| **total per query** | **~14000** | end-to-end, cold |

**Where the time goes:** synthesize is the single biggest line (~50% on
general/dense probes, less on hyde probes where retrieve is also large). The
three LLM calls (classify, grade, synthesize) together dominate every route at
roughly 12-13s combined. The hyde retrieve cost (+6s LLM call) is the only
non-LLM-dominant route-specific term — and it's where the routing-vs-best-fixed
tradeoff has a real latency dimension, not just a quality one.

**Cache impact (warm pass):** retrieve drops to 0 ms on every probe (confirmed
via the `cache hit` flag in the trace). End-to-end totals fall ~10-30% on
hyde-routed probes (Smaug 14.8s → 9.8s, Dwarves 17.5s → 11.0s — saving ~5-6s
each on the +1-LLM-call retrieve) and ~0-5% on dense-routed probes (retrieve
was already 260ms; the gain is lost in LLM noise). Tom Bombadil's warm pass
was actually slower than cold (14.5s → 15.8s) because the grade LLM call
varied by ~4s on that probe — judge latency carries ~2-3s of run-to-run noise
that swamps small wins.

**Production-engineering reading:**
- The cache is genuinely useful where it matters most (hyde routes). On
  dense, it's free correctness insurance against repeated queries but doesn't
  move the latency meaningfully.
- Synthesize is the next-largest fixed cost. Streaming via FastAPI
  `StreamingResponse` + `langgraph.astream` would not reduce total tokens but
  would give perceived ~2-3s time-to-first-byte. Deferred (1-2h refactor to
  async-all-the-way-down for a UX-only win — matches the README's "manual-only
  CI" cost-aware framing).
- Considered semantic caching (similar-question lookup via embedding distance)
  and rejected for portfolio scope. Worth adding when there's production
  traffic with paraphrased repeats; for the 7-probe demo loop, exact-string
  LRU catches every meaningful hit and avoids the silent-wrong-answer failure
  mode of fuzzy-match caches.

## Day 11 — LangGraph routing agent

Built a LangGraph state machine in `app/rag/agent/graph.py` that classifies each
question into `definitional` / `multi_hop` / `general` and routes to the
retriever each class wins on per Days 7-9 (semantic / hyde / dense). The graph
also grades retrieved docs and can rewrite + retry once on a `poor` grade.
Exposed as `/api/v1/rag/agent_query` (production) and `/api/v1/rag/agent_query_debug`
(returns route, grade, attempt, trace). RAGAS deferred to end-of-project to save
cost; today's measurement is the routing layer itself.

**7-probe sweep against `/agent_query_debug`** (`run_agent_v2_probes.py`):

| probe | expected (plan) | got | match | grade | attempts |
|---|---|---|---|---|---|
| Who is Gandalf? | definitional | definitional | ✓ | partial | 1 |
| Who killed Smaug? | multi_hop | multi_hop | ✓ | relevant | 1 |
| What rings did the Dwarves get? | multi_hop | multi_hop | ✓ | relevant | 1 |
| Battle of Five Armies | general | general | ✓ | partial | 1 |
| Beren and Lúthien | general | **definitional** | ✗ | relevant | 1 |
| What is mithril? | definitional | definitional | ✓ | relevant | 1 |
| Tom Bombadil | definitional | definitional | ✓ | partial | 1 |

**Classifier match: 6/7 (86%) against the plan's hypothesis.** Grades: 4 relevant
/ 3 partial / 0 poor. **Zero retries triggered** — the design's "retry only on
`poor`, not on `partial`" decision means the agent commits to synthesis whenever
the grader is at least lukewarm. That's a deliberate tradeoff; a more aggressive
retry policy could lift quality at meaningful cost.

**Beren & Lúthien is the only "miss" — and the plan's expectation is the part
that's wrong.** My Hour 1 pre-registered prediction was `definitional` for that
probe (a request to identify two specific entities), which is what the
classifier chose. The plan's table had `general`. The classifier's reasoning
("requesting concise information about who they are") is sound, and semantic
retrieved 4 strong Lúthien chunks graded `relevant`. So this is the classifier
agreeing with my prior, not with the plan's prior — a useful reminder that the
expected-routes table is a hypothesis, not ground truth.

**Two qualitative wins from per-query routing that the prior single-retriever
runs left on the table:**
- *Smaug → multi_hop → hyde*: hyde surfaced `Destruction of Lake-town` at rank 1
  (the Day 8 hyde-only precision win), and the grader returned `relevant` first
  try. Dense never reached that chunk; the agent picked the right tool
  automatically.
- *Dwarves → multi_hop → hyde*: on Day 6 hyde+sparse hybrids regressed this
  probe via BM25 noise (Ori / Farmer Maggot); standalone hyde on Day 8 was
  fine; the agent now selectively *uses* hyde here. This is the per-query
  routing extracting Day 8 wins without paying Day 6's hybrid-tail precision
  cost.

**Three `partial` grades worth noting** (Gandalf, Battle, Bombadil): all three
were graded `partial` not `poor`, so the no-retry-on-partial policy held. But
the grader's reasoning was honest in each case ("don't directly explain who or
what Gandalf fundamentally is", "lack comprehensive details about the battle's
causes/progression", "retrieved text is fragmented"). Those are real
sub-ceiling observations the grader is correctly surfacing — and a future
refinement could be a `partial`-with-rewrite policy on probes where the answer
might still improve, paid in extra LLM calls.

**Net:** the routing layer works as designed. The classifier is reliable
(6/7, only "miss" is the plan's hypothesis being wrong, not the classifier).
The grader is honest and inspectable. End-to-end RAGAS quality vs single
retrievers is deferred to end-of-project to manage cost.

## Day 9 — Chunk-granularity: parent-document retrieval & semantic chunking

Tested whether changing what "a chunk" is can move the ~0.82 recall ceiling that
Day 8's query transforms could not. Two chunk-axis interventions, both in separate
Chroma collections (production `middle_earth` untouched):
- **PDR** (`build_pdr_index.py`, `pdr.py`): 400-char children for embedding, 2000-char
  parents for context (2340 parents / 12378 children, no article truncated). Retrieve
  children, dedupe by parent_id, return top-k parents.
- **Semantic** (`build_semantic_index.py`, `semantic.py`): the Day-2 deferred variant.
  SemanticChunker splits at embedding-distance topic boundaries (2268 chunks, avg 1464
  chars vs the recursive baseline's 800).

Four-way RAGAS (mean over 7 probes, k=4; judge = Claude):

| retriever | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| dense (baseline) | 0.949 | 0.801 | **0.881** | 0.821 |
| pdr      | 0.958 | 0.840 | 0.873 | **0.595** |
| semantic | **0.988** | 0.842 | 0.758 | **0.833** |

**Answer to the substrate-ceiling question: granularity moves recall, but only a little,
and not for free.** Semantic chunking is the only intervention all week to lift recall
above dense (0.821 → 0.833) — and it specifically *fixed the two worst dense recall
probes*: Mithril 0.33 → 1.0 and Bombadil 0.67 → 1.0. So the recall gaps on those probes
were genuinely granularity-bound: the recursive 800-char splitter was cutting their facts
across chunk boundaries, and topic-aligned chunks recovered them. But the recall lift is
small in aggregate and **bought with precision**: semantic's context_precision fell
0.881 → 0.758 (bigger topical chunks carry more off-target text; Battle of Five Armies
precision cratered to 0.33). It's a precision/recall *trade*, not a free lift — which
reads as "near the corpus-bound bottom" more than "granularity is the unlock."

**PDR regressed recall hard (0.595, −0.23) — a real mechanism finding, not a bug.**
Dedup-by-parent forces k *distinct* parents. When a probe's ground-truth facts are
concentrated in one article (the common case here), dense returns 4 chunks all from that
article, but PDR returns 4 different parents and diversifies *away* from the fact-dense
one. Verified on the Dwarves probe: dense = [Dwarves×3, Dwarves-in-ME]; PDR = [Durin III,
Seven Hoards of the Dwarf-kings, Dwarves, Dwarves] — it spent two of its four slots on
tangential articles. PDR's diversification helps spread-out facts and hurts concentrated
ones; this corpus has mostly concentrated facts, so PDR loses. (Mithril was the exception:
PDR lifted it 0.33 → 0.67, because Mithril facts *were* split across parents.)

**Faithfulness rose with chunk size** (dense 0.95 → pdr 0.96 → semantic 0.99): larger,
more coherent chunks give the LLM more complete context, so it makes fewer unsupported
claims. Monotonic with chunk size across all three — a clean secondary signal.

**Decision:** dense stays the default. Neither intervention is a clear win: PDR loses on
recall, semantic trades precision for a small recall gain. But semantic's targeted fix of
the Mithril/Bombadil recall gaps is worth keeping available (`kind="semantic"`) and points
at a possible Day 11+ refinement — a *hybrid of granularities* (semantic chunks for
recall-hard entity probes, recursive for precision-sensitive event probes) rather than one
global chunk size.

**Dependency note:** ragas 0.4.0 top-level-imports `langchain_community.chat_models
.vertexai.ChatVertexAI`, which community ≥0.4.2 removed — but langchain-experimental 0.4.2
(needed for SemanticChunker) forces community ≥0.4.2. Resolved with a small compat shim
(`app/rag/eval/_ragas_compat.py`) that stubs the dead import before ragas loads; ragas
never instantiates ChatVertexAI under our Claude judge, so the stub is inert. requirements
bumped: langchain-community 0.4.1→0.4.2, langchain-classic 1.0.0→1.0.7, +langchain-experimental 0.4.2.

## Day 8 — Query transformation: HyDE and Multi-Query

Built two query-transformation retrievers and measured both with the Day 7 RAGAS harness:
- `app/rag/retrieval/hyde.py` — generate a hypothetical answer, embed it, retrieve on it.
- `app/rag/retrieval/multi_query.py` — generate ~3 phrasings, retrieve each, dedupe,
  re-rank the union by similarity to the original query, keep top-k.

Both wired into `get_retriever(kind=hyde|multi_query)`. LLM calls (judge + both
transforms) hardened with `max_retries=5` after Day 7's transient-500 trouble.

Five-way table (mean over 7 in-corpus probes, k=4):

| retriever | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| dense (Day 7 baseline) | 0.913 | 0.803 | 0.881 | 0.821 |
| dense (Day 8 rerun)    | 0.949 | 0.801 | 0.881 | 0.821 |
| sparse                 | 0.832 | 0.691 | 0.440 | 0.179 |
| hyde                   | 0.928 | 0.812 | **0.937** | 0.821 |
| multi_query            | 0.900 | 0.809 | 0.881 | 0.821 |

**Judge-variance baseline (the point of the dense rerun):** the two *context* metrics
are identical across the two dense runs (0.881, 0.821) — they're deterministic given
fixed retrieval (per-chunk relevance judgments are stable). Faithfulness moved +0.036
and relevancy wobbled. So: trust any delta on context_precision/recall, but only believe
faithfulness/relevancy deltas larger than ~0.04.

**HyDE is the one real win — on precision, not recall.** context_precision 0.937 vs
dense 0.881 (+0.056 on a deterministic metric = real). HyDE's answer-shaped queries pull
cleaner chunks: Mithril precision 0.75→1.0, Dwarves 0.58→0.83. The Smaug case (HyDE's
textbook target) confirmed it qualitatively too — HyDE surfaced a "Destruction of
Lake-town" chunk at rank 1 that dense never reached.

**But HyDE's recall mean (0.821) hides a per-probe swap, not a tie:**
- Mithril recall 0.33 → 0.667 — HyDE *fixed* the worst dense recall probe. Mithril was a
  "query-shape" problem (the plan's diagnostic): reshaping the query reached more of the
  article.
- Battle of Five Armies recall 1.0 → 0.667 — HyDE *regressed* this. The hypothetical
  answer pulled it slightly off the exact battle chunks.

So HyDE improves precision clearly and redistributes recall (helps shape-sensitive
probes, costs already-good ones) rather than lifting the recall ceiling. Net: a genuine
precision lever worth keeping available, but not an unambiguous default-replacement.

**Multi-query is indistinguishable from dense** (precision 0.881, recall 0.821 — both
identical; faith/relevancy within noise). After re-ranking the variant union by
similarity to the original query, it converges to almost exactly dense's chunks — so it
adds the LLM cost of variant generation for no measurable gain on this corpus.

**Method note — a real bug found and fixed mid-eval:** the first multi_query run scored
catastrophically (recall 0.41, three probes at 0.0 relevancy). Investigation showed it
was a *truncation artifact*, not the technique: `MultiQueryRetriever` returns the deduped
union grouped by variant in generation order, not relevance-ranked, so naive `[:k]` kept
a weak variant's off-topic chunks and dropped good ones at positions 5-7. Fixed by
embedding the original query and ranking the union by cosine similarity before truncating.
The 0.41 number was measuring the bug, not multi-query — re-ran fairly to get 0.821.

**Net / decision:** dense stays the default. HyDE is the only technique that moved a
trustworthy metric (precision +0.056) and is retained as an opt-in `kind`. Recall sits at
~0.82 across dense/hyde/multi_query and was not lifted by any query transform — evidence
that the recall ceiling on this corpus is set by chunking/corpus coverage, not query
phrasing. That points at parent-document / chunk-level work (Day 9) as the next lever for
the Mithril/Bombadil recall gaps, not more query engineering.

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
