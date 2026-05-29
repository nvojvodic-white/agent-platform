# RAG findings

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
