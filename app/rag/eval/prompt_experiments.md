# Day 5 — Prompt experiments: where the prompt has leverage over Claude

All experiments hold retrieval fixed (k=4, verified deterministic across 5 runs per
query) so the system prompt is the only variable. Scores are draft, by-eye (1-5), read
from the actual transcripts in this run. Generation model: claude-sonnet-4-5.

## Headline

On this corpus Claude's faithfulness floor is already high: the "strict" instruction
barely changed refusal behaviour because baseline already refused every adversarial
probe. Where the prompt showed clear, repeatable leverage was **refusal quality** (how
it declines), not faithfulness (whether it declines). And prompts could **suppress** a
content prior and even **rebalance** it (Exp 3), though suppression visibly fights the
prior rather than erasing it.

## Methodology finding (a result before the experiments produced answers)

- Of 15 naive "facts everyone knows" probes, 12 were PRESENT in the retrieved context
  (Shadowfax, Nine, Sting, Andúril, Barad-dûr, Glóin, Witch-king, ...). The corpus is
  dense enough that obvious facts are in-context, so answering them is not a leak. Only
  3 were genuine gaps.
- One probe was silently invalidated by REWORDING it between gap-validation and
  execution: "How long did Bilbo live before leaving the Shire?" retrieves 4 Bilbo
  chunks with no "eleventy-one"; the longer reword "...his age at the farewell party"
  pulls a different 4th chunk (Frodo party text) that DOES contain it. Confirmed stable
  across 5 runs each way. Lesson: probe text must be byte-identical between
  gap-validation and execution.
- To get reliable gaps regardless of corpus density, added restricted-context probes:
  ask question X but deliberately retrieve context for an unrelated query, so the answer
  is absent by construction.

## Experiment 1 — Faithfulness under pressure (baseline vs strict)

Restricted-context probes (answer absent by construction):

| Probe | forced context from | baseline | strict |
|---|---|---|---|
| Sauron's tower (Barad-dûr) | hobbit meals | refused cleanly | refused cleanly |
| Who forged the One Ring (Sauron) | Shire geography | refused cleanly | refused cleanly |

Real corpus gaps:

| Probe | baseline | strict |
|---|---|---|
| Bilbo's ring-finding year (TA 2941) | no leak — declined the year, cited related dates | no leak |
| Treebeard's forest (Fangorn) | no leak | no leak |

Result: zero leaks in either variant, even for "who forged the One Ring" under forced
off-topic context. Strict had no measurable effect on faithfulness because baseline was
already at the ceiling. Confirms the plan's flagged possibility: faithfulness is robust
enough here that prompt strictness is near-inert, so a leaner prompt is viable.

## Experiment 2 — Faithfulness / helpfulness frontier (4 variants × 10 probes, verified run)

On the 7 clean in-corpus probes all four variants gave solid grounded, cited answers, so
behaviour differentiated entirely on the gap probes. Verbatim from this run:

Bilbo's ring-finding year (TA 2941):
- permissive — refused from context, then LEAKED via "Beyond the corpus: Bilbo found the
  One Ring in TA 2941 ...".
- strict — clean refusal, no year.
- strict_hedge — refusal plus "What the context covers / What the context does not cover".

Treebeard's forest (Fangorn):
- permissive — refused from context, then LEAKED "Beyond the corpus: ... Fangorn Forest".
- strict — clean refusal.
- strict_hedge — refusal plus covers/missing structure; did NOT name Fangorn.

Draft scores (F = all stated facts in context; H = helpfulness):

| Variant | F | H |
|---|---|---|
| permissive | 2 | 4 |
| baseline | 5 | 3 |
| strict | 5 | 3 |
| strict_hedge | 5 | 5 |

Winner: strict_hedge — matches baseline faithfulness, restores the helpfulness bare
strict loses by turning refusals into "here's what I found / here's the gap". (Earlier
exploratory run once had strict_hedge name Fangorn in its "missing" line; it did not
recur in this verified run, where it said "the specific name". Run-to-run risk to watch.)

## Experiment 3 — Routing vs ranking (bare model, no retrieval)

Q: "Who is the wisest character in fantasy literature?"

- neutral — led with Gandalf, then Dumbledore, Galadriel. Tolkien-first, matching Day 4.
- diversity — genuinely rebalanced across traditions and named its own skew; Gandalf demoted.
- anti_tolkien — suppressed Tolkien but visibly fought the prior, starting a Galadriel
  reference then self-correcting mid-sentence.

Result: the prompt CAN reach the ranking layer, not only gate tool routing (the Day 4
question). Diversity produced real rebalancing, not just Tolkien removal — stronger than
the Day 4 hypothesis that prompts could only suppress.

## What I now believe about prompt leverage over this model

1. Faithfulness/abstention: little marginal leverage here — base behaviour is near-ceiling.
   Keep one strong "only from context" rule; stacking strictness buys nothing on this corpus.
2. Refusal quality (helpfulness): strong, reliable leverage. "Say what you found and
   what's missing" reliably converts a curt refusal into a useful one.
3. Content ranking/priors: partial leverage — can both suppress and genuinely rebalance,
   but suppression visibly fights the prior rather than erasing it.

Promoted to the live RAG prompt: strict_hedge (rules 7-8 in app/rag/chain/prompts.py).
