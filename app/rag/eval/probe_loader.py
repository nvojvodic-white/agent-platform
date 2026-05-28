"""Load probe slices from probe_queries.json.

Slices:
  in_corpus              - normal RAG quality probes (have retrievable context)
  out_of_corpus          - refusal probes (no grounding exists)
  adversarial_real       - prior-vs-context probes where the well-known answer
                           is genuinely ABSENT from the retrieved context
  adversarial_restricted - probes where the answer IS in the corpus but we force
                           off-topic retrieval (force_context_query) to isolate
                           prior-vs-context
  all                    - in_corpus + adversarial_real (the comparable set)
"""
import json
from pathlib import Path

_PATH = Path(__file__).parent / "probe_queries.json"


def _data() -> dict:
    return json.loads(_PATH.read_text())


def load(slice_name: str = "all") -> list[dict]:
    d = _data()
    adv = d["adversarial_faithfulness"]
    if slice_name == "in_corpus":
        return d["in_corpus"]
    if slice_name == "out_of_corpus":
        return d["out_of_corpus"]
    if slice_name == "adversarial_real":
        return adv["real_gaps"]
    if slice_name == "adversarial_restricted":
        return adv["restricted_context"]
    if slice_name == "all":
        return d["in_corpus"] + adv["real_gaps"]
    raise ValueError(f"Unknown slice: {slice_name}")
