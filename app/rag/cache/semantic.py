"""Semantic response cache: embed the incoming question, return a cached
response if any prior question is within `threshold` cosine similarity.

Rejected for the demo workload (exact-string LRU catches every
repeat; semantic caching adds silent-wrong-answer risk and staleness on prompt
/ retriever / corpus changes). This implementation is the 'minimal, opt-in,
documented' version: disabled by default, behind SEMANTIC_CACHE_ENABLED=1,
wired only into /agent_query (NOT streaming, NOT multi-turn).

Known limitations, all by design rather than oversight:
  - In-memory only. Dies on process restart. By design: a process restart is
    the explicit invalidation signal when the corpus/prompt/retriever changes.
  - No cache invalidation hooks. If you change the prompt or re-ingest the
    corpus while the process keeps running, cached entries become stale and
    will return wrong-for-now answers. Restart to clear.
  - The threshold is a guess (0.97 default). Without paraphrase probe data we
    cannot claim it is tuned. Configurable via SEMANTIC_CACHE_THRESHOLD.
  - Embedding the query adds ~100-200ms even on a hit. Cache hits are much
    faster than the full ~14s pipeline but not free.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings


@dataclass
class CacheEntry:
    embedding: np.ndarray
    question: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class SemanticCache:
    """Embedding-similarity cache. Singleton-ish; one shared instance per
    process via `get_cache()`. Thread-safe writes via a single lock (the
    expected concurrency on this service is low; a single lock is fine).
    """

    def __init__(
        self,
        threshold: float = 0.97,
        max_size: int = 256,
        embed_model: str = "text-embedding-3-small",
    ):
        self.threshold = threshold
        self.max_size = max_size
        self._embed = OpenAIEmbeddings(model=embed_model)
        self._entries: list[CacheEntry] = []
        self._lock = Lock()
        # Observability: cheap counters for hit-rate visibility.
        self.hits = 0
        self.misses = 0

    def get(self, question: str) -> tuple[dict[str, Any] | None, float]:
        """Return (cached_payload, similarity) on hit, (None, best_sim) on miss.
        Returning the similarity on miss lets the caller log near-misses for
        threshold tuning."""
        if not self._entries:
            self.misses += 1
            return None, 0.0
        q_vec = np.array(self._embed.embed_query(question))
        q_norm = np.linalg.norm(q_vec) + 1e-12
        # Stack all entries' embeddings for vectorized similarity.
        with self._lock:
            mat = np.stack([e.embedding for e in self._entries])
            entries_snapshot = list(self._entries)
        mat_norms = np.linalg.norm(mat, axis=1) + 1e-12
        sims = (mat @ q_vec) / (mat_norms * q_norm)
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        if best >= self.threshold:
            self.hits += 1
            return entries_snapshot[idx].payload, best
        self.misses += 1
        return None, best

    def put(self, question: str, payload: dict[str, Any]) -> None:
        q_vec = np.array(self._embed.embed_query(question))
        with self._lock:
            self._entries.append(
                CacheEntry(embedding=q_vec, question=question, payload=payload)
            )
            # Naive FIFO eviction when over capacity. LRU would need access
            # tracking; FIFO is good enough for the demo-scope this is sized for.
            if len(self._entries) > self.max_size:
                self._entries = self._entries[-self.max_size :]

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "enabled": True,
            "threshold": self.threshold,
            "size": len(self._entries),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }


_cache: SemanticCache | None = None


def is_enabled() -> bool:
    """Read the env at call time, not import time, so tests can flip it."""
    return os.getenv("SEMANTIC_CACHE_ENABLED", "0").lower() in ("1", "true", "yes")


def get_cache() -> SemanticCache:
    """Singleton accessor. Built lazily on first call so importing this module
    does not eagerly construct an OpenAI client (which would force every test
    importing routes.py to need an OPENAI_API_KEY)."""
    global _cache
    if _cache is None:
        threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.97"))
        max_size = int(os.getenv("SEMANTIC_CACHE_MAX_SIZE", "256"))
        _cache = SemanticCache(threshold=threshold, max_size=max_size)
    return _cache
