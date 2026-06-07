"""Turbovec retriever: 4-bit product-quantization-style vector index.

Uses the community turbovec library (Rust impl of Google Research's TurboQuant
algorithm) over the same chunks as the dense Chroma index. Isolates the
storage/index format as the only variable vs. the dense baseline. Not on the
routing path; available as kind="turbovec" for measurement.
"""
import pickle
from functools import lru_cache

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from turbovec import TurboQuantIndex

from app.rag.retrieval.vectorstore import CHUNKS_PATH

INDEX_PATH = "data/turbovec_index.tq"


@lru_cache(maxsize=1)
def _index() -> TurboQuantIndex:
    return TurboQuantIndex.load(INDEX_PATH)


@lru_cache(maxsize=1)
def _chunks() -> tuple[Document, ...]:
    with open(CHUNKS_PATH, "rb") as f:
        return tuple(pickle.load(f))


@lru_cache(maxsize=1)
def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


class TurbovecRetriever(BaseRetriever):
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # turbovec's search expects a 2D batched query array (n_queries, dim)
        # and returns 2D (n_queries, k) for both scores and indices. We send
        # one query, so unwrap [0].
        query_vec = np.array(
            [_embeddings().embed_query(query)], dtype=np.float32
        )
        _scores, indices = _index().search(query_vec, k=self.k)
        chunks = _chunks()
        return [chunks[i] for i in indices[0]]


def get_turbovec_retriever(k: int = 4) -> TurbovecRetriever:
    return TurbovecRetriever(k=k)
