"""Semantic-chunking retriever: dense search over the semantic-chunked index.

Same retrieval strategy as dense, but the underlying chunks were split at
embedding-distance topic boundaries (SemanticChunker) rather than fixed 800-char
windows. Isolates chunk granularity as the only variable vs. the dense baseline.
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

CHROMA_DIR = "data/chroma_semantic"
COLLECTION = "middle_earth_semantic"


@lru_cache(maxsize=1)
def _semantic_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_DIR,
    )


def get_semantic_retriever(k: int = 4) -> VectorStoreRetriever:
    return _semantic_store().as_retriever(search_kwargs={"k": k})
