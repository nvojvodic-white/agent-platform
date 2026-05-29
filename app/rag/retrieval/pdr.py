"""Parent-document retrieval: search small child chunks, return their parents.

Children (~400 chars) are tight for embedding precision; parents (~2000 chars)
give the LLM more synthesis context. At query time: similarity-search the child
collection, dedupe by parent_id preserving rank order, fetch the parent docs,
return the top-k parents. k means k *parents* (comparable context budget to
dense's k chunks, just larger units).

Reads the index built by app.rag.ingestion.build_pdr_index (separate Chroma
collection 'middle_earth_pdr' + a parent pickle), leaving the production
'middle_earth' collection untouched.
"""
import pickle
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings

CHROMA_DIR = "data/chroma_pdr"
CHILDREN_COLLECTION = "middle_earth_pdr"
PARENT_DOCSTORE_PATH = "data/pdr_parents.pkl"
# Fetch more children than k so dedupe-by-parent can still yield k parents.
CHILD_FETCH_MULTIPLIER = 6


@lru_cache(maxsize=1)
def _child_store() -> Chroma:
    return Chroma(
        collection_name=CHILDREN_COLLECTION,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_DIR,
    )


@lru_cache(maxsize=1)
def _parent_store() -> dict[str, Document]:
    with open(PARENT_DOCSTORE_PATH, "rb") as f:
        return pickle.load(f)


class ParentDocumentRetriever(BaseRetriever):
    """Search children, return deduped top-k parents."""

    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        children = _child_store().similarity_search(
            query, k=self.k * CHILD_FETCH_MULTIPLIER
        )
        parents = _parent_store()
        seen: set[str] = set()
        out: list[Document] = []
        for child in children:
            pid = child.metadata.get("parent_id")
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            parent = parents.get(pid)
            if parent is not None:
                out.append(parent)
            if len(out) >= self.k:
                break
        return out


def get_pdr_retriever(k: int = 4) -> ParentDocumentRetriever:
    return ParentDocumentRetriever(k=k)
