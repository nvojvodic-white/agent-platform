from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

CHROMA_DIR = "data/chroma_middle_earth"
COLLECTION = "middle_earth"


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_DIR,
    )


@lru_cache(maxsize=8)
def get_retriever(k: int = 4) -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
