"""Export chunks already stored in Chroma to a pickle for BM25 / sparse retrieval.

Chroma holds the embedded chunks; BM25 needs the same chunks as plain Document
objects. Rather than re-chunk and re-embed (which costs OpenAI calls), pull the
stored documents + metadata back out of Chroma via .get() and reconstitute
Documents. The pickle is the shared chunk source for the sparse retriever
(Day 6) and parent-document retrieval (Day 9).
"""
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

from app.rag.retrieval.vectorstore import CHUNKS_PATH, get_vectorstore

load_dotenv()


def main() -> None:
    vs = get_vectorstore()
    data = vs.get(include=["documents", "metadatas"])
    texts = data["documents"]
    metas = data["metadatas"]
    chunks = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(texts, metas)
    ]
    Path(CHUNKS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Exported {len(chunks)} chunks from Chroma to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
