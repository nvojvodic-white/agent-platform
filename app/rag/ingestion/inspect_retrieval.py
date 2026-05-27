"""Manual retrieval inspection — probe queries against the Chroma index.

This is for hour-4 sanity checking, not production. Run after build_index.py
to see what similarity_search returns for a fixed set of queries that probe
different failure modes (easy lookups, specific facts, multi-hop, book-only
characters, etc.).
"""
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.rag.ingestion.build_index import CHROMA_DIR, COLLECTION

load_dotenv()

PROBE_QUERIES = [
    "Who is Gandalf?",
    "Who killed Smaug?",
    "What rings did the Dwarves get?",
    "Tell me about the Battle of Five Armies",
    "Beren and Luthien",
    "What is mithril?",
    "Tom Bombadil",
]


def main() -> None:
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_DIR,
    )
    for q in PROBE_QUERIES:
        print(f"\n=== {q} ===")
        for r in vs.similarity_search(q, k=3):
            title = r.metadata.get("title", "?")
            src = r.metadata.get("source", "?")
            snippet = r.page_content[:140].replace("\n", " ")
            print(f"  [{src} :: {title}] {snippet}...")


if __name__ == "__main__":
    main()
