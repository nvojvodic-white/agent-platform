"""Chunk Middle-earth articles and build a Chroma index."""
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()

RAW_DIR = Path("data/raw")
CHROMA_DIR = "data/chroma_middle_earth"
COLLECTION = "middle_earth"
# Throttled to stay under OpenAI free-tier 40k TPM ceiling for text-embedding-3-small.
# ~50 chunks/batch * ~600 chars ≈ 7.5k tokens, sleeping 12s -> ~37k TPM with headroom.
EMBED_BATCH = 50
BATCH_SLEEP_SEC = 12


def load_documents() -> list[Document]:
    docs = []
    for f in RAW_DIR.glob("*/*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        docs.append(
            Document(
                page_content=data["text"],
                metadata={
                    "title": data["title"],
                    "url": data["url"],
                    "source": data.get("source", f.parent.name),
                },
            )
        )
    return docs


def main() -> None:
    docs = load_documents()
    print(f"Loaded {len(docs)} articles")

    # 800/120 balances definitional coherence (entity definitions cluster in
    # 200-400 char spans) against event-narrative continuity (paragraphs run
    # ~600-1000 chars). To be A/B tested against parent-document retrieval on
    # day 9.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Produced {len(chunks)} chunks")
    print(
        f"Avg chunk length: "
        f"{sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars"
    )
    print(f"Sample chunk:\n{chunks[0].page_content[:300]}...\n")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    batches = list(range(0, len(chunks), EMBED_BATCH))
    for idx, i in enumerate(tqdm(batches, desc="Embedding")):
        vs.add_documents(chunks[i : i + EMBED_BATCH])
        if idx < len(batches) - 1:
            time.sleep(BATCH_SLEEP_SEC)

    print(f"\nIndexed into {CHROMA_DIR} (collection: {COLLECTION})")

    print("\n--- Sanity check: similarity search for 'Who is Gandalf?' ---")
    for i, r in enumerate(vs.similarity_search("Who is Gandalf?", k=3)):
        print(f"\n[{i+1}] {r.metadata['title']}")
        print(r.page_content[:200])


if __name__ == "__main__":
    main()
