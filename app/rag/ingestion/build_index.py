"""Chunk Middle-earth articles and build a Chroma index."""
import json
import pickle
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from app.rag.retrieval.vectorstore import CHUNKS_PATH

load_dotenv()

RAW_DIR = Path("data/raw")
CHROMA_DIR = "data/chroma_middle_earth"
COLLECTION = "middle_earth"
# Account is on the 1M TPM tier (measured). Our throughput (~15k tokens/batch)
# is far under that, so a small safety sleep is plenty. Matches build_pdr_index.
EMBED_BATCH = 100
BATCH_SLEEP_SEC = 1


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
    # ~600-1000 chars). To be A/B tested against parent-document retrieval.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Produced {len(chunks)} chunks")

    # Persist chunks for BM25 / sparse retrieval, kept in sync with Chroma.
    Path(CHUNKS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Persisted {len(chunks)} chunks to {CHUNKS_PATH}")
    print(
        f"Avg chunk length: "
        f"{sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars"
    )
    print(f"Sample chunk:\n{chunks[0].page_content[:300]}...\n")

    # Wipe-and-rebuild. Chroma(...) opens an EXISTING collection; subsequent
    # add_documents APPENDS, which on a re-run silently produces duplicate
    # chunks (every original chunk gets a second copy alongside any new ones).
    # Delete the persist_directory so the re-created Chroma starts empty.
    if Path(CHROMA_DIR).exists():
        print(f"Wiping existing {CHROMA_DIR} for clean rebuild...")
        shutil.rmtree(CHROMA_DIR)

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
