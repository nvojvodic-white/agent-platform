"""Build a semantic-chunking index (Day 2's deferred chunking variant).

Instead of fixed 800-char windows, SemanticChunker splits where the embedding
distance between consecutive sentences spikes - i.e. at topic boundaries. The
hypothesis: topically-coherent chunks retrieve more cleanly and carry complete
thoughts, which could lift recall on probes whose facts were split mid-topic by
the recursive splitter.

Separate Chroma collection 'middle_earth_semantic' so it can be A/B'd against
dense (recursive 800/120) and PDR without disturbing them.
"""
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

load_dotenv()

RAW_DIR = Path("data/raw")
CHROMA_DIR = "data/chroma_semantic"
COLLECTION = "middle_earth_semantic"
EMBED_BATCH = 100
BATCH_SLEEP_SEC = 1


def load_articles() -> list[Document]:
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
    articles = load_articles()
    print(f"Loaded {len(articles)} articles")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # SemanticChunker embeds sentences to find breakpoints (uses the embeddings
    # model directly). percentile breakpoint = default, conservative.
    chunker = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile"
    )

    print("Semantic-chunking articles (embeds sentences to find breakpoints)...")
    chunks = []
    for article in tqdm(articles, desc="Chunking"):
        for c in chunker.split_documents([article]):
            chunks.append(c)
    print(
        f"Produced {len(chunks)} semantic chunks "
        f"(avg {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars)"
    )

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
    print(f"Indexed {len(chunks)} chunks into '{COLLECTION}'")

    hits = vs.similarity_search("What is mithril?", k=4)
    print("\nSanity check 'What is mithril?':")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h.metadata['title']}: {h.page_content[:70]}...")


if __name__ == "__main__":
    main()
