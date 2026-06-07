"""Build a TurboQuant vector index over the same chunks as the dense Chroma index.

Uses turbovec (community Rust impl of Google Research's TurboQuant algorithm) at
4-bit quantization. Same chunks, same embedding model as the dense index, so the
A/B is purely on the storage/index format (Chroma's HNSW vs TurboQuant's
product-quantization-style compression). Not on the routing path; available as
kind="turbovec" for measurement.
"""
import pickle
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
from turbovec import TurboQuantIndex

from app.rag.retrieval.vectorstore import CHUNKS_PATH

load_dotenv()

INDEX_PATH = "data/turbovec_index.tq"
EMBED_DIM = 1536  # text-embedding-3-small
BIT_WIDTH = 4
EMBED_BATCH = 100
BATCH_SLEEP_SEC = 1


def main() -> None:
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    all_vecs: list[list[float]] = []
    batches = list(range(0, len(chunks), EMBED_BATCH))
    for idx, i in enumerate(tqdm(batches, desc="Embedding")):
        batch = chunks[i : i + EMBED_BATCH]
        vecs = embeddings.embed_documents([c.page_content for c in batch])
        all_vecs.extend(vecs)
        if idx < len(batches) - 1:
            time.sleep(BATCH_SLEEP_SEC)

    arr = np.array(all_vecs, dtype=np.float32)
    print(f"Embedded {arr.shape[0]} vectors of dim {arr.shape[1]}")

    # Wipe-and-rebuild: turbovec writes a single file; building fresh from
    # in-memory vectors guarantees no stale entries from prior runs.
    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    if Path(INDEX_PATH).exists():
        Path(INDEX_PATH).unlink()

    index = TurboQuantIndex(dim=EMBED_DIM, bit_width=BIT_WIDTH)
    index.add(arr)
    index.write(INDEX_PATH)
    size_mib = Path(INDEX_PATH).stat().st_size / 1024 / 1024
    print(f"Wrote turbovec index to {INDEX_PATH} ({size_mib:.1f} MiB)")

    q = embeddings.embed_query("Who is Gandalf?")
    scores, indices = index.search(np.array([q], dtype=np.float32), k=3)
    print("\n--- Sanity: 'Who is Gandalf?' ---")
    for rank, (score, i) in enumerate(zip(scores[0], indices[0]), 1):
        print(f"[{rank}] score={score:.3f} title={chunks[i].metadata.get('title')}")
        print(f"    {chunks[i].page_content[:120]}...")


if __name__ == "__main__":
    main()
