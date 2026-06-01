"""Build a parent-document retrieval (PDR) index.

Separate from the production 'middle_earth' collection so PDR can be A/B'd
without disturbing dense/hyde/etc.

Parents: ~2000-char recursive chunks (NOT whole articles - 17% of articles
exceed 8000 chars and truncating them would drop late-article content, which is
where the Mithril/Bombadil recall gaps may live). No article is ever truncated.
Children: ~400-char chunks embedded into Chroma collection 'middle_earth_pdr'.
Parents stored in a pickle keyed by uuid (loaded into an InMemoryStore at
retrieval time).

Throttled embedding to stay under the OpenAI free-tier 40k TPM ceiling, same as
build_index.py.
"""
import json
import pickle
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()

RAW_DIR = Path("data/raw")
PARENT_DOCSTORE_PATH = "data/pdr_parents.pkl"
CHROMA_DIR = "data/chroma_pdr"
CHILDREN_COLLECTION = "middle_earth_pdr"

PARENT_SIZE = 2000
PARENT_OVERLAP = 200
CHILD_SIZE = 400
CHILD_OVERLAP = 50
EMBED_BATCH = 100
# Account is on the 1M TPM tier (probed); our throughput (~15k tokens/batch) is
# far under that, so a small safety sleep is all that's needed.
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

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_SIZE,
        chunk_overlap=PARENT_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_SIZE,
        chunk_overlap=CHILD_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_store: dict[str, Document] = {}
    children: list[Document] = []
    for article in articles:
        for parent in parent_splitter.split_documents([article]):
            pid = str(uuid.uuid4())
            parent.metadata["parent_id"] = pid
            parent_store[pid] = parent
            for child in child_splitter.split_documents([parent]):
                child.metadata["parent_id"] = pid
                child.metadata["title"] = parent.metadata["title"]
                child.metadata["source"] = parent.metadata["source"]
                children.append(child)

    print(f"Produced {len(parent_store)} parents, {len(children)} children "
          f"(avg child {sum(len(c.page_content) for c in children) / len(children):.0f} chars)")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(
        collection_name=CHILDREN_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    batches = list(range(0, len(children), EMBED_BATCH))
    for idx, i in enumerate(tqdm(batches, desc="Embedding children")):
        vs.add_documents(children[i : i + EMBED_BATCH])
        if idx < len(batches) - 1:
            time.sleep(BATCH_SLEEP_SEC)
    print(f"Indexed {len(children)} children into '{CHILDREN_COLLECTION}'")

    Path(PARENT_DOCSTORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PARENT_DOCSTORE_PATH, "wb") as f:
        pickle.dump(parent_store, f)
    print(f"Persisted {len(parent_store)} parents to {PARENT_DOCSTORE_PATH}")

    hits = vs.similarity_search("Who is Tom Bombadil?", k=4)
    print("\nSanity check 'Who is Tom Bombadil?':")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h.metadata['title']}: {h.page_content[:70]}...")


if __name__ == "__main__":
    main()
