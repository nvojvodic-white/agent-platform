"""Targeted fetch of the Wikipedia 'Tom Bombadil' article + index delta.

The retriever comparison found the Bombadil miss was a corpus gap, not a dense-vs-sparse
problem: the corpus had no standalone Bombadil article (only the poetry
collection 'The Adventures of Tom Bombadil'). This closes that gap.

Fetches the article via the existing extracts path, saves it under
data/raw/wikipedia/, then adds ONLY its chunks to the existing Chroma collection
and appends them to the chunks pickle (no full re-embed of the 5763 existing
chunks). Re-run app.rag.eval.compare_retrievers afterwards to retest the
original hypothesis on a fair corpus.
"""
import json
import pickle
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.ingestion.fetch import UA
from app.rag.retrieval.vectorstore import CHROMA_DIR, CHUNKS_PATH, COLLECTION

load_dotenv()

TITLE = "Tom Bombadil"
WIKI_API = "https://en.wikipedia.org/w/api.php"
OUT_FILE = Path("data/raw/wikipedia/Tom Bombadil.json")


def fetch_article() -> dict:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "titles": TITLE,
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "inprop": "url",
    }
    r = requests.get(WIKI_API, params=params, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    if "extract" not in page or not page["extract"].strip():
        raise SystemExit("No extract returned for Tom Bombadil")
    return {
        "title": page["title"],
        "url": page.get("fullurl", ""),
        "text": page["extract"],
        "source": "wikipedia",
    }


def main() -> None:
    from langchain_chroma import Chroma

    data = fetch_article()
    OUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {OUT_FILE} ({len(data['text'])} chars, title={data['title']!r})")

    doc = Document(
        page_content=data["text"],
        metadata={"title": data["title"], "url": data["url"], "source": data["source"]},
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    new_chunks = splitter.split_documents([doc])
    print(f"Split into {len(new_chunks)} chunks")

    # Add only the new chunks to Chroma (existing 5763 untouched).
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_DIR,
    )
    before = vs._collection.count()
    vs.add_documents(new_chunks)
    after = vs._collection.count()
    print(f"Chroma count {before} -> {after}")

    # Append to the chunks pickle (BM25 source).
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    chunks.extend(new_chunks)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks pickle now holds {len(chunks)} chunks")


if __name__ == "__main__":
    main()
