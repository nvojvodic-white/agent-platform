"""Fetch Middle-earth articles via the MediaWiki API.

Pulls from two CC BY-SA sources:
  - Fandom LotR wiki  (lotr.fandom.com)  — uses action=parse + HTML strip
                                            because Fandom doesn't have the
                                            TextExtracts extension installed.
  - Wikipedia         (en.wikipedia.org) — uses prop=extracts (plain text).

Both paths dedupe by the canonical page title (post-redirect) so that
50 different redirect aliases for the same target page only produce one
saved file.

Tolkien Gateway (the canonical fan wiki) was originally planned but blocks
this network with HTTP 403 on every request.

Respectful scraping: 1 req/sec, identifies itself with contact info, caches
to disk so reruns skip already-downloaded pages.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

UA = (
    "agent-platform-rag-learning/0.1 "
    "(educational; n.vojvodic@icloud.com)"
)
OUT_DIR = Path("data/raw")
REQUEST_DELAY_SEC = 1.0
MIN_EXTRACT_CHARS = 200
CATEGORY_LIMIT = 500

# HTML elements to strip from Fandom rendered pages before extracting text.
# Removes infoboxes, navboxes, references, edit links, image thumbnails, etc.
FANDOM_STRIP_SELECTORS = [
    "table.infobox",
    "table.navbox",
    "div.references",
    ".reference",
    ".mw-editsection",
    ".reflist",
    "div.thumb",
    "figure",
    "aside",
    "style",
    "script",
]


@dataclass
class Source:
    name: str
    api_url: str
    categories: list[str]
    fetch_mode: str  # "extracts" (Wikipedia) or "parse" (Fandom)
    site_base: str   # for building canonical URLs


SOURCES = [
    Source(
        name="fandom_lotr",
        api_url="https://lotr.fandom.com/api.php",
        site_base="https://lotr.fandom.com/wiki/",
        fetch_mode="parse",
        # Only categories with real page counts (probed beforehand)
        categories=[
            "Category:Hobbits",      # 272
            "Category:Elves",        # 130
            "Category:Battles",      # 85
            "Category:Dwarves",      # 62
            "Category:Realms",       # 28
            "Category:Kings",        # 21
            "Category:Wars",         # 16
            "Category:Dragons",      # 13
            "Category:Wizards",      # 9
            "Category:Men",          # 7
        ],
    ),
    Source(
        name="wikipedia",
        api_url="https://en.wikipedia.org/w/api.php",
        site_base="https://en.wikipedia.org/wiki/",
        fetch_mode="extracts",
        categories=[
            "Category:Middle-earth_Elves",
            "Category:Middle-earth_Dwarves",
            "Category:Middle-earth_Men",
            "Category:Middle-earth_races",
            "Category:Middle-earth_locations",
            "Category:Middle-earth_objects",
        ],
    ),
    Source(
        # Day-16 add. Tolkien Gateway is the canonical fan wiki and was the
        # original Day-2 primary source, blocked by HTTP 403 on every request
        # at the time. The block has since lifted; the API works fine now.
        # Categories below come from a seed-page discovery pass (see commit
        # message); maintenance categories like "Pages with short description"
        # and name-language categories (Quenya/Sindarin/Gnomish/Noldorin names)
        # are deliberately excluded - they'd add duplicates with no new content.
        # The four "<Age> characters" categories are TG's specialty and cover
        # the Silmarillion First Age content Fandom basically lacks.
        name="tolkien_gateway",
        api_url="https://tolkiengateway.net/w/api.php",
        site_base="https://tolkiengateway.net/wiki/",
        fetch_mode="extracts",
        categories=[
            # Characters by book
            "Category:Characters in The Lord of the Rings",
            "Category:Characters in The Hobbit",
            "Category:Characters in The Silmarillion",
            "Category:Characters in The Adventures of Tom Bombadil",
            # Characters by age (TG's strength vs Fandom)
            "Category:First Age characters",
            "Category:Second Age characters",
            "Category:Third Age characters",
            "Category:Fourth Age characters",
            # Places
            "Category:Regions",
            "Category:Sindarin locations",
            "Category:Gondor",
            "Category:Arnor",
            "Category:Eriador",
            # Peoples / lineages
            "Category:Edain",
            "Category:Númenóreans",
            "Category:Dúnedain",
            "Category:House of Isildur",
            "Category:House of Bëor",
            # Conflicts (the categories Day 2 totally missed on Fandom too)
            "Category:Conflicts of the War of the Ring",
            "Category:Sieges",
            # Artifacts (canonical-wiki advantage: Silmarils, Andúril, etc.)
            "Category:Rings and jewels",
            "Category:Swords",
            "Category:Heirlooms",
            # Misc useful
            "Category:Enigmas",  # Bombadil et al.
            "Category:Spirits",
            "Category:Rulers of Gondor",
            "Category:Rulers of Arnor",
        ],
    ),
]


def _safe_filename(title: str) -> str:
    return title.replace("/", "_").replace(":", "_").replace("?", "_")


def get_category_members(
    session: requests.Session, api: str, category: str
) -> list[str]:
    """Page through a category and return all member titles."""
    titles: list[str] = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": CATEGORY_LIMIT,
            "format": "json",
            "cmtype": "page",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        r = session.get(api, params=params, timeout=30)
        r.raise_for_status()
        body = r.json()
        titles.extend(m["title"] for m in body["query"]["categorymembers"])
        if "continue" in body and "cmcontinue" in body["continue"]:
            cmcontinue = body["continue"]["cmcontinue"]
            time.sleep(REQUEST_DELAY_SEC)
        else:
            return titles


def fetch_via_extracts(
    session: requests.Session, source: Source, title: str
) -> dict | None:
    """Wikipedia path: TextExtracts API gives clean plain text directly."""
    params = {
        "action": "query",
        "prop": "extracts|info",
        "titles": title,
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "inprop": "url",
    }
    r = session.get(source.api_url, params=params, timeout=30)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page or not page["extract"].strip():
        return None
    return {
        "title": page["title"],  # canonical title after redirect resolution
        "url": page.get("fullurl", ""),
        "text": page["extract"],
    }


def fetch_via_parse(
    session: requests.Session, source: Source, title: str
) -> dict | None:
    """Fandom path: action=parse returns rendered HTML; strip with BeautifulSoup."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "redirects": 1,
        "disablelimitreport": 1,
        "disableeditsection": 1,
        "disabletoc": 1,
    }
    r = session.get(source.api_url, params=params, timeout=30)
    r.raise_for_status()
    body = r.json()
    if "parse" not in body:
        return None
    parse = body["parse"]
    canonical_title = parse["title"]
    html = parse["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")
    for sel in FANDOM_STRIP_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    text = soup.get_text(" ", strip=True)
    if not text:
        return None
    return {
        "title": canonical_title,
        "url": source.site_base + canonical_title.replace(" ", "_"),
        "text": text,
    }


FETCHERS = {
    "extracts": fetch_via_extracts,
    "parse": fetch_via_parse,
}


def fetch_source(source: Source) -> None:
    out_dir = OUT_DIR / source.name
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    fetcher = FETCHERS[source.fetch_mode]

    titles: set[str] = set()
    for cat in source.categories:
        print(f"[{source.name}] Fetching {cat}...")
        try:
            members = get_category_members(session, source.api_url, cat)
            titles.update(members)
            print(f"  -> {len(members)} pages (running total: {len(titles)})")
        except Exception as e:
            print(f"  failed: {e}")
        time.sleep(REQUEST_DELAY_SEC)

    print(f"[{source.name}] {len(titles)} unique requested titles\n")

    # Track which canonical titles we've already saved (handles redirect collapse).
    seen_canonical: set[str] = set()
    for f in out_dir.glob("*.json"):
        try:
            seen_canonical.add(json.loads(f.read_text(encoding="utf-8"))["title"])
        except Exception:
            pass

    fetched = skipped_dup = skipped_short = failed = 0
    try:
        for title in tqdm(sorted(titles), desc=f"[{source.name}] pages"):
            try:
                data = fetcher(session, source, title)
                if data is None:
                    skipped_short += 1
                    time.sleep(REQUEST_DELAY_SEC)
                    continue
                canonical = data["title"]
                if canonical in seen_canonical:
                    skipped_dup += 1
                    time.sleep(REQUEST_DELAY_SEC)
                    continue
                if len(data["text"]) <= MIN_EXTRACT_CHARS:
                    skipped_short += 1
                    time.sleep(REQUEST_DELAY_SEC)
                    continue
                data["source"] = source.name
                # Filename keyed off canonical title so reruns dedupe naturally.
                out_file = out_dir / f"{_safe_filename(canonical)}.json"
                out_file.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                seen_canonical.add(canonical)
                fetched += 1
                time.sleep(REQUEST_DELAY_SEC)
            except Exception as e:
                failed += 1
                print(f"Failed {title}: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(
        f"[{source.name}] done. "
        f"fetched={fetched} skipped_dup={skipped_dup} "
        f"skipped_short={skipped_short} failed={failed}"
    )


def main(only: str | None = None) -> None:
    """When `only` is set, fetch just that source by name (one of:
    fandom_lotr, wikipedia, tolkien_gateway). Default iterates everything.
    Use --source to avoid re-fetching sources whose data/raw/<name>/ already
    exists (the per-title skip in fetch_source checks AFTER the API call, so
    re-running ALL is wasteful even when files are cached on disk)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = [s for s in SOURCES if only is None or s.name == only]
    if not sources:
        names = ", ".join(s.name for s in SOURCES)
        raise SystemExit(f"unknown source {only!r}; valid: {names}")
    for source in sources:
        fetch_source(source)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        default=None,
        help="Fetch only this source. Default: all. "
        "Choices: fandom_lotr, wikipedia, tolkien_gateway.",
    )
    args = p.parse_args()
    main(only=args.source)
