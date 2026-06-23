#!/usr/bin/env python3
"""
Fetch Middle-earth Wikipedia pages via the MediaWiki API.

Run this BEFORE chunking to build the wiki corpus.
Output: data/processed/wiki/*.json (one file per page)
        + stats printed to stdout

Usage:
    python scripts/fetch_wikipedia.py --inspect-only
    python scripts/fetch_wikipedia.py
"""
import argparse
import json
import time
from pathlib import Path

import wikipediaapi


OUTPUT_DIR = Path("data/processed/wiki")

# Starter set — expand deliberately.
# Think about: major characters only vs. all places vs. factions,
# page quality (stubs vs. full articles), overlap with LOTR corpus.
# Alternatively, fetch an entire category:
#   wiki.page("Category:Middle-earth characters").categorymembers
PAGES = [
    "Frodo Baggins",
    "Samwise Gamgee",
    "Gandalf",
    "Aragorn",
    "Legolas",
    "Gimli (Middle-earth)",
    "Saruman",
    "Sauron",
    "The One Ring",
    "Shire",
    "Mordor",
    "Rivendell",
    "Rohan",
    "Gondor",
    "Minas Tirith",
    "Mount Doom",
    "Balrog",
    "Ents",
    # TODO (you): add more — aim for ~50-100 pages for a meaningful corpus
]

# Pages below this threshold are usually stubs or disambiguation pages
MIN_WORDS = 200


def word_count(text: str) -> int:
    return len(text.split())


def print_stats(docs: list[dict]) -> None:
    counts = sorted(d["word_count"] for d in docs)
    total = sum(counts)
    median = counts[len(counts) // 2]
    print(f"\n{'='*40}")
    print(f"Documents : {len(docs)}")
    print(f"Total words: {total:,}")
    print(f"Min / Median / Max: {counts[0]:,} / {median:,} / {counts[-1]:,}")
    print(f"\nAll pages (title | words):")
    for d in docs:
        print(f"  {d['title']:<40}  |  {d['word_count']:>6,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print structure without writing output files",
    )
    args = parser.parse_args()

    # MediaWiki policy requires a descriptive user agent
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="lotr-rag-learning/0.1 (github.com/samuxyz/lotr-rag)",
    )

    docs = []
    skipped = []

    for title in PAGES:
        page = wiki.page(title)

        if not page.exists():
            print(f"  [skip] '{title}' — not found")
            skipped.append(title)
            continue

        wc = word_count(page.text)
        if wc < MIN_WORDS:
            print(f"  [skip] '{title}' — too short ({wc} words)")
            skipped.append(title)
            continue

        docs.append({
            "id": f"wiki_{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}",
            "title": page.title,
            "text": page.text,
            "word_count": wc,
            "source": "wiki",
            "chapter_or_page": page.title,
            "url": page.fullurl,
            # Keeping only first 10 categories — the full list is noisy
            "categories": list(page.categories.keys())[:10],
        })
        print(f"  [ok]   '{title}' ({wc:,} words)")

        time.sleep(0.5)  # be polite to the MediaWiki API

    print_stats(docs)

    if skipped:
        print(f"\nSkipped ({len(skipped)}): {skipped}")

    if args.inspect_only:
        print("\n--inspect-only: no files written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        out_path = OUTPUT_DIR / f"{doc['id']}.json"
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    print(f"\nWrote {len(docs)} files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
