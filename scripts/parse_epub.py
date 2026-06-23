#!/usr/bin/env python3
"""
Inspect and extract text from the LOTR epub.

Run this BEFORE chunking to understand the corpus structure.
Output: data/processed/lotr/*.json (one file per spine item)
        + stats printed to stdout

Usage:
    python scripts/parse_epub.py path/to/lotr.epub --inspect-only
    python scripts/parse_epub.py path/to/lotr.epub
"""
import argparse
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from bs4 import BeautifulSoup


OUTPUT_DIR = Path("data/processed/lotr")


def extract_text(html_content: bytes) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Extract each block element as its own paragraph so chunk_by_paragraph
    # can split on real prose boundaries rather than word-count alone.
    paragraphs = [
        elem.get_text(separator=" ", strip=True)
        for elem in soup.find_all(["p", "h1", "h2", "h3", "h4", "blockquote"])
        if elem.get_text(strip=True)
    ]
    return "\n\n".join(paragraphs)


def extract_title(html_content: bytes) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    heading = soup.find("h1", class_="ct")
    return heading.get_text(strip=True) if heading else ""

def extract_chapter_num(html_content: bytes) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    heading = soup.find("h1", class_="cn")
    return heading.get_text(strip=True) if heading else ""

def extract_book(html_content: bytes) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    heading = soup.find("p", class_="pt")
    return heading.get_text(strip=True) if heading else ""


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
    print(f"\n{'id':<12} {'volume':<35} {'book':<12} {'chapter':<10} {'title':<40} {'words':>6}")
    print("-" * 120)
    for d in docs:
        print(
            f"  {d['id']:<10} "
            f"{d.get('volume', ''):<35} "
            f"{d.get('book', ''):<12} "
            f"{d.get('chapter_num', ''):<10} "
            f"{d.get('title', '')[:38]:<40} "
            f"{d['word_count']:>6,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("epub_path", help="Path to the LOTR epub file")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print structure without writing output files",
    )
    args = parser.parse_args()

    epub_path = Path(args.epub_path)
    if not epub_path.exists():
        print(f"Error: {epub_path} not found", file=sys.stderr)
        sys.exit(1)

    # An epub is just a ZIP. We parse it manually because ebooklib has
    # bugs with epubs whose manifests reference missing files.
    docs = []
    with zipfile.ZipFile(epub_path) as zf:
        # container.xml tells us where the OPF (package) file lives
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        ns_c = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
        opf_path = container.find(".//c:rootfile", ns_c).get("full-path")
        opf_dir = str(Path(opf_path).parent)

        # OPF has two things we need: manifest (id->href) and spine (reading order)
        opf = ET.fromstring(zf.read(opf_path))
        ns_opf = {"opf": "http://www.idpf.org/2007/opf"}

        manifest = {
            item.get("id"): item.get("href")
            for item in opf.findall(".//opf:manifest/opf:item", ns_opf)
        }

        current_volume = ""
        current_book = ""
        i = 0
        for itemref in opf.findall(".//opf:spine/opf:itemref", ns_opf):
            href = manifest.get(itemref.get("idref"))
            if not href:
                continue

            full_path = str(Path(opf_dir) / href) if opf_dir != "." else href
            try:
                content = zf.read(full_path)
            except KeyError:
                # Manifest references a file that doesn't exist — skip it
                continue

            if "_part-" in href:
                pt = extract_book(content)
                if "BOOK" in pt:
                    current_book = pt.title()
                else:
                    current_volume = pt.title()
                continue

            text = extract_text(content)
            if len(text.strip()) < 100:
                continue

            if "_chap-" in href:
                chapter_title = extract_title(content)
                chapter_num = extract_chapter_num(content)
                doc = {
                    "id": f"lotr_{i:03d}",
                    "volume": current_volume,
                    "book": current_book,
                    "chapter_num": chapter_num,
                    "title": chapter_title,
                    "text": text,
                    "word_count": word_count(text),
                    "source": "lotr",
                    "chapter_or_page": f"{current_volume}, {current_book}, {chapter_num}",
                }
            elif "_prol-" in href:
                doc = {
                    "id": f"lotr_{i:03d}",
                    "volume": "Ancillary",
                    "book": "",
                    "chapter_num": "",
                    "title": "Prologue",
                    "text": text,
                    "word_count": word_count(text),
                    "source": "lotr",
                    "chapter_or_page": "Prologue",
                }
            elif "_appe-" in href:
                # Try to get appendix title from HTML; appendices use different heading styles
                soup = BeautifulSoup(content, "html.parser")
                heading = soup.find("h1") or soup.find("h2")
                title = heading.get_text(strip=True) if heading else Path(href).stem
                doc = {
                    "id": f"lotr_{i:03d}",
                    "volume": "Ancillary",
                    "book": "",
                    "chapter_num": "",
                    "title": title,
                    "text": text,
                    "word_count": word_count(text),
                    "source": "lotr",
                    "chapter_or_page": title,
                }
            else:
                # Skip all other front/back matter (TOC, foreword, index, copyright, etc.)
                continue

            docs.append(doc)
            i += 1

    print_stats(docs)

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
