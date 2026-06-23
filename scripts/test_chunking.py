#!/usr/bin/env python3
"""
Quick sanity check for chunking functions.
Run this after implementing a chunking strategy to verify output before ingestion.

Usage:
    python scripts/test_chunking.py
"""
import json
from pathlib import Path
from app.chunking import chunk_fixed_size

doc = json.loads(Path("data/processed/lotr/lotr_006.json").read_text())

chunks = chunk_fixed_size(
    text=doc["text"],
    chunk_size=300,
    overlap=50,
    doc_id=doc["id"],
    metadata={
        "source": doc["source"],
        "volume": doc["volume"],
        "book": doc["book"],
        "chapter_or_page": doc["chapter_or_page"],
        "title": doc["title"],
    },
)

print(f"Document : {doc['title']}")
print(f"Words    : {doc['word_count']:,}")
print(f"Chunks   : {len(chunks)}")
print(f"Avg words/chunk: {doc['word_count'] // len(chunks)}")
print()

print("=== Chunk 0 (first 200 chars) ===")
print(chunks[0].text[:200])
print()

print("=== Chunk 1 (first 200 chars) ===")
print(chunks[1].text[:200])
print()

overlap = 50
chunk_size = 300
print(f"=== Overlap check (last {overlap} words of chunk 0 vs first {overlap} words of chunk 1) ===")
end_of_0 = chunks[0].text.split()[-overlap:]
start_of_1 = chunks[1].text.split()[:overlap]
print(f"End of chunk 0   : {' '.join(end_of_0)}")
print()
print(f"Start of chunk 1 : {' '.join(start_of_1)}")
print()
print(f"Match: {end_of_0 == start_of_1}")
