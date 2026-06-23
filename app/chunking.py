from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    doc_id: str        # id of the source document (e.g. "lotr_006")
    chunk_index: int   # position of this chunk within the document (0-based)
    metadata: dict = field(default_factory=dict)
    # metadata carries everything needed for the `sources` field in API responses:
    # source, volume, book, chapter_or_page, title, etc.


def chunk_fixed_size(text: str, chunk_size: int, overlap: int, doc_id: str, metadata: dict) -> list[Chunk]:
    """
    Split text into overlapping fixed-size chunks measured in words.

    Args:
        text:       raw text of the document
        chunk_size: number of words per chunk (try 200-400 to start)
        overlap:    number of words to repeat at the start of the next chunk
                    (try 10-20% of chunk_size — e.g. overlap=50 for chunk_size=300)
        doc_id:     source document id, passed through to each Chunk
        metadata:   source metadata (volume, book, chapter_or_page, etc.), passed through

    Returns:
        list of Chunk objects in document order

    Why overlap?
        A sentence that falls on a chunk boundary gets split across two chunks.
        Without overlap, neither chunk contains the full sentence — retrieval
        may miss it entirely. Overlap repeats a window of words so boundary
        content appears intact in at least one chunk.

    Implementation hint:
        1. Split text into a list of words
        2. Use a sliding window: start at index 0, advance by (chunk_size - overlap) each step
        3. Each window is words[start : start + chunk_size], joined back to a string
        4. Stop when start >= len(words)

    TODO: implement this
    """
    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for i, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size]
        chunks.append(Chunk(
            text=" ".join(chunk_words),
            doc_id=doc_id,
            chunk_index=i,
            metadata=metadata,
        ))

    return chunks


def chunk_by_paragraph(text: str, min_words: int, max_words: int, doc_id: str, metadata: dict) -> list[Chunk]:
    """
    Split text at paragraph boundaries (double newlines).

    Args:
        text:      raw text of the document
        min_words: paragraphs shorter than this are merged with the next one
                   (avoids tiny chunks from section headers or single-line paragraphs)
        max_words: paragraphs longer than this are split using fixed-size logic
                   (avoids oversized chunks from dense prose)
        doc_id:    source document id, passed through to each Chunk
        metadata:  source metadata, passed through

    Returns:
        list of Chunk objects in document order

    Why paragraph-aware?
        Fixed-size chunking ignores sentence and paragraph boundaries — a chunk
        might start mid-sentence and end mid-thought. Paragraphs are the natural
        unit of meaning in prose: Tolkien writes one idea per paragraph more often
        than not. Splitting here produces chunks that are more semantically coherent.
    """
    # Try double newline first (epub-extracted text), fall back to single newline (wiki).
    delimiter = "\n\n" if "\n\n" in text else "\n"
    paragraphs = [p.strip() for p in text.split(delimiter) if p.strip()]

    chunks = []
    buffer: list[str] = []
    buffer_para_start = 0
    chunk_index = 0

    def emit(words: list[str], para_start: int, para_end: int) -> None:
        nonlocal chunk_index
        chunks.append(Chunk(
            text=" ".join(words),
            doc_id=doc_id,
            chunk_index=chunk_index,
            metadata={**metadata, "para_start": para_start, "para_end": para_end},
        ))
        chunk_index += 1

    for para_idx, para in enumerate(paragraphs):
        words = para.split()

        if len(words) > max_words:
            if buffer:
                emit(buffer, buffer_para_start, para_idx - 1)
                buffer = []
            for sub in chunk_fixed_size(para, chunk_size=max_words, overlap=50, doc_id=doc_id, metadata=metadata):
                sub.chunk_index = chunk_index
                sub.metadata = {**sub.metadata, "para_start": para_idx, "para_end": para_idx}
                chunks.append(sub)
                chunk_index += 1
        else:
            if not buffer:
                buffer_para_start = para_idx
            buffer.extend(words)
            if len(buffer) >= min_words:
                emit(buffer, buffer_para_start, para_idx)
                buffer = []

    if buffer:
        emit(buffer, buffer_para_start, len(paragraphs) - 1)

    return chunks
