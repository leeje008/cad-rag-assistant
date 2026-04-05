"""Heading + length aware markdown chunker.

Walks `ParsedBlock`s (already page/section tagged), accumulating lines
until either a markdown heading or the `max_chars` budget is hit. Produces
chunks with overlap for downstream embedding. Intentionally simple — a
full hierarchical / AutoMergingRetriever approach is planned for Phase 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .parsers import ParsedBlock, ParsedDoc

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    text: str
    page: int | None
    section: str | None
    char_start: int
    char_end: int


def chunk_parsed(
    doc: ParsedDoc,
    *,
    max_chars: int = 1500,
    min_chars: int = 400,
    overlap: int = 150,
) -> list[Chunk]:
    """Split a ParsedDoc into retrieval-sized chunks."""

    chunks: list[Chunk] = []
    counter = 0
    cursor = 0  # global character cursor across the whole document

    buffer_text: list[str] = []
    buffer_len = 0
    buffer_start = 0
    buffer_page: int | None = None
    buffer_section: str | None = None

    def flush() -> None:
        nonlocal counter, buffer_text, buffer_len, buffer_start
        nonlocal buffer_page, buffer_section

        if not buffer_text:
            return
        text = "\n".join(buffer_text).strip()
        if not text:
            buffer_text = []
            buffer_len = 0
            return

        end = buffer_start + len(text)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{counter}",
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                title=doc.title,
                text=text,
                page=buffer_page,
                section=buffer_section,
                char_start=buffer_start,
                char_end=end,
            )
        )
        counter += 1

        # Seed the next buffer with the tail overlap.
        if overlap > 0 and len(text) > overlap:
            tail = text[-overlap:]
            buffer_text = [tail]
            buffer_len = len(tail)
            buffer_start = end - overlap
        else:
            buffer_text = []
            buffer_len = 0
            buffer_start = end

    for block in doc.blocks:
        block_section = block.section
        block_page = block.page

        # Advance the global cursor to the start of this block.
        if buffer_len == 0:
            buffer_start = cursor
            buffer_page = block_page
            buffer_section = block_section

        for line in block.text.splitlines():
            stripped = line.strip()
            is_heading = stripped.startswith("#")

            if is_heading and buffer_len >= min_chars:
                flush()
                buffer_page = block_page
                buffer_section = stripped.lstrip("#").strip() or block_section

            if is_heading:
                buffer_section = stripped.lstrip("#").strip() or buffer_section

            buffer_text.append(line)
            buffer_len += len(line) + 1

            if buffer_len >= max_chars:
                flush()
                buffer_page = block_page
                buffer_section = block_section

        cursor += len(block.text) + 1

    if buffer_len >= min_chars or (buffer_len > 0 and not chunks):
        flush()

    logger.info("chunked %s into %d chunks", doc.title, len(chunks))
    return chunks
