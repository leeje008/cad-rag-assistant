"""Hierarchical (parent-child) section-aware chunker.

Pass 1 groups `ParsedBlock`s into sections (splitting text blocks at
markdown headings; table/image blocks stay whole). Pass 2 emits, per
section, one `parent` chunk (level 0, full-section text) plus its
`child` chunks (level 1): text blocks are length+overlap split, while
table/image blocks become a single typed child carrying table_id /
figure_id. Children are what retrieval ranks; the parent is fetched at
expansion time to give the LLM the surrounding section.
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
    chunk_type: str = "text"  # "text"|"table"|"table_summary"|"image"|"parent"
    parent_id: str | None = None  # chunk_id of the section parent
    table_id: str | None = None
    figure_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    level: int = 1  # 0 = parent/section, 1 = child
    table_html: str | None = None
    image_b64: str | None = None  # transient: VLM input, never written to LanceDB


def _split_length(text: str, max_chars: int, min_chars: int, overlap: int) -> list[str]:
    """Sliding-window split of a single text into max_chars pieces with overlap."""

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    n = len(text)
    step = max_chars - overlap if overlap < max_chars else max_chars
    while start < n:
        end = min(start + max_chars, n)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start += step

    # Fold a too-small tail into the previous piece — but only when the
    # merge stays within max_chars, otherwise keep the small tail as-is
    # rather than producing an oversized chunk.
    if len(pieces) > 1 and len(pieces[-1]) < min_chars:
        merged = f"{pieces[-2]}\n{pieces[-1]}"
        if len(merged) <= max_chars:
            pieces[-2] = merged
            pieces.pop()
    return pieces


def _group_sections(blocks: list[ParsedBlock]) -> list[dict]:
    """Group blocks into sections, splitting text blocks at heading lines.

    Each section is {"section": str|None, "page": int|None, "blocks": [ParsedBlock]}.
    Table/image blocks are kept whole; text is re-emitted as text sub-blocks.
    """

    sections: list[dict] = []

    def ensure(name: str | None, page: int | None) -> dict:
        if not sections or sections[-1]["section"] != name:
            sections.append({"section": name, "page": page, "blocks": []})
        return sections[-1]

    for block in blocks:
        if block.block_type in ("table", "image"):
            ensure(block.section, block.page)["blocks"].append(block)
            continue

        cur = block.section
        buf: list[str] = []

        def emit(section: str | None) -> None:
            text = "\n".join(buf).strip()
            if text:
                ensure(section, block.page)["blocks"].append(
                    ParsedBlock(text=text, page=block.page, section=section)
                )

        for line in block.text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                emit(cur)
                buf = []
                cur = stripped.lstrip("#").strip() or cur
            buf.append(line)
        emit(cur)

    return sections


def chunk_parsed(
    doc: ParsedDoc,
    *,
    max_chars: int = 1500,
    min_chars: int = 400,
    overlap: int = 150,
) -> list[Chunk]:
    """Split a ParsedDoc into hierarchical parent/child retrieval chunks."""

    chunks: list[Chunk] = []
    child_counter = 0
    cursor = 0  # global character cursor across the whole document

    def make_child(text: str, *, page, section, chunk_type, parent_id,
                   table_id=None, figure_id=None, bbox=None, table_html=None,
                   image_b64=None) -> Chunk:
        nonlocal child_counter, cursor
        start = cursor
        end = start + len(text)
        cursor = end + 1
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}:{child_counter}",
            doc_id=doc.doc_id,
            source_path=doc.source_path,
            title=doc.title,
            text=text,
            page=page,
            section=section,
            char_start=start,
            char_end=end,
            chunk_type=chunk_type,
            parent_id=parent_id,
            table_id=table_id,
            figure_id=figure_id,
            bbox=bbox,
            level=1,
            table_html=table_html,
            image_b64=image_b64,
        )
        child_counter += 1
        return chunk

    for si, sec in enumerate(_group_sections(doc.blocks)):
        parent_id = f"{doc.doc_id}:p{si}"
        section = sec["section"]
        children: list[Chunk] = []

        for block in sec["blocks"]:
            if block.block_type == "table":
                text = block.text.strip()
                if not text:
                    continue
                children.append(
                    make_child(
                        text, page=block.page, section=section,
                        chunk_type="table", parent_id=parent_id,
                        table_id=block.table_id, bbox=block.bbox,
                        table_html=block.table_html,
                    )
                )
            elif block.block_type == "image":
                # Placeholder text; the VLM captioner (PR6) overwrites this.
                label = f"[이미지] {doc.title}"
                if section:
                    label += f" · {section}"
                children.append(
                    make_child(
                        label, page=block.page, section=section,
                        chunk_type="image", parent_id=parent_id,
                        figure_id=block.figure_id, bbox=block.bbox,
                        image_b64=block.image_b64,
                    )
                )
            else:
                for piece in _split_length(block.text, max_chars, min_chars, overlap):
                    children.append(
                        make_child(
                            piece, page=block.page, section=section,
                            chunk_type="text", parent_id=parent_id,
                        )
                    )

        if not children:
            continue

        parent_text = "\n\n".join(c.text for c in children)[: max_chars * 4]
        chunks.append(
            Chunk(
                chunk_id=parent_id,
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                title=doc.title,
                text=parent_text,
                page=sec["page"],
                section=section,
                char_start=children[0].char_start,
                char_end=children[-1].char_end,
                chunk_type="parent",
                parent_id=None,
                level=0,
            )
        )
        chunks.extend(children)

    n_parents = sum(1 for c in chunks if c.chunk_type == "parent")
    logger.info(
        "chunked %s into %d chunks (%d parents, %d children)",
        doc.title,
        len(chunks),
        n_parents,
        len(chunks) - n_parents,
    )
    return chunks
