"""Shared ingest pipeline: ParsedDoc → enriched chunks + embedding vectors.

Both the live ingest route (`ingest.run_ingest`) and the offline rebuild
(`scripts/rebuild_index.py`) call `build_chunks_and_vectors` so the chunk →
enrich → embed sequence stays identical and never drifts.

Enrichment stages (each gated by a setting, all one-off batch cost):
- PR4 table summaries: a `table_summary` sibling per table chunk.
- PR6 image captions: VLM caption overwrites the image chunk placeholder.
- PR5 contextual prepend: a one-line section context is prepended to the
  text that gets embedded, while the stored chunk text stays clean.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import replace

import httpx

from .captioner import caption_image, placeholder_caption
from .chunker import Chunk, chunk_parsed
from .contextualizer import section_context
from .embedder import embed_texts
from .parsers import ParsedDoc
from .settings import settings
from .table_summary import summarize_table

logger = logging.getLogger(__name__)


async def _add_table_summaries(
    client: httpx.AsyncClient, chunks: list[Chunk], doc: ParsedDoc
) -> list[Chunk]:
    """Append a `table_summary` sibling chunk for each table chunk."""

    extra: list[Chunk] = []
    for c in chunks:
        if c.chunk_type != "table":
            continue
        summary = await summarize_table(
            client, c.text, doc_title=doc.title, section=c.section
        )
        if not summary:
            continue
        extra.append(
            replace(
                c,
                chunk_id=f"{c.chunk_id}:sum",
                text=summary,
                chunk_type="table_summary",
                table_html=None,  # summary doesn't carry the original markup
            )
        )
    if extra:
        logger.info("added %d table summaries for %s", len(extra), doc.title)
    return chunks + extra


async def _build_embed_inputs(
    client: httpx.AsyncClient, chunks: list[Chunk], doc: ParsedDoc
) -> list[str]:
    """Per-chunk embedding text, with a one-line section context prepended.

    Context is generated once per section (parent) and reused across its
    children. Returns the chunk's own text unchanged when contextual mode is
    off or no parent context is available. Parents are embedded without a
    prefix (they are never ranked, only fetched for expansion).
    """

    if not settings.use_contextual:
        return [c.text for c in chunks]

    parents = [c for c in chunks if c.chunk_type == "parent"]
    ctx_by_parent: dict[str, str] = {}
    for p in parents:
        ctx_by_parent[p.chunk_id] = await section_context(
            client, doc_title=doc.title, section=p.section, parent_text=p.text
        )

    inputs: list[str] = []
    for c in chunks:
        ctx = ctx_by_parent.get(c.parent_id or "")
        if ctx and c.chunk_type != "parent":
            inputs.append(f"{ctx}\n{c.text}")
        else:
            inputs.append(c.text)
    return inputs


def _persist_figure_images(chunks: list[Chunk]) -> None:
    """Write each image chunk's transient PNG under settings.assets_dir.

    Path is deterministic per doc_id (`<doc_id>/fig_<n>.png`), so re-ingest
    overwrites in place. Persisted even when the later caption falls back to
    a placeholder — the figure must stay citable. Failures only log; ingest
    never blocks on asset persistence.
    """

    for c in chunks:
        if c.chunk_type != "image" or not c.image_b64 or not c.figure_id:
            continue
        n = c.figure_id.rsplit(":", 1)[-1]
        rel = f"{c.doc_id}/fig_{n}.png"
        try:
            (settings.assets_dir / c.doc_id).mkdir(parents=True, exist_ok=True)
            (settings.assets_dir / rel).write_bytes(base64.b64decode(c.image_b64))
            c.image_path = rel
        except Exception as exc:  # noqa: BLE001
            logger.warning("figure persist failed for %s: %s", c.figure_id, exc)


async def _caption_images(
    client: httpx.AsyncClient, chunks: list[Chunk], doc: ParsedDoc
) -> None:
    """Overwrite each image chunk's placeholder text with a VLM caption.

    Mutates chunks in place. On caption failure/drift, falls back to a
    locatable placeholder so the figure stays findable by doc/section.
    The transient `image_b64` is cleared afterwards so it never reaches the DB.
    """

    for c in chunks:
        if c.chunk_type != "image":
            continue
        caption = await caption_image(
            client, c.image_b64 or "", context_hint=c.section
        )
        c.text = caption or placeholder_caption(doc.title, c.section, c.page)
        c.image_b64 = None  # drop transient bytes


async def build_chunks_and_vectors(
    client: httpx.AsyncClient, doc: ParsedDoc
) -> tuple[list[Chunk], list[list[float]]]:
    """Chunk a parsed doc, run LLM enrichment, and embed — ready for upsert."""

    chunks = chunk_parsed(
        doc,
        max_chars=settings.chunk_max_chars,
        min_chars=settings.chunk_min_chars,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        return [], []

    if settings.use_table_summary:
        chunks = await _add_table_summaries(client, chunks, doc)

    if settings.persist_figure_images:
        _persist_figure_images(chunks)

    if settings.use_vlm_caption:
        await _caption_images(client, chunks, doc)

    # Contextual prepend affects the EMBEDDING input only; stored text (and
    # therefore citations) stay clean.
    embed_inputs = await _build_embed_inputs(client, chunks, doc)
    vectors = await embed_texts(client, embed_inputs)
    return chunks, vectors
