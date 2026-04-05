"""Query → embed → LanceDB search → list[Source]."""

from __future__ import annotations

import logging

import httpx

from ..models.schemas import Source
from .embedder import embed_query
from .settings import settings
from .vectorstore import search

logger = logging.getLogger(__name__)


async def retrieve(
    client: httpx.AsyncClient,
    table,
    query: str,
    k: int | None = None,
) -> list[Source]:
    if table is None:
        logger.warning("retrieve called with no LanceDB table — returning []")
        return []

    top_k = k or settings.top_k
    vector = await embed_query(client, query)
    rows = search(table, vector, top_k)

    sources: list[Source] = []
    for row in rows:
        page_val = row.get("page")
        if isinstance(page_val, int) and page_val < 0:
            page_val = None
        sources.append(
            Source(
                document=row.get("title", ""),
                page=page_val,
                section=row.get("section") or None,
                relevance=row.get("_relevance"),
                text=row.get("text", ""),
                source_path=row.get("source_path", ""),
                chunk_id=row.get("chunk_id", ""),
            )
        )
    return sources


def format_context(sources: list[Source], *, max_chars_per_source: int = 1200) -> str:
    """Build the `[CONTEXT]` block injected into the LLM system prompt."""

    if not sources:
        return ""

    parts: list[str] = []
    for i, src in enumerate(sources, start=1):
        header = f"[{i}] {src.document}"
        if src.page is not None:
            header += f" p.{src.page}"
        if src.section:
            header += f" · {src.section}"
        text = src.text.strip()
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rstrip() + "…"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)
