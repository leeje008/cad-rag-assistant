"""Retrieval pipeline.

Phase 2 adds a hybrid dense+FTS path with RRF fusion and an optional
BGE-reranker re-ranking step. The legacy dense-only `retrieve()` is
preserved as a fallback and is still used by `/api/retrieve` when the
caller passes no flags.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..models.schemas import Source
from .embedder import embed_query
from .reranker import get_reranker
from .settings import settings
from .vectorstore import search, search_fts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers


def _row_to_source(row: dict[str, Any]) -> Source:
    page_val = row.get("page")
    if isinstance(page_val, int) and page_val < 0:
        page_val = None
    return Source(
        document=row.get("title", ""),
        page=page_val,
        section=row.get("section") or None,
        relevance=row.get("_relevance"),
        text=row.get("text", ""),
        source_path=row.get("source_path", ""),
        chunk_id=row.get("chunk_id", ""),
    )


def _rrf_merge(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int | None = None,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Combines rows by `chunk_id`. Each row gets a fused score
    `Σ 1 / (k + rank)` where rank is 1-based per list.
    """

    rrf_k = k or settings.rrf_k
    fused: dict[str, dict[str, Any]] = {}
    for results in ranked_lists:
        for rank, row in enumerate(results, start=1):
            cid = row.get("chunk_id")
            if not cid:
                continue
            score = 1.0 / (rrf_k + rank)
            if cid in fused:
                fused[cid]["_rrf"] += score
            else:
                entry = dict(row)
                entry["_rrf"] = score
                fused[cid] = entry
    ordered = sorted(fused.values(), key=lambda r: r["_rrf"], reverse=True)
    return ordered


# ---------------------------------------------------------------------------
# Public API


async def retrieve(
    client: httpx.AsyncClient,
    table,
    query: str,
    k: int | None = None,
) -> list[Source]:
    """Legacy dense-only retrieval (kept for the debug /api/retrieve route)."""

    if table is None:
        return []

    top_k = k or settings.top_k
    vector = await embed_query(client, query)
    rows = search(table, vector, top_k)
    return [_row_to_source(r) for r in rows]


async def retrieve_hybrid(
    client: httpx.AsyncClient,
    table,
    query: str,
    *,
    k: int | None = None,
    candidate_n: int | None = None,
    use_reranker: bool | None = None,
) -> list[Source]:
    """Dense + FTS retrieval fused via RRF, optionally reranked.

    1. dense vector search → top `candidate_n`
    2. FTS search on kiwi-tokenised column → top `candidate_n`
    3. RRF merge by `chunk_id`
    4. (optional) BGE cross-encoder rerank → top `k`
    """

    if table is None:
        return []

    final_k = k or settings.final_top_k
    cand = candidate_n or settings.rerank_candidate_n
    rerank = settings.use_reranker if use_reranker is None else use_reranker

    # Dense
    vector = await embed_query(client, query)
    dense_rows = search(table, vector, cand)

    # Lexical
    fts_rows: list[dict[str, Any]] = []
    if settings.use_hybrid:
        fts_rows = search_fts(table, query, cand)

    # Fuse
    fused = _rrf_merge([dense_rows, fts_rows])[: max(cand, final_k)]
    if not fused:
        return []

    if rerank and len(fused) > 1:
        reranker = get_reranker()
        passages = [r.get("text", "") for r in fused]
        try:
            scores = await reranker.score(query, passages)
            for row, s in zip(fused, scores, strict=True):
                row["_rerank"] = float(s)
            fused.sort(key=lambda r: r.get("_rerank", 0.0), reverse=True)
            # Expose the rerank score as the relevance for downstream UI.
            for row in fused:
                row["_relevance"] = row.get("_rerank")
        except Exception as exc:  # noqa: BLE001
            logger.warning("reranker failed, falling back to RRF order: %s", exc)
            for row in fused:
                row["_relevance"] = row.get("_rrf")
    else:
        for row in fused:
            row["_relevance"] = row.get("_rrf")

    top = fused[:final_k]
    return [_row_to_source(r) for r in top]


def format_context(sources: list[Source], *, max_chars_per_source: int = 1200) -> str:
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
