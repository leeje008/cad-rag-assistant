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

# Parent chunks exist only as expansion targets — never ranked directly.
_PARENT_FILTER = "chunk_type != 'parent'"


# ---------------------------------------------------------------------------
# Helpers


def _row_to_source(row: dict[str, Any]) -> Source:
    page_val = row.get("page")
    if isinstance(page_val, int) and page_val < 0:
        page_val = None
    # Context preference: a table_summary hit substitutes the ORIGINAL table
    # (resolved by table_id); a text hit expands to its parent section; tables
    # and images keep their own content. Citation metadata stays on the hit.
    text = (
        row.get("_table_text")
        or row.get("_parent_text")
        or row.get("text", "")
    )
    return Source(
        document=row.get("title", ""),
        page=page_val,
        section=row.get("section") or None,
        relevance=row.get("_relevance"),
        text=text,
        source_path=row.get("source_path", ""),
        chunk_id=row.get("chunk_id", ""),
        chunk_type=row.get("chunk_type") or "text",
        table_id=row.get("table_id") or None,
        parent_id=row.get("parent_id") or None,
    )


def _is_text(row: dict[str, Any]) -> bool:
    return (row.get("chunk_type") or "text") == "text"


def _expand_to_parents(table, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the parent-section text as `_parent_text` to text children.

    Table/image children keep their own content (they are standalone
    citable artifacts; tables get original-table substitution in PR4).
    One filtered fetch covers all distinct parent_ids.
    """

    parent_ids = {
        r.get("parent_id") for r in rows if _is_text(r) and r.get("parent_id")
    }
    parent_ids.discard("")
    if not parent_ids:
        return rows
    quoted = ",".join("'" + str(p).replace("'", "''") + "'" for p in parent_ids)
    try:
        parents = (
            table.search()
            .where(f"chunk_id IN ({quoted})")
            .limit(len(parent_ids))
            .to_list()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("parent expansion fetch failed: %s", exc)
        return rows
    pmap = {p.get("chunk_id"): p.get("text", "") for p in parents}
    for r in rows:
        if not _is_text(r):
            continue
        ptext = pmap.get(r.get("parent_id"))
        if ptext:
            r["_parent_text"] = ptext
    return rows


def _resolve_table_summaries(table, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Substitute the original table markdown for `table_summary` hits.

    Retrieval matches the summary; generation must see the original table.
    Fetches sibling `table` chunks by table_id and attaches `_table_text`.
    """

    table_ids = {
        r.get("table_id")
        for r in rows
        if r.get("chunk_type") == "table_summary" and r.get("table_id")
    }
    table_ids.discard("")
    if not table_ids:
        return rows
    quoted = ",".join("'" + str(t).replace("'", "''") + "'" for t in table_ids)
    try:
        originals = (
            table.search()
            .where(f"table_id IN ({quoted}) AND chunk_type = 'table'")
            .limit(len(table_ids))
            .to_list()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("table-original resolution failed: %s", exc)
        return rows
    omap = {o.get("table_id"): o.get("text", "") for o in originals}
    for r in rows:
        if r.get("chunk_type") == "table_summary":
            original = omap.get(r.get("table_id"))
            if original:
                r["_table_text"] = original
    return rows


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse redundant hits, keeping the best-ranked first.

    Assumes `rows` is sorted best-first. Collapses: text children sharing a
    parent (same section), and `table`/`table_summary` hits sharing a table_id
    (same table). Images and parent-less rows stay distinct.
    """

    seen_parents: set[str] = set()
    seen_tables: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        ctype = r.get("chunk_type") or "text"
        if ctype == "text" and r.get("parent_id"):
            if r["parent_id"] in seen_parents:
                continue
            seen_parents.add(r["parent_id"])
        elif ctype in ("table", "table_summary") and r.get("table_id"):
            if r["table_id"] in seen_tables:
                continue
            seen_tables.add(r["table_id"])
        out.append(r)
    return out


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
    rows = search(table, vector, top_k, where=_PARENT_FILTER)
    rows = _expand_to_parents(table, rows)
    rows = _resolve_table_summaries(table, rows)
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

    # Dense (parents excluded — they are expansion targets, not candidates)
    vector = await embed_query(client, query)
    dense_rows = search(table, vector, cand, where=_PARENT_FILTER)

    # Lexical
    fts_rows: list[dict[str, Any]] = []
    if settings.use_hybrid:
        fts_rows = search_fts(table, query, cand, where=_PARENT_FILTER)

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

    # Distinct sections/tables: collapse redundant hits before slicing.
    fused = _dedupe(fused)
    top = fused[:final_k]
    top = _expand_to_parents(table, top)
    top = _resolve_table_summaries(table, top)
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
