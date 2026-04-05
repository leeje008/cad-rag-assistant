"""LanceDB wrapper: schema, open-or-create, doc-level upsert, search.

Phase 2 additions:
- `text_fts` column populated with kiwi-tokenised text for Tantivy FTS
- `ensure_fts_index(table)` — creates whitespace tokenizer index idempotently
- `search_fts(table, query_tokens, k)` — lexical search over the FTS column
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from .chunker import Chunk
from .settings import settings
from .tokenizer_ko import tokenize_for_fts

logger = logging.getLogger(__name__)


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("title", pa.string()),
            pa.field("text", pa.string()),
            pa.field("text_fts", pa.string()),
            pa.field("section", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("char_start", pa.int32()),
            pa.field("char_end", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def open_or_create_table(
    db_path: Path | None = None,
    table_name: str | None = None,
    dim: int | None = None,
):
    db_path = Path(db_path or settings.lancedb_path)
    db_path.mkdir(parents=True, exist_ok=True)
    name = table_name or settings.lancedb_table
    dimension = dim or settings.embed_dimension

    db = lancedb.connect(str(db_path))
    if name in db.table_names():
        table = db.open_table(name)
        logger.info("opened LanceDB table %s (rows=%d)", name, table.count_rows())
        return table

    table = db.create_table(name, schema=_schema(dimension), mode="create")
    logger.info("created LanceDB table %s (dim=%d)", name, dimension)
    return table


def ensure_fts_index(table) -> None:
    """Create the whitespace FTS index on `text_fts` if it doesn't exist.

    LanceDB re-indexing is idempotent but noisy, so we try once and log on
    failure rather than blocking startup.
    """

    if table is None:
        return
    try:
        table.create_fts_index("text_fts", replace=True)
        logger.info("ensured FTS index on text_fts")
    except Exception as exc:  # noqa: BLE001
        logger.warning("create_fts_index skipped: %s", exc)


def upsert_chunks(table, chunks: list[Chunk], vectors: list[list[float]]) -> int:
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunk/vector length mismatch: {len(chunks)} vs {len(vectors)}"
        )

    # Delete any existing rows for these doc_ids first (doc-level upsert).
    doc_ids = {c.doc_id for c in chunks}
    for doc_id in doc_ids:
        safe = doc_id.replace("'", "''")
        try:
            table.delete(f"doc_id = '{safe}'")
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete for doc_id=%s failed (maybe empty): %s", doc_id, exc)

    rows: list[dict[str, Any]] = []
    for chunk, vec in zip(chunks, vectors, strict=True):
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_path": chunk.source_path,
                "title": chunk.title,
                "text": chunk.text,
                "text_fts": tokenize_for_fts(chunk.text),
                "section": chunk.section or "",
                "page": chunk.page if chunk.page is not None else -1,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "vector": [float(v) for v in vec],
            }
        )
    table.add(rows)
    logger.info("upserted %d chunks (%d docs)", len(rows), len(doc_ids))
    return len(rows)


def search(table, query_vector: list[float], k: int) -> list[dict[str, Any]]:
    """Dense-only vector search (kept for the legacy retriever path)."""

    if table is None:
        return []
    result = table.search(query_vector).limit(k).to_list()
    for row in result:
        distance = row.get("_distance")
        if distance is None:
            row["_relevance"] = None
        else:
            row["_relevance"] = 1.0 / (1.0 + float(distance))
    return result


def search_fts(table, query: str, k: int) -> list[dict[str, Any]]:
    """Lexical (BM25 via Tantivy) search over the kiwi-tokenised column.

    Returns rows with a `_score` field (higher is better).
    """

    if table is None:
        return []
    tokenised_query = tokenize_for_fts(query)
    if not tokenised_query:
        return []
    try:
        result = (
            table.search(tokenised_query, query_type="fts", fts_columns="text_fts")
            .limit(k)
            .to_list()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("FTS search failed, returning empty: %s", exc)
        return []
    for row in result:
        score = row.get("_score")
        row["_relevance"] = float(score) if score is not None else None
    return result


def count_rows(table) -> int:
    if table is None:
        return 0
    return table.count_rows()
