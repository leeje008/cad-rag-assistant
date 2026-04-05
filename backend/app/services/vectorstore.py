"""LanceDB wrapper: schema, open-or-create, doc-level upsert, search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from .chunker import Chunk
from .settings import settings

logger = logging.getLogger(__name__)


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("title", pa.string()),
            pa.field("text", pa.string()),
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


def count_rows(table) -> int:
    if table is None:
        return 0
    return table.count_rows()
