"""Rebuild the LanceDB table from scratch using the currently configured
embedding model (e.g. after switching from mxbai-embed-large to bge-m3).

Strategy: create a NEW table (<table>__new), ingest everything under
SPEC_DIR into it, then atomically swap by dropping the old table and
renaming. This keeps `/api/chat` responsive during rebuilds.

Usage:
    python scripts/rebuild_index.py                     # full rebuild of spec_dir
    python scripts/rebuild_index.py --path SPEC/sub     # limited rebuild
    python scripts/rebuild_index.py --dry-run           # parse/chunk only, no embed/upsert
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
import time
from pathlib import Path

# Make backend/app importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import httpx  # noqa: E402
import lancedb  # noqa: E402

from app.services.chunker import chunk_parsed  # noqa: E402
from app.services.embedder import embed_texts  # noqa: E402
from app.services.parsers import SUPPORTED_SUFFIXES, parse_any  # noqa: E402
from app.services.settings import settings  # noqa: E402
from app.services.vectorstore import (  # noqa: E402
    _schema,
    ensure_fts_index,
    open_or_create_table,
    upsert_chunks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rebuild_index")


def _discover(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


async def _run(path: Path, dry_run: bool) -> int:
    db_path = Path(settings.lancedb_path)
    target_name = settings.lancedb_table
    staging_name = f"{target_name}__new"

    db = lancedb.connect(str(db_path))
    if staging_name in db.table_names():
        logger.warning("dropping stale staging table %s", staging_name)
        db.drop_table(staging_name)

    if dry_run:
        staging_table = None
    else:
        staging_table = db.create_table(
            staging_name,
            schema=_schema(settings.embed_dimension),
            mode="create",
        )
        logger.info("created staging table %s", staging_name)

    files = _discover(path)
    if not files:
        logger.error("no supported files under %s", path)
        return 1
    logger.info("discovered %d files under %s", len(files), path)

    total_chunks = 0
    errors: list[str] = []

    timeout = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for i, file_path in enumerate(files, start=1):
            t0 = time.perf_counter()
            try:
                doc = parse_any(file_path)
                if not doc.blocks:
                    errors.append(f"{file_path.name}: empty")
                    logger.warning("[%d/%d] %s: no content", i, len(files), file_path.name)
                    continue

                chunks = chunk_parsed(
                    doc,
                    max_chars=settings.chunk_max_chars,
                    min_chars=settings.chunk_min_chars,
                    overlap=settings.chunk_overlap,
                )
                if not chunks:
                    errors.append(f"{file_path.name}: 0 chunks")
                    continue

                if dry_run:
                    total_chunks += len(chunks)
                    logger.info(
                        "[%d/%d] DRY %s → %d chunks", i, len(files), file_path.name, len(chunks)
                    )
                    continue

                vectors = await embed_texts(client, [c.text for c in chunks])
                upsert_chunks(staging_table, chunks, vectors)
                total_chunks += len(chunks)
                dt = time.perf_counter() - t0
                logger.info(
                    "[%d/%d] %s → %d chunks (%.1fs)",
                    i,
                    len(files),
                    file_path.name,
                    len(chunks),
                    dt,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{file_path.name}: {exc}")
                logger.exception("failed on %s", file_path)

    if dry_run:
        logger.info("DRY RUN: would have indexed %d chunks", total_chunks)
        return 0

    # Atomic-ish swap: drop old, rename staging → target.
    if target_name in db.table_names():
        logger.info("dropping old table %s", target_name)
        db.drop_table(target_name)

    # LanceDB has no direct rename; re-create target with staging data.
    staging_table = db.open_table(staging_name)
    rows = staging_table.to_arrow()
    final_table = db.create_table(target_name, data=rows, mode="create")
    db.drop_table(staging_name)
    ensure_fts_index(final_table)
    logger.info(
        "rebuild done: table=%s rows=%d chunks=%d errors=%d",
        target_name,
        final_table.count_rows(),
        total_chunks,
        len(errors),
    )
    if errors:
        for e in errors:
            logger.error("  %s", e)
    return 0 if not errors else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default=str(settings.spec_dir),
        help="file or directory to index (default: settings.spec_dir)",
    )
    ap.add_argument("--dry-run", action="store_true", help="parse/chunk only")
    args = ap.parse_args()

    target = Path(args.path)
    if not target.is_absolute():
        target = REPO_ROOT / target

    return asyncio.run(_run(target, dry_run=args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
