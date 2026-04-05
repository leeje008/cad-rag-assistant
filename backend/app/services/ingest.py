"""Ingest orchestration: path → parse → chunk → embed → upsert.

Job state is kept in an in-memory dict. Dev-only: on `uvicorn --reload`
the state is lost between reloads. Phase 2 will move to a task queue.
"""

from __future__ import annotations

import logging
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx

from .chunker import chunk_parsed
from .embedder import embed_texts
from .parsers import SUPPORTED_SUFFIXES, parse_any
from .settings import settings
from .vectorstore import upsert_chunks

logger = logging.getLogger(__name__)


Status = Literal["pending", "running", "done", "error"]


@dataclass
class IngestJob:
    job_id: str
    path: str
    status: Status = "pending"
    total: int = 0
    processed: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None


INGEST_JOBS: dict[str, IngestJob] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _discover(path: Path, recursive: bool) -> list[Path]:
    path = Path(unicodedata.normalize("NFC", str(path)))
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    if not path.is_dir():
        return []
    glob = "**/*" if recursive else "*"
    files: list[Path] = []
    for p in path.glob(glob):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(p)
    return sorted(files)


def create_job(path: str) -> IngestJob:
    job = IngestJob(job_id=uuid.uuid4().hex[:12], path=path)
    INGEST_JOBS[job.job_id] = job
    return job


async def run_ingest(
    job_id: str,
    path: str,
    client: httpx.AsyncClient,
    table,
    *,
    recursive: bool = True,
) -> None:
    job = INGEST_JOBS.get(job_id)
    if job is None:
        return
    job.status = "running"
    job.started_at = _now()

    try:
        raw_path = Path(path)
        if not raw_path.is_absolute():
            # Relative paths resolve against the repo root (parent of backend/).
            raw_path = settings.spec_dir.parent / raw_path

        files = _discover(raw_path, recursive)
        job.total = len(files)
        if not files:
            job.errors.append(f"no supported files under {raw_path}")
            job.status = "error"
            job.finished_at = _now()
            return

        for file_path in files:
            try:
                doc = parse_any(file_path)
                if not doc.blocks:
                    job.errors.append(f"{file_path.name}: no content extracted")
                    job.processed += 1
                    continue

                chunks = chunk_parsed(
                    doc,
                    max_chars=settings.chunk_max_chars,
                    min_chars=settings.chunk_min_chars,
                    overlap=settings.chunk_overlap,
                )
                if not chunks:
                    job.errors.append(f"{file_path.name}: chunker produced 0 chunks")
                    job.processed += 1
                    continue

                vectors = await embed_texts(client, [c.text for c in chunks])
                upsert_chunks(table, chunks, vectors)
                job.processed += 1
                logger.info(
                    "ingested %s (%d chunks)", file_path.name, len(chunks)
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("ingest failed for %s", file_path)
                job.errors.append(f"{file_path.name}: {exc}")
                job.processed += 1

        job.status = "done" if not job.errors or job.processed > 0 else "error"
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingest job %s crashed", job_id)
        job.errors.append(str(exc))
        job.status = "error"
    finally:
        job.finished_at = _now()
