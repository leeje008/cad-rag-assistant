"""Document parsers producing section/page-aware markdown blocks.

PDF: pymupdf4llm (markdown per page)
XLSX: openpyxl + pandas (markdown per sheet, merged cells forward-filled)

Both yield `ParsedDoc` with enough metadata for the chunker to keep page
and section information through the retrieval pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParsedBlock:
    text: str
    page: int | None = None
    section: str | None = None


@dataclass
class ParsedDoc:
    doc_id: str
    source_path: str
    title: str
    blocks: list[ParsedBlock] = field(default_factory=list)


def _normalise_path(path: Path) -> Path:
    return Path(unicodedata.normalize("NFC", str(path)))


def _doc_id(path: Path) -> str:
    path = _normalise_path(path)
    stat = path.stat()
    key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def parse_pdf(path: Path) -> ParsedDoc:
    """Parse a PDF into one ParsedBlock per page using pymupdf4llm."""

    import pymupdf4llm

    path = _normalise_path(path)
    raw = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    blocks: list[ParsedBlock] = []
    for item in raw:
        # Newer versions return dicts; older tuple builds are unlikely but defended.
        if isinstance(item, dict):
            text = item.get("text", "") or ""
            meta = item.get("metadata") or {}
            page = meta.get("page")
            if page is None:
                page = item.get("page")
        elif isinstance(item, tuple) and item:
            text = item[0] if isinstance(item[0], str) else ""
            page = item[1] if len(item) > 1 and isinstance(item[1], int) else None
        else:
            continue

        text = (text or "").strip()
        if not text:
            continue

        section = _extract_first_heading(text)
        blocks.append(
            ParsedBlock(text=text, page=_coerce_page(page), section=section)
        )

    return ParsedDoc(
        doc_id=_doc_id(path),
        source_path=str(path),
        title=path.name,
        blocks=blocks,
    )


def parse_xlsx(path: Path) -> ParsedDoc:
    """Parse an XLSX workbook into one ParsedBlock per sheet."""

    import openpyxl
    import pandas as pd

    path = _normalise_path(path)
    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=False)

    blocks: list[ParsedBlock] = []
    for sheet_index, sheet_name in enumerate(wb.sheetnames, start=1):
        ws = wb[sheet_name]

        # Forward-fill merged cells so each logical row sees the header.
        for merged in list(ws.merged_cells.ranges):
            min_col, min_row, max_col, max_row = merged.bounds
            top_value = ws.cell(row=min_row, column=min_col).value
            ws.unmerge_cells(str(merged))
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    ws.cell(row=row, column=col).value = top_value

        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        # Drop fully-empty leading rows.
        while rows and all(cell is None for cell in rows[0]):
            rows = rows[1:]
        if not rows:
            continue

        header = [
            str(cell) if cell is not None else f"col_{i}"
            for i, cell in enumerate(rows[0])
        ]
        data_rows = rows[1:]
        df = pd.DataFrame(data_rows, columns=header)
        if df.empty:
            continue

        md_table = df.fillna("").astype(str).to_markdown(index=False)
        block_text = f"# {sheet_name}\n\n{md_table}"
        blocks.append(
            ParsedBlock(text=block_text, page=sheet_index, section=sheet_name)
        )

    wb.close()
    return ParsedDoc(
        doc_id=_doc_id(path),
        source_path=str(path),
        title=path.name,
        blocks=blocks,
    )


def parse_any(path: Path) -> ParsedDoc:
    path = _normalise_path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in {".xlsx", ".xlsm"}:
        return parse_xlsx(path)
    raise ValueError(f"Unsupported file type: {suffix} ({path.name})")


# ---------------------------------------------------------------------------


def _coerce_page(page: object) -> int | None:
    if page is None:
        return None
    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def _extract_first_heading(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip() or None
    return None


SUPPORTED_SUFFIXES = {".pdf", ".xlsx", ".xlsm"}
