"""Serve persisted figure images extracted at ingest time (Phase C).

Filenames are fully deterministic (`<16-hex doc_id>/fig_<n>.png`), so a
regex whitelist is both simpler and stricter than path-resolution checks.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..services.settings import settings

router = APIRouter()

_DOC_ID_RE = re.compile(r"^[0-9a-f]{16}$")
# fig_N: parsed figure crops (C1); page_N: ColQwen pilot page renders (C6).
_FILENAME_RE = re.compile(r"^(fig|page)_\d+\.png$")

# Assets are content-addressed: doc_id changes whenever the source file does.
_CACHE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}


@router.get("/api/assets/{doc_id}/{filename}")
async def get_asset(doc_id: str, filename: str) -> FileResponse:
    if not _DOC_ID_RE.match(doc_id) or not _FILENAME_RE.match(filename):
        raise HTTPException(status_code=404, detail="asset not found")
    path = settings.assets_dir / doc_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(path, media_type="image/png", headers=_CACHE_HEADERS)
