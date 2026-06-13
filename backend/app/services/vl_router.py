"""Route answer generation to the vision model when retrieval surfaces figures.

The routing rule is deliberately a heuristic over retrieved chunk types, not
an LLM intent classifier: an image chunk only reaches top-k when its VLM
caption matched the query, which is a stronger signal than pre-retrieval
intent — and a classifier would add latency plus an extra Ollama model swap
under the one-large-model-at-a-time memory constraint. When no figure is
retrieved, the text path (captions in context) is the correct behaviour
anyway, so the failure mode of a missed route is benign.

Shared by the chat router and scripts/eval.py so both measure the same path.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from ..models.schemas import Source
from .settings import settings

logger = logging.getLogger(__name__)

# page_image rows come from the ColQwen pilot (C6) and carry image_path too.
_VISUAL_TYPES = ("image", "page_image")

_ASSETS_URL_PREFIX = "/api/assets/"


def _asset_path(source: Source) -> Path | None:
    """Resolve a source's image_url back to a readable file under assets_dir."""

    if not source.image_url or not source.image_url.startswith(_ASSETS_URL_PREFIX):
        return None
    rel = source.image_url[len(_ASSETS_URL_PREFIX):]
    path = (settings.assets_dir / rel).resolve()
    # Belt-and-braces: never follow a URL outside the assets dir.
    if not path.is_relative_to(settings.assets_dir.resolve()):
        return None
    return path if path.is_file() else None


def should_route_vl(sources: list[Source]) -> bool:
    """True when VL answering is enabled and a retrieved figure exists on disk."""

    if not settings.use_vl_answer:
        return False
    return any(
        s.chunk_type in _VISUAL_TYPES and _asset_path(s) is not None
        for s in sources
    )


def load_source_images(sources: list[Source]) -> list[str]:
    """Base64 PNGs for the retrieved figures, capped at `vl_max_images`.

    Unreadable files are skipped. An empty list tells the caller to fall back
    to the text path.
    """

    images: list[str] = []
    for s in sources:
        if len(images) >= settings.vl_max_images:
            break
        if s.chunk_type not in _VISUAL_TYPES:
            continue
        path = _asset_path(s)
        if path is None:
            continue
        try:
            images.append(base64.b64encode(path.read_bytes()).decode("ascii"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load figure %s: %s", s.figure_id, exc)
    return images
