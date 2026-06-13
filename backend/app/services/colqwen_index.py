"""ColQwen page-image retrieval (C6 pilot, flag-off by default).

Sidecar multi-vector index, deliberately NOT in LanceDB: the pinned
lancedb 0.15 has no usable multivector/MaxSim support, and the pilot corpus
(P&ID binders, key drawing pages — well under 1k pages) is small enough for
brute-force MaxSim over in-memory tensors on MPS/CPU.

Layout under settings.colqwen_dir (built by scripts/build_colqwen_index.py):
    manifest.json            [{doc_id, page, title, source_path, image_path, tensor}]
    {doc_id}/page_{n}.pt     per-page multi-vector embedding (fp16)

Page PNGs are written under settings.assets_dir/{doc_id}/page_{n}.png so the
existing citation chain (C2 image_url) and VL routing (C4) work unchanged.

Search results are pseudo-rows (chunk_type="page_image") shaped like LanceDB
rows so they flow straight into retriever RRF fusion and _row_to_source.
The model loads lazily on the first query after the flag is enabled; a failed
load is cached so a missing optional dependency degrades to text-only
retrieval instead of paying the import cost per query.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .settings import settings

logger = logging.getLogger(__name__)


class _ColQwenSearcher:
    def __init__(self) -> None:
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = ColQwen2.from_pretrained(
            settings.colqwen_model,
            torch_dtype=torch.float16,
            device_map=self.device,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(settings.colqwen_model)

        manifest_path = settings.colqwen_dir / "manifest.json"
        self.entries: list[dict[str, Any]] = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else []
        )
        self.page_embeddings = [
            torch.load(settings.colqwen_dir / e["tensor"], map_location=self.device)
            for e in self.entries
        ]
        logger.info(
            "ColQwen index loaded: %d pages on %s", len(self.entries), self.device
        )

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        if not self.entries:
            return []
        torch = self.torch
        batch = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            q_embedding = self.model(**batch)
        scores = self.processor.score_multi_vector(
            q_embedding, self.page_embeddings
        )[0]
        top = torch.topk(scores, min(k, len(self.entries)))
        rows: list[dict[str, Any]] = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            e = self.entries[idx]
            page = e.get("page", -1)
            rows.append(
                {
                    "chunk_id": f"{e['doc_id']}:page:{page}",
                    "doc_id": e["doc_id"],
                    "title": e.get("title", ""),
                    "source_path": e.get("source_path", ""),
                    "text": f"[도면 페이지] {e.get('title', '')} p.{page}",
                    "section": "",
                    "page": page,
                    "chunk_type": "page_image",
                    "parent_id": "",
                    "table_id": "",
                    "figure_id": f"{e['doc_id']}:page:{page}",
                    "level": 1,
                    "table_html": "",
                    "image_path": e.get("image_path", ""),
                    "_maxsim": float(score),
                }
            )
        return rows


# False marks a failed init so a missing optional dep is paid once, not per query.
_searcher: _ColQwenSearcher | None | bool = None


def search_pages(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """Top-k drawing pages by MaxSim, or [] when the pilot is unavailable."""

    global _searcher
    if _searcher is False:
        return []
    if _searcher is None:
        try:
            _searcher = _ColQwenSearcher()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ColQwen pilot unavailable, disabling: %s", exc)
            _searcher = False
            return []
    try:
        return _searcher.search(query, k or settings.colqwen_top_k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ColQwen search failed: %s", exc)
        return []
