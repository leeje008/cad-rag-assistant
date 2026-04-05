"""BGE-Reranker-v2-m3 cross-encoder wrapper.

Uses sentence-transformers `CrossEncoder` rather than FlagEmbedding —
same underlying BAAI/bge-reranker-v2-m3 weights, but with a lighter
dependency footprint and better compatibility with current transformers.

The model is loaded lazily on first use (several seconds on CPU) so
that uvicorn startup stays fast. Subsequent calls re-use the cached
instance via `asyncio.to_thread` to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from .settings import settings

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.reranker_model
        self._model: Any = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import CrossEncoder  # heavy import

            logger.info("loading reranker model %s (CPU)", self.model_name)
            # M4 MPS isn't officially supported for CrossEncoder fp16, so we
            # stay on CPU. fp16 halves memory but adds a tiny accuracy drop.
            self._model = CrossEncoder(self.model_name, max_length=512, device="cpu")
            logger.info("reranker loaded")

    def _score_sync(self, query: str, passages: list[str]) -> list[float]:
        self._ensure_loaded()
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return await asyncio.to_thread(self._score_sync, query, passages)


_singleton: Reranker | None = None


def get_reranker() -> Reranker:
    global _singleton
    if _singleton is None:
        _singleton = Reranker()
    return _singleton
