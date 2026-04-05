"""Ollama embedding client.

Calls `/api/embed` (plural input supported in current Ollama builds) in
batches. Shared httpx client is passed in from the lifespan.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .settings import settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


async def _embed_batch(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    resp = await client.post(
        f"{settings.ollama_base_url}/api/embed",
        json={"model": settings.embed_model, "input": texts},
        timeout=httpx.Timeout(120.0),
    )
    resp.raise_for_status()
    data = resp.json()
    vectors = data.get("embeddings")
    if vectors is None:
        # Older Ollama versions return "embedding" (singular) for a single input.
        single = data.get("embedding")
        if single is not None:
            return [single]
        raise EmbeddingError(f"unexpected embed response keys: {list(data.keys())}")
    return vectors


async def embed_texts(
    client: httpx.AsyncClient,
    texts: list[str],
    *,
    batch_size: int | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    batch = batch_size or settings.embed_batch
    out: list[list[float]] = []
    for start in range(0, len(texts), batch):
        slice_ = texts[start : start + batch]
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.HTTPError, EmbeddingError)),
            reraise=True,
        ):
            with attempt:
                vectors = await _embed_batch(client, slice_)
        if len(vectors) != len(slice_):
            raise EmbeddingError(
                f"embed batch size mismatch: requested {len(slice_)} got {len(vectors)}"
            )
        out.extend(vectors)
        logger.debug("embedded %d/%d", len(out), len(texts))

    return out


async def embed_query(client: httpx.AsyncClient, query: str) -> list[float]:
    result = await embed_texts(client, [query])
    return result[0]
