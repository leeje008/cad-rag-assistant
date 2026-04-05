"""Ollama chat/embedding/health helpers that share a single httpx client.

The client is created once in `main.lifespan` and stashed on
`app.state.http`. Routers pass it explicitly so nothing here owns the
lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import httpx

from .settings import SYSTEM_PROMPT_BASE, SYSTEM_PROMPT_WITH_CONTEXT, settings

logger = logging.getLogger(__name__)


def _build_system_prompt(context: str | None) -> str:
    if context and context.strip():
        return SYSTEM_PROMPT_WITH_CONTEXT.format(context=context)
    return SYSTEM_PROMPT_BASE


async def stream_chat(
    client: httpx.AsyncClient,
    message: str,
    *,
    model: str | None = None,
    context: str | None = None,
    history: list[dict] | None = None,
) -> AsyncIterator[str]:
    """Stream raw JSON lines from Ollama.

    The caller decides how to translate them into the Vercel AI SDK data
    stream protocol.
    """

    selected_model = model or settings.llm_model
    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(context)},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    payload = {
        "model": selected_model,
        "messages": messages,
        "stream": True,
    }

    try:
        async with client.stream(
            "POST",
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    yield line
    except asyncio.CancelledError:
        logger.info("stream_chat cancelled by client disconnect")
        raise
    except httpx.HTTPError as exc:
        logger.exception("Ollama chat request failed: %s", exc)
        raise


async def list_models(client: httpx.AsyncClient) -> list[dict]:
    resp = await client.get(
        f"{settings.ollama_base_url}/api/tags",
        timeout=httpx.Timeout(10.0),
    )
    resp.raise_for_status()
    return resp.json().get("models", [])


async def check_ollama_health(client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=httpx.Timeout(5.0),
        )
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return False
