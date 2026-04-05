from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..models.schemas import ChatRequest
from ..services.llm import stream_chat
from ..services.retriever import format_context, retrieve

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request):
    """Chat endpoint that retrieves top-K context then streams tokens.

    Wire format (custom superset of Vercel AI SDK Data Stream Protocol
    v1):
        s:[{...Source}, ...]\\n   # single leading source part (custom)
        0:"token"\\n             # text parts
        e:{...}\\n               # finish part
        d:{...}\\n               # done part
    """

    client = http_request.app.state.http
    table = http_request.app.state.table

    sources = []
    if table is not None:
        try:
            sources = await retrieve(client, table, request.message)
        except Exception as exc:  # noqa: BLE001
            logger.exception("retrieve failed, falling back to base prompt: %s", exc)
            sources = []

    context = format_context(sources)
    history = [msg.model_dump() for msg in request.history] if request.history else None

    async def generate():
        # Leading custom `s:` source part (always emitted, even when empty).
        source_payload = [s.model_dump() for s in sources]
        yield f"s:{json.dumps(source_payload, ensure_ascii=False)}\n"

        finish_reason = "stop"
        try:
            async for line in stream_chat(
                client,
                request.message,
                model=request.model,
                context=context,
                history=history,
            ):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = data.get("message", {}).get("content", "")
                done = data.get("done", False)
                if content:
                    yield f"0:{json.dumps(content)}\n"
                if done:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat stream failed: %s", exc)
            finish_reason = "error"
            yield f"3:{json.dumps({'message': str(exc)})}\n"

        finish_payload = {
            "finishReason": finish_reason,
            "usage": {"promptTokens": 0, "completionTokens": 0},
        }
        yield f"e:{json.dumps(finish_payload)}\n"
        yield f"d:{json.dumps({'finishReason': finish_reason})}\n"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Vercel-AI-Data-Stream": "v1",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
