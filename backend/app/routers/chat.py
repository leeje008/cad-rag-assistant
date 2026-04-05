from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..models.schemas import ChatRequest
from ..services.llm import stream_chat

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request):
    """Chat endpoint producing a Vercel AI SDK Data Stream (text parts only).

    Phase 1 will prepend a custom `s:` source part and inject retrieval
    context before the token stream. For Phase 0 this remains a straight
    Ollama passthrough but now reuses the shared httpx client.
    """

    client = http_request.app.state.http
    history = [msg.model_dump() for msg in request.history] if request.history else None

    async def generate():
        finish_reason = "stop"
        try:
            async for line in stream_chat(
                client,
                request.message,
                model=request.model,
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
            err_payload = {"message": str(exc)}
            yield f'3:{json.dumps(err_payload)}\n'

        finish_payload = {
            "finishReason": finish_reason,
            "usage": {"promptTokens": 0, "completionTokens": 0},
        }
        yield f"e:{json.dumps(finish_payload)}\n"
        yield f'd:{json.dumps({"finishReason": finish_reason})}\n'

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
