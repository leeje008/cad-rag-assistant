import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..models.schemas import ChatRequest
from ..services.llm import stream_chat

router = APIRouter()


@router.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint that streams responses from Ollama in AI SDK compatible format."""

    async def generate():
        async for line in stream_chat(request.message, request.model):
            try:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                done = data.get("done", False)

                if content:
                    # Vercel AI SDK Data Stream Protocol: text part
                    yield f"0:{json.dumps(content)}\n"

                if done:
                    # Send finish reason
                    yield f"e:{json.dumps({"finishReason": "stop", "usage": {"promptTokens": 0, "completionTokens": 0}})}\n"
                    # Send done signal
                    yield f"d:{json.dumps({"finishReason": "stop"})}\n"

            except json.JSONDecodeError:
                continue

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Vercel-AI-Data-Stream": "v1",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
