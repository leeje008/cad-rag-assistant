from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..models.schemas import ChatRequest, Source
from ..services.llm import stream_chat
from ..services.query_rewriter import expand_queries
from ..services.retriever import format_context, retrieve_hybrid
from ..services.settings import settings
from ..services.vl_router import load_source_images, should_route_vl

router = APIRouter()
logger = logging.getLogger(__name__)


async def _multi_query_retrieve(client, table, query: str) -> list[Source]:
    """Run hybrid retrieval on N query variants and fuse by chunk_id."""

    queries = await expand_queries(client, query)
    logger.info("multi-query variants (%d): %s", len(queries), queries)

    seen: dict[str, Source] = {}
    order: list[str] = []
    for q in queries:
        hits = await retrieve_hybrid(
            client,
            table,
            q,
            k=settings.final_top_k,
            candidate_n=settings.rerank_candidate_n,
        )
        for s in hits:
            # Across variants: collapse text children by section (parent_id)
            # and table/table_summary by table_id; else key by chunk_id.
            if s.chunk_type == "text" and s.parent_id:
                key = f"p:{s.parent_id}"
            elif s.chunk_type in ("table", "table_summary") and s.table_id:
                key = f"t:{s.table_id}"
            else:
                key = s.chunk_id or f"{s.document}:{s.page}"
            if key in seen:
                # Keep the highest relevance seen across variants.
                existing = seen[key]
                if (s.relevance or 0) > (existing.relevance or 0):
                    seen[key] = s
            else:
                seen[key] = s
                order.append(key)

    merged = [seen[k] for k in order]
    merged.sort(key=lambda s: (s.relevance or 0), reverse=True)
    return merged[: settings.final_top_k]


@router.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request):
    """Chat endpoint: multi-query hybrid retrieval → context → stream tokens.

    Wire format (custom superset of Vercel AI SDK Data Stream Protocol v1):
        s:[{...Source}, ...]\\n   # single leading source part (custom)
        0:"token"\\n             # text parts
        e:{...}\\n               # finish part
        d:{...}\\n               # done part
    """

    client = http_request.app.state.http
    table = http_request.app.state.table

    sources: list[Source] = []
    if table is not None:
        try:
            if settings.use_multi_query:
                sources = await _multi_query_retrieve(client, table, request.message)
            else:
                sources = await retrieve_hybrid(client, table, request.message)
        except Exception as exc:  # noqa: BLE001
            logger.exception("retrieve failed, falling back to base prompt: %s", exc)
            sources = []

    context = format_context(sources)
    history = [msg.model_dump() for msg in request.history] if request.history else None

    # C4: when retrieval surfaced a figure, answer with the vision model and
    # the original image(s). The explicit `request.model` override applies to
    # the text path only — the VL route is system-decided. Routing happens
    # before generation, so exactly one large model is resident per request;
    # alternating text/VL queries pay an Ollama swap (~10-30s cold).
    vl_images = load_source_images(sources) if should_route_vl(sources) else []
    use_vl = bool(vl_images)
    selected_model = (
        settings.vision_model if use_vl else (request.model or settings.llm_model)
    )
    if use_vl:
        logger.info(
            "VL route: attaching %d figure(s), model=%s", len(vl_images), selected_model
        )

    async def generate():
        source_payload = [s.model_dump() for s in sources]
        yield f"s:{json.dumps(source_payload, ensure_ascii=False)}\n"

        finish_reason = "stop"
        try:
            async for line in stream_chat(
                client,
                request.message,
                model=selected_model,
                context=context,
                history=history,
                images=vl_images or None,
                system_prompt=(
                    settings.prompt_vl_with_context.format(context=context)
                    if use_vl
                    else None
                ),
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
            "model": selected_model,
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
