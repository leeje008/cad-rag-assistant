from __future__ import annotations

from fastapi import APIRouter, Request

from ..models.schemas import HealthResponse
from ..services.llm import check_ollama_health, list_models

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health(request: Request):
    client = request.app.state.http
    ollama_ok = await check_ollama_health(client)
    models: list[str] = []
    if ollama_ok:
        try:
            raw = await list_models(client)
            models = [m["name"] for m in raw]
        except Exception:  # noqa: BLE001
            models = []

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama=ollama_ok,
        models=models,
    )


@router.get("/api/models")
async def get_models(request: Request):
    client = request.app.state.http
    models = await list_models(client)
    return {
        "models": [
            {"name": m["name"], "size": str(m.get("size", ""))} for m in models
        ]
    }
