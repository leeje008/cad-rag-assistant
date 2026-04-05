from fastapi import APIRouter

from ..services.llm import list_models, check_ollama_health
from ..models.schemas import HealthResponse

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    ollama_ok = await check_ollama_health()
    models = []
    if ollama_ok:
        model_list = await list_models()
        models = [m["name"] for m in model_list]

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama=ollama_ok,
        models=models,
    )


@router.get("/api/models")
async def get_models():
    """List available Ollama models."""
    models = await list_models()
    return {"models": [{"name": m["name"], "size": str(m.get("size", ""))} for m in models]}
