from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    model: str | None = None


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    models: list[str]


class ModelInfo(BaseModel):
    name: str
    size: str
    modified: str
