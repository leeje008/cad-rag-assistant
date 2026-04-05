from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    model: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    models: list[str]


class ModelInfo(BaseModel):
    name: str
    size: str
    modified: str = ""


class Source(BaseModel):
    document: str
    page: int | None = None
    section: str | None = None
    relevance: float | None = None
    text: str = ""
    source_path: str = ""
    chunk_id: str = ""


class IngestRequest(BaseModel):
    path: str
    recursive: bool = True


class IngestJobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]
    processed: int = 0
    total: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None


class RetrieveRequest(BaseModel):
    query: str
    k: int | None = None


class RetrieveResponse(BaseModel):
    sources: list[Source]
