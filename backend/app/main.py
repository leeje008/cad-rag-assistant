from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import chat, documents, ingest, retrieve
from .services.settings import settings
from .services.vectorstore import open_or_create_table

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own the process-wide httpx client and LanceDB table handle."""

    timeout = httpx.Timeout(
        connect=settings.ollama_connect_timeout,
        read=settings.ollama_read_timeout,
        write=settings.ollama_write_timeout,
        pool=settings.ollama_pool_timeout,
    )
    limits = httpx.Limits(
        max_keepalive_connections=20,
        max_connections=40,
        keepalive_expiry=60.0,
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        app.state.http = client
        try:
            app.state.table = open_or_create_table()
        except Exception as exc:  # noqa: BLE001
            logger.exception("LanceDB init failed: %s", exc)
            app.state.table = None

        logger.info(
            "Backend ready. Ollama=%s llm=%s embed=%s lancedb=%s",
            settings.ollama_base_url,
            settings.llm_model,
            settings.embed_model,
            settings.lancedb_path,
        )
        yield
    logger.info("Backend shutdown complete")


app = FastAPI(
    title="CAD RAG Assistant API",
    description="CAD 설계 문서 기반 RAG 질의응답 백엔드",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(ingest.router)
app.include_router(retrieve.router)


@app.get("/")
async def root():
    return {"message": "CAD RAG Assistant API", "docs": "/docs"}
