from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import chat, documents
from .services.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own the process-wide httpx client and (Phase 1) LanceDB handles."""

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
        app.state.table = None  # populated in Phase 1 once LanceDB is wired
        logger.info(
            "Backend ready. Ollama=%s llm=%s embed=%s",
            settings.ollama_base_url,
            settings.llm_model,
            settings.embed_model,
        )
        yield
    logger.info("Backend shutdown complete")


app = FastAPI(
    title="CAD RAG Assistant API",
    description="CAD 설계 문서 기반 RAG 질의응답 백엔드",
    version="0.2.0",
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


@app.get("/")
async def root():
    return {"message": "CAD RAG Assistant API", "docs": "/docs"}
