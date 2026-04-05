from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import chat, documents

app = FastAPI(
    title="CAD RAG Assistant API",
    description="CAD 설계 문서 기반 RAG 질의응답 백엔드",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/")
async def root():
    return {"message": "CAD RAG Assistant API", "docs": "/docs"}
