"""Centralised runtime settings (pydantic-settings).

Single source of truth for every tunable used by the backend.
Import `settings` anywhere; legacy `config.py` re-exports the same values
for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_DIR.parent


SYSTEM_PROMPT_BASE = """당신은 CAD 설계 문서 기반 기술 질의응답 전문 AI 어시스턴트입니다.

규칙:
1. 모르는 내용은 추측하지 말고 "관련 문서를 찾지 못했습니다"라고 답하세요.
2. 기술 용어는 원문 그대로 사용하세요 (한국어/영어 혼용 가능).
3. 답변은 구조적으로 작성하세요 (번호/불릿 활용).
"""


SYSTEM_PROMPT_WITH_CONTEXT = """당신은 CAD 설계 문서 기반 기술 질의응답 전문 AI 어시스턴트입니다.
아래 [CONTEXT] 안의 정보만을 근거로 답변하세요.

규칙:
1. [CONTEXT]에 없는 내용은 추측하지 말고 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
2. 가능하면 답변 안에 인용 번호를 `[1]`, `[2]` 형태로 표기하세요 (번호는 [CONTEXT] 항목 순서).
3. 기술 용어는 원문 그대로 사용하세요 (한국어/영어 혼용 가능).
4. 테이블 데이터는 가능한 한 표 형식으로 제시하세요.
5. 답변은 구조적으로 작성하세요 (번호/불릿 활용).

[CONTEXT]
{context}
"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:27b"
    embed_model: str = "mxbai-embed-large"
    vision_model: str = "gemma4:e4b"
    fallback_llm_model: str = "llama3.1:8b"
    embed_dimension: int = 1024
    embed_batch: int = 16

    # Ollama timeouts (seconds)
    ollama_connect_timeout: float = 5.0
    ollama_read_timeout: float = 300.0
    ollama_write_timeout: float = 30.0
    ollama_pool_timeout: float = 5.0

    # RAG — chunker
    chunk_max_chars: int = 1500
    chunk_min_chars: int = 400
    chunk_overlap: int = 150
    top_k: int = 5

    # Storage paths
    spec_dir: Path = _REPO_ROOT / "SPEC"
    lancedb_path: Path = _BACKEND_DIR / ".lancedb"
    lancedb_table: str = "cad_chunks"

    # CORS
    cors_allow_origins: list[str] = ["http://localhost:3000"]
    cors_allow_origin_regex: str = r"https://([a-z0-9-]+\.)*vercel\.app"

    # Misc
    ingest_max_file_mb: int = 200

    @property
    def prompt_base(self) -> str:
        return SYSTEM_PROMPT_BASE

    @property
    def prompt_with_context(self) -> str:
        return SYSTEM_PROMPT_WITH_CONTEXT


settings = Settings()
