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


SYSTEM_PROMPT_VL_WITH_CONTEXT = """당신은 CAD 설계 문서 기반 기술 질의응답 전문 AI 어시스턴트입니다.
아래 [CONTEXT]의 정보와 첨부된 이미지(검색된 도면/그림 원본)만을 근거로 답변하세요.

규칙:
1. [CONTEXT]와 이미지에 없는 내용은 추측하지 말고 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
2. 가능하면 답변 안에 인용 번호를 `[1]`, `[2]` 형태로 표기하고, 이미지를 근거로 한 내용은 "(그림 [n])" 형태로 표기하세요 (번호는 [CONTEXT] 항목 순서).
3. 이미지에서 판독할 수 없는 내용은 추측하지 말고 "이미지에서 확인 불가"라고 명시하세요.
4. 텍스트 컨텍스트와 이미지 내용이 상충하면 둘 다 언급하세요.
5. 기술 용어는 원문 그대로 사용하세요 (한국어/영어 혼용 가능).
6. 테이블 데이터는 가능한 한 표 형식으로 제시하세요.
7. 답변은 구조적으로 작성하세요 (번호/불릿 활용).

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
    embed_model: str = "bge-m3"
    vision_model: str = "qwen2.5vl"
    fallback_llm_model: str = "llama3.1:8b"
    embed_dimension: int = 1024
    embed_batch: int = 16

    # Retrieval — Phase 2
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_hybrid: bool = True
    use_reranker: bool = True
    rerank_candidate_n: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    multi_query_n: int = 3
    use_multi_query: bool = True
    # Query rewriting is a light task; prefer a fast fallback model so it
    # doesn't block the hybrid retrieval path behind a 27B warmup.
    query_rewriter_model: str = "llama3.1:8b"

    # Ollama timeouts (seconds)
    ollama_connect_timeout: float = 5.0
    ollama_read_timeout: float = 300.0
    ollama_write_timeout: float = 30.0
    ollama_pool_timeout: float = 5.0

    # Parsing — Phase A (Docling)
    use_docling: bool = True  # Docling primary, pymupdf4llm fallback on failure
    docling_do_ocr: bool = True  # OCR scanned P&ID/drawings (slow, no GPU)
    docling_images_scale: float = 2.0  # render scale for extracted figure images
    # MinerU option for CJK multi-header docs — not implemented this round.

    # RAG — chunker
    chunk_max_chars: int = 1500
    chunk_min_chars: int = 400
    chunk_overlap: int = 150
    top_k: int = 5

    # Ingest enrichment (LLM stages — one-off batch cost)
    use_table_summary: bool = True  # PR4: dual table representation
    use_contextual: bool = True  # PR5: per-section contextual prepend
    use_vlm_caption: bool = True  # PR6: VLM image captions

    # Multimodal — Phase C
    persist_figure_images: bool = True  # C1: keep figure PNGs for citation/VL input
    # C4: route generation to the vision model when retrieval surfaces figures.
    # Ships off; promote to True once the C5 eval gate passes.
    use_vl_answer: bool = False
    vl_max_images: int = 3  # cap on figures attached to a VL request
    # C6 pilot: ColQwen page-image retrieval over a hand-picked drawing
    # corpus (see scripts/build_colqwen_index.py). Needs the optional
    # colpali-engine/torch deps; keep off unless the index is built.
    use_colqwen_pages: bool = False
    colqwen_model: str = "vidore/colqwen2-v1.0"
    colqwen_top_k: int = 3
    colqwen_dir: Path = _BACKEND_DIR / ".colqwen"

    # Storage paths
    spec_dir: Path = _REPO_ROOT / "SPEC"
    lancedb_path: Path = _BACKEND_DIR / ".lancedb"
    lancedb_table: str = "cad_chunks"
    assets_dir: Path = _BACKEND_DIR / ".assets"

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

    @property
    def prompt_vl_with_context(self) -> str:
        return SYSTEM_PROMPT_VL_WITH_CONTEXT


settings = Settings()
