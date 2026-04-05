"""Legacy config surface. Real values live in `services.settings.settings`.

Kept as thin re-export so older imports keep working.
"""

from .services.settings import (
    SYSTEM_PROMPT_BASE,
    SYSTEM_PROMPT_WITH_CONTEXT,
    settings,
)

OLLAMA_BASE_URL = settings.ollama_base_url
LLM_MODEL = settings.llm_model
EMBED_MODEL = settings.embed_model
VISION_MODEL = settings.vision_model
FALLBACK_LLM_MODEL = settings.fallback_llm_model
EMBED_DIMENSION = settings.embed_dimension

CHUNK_SIZE_PARENT = settings.chunk_max_chars
CHUNK_SIZE_CHILD = settings.chunk_min_chars
CHUNK_OVERLAP = settings.chunk_overlap
TOP_K_RETRIEVAL = settings.top_k

SPEC_DIR = str(settings.spec_dir)

# Backwards-compat system prompt (base + context directive).
SYSTEM_PROMPT = SYSTEM_PROMPT_BASE

__all__ = [
    "settings",
    "OLLAMA_BASE_URL",
    "LLM_MODEL",
    "EMBED_MODEL",
    "VISION_MODEL",
    "FALLBACK_LLM_MODEL",
    "EMBED_DIMENSION",
    "CHUNK_SIZE_PARENT",
    "CHUNK_SIZE_CHILD",
    "CHUNK_OVERLAP",
    "TOP_K_RETRIEVAL",
    "SPEC_DIR",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_BASE",
    "SYSTEM_PROMPT_WITH_CONTEXT",
]
