import os

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model Configuration (mapped to locally installed models)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:27b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
VISION_MODEL = os.getenv("VISION_MODEL", "gemma4:e4b")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "llama3.1:8b")

# Embedding dimensions
EMBED_DIMENSION = 1024  # mxbai-embed-large output dimension

# CORS - allowed origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://*.vercel.app",
]

# Document paths
SPEC_DIR = os.getenv("SPEC_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "SPEC"))

# RAG Configuration
CHUNK_SIZE_PARENT = 1500
CHUNK_SIZE_CHILD = 400
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 5

# System prompt for RAG
SYSTEM_PROMPT = """당신은 CAD 설계 문서 기반 기술 질의응답 전문 AI 어시스턴트입니다.
제공된 컨텍스트 정보만을 기반으로 정확하게 답변하세요.

규칙:
1. 컨텍스트에 없는 정보는 추측하지 마세요. "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
2. 모든 주장에 출처를 명시하세요: [Source: 문서명, p.N] 형식
3. 기술 용어는 원문 그대로 사용하세요 (한국어/영어 혼용 가능)
4. 테이블 데이터는 가능한 표 형식으로 제시하세요.
5. 답변은 구조적으로 작성하세요 (번호/불릿 포인트 활용).
"""
