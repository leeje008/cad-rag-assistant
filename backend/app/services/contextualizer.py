"""Contextual Retrieval: one-line section context for chunk self-containment.

Anthropic's Contextual Retrieval prepends a short situating context to each
chunk before embedding so a chunk like "허용 하중은 Fx 100kN" still carries
"which equipment / which spec" it belongs to. We generate the context once
per section (parent) and reuse it across all of that section's children, so
the LLM cost is ~1 call per section rather than per chunk.

The context is prepended to the EMBEDDING input only; the stored chunk text
stays clean so citations never show the injected prefix.
"""

from __future__ import annotations

import logging

import httpx

from .settings import settings

logger = logging.getLogger(__name__)

MAX_PARENT_CHARS = 2000  # cap parent text fed to the LLM

_CONTEXT_SYSTEM_PROMPT = """당신은 CAD 엔지니어링 문서 검색을 돕는 맥락 생성기입니다.
주어진 문서명·섹션·본문을 보고, 이 섹션이 "무엇에 대한 내용인지"를 한 줄로 요약하세요.

규칙:
1. 대상 장비/시스템과 규격(ISO/ASME/DOSH/PTS 등)을 가능하면 포함하세요.
2. 한 문장, 한 줄로만 출력하세요. 수치 나열 금지.
3. 한국어/영어 전문용어는 원문 그대로 사용하세요.
4. 맥락 문장만 출력하세요. 머리말/설명 금지.
"""


def _template_context(doc_title: str, section: str | None) -> str:
    """Deterministic, LLM-free fallback context line."""

    if section:
        return f"문서: {doc_title} · 섹션: {section}"
    return f"문서: {doc_title}"


async def section_context(
    client: httpx.AsyncClient,
    *,
    doc_title: str,
    section: str | None,
    parent_text: str,
    model: str | None = None,
) -> str:
    """Return a one-line situating context for a section.

    Never raises — on any failure it returns the deterministic template so
    the ingest batch is robust to a flaky local model.
    """

    fallback = _template_context(doc_title, section)
    if not settings.use_contextual:
        return fallback

    header = f"문서: {doc_title}"
    if section:
        header += f" · 섹션: {section}"
    user_content = f"{header}\n\n본문:\n{(parent_text or '').strip()[:MAX_PARENT_CHARS]}"

    payload = {
        "model": model or settings.query_rewriter_model,
        "messages": [
            {"role": "system", "content": _CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=httpx.Timeout(60.0),
        )
        resp.raise_for_status()
        line = resp.json().get("message", {}).get("content", "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("section context failed for %s, using template: %s", doc_title, exc)
        return fallback

    # Collapse to a single line; fall back if the model returned nothing useful.
    line = " ".join(line.split())
    return line or fallback
