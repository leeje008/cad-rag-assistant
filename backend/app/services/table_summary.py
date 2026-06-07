"""LLM table summarisation for the dual-representation table strategy.

For each parsed table we keep the original (Markdown-KV / HTML) for the
generation step and index a short retrieval-oriented summary alongside it.
Retrieval matches the summary; generation is fed the original table. The
two are linked by ``table_id``. Summaries use the light fallback model so
the per-table cost stays low during the one-off batch reindex.
"""

from __future__ import annotations

import logging

import httpx

from .settings import settings

logger = logging.getLogger(__name__)

# Tables shorter than this keep only their original chunk (a summary would
# add cost without improving recall for a trivially small table).
MIN_TABLE_CHARS = 200
# Cap the table text fed to the LLM so a huge sheet can't blow the context.
MAX_TABLE_CHARS = 3000

_SUMMARY_SYSTEM_PROMPT = """당신은 CAD 엔지니어링 문서의 표를 검색용으로 요약하는 도우미입니다.
주어진 표를 한국어 1~3문장으로 요약하세요.

규칙:
1. 표의 목적(무슨 데이터인지), 행/열 축, 단위, 대상 장비/규격을 포함하세요.
2. 개별 수치를 나열하지 말고 표가 "무엇에 대한 표인지" 검색되도록 핵심만 쓰세요.
3. 한국어/영어 전문용어는 원문 그대로 사용하세요.
4. 요약 문장만 출력하세요. 다른 설명/머리말 금지.
"""


async def summarize_table(
    client: httpx.AsyncClient,
    table_markdown: str,
    *,
    doc_title: str,
    section: str | None = None,
    model: str | None = None,
) -> str | None:
    """Return a 1-3 sentence Korean retrieval summary of a table, or None.

    Returns None when the table is too small to be worth summarising or the
    LLM call fails — callers fall back to indexing only the original table.
    """

    text = (table_markdown or "").strip()
    if len(text) < MIN_TABLE_CHARS:
        return None

    header = f"문서: {doc_title}"
    if section:
        header += f" · 섹션: {section}"
    user_content = f"{header}\n\n{text[:MAX_TABLE_CHARS]}"

    payload = {
        "model": model or settings.query_rewriter_model,
        "messages": [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=httpx.Timeout(120.0),
        )
        resp.raise_for_status()
        summary = resp.json().get("message", {}).get("content", "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("table summary failed for %s: %s", doc_title, exc)
        return None

    return summary or None
