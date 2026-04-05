"""LLM-based query expansion for multi-query retrieval.

Given one user query, we ask the main LLM to produce N alternative
phrasings that cover synonyms, Korean↔English pairings, and abstracted
/ specialised forms. Each candidate is run through the hybrid retriever
and the results are fused with RRF in the chat route.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from .settings import settings

logger = logging.getLogger(__name__)


_REWRITE_SYSTEM_PROMPT = """당신은 CAD 엔지니어링 문서 검색을 돕는 쿼리 재작성기입니다.
사용자 질문을 받으면 의미는 같지만 문구/용어가 다른 검색 쿼리 {n}개를
JSON 배열로만 출력하세요.

규칙:
1. 한국어 ↔ 영어 전문용어를 섞어 다양성 확보. (예: "노즐 하중" → "nozzle load")
2. 규격 번호(ISO, ASME, DOSH 등)가 있다면 보존하거나 추가로 나열.
3. 질문을 더 구체적으로 만든 것 1개, 더 추상적으로 만든 것 1개를 포함.
4. JSON 배열(문자열 {n}개)만 출력. 다른 텍스트 금지.
"""


_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


async def expand_queries(
    client: httpx.AsyncClient,
    query: str,
    n: int | None = None,
    *,
    model: str | None = None,
) -> list[str]:
    """Return the original query plus ``n - 1`` LLM-generated variants.

    The original query is always first in the list so a multi-query
    retriever gives it full weight. Variants that fail to parse are
    silently dropped; if the LLM errors out we fall back to just the
    original query.
    """

    count = n or settings.multi_query_n
    if count <= 1 or not settings.use_multi_query:
        return [query]

    selected_model = model or settings.query_rewriter_model
    prompt = _REWRITE_SYSTEM_PROMPT.format(n=count)
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }

    try:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=httpx.Timeout(60.0),
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("query expansion failed, using original only: %s", exc)
        return [query]

    variants = _parse_variants(raw, count)
    # Ensure the original is present and first; then dedupe preserving order.
    candidates: list[str] = [query] + variants
    seen: set[str] = set()
    deduped: list[str] = []
    for q in candidates:
        key = q.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
        if len(deduped) >= count:
            break
    return deduped


def _parse_variants(raw: str, n: int) -> list[str]:
    if not raw:
        return []
    text = raw.strip()
    # Strip common markdown fences the model likes to add.
    if text.startswith("```"):
        text = text.strip("`")
        # Remove a leading "json" language hint if present.
        if text.lower().startswith("json"):
            text = text[4:]
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    out: list[str] = []
    for item in arr:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        if len(out) >= n:
            break
    return out
