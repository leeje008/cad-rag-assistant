"""VLM image captioning (1차 multimodal).

Each parsed figure/drawing is captioned in Korean by a local vision model
via Ollama and the caption is indexed as the image chunk's text in the
existing dense+FTS index — making drawings/P&IDs searchable with minimal
stack change. The original image bytes are NOT stored (page/bbox/figure_id
persist for a future ColQwen page-image round).

Anti-hallucination: the prompt forces "판독 불가" for unreadable regions,
and captions that come back empty/too short are replaced with a locatable
placeholder rather than indexed as confident-but-wrong text.
"""

from __future__ import annotations

import logging

import httpx

from .settings import settings

logger = logging.getLogger(__name__)

# Below this length a caption is treated as a non-answer (model refusal,
# blank, or drift) and swapped for the deterministic placeholder.
MIN_CAPTION_CHARS = 15

_CAPTION_PROMPT = """이 엔지니어링 도면/이미지를 한국어로 설명하세요.
포함할 내용: 도면 유형(P&ID / 노즐 / 체크리스트 / 심볼 범례 등), 보이는 라벨·기호·수치, 대상 장비.
이미지에서 판독할 수 없는 내용은 추측하지 말고 '판독 불가'로 표기하세요.
설명 문장만 출력하세요."""


def placeholder_caption(doc_title: str, section: str | None, page: int | None) -> str:
    """Locatable fallback when captioning fails or drifts."""

    loc = f"[이미지] {doc_title}"
    if section:
        loc += f" · {section}"
    if page is not None and page >= 0:
        loc += f" (p.{page})"
    return f"{loc} — 캡션 생성 실패, 원문 확인 필요"


async def caption_image(
    client: httpx.AsyncClient,
    image_b64: str,
    *,
    model: str | None = None,
    context_hint: str | None = None,
) -> str | None:
    """Return a Korean caption for a base64 PNG, or None on failure.

    None signals the caller to fall back to a placeholder (or skip). The
    call never raises.
    """

    if not image_b64:
        return None

    prompt = _CAPTION_PROMPT
    if context_hint:
        prompt = f"{prompt}\n참고 맥락: {context_hint}"

    payload = {
        "model": model or settings.vision_model,
        "messages": [
            {"role": "user", "content": prompt, "images": [image_b64]},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=httpx.Timeout(180.0),
        )
        resp.raise_for_status()
        caption = resp.json().get("message", {}).get("content", "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("VLM caption failed: %s", exc)
        return None

    caption = " ".join(caption.split())
    if len(caption) < MIN_CAPTION_CHARS:
        return None
    return caption
