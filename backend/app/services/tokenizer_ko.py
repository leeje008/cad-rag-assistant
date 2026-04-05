"""Korean morpheme tokenizer for LanceDB FTS.

LanceDB's FTS backend (Tantivy) doesn't ship a Korean tokenizer. We pre-
tokenize documents and queries with kiwipiepy and store the result in a
separate `text_fts` column that Tantivy treats as plain whitespace tokens.

The same function MUST be used at index time and query time; otherwise
the token spaces won't align.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Morpheme tags we want to keep for retrieval. We drop particles (J*),
# endings (E*), symbols/punctuation (S*, SF, SP, ...), and foreign tags
# that only add noise.
_KEEP_PREFIXES: tuple[str, ...] = (
    "N",   # nouns (NNG, NNP, NNB, NR, NP, ...)
    "V",   # verbs (VV, VA, ...) — stems only
    "M",   # adverbs / determiners
    "SL",  # foreign language (latin words — preserves English terms)
    "SH",  # Chinese characters
    "SN",  # numbers
    "XR",  # root morpheme
)


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def tokenize_for_fts(text: str) -> str:
    """Return a whitespace-joined token string suitable for a Tantivy FTS column.

    - Keeps content morphemes (nouns, verb/adj stems, numbers, Latin terms)
    - Drops particles, endings, punctuation, and single-char symbols
    - Lower-cases everything so English terms match case-insensitively
    """

    if not text:
        return ""

    kiwi = _kiwi()
    try:
        result = kiwi.tokenize(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("kiwi tokenize failed, falling back to whitespace: %s", exc)
        return text.lower()

    tokens: list[str] = []
    for tok in result:
        form = tok.form.strip()
        tag = tok.tag
        if not form:
            continue
        if len(form) == 1 and not form.isalnum():
            continue
        if not any(tag.startswith(p) for p in _KEEP_PREFIXES):
            continue
        tokens.append(form.lower())
    return " ".join(tokens)
