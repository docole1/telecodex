from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


MAX_GROUP_TRIGGER_WORDS = 20
MAX_GROUP_TRIGGER_LENGTH = 32
_WORD_PATTERN = re.compile(r"^[^\W_]+(?:[-_][^\W_]+)*$", re.UNICODE)


def parse_trigger_words(value: str) -> tuple[str, ...]:
    """Parse and normalize a comma-separated list of group trigger words."""
    if not isinstance(value, str):
        raise ValueError("Trigger words must be text")

    words: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        word = part.strip()
        if not word:
            continue
        if len(word) > MAX_GROUP_TRIGGER_LENGTH or not _WORD_PATTERN.fullmatch(word):
            raise ValueError("Each trigger must be one word containing letters or numbers")
        normalized = word.casefold()
        if normalized in seen:
            continue
        words.append(normalized)
        seen.add(normalized)
        if len(words) > MAX_GROUP_TRIGGER_WORDS:
            raise ValueError("Too many trigger words")

    if not words:
        raise ValueError("At least one trigger word is required")
    return tuple(words)


def normalize_stored_trigger_words(value: Any) -> tuple[str, ...]:
    """Validate trigger words loaded from persistent state."""
    if not isinstance(value, list) or any(not isinstance(word, str) for word in value):
        raise ValueError("Runtime state has invalid group trigger words")
    if not value:
        return ()
    try:
        normalized = parse_trigger_words(",".join(value))
    except ValueError as exc:
        raise ValueError("Runtime state has invalid group trigger words") from exc
    return normalized


def strip_group_trigger(text: str, trigger_words: Sequence[str]) -> str | None:
    """Return the message without a matching first-word trigger, or None."""
    if not text or not trigger_words:
        return None
    alternatives = "|".join(re.escape(word) for word in sorted(trigger_words, key=len, reverse=True))
    match = re.match(rf"^\s*(?:{alternatives})(?=$|[^\w])", text, re.IGNORECASE | re.UNICODE)
    if match is None:
        return None
    return text[match.end() :].lstrip(" \t\r\n,.:;!?—–-")
