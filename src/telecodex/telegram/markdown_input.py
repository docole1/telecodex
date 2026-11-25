"""Build model-ready Markdown from Telegram's separate text entities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, MessageEntity, TelegramObject
from aiogram.utils.text_decorations import MarkdownDecoration


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _longest_backtick_run(value: str) -> int:
    longest = 0
    current = 0
    for character in value:
        if character == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _code_fence(value: str, language: str | None = None) -> str:
    fence = "`" * max(3, _longest_backtick_run(value) + 1)
    safe_language = language.strip() if language and language.strip().replace("_", "a").isalnum() else ""
    return f"{fence}{safe_language}\n{value}\n{fence}"


class _ModelMarkdownDecoration(MarkdownDecoration):
    """Use model-friendly markers while preserving the user's visible text."""

    def quote(self, value: str) -> str:
        return value

    def bold(self, value: str) -> str:
        return f"**{value}**"

    def italic(self, value: str) -> str:
        return f"*{value}*"

    def underline(self, value: str) -> str:
        return f"<u>{value}</u>"

    def strikethrough(self, value: str) -> str:
        return f"~~{value}~~"

    def code(self, value: str) -> str:
        fence = "`" * (_longest_backtick_run(value) + 1)
        return f"{fence}{value}{fence}"

    def pre(self, value: str) -> str:
        return _code_fence(value)

    def pre_language(self, value: str, language: str) -> str:
        return _code_fence(value, language)

    def spoiler(self, value: str) -> str:
        return f"||{value}||"

    def blockquote(self, value: str) -> str:
        return "\n".join(f"> {line}" for line in value.splitlines())

    def expandable_blockquote(self, value: str) -> str:
        return "\n".join(f"> {line}" for line in value.splitlines())

    def custom_emoji(self, value: str, custom_emoji_id: str) -> str:
        return value

    def date_time(
        self,
        value: str,
        unix_time: int | datetime,
        date_time_format: str | None = None,
    ) -> str:
        return value


_DECORATION = _ModelMarkdownDecoration()


def _validated_entities(
    entities: Sequence[MessageEntity | dict[str, Any]] | None,
) -> list[MessageEntity]:
    result: list[MessageEntity] = []
    for entity in entities or ():
        try:
            result.append(
                entity if isinstance(entity, MessageEntity) else MessageEntity.model_validate(entity)
            )
        except (TypeError, ValueError):
            continue
    return result


def _entity_slice(
    text: str,
    entities: Sequence[MessageEntity | dict[str, Any]] | None,
    start: int,
    end: int,
) -> tuple[str, list[MessageEntity]]:
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    start_units = _utf16_length(text[:start])
    end_units = _utf16_length(text[:end])
    sliced_entities: list[MessageEntity] = []
    for entity in _validated_entities(entities):
        entity_start = entity.offset
        entity_end = entity.offset + entity.length
        overlap_start = max(entity_start, start_units)
        overlap_end = min(entity_end, end_units)
        if overlap_end <= overlap_start:
            continue
        sliced_entities.append(
            entity.model_copy(
                update={
                    "offset": overlap_start - start_units,
                    "length": overlap_end - overlap_start,
                }
            )
        )
    return text[start:end], sliced_entities


def slice_entities(
    text: str,
    entities: Sequence[MessageEntity | dict[str, Any]] | None,
    start: int,
    end: int | None = None,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Slice visible text and clip/rebase entities to that slice."""
    sliced_text, sliced_entities = _entity_slice(
        text,
        entities,
        start,
        len(text) if end is None else end,
    )
    return sliced_text, tuple(
        entity.model_dump(mode="python", by_alias=True, exclude_none=True)
        for entity in sliced_entities
    )


def to_model_markdown(
    text: str | None,
    entities: Sequence[MessageEntity | dict[str, Any]] | None = None,
) -> str:
    """Restore entity formatting; Telegram offsets are measured in UTF-16 units."""
    if not text:
        return text or ""
    parsed_entities = _validated_entities(entities)
    if not parsed_entities:
        return text
    return _DECORATION.unparse(text=text, entities=parsed_entities)


def to_model_markdown_slice(
    text: str,
    entities: Sequence[MessageEntity | dict[str, Any]] | None,
    start: int,
    end: int | None = None,
) -> str:
    sliced_text, sliced_entities = _entity_slice(text, entities, start, len(text) if end is None else end)
    return to_model_markdown(sliced_text, sliced_entities)


class MarkdownInputMiddleware(BaseMiddleware):
    """Expose a model-formatted view without modifying routing text or captions."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Message):
            if event.text is not None:
                text, entities = event.text, event.entities
            elif event.caption is not None:
                text, entities = event.caption, event.caption_entities
            else:
                text, entities = "", None
            data["telecodex_model_text"] = to_model_markdown(text, entities)
        return await handler(event, data)
