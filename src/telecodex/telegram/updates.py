"""Typed Telegram updates and their narrow, side-effect-free parsing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from aiogram.types import CallbackQuery, Message, Update

from telecodex.documents.models import TelegramAttachment
from telecodex.documents.processor import AUDIO_EXTENSIONS
from telecodex.telegram.markdown_input import slice_entities, to_model_markdown, to_model_markdown_slice


@dataclass(frozen=True, slots=True)
class TelegramMessageContext:
    """TeleCodex metadata carried through Aiogram handler data."""

    event: Message
    update_id: int
    model_text: str
    attachments: tuple[TelegramAttachment, ...] = ()
    group_trigger_checked: bool = False
    group_trigger_required: bool = False
    text_override: str | None = None
    entities_override: tuple[Any, ...] | None = None

    @property
    def message_id(self) -> int:
        return self.event.message_id

    @property
    def user_id(self) -> int:
        return self.event.from_user.id  # type: ignore[union-attr]

    @property
    def chat_id(self) -> int:
        return self.event.chat.id

    @property
    def chat_type(self) -> str:
        return str(self.event.chat.type)

    @property
    def chat_title(self) -> str | None:
        return self.event.chat.title

    @property
    def media_group_id(self) -> str | None:
        return self.event.media_group_id

    @property
    def text(self) -> str:
        if self.text_override is not None:
            return self.text_override
        return self.event.text if self.event.text is not None else self.event.caption or ""

    @property
    def entities(self) -> tuple[Any, ...]:
        if self.entities_override is not None:
            return self.entities_override
        values = self.event.entities if self.event.text is not None else self.event.caption_entities
        return tuple(values or ())

    @property
    def reply_to_user_id(self) -> int | None:
        reply = self.event.reply_to_message
        return reply.from_user.id if reply is not None and reply.from_user is not None else None

    @property
    def prompt_text(self) -> str:
        return self.model_text

    def without_prefix(self, prefix_length: int) -> TelegramMessageContext:
        text, entities = slice_entities(self.text, self.entities, prefix_length)
        prompt_text = to_model_markdown(text, entities)
        return replace(
            self,
            text_override=text,
            entities_override=tuple(entities),
            model_text=prompt_text,
        )


@dataclass(frozen=True, slots=True)
class TelegramCallbackContext:
    """TeleCodex callback metadata carried through Aiogram handler data."""

    event: CallbackQuery
    update_id: int

    @property
    def callback_id(self) -> str:
        return self.event.id

    @property
    def user_id(self) -> int:
        return self.event.from_user.id

    @property
    def chat_id(self) -> int:
        message = self.event.message
        if message is None:
            raise ValueError("Callback query has no accessible chat message")
        return message.chat.id

    @property
    def message_id(self) -> int:
        message = self.event.message
        if message is None:
            raise ValueError("Callback query has no accessible chat message")
        return message.message_id

    @property
    def data(self) -> str:
        return self.event.data or ""


TelegramEventContext = TelegramMessageContext | TelegramCallbackContext


def parse_update(
    update: dict[str, Any],
    *,
    model_text: str | None = None,
) -> TelegramEventContext | None:
    try:
        typed_update = Update.model_validate(_normalize_update_for_aiogram(update))
    except Exception:
        return None
    return parse_aiogram_update(typed_update, model_text=model_text)


def parse_aiogram_update(
    update: Update,
    *,
    model_text: str | None = None,
) -> TelegramEventContext | None:
    """Build TeleCodex metadata separately from the original Aiogram event."""
    if not isinstance(update.update_id, int):
        return None
    message = update.message
    if message is not None:
        user_id = message.from_user.id if message.from_user is not None else None
        text = message.text if message.text is not None else message.caption or ""
        entities = message.entities if message.text is not None else message.caption_entities
        raw_message = message.model_dump(mode="json", by_alias=True, exclude_none=True)
        attachments = _parse_message_attachments(raw_message)
        if user_id is None or (not text and not attachments):
            return None
        markdown = model_text if model_text is not None else to_model_markdown(text, entities)
        return TelegramMessageContext(
            event=message,
            update_id=update.update_id,
            model_text=markdown,
            attachments=attachments,
        )

    callback = update.callback_query
    if callback is None or callback.data is None:
        return None
    message = callback.message
    if message is None:
        return None
    return TelegramCallbackContext(event=callback, update_id=update.update_id)


def command_argument_markdown(message: TelegramMessageContext) -> str:
    text = message.text
    _command, separator, _argument = text.partition(" ")
    if not separator:
        return ""
    start = len(_command) + 1
    end = len(text.rstrip())
    while start < end and text[start].isspace():
        start += 1
    return to_model_markdown_slice(text, message.entities, start, end)


def without_message_prefix(
    message: TelegramMessageContext,
    prefix_length: int,
) -> TelegramMessageContext:
    return message.without_prefix(prefix_length)


def copy_message(message: TelegramMessageContext, **updates: Any) -> TelegramMessageContext:
    return replace(message, **updates)


def _normalize_update_for_aiogram(update: dict[str, Any]) -> dict[str, Any]:
    """Fill fields required by Aiogram when reading older minimal test fixtures."""
    normalized = deepcopy(update)

    def normalize_message(message: Any) -> None:
        if not isinstance(message, dict):
            return
        message.setdefault("date", 0)
        sender = message.get("from")
        if isinstance(sender, dict):
            sender.setdefault("is_bot", False)
            sender.setdefault("first_name", "Telegram user")
        for field in ("voice", "audio", "document"):
            media = message.get(field)
            if isinstance(media, dict):
                media.setdefault("file_unique_id", media.get("file_id", "test-file"))
        photos = message.get("photo")
        if isinstance(photos, list):
            for photo in photos:
                if isinstance(photo, dict):
                    photo.setdefault("file_unique_id", photo.get("file_id", "test-photo"))
        normalize_message(message.get("reply_to_message"))

    normalize_message(normalized.get("message"))
    callback = normalized.get("callback_query")
    if isinstance(callback, dict):
        sender = callback.get("from")
        if isinstance(sender, dict):
            sender.setdefault("is_bot", False)
            sender.setdefault("first_name", "Telegram user")
        callback.setdefault("chat_instance", "test-chat-instance")
        normalize_message(callback.get("message"))
    return normalized


def aiogram_update_to_dict(update: Update) -> dict[str, Any]:
    """Serialize a typed update into Bot API JSON-compatible fields."""
    return update.model_dump(mode="json", by_alias=True, exclude_none=True)


def parse_command(text: str) -> tuple[str, str] | None:
    if not text.startswith("/"):
        return None
    first, _, remainder = text.partition(" ")
    command = first[1:].split("@", 1)[0].lower()
    return command, remainder.strip()


def _parse_message_attachments(message: dict[str, Any]) -> tuple[TelegramAttachment, ...]:
    voice = message.get("voice")
    if isinstance(voice, dict) and isinstance(voice.get("file_id"), str):
        file_size = voice.get("file_size")
        duration = voice.get("duration")
        mime_type = voice.get("mime_type")
        return (
            TelegramAttachment(
                file_id=voice["file_id"],
                kind="voice",
                file_size=file_size if isinstance(file_size, int) else None,
                file_name="voice.ogg",
                mime_type=mime_type if isinstance(mime_type, str) else "audio/ogg",
                duration_seconds=duration if isinstance(duration, int) else None,
            ),
        )

    audio = message.get("audio")
    if isinstance(audio, dict) and isinstance(audio.get("file_id"), str):
        file_size = audio.get("file_size")
        duration = audio.get("duration")
        file_name = audio.get("file_name")
        mime_type = audio.get("mime_type")
        return (
            TelegramAttachment(
                file_id=audio["file_id"],
                kind="audio",
                file_size=file_size if isinstance(file_size, int) else None,
                file_name=file_name if isinstance(file_name, str) else None,
                mime_type=mime_type if isinstance(mime_type, str) else None,
                duration_seconds=duration if isinstance(duration, int) else None,
            ),
        )

    document = message.get("document")
    if isinstance(document, dict) and isinstance(document.get("file_id"), str):
        file_size = document.get("file_size")
        file_name = document.get("file_name")
        mime_type = document.get("mime_type")
        is_audio = (
            isinstance(mime_type, str) and mime_type.lower().startswith("audio/")
        ) or (isinstance(file_name, str) and Path(file_name).suffix.lower() in AUDIO_EXTENSIONS)
        return (
            TelegramAttachment(
                file_id=document["file_id"],
                kind="audio" if is_audio else "document",
                file_size=file_size if isinstance(file_size, int) else None,
                file_name=file_name if isinstance(file_name, str) else None,
                mime_type=mime_type if isinstance(mime_type, str) else None,
            ),
        )

    photos = message.get("photo")
    if isinstance(photos, list):
        candidates = [
            photo
            for photo in photos
            if isinstance(photo, dict) and isinstance(photo.get("file_id"), str)
        ]
        if candidates:
            photo = max(candidates, key=lambda item: item.get("file_size", 0))
            file_size = photo.get("file_size")
            return (
                TelegramAttachment(
                    file_id=photo["file_id"],
                    kind="photo",
                    file_size=file_size if isinstance(file_size, int) else None,
                    file_name="photo.jpg",
                    mime_type="image/jpeg",
                ),
            )
    return ()
