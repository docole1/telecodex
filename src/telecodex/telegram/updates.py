"""Typed Telegram updates and their narrow, side-effect-free parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiogram.types import Update

from telecodex.documents.models import TelegramAttachment
from telecodex.documents.processor import AUDIO_EXTENSIONS
from telecodex.telegram.markdown_input import slice_entities, to_model_markdown, to_model_markdown_slice


@dataclass(frozen=True)
class IncomingMessage:
    update_id: int
    message_id: int
    user_id: int
    chat_id: int
    chat_type: str
    text: str
    attachments: tuple[TelegramAttachment, ...] = ()
    media_group_id: str | None = None
    chat_title: str | None = None
    group_trigger_checked: bool = False
    group_trigger_required: bool = False
    reply_to_user_id: int | None = None
    model_text: str | None = None
    entities: tuple[dict[str, Any], ...] = ()

    @property
    def prompt_text(self) -> str:
        return self.model_text if self.model_text is not None else to_model_markdown(self.text, self.entities)

    def without_prefix(self, prefix_length: int) -> IncomingMessage:
        text, entities = slice_entities(self.text, self.entities, prefix_length)
        return IncomingMessage(
            update_id=self.update_id,
            message_id=self.message_id,
            user_id=self.user_id,
            chat_id=self.chat_id,
            chat_type=self.chat_type,
            text=text,
            attachments=self.attachments,
            media_group_id=self.media_group_id,
            chat_title=self.chat_title,
            group_trigger_checked=self.group_trigger_checked,
            group_trigger_required=self.group_trigger_required,
            reply_to_user_id=self.reply_to_user_id,
            model_text=to_model_markdown(text, entities),
            entities=entities,
        )

    def command_argument_markdown(self) -> str:
        _command, separator, _argument = self.text.partition(" ")
        if not separator:
            return ""
        start = len(_command) + 1
        end = len(self.text.rstrip())
        while start < end and self.text[start].isspace():
            start += 1
        return to_model_markdown_slice(self.text, self.entities, start, end)


@dataclass(frozen=True)
class IncomingCallback:
    update_id: int
    callback_id: str
    user_id: int
    chat_id: int
    message_id: int
    data: str


IncomingUpdate = IncomingMessage | IncomingCallback


def parse_update(
    update: dict[str, Any],
    *,
    model_text: str | None = None,
) -> IncomingUpdate | None:
    update_id = update.get("update_id")
    if not isinstance(update_id, int):
        return None

    message = update.get("message")
    if isinstance(message, dict):
        user = message.get("from", {})
        chat = message.get("chat", {})
        text = message.get("text")
        entities_value = message.get("entities")
        user_id = user.get("id") if isinstance(user, dict) else None
        chat_id = chat.get("id") if isinstance(chat, dict) else None
        message_id = message.get("message_id")
        chat_type = chat.get("type") if isinstance(chat, dict) else None
        chat_title = chat.get("title") if isinstance(chat, dict) else None
        reply = message.get("reply_to_message")
        reply_user = reply.get("from") if isinstance(reply, dict) else None
        reply_to_user_id = reply_user.get("id") if isinstance(reply_user, dict) else None
        attachments = _parse_message_attachments(message)
        caption = message.get("caption")
        if not isinstance(text, str):
            text = caption if isinstance(caption, str) else ""
            entities_value = message.get("caption_entities")
        entities = tuple(
            dict(entity)
            for entity in entities_value
            if isinstance(entity, dict)
        ) if isinstance(entities_value, list) else ()
        media_group_id = message.get("media_group_id")
        if not isinstance(media_group_id, str):
            media_group_id = None
        if (
            all(isinstance(value, int) for value in (message_id, user_id, chat_id))
            and (isinstance(text, str) or attachments)
        ):
            return IncomingMessage(
                update_id,
                message_id,
                user_id,
                chat_id,
                str(chat_type or ""),
                text,
                attachments,
                media_group_id,
                chat_title if isinstance(chat_title, str) else None,
                False,
                False,
                reply_to_user_id if isinstance(reply_to_user_id, int) else None,
                model_text if model_text is not None else to_model_markdown(text, entities),
                entities,
            )
        return None

    callback = update.get("callback_query")
    if isinstance(callback, dict):
        user = callback.get("from", {})
        message = callback.get("message", {})
        chat = message.get("chat", {}) if isinstance(message, dict) else {}
        user_id = user.get("id") if isinstance(user, dict) else None
        chat_id = chat.get("id") if isinstance(chat, dict) else None
        message_id = message.get("message_id") if isinstance(message, dict) else None
        callback_id = callback.get("id")
        data = callback.get("data")
        if (
            isinstance(user_id, int)
            and isinstance(chat_id, int)
            and isinstance(message_id, int)
            and isinstance(callback_id, str)
            and isinstance(data, str)
        ):
            return IncomingCallback(update_id, callback_id, user_id, chat_id, message_id, data)
    return None


def parse_aiogram_update(
    update: Update,
    *,
    model_text: str | None = None,
) -> IncomingUpdate | None:
    """Adapt an Aiogram update to TeleCodex's framework-neutral DTOs."""
    raw_update = aiogram_update_to_dict(update)
    return parse_update(raw_update, model_text=model_text)


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
