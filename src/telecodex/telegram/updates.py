"""Typed Telegram updates and their narrow, side-effect-free parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from telecodex.documents.models import TelegramAttachment
from telecodex.documents.processor import AUDIO_EXTENSIONS


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


@dataclass(frozen=True)
class IncomingCallback:
    update_id: int
    callback_id: str
    user_id: int
    chat_id: int
    message_id: int
    data: str


IncomingUpdate = IncomingMessage | IncomingCallback


def parse_update(update: dict[str, Any]) -> IncomingUpdate | None:
    update_id = update.get("update_id")
    if not isinstance(update_id, int):
        return None

    message = update.get("message")
    if isinstance(message, dict):
        user = message.get("from", {})
        chat = message.get("chat", {})
        text = message.get("text")
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
