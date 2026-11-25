from __future__ import annotations

import asyncio
import importlib
import logging
import mimetypes
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from telecodex.documents.models import PreparedAttachment, TelegramAttachment
from telecodex.logging_config import log_exception
from telecodex.telegram.api import (
    TelegramFileDownloadError,
    TelegramFileStorageError,
    TelegramFileTooLargeError,
    download_telegram_file,
)

logger = logging.getLogger(__name__)

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_EXTRACTED_MARKDOWN_BYTES = 4 * 1024 * 1024
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".oga", ".ogg", ".opus", ".wav", ".webm"}


class AttachmentProcessingError(RuntimeError):
    def __init__(self, text_key: str, *, filename: str = "") -> None:
        super().__init__(text_key)
        self.text_key = text_key
        self.filename = filename


async def prepare_attachments(
    bot: Bot,
    attachments: Sequence[TelegramAttachment],
    working_directory: Path,
    user_id: int,
) -> list[PreparedAttachment]:
    attachment_directory = _attachment_directory(working_directory, user_id)
    prepared: list[PreparedAttachment] = []

    for attachment in attachments:
        name = _display_name(attachment)
        if attachment.file_size is not None and attachment.file_size > MAX_ATTACHMENT_BYTES:
            raise AttachmentProcessingError("documents.label.too_large", filename=name)

        path = attachment_directory / _stored_name(name)
        try:
            await download_telegram_file(
                bot,
                attachment.file_id,
                path,
                max_bytes=MAX_ATTACHMENT_BYTES,
            )
        except TelegramFileTooLargeError as exc:
            path.unlink(missing_ok=True)
            raise AttachmentProcessingError("documents.label.too_large", filename=name) from exc
        except TelegramFileStorageError as exc:
            path.unlink(missing_ok=True)
            raise AttachmentProcessingError("documents.label.save_failed", filename=name) from exc
        except TelegramFileDownloadError as exc:
            path.unlink(missing_ok=True)
            raise AttachmentProcessingError("documents.label.download_failed", filename=name) from exc
        except TelegramAPIError as exc:
            path.unlink(missing_ok=True)
            raise AttachmentProcessingError("documents.label.download_failed", filename=name) from exc

        if _is_image(attachment, path):
            prepared.append(
                PreparedAttachment(name, str(path), str(path), is_image=True)
            )
            continue

        if _is_audio(attachment, path):
            prepared.append(
                PreparedAttachment(name, str(path), str(path), is_image=False, is_audio=True)
            )
            continue

        if path.suffix.lower() in DOCUMENT_EXTENSIONS:
            markdown = await _convert_document(path, name)
            if not markdown.strip():
                raise AttachmentProcessingError("documents.label.empty", filename=name)
            encoded = markdown.encode("utf-8")
            if len(encoded) > MAX_EXTRACTED_MARKDOWN_BYTES:
                encoded = encoded[:MAX_EXTRACTED_MARKDOWN_BYTES]
                markdown = encoded.decode("utf-8", errors="ignore")
                markdown += "\n\n[TeleCodex: extracted text was limited to 4 MiB.]\n"
            markdown_path = path.with_name(f"{path.name}.md")
            try:
                markdown_path.write_text(markdown, encoding="utf-8")
                markdown_path.chmod(0o600)
            except OSError as exc:
                raise AttachmentProcessingError("documents.label.save_failed", filename=name) from exc
            prepared.append(
                PreparedAttachment(name, str(path), str(markdown_path), is_image=False)
            )
            continue

        prepared.append(PreparedAttachment(name, str(path), str(path), is_image=False))

    return prepared


def _attachment_directory(working_directory: Path, user_id: int) -> Path:
    cwd = working_directory.resolve()
    base = cwd / ".telecodex"
    root = base / "attachments"
    user_directory = root / str(user_id)
    try:
        for directory in (base, root, user_directory):
            if directory.is_symlink():
                raise AttachmentProcessingError("documents.label.save_failed")
            directory.mkdir(mode=0o700, parents=False, exist_ok=True)
            try:
                directory.resolve().relative_to(cwd)
            except ValueError as exc:
                raise AttachmentProcessingError("documents.label.save_failed") from exc
            directory.chmod(0o700)
    except OSError as exc:
        raise AttachmentProcessingError("documents.label.save_failed") from exc
    return user_directory


def _stored_name(name: str) -> str:
    # Telegram file names are display metadata, never a path.
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    basename = "".join(char for char in basename if unicodedata.category(char)[0] != "C")
    basename = re.sub(r"\s+", "_", basename).strip("._ ")[:100] or "attachment"
    return f"{uuid.uuid4().hex}-{basename}"


def _display_name(attachment: TelegramAttachment) -> str:
    if attachment.file_name:
        basename = attachment.file_name.replace("\\", "/").rsplit("/", 1)[-1]
        basename = "".join(char for char in basename if unicodedata.category(char)[0] != "C")
        return basename[:120] or "attachment"
    if attachment.kind == "photo":
        return "photo.jpg"
    suffix_by_mime = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    }
    guessed = suffix_by_mime.get(
        (attachment.mime_type or "").lower(),
        mimetypes.guess_extension(attachment.mime_type or ""),
    )
    return f"attachment{guessed or ''}"


def _is_image(attachment: TelegramAttachment, path: Path) -> bool:
    if attachment.kind == "photo":
        return True
    return (
        (attachment.mime_type or "").lower() in IMAGE_MIME_TYPES
        or path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _is_audio(attachment: TelegramAttachment, path: Path) -> bool:
    return (
        attachment.kind in {"voice", "audio"}
        or (attachment.mime_type or "").lower().startswith("audio/")
        or path.suffix.lower() in AUDIO_EXTENSIONS
    )


async def _convert_document(path: Path, filename: str) -> str:
    try:
        markdown = await asyncio.to_thread(_convert_document_sync, path)
    except AttachmentProcessingError as exc:
        if not exc.filename:
            raise AttachmentProcessingError(exc.text_key, filename=filename) from exc
        raise
    except Exception as exc:
        try:
            file_size = path.stat().st_size
        except OSError:
            file_size = None
        log_exception(
            logger,
            "Could not convert Telegram document",
            exc,
            event="document.conversion.failed",
            level=logging.WARNING,
            backend="markitdown",
            file_type=path.suffix.lower().lstrip("."),
            file_size=file_size,
        )
        raise AttachmentProcessingError("documents.label.conversion_failed", filename=filename) from exc
    return markdown


def _convert_document_sync(path: Path) -> str:
    try:
        module = importlib.import_module("markitdown")
    except ImportError as exc:
        raise AttachmentProcessingError("documents.label.converter_missing") from exc

    converter_type = getattr(module, "MarkItDown", None)
    if converter_type is None:
        raise AttachmentProcessingError("documents.label.converter_missing")
    converter = converter_type(enable_plugins=False)
    try:
        result = converter.convert_local(str(path))
    except Exception as exc:
        if type(exc).__name__ in {"MissingDependencyException", "ModuleNotFoundError"}:
            raise AttachmentProcessingError("documents.label.converter_missing") from exc
        raise
    markdown = getattr(result, "markdown", None)
    return markdown if isinstance(markdown, str) else ""
