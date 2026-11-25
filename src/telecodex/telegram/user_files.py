"""Telegram file browsing and delivery for a selected Codex workspace."""

from __future__ import annotations

import asyncio
import html
import secrets
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import FSInputFile

from telecodex.telegram.activity import utf16_length
from telecodex.telegram.formatting import TELEGRAM_TEXT_LIMIT, truncate
from telecodex.telegram.pagination import PAGE_SIZE, keyboard, navigation_row, page_count, two_column_rows
from telecodex.telegram.pickers import FilesPicker
from telecodex.telegram.updates import TelegramCallbackContext
from telecodex.workspace_files import (
    WorkspaceEntry,
    WorkspaceFileBrowser,
    WorkspaceFileTooLarge,
    WorkspaceFilesError,
    WorkspacePreviewUnavailable,
)


class TelegramWorkspaceFilesController:
    """Own the Telegram file picker, previews, and downloads for one context."""

    def __init__(
        self,
        *,
        telegram: Bot,
        codex: Callable[[], Any],
        workspace_root: Callable[[], Path],
        text: Callable[..., str],
        texts: Any,
        edit_message: Callable[..., Awaitable[Any]],
        log_failure: Callable[..., None],
    ) -> None:
        self.telegram = telegram
        self._codex = codex
        self._workspace_root = workspace_root
        self._text = text
        self.texts = texts
        self._edit_message = edit_message
        self._log_failure = log_failure
        self.picker: FilesPicker | None = None

    async def _show_files_picker(self, chat_id: int, user_id: int) -> None:
        codex = self._codex()
        picker = FilesPicker(
            token=secrets.token_urlsafe(6),
            owner_user_id=user_id,
            current_path=codex.working_directory,
            entries=[],
        )
        if not await self._load_files_picker(chat_id, picker, send_new=True):
            await self.telegram.send_message(chat_id, self._text("files.label.read_failed"))


    async def _load_files_picker(
        self,
        chat_id: int,
        picker: FilesPicker,
        *,
        send_new: bool = False,
    ) -> bool:
        try:
            browser = WorkspaceFileBrowser(self._workspace_root())
            picker.current_path = browser.resolve_directory(picker.current_path)
            picker.entries = await asyncio.to_thread(
                browser.list_entries,
                picker.current_path,
            )
            picker.page = min(picker.page, page_count(len(picker.entries)) - 1)
            picker.selected_index = None
        except WorkspaceFilesError as exc:
            self._log_failure(
                "Could not list project files on the host",
                exc,
                event="telegram.files.list_failed",
                level=logging.WARNING,
            )
            return False

        text = self._files_picker_text(picker)
        markup = self._files_picker_keyboard(picker)
        if send_new:
            message = await self.telegram.send_message(chat_id, text, reply_markup=markup)
            message_id = message.message_id
            if isinstance(message_id, int):
                picker.message_id = message_id
                self.picker = picker
        elif picker.message_id is not None:
            await self._edit_message(chat_id, picker.message_id, text, markup)
        return True


    def _files_picker_text(self, picker: FilesPicker) -> str:
        return self._text("files.label.browser", path=str(picker.current_path))


    def _files_picker_keyboard(self, picker: FilesPicker) -> dict[str, Any]:
        rows: list[list[dict[str, str]]] = []
        if picker.current_path != self._workspace_root():
            rows.append([{"text": self._text("files.buttons.up"), "callback_data": f"fb:{picker.token}:u"}])
        buttons: list[dict[str, str]] = []
        start = picker.page * PAGE_SIZE
        end = min(start + PAGE_SIZE, len(picker.entries))
        for index in range(start, end):
            entry = picker.entries[index]
            if entry.kind == "directory":
                label = f"📁 {entry.name}"
                action = "o"
            else:
                label = f"📄 {entry.name} ({self._format_workspace_file_size(entry.size or 0)})"
                action = "f"
            buttons.append(
                {
                    "text": truncate(label, 60, self.texts.system("values.ellipsis")),
                    "callback_data": f"fb:{picker.token}:{action}:{index}",
                }
            )
        rows.extend(two_column_rows(buttons, 0))
        count = page_count(len(picker.entries))
        if count > 1:
            rows.append(
                navigation_row(
                    picker.page,
                    count,
                    self.texts.system("buttons.pager.previous"),
                    self.texts.system("buttons.pager.page", current="{current}", total="{total}"),
                    self.texts.system("buttons.pager.next"),
                    f"fb:{picker.token}",
                )
            )
        return keyboard(rows)


    def _files_detail_text(self, entry: WorkspaceEntry) -> str:
        return self._text(
            "files.label.detail",
            path=str(entry.path),
            size=self._format_workspace_file_size(entry.size or 0),
        )


    def _files_detail_keyboard(self, picker: FilesPicker) -> dict[str, Any]:
        return keyboard(
            [
                [
                    {"text": self._text("files.buttons.download"), "callback_data": f"fb:{picker.token}:d"},
                    {"text": self._text("files.buttons.preview"), "callback_data": f"fb:{picker.token}:v"},
                ],
                [{"text": self._text("files.buttons.back"), "callback_data": f"fb:{picker.token}:b"}],
            ]
        )


    @staticmethod
    def _format_workspace_file_size(size: int) -> str:
        if size >= 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        if size >= 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size} B"


    async def _handle_files_callback(
        self,
        incoming: TelegramCallbackContext,
        parts: list[str],
    ) -> None:
        picker = self.picker
        if (
            picker is None
            or len(parts) < 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
            or incoming.user_id != picker.owner_user_id
        ):
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("files.label.stale"))
            return
        action = parts[2]
        if action == "p" and len(parts) == 4:
            try:
                page = int(parts[3])
            except ValueError:
                page = -1
            if not 0 <= page < page_count(len(picker.entries)):
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("files.label.stale"))
                return
            picker.page = page
            picker.selected_index = None
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._files_picker_text(picker),
                self._files_picker_keyboard(picker),
            )
            return
        if action == "i" and len(parts) == 3:
            await self.telegram.answer_callback_query(incoming.callback_id)
            return
        if action == "u" and len(parts) == 3:
            if picker.current_path == self._workspace_root():
                await self.telegram.answer_callback_query(incoming.callback_id)
                return
            picker.current_path = picker.current_path.parent
            picker.page = 0
            await self.telegram.answer_callback_query(incoming.callback_id)
            if not await self._load_files_picker(incoming.chat_id, picker):
                await self.telegram.send_message(incoming.chat_id, self._text("files.label.read_failed"))
            return
        if action in {"o", "f"} and len(parts) == 4:
            try:
                index = int(parts[3])
                start = picker.page * PAGE_SIZE
                if not start <= index < min(start + PAGE_SIZE, len(picker.entries)):
                    raise ValueError
                entry = picker.entries[index]
            except (ValueError, IndexError):
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("files.label.stale"))
                return
            if action == "o" and entry.kind == "directory":
                picker.current_path = entry.path
                picker.page = 0
                await self.telegram.answer_callback_query(incoming.callback_id)
                if not await self._load_files_picker(incoming.chat_id, picker):
                    await self.telegram.send_message(incoming.chat_id, self._text("files.label.read_failed"))
                return
            if action == "f" and entry.kind == "file":
                picker.selected_index = index
                await self.telegram.answer_callback_query(incoming.callback_id)
                await self._edit_message(
                    incoming.chat_id,
                    incoming.message_id,
                    self._files_detail_text(entry),
                    self._files_detail_keyboard(picker),
                )
                return
        if action == "b" and len(parts) == 3 and self._selected_workspace_file(picker):
            picker.selected_index = None
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._files_picker_text(picker),
                self._files_picker_keyboard(picker),
            )
            return
        entry = self._selected_workspace_file(picker)
        if entry is None or action not in {"d", "v"} or len(parts) != 3:
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("files.label.stale"))
            return
        await self.telegram.answer_callback_query(incoming.callback_id)
        if action == "v":
            await self._send_workspace_preview(incoming.chat_id, picker, entry)
        else:
            await self._send_workspace_file(incoming.chat_id, picker, entry)


    @staticmethod
    def _selected_workspace_file(picker: FilesPicker) -> WorkspaceEntry | None:
        index = picker.selected_index
        if index is None or not 0 <= index < len(picker.entries):
            return None
        entry = picker.entries[index]
        return entry if entry.kind == "file" else None


    async def _send_workspace_preview(
        self,
        chat_id: int,
        picker: FilesPicker,
        entry: WorkspaceEntry,
    ) -> None:
        browser = WorkspaceFileBrowser(self._workspace_root())
        try:
            preview = await asyncio.to_thread(browser.read_text_preview, entry.path)
        except WorkspacePreviewUnavailable:
            await self.telegram.send_message(chat_id, self._text("files.label.preview_unavailable"))
            return
        except WorkspaceFilesError as exc:
            self._log_failure(
                "Could not preview project file on the host",
                exc,
                event="telegram.files.preview_failed",
                level=logging.WARNING,
            )
            await self.telegram.send_message(chat_id, self._text("files.label.preview_unavailable"))
            return
        header = self._text(
            "files.label.preview_header",
            path=str(entry.path),
        )
        suffix = self._text("files.values.preview_truncated") if preview.truncated else ""
        code_wrapper = "\n<pre></pre>"
        available = max(
            0,
            TELEGRAM_TEXT_LIMIT
            - utf16_length(header + code_wrapper)
            - utf16_length(suffix),
        )
        body = self._truncate_utf16(preview.text, available)
        clipped = len(body) < len(preview.text)
        if clipped and not suffix:
            suffix = self._text("files.values.preview_truncated")
            available = max(
                0,
                TELEGRAM_TEXT_LIMIT
                - utf16_length(header + code_wrapper)
                - utf16_length(suffix),
            )
            body = self._truncate_utf16(preview.text, available)
        code = f"\n<pre>{html.escape(body, quote=False)}</pre>" if body else ""
        text = f"{header}{code}{suffix}"
        await self.telegram.send_message(
            chat_id,
            text,
            reply_to_message_id=picker.message_id,
        )


    @staticmethod
    def _truncate_utf16(text: str, limit: int) -> str:
        result: list[str] = []
        length = 0
        for character in text:
            units = 2 if ord(character) > 0xFFFF else 1
            if length + units > limit:
                break
            result.append(character)
            length += units
        return "".join(result)


    async def _send_workspace_file(
        self,
        chat_id: int,
        picker: FilesPicker,
        entry: WorkspaceEntry,
    ) -> None:
        if entry.size is not None and entry.size > WorkspaceFileBrowser.max_download_bytes:
            await self.telegram.send_message(chat_id, self._text("files.label.too_large"))
            return
        browser = WorkspaceFileBrowser(self._workspace_root())
        with tempfile.TemporaryDirectory(prefix="telecodex-workspace-file-") as directory:
            destination = Path(directory) / "download"
            try:
                filename, path = await asyncio.to_thread(browser.copy_file, entry.path, destination)
                if not filename or "/" in filename or "\\" in filename or "\0" in filename:
                    raise ValueError("Workspace filename cannot be sent as a Telegram document")
                await self.telegram.send_document(
                    chat_id=chat_id,
                    document=FSInputFile(path, filename=filename),
                    reply_to_message_id=picker.message_id,
                )
            except WorkspaceFileTooLarge:
                await self.telegram.send_message(chat_id, self._text("files.label.too_large"))
            except TelegramAPIError as exc:
                self._log_failure(
                    "Could not send host project file through Telegram",
                    exc,
                    event="telegram.files.telegram_upload_failed",
                    level=logging.WARNING,
                )
                await self.telegram.send_message(chat_id, self._text("files.label.download_failed"))
            except WorkspaceFilesError as exc:
                self._log_failure(
                    "Could not copy host project file",
                    exc,
                    event="telegram.files.download_failed",
                    level=logging.WARNING,
                )
                await self.telegram.send_message(chat_id, self._text("files.label.download_failed"))
