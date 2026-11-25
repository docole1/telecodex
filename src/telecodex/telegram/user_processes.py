"""Telegram controls for Codex background terminals."""

from __future__ import annotations

import logging
import re
import secrets
from typing import Any, Callable

from telecodex.codex.types import CodexBackgroundTerminal
from telecodex.logging_config import redact_text
from telecodex.telegram.api import TelegramAPI
from telecodex.telegram.formatting import truncate
from telecodex.telegram.pagination import PAGE_SIZE, keyboard, navigation_row, page_count, two_column_rows
from telecodex.telegram.pickers import ProcessesPicker
from telecodex.telegram.updates import IncomingCallback

_SENSITIVE_COMMAND_ARGUMENT = re.compile(
    r"(?i)(--?(?:password|passwd|token|secret|api[-_]?key|credential|auth)(?:=|\s+))"
    r"(\"[^\"]*\"|'[^']*'|[^\s]+)"
)


class TelegramProcessesController:
    """Own the `/ps` picker for one Telegram conversation."""

    def __init__(
        self,
        *,
        telegram: TelegramAPI,
        codex: Callable[[], Any],
        text: Callable[..., str],
        texts: Any,
        edit_message: Callable[..., Any],
        log_failure: Callable[..., None],
    ) -> None:
        self.telegram = telegram
        self._codex = codex
        self._text = text
        self.texts = texts
        self._edit_message = edit_message
        self._log_failure = log_failure
        self.picker: ProcessesPicker | None = None

    async def _show_processes(self, chat_id: int) -> None:
        codex = self._codex()
        thread_id = codex.state.thread_id
        if not thread_id:
            await self.telegram.send_message(chat_id, self._text("ps.label.no_chat"))
            return
        try:
            terminals = await codex.list_background_terminals()
        except Exception as exc:
            self._log_failure(
                "Could not list Codex background terminals",
                exc,
                event="telegram.processes.list_failed",
                rpc_method="thread/backgroundTerminals/list",
                level=logging.WARNING,
            )
            await self.telegram.send_message(chat_id, self._text("ps.label.list_failed"))
            return
        if not terminals:
            await self.telegram.send_message(chat_id, self._text("ps.label.empty"))
            return

        picker = ProcessesPicker(
            token=secrets.token_urlsafe(6),
            thread_id=thread_id,
            terminals=terminals,
        )
        message = await self.telegram.send_message(
            chat_id,
            self._processes_picker_text(picker),
            self._processes_picker_keyboard(picker),
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.picker = picker


    def _terminal_command_text(self, terminal: CodexBackgroundTerminal) -> str:
        command = redact_text(terminal.command, max_length=400)
        command = _SENSITIVE_COMMAND_ARGUMENT.sub(r"\1[REDACTED]", command)
        return truncate(command, 180, "…")


    def _terminal_cwd_text(
        self,
        terminal: CodexBackgroundTerminal,
        limit: int,
    ) -> str:
        cwd = terminal.cwd or self._text("ps.values.cwd_unknown")
        return truncate(redact_text(cwd, max_length=600), limit, "…")


    def _terminal_resources_text(self, terminal: CodexBackgroundTerminal) -> str:
        pid = str(terminal.os_pid) if terminal.os_pid is not None else self._text("common.values.no_data")
        cpu = (
            f"{terminal.cpu_percent:.1f}%"
            if terminal.cpu_percent is not None
            else self._text("common.values.no_data")
        )
        memory = (
            f"{terminal.rss_kb / 1024:.1f} МБ"
            if terminal.rss_kb is not None
            else self._text("common.values.no_data")
        )
        return self._text("ps.values.resources", pid=pid, cpu=cpu, memory=memory)


    def _processes_picker_text(self, picker: ProcessesPicker) -> str:
        total_pages = page_count(len(picker.terminals))
        lines = [
            self._text(
                "ps.label.list_title",
                current=picker.page + 1,
                total=total_pages,
            )
        ]
        start = picker.page * PAGE_SIZE
        end = min(start + PAGE_SIZE, len(picker.terminals))
        for index in range(start, end):
            terminal = picker.terminals[index]
            lines.append(
                self._text(
                    "ps.values.list_item",
                    command=truncate(self._terminal_command_text(terminal), 120, "…"),
                    cwd=self._terminal_cwd_text(terminal, 100),
                    resources=self._terminal_resources_text(terminal),
                )
            )
        return "\n\n".join(lines)


    def _processes_picker_keyboard(self, picker: ProcessesPicker) -> dict[str, Any]:
        count = page_count(len(picker.terminals))
        start = picker.page * PAGE_SIZE
        end = min(start + PAGE_SIZE, len(picker.terminals))
        buttons = [
            {
                "text": truncate(
                    self._terminal_command_text(picker.terminals[index]),
                    64,
                    self.texts.system("values.ellipsis"),
                ),
                "callback_data": f"ps:{picker.token}:v:{index}",
            }
            for index in range(start, end)
        ]
        rows = two_column_rows(buttons, 0)
        if count > 1:
            rows.append(
                navigation_row(
                    picker.page,
                    count,
                    self.texts.system("buttons.pager.previous"),
                    self.texts.system("buttons.pager.page", current="{current}", total="{total}"),
                    self.texts.system("buttons.pager.next"),
                    f"ps:{picker.token}",
                )
            )
        return keyboard(rows)


    def _process_detail_text(self, terminal: CodexBackgroundTerminal) -> str:
        return self._text(
            "ps.label.detail",
            command=truncate(self._terminal_command_text(terminal), 300, "…"),
            cwd=self._terminal_cwd_text(terminal, 500),
            resources=self._terminal_resources_text(terminal),
        )


    def _process_detail_keyboard(self, picker: ProcessesPicker) -> dict[str, Any]:
        return keyboard(
            [
                [
                    {
                        "text": self._text("ps.buttons.stop_button"),
                        "callback_data": f"ps:{picker.token}:s",
                    }
                ],
                [
                    {
                        "text": self._text("common.buttons.back_to_list"),
                        "callback_data": f"ps:{picker.token}:b",
                    }
                ],
            ]
        )


    async def _refresh_process_picker(
        self,
        incoming: IncomingCallback,
        picker: ProcessesPicker,
    ) -> None:
        picker.terminals = await self._codex().list_background_terminals()
        picker.selected_process_id = None
        if not picker.terminals:
            self.picker = None
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("ps.label.empty"),
                {"inline_keyboard": []},
            )
            return
        picker.page = min(picker.page, page_count(len(picker.terminals)) - 1)
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self._processes_picker_text(picker),
            self._processes_picker_keyboard(picker),
        )


    async def _handle_processes_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        picker = self.picker
        if (
            picker is None
            or len(parts) < 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
            or self._codex().state.thread_id != picker.thread_id
        ):
            await self.telegram.answer_callback(incoming.callback_id, self._text("ps.label.stale"))
            return

        action = parts[2]
        if action == "p" and len(parts) == 4:
            try:
                page = int(parts[3])
            except ValueError:
                page = -1
            if page < 0 or page >= page_count(len(picker.terminals)):
                await self.telegram.answer_callback(incoming.callback_id, self._text("ps.label.stale"))
                return
            picker.page = page
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._processes_picker_text(picker),
                self._processes_picker_keyboard(picker),
            )
            return
        if action == "i" and len(parts) == 3:
            await self.telegram.answer_callback(incoming.callback_id)
            return
        if action == "v" and len(parts) == 4:
            try:
                index = int(parts[3])
            except ValueError:
                index = -1
            if index < 0 or index >= len(picker.terminals):
                await self.telegram.answer_callback(incoming.callback_id, self._text("ps.label.stale"))
                return
            process_id = picker.terminals[index].process_id
            try:
                terminals = await self._codex().list_background_terminals()
                terminal = next((item for item in terminals if item.process_id == process_id), None)
            except Exception as exc:
                self._log_failure(
                    "Could not refresh Codex background terminal details",
                    exc,
                    event="telegram.processes.read_failed",
                    rpc_method="thread/backgroundTerminals/list",
                    level=logging.WARNING,
                )
                terminal = None
            if terminal is None:
                await self.telegram.answer_callback(incoming.callback_id, self._text("ps.label.stale"))
                return
            picker.terminals = terminals
            picker.selected_process_id = process_id
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._process_detail_text(terminal),
                self._process_detail_keyboard(picker),
            )
            return
        if action == "b" and len(parts) == 3:
            try:
                await self._refresh_process_picker(incoming, picker)
            except Exception as exc:
                self._log_failure(
                    "Could not refresh Codex background terminal list",
                    exc,
                    event="telegram.processes.list_failed",
                    rpc_method="thread/backgroundTerminals/list",
                    level=logging.WARNING,
                )
                await self._edit_message(
                    incoming.chat_id,
                    incoming.message_id,
                    self._text("ps.label.list_failed"),
                    {"inline_keyboard": []},
                )
            await self.telegram.answer_callback(incoming.callback_id)
            return

        if action == "s" and len(parts) == 3 and picker.selected_process_id:
            try:
                terminals = await self._codex().list_background_terminals()
                selected = next(
                    (item for item in terminals if item.process_id == picker.selected_process_id),
                    None,
                )
                if selected is None:
                    await self.telegram.answer_callback(incoming.callback_id, self._text("ps.label.stale"))
                    return
                stopped = await self._codex().terminate_background_terminal(
                    picker.selected_process_id
                )
                if not stopped:
                    await self.telegram.answer_callback(incoming.callback_id, self._text("ps.label.stale"))
                    return
                await self._refresh_process_picker(incoming, picker)
                await self.telegram.answer_callback(
                    incoming.callback_id,
                    self._text("ps.label.stopped"),
                )
            except Exception as exc:
                self._log_failure(
                    "Could not stop Codex background terminal",
                    exc,
                    event="telegram.processes.stop_failed",
                    rpc_method="thread/backgroundTerminals/terminate",
                    level=logging.WARNING,
                )
                await self.telegram.answer_callback(
                    incoming.callback_id,
                    self._text("ps.label.action_failed"),
                )
            return
        await self.telegram.answer_callback(incoming.callback_id, self._text("ps.label.stale"))
