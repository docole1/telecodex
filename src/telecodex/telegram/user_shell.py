"""Interactive shell commands and output delivery for one Telegram context."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiogram import Bot
from aiogram.types import Message

from telecodex.config import AppConfig
from telecodex.shell.runner import EmptyShellCommandError, ShellRunner
from telecodex.shell.session import ShellOutput, ShellSession
from telecodex.telegram.activity import utf16_length
from telecodex.telegram.formatting import TELEGRAM_TEXT_LIMIT, split_plain_text

_TERMINAL_ESCAPE_SEQUENCE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))"
)


def clean_terminal_output(text: str) -> str:
    text = _TERMINAL_ESCAPE_SEQUENCE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        character
        for character in text
        if character in "\n\t" or (ord(character) >= 32 and not 0x7F <= ord(character) <= 0x9F)
    )


class TelegramShellController:
    """Own interactive shell state and adapt shell output to Telegram."""

    def __init__(
        self,
        *,
        config: AppConfig,
        telegram: Bot,
        shell: ShellRunner,
        chat_id: int,
        is_group: bool,
        is_attached: Callable[[], bool],
        codex: Callable[[], Any],
        text: Callable[..., str],
        log_context_fields: Callable[..., dict[str, str | None]],
        log_failure: Callable[..., None],
        consume_group_trigger: Callable[[str], str | None],
        truncate: Callable[..., str],
        queue_group_activity: Callable[..., Awaitable[dict[str, Any]]],
        activity_lock: asyncio.Lock,
        group_message_limit: int,
    ) -> None:
        self.config = config
        self.telegram = telegram
        self.shell = shell
        self.chat_id = chat_id
        self.is_group = is_group
        self._is_attached = is_attached
        self._codex = codex
        self._text = text
        self._log_context_fields = log_context_fields
        self._log_failure = log_failure
        self._consume_group_trigger = consume_group_trigger
        self._truncate = truncate
        self._queue_group_activity = queue_group_activity
        self._codex_activity_lock = activity_lock
        self.group_message_limit = group_message_limit
        self.session: ShellSession | None = None

    @property
    def attached(self) -> bool:
        return self._is_attached()

    async def close_shell_session(self) -> None:
        session = self.session
        self.session = None
        if session is not None:
            await session.close()


    async def _enter_shell(self, incoming: Message) -> None:
        if self.session is not None and self.session.is_open:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("shell.label.already_open", cwd=self._codex().working_directory),
                reply_to_message_id=incoming.message_id,
            )
            return

        session = ShellSession(
            self.shell,
            self._codex().working_directory,
            self.config.shell_timeout_seconds,
            self._on_shell_output,
            self._on_shell_closed,
            self._log_context_fields(),
        )
        self.session = session
        try:
            await session.start()
        except Exception as exc:
            if self.session is session:
                self.session = None
            self._log_failure(
                "Could not start interactive host shell",
                exc,
                event="shell.session.start_failed",
            )
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("shell.label.start_failed"),
                reply_to_message_id=incoming.message_id,
            )
            return

        await self.telegram.send_message(
            incoming.chat_id,
            self._text("shell.label.started", cwd=self._codex().working_directory),
            reply_to_message_id=incoming.message_id,
        )


    async def _exit_shell(self, incoming: Message) -> None:
        session = self.session
        if session is None:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("shell.label.not_open"),
                reply_to_message_id=incoming.message_id,
            )
            return
        self.session = None
        await session.close()
        await self.telegram.send_message(
            incoming.chat_id,
            self._text("shell.label.closed"),
            reply_to_message_id=incoming.message_id,
        )


    async def _send_shell_input(self, incoming: Message) -> None:
        session = self.session
        if session is None:
            return
        text = incoming.text
        if self.is_group:
            triggered_text = self._consume_group_trigger(text)
            if triggered_text is not None:
                text = triggered_text
                if not text:
                    await self.telegram.send_message(
                        incoming.chat_id,
                        self._text("groups.label.trigger_empty"),
                        reply_to_message_id=incoming.message_id,
                    )
                    return
        try:
            await session.send_input(text, incoming.message_id)
        except ValueError:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("shell.label.input_too_long"),
                reply_to_message_id=incoming.message_id,
            )
        except (OSError, RuntimeError) as exc:
            if self.session is session:
                self.session = None
            self._log_failure(
                "Could not write to interactive shell",
                exc,
                event="shell.session.input_failed",
                level=logging.WARNING,
            )
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("shell.label.not_open"),
                reply_to_message_id=incoming.message_id,
            )


    async def _stop_shell(self, incoming: Message) -> None:
        session = self.session
        stopped = await session.interrupt() if session is not None else False
        await self.telegram.send_message(
            incoming.chat_id,
            self._text("shell.label.stop_requested" if stopped else "shell.label.stop_idle"),
            reply_to_message_id=incoming.message_id,
        )


    async def _on_shell_output(self, output: ShellOutput) -> None:
        if self.is_group and not self.attached:
            return
        text = clean_terminal_output(output.text)
        if output.partial:
            if text:
                await self._send_shell_output_chunks(text, output.reply_to_message_id)
            return

        if output.return_code is None and not output.timed_out:
            if text:
                await self._send_shell_output_chunks(text, output.reply_to_message_id)
            return

        if output.timed_out:
            state = self._text("shell.values.timeout")
        elif output.return_code is not None:
            state = self._text("shell.values.exit_code", code=output.return_code)
        else:
            state = self._text("shell.label.process_closed")
        if output.truncated:
            state += self._text("shell.values.output_truncated")

        if text and output.previous_output_sent:
            await self._send_shell_output_chunks(text, output.reply_to_message_id)
            await self._send_shell_status(state, output.reply_to_message_id)
            return
        if text:
            await self._send_shell_result(state, text, output.reply_to_message_id)
            return
        if output.previous_output_sent:
            await self._send_shell_status(state, output.reply_to_message_id)
            return
        await self._send_shell_status(
            f"{state}\n\n{self._text('shell.values.no_output')}",
            output.reply_to_message_id,
        )


    async def _send_shell_result(
        self,
        state: str,
        output: str,
        reply_to_message_id: int | None,
    ) -> None:
        limit = self.group_message_limit if self.is_group else TELEGRAM_TEXT_LIMIT
        prefix = f"{state}\n\n"
        first_limit = max(1, limit - utf16_length(prefix))
        chunks = split_plain_text(output, limit=first_limit)
        if not chunks:
            await self._send_shell_status(
                f"{state}\n\n{self._text('shell.values.no_output')}",
                reply_to_message_id,
            )
            return
        first = chunks.pop(0)
        first_text = prefix + first
        first_entities = [
            {
                "type": "pre",
                "offset": utf16_length(prefix),
                "length": utf16_length(first),
            }
        ]
        await self._send_shell_message(first_text, first_entities, reply_to_message_id)
        await self._send_shell_output_chunks("\n".join(chunks), reply_to_message_id)


    async def _send_shell_output_chunks(
        self,
        output: str,
        reply_to_message_id: int | None,
    ) -> None:
        limit = self.group_message_limit if self.is_group else TELEGRAM_TEXT_LIMIT
        for chunk in split_plain_text(output, limit=limit):
            entities = [
                {"type": "pre", "offset": 0, "length": utf16_length(chunk)}
            ]
            await self._send_shell_message(chunk, entities, reply_to_message_id)


    async def _send_shell_status(
        self,
        text: str,
        reply_to_message_id: int | None,
    ) -> None:
        for chunk in split_plain_text(text, limit=TELEGRAM_TEXT_LIMIT):
            await self._send_shell_message(chunk, [], reply_to_message_id)


    async def _send_shell_message(
        self,
        text: str,
        entities: list[dict[str, Any]],
        reply_to_message_id: int | None,
    ) -> None:
        async with self._codex_activity_lock:
            if self.is_group:
                await self._queue_group_activity(
                    text,
                    entities,
                    reply_to_message_id,
                    batchable=False,
                    wait_for_delivery=True,
                )
            elif self.attached:
                await self.telegram.send_message(
                    self.chat_id,
                    text,
                    reply_to_message_id=reply_to_message_id,
                    entities=entities,
                )


    async def _on_shell_closed(self, session: ShellSession, reason: str) -> None:
        if self.session is not session:
            return
        self.session = None
        await session.close()
        if self.is_group and not self.attached:
            return
        key = {
            "idle_timeout": "shell.label.idle_closed",
            "process_exit": "shell.label.process_closed",
            "reader_error": "shell.label.reader_closed",
        }.get(reason, "shell.label.process_closed")
        await self.telegram.send_message(self.chat_id, self._text(key))


    async def _run_shell(self, chat_id: int, command: str) -> None:
        if not command.strip():
            await self.telegram.send_message(chat_id, self._text("shell.label.usage"))
            return
        try:
            result = await self.shell.run(
                command,
                working_directory=self._codex().working_directory,
            )
        except EmptyShellCommandError:
            await self.telegram.send_message(chat_id, self._text("shell.label.usage"))
            return
        except OSError:
            await self.telegram.send_message(chat_id, self._text("errors.label.shell_exec"))
            return
        state = (
            self._text("shell.values.timeout")
            if result.timed_out
            else self._text("shell.values.exit_code", code=result.return_code)
        )
        output = result.output or self._text("shell.values.no_output")
        output_suffix = self._text("shell.values.output_truncated") if result.output_truncated else ""
        prefix = self._text("shell.label.result", state=state, output="")
        available_length = max(0, TELEGRAM_TEXT_LIMIT - len(prefix) - len(output_suffix))
        output = self._truncate(output, available_length) + output_suffix
        await self.telegram.send_message(chat_id, self._text("shell.label.result", state=state, output=output))
