from __future__ import annotations

import html
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from multi_user_test_support import FakeSentMessage, callback_event, message_event

from telecodex.config import AppConfig
from telecodex.codex.session import CodexThread
from telecodex.state import PendingTurnState, RuntimeState
from telecodex.telegram.user_controller import TelegramUserController


_HTML_FORMAT_TAG = re.compile(r"</?(?:strong|code|pre)>")


def _plain_html(text: str) -> str:
    return html.unescape(_HTML_FORMAT_TAG.sub("", text))


class FakeTelegram:
    def __init__(self) -> None:
        self.id = 999
        self.messages: list[tuple[int, str]] = []
        self.markups: list[Any] = []
        self.edits: list[tuple[int, int, str, Any]] = []
        self.callbacks: list[tuple[str, str]] = []
        self.next_message_id = 1

    async def send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: Any = None,
        *,
        reply_to_message_id: int | None = None,
        entities: list[dict[str, Any]] | None = None,
        disable_notification: bool = False,
    ) -> FakeSentMessage:
        self.messages.append((chat_id, text))
        self.markups.append(reply_markup)
        message_id = self.next_message_id
        self.next_message_id += 1
        return FakeSentMessage(message_id=message_id)

    async def send_rich_message(
        self,
        chat_id: int,
        rich_message: Any,
        *,
        reply_parameters: Any = None,
        disable_notification: bool = False,
    ) -> FakeSentMessage:
        markdown = (
            rich_message.get("markdown", "")
            if isinstance(rich_message, dict)
            else getattr(rich_message, "markdown", "")
        )
        reply_to_message_id = reply_parameters.message_id if reply_parameters is not None else None
        return await self.send_message(
            chat_id,
            markdown,
            reply_to_message_id=reply_to_message_id,
            disable_notification=disable_notification,
        )

    async def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: Any = None,
    ) -> None:
        self.edits.append((chat_id, message_id, text, reply_markup))

    async def answer_callback_query(self, callback_query_id: str, text: str | None = None) -> None:
        self.callbacks.append((callback_query_id, text or ""))

    async def send_chat_action(self, _chat_id: int, _action: str = "typing") -> None:
        return None


class FakeShellSession:
    def __init__(
        self,
        _runner: object,
        cwd: Path,
        _timeout: int,
        _on_output: object,
        _on_closed: object,
        _log_fields: object,
    ) -> None:
        self.working_directory = cwd
        self.is_open = False
        self.inputs: list[str] = []

    async def start(self) -> None:
        self.is_open = True

    async def close(self) -> None:
        self.is_open = False

    async def send_input(self, text: str, _message_id: int | None = None) -> None:
        self.inputs.append(text)
        if text.strip() == "cd ..":
            self.working_directory = self.working_directory.parent

    async def interrupt(self) -> bool:
        return True


class FakeCodex:
    def __init__(self, workspace: Path) -> None:
        self.state = RuntimeState(thread_id="thread-current", cwd=str(workspace))
        self.working_directory = workspace
        self.is_busy = False
        self.has_active_command_execution = False
        self.compaction_barrier_pending = False
        self.threads: list[CodexThread] = []
        self.messages: list[str] = []
        self.new_thread_calls: list[tuple[Path, bool]] = []
        self.select_calls: list[tuple[str, Path, bool]] = []
        self.interrupt_calls = 0

    @property
    def pending_turn(self) -> PendingTurnState | None:
        return self.state.pending_turn

    async def send_message(self, text: str, *_args: Any, **_kwargs: Any) -> str:
        self.messages.append(text)
        return "started"

    async def list_workspace_directories(self, _parent: Path) -> list[Path]:
        return []

    async def new_thread(self, cwd: Path, *, force: bool = False) -> None:
        self.new_thread_calls.append((cwd, force))
        self.state.thread_id = None
        self.state.cwd = str(cwd)
        self.state.pending_turn = None
        self.working_directory = cwd
        self.is_busy = False

    async def fork_thread(self) -> str:
        self.state.thread_id = "thread-forked"
        return "thread-forked"

    async def list_threads(self) -> list[CodexThread]:
        return self.threads

    async def last_assistant_message(self, _thread_id: str) -> str:
        return "previous answer"

    async def select_thread(self, thread_id: str, cwd: Path, *, force: bool = False) -> None:
        self.select_calls.append((thread_id, cwd, force))
        self.state.thread_id = thread_id
        self.state.cwd = str(cwd)
        self.state.pending_turn = None
        self.working_directory = cwd
        self.is_busy = False

    async def interrupt_and_abandon_for_switch(self) -> str:
        self.interrupt_calls += 1
        self.state.pending_turn = None
        self.is_busy = False
        self.has_active_command_execution = False
        return "unknown"


class NonIdealUserFlowTests(unittest.IsolatedAsyncioTestCase):
    def _controller(self, workspace: Path) -> tuple[TelegramUserController, FakeTelegram, FakeCodex]:
        config = AppConfig(
            telegram_bot_token="token",
            telegram_allowed_user_ids=(101,),
            workspace_root=workspace,
            codex_bin="codex",
            codex_model=None,
            network_access=False,
            approval_policy="on-request",
            shell_timeout_seconds=60,
            database_file=workspace / "state.sqlite3",
        )
        telegram = FakeTelegram()
        codex = FakeCodex(workspace)
        controller = TelegramUserController(config, telegram, object(), 101)  # type: ignore[arg-type]
        controller.bind_codex(codex)  # type: ignore[arg-type]
        return controller, telegram, codex

    @staticmethod
    def _callback(markup: Any, suffix: str) -> str:
        for row in markup["inline_keyboard"]:
            for button in row:
                if button["callback_data"].endswith(suffix):
                    return button["callback_data"]
        raise AssertionError(f"No callback ending in {suffix!r}")

    async def test_shell_cd_does_not_change_codex_directory_and_new_closes_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "project"
            workspace.mkdir()
            controller, telegram, codex = self._controller(workspace)
            with patch("telecodex.telegram.user_shell.ShellSession", FakeShellSession):
                await controller._handle_message(message_event(1, 1, 101, 101, "private", "/shell"))
                shell = controller.shell_session
                self.assertIsNotNone(shell)

                await controller._handle_message(message_event(2, 2, 101, 101, "private", "cd .."))
                self.assertEqual(shell.working_directory, workspace.parent)  # type: ignore[union-attr]
                self.assertEqual(codex.working_directory, workspace)

                await controller._handle_message(message_event(3, 3, 101, 101, "private", "/new"))
                picker = controller.directory_picker
                self.assertIsNotNone(picker)
                await controller._handle_callback(
                    callback_event(
                        4,
                        "choose-directory",
                        101,
                        101,
                        picker.message_id or 0,  # type: ignore[union-attr]
                        f"np:{picker.token}:s",  # type: ignore[union-attr]
                    )
                )

                self.assertIsNone(controller.shell_session)
                self.assertEqual(codex.new_thread_calls, [(workspace, False)])
                await controller._handle_message(
                    message_event(5, 5, 101, 101, "private", "continue in Codex")
                )
                self.assertEqual(codex.messages, ["continue in Codex"])
                self.assertEqual(
                    _plain_html(telegram.edits[-1][2]),
                    f"Новый диалог создан.\nДиректория: {workspace}\n\nИнтерактивный shell закрыт; обычные сообщения теперь идут в Codex.",
                )

    async def test_chat_selection_closes_shell_and_routes_followup_to_selected_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "other-project"
            codex_thread = CodexThread("thread-other", "Other project", None, target, None, "cli")
            controller, telegram, codex = self._controller(workspace)
            codex.threads = [codex_thread]
            with patch("telecodex.telegram.user_shell.ShellSession", FakeShellSession):
                await controller._handle_message(message_event(1, 1, 101, 101, "private", "/shell"))
                await controller._handle_message(message_event(2, 2, 101, 101, "private", "/chats"))
                picker = controller.chats_picker
                self.assertIsNotNone(picker)
                detail = self._callback(telegram.markups[-1], ":v:0")
                await controller._handle_callback(
                    callback_event(3, "open-details", 101, 101, picker.message_id or 0, detail)  # type: ignore[union-attr]
                )
                select = self._callback(telegram.edits[-1][3], ":s:0")
                await controller._handle_callback(
                    callback_event(4, "select-chat", 101, 101, picker.message_id or 0, select)  # type: ignore[union-attr]
                )

                self.assertIsNone(controller.shell_session)
                self.assertEqual(codex.select_calls, [("thread-other", target, False)])
                await controller._handle_message(
                    message_event(5, 5, 101, 101, "private", "continue in selected chat")
                )
                self.assertEqual(codex.messages, ["continue in selected chat"])

    async def test_fork_closes_shell_after_creating_the_new_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            controller, _telegram, codex = self._controller(workspace)
            with patch("telecodex.telegram.user_shell.ShellSession", FakeShellSession):
                await controller._handle_message(message_event(1, 1, 101, 101, "private", "/shell"))
                shell = controller.shell_session
                await controller._handle_message(message_event(2, 2, 101, 101, "private", "/fork"))

                self.assertEqual(codex.state.thread_id, "thread-forked")
                self.assertIsNone(controller.shell_session)
                self.assertFalse(shell.is_open)  # type: ignore[union-attr]

    async def test_new_directory_switch_during_active_command_requires_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            controller, telegram, codex = self._controller(workspace)
            codex.is_busy = True
            codex.has_active_command_execution = True
            codex.state.pending_turn = PendingTurnState(
                "thread-current", "turn-current", 1, 1, phase="running"
            )
            with patch("telecodex.telegram.user_shell.ShellSession", FakeShellSession):
                await controller._handle_message(message_event(1, 1, 101, 101, "private", "/shell"))
                await controller._handle_message(message_event(2, 2, 101, 101, "private", "/new"))
                picker = controller.directory_picker
                await controller._handle_callback(
                    callback_event(
                        3,
                        "choose-directory",
                        101,
                        101,
                        picker.message_id or 0,  # type: ignore[union-attr]
                        f"np:{picker.token}:s",  # type: ignore[union-attr]
                    )
                )

                self.assertEqual(codex.new_thread_calls, [])
                self.assertTrue(codex.is_busy)
                confirm = self._callback(telegram.markups[-1], ":yes")
                message_id = telegram.next_message_id - 1
                await controller._handle_callback(
                    callback_event(4, "confirm-new", 101, 101, message_id, confirm)
                )

                self.assertEqual(codex.interrupt_calls, 1)
                self.assertEqual(codex.new_thread_calls, [(workspace, True)])
                self.assertEqual(codex.messages, [])
                self.assertIsNone(controller.shell_session)

    async def test_declining_switch_keeps_active_turn_and_shell_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "other-project"
            controller, telegram, codex = self._controller(workspace)
            codex.threads = [CodexThread("thread-other", "Other project", None, target, None, "cli")]
            codex.is_busy = True
            codex.has_active_command_execution = True
            codex.state.pending_turn = PendingTurnState(
                "thread-current", "turn-current", 1, 1, phase="running"
            )
            with patch("telecodex.telegram.user_shell.ShellSession", FakeShellSession):
                await controller._handle_message(message_event(1, 1, 101, 101, "private", "/shell"))
                await controller._handle_message(message_event(2, 2, 101, 101, "private", "/chats"))
                picker = controller.chats_picker
                detail = self._callback(telegram.markups[-1], ":v:0")
                await controller._handle_callback(
                    callback_event(3, "open-details", 101, 101, picker.message_id or 0, detail)  # type: ignore[union-attr]
                )
                select = self._callback(telegram.edits[-1][3], ":s:0")
                await controller._handle_callback(
                    callback_event(4, "select-chat", 101, 101, picker.message_id or 0, select)  # type: ignore[union-attr]
                )

                confirmation = telegram.markups[-1]
                cancel = self._callback(confirmation, ":no")
                message_id = telegram.next_message_id - 1
                await controller._handle_callback(
                    callback_event(5, "cancel-switch", 101, 101, message_id, cancel)
                )

                self.assertTrue(codex.is_busy)
                self.assertEqual(codex.interrupt_calls, 0)
                self.assertEqual(codex.select_calls, [])
                self.assertIsNotNone(controller.shell_session)

    async def test_confirmed_switch_releases_busy_context_without_replaying_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "other-project"
            controller, telegram, codex = self._controller(workspace)
            codex.threads = [CodexThread("thread-other", "Other project", None, target, None, "cli")]
            codex.is_busy = True
            codex.has_active_command_execution = True
            codex.state.pending_turn = PendingTurnState(
                "thread-current", "turn-current", 1, 1, phase="running"
            )
            with patch("telecodex.telegram.user_shell.ShellSession", FakeShellSession):
                await controller._handle_message(message_event(1, 1, 101, 101, "private", "/shell"))
                await controller._handle_message(message_event(2, 2, 101, 101, "private", "/chats"))
                picker = controller.chats_picker
                detail = self._callback(telegram.markups[-1], ":v:0")
                await controller._handle_callback(
                    callback_event(3, "open-details", 101, 101, picker.message_id or 0, detail)  # type: ignore[union-attr]
                )
                select = self._callback(telegram.edits[-1][3], ":s:0")
                await controller._handle_callback(
                    callback_event(4, "select-chat", 101, 101, picker.message_id or 0, select)  # type: ignore[union-attr]
                )

                confirmation = telegram.markups[-1]
                confirm = self._callback(confirmation, ":yes")
                message_id = telegram.next_message_id - 1
                await controller._handle_callback(
                    callback_event(5, "confirm-switch", 101, 101, message_id, confirm)
                )

                self.assertEqual(codex.interrupt_calls, 1)
                self.assertEqual(codex.select_calls, [("thread-other", target, True)])
                self.assertEqual(codex.messages, [])
                self.assertIsNone(controller.shell_session)
                await controller._handle_message(
                    message_event(6, 6, 101, 101, "private", "new task")
                )
                self.assertEqual(codex.messages, ["new task"])

    async def test_quiet_turn_switch_button_is_explicit_and_keeps_old_thread_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            controller, telegram, codex = self._controller(workspace)
            codex.threads = [
                CodexThread("thread-current", "Current task", None, workspace, None, "cli")
            ]
            codex.is_busy = True
            codex.has_active_command_execution = False
            codex.state.pending_turn = PendingTurnState(
                "thread-current", "turn-current", 1, 1, phase="running"
            )

            await controller.send_recovery_switch_offer()
            recovery_rows = telegram.markups[-1]["inline_keyboard"]
            self.assertEqual([len(row) for row in recovery_rows], [1, 1])
            switch = self._callback(telegram.markups[-1], ":switch")
            message_id = telegram.next_message_id - 1
            await controller._handle_callback(
                callback_event(2, "switch-quiet-turn", 101, 101, message_id, switch)
            )

            self.assertEqual(codex.interrupt_calls, 1)
            self.assertEqual(codex.new_thread_calls, [(workspace, True)])
            self.assertEqual(codex.state.thread_id, None)
            self.assertEqual([thread.thread_id for thread in codex.threads], ["thread-current"])
            self.assertEqual(
                _plain_html(telegram.edits[-1][2]),
                f"Новый диалог создан в директории {workspace}. Итог предыдущей задачи неизвестен; исходный запрос повторно не запускался.",
            )
            self.assertEqual(codex.messages, [])

    async def test_quiet_turn_wait_button_keeps_the_current_turn_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            controller, telegram, codex = self._controller(workspace)
            codex.is_busy = True
            codex.state.pending_turn = PendingTurnState(
                "thread-current", "turn-current", 1, 1, phase="running"
            )

            await controller.send_recovery_switch_offer()
            wait = self._callback(telegram.markups[-1], ":wait")
            message_id = telegram.next_message_id - 1
            await controller._handle_callback(
                callback_event(2, "wait-for-turn", 101, 101, message_id, wait)
            )

            self.assertEqual(codex.interrupt_calls, 0)
            self.assertTrue(codex.is_busy)
            self.assertEqual(codex.pending_turn.turn_id, "turn-current")  # type: ignore[union-attr]
            self.assertEqual(
                telegram.edits[-1][2],
                "Ожидание ответа Codex продолжается. Текущая задача не прервана.",
            )

    async def test_thread_switch_confirmation_buttons_use_separate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            controller, telegram, _codex = self._controller(workspace)

            await controller._request_thread_switch(
                callback_event(1, "switch-confirm", 101, 101, 50, ""),
                action="new",
                path=workspace,
            )

            rows = telegram.markups[-1]["inline_keyboard"]
            self.assertEqual([len(row) for row in rows], [1, 1])

    async def test_typo_suggests_commands_but_typo_inside_shell_stays_shell_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, telegram, _codex = self._controller(Path(temporary))
            await controller._handle_message(message_event(1, 1, 101, 101, "private", "/neww"))
            self.assertIn("/new", telegram.messages[-1][1])
            self.assertEqual(_codex.messages, [])

            with patch("telecodex.telegram.user_shell.ShellSession", FakeShellSession):
                await controller._handle_message(message_event(2, 2, 101, 101, "private", "/shell"))
                session = controller.shell_session
                await controller._handle_message(message_event(3, 3, 101, 101, "private", "/neww"))
                self.assertEqual(session.inputs, ["/neww"])  # type: ignore[union-attr]
                await controller.close_shell_session()

