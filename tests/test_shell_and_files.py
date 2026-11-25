from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
from telecodex.config import AppConfig
from telecodex.shell.runner import ShellResult
from telecodex.telegram.formatting import TELEGRAM_TEXT_LIMIT
from telecodex.telegram.user_controller import TelegramUserController

from multi_user_test_support import FakeCodex, FakeTelegram, _catalog_text, callback_event, message_event

class ShellHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_shell_result_is_sent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101,),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=Path(temp_dir) / "state.sqlite3",
            )
            telegram = FakeTelegram()

            class Shell:
                async def run(
                    self,
                    command: str,
                    *,
                    working_directory: Path | None = None,
                ) -> ShellResult:
                    self.command = command
                    self.working_directory = working_directory
                    return ShellResult(command, 0, "/root/telecodex\n", False, False)

            shell = Shell()
            controller = TelegramUserController(config, telegram, shell, 101)  # type: ignore[arg-type]
            codex = FakeCodex()
            codex.working_directory = Path(temp_dir)
            controller.bind_codex(codex)  # type: ignore[arg-type]
            await controller._run_shell(101, "pwd")

        self.assertEqual(len(telegram.messages), 1)
        self.assertIn(_catalog_text("shell.values.exit_code", code=0), telegram.messages[0][1])
        self.assertIn("/root/telecodex", telegram.messages[0][1])
        self.assertEqual(shell.working_directory, Path(temp_dir))

    async def test_large_shell_result_includes_truncation_markers_within_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101,),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=Path(temp_dir) / "state.sqlite3",
            )
            telegram = FakeTelegram()

            class Shell:
                async def run(
                    self,
                    command: str,
                    *,
                    working_directory: Path | None = None,
                ) -> ShellResult:
                    return ShellResult(command, 0, "x" * 65536, False, True)

            controller = TelegramUserController(config, telegram, Shell(), 101)  # type: ignore[arg-type]
            codex = FakeCodex()
            codex.working_directory = Path(temp_dir)
            controller.bind_codex(codex)  # type: ignore[arg-type]
            await controller._run_shell(101, "long-output")

        text = telegram.messages[0][1]
        self.assertLessEqual(len(text), TELEGRAM_TEXT_LIMIT)
        self.assertIn(_catalog_text("format.values.truncated"), text)
        self.assertIn(_catalog_text("shell.values.output_truncated"), text)


class WorkspaceFilesUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_files_command_previews_and_downloads_host_project_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            (project / "note.txt").write_text("project contents", encoding="utf-8")
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101,),
                workspace_root=root,
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=root / "state.sqlite3",
            )
            telegram = FakeTelegram()
            controller = TelegramUserController(config, telegram, object(), 101)  # type: ignore[arg-type]
            codex = FakeCodex()
            codex.working_directory = project
            controller.bind_codex(codex)  # type: ignore[arg-type]

            await controller._handle_message(message_event(1, 1, 101, 101, "private", "/files"))

            picker = controller.files_picker
            self.assertIsNotNone(picker)
            assert picker is not None
            self.assertEqual(picker.current_path, project)
            self.assertTrue(
                any(
                    "note.txt" in button["text"]
                    for row in telegram.reply_markups[0]["inline_keyboard"]
                    for button in row
                )
            )
            await controller._handle_callback(
                callback_event(
                    2,
                    "open-file",
                    101,
                    101,
                    picker.message_id or 0,
                    f"fb:{picker.token}:f:0",
                )
            )
            self.assertIn("note.txt", telegram.edits[-1][2])

            await controller._handle_callback(
                callback_event(
                    3,
                    "preview-file",
                    101,
                    101,
                    picker.message_id or 0,
                    f"fb:{picker.token}:v",
                )
            )
            self.assertIn("<pre>project contents</pre>", telegram.messages[-1][1])
            self.assertEqual(telegram.message_entities[-1], [])

            await controller._handle_callback(
                callback_event(
                    4,
                    "download-file",
                    101,
                    101,
                    picker.message_id or 0,
                    f"fb:{picker.token}:d",
                )
            )
            self.assertEqual(telegram.documents[-1][2], "note.txt")
