from __future__ import annotations
from pathlib import Path
from typing import Any
from unittest.mock import patch
import tempfile
import unittest
from telecodex.codex.session import CodexBackgroundTerminal, CodexModel, CodexRecoverySnapshot, CodexSession
from telecodex.config import AppConfig
from telecodex.state import PendingTurnState, RuntimeState, RuntimeStateStore
from telecodex.telegram.pagination import navigation_row, page_count, two_column_rows
from telecodex.telegram.updates import IncomingCallback, IncomingMessage
from telecodex.telegram.user_controller import TelegramUserController

from multi_user_test_support import (
    FakeAppServer,
    FakeCodex,
    FakeTelegram,
    _catalog_text,
    _plain_html,
)

class ThreadCommandHandlerTests(unittest.IsolatedAsyncioTestCase):
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
        codex = FakeCodex()
        controller = TelegramUserController(config, telegram, object(), 101)  # type: ignore[arg-type]
        controller.bind_codex(codex)  # type: ignore[arg-type]
        return controller, telegram, codex

    async def test_permissions_picker_is_available_during_active_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, codex = self._controller(Path(temp_dir))
            codex.state.thread_id = "thread-active-permissions"
            codex.is_busy = True
            codex.permission_mode = "ask"

            await controller._handle_message(
                IncomingMessage(1, 83, 101, 101, "private", "/permissions")
            )

            self.assertEqual(len(telegram.messages), 1)
            self.assertIn("Текущая задача продолжит работу", telegram.messages[0][1])
            self.assertEqual(
                telegram.reply_markups[-1]["inline_keyboard"][0][0]["text"],
                "› Спрашивать при необходимости ‹",
            )
            self.assertTrue(telegram.reply_markups[-1]["inline_keyboard"])

    async def test_compact_command_does_not_send_redundant_acceptance_when_idle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, codex = self._controller(Path(temp_dir))
            codex.state.thread_id = "idle-thread"

            await controller._handle_message(
                IncomingMessage(1, 84, 101, 101, "private", "/compact")
            )
            await controller._finish_typing()

            self.assertTrue(codex.compacted)
            self.assertEqual(telegram.messages, [])

    async def test_non_destructive_pickers_and_rename_are_available_during_active_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            controller, telegram, codex = self._controller(workspace)
            codex.state.thread_id = "thread-active-pickers"
            codex.working_directory = workspace
            codex.is_busy = True
            codex.permission_mode = "ask"
            codex.models = [CodexModel("gpt-6-luna", "GPT-6-Luna", True, ("medium",), "medium")]

            commands = (
                "/rename active-test",
                "/files",
                "/chats",
                "/model",
                "/fast",
                "/skills",
                "/memories",
                "/mcp",
                "/language",
                "/mode",
            )
            for update_id, command in enumerate(commands, start=1):
                await controller._handle_message(
                    IncomingMessage(update_id, 100 + update_id, 101, 101, "private", command)
                )

            self.assertEqual(codex.renamed_chat_name, "active-test")
            self.assertTrue(codex.fast_enabled)
            self.assertEqual(codex.state.thread_id, "thread-active-pickers")
            self.assertTrue(all("Дождись завершения" not in text for _, text in telegram.messages))
            self.assertEqual(len(telegram.messages), len(commands))

    async def test_fork_during_active_turn_waits_without_interrupting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, codex = self._controller(Path(temp_dir))
            codex.state.thread_id = "thread-fork-after-turn"
            codex.is_busy = True

            await controller._handle_message(
                IncomingMessage(1, 120, 101, 101, "private", "/fork")
            )

            self.assertFalse(codex.forked)
            self.assertTrue(codex.is_busy)
            self.assertIn("после завершения текущей задачи", telegram.messages[-1][1])
            task = controller._fork_after_turn_task
            self.assertIsNotNone(task)

            codex.is_busy = False
            await task  # type: ignore[arg-type]

            self.assertTrue(codex.forked)
            self.assertEqual(codex.state.thread_id, "thread-forked")
            self.assertIn("Копия диалога создана", _plain_html(telegram.messages[-1][1]))

    async def test_recover_cancel_leaves_session_untouched_and_resumes_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, codex = self._controller(Path(temp_dir))
            codex.state.thread_id = "thread-recover-confirm"
            codex.compaction_barrier_pending = True
            codex.compaction_barrier_event.clear()
            await controller._send_to_codex(101, "queued before recover", 79)

            await controller._handle_message(
                IncomingMessage(1, 80, 101, 101, "private", "/recover")
            )

            pending = controller.pending_chat_recovery
            self.assertIsNotNone(pending)
            self.assertTrue(codex.recovery_held)
            self.assertFalse(controller._recovery_ready.is_set())
            self.assertEqual(codex.recovery_inspections, 0)
            recovery_rows = telegram.reply_markups[-1]["inline_keyboard"]
            self.assertEqual([len(row) for row in recovery_rows], [1, 1])
            cancel_data = recovery_rows[1][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(2, "cancel-recovery", 101, 101, pending.message_id, cancel_data)  # type: ignore[union-attr]
            )

            self.assertIsNone(controller.pending_chat_recovery)
            self.assertFalse(codex.recovery_held)
            self.assertTrue(controller._recovery_ready.is_set())
            self.assertEqual(codex.recovery_calls, [])
            self.assertIn("отменено", telegram.edits[-1][2].lower())
            self.assertEqual(len(controller._compaction_input_queue), 1)

            codex.compaction_barrier_pending = False
            codex.compaction_barrier_event.set()
            queue_task = controller._compaction_input_queue_task
            self.assertIsNotNone(queue_task)
            await queue_task  # type: ignore[arg-type]
            self.assertEqual(codex.messages, ["queued before recover"])

    async def test_recover_lists_and_requires_confirmation_for_active_work_and_terminals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, codex = self._controller(Path(temp_dir))
            codex.state.thread_id = "thread-recover-impact"
            codex.recovery_snapshot = CodexRecoverySnapshot(
                "thread-recover-impact",
                "active",
                "turn-recover-impact",
                (
                    CodexBackgroundTerminal(
                        "terminal-impact",
                        "python worker.py --token secret-value",
                        "/workspace",
                        9001,
                        0.0,
                        0,
                    ),
                ),
            )

            await controller._handle_message(
                IncomingMessage(1, 81, 101, 101, "private", "/recover")
            )
            pending = controller.pending_chat_recovery
            self.assertIsNotNone(pending)
            reset_data = telegram.reply_markups[-1]["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(2, "inspect-recovery", 101, 101, pending.message_id, reset_data)  # type: ignore[union-attr]
            )

            pending = controller.pending_chat_recovery
            self.assertIsNotNone(pending)
            self.assertEqual(pending.stage, "impact")  # type: ignore[union-attr]
            self.assertIn("python worker.py", telegram.edits[-1][2])
            self.assertNotIn("secret-value", telegram.edits[-1][2])
            await controller._handle_message(
                IncomingMessage(3, 82, 101, 101, "private", "send another task")
            )
            self.assertEqual(codex.messages, [])

            apply_data = telegram.edit_markups[-1]["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(4, "apply-recovery", 101, 101, pending.message_id, apply_data)  # type: ignore[union-attr]
            )

            self.assertIsNone(controller.pending_chat_recovery)
            self.assertEqual(
                codex.recovery_calls,
                [("thread-recover-impact", {"terminal-impact"}, True)],
            )
            self.assertIn("Codex перезапущен", telegram.edits[-1][2])

    async def test_recovered_result_replies_to_source_and_closes_old_prompt_buttons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            telegram = FakeTelegram()
            codex = FakeCodex()
            codex.state.pending_turn = PendingTurnState(
                thread_id="thread-recover",
                turn_id="turn-recover",
                prompt_message_id=55,
                started_at=123,
                phase="delivery_pending",
                result_status="interrupted",
                prompt_message_ids=[66],
            )
            controller = TelegramUserController(
                AppConfig(
                    telegram_bot_token="token",
                    telegram_allowed_user_ids=(101,),
                    workspace_root=Path(temp_dir),
                    codex_bin="codex",
                    codex_model=None,
                    network_access=False,
                    approval_policy="on-request",
                    shell_timeout_seconds=60,
                    database_file=Path(temp_dir) / "state.sqlite3",
                ),
                telegram,
                object(),
                101,
            )  # type: ignore[arg-type]
            controller.bind_codex(codex)  # type: ignore[arg-type]
            controller._turn_prompt_message_id = 55

            await controller._on_codex_finished(
                {"status": "interrupted", "text": "partial output", "recovered": True}
            )

            self.assertEqual(telegram.replies, [55])
            self.assertIn(
                _catalog_text("recovery.label.interrupted", message="partial output"),
                telegram.messages[0][1],
            )
            self.assertEqual(telegram.edits[0][1], 66)
            self.assertEqual(telegram.edit_markups[0], {"inline_keyboard": []})
            self.assertIsNone(codex.state.pending_turn)

    async def test_delete_is_cancelable_and_requires_explicit_confirmation(self) -> None:
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
            codex = FakeCodex()
            codex.state.thread_id = "thread-to-delete"
            controller = TelegramUserController(config, telegram, object(), 101)  # type: ignore[arg-type]
            controller.bind_codex(codex)  # type: ignore[arg-type]

            await controller._handle_message(
                IncomingMessage(1, 1, 101, 101, "private", "/delete")
            )
            confirmation = controller.delete_confirmation
            self.assertIsNotNone(confirmation)
            self.assertIn("inline_keyboard", telegram.reply_markups[-1])
            self.assertIsNone(codex.deleted_thread)

            await controller._handle_callback(
                IncomingCallback(
                    2,
                    "cancel-delete",
                    101,
                    101,
                    confirmation.message_id,  # type: ignore[union-attr]
                    f"td:{confirmation.token}:cancel",  # type: ignore[union-attr]
                )
            )
            self.assertIsNone(codex.deleted_thread)
            self.assertEqual(telegram.edits[-1][2], _catalog_text("thread_commands.label.delete_cancelled"))

            codex.state.thread_id = "thread-to-delete"
            await controller._handle_message(
                IncomingMessage(3, 3, 101, 101, "private", "/delete")
            )
            confirmation = controller.delete_confirmation
            self.assertIsNotNone(confirmation)
            await controller._handle_callback(
                IncomingCallback(
                    4,
                    "confirm-delete",
                    101,
                    101,
                    confirmation.message_id,  # type: ignore[union-attr]
                    f"td:{confirmation.token}:confirm",  # type: ignore[union-attr]
                )
            )
            self.assertEqual(codex.deleted_thread, "thread-to-delete")
            self.assertEqual(telegram.edits[-1][2], _catalog_text("thread_commands.label.deleted"))


class ModelEffortSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_model_settings_use_active_thread_before_global_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101,),
                workspace_root=workspace,
                codex_bin="codex",
                codex_model="gpt-6.2-sol",
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=workspace / "state.sqlite3",
            )
            store = RuntimeStateStore(config.database_file).for_user(101)
            store.save(RuntimeState(thread_id="thread-existing", cwd=str(workspace)))
            app_server = FakeAppServer(missing_thread=False)
            app_server.responses["model/list"] = {
                "data": [
                    {
                        "model": "gpt-6.2-sol",
                        "displayName": "GPT-6.2-Sol",
                        "isDefault": True,
                    },
                    {
                        "model": "gpt6-luna",
                        "displayName": "GPT-6-Luna",
                        "isDefault": False,
                        "defaultReasoningEffort": "medium",
                    },
                ],
                "nextCursor": None,
            }
            app_server.responses["thread/read"] = {
                "thread": {
                    "id": "thread-existing",
                    "model": "gpt6-luna",
                    "reasoningEffort": "high",
                }
            }

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                models = await session.list_models()
                model, effort = await session.current_model_settings(models)

            self.assertEqual(model, "gpt6-luna")
            self.assertEqual(effort, "high")
            self.assertIn(
                ("thread/read", {"threadId": "thread-existing"}),
                app_server.requests,
            )

            store.save(RuntimeState())
            model, effort = await session.current_model_settings(models)
            self.assertEqual(model, "gpt-6.2-sol")
            self.assertIsNone(effort)

    async def test_model_list_efforts_are_parsed_and_sent_on_turn_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
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
            store = RuntimeStateStore(config.database_file).for_user(101)
            app_server = FakeAppServer()
            app_server.responses["model/list"] = {
                "data": [
                    {
                        "model": "gpt-6-luna",
                        "displayName": "GPT-6-Luna",
                        "isDefault": True,
                        "defaultReasoningEffort": "medium",
                        "supportedReasoningEfforts": [
                            {"reasoningEffort": effort, "description": effort}
                            for effort in ("low", "medium", "high", "xhigh", "max")
                        ],
                    }
                ],
                "nextCursor": None,
            }
            app_server.responses["thread/start"] = {"thread": {"id": "thread-luna"}}
            app_server.responses["turn/start"] = {"turn": {"id": "turn-luna"}}

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                models = await session.list_models()
                await session.select_model("gpt-6-luna", "high")
                outcome = await session.send_message("Say OK")

            self.assertEqual(models[0].supported_reasoning_efforts, ("low", "medium", "high", "xhigh", "max"))
            self.assertEqual(models[0].default_reasoning_effort, "medium")
            self.assertEqual(outcome, "started")
            turn_params = next(params for method, params in app_server.requests if method == "turn/start")
            self.assertEqual(turn_params["model"], "gpt-6-luna")
            self.assertEqual(turn_params["effort"], "high")
            self.assertEqual(store.load().selected_effort, "high")

    async def test_thread_settings_update_saves_selected_effort(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
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
            store = RuntimeStateStore(config.database_file).for_user(101)
            store.save(RuntimeState(thread_id="thread-existing", cwd=str(workspace)))
            app_server = FakeAppServer(missing_thread=False)

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                await session.select_model("gpt-6-luna", "xhigh")

            setting_params = next(
                params for method, params in app_server.requests if method == "thread/settings/update"
            )
            self.assertEqual(setting_params["model"], "gpt-6-luna")
            self.assertEqual(setting_params["effort"], "xhigh")
            self.assertEqual(store.load().selected_effort, "xhigh")

    async def test_model_picker_uses_same_message_for_model_and_effort(self) -> None:
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
            codex = FakeCodex()
            codex.models = [
                CodexModel(
                    "gpt-6-luna",
                    "GPT-6-Luna",
                    True,
                    ("low", "medium", "high", "xhigh", "max"),
                    "medium",
                )
            ]
            controller = TelegramUserController(config, telegram, object(), 101)  # type: ignore[arg-type]
            controller.bind_codex(codex)  # type: ignore[arg-type]

            await controller._handle_message(IncomingMessage(1, 1, 101, 101, "private", "/model"))
            picker = controller.model_picker
            self.assertIsNotNone(picker)
            message_id = picker.message_id  # type: ignore[union-attr]
            token = picker.token  # type: ignore[union-attr]
            await controller._handle_callback(
                IncomingCallback(2, "choose-luna", 101, 101, message_id, f"m:{token}:s:0")
            )

            self.assertIsNone(codex.state.selected_model)
            self.assertEqual(
                telegram.edits[-1][2],
                _catalog_text(
                    "models.label.effort_prompt",
                    model="GPT-6-Luna",
                    default_effort=_catalog_text("models.buttons.effort_names.medium"),
                ),
            )
            effort_keyboard = telegram.edit_markups[-1]
            self.assertEqual(effort_keyboard["inline_keyboard"][0][0]["callback_data"], f"m:{token}:d")
            await controller._handle_callback(
                IncomingCallback(3, "choose-high", 101, 101, message_id, f"m:{token}:e:1")
            )

            self.assertEqual((codex.state.selected_model, codex.state.selected_effort), ("gpt-6-luna", "high"))
            self.assertEqual(telegram.edits[-1][1], message_id)

            await controller._handle_callback(
                IncomingCallback(4, "choose-luna-again", 101, 101, message_id, f"m:{token}:s:0")
            )
            await controller._handle_callback(
                IncomingCallback(5, "choose-default", 101, 101, message_id, f"m:{token}:d")
            )
            self.assertEqual((codex.state.selected_model, codex.state.selected_effort), ("gpt-6-luna", "medium"))

            await controller._handle_callback(
                IncomingCallback(6, "reset-model", 101, 101, message_id, f"m:{token}:r")
            )
            self.assertEqual((codex.state.selected_model, codex.state.selected_effort), (None, None))

    async def test_model_picker_uses_active_thread_model_before_global_default(self) -> None:
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
            codex = FakeCodex()
            codex.state.thread_id = "thread-luna"
            codex.thread_model = "gpt6-luna"
            codex.thread_effort = "high"
            codex.models = [
                CodexModel("gpt-6.2-sol", "GPT-6.2-Sol", True, ("low",), "low"),
                CodexModel("gpt6-luna", "GPT-6-Luna", False, ("low", "high"), "low"),
            ]
            controller = TelegramUserController(config, telegram, object(), 101)  # type: ignore[arg-type]
            controller.bind_codex(codex)  # type: ignore[arg-type]

            await controller._handle_message(
                IncomingMessage(1, 1, 101, 101, "private", "/model")
            )

            text = telegram.messages[-1][1]
            self.assertIn("GPT-6-Luna", text)
            self.assertNotIn("GPT-6.2-Sol", text)
            self.assertIn("Высокий (high)", text)
            labels = [
                button["text"]
                for row in telegram.reply_markups[-1]["inline_keyboard"]
                for button in row
            ]
            self.assertIn("› GPT-6-Luna ‹", labels)

    async def test_model_picker_uses_shared_page_counter_and_callbacks(self) -> None:
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
            codex = FakeCodex()
            codex.models = [
                CodexModel(f"model-{index}", f"Model {index}", index == 0, ("low",), "low")
                for index in range(9)
            ]
            controller = TelegramUserController(config, telegram, object(), 101)  # type: ignore[arg-type]
            controller.bind_codex(codex)  # type: ignore[arg-type]

            await controller._handle_message(
                IncomingMessage(1, 1, 101, 101, "private", "/model")
            )

            picker = controller.model_picker
            self.assertIsNotNone(picker)
            assert picker is not None
            self.assertNotIn("Страница 1/2", telegram.messages[-1][1])
            first_pager = telegram.reply_markups[-1]["inline_keyboard"][-1]
            self.assertEqual([button["text"] for button in first_pager], ["«", "1/2", "»"])
            self.assertEqual(first_pager[2]["callback_data"], f"m:{picker.token}:p:1")
            await controller._handle_callback(
                IncomingCallback(
                    2,
                    "current-model-page",
                    101,
                    101,
                    picker.message_id or 0,
                    first_pager[1]["callback_data"],
                )
            )
            self.assertEqual(picker.page, 0)
            self.assertEqual(telegram.edits, [])

            await controller._handle_callback(
                IncomingCallback(
                    3,
                    "next-model-page",
                    101,
                    101,
                    picker.message_id or 0,
                    first_pager[2]["callback_data"],
                )
            )
            second_pager = telegram.edit_markups[-1]["inline_keyboard"][-1]
            self.assertEqual([button["text"] for button in second_pager], ["«", "2/2", "»"])
            self.assertEqual(second_pager[0]["callback_data"], f"m:{picker.token}:p:0")

            await controller._handle_callback(
                IncomingCallback(
                    4,
                    "previous-model-page",
                    101,
                    101,
                    picker.message_id or 0,
                    second_pager[0]["callback_data"],
                )
            )
            self.assertEqual(
                [button["text"] for button in telegram.edit_markups[-1]["inline_keyboard"][-1]],
                ["«", "1/2", "»"],
            )


class PickerPaginationTests(unittest.TestCase):
    def test_two_column_pages_hold_ten_items_and_show_total(self) -> None:
        buttons = [
            {"text": f"item {index}", "callback_data": f"item:{index}"}
            for index in range(11)
        ]

        self.assertEqual(page_count(10), 1)
        self.assertEqual(page_count(11), 2)
        self.assertEqual([len(row) for row in two_column_rows(buttons, 0)], [2, 2, 2, 2, 2])
        self.assertEqual([len(row) for row in two_column_rows(buttons, 1)], [1])
        self.assertEqual(
            [button["text"] for button in navigation_row(0, 2, "<", "{current}/{total}", ">", "x")],
            ["<", "1/2", ">"],
        )
