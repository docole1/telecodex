from __future__ import annotations
from pathlib import Path
from typing import Any
from unittest.mock import patch
import asyncio
import tempfile
import unittest
from telecodex.codex.rpc import RpcError
from telecodex.codex.session import ActiveThreadWriterError, CodexServiceTier, CodexSession, CodexThread, MemoryFeatureUnavailableError, SelectedSkillUnavailableError
from telecodex.config import AppConfig
from telecodex.state import PendingTurnState, RuntimeState, RuntimeStateStore, ThreadRuntimeState
from telecodex.telegram.controller import TelegramController
from telecodex.telegram.updates import IncomingCallback, IncomingMessage
from telecodex.telegram.user_controller import TelegramUserController

from multi_user_test_support import (
    FakeAppServer,
    FakeCodex,
    FakeTelegram,
    _catalog_text,
)

class ActiveWriterRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_cli_thread_does_not_prevent_polling_startup(self) -> None:
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
            store = RuntimeStateStore(config.database_file)
            store.for_user(101).save(RuntimeState(thread_id="locked-thread", cwd=str(workspace)))
            telegram = FakeTelegram()
            codex = FakeCodex()

            async def start_with_locked_thread() -> bool:
                raise ActiveThreadWriterError("locked in CLI")

            codex.start = start_with_locked_thread  # type: ignore[method-assign]
            controller = TelegramController(
                config,
                telegram,
                object(),
                lambda _user_id, _controller: codex,
                state_store=store,
            )  # type: ignore[arg-type]

            await controller.start()
            self.assertTrue(codex.closed)
            await controller._handle_update({
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "from": {"id": 101},
                    "chat": {"id": 101, "type": "private"},
                    "text": "polling remains alive",
                },
            })
            self.assertEqual(codex.messages, ["polling remains alive"])
            await controller.close()

    def _controller(
        self,
        workspace: Path,
    ) -> tuple[TelegramUserController, FakeTelegram, FakeCodex]:
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
        codex.threads = [
            CodexThread(
                thread_id="thread-locked",
                name="CLI chat",
                preview=None,
                cwd=workspace,
                updated_at=None,
                source="cli",
            )
        ]
        codex.select_error = ActiveThreadWriterError("active writer")
        controller = TelegramUserController(config, telegram, object(), 101)  # type: ignore[arg-type]
        controller.bind_codex(codex)  # type: ignore[arg-type]
        return controller, telegram, codex

    async def test_selected_chat_name_is_wrapped_and_kept_as_a_full_width_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            controller, telegram, codex = self._controller(workspace)
            codex.state.thread_id = "thread-locked"

            await controller._handle_message(
                IncomingMessage(1, 1, 101, 101, "private", "/chats")
            )

            markup = telegram.reply_markups[-1]
            label = markup["inline_keyboard"][0][0]["text"]
            self.assertTrue(label.startswith("› CLI chat · "))
            self.assertTrue(label.endswith(" ‹"))
            self.assertEqual(len(markup["inline_keyboard"][0]), 1)

    @staticmethod
    def _button(markup: Any, action: str) -> str:
        for row in markup["inline_keyboard"]:
            for button in row:
                if f":{action}:" in button["callback_data"] or button["callback_data"].endswith(f":{action}"):
                    return button["callback_data"]
        raise AssertionError(f"No {action!r} callback button")

    async def _open_picker_and_trigger_conflict(
        self,
        controller: TelegramUserController,
        telegram: FakeTelegram,
    ) -> None:
        await controller._handle_message(IncomingMessage(1, 1, 101, 101, "private", "/chats"))
        detail_data = telegram.reply_markups[-1]["inline_keyboard"][0][0]["callback_data"]
        await controller._handle_callback(
            IncomingCallback(2, "open-chat-details", 101, 101, 1, detail_data)
        )
        select_data = telegram.edit_markups[-1]["inline_keyboard"][0][0]["callback_data"]
        await controller._handle_callback(
            IncomingCallback(3, "select-chat", 101, 101, 1, select_data)
        )

    async def test_active_writer_offers_retry_and_retry_selects_after_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            controller, telegram, codex = self._controller(workspace)
            await self._open_picker_and_trigger_conflict(controller, telegram)

            self.assertEqual(
                telegram.edits[-1][2],
                _catalog_text("chats.label.active_writer", label="CLI chat"),
            )
            recovery_markup = telegram.edit_markups[-1]
            retry_data = self._button(recovery_markup, "r")
            self._button(recovery_markup, "f")
            self.assertEqual(controller.chats_picker.blocked_thread_index, 0)  # type: ignore[union-attr]

            codex.select_error = None
            await controller._handle_callback(
                IncomingCallback(3, "retry-chat", 101, 101, 1, retry_data)
            )

            self.assertEqual(codex.select_calls, [("thread-locked", workspace), ("thread-locked", workspace)])
            self.assertEqual(codex.state.thread_id, "thread-locked")
            self.assertEqual(
                telegram.edits[-1][2],
                _catalog_text("chats.label.selected", label="CLI chat", path=workspace),
            )
            self.assertEqual(telegram.edit_markups[-1], {"inline_keyboard": []})

    async def test_active_writer_can_be_opened_as_a_copy_without_resuming_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            controller, telegram, codex = self._controller(workspace)
            await self._open_picker_and_trigger_conflict(controller, telegram)

            fork_data = self._button(telegram.edit_markups[-1], "f")
            await controller._handle_callback(
                IncomingCallback(3, "fork-chat", 101, 101, 1, fork_data)
            )

            self.assertEqual(codex.fork_by_id_calls, [("thread-locked", workspace)])
            self.assertEqual(codex.select_calls, [("thread-locked", workspace)])
            self.assertEqual(codex.state.thread_id, "thread-forked-from-picker")
            self.assertEqual(
                telegram.edits[-1][2],
                _catalog_text("chats.label.forked", label="CLI chat", path=workspace),
            )
            self.assertEqual(telegram.edit_markups[-1], {"inline_keyboard": []})


class MissingThreadRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def _build_session(
        self,
        workspace: Path,
        thread_id: str | None,
        *,
        on_status: Any,
        on_finished: Any,
    ) -> tuple[CodexSession, FakeAppServer, Any]:
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
        store.save(RuntimeState(thread_id=thread_id, cwd=str(workspace)))
        app_server = FakeAppServer(missing_thread=False)

        async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
            return {}

        async def no_command(_item: dict[str, Any]) -> None:
            return None

        with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
            session = CodexSession(config, store, on_status, no_command, on_finished, no_prompt)
        return session, app_server, store

    async def test_active_writer_has_specific_error_and_restores_previous_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            project = workspace / "project"
            project.mkdir()
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
            store.save(RuntimeState("thread-current", cwd=str(workspace)))
            app_server = FakeAppServer(missing_thread=False, active_writer_on_resume=True)

            async def no_status(_text: str) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                with self.assertRaises(ActiveThreadWriterError):
                    await session.select_thread("thread-locked", project)

            self.assertEqual(session.state.thread_id, "thread-current")
            self.assertEqual(session.state.cwd, str(workspace))
            self.assertEqual([method for method, _params in app_server.requests], ["thread/resume"])

    async def test_fork_by_id_does_not_resume_locked_source_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            project = workspace / "project"
            project.mkdir()
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
            store.save(RuntimeState("thread-current", cwd=str(workspace)))
            app_server = FakeAppServer(missing_thread=False)
            app_server.responses["thread/fork"] = {
                "thread": {"id": "thread-copy", "cwd": str(project)}
            }

            async def no_status(_text: str) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                forked_id = await session.fork_thread_by_id("thread-locked", project)

            self.assertEqual(forked_id, "thread-copy")
            self.assertEqual(
                [method for method, _params in app_server.requests],
                [
                    "thread/goal/get",
                    "thread/fork",
                    "thread/goal/get",
                    "thread/backgroundTerminals/list",
                    "thread/unsubscribe",
                ],
            )
            self.assertEqual(store.load().thread_id, "thread-copy")
            self.assertEqual(store.load().cwd, str(project))

    async def test_missing_codex_rollout_is_replaced_with_a_new_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            project = workspace / "selected"
            project.mkdir()
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
            store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3").for_user(101)
            store.save(RuntimeState("thread-gone", cwd=str(project)))
            app_server = FakeAppServer()

            async def no_status(_text: str) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                await session.start()

            self.assertEqual(session.state.thread_id, "thread-fresh")
            self.assertEqual(
                [method for method, _params in app_server.requests],
                [
                    "thread/resume",
                    "thread/start",
                    "thread/backgroundTerminals/list",
                    "thread/unsubscribe",
                ],
            )
            start_params = next(params for method, params in app_server.requests if method == "thread/start")
            self.assertEqual(start_params["cwd"], str(project))
            self.assertEqual(store.load().thread_id, "thread-fresh")
            self.assertEqual(store.load().cwd, str(project))

    async def test_resume_retries_while_codex_is_closing_a_thread(self) -> None:
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
            store.save(RuntimeState("thread-existing", cwd=str(workspace)))
            app_server = FakeAppServer(closing_resume_failures=2, missing_thread=False)

            async def no_status(_text: str) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                await session.start()

            resumes = [method for method, _params in app_server.requests if method == "thread/resume"]
            self.assertEqual(len(resumes), 3)
            self.assertEqual(session.state.thread_id, "thread-existing")

    async def test_new_thread_defers_creation_and_keeps_selected_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            project = workspace / "selected"
            project.mkdir()
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

            async def no_status(_text: str) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                await session.new_thread(project)

            self.assertIsNone(store.load().thread_id)
            self.assertEqual(store.load().cwd, str(project))
            self.assertEqual(app_server.requests, [])

    async def test_new_thread_uses_saved_permission_mode(self) -> None:
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
            state = store.load()
            state.default_permission_mode = "read_only"
            store.save(state)
            app_server = FakeAppServer()

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                await session._ensure_thread()

            start_params = next(params for method, params in app_server.requests if method == "thread/start")
            self.assertEqual(start_params["permissions"], "telecodex_read_only")
            self.assertEqual(start_params["approvalsReviewer"], "user")
            self.assertEqual(
                store.load_thread("thread-fresh"),
                ThreadRuntimeState(permission_mode="read_only", managed_by_telecodex=True),
            )

    async def test_existing_thread_mode_is_applied_to_next_turn(self) -> None:
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
            store.save_thread(
                "thread-existing",
                ThreadRuntimeState(permission_mode="auto_review", managed_by_telecodex=True),
            )
            app_server = FakeAppServer(missing_thread=False)

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                with self.assertRaisesRegex(RuntimeError, "did not return a turn ID"):
                    await session.send_message("harmless check")

            turn_params = next(params for method, params in app_server.requests if method == "turn/start")
            self.assertEqual(turn_params["approvalsReviewer"], "auto_review")
            self.assertEqual(turn_params["permissions"], "telecodex_ask")

    async def test_token_usage_notification_stores_latest_turn_context(self) -> None:
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

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=FakeAppServer()):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                session._on_event(
                    {
                        "method": "thread/tokenUsage/updated",
                        "params": {
                            "threadId": "shared-thread",
                            "tokenUsage": {
                                "last": {"totalTokens": 51500},
                                "total": {"totalTokens": 120000},
                                "modelContextWindow": 258000,
                            },
                        },
                    }
                )

            self.assertEqual(
                store.load_thread("shared-thread").token_usage,
                {"context_tokens": 51500, "model_context_window": 258000},
            )

    async def test_plan_mode_toggle_preserves_thread_model_and_effort(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            session, app_server, _store = self._build_session(
                workspace,
                "thread-plan",
                on_status=no_status,
                on_finished=no_status,
            )
            app_server.response_sequences["thread/resume"] = [
                {
                    "thread": {"id": "thread-plan", "cwd": str(workspace)},
                    "model": "gpt-6-luna",
                    "reasoningEffort": "high",
                    "collaborationMode": {"mode": "default", "settings": {"model": "gpt-6-luna"}},
                },
                {
                    "thread": {"id": "thread-plan", "cwd": str(workspace)},
                    "model": "gpt-6-luna",
                    "reasoningEffort": "high",
                    "collaborationMode": {"mode": "plan", "settings": {"model": "gpt-6-luna"}},
                },
            ]
            app_server.responses["collaborationMode/list"] = {
                "data": [
                    {"name": "Plan", "mode": "plan", "model": "gpt-6-luna"},
                    {"name": "Default", "mode": "default", "model": "gpt-6-luna"},
                ]
            }

            self.assertEqual(await session.toggle_plan_mode(), "plan")
            self.assertEqual(await session.toggle_plan_mode(), "default")

            updates = [params for method, params in app_server.requests if method == "thread/settings/update"]
            self.assertEqual(len(updates), 2)
            for params, mode in zip(updates, ("plan", "default"), strict=True):
                self.assertEqual(params["threadId"], "thread-plan")
                self.assertEqual(
                    params["collaborationMode"],
                    {
                        "mode": mode,
                        "settings": {
                            "model": "gpt-6-luna",
                            "reasoning_effort": "high",
                            "developer_instructions": None,
                        },
                    },
                )

    async def test_plan_mode_for_new_thread_keeps_it_loaded_for_its_first_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, _store = self._build_session(
                workspace,
                None,
                on_status=no_status,
                on_finished=no_finished,
            )
            app_server.responses["turn/start"] = {"turn": {"id": "turn-first"}}

            await session.set_collaboration_mode("plan", create_if_missing=True)
            await session.send_message("Give a short plan")

            self.assertEqual(session.state.thread_id, "thread-fresh")
            self.assertEqual(session.collaboration_mode, "plan")
            self.assertNotIn("thread/resume", [method for method, _params in app_server.requests])
            self.assertIn("thread/settings/update", [method for method, _params in app_server.requests])
            turn_params = next(params for method, params in app_server.requests if method == "turn/start")
            self.assertEqual(turn_params["collaborationMode"]["mode"], "plan")
            self.assertEqual(turn_params["collaborationMode"]["settings"]["developer_instructions"], None)

    async def test_plan_item_is_preferred_for_final_delivery_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            session, app_server, _store = self._build_session(
                Path(temp_dir),
                "thread-plan",
                on_status=no_status,
                on_finished=no_status,
            )
            session._on_event(
                {"method": "turn/started", "params": {"threadId": "thread-plan", "turn": {"id": "turn-plan"}}}
            )
            session._on_event(
                {
                    "method": "item/plan/delta",
                    "params": {
                        "threadId": "thread-plan",
                        "turnId": "turn-plan",
                        "itemId": "plan-item",
                        "delta": "Proposed plan",
                    },
                }
            )
            session._on_event(
                {
                    "method": "item/completed",
                    "params": {"item": {"type": "plan", "id": "plan-item", "text": "Authoritative plan"}},
                }
            )
            self.assertEqual(session._final_message(), "Authoritative plan")

            app_server.responses["thread/items/list"] = {
                "data": [
                    {"item": {"type": "agentMessage", "id": "answer", "phase": "final_answer", "text": "fallback"}},
                    {"item": {"type": "plan", "id": "plan-item", "text": "Recovered plan"}},
                ],
                "nextCursor": None,
            }
            self.assertEqual(await session._read_turn_text("thread-plan", "turn-plan"), "Recovered plan")

    async def test_review_item_is_recovered_when_no_assistant_final_item_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            async def no_status(_key: str, _details: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, _store = self._build_session(
                Path(temp_dir),
                "thread-review-recovery",
                on_status=no_status,
                on_finished=no_finished,
            )
            app_server.responses["thread/items/list"] = {
                "data": [
                    {
                        "item": {
                            "type": "agentMessage",
                            "id": "empty-final",
                            "phase": "final_answer",
                            "text": "",
                        }
                    },
                    {
                        "item": {
                            "type": "exitedReviewMode",
                            "id": "review-turn",
                            "review": "Recovered review findings.",
                        }
                    }
                ],
                "nextCursor": None,
            }

            self.assertEqual(
                await session._read_turn_text("thread-review-recovery", "review-turn"),
                "Recovered review findings.",
            )

    async def test_new_thread_applies_selected_speed_and_memory_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, store = self._build_session(
                workspace,
                None,
                on_status=no_status,
                on_finished=no_finished,
            )
            session.state.selected_model = "gpt-6-luna"
            session.state.selected_service_tier = "priority"
            session.state.selected_memory_mode = "enabled"
            store.save(session.state)
            app_server.responses["thread/start"] = {
                "thread": {"id": "thread-fresh", "cwd": str(workspace)},
                "model": "gpt-6-luna",
                "reasoningEffort": "high",
                "serviceTier": "priority",
            }
            app_server.responses["turn/start"] = {"turn": {"id": "turn-fresh"}}

            await session.send_message("Use the chosen settings")

            start_params = next(params for method, params in app_server.requests if method == "thread/start")
            turn_params = next(params for method, params in app_server.requests if method == "turn/start")
            self.assertEqual(start_params["serviceTier"], "priority")
            self.assertEqual(turn_params["serviceTier"], "priority")
            self.assertIn(
                ("thread/memoryMode/set", {"threadId": "thread-fresh", "mode": "enabled"}),
                app_server.requests,
            )
            self.assertEqual(store.load_thread("thread-fresh").memory_mode, "enabled")

    async def test_selected_skill_is_injected_into_the_next_turn_and_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            statuses: list[tuple[str, object | None]] = []

            async def on_status(
                key: str, details: object = None, _item_id: str | None = None
            ) -> None:
                statuses.append((key, details))

            async def no_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, store = self._build_session(
                workspace,
                "thread-skills",
                on_status=on_status,
                on_finished=no_finished,
            )
            session.state.pending_skill_name = "reviewer"
            session.state.pending_skill_path = str(workspace / ".agents/skills/reviewer/SKILL.md")
            store.save(session.state)
            app_server.skills = [{
                "name": "reviewer",
                "description": "Review changes",
                "path": str(workspace / ".agents/skills/reviewer/SKILL.md"),
                "enabled": True,
                "interface": {"displayName": "Reviewer", "shortDescription": "Review changes"},
            }]
            app_server.responses["turn/start"] = {"turn": {"id": "turn-skill"}}

            await session.send_message("Review this change")

            turn_params = next(params for method, params in app_server.requests if method == "turn/start")
            self.assertEqual(turn_params["input"][0]["text"], "$reviewer\n\nReview this change")
            self.assertEqual(
                turn_params["input"][1],
                {"type": "skill", "name": "reviewer", "path": str(workspace / ".agents/skills/reviewer/SKILL.md")},
            )
            self.assertIsNone(store.load().pending_skill_name)
            self.assertIsNone(store.load().pending_skill_path)
            await asyncio.sleep(0)
            self.assertEqual(statuses, [("detailed_log.values.skill_attached", "Reviewer")])

    async def test_skill_attached_is_reported_after_accepted_steering_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            statuses: list[tuple[str, object | None]] = []

            async def on_status(
                key: str, details: object = None, _item_id: str | None = None
            ) -> None:
                statuses.append((key, details))

            async def no_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, store = self._build_session(
                workspace,
                "thread-steer-skill",
                on_status=on_status,
                on_finished=no_finished,
            )
            app_server.skills = [{
                "name": "docs",
                "description": "Documentation",
                "path": str(workspace / "docs/SKILL.md"),
                "enabled": True,
                "interface": {"displayName": "OpenAI Docs"},
            }]
            session.state.pending_skill_name = "docs"
            session.state.pending_skill_path = str(workspace / "docs/SKILL.md")
            session.active_turn_id = "turn-steer"
            session.state.pending_turn = PendingTurnState(
                thread_id="thread-steer-skill",
                turn_id="turn-steer",
                prompt_message_id=None,
                started_at=1,
                phase="running",
            )
            store.save(session.state)
            app_server.is_running = True

            self.assertEqual(await session.send_message("Continue with the docs"), "steered")
            await asyncio.sleep(0)

            self.assertEqual(statuses, [("detailed_log.values.skill_attached", "OpenAI Docs")])
            self.assertIsNone(store.load().pending_skill_name)
            self.assertEqual(app_server.requests[-1][0], "turn/steer")

    async def test_skill_is_not_reported_when_turn_start_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            statuses: list[tuple[str, object | None]] = []

            async def on_status(
                key: str, details: object = None, _item_id: str | None = None
            ) -> None:
                statuses.append((key, details))

            async def no_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, store = self._build_session(
                workspace,
                "thread-rejected-skill",
                on_status=on_status,
                on_finished=no_finished,
            )
            skill_path = str(workspace / "docs/SKILL.md")
            session.state.pending_skill_name = "docs"
            session.state.pending_skill_path = skill_path
            store.save(session.state)
            app_server.skills = [{
                "name": "docs",
                "description": "Documentation",
                "path": skill_path,
                "enabled": True,
                "interface": {"displayName": "OpenAI Docs"},
            }]
            app_server.missing_thread = False
            app_server.response_errors["turn/start"] = RpcError(-32000, "rejected")

            with self.assertRaises(RpcError):
                await session.send_message("Check the docs")
            await asyncio.sleep(0)

            self.assertEqual(statuses, [])
            self.assertEqual(store.load().pending_skill_name, "docs")

    async def test_memory_mode_is_per_thread_and_memory_generation_updates_codex_home(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, store = self._build_session(
                workspace,
                "thread-memory",
                on_status=no_status,
                on_finished=no_finished,
            )

            await session.set_memory_mode("disabled")
            self.assertEqual(store.load_thread("thread-memory").memory_mode, "disabled")
            self.assertIn(
                ("thread/memoryMode/set", {"threadId": "thread-memory", "mode": "disabled"}),
                app_server.requests,
            )

            self.assertTrue(await session.toggle_memory_generation())
            write = next(params for method, params in app_server.requests if method == "config/batchWrite")
            self.assertEqual(write["expectedVersion"], "config-v1")
            self.assertEqual(
                write["edits"],
                [{"keyPath": "memories.generate_memories", "mergeStrategy": "replace", "value": True}],
            )
            self.assertTrue(app_server.codex_config["memories"]["generate_memories"])

    async def test_fast_tier_uses_model_catalog_and_explicit_standard_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            session, app_server, store = self._build_session(
                workspace,
                "thread-fast",
                on_status=no_status,
                on_finished=no_status,
            )
            app_server.response_sequences["thread/resume"] = [
                {
                    "thread": {"id": "thread-fast", "cwd": str(workspace)},
                    "model": "gpt-6-luna",
                    "reasoningEffort": "high",
                    "serviceTier": "default",
                },
                {
                    "thread": {"id": "thread-fast", "cwd": str(workspace)},
                    "model": "gpt-6-luna",
                    "reasoningEffort": "high",
                    "serviceTier": "priority",
                },
            ]

            models = await session.list_models()
            self.assertEqual(
                models[0].service_tiers,
                (CodexServiceTier("priority", "Fast", "fast inference"),),
            )
            self.assertTrue(await session.toggle_fast_mode())
            self.assertFalse(await session.toggle_fast_mode())

            tier_updates = [
                params["serviceTier"]
                for method, params in app_server.requests
                if method == "thread/settings/update" and "serviceTier" in params
            ]
            self.assertEqual(tier_updates, ["priority", "default"])
            self.assertEqual(store.load().selected_service_tier, "default")

    async def test_fast_tier_can_be_selected_before_the_first_codex_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            session, app_server, store = self._build_session(
                Path(temp_dir),
                None,
                on_status=no_status,
                on_finished=no_status,
            )
            session.state.selected_model = "gpt-6-luna"

            self.assertTrue(await session.toggle_fast_mode())

            self.assertIsNone(session.state.thread_id)
            self.assertEqual(store.load().selected_service_tier, "priority")
            self.assertFalse(any(method == "thread/start" for method, _params in app_server.requests))

    async def test_memory_mode_rejects_disabled_feature(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            session, _app_server, store = self._build_session(
                Path(temp_dir),
                None,
                on_status=no_status,
                on_finished=no_status,
            )
            session.app_server.codex_config["features"]["memories"] = False
            settings = await session.read_memory_settings()
            self.assertFalse(settings.feature_enabled)
            with self.assertRaises(MemoryFeatureUnavailableError):
                await session.set_memory_mode("enabled")
            with self.assertRaises(MemoryFeatureUnavailableError):
                await session.reset_memories()
            self.assertIsNone(store.load().selected_memory_mode)

    async def test_reset_memories_uses_the_codex_app_server_api(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            session, app_server, _store = self._build_session(
                Path(temp_dir),
                None,
                on_status=no_status,
                on_finished=no_status,
            )

            await session.reset_memories()

            self.assertIn(("memory/reset", {}), app_server.requests)

    async def test_memory_generation_retries_after_a_config_version_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            session, app_server, _store = self._build_session(
                Path(temp_dir),
                None,
                on_status=no_status,
                on_finished=no_status,
            )
            app_server.response_error_sequences["config/batchWrite"] = [
                RpcError(-32602, "Config version conflict; reread and retry")
            ]

            self.assertTrue(await session.toggle_memory_generation())
            writes = [method for method, _params in app_server.requests if method == "config/batchWrite"]
            self.assertEqual(len(writes), 2)
            self.assertTrue(app_server.codex_config["memories"]["generate_memories"])

    async def test_selected_skill_unavailable_is_not_sent_as_a_plain_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, store = self._build_session(
                workspace,
                "thread-skill-stale",
                on_status=no_status,
                on_finished=no_finished,
            )
            session.state.pending_skill_name = "missing-skill"
            session.state.pending_skill_path = str(workspace / "missing/SKILL.md")
            store.save(session.state)

            with self.assertRaises(SelectedSkillUnavailableError):
                await session.send_message("Do the task")

            self.assertIsNone(store.load().pending_skill_name)
            self.assertFalse(any(method == "turn/start" for method, _params in app_server.requests))

    async def test_status_includes_thread_account_context_and_quota_data(self) -> None:
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
            store.save(RuntimeState(thread_id="thread-status", cwd=str(workspace)))
            store.save_thread(
                "thread-status",
                ThreadRuntimeState(
                    managed_by_telecodex=True,
                    token_usage={"context_tokens": 51500, "model_context_window": 258000},
                ),
            )
            app_server = FakeAppServer()
            app_server.missing_thread = False
            app_server.responses["thread/read"] = {
                "thread": {"name": "Release prep", "model": "gpt-6.1-sol", "cwd": str(workspace)}
            }
            app_server.responses["account/rateLimits/read"] = {
                "rateLimitsByLimitId": {
                    "codex": {
                        "primary": {"usedPercent": 63, "windowDurationMins": 300, "resetsAt": 0},
                        "secondary": {"usedPercent": 18, "windowDurationMins": 10080, "resetsAt": 0},
                    }
                }
            }

            async def no_status(_text: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def no_prompt(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
                return {}

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
                session = CodexSession(config, store, no_status, no_status, no_status, no_prompt)
                status = await session.status_text()

            expected_status = _catalog_text(
                "status.label.summary",
                activity=_catalog_text("status.values.ready"),
                title="Release prep",
                model=f"gpt-6.1-sol ({_catalog_text('status.values.effort_unknown')})",
                service_tier=_catalog_text("status.values.speed_standard"),
                collaboration_mode=_catalog_text("status.values.collaboration_mode_default"),
                project=str(workspace),
                permissions=_catalog_text("permissions.buttons.mode_names.ask"),
                account="codex@example.com (Plus)",
                context=_catalog_text("status.values.context_usage", percent=20, used="52K", window="258K"),
                short_limit=_catalog_text("status.values.rate_limit", remaining=37, reset="01.01 03:00"),
                weekly_limit=_catalog_text("status.values.rate_limit", remaining=82, reset="01.01 03:00"),
            )
            self.assertEqual(status, expected_status)
