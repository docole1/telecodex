from __future__ import annotations
from pathlib import Path
from typing import Any
from unittest.mock import patch
import asyncio
import tempfile
import unittest
from telecodex.codex.command_prompts import GOAL_CONTINUATION_PROMPT
from telecodex.codex.rpc import RpcError
from telecodex.codex.session import ActiveThreadWriterError, CodexChatBusyError, CodexGoal, CodexSession, GoalNotFoundError, GoalUnavailableError
from telecodex.config import AppConfig
from telecodex.state import PendingTurnState, RuntimeState, RuntimeStateStore, ThreadRuntimeState

from multi_user_test_support import FakeAppServer

class ThreadLifecycleSessionTests(unittest.IsolatedAsyncioTestCase):
    def _build_session(
        self,
        workspace: Path,
        thread_id: str | None,
        *,
        on_status: Any,
        on_finished: Any,
        on_assistant_message: Any = None,
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
            session = CodexSession(
                config,
                store,
                on_status,
                no_command,
                on_finished,
                no_prompt,
                on_assistant_message=on_assistant_message,
            )
        return session, app_server, store

    async def test_command_execution_activity_is_tracked_until_item_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session, _app_server, _store = self._build_session(
                Path(temp_dir),
                "thread-command-tracking",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )

            session._on_event(
                {
                    "method": "item/started",
                    "params": {
                        "item": {
                            "id": "command-1",
                            "type": "commandExecution",
                            "command": "sleep 600",
                        }
                    },
                }
            )
            self.assertTrue(session.has_active_command_execution)
            self.assertEqual(session.active_command_executions, {"command-1"})

            session._on_event(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "id": "command-1",
                            "type": "commandExecution",
                            "status": "completed",
                            "exitCode": 0,
                        }
                    },
                }
            )
            self.assertFalse(session.has_active_command_execution)
            await asyncio.sleep(0)

    async def test_waiting_for_turn_completion_has_a_bound_and_preserves_live_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session, _app_server, _store = self._build_session(
                Path(temp_dir),
                "thread-wait-bound",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            stalled_task = asyncio.create_task(asyncio.Event().wait())
            session._turn_completion_task = stalled_task

            with patch("telecodex.codex.session.TURN_COMPLETION_WAIT_SECONDS", 0.01):
                with self.assertRaisesRegex(RuntimeError, "completion delivery is still pending"):
                    await session._wait_for_turn_completion()

            self.assertFalse(stalled_task.done())
            stalled_task.cancel()
            await asyncio.gather(stalled_task, return_exceptions=True)

    async def test_confirmed_switch_preserves_terminal_result_reported_by_codex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            finished: list[dict[str, Any]] = []
            session_ref: dict[str, CodexSession] = {}

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)
                session_ref["session"].clear_pending_turn()

            session, app_server, _store = self._build_session(
                workspace,
                "thread-terminal-switch",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=on_finished,
            )
            session_ref["session"] = session
            session.state.pending_turn = PendingTurnState(
                "thread-terminal-switch", "turn-terminal-switch", 10, 1, phase="running"
            )
            session.active_turn_id = "turn-terminal-switch"
            app_server.is_running = True
            app_server.responses["thread/turns/list"] = {
                "data": [{"id": "turn-terminal-switch", "status": "interrupted"}],
                "nextCursor": None,
            }
            app_server.responses["thread/items/list"] = {"data": [], "nextCursor": None}
            original_request = app_server.request

            async def request_with_completion_event(
                method: str,
                params: dict[str, Any] | None = None,
                timeout_seconds: int = 90,
            ) -> dict[str, Any]:
                result = await original_request(method, params, timeout_seconds)
                if method == "turn/interrupt":
                    session._on_event(
                        {
                            "method": "turn/completed",
                            "params": {
                                "threadId": "thread-terminal-switch",
                                "turn": {"id": "turn-terminal-switch", "status": "interrupted"},
                            },
                        }
                    )
                return result

            app_server.request = request_with_completion_event  # type: ignore[method-assign]

            result = await session.interrupt_and_abandon_for_switch()

            self.assertEqual(result, "interrupted")
            self.assertTrue(app_server.is_running)
            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0]["status"], "interrupted")
            self.assertIsNone(session.pending_turn)
            self.assertEqual(session.state.thread_id, "thread-terminal-switch")
            self.assertIsNone(session._turn_completion_task)

    async def test_confirmed_switch_marks_unconfirmed_result_unknown_and_never_replays(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            finished: list[dict[str, Any]] = []
            session_ref: dict[str, CodexSession] = {}

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)
                session_ref["session"].clear_pending_turn()

            session, app_server, _store = self._build_session(
                workspace,
                "thread-stuck-switch",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=on_finished,
            )
            session_ref["session"] = session
            session.state.pending_turn = PendingTurnState(
                "thread-stuck-switch", "turn-stuck-switch", 10, 1, phase="running"
            )
            session.active_turn_id = "turn-stuck-switch"
            session.active_command_executions.add("command-sleep")
            app_server.is_running = True
            app_server.responses["thread/turns/list"] = {
                "data": [{"id": "turn-stuck-switch", "status": "inProgress"}],
                "nextCursor": None,
            }

            result = await session.interrupt_and_abandon_for_switch()

            self.assertEqual(result, "unknown")
            self.assertFalse(app_server.is_running)
            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0]["status"], "unknown")
            self.assertIsNone(session.pending_turn)
            self.assertEqual(session.state.thread_id, "thread-stuck-switch")
            self.assertEqual(session.active_command_executions, set())
            self.assertNotIn("turn/start", [method for method, _params in app_server.requests])

    async def test_goal_rpc_lifecycle_persists_thread_state_without_custom_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, _store = self._build_session(
                workspace,
                None,
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            continuation_prompts: list[tuple[str, int | None]] = []

            async def send_goal_continuation(text: str, prompt_message_id: int | None = None, **_kwargs: Any) -> str:
                continuation_prompts.append((text, prompt_message_id))
                return "started"

            session.send_message = send_goal_continuation  # type: ignore[method-assign]

            created = await session.set_goal("Finish the migration and verify it", prompt_message_id=91)
            self.assertEqual(created.status, "active")
            self.assertEqual(session.state.thread_id, "thread-fresh")
            self.assertEqual(created.objective, "Finish the migration and verify it")
            set_request = next(
                params for method, params in app_server.requests if method == "thread/goal/set"
            )
            self.assertEqual(set_request["origin"], "user")
            self.assertNotIn("tokenBudget", set_request)

            loaded = await session.get_goal()
            self.assertEqual(loaded, created)
            await session.set_goal_status("paused")
            self.assertEqual((await session.get_goal()).status, "paused")  # type: ignore[union-attr]

            await session.set_goal_status("active", prompt_message_id=92)
            self.assertEqual((await session.get_goal()).status, "active")  # type: ignore[union-attr]
            self.assertEqual(
                continuation_prompts,
                [(GOAL_CONTINUATION_PROMPT, 91), (GOAL_CONTINUATION_PROMPT, 92)],
            )
            self.assertTrue(await session.clear_goal())
            self.assertIsNone(await session.get_goal())
            methods = [method for method, _params in app_server.requests]
            self.assertIn("thread/goal/get", methods)
            self.assertIn("thread/goal/clear", methods)
            self.assertIsNone(session.pending_turn)

    async def test_startup_keeps_active_goal_thread_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, _store = self._build_session(
                workspace,
                "thread-goal-resume",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            app_server.goals["thread-goal-resume"] = {
                "threadId": "thread-goal-resume",
                "objective": "Resume the long-running task",
                "status": "active",
                "tokensUsed": 42,
                "timeUsedSeconds": 3,
            }

            await session.start()

            self.assertTrue(session._goal_is_active())
            self.assertEqual(session._loaded_thread_id, "thread-goal-resume")
            self.assertFalse(any(method == "thread/unsubscribe" for method, _ in app_server.requests))

    async def test_goal_methods_report_when_app_server_has_no_goal_api(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, _store = self._build_session(
                workspace,
                "thread-no-goal-api",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            app_server.response_errors["thread/goal/get"] = RpcError(-32601, "Method not found")

            with self.assertRaises(GoalUnavailableError):
                await session.get_goal()

    async def test_goal_pause_requires_an_existing_goal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, _app_server, _store = self._build_session(
                workspace,
                "thread-without-goal",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )

            with self.assertRaises(GoalNotFoundError):
                await session.set_goal_status("paused")

    async def test_clear_goal_resumes_thread_after_stale_loaded_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, _store = self._build_session(
                workspace,
                "thread-goal-clear-retry",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            app_server.goals["thread-goal-clear-retry"] = {
                "threadId": "thread-goal-clear-retry",
                "objective": "Finish the task",
                "status": "complete",
            }
            app_server.response_error_sequences["thread/goal/clear"] = [
                RpcError(-32603, "failed to record goal instruction: thread id not found")
            ]
            session._goal = CodexGoal("Finish the task", "complete", None, 10, 3)
            session._goal_thread_id = "thread-goal-clear-retry"
            session._loaded_thread_id = "thread-goal-clear-retry"
            await app_server.start()

            self.assertTrue(await session.clear_goal())

            methods = [method for method, _params in app_server.requests]
            self.assertIn("thread/resume", methods)
            self.assertEqual(methods.count("thread/goal/clear"), 2)
            self.assertIsNone(await session.get_goal())

    async def test_stop_pauses_idle_goal_without_interrupting_a_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, _store = self._build_session(
                workspace,
                "thread-idle-goal",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            app_server.goals["thread-idle-goal"] = {
                "threadId": "thread-idle-goal",
                "objective": "Continue until done",
                "status": "active",
            }
            session._goal = CodexGoal("Continue until done", "active", None, 0, 0)
            session._goal_thread_id = "thread-idle-goal"
            session._loaded_thread_id = "thread-idle-goal"
            await app_server.start()

            self.assertTrue(await session.interrupt())

            self.assertEqual(app_server.goals["thread-idle-goal"]["status"], "paused")
            self.assertFalse(any(method == "turn/interrupt" for method, _ in app_server.requests))

    async def test_stop_pauses_goal_and_interrupts_its_current_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, store = self._build_session(
                workspace,
                "thread-running-goal",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            app_server.goals["thread-running-goal"] = {
                "threadId": "thread-running-goal",
                "objective": "Continue until done",
                "status": "active",
            }
            session._goal = CodexGoal("Continue until done", "active", None, 0, 0)
            session._goal_thread_id = "thread-running-goal"
            session._loaded_thread_id = "thread-running-goal"
            session.active_turn_id = "goal-turn"
            session.state.pending_turn = PendingTurnState(
                thread_id="thread-running-goal",
                turn_id="goal-turn",
                prompt_message_id=89,
                started_at=123,
                phase="running",
            )
            store.save(session.state)
            await app_server.start()

            self.assertTrue(await session.interrupt())

            self.assertEqual(app_server.goals["thread-running-goal"]["status"], "paused")
            self.assertIn(
                ("turn/interrupt", {"threadId": "thread-running-goal", "turnId": "goal-turn"}),
                app_server.requests,
            )

    async def test_goal_auto_turn_is_tracked_and_active_goal_keeps_thread_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            finished: list[dict[str, Any]] = []
            status_events: list[str] = []

            async def on_status(key: str, _details: object = None, _item_id: str | None = None) -> None:
                status_events.append(key)

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            session, app_server, store = self._build_session(
                workspace,
                "thread-goal-active",
                on_status=on_status,
                on_finished=on_finished,
            )
            app_server.goals["thread-goal-active"] = {
                "threadId": "thread-goal-active",
                "objective": "Finish the task",
                "status": "active",
                "tokensUsed": 0,
                "timeUsedSeconds": 0,
            }
            session._goal = CodexGoal("Finish the task", "active", None, 0, 0)
            session._goal_thread_id = "thread-goal-active"
            await app_server.start()
            session._loaded_thread_id = "thread-goal-active"

            session._on_event(
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": "thread-goal-active",
                        "turn": {"id": "goal-turn-1", "rootTurnId": "goal-turn-1"},
                    },
                }
            )
            self.assertIsNotNone(session.pending_turn)
            self.assertEqual(session.pending_turn.turn_id, "goal-turn-1")  # type: ignore[union-attr]
            self.assertEqual(session.pending_turn.phase, "running")  # type: ignore[union-attr]

            session._on_event(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-goal-active",
                        "turn": {"id": "goal-turn-1", "status": "completed"},
                    },
                }
            )
            await session._wait_for_turn_completion()
            await asyncio.sleep(0)

            self.assertEqual(finished[0]["status"], "completed")
            self.assertTrue(session._goal_is_active())
            self.assertFalse(any(method == "thread/unsubscribe" for method, _ in app_server.requests))
            self.assertIn("status.goal_continuing", status_events)
            self.assertIsNotNone(store.load().pending_turn)

            app_server.goals["thread-goal-active"]["status"] = "complete"
            session._on_event(
                {
                    "method": "thread/goal/updated",
                    "params": {
                        "threadId": "thread-goal-active",
                        "goal": app_server.goals["thread-goal-active"],
                    },
                }
            )
            await session._unsubscribe_thread()
            self.assertTrue(any(method == "thread/unsubscribe" for method, _ in app_server.requests))
            await asyncio.sleep(0)

    async def test_goal_auto_continuation_replaces_previous_delivery_pending_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, _app_server, store = self._build_session(
                workspace,
                "thread-goal-continued",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            session._goal = CodexGoal("Finish the task", "active", None, 12, 2)
            session._goal_thread_id = "thread-goal-continued"
            session.state.pending_turn = PendingTurnState(
                thread_id="thread-goal-continued",
                turn_id="goal-turn-1",
                prompt_message_id=93,
                started_at=123,
                phase="delivery_pending",
                result_status="completed",
            )
            session.app_server.is_running = True
            store.save(session.state)

            session._on_event(
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": "thread-goal-continued",
                        "turn": {"id": "goal-turn-2", "rootTurnId": "goal-turn-2"},
                    },
                }
            )
            await asyncio.sleep(0)

            pending = session.pending_turn
            self.assertIsNotNone(pending)
            self.assertEqual(pending.turn_id, "goal-turn-2")  # type: ignore[union-attr]
            self.assertEqual(pending.phase, "running")  # type: ignore[union-attr]
            self.assertIsNone(pending.result_status)  # type: ignore[union-attr]
            self.assertTrue(session.is_busy)

    async def test_fork_defers_and_restarts_a_copied_active_goal_after_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, _store = self._build_session(
                workspace,
                "thread-goal-source",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            app_server.goals["thread-goal-source"] = {
                "threadId": "thread-goal-source",
                "objective": "Complete the release task",
                "status": "active",
                "tokensUsed": 5,
                "timeUsedSeconds": 1,
            }
            app_server.goals["thread-goal-copy"] = dict(app_server.goals["thread-goal-source"])
            app_server.goals["thread-goal-copy"]["threadId"] = "thread-goal-copy"
            app_server.responses["thread/fork"] = {
                "thread": {"id": "thread-goal-copy", "cwd": str(workspace)}
            }
            continuation_prompts: list[str] = []

            async def send_goal_continuation(text: str, _prompt_message_id: int | None = None, **_kwargs: Any) -> str:
                continuation_prompts.append(text)
                return "started"

            session.send_message = send_goal_continuation  # type: ignore[method-assign]

            forked_id = await session.fork_thread()

            self.assertEqual(forked_id, "thread-goal-copy")
            fork_request = next(
                params for method, params in app_server.requests if method == "thread/fork"
            )
            self.assertTrue(fork_request["deferGoalContinuation"])
            self.assertEqual(session.state.thread_id, "thread-goal-copy")
            self.assertTrue(session._goal_is_active())
            self.assertTrue(
                any(
                    method == "thread/goal/set"
                    and params.get("status") == "active"
                    for method, params in app_server.requests
                )
            )
            self.assertEqual(continuation_prompts, [GOAL_CONTINUATION_PROMPT])

    async def test_background_terminal_api_paginates_and_stops_in_current_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            async def no_status(
                _key: str, _details: object = None, _item_id: str | None = None
            ) -> None:
                return None

            async def no_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, _store = self._build_session(
                workspace,
                "thread-background",
                on_status=no_status,
                on_finished=no_finished,
            )
            app_server.response_sequences["thread/backgroundTerminals/list"] = [
                {
                    "data": [
                        {
                            "processId": "terminal-1",
                            "command": "python3 -m http.server",
                            "cwd": str(workspace),
                            "osPid": 2020,
                            "cpuPercent": 0.5,
                            "rssKb": 1024,
                        }
                    ],
                    "nextCursor": "cursor-1",
                },
                {
                    "data": [
                        {
                            "processId": "terminal-2",
                            "command": "npm run dev",
                            "cwd": str(workspace),
                            "osPid": None,
                        }
                    ],
                    "nextCursor": None,
                },
            ]

            terminals = await session.list_background_terminals()

            self.assertEqual(
                [terminal.process_id for terminal in terminals],
                ["terminal-1", "terminal-2"],
            )
            self.assertEqual(terminals[0].os_pid, 2020)
            self.assertEqual(terminals[0].rss_kb, 1024)
            list_requests = [
                params
                for method, params in app_server.requests
                if method == "thread/backgroundTerminals/list"
            ]
            self.assertEqual(list_requests[0]["threadId"], "thread-background")
            self.assertEqual(list_requests[1]["cursor"], "cursor-1")
            self.assertNotIn(
                "thread/unsubscribe",
                [method for method, _params in app_server.requests],
            )

            app_server.responses["thread/backgroundTerminals/list"] = {
                "data": [{"processId": "terminal-1"}],
                "nextCursor": None,
            }
            await session._unsubscribe_thread()
            self.assertNotIn(
                "thread/unsubscribe",
                [method for method, _params in app_server.requests],
            )

            app_server.responses["thread/backgroundTerminals/terminate"] = {
                "terminated": True
            }
            self.assertTrue(
                await session.terminate_background_terminal("terminal-1")
            )
            self.assertIn(
                (
                    "thread/backgroundTerminals/terminate",
                    {"threadId": "thread-background", "processId": "terminal-1"},
                ),
                app_server.requests,
            )

    async def test_empty_background_terminal_list_releases_idle_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            async def no_status(
                _key: str, _details: object = None, _item_id: str | None = None
            ) -> None:
                return None

            async def no_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, _store = self._build_session(
                workspace,
                "thread-no-background",
                on_status=no_status,
                on_finished=no_finished,
            )
            app_server.responses["thread/backgroundTerminals/list"] = {
                "data": [],
                "nextCursor": None,
            }

            self.assertEqual(await session.list_background_terminals(), [])
            self.assertIn(
                ("thread/unsubscribe", {"threadId": "thread-no-background"}),
                app_server.requests,
            )

    async def test_unsubscribe_falls_back_when_background_terminal_api_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            async def no_status(
                _key: str, _details: object = None, _item_id: str | None = None
            ) -> None:
                return None

            async def no_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, _store = self._build_session(
                workspace,
                "thread-old-codex",
                on_status=no_status,
                on_finished=no_finished,
            )
            await session._ensure_thread(create_if_missing=False)
            app_server.response_errors["thread/backgroundTerminals/list"] = RpcError(
                -32601,
                "Method not found",
            )

            await session._unsubscribe_thread()

            self.assertFalse(session._background_terminal_api_supported)
            self.assertIn(
                ("thread/unsubscribe", {"threadId": "thread-old-codex"}),
                app_server.requests,
            )

    async def test_app_server_activity_events_forward_file_and_web_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            statuses: list[tuple[str, object | None]] = []

            async def on_status(key: str, details: object = None, _item_id: str | None = None) -> None:
                statuses.append((key, details))

            async def on_finished(_result: dict[str, Any]) -> None:
                return None

            session, _app_server, _store = self._build_session(
                workspace,
                "thread-activity",
                on_status=on_status,
                on_finished=on_finished,
            )
            web_item = {"id": "web-1", "type": "webSearch", "query": "Codex"}
            file_item = {
                "id": "file-1",
                "type": "fileChange",
                "status": "completed",
                "changes": [{"path": "src/main.py", "kind": {"type": "update"}, "diff": "+line"}],
            }

            session._on_event({"method": "item/started", "params": {"item": web_item}})
            session._on_event({"method": "item/completed", "params": {"item": web_item}})
            session._on_event({"method": "item/completed", "params": {"item": file_item}})
            await asyncio.sleep(0)

            self.assertEqual(
                [key for key, _details in statuses],
                ["detailed_log.values.web_search_started", "detailed_log.values.web_search_completed", "status.file_change"],
            )
            self.assertEqual(statuses[0][1], web_item)
            self.assertEqual(statuses[1][1], web_item)
            self.assertEqual(statuses[2][1], file_item)

    async def test_app_server_review_image_and_subagent_events_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            statuses: list[tuple[str, object | None, str | None]] = []
            finished: list[dict[str, Any]] = []

            async def on_status(
                key: str, details: object = None, item_id: str | None = None
            ) -> None:
                statuses.append((key, details, item_id))

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            session, _app_server, _store = self._build_session(
                workspace,
                "thread-activity-events",
                on_status=on_status,
                on_finished=on_finished,
            )
            review_started = {"id": "review-turn", "type": "enteredReviewMode", "review": "changes"}
            image_started = {"id": "image-1", "type": "imageGeneration", "status": "inProgress"}
            agent_started = {
                "id": "agent-activity-1",
                "type": "subAgentActivity",
                "kind": "started",
                "agentPath": "root/researcher",
                "agentThreadId": "agent-thread-1",
            }
            collab_running = {
                "id": "agent-call-1",
                "type": "collabAgentToolCall",
                "tool": "spawnAgent",
                "status": "completed",
                "receiverThreadIds": ["agent-thread-1"],
                "agentsStates": {"agent-thread-1": {"status": "running"}},
            }

            for item in (review_started, image_started, agent_started, collab_running):
                session._on_event({"method": "item/started", "params": {"item": item}})
            session._on_event({"method": "item/completed", "params": {"item": review_started}})
            session._on_event({"method": "item/completed", "params": {"item": agent_started}})
            session._on_event({"method": "item/started", "params": {"item": collab_running}})
            session._on_event(
                {
                    "method": "item/started",
                    "params": {
                        "item": {
                            "id": "agent-interaction",
                            "type": "subAgentActivity",
                            "kind": "interacted",
                            "agentPath": "root",
                            "agentThreadId": "agent-thread-interaction",
                        }
                    },
                }
            )

            review_finished = {
                "id": "review-turn",
                "type": "exitedReviewMode",
                "review": "Found one issue in src/main.py.",
            }
            image_finished = {
                "id": "image-1",
                "type": "imageGeneration",
                "status": "completed",
                "savedPath": str(workspace / "assets/generated.png"),
                "revisedPrompt": "Do not forward this internal prompt.",
                "failure": None,
            }
            image_failed = {
                "id": "image-2",
                "type": "imageGeneration",
                "status": "failed",
                "failure": {"type": "usageLimitExceeded", "limitId": "daily"},
                "revisedPrompt": "Do not forward this internal prompt either.",
            }
            agent_finished = {**agent_started, "id": "agent-activity-2", "kind": "completed"}
            collab_finished = {
                **collab_running,
                "status": "completed",
                "agentsStates": {"agent-thread-1": {"status": "completed"}},
            }
            for item in (
                review_finished,
                image_finished,
                image_failed,
                agent_finished,
                collab_finished,
            ):
                session._on_event({"method": "item/completed", "params": {"item": item}})
            session._on_event({"method": "item/completed", "params": {"item": review_finished}})
            session._on_event(
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": "review-turn", "status": "completed"}},
                }
            )
            await asyncio.sleep(0)

            self.assertEqual(
                [key for key, _details, _item_id in statuses],
                [
                    "detailed_log.values.review_started",
                    "detailed_log.values.image_generation_started",
                    "detailed_log.values.subagent_started",
                    "detailed_log.values.review_completed",
                    "detailed_log.values.image_generation_completed",
                    "detailed_log.values.image_generation_failed",
                    "detailed_log.values.subagent_completed",
                ],
            )
            self.assertEqual(statuses[2][1], {"agentPath": "root/researcher"})
            self.assertEqual(
                statuses[4][1],
                {"savedPath": str(workspace / "assets/generated.png")},
            )
            self.assertEqual(statuses[5][1], {"failureType": "usageLimitExceeded"})
            self.assertEqual(finished[0]["text"], "Found one issue in src/main.py.")

    async def test_review_child_turn_completes_against_its_root_turn_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            async def no_status(
                _key: str, _details: object = None, _item_id: str | None = None
            ) -> None:
                return None

            finished: list[dict[str, Any]] = []

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            session, _app_server, _store = self._build_session(
                Path(temp_dir),
                "thread-review-root-correlation",
                on_status=no_status,
                on_finished=on_finished,
            )
            session._on_event(
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": "thread-review-root-correlation",
                        "turn": {"id": "review-child-turn", "rootTurnId": "review-root-turn"},
                    },
                }
            )
            self.assertEqual(session.active_turn_id, "review-root-turn")
            session._on_event(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "id": "review-result",
                            "type": "exitedReviewMode",
                            "review": "Review findings from the nested review turn.",
                        }
                    },
                }
            )
            session._on_event(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "id": "empty-review-final",
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": "",
                        }
                    },
                }
            )
            session._on_event(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-review-root-correlation",
                        "turn": {"id": "review-root-turn", "status": "completed"},
                    },
                }
            )
            await session._wait_for_turn_completion()

            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0]["status"], "completed")
            self.assertEqual(finished[0]["text"], "Review findings from the nested review turn.")

    async def test_commentary_messages_are_delivered_once_before_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            delivered: list[tuple[str, str]] = []
            finished: list[dict[str, Any]] = []
            order: list[str] = []

            async def no_status(
                _key: str, _details: object = None, _item_id: str | None = None
            ) -> None:
                return None

            async def on_assistant_message(item_id: str, text: str) -> None:
                delivered.append((item_id, text))
                order.append("commentary")
                await asyncio.sleep(0)

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)
                order.append("final")

            session, _app_server, _store = self._build_session(
                workspace,
                "thread-commentary",
                on_status=no_status,
                on_finished=on_finished,
                on_assistant_message=on_assistant_message,
            )
            session._on_event(
                {"method": "turn/started", "params": {"turn": {"id": "turn-commentary"}}}
            )
            session._on_event(
                {
                    "method": "item/started",
                    "params": {
                        "item": {
                            "id": "commentary-1",
                            "type": "agentMessage",
                            "phase": "commentary",
                        }
                    },
                }
            )
            session._on_event(
                {
                    "method": "item/agentMessage/delta",
                "params": {"itemId": "commentary-1", "delta": "Checking "},
                }
            )
            session._on_event(
                {
                    "method": "item/agentMessage/delta",
                "params": {"itemId": "commentary-1", "delta": "logs."},
                }
            )
            commentary_item = {
                "id": "commentary-1",
                "type": "agentMessage",
                "phase": "commentary",
                "text": "Checking logs.",
            }
            session._on_event(
                {"method": "item/completed", "params": {"item": commentary_item}}
            )
            session._on_event(
                {"method": "item/completed", "params": {"item": commentary_item}}
            )
            session._on_event(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "id": "analysis-1",
                            "type": "agentMessage",
                            "phase": "analysis",
                            "text": "internal content",
                        }
                    },
                }
            )
            session._on_event(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "id": "answer-1",
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": "Done.",
                        }
                    },
                }
            )
            session._on_event(
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": "turn-commentary", "status": "completed"}},
                }
            )
            assert session._turn_completion_task is not None
            await session._turn_completion_task

            self.assertEqual(delivered, [("commentary-1", "Checking logs.")])
            self.assertEqual(finished[0]["text"], "Done.")
            self.assertEqual(order, ["commentary", "final"])

    async def test_interrupted_turn_does_not_repeat_delivered_commentary_as_final(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            delivered: list[str] = []
            finished: list[dict[str, Any]] = []

            async def no_status(
                _key: str, _details: object = None, _item_id: str | None = None
            ) -> None:
                return None

            async def on_assistant_message(_item_id: str, text: str) -> None:
                delivered.append(text)

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            session, _app_server, _store = self._build_session(
                Path(temp_dir),
                "thread-interrupted-commentary",
                on_status=no_status,
                on_finished=on_finished,
                on_assistant_message=on_assistant_message,
            )
            session._on_event(
                {
                    "method": "turn/started",
                    "params": {"turn": {"id": "turn-interrupted-commentary"}},
                }
            )
            session._on_event(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "id": "commentary-interrupted",
                            "type": "agentMessage",
                            "phase": "commentary",
                            "text": "Checking.",
                        }
                    },
                }
            )
            session._on_event(
                {
                    "method": "turn/completed",
                    "params": {
                        "turn": {
                            "id": "turn-interrupted-commentary",
                            "status": "interrupted",
                        }
                    },
                }
            )
            assert session._turn_completion_task is not None
            await session._turn_completion_task

            self.assertEqual(delivered, ["Checking."])
            self.assertEqual(finished[0]["text"], "")

    async def test_delete_clears_all_references_and_preserves_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            store_root = RuntimeStateStore(workspace / "state.sqlite3")
            store_root.for_user(202).save(RuntimeState("thread-old", cwd=str(workspace)))
            statuses: list[str] = []

            async def on_status(key: str, _command: object = None, _item_id: str | None = None) -> None:
                statuses.append(key)

            async def on_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, store = self._build_session(
                workspace,
                "thread-old",
                on_status=on_status,
                on_finished=on_finished,
            )
            app_server.responses["thread/delete"] = {}

            await session.delete_thread("thread-old")

            self.assertIn(("thread/delete", {"threadId": "thread-old"}), app_server.requests)
            self.assertIsNone(store.load().thread_id)
            self.assertEqual(store.load().cwd, str(workspace))
            self.assertIsNone(store_root.for_user(202).load().thread_id)
            self.assertEqual(store.load_thread("thread-old"), ThreadRuntimeState())

    async def test_fork_switches_only_current_user_and_copies_thread_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            async def on_status(_key: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def on_finished(_result: dict[str, Any]) -> None:
                return None

            session, app_server, store = self._build_session(
                workspace,
                "thread-source",
                on_status=on_status,
                on_finished=on_finished,
            )
            other_user = RuntimeStateStore(workspace / "state.sqlite3").for_user(202)
            other_user.save(RuntimeState("thread-source", cwd=str(workspace)))
            store.save_thread(
                "thread-source",
                ThreadRuntimeState(
                    permission_mode="read_only",
                    managed_by_telecodex=True,
                    token_usage={"context_tokens": 900, "model_context_window": 1000},
                ),
            )
            app_server.responses["thread/fork"] = {
                "thread": {"id": "thread-forked", "cwd": str(workspace)}
            }

            forked_id = await session.fork_thread()

            self.assertEqual(forked_id, "thread-forked")
            self.assertIn(
                ("thread/fork", {"threadId": "thread-source", "excludeTurns": True}),
                app_server.requests,
            )
            self.assertEqual(store.load().thread_id, "thread-forked")
            self.assertEqual(other_user.load().thread_id, "thread-source")
            self.assertEqual(
                store.load_thread("thread-forked"),
                ThreadRuntimeState(
                    permission_mode="read_only",
                    managed_by_telecodex=True,
                    token_usage={"context_tokens": 900, "model_context_window": 1000},
                ),
            )

    async def test_manual_compaction_reports_events_without_empty_task_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            statuses: list[str] = []
            finished: list[dict[str, Any]] = []

            async def on_status(key: str, _command: object = None, _item_id: str | None = None) -> None:
                statuses.append(key)

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            session, app_server, _store = self._build_session(
                workspace,
                "thread-compact",
                on_status=on_status,
                on_finished=on_finished,
            )
            app_server.responses["thread/read"] = {
                "thread": {"id": "thread-compact", "status": {"type": "idle"}}
            }

            await session.compact_thread()
            self.assertTrue(session.is_busy)
            for _ in range(20):
                if any(method == "thread/compact/start" for method, _params in app_server.requests):
                    break
                await asyncio.sleep(0)
            self.assertIn(
                ("thread/compact/start", {"threadId": "thread-compact"}),
                app_server.requests,
            )
            session._on_event({"method": "turn/started", "params": {"turn": {"id": "compact-turn"}}})
            session._on_event(
                {
                    "method": "item/started",
                    "params": {"item": {"id": "compact-item", "type": "contextCompaction"}},
                }
            )
            session._on_event(
                {
                    "method": "item/completed",
                    "params": {"item": {"id": "compact-item", "type": "contextCompaction"}},
                }
            )
            session._on_event(
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": "compact-turn", "status": "completed"}},
                }
            )
            await session.wait_for_manual_compaction_barrier()
            await asyncio.sleep(0)

            self.assertFalse(session.is_busy)
            self.assertEqual(statuses, ["status.label.compaction_started", "status.label.manual_compaction_completed"])
            self.assertEqual(finished, [])
            self.assertIn(
                ("thread/unsubscribe", {"threadId": "thread-compact"}),
                app_server.requests,
            )

    async def test_manual_compaction_waits_for_an_active_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session, app_server, _store = self._build_session(
                Path(temp_dir),
                "thread-compact-busy",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            app_server.responses["thread/read"] = {
                "thread": {"id": "thread-compact-busy", "status": {"type": "idle"}}
            }
            session.active_turn_id = "turn-running"
            session._turn_completed_event = asyncio.Event()

            await session.compact_thread()
            await asyncio.sleep(0)

            self.assertTrue(session.compaction_barrier_pending)
            self.assertFalse(
                any(method == "thread/compact/start" for method, _params in app_server.requests)
            )

            session._on_event(
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": "turn-running", "status": "completed"}},
                }
            )
            for _ in range(40):
                if any(method == "thread/compact/start" for method, _params in app_server.requests):
                    break
                await asyncio.sleep(0)
            self.assertIn(
                ("thread/compact/start", {"threadId": "thread-compact-busy"}),
                app_server.requests,
            )
            session._on_event(
                {"method": "turn/started", "params": {"turn": {"id": "compact-turn"}}}
            )
            session._on_event(
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": "compact-turn", "status": "completed"}},
                }
            )
            await session.wait_for_manual_compaction_barrier()
            self.assertFalse(session.is_busy)

    async def test_manual_compaction_waits_for_background_terminals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            statuses: list[str] = []

            async def on_status(key: str, *_args: Any) -> None:
                statuses.append(key)

            session, app_server, _store = self._build_session(
                Path(temp_dir),
                "thread-compact-background",
                on_status=on_status,
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            app_server.responses["thread/read"] = {
                "thread": {"id": "thread-compact-background", "status": {"type": "idle"}}
            }
            app_server.responses["thread/backgroundTerminals/list"] = {
                "data": [{"processId": "server-1", "command": "python -m http.server"}],
                "nextCursor": None,
            }

            with patch("telecodex.codex.session.MANUAL_COMPACTION_IDLE_POLL_SECONDS", 0.001):
                await session.compact_thread()
                for _ in range(40):
                    if any(
                        method == "thread/backgroundTerminals/list"
                        for method, _params in app_server.requests
                    ):
                        break
                    await asyncio.sleep(0)
                for _ in range(10):
                    if "status.compaction_queued" in statuses:
                        break
                    await asyncio.sleep(0)
                self.assertEqual(statuses.count("status.compaction_queued"), 1)
                self.assertFalse(
                    any(method == "thread/compact/start" for method, _params in app_server.requests)
                )

                app_server.responses["thread/backgroundTerminals/list"] = {
                    "data": [],
                    "nextCursor": None,
                }
                for _ in range(100):
                    if any(method == "thread/compact/start" for method, _params in app_server.requests):
                        break
                    await asyncio.sleep(0.001)
                self.assertIn(
                    (
                        "thread/compact/start",
                        {"threadId": "thread-compact-background"},
                    ),
                    app_server.requests,
                )
                session._on_event(
                    {"method": "turn/started", "params": {"turn": {"id": "compact-turn"}}}
                )
                session._on_event(
                    {
                        "method": "turn/completed",
                        "params": {"turn": {"id": "compact-turn", "status": "completed"}},
                    }
                )
                await session.wait_for_manual_compaction_barrier()

            self.assertFalse(session.is_busy)

    async def test_stale_manual_compaction_is_released_when_thread_is_idle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            statuses: list[str] = []

            async def on_status(key: str, *_args: Any) -> None:
                statuses.append(key)

            session, app_server, _store = self._build_session(
                Path(temp_dir),
                "thread-compact-stale",
                on_status=on_status,
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            app_server.is_running = True
            app_server.responses["thread/read"] = {
                "thread": {"id": "thread-compact-stale", "status": {"type": "idle"}}
            }
            future = asyncio.get_running_loop().create_future()
            session._manual_compaction_future = future
            session._manual_compaction_started_at = 0.0

            await session._wait_for_turn_completion()
            await asyncio.sleep(0)

            self.assertFalse(session.is_busy)
            self.assertTrue(future.done())
            self.assertIsNone(session._manual_compaction_future)
            self.assertIn(("thread/read", {"threadId": "thread-compact-stale"}), app_server.requests)
            self.assertEqual(statuses, ["status.label.compaction_failed"])

    async def test_manual_compaction_stays_busy_when_thread_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session, app_server, _store = self._build_session(
                Path(temp_dir),
                "thread-compact-active",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            app_server.is_running = True
            app_server.responses["thread/read"] = {
                "thread": {"id": "thread-compact-active", "status": {"type": "active"}}
            }
            future = asyncio.get_running_loop().create_future()
            session._manual_compaction_future = future
            session._manual_compaction_started_at = 0.0
            session._manual_compaction_turn_id = "compact-turn"
            session.active_turn_id = "compact-turn"

            with self.assertRaises(CodexChatBusyError):
                await session._wait_for_turn_completion()

            self.assertTrue(session.is_busy)
            self.assertIs(session._manual_compaction_future, future)
            future.cancel()
            session._manual_compaction_future = None

    async def test_automatic_compaction_events_do_not_interrupt_turn_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            statuses: list[str] = []
            finished: list[dict[str, Any]] = []

            async def on_status(key: str, _command: object = None, _item_id: str | None = None) -> None:
                statuses.append(key)

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            session, _app_server, _store = self._build_session(
                workspace,
                None,
                on_status=on_status,
                on_finished=on_finished,
            )
            session._on_event({"method": "turn/started", "params": {"turn": {"id": "task-turn"}}})
            session._on_event(
                {
                    "method": "item/started",
                    "params": {"item": {"id": "auto-compact", "type": "contextCompaction"}},
                }
            )
            session._on_event(
                {
                    "method": "item/completed",
                    "params": {"item": {"id": "auto-compact", "type": "contextCompaction"}},
                }
            )
            session._on_event(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "id": "answer",
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": "done",
                        }
                    },
                }
            )
            session._on_event(
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": "task-turn", "status": "completed"}},
                }
            )
            task = session._turn_completion_task
            self.assertIsNotNone(task)
            await task
            await asyncio.sleep(0)

            self.assertEqual(statuses, ["status.label.compaction_started", "status.label.compaction_completed"])
            self.assertEqual(finished[0]["status"], "completed")
            self.assertEqual(finished[0]["text"], "done")

    async def test_full_access_selection_is_persisted_and_sent_on_the_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, store = self._build_session(
                workspace,
                "thread-full-access",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            store.save_thread("thread-full-access", ThreadRuntimeState())
            session.select_permission_mode("full_access")
            app_server.responses["turn/start"] = {
                "turn": {"id": "turn-full-access", "status": "inProgress"}
            }

            await session.send_message("run the requested command")

            self.assertEqual(
                store.load_thread("thread-full-access").permission_mode,
                "full_access",
            )
            turn_params = next(
                params for method, params in app_server.requests if method == "turn/start"
            )
            self.assertEqual(turn_params["approvalPolicy"], "never")
            self.assertEqual(turn_params["sandboxPolicy"], {"type": "dangerFullAccess"})
            self.assertNotIn("permissions", turn_params)
            session.active_turn_id = None
            session.state.pending_turn = None
            await session.close()

    async def test_turn_start_persists_telegram_correlation_and_completion_outbox(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            finished: list[dict[str, Any]] = []

            async def no_status(_key: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            session, app_server, store = self._build_session(
                workspace,
                "thread-active",
                on_status=no_status,
                on_finished=on_finished,
            )
            app_server.responses["turn/start"] = {"turn": {"id": "turn-active"}}

            self.assertEqual(await session.send_message("do work", 77), "started")
            self.assertEqual(
                store.load().pending_turn,
                PendingTurnState(
                    thread_id="thread-active",
                    turn_id="turn-active",
                    prompt_message_id=77,
                    started_at=store.load().pending_turn.started_at,  # type: ignore[union-attr]
                    phase="running",
                ),
            )

            session._on_event(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "id": "answer",
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": "finished",
                        }
                    },
                }
            )
            session._on_event(
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": "turn-active", "status": "completed"}},
                }
            )
            await session._wait_for_turn_completion()

            recovered = store.load().pending_turn
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.phase, "delivery_pending")  # type: ignore[union-attr]
            self.assertEqual(recovered.result_status, "completed")  # type: ignore[union-attr]
            self.assertEqual(finished[0]["text"], "finished")

    async def test_immediate_completion_before_turn_start_response_does_not_leave_busy_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            finished: list[dict[str, Any]] = []

            async def no_status(_key: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            session, app_server, store = self._build_session(
                workspace,
                "thread-fast",
                on_status=no_status,
                on_finished=on_finished,
            )
            original_request = app_server.request

            async def complete_before_response(
                method: str,
                params: dict[str, Any] | None = None,
                timeout_seconds: int = 90,
            ) -> dict[str, Any]:
                if method == "turn/start":
                    session._on_event(
                        {
                            "method": "item/completed",
                            "params": {
                                "item": {
                                    "id": "fast-answer",
                                    "type": "agentMessage",
                                    "phase": "final_answer",
                                    "text": "done immediately",
                                }
                            },
                        }
                    )
                    session._on_event(
                        {
                            "method": "turn/completed",
                            "params": {"turn": {"id": "turn-fast", "status": "completed"}},
                        }
                    )
                    return {"turn": {"id": "turn-fast"}}
                return await original_request(method, params, timeout_seconds)

            app_server.request = complete_before_response  # type: ignore[method-assign]
            await session.send_message("quick task", 99)
            await session._wait_for_turn_completion()

            self.assertFalse(session.is_busy)
            self.assertEqual(finished[0]["text"], "done immediately")
            self.assertEqual(store.load().pending_turn.phase, "delivery_pending")  # type: ignore[union-attr]

    async def test_startup_recovers_completed_turn_from_persisted_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            store_root = RuntimeStateStore(workspace / "state.sqlite3")
            store = store_root.for_user(101)
            store.save(
                RuntimeState(
                    thread_id="thread-recover",
                    cwd=str(workspace),
                    pending_turn=PendingTurnState(
                        thread_id="thread-recover",
                        turn_id="turn-recover",
                        prompt_message_id=88,
                        started_at=123,
                    ),
                )
            )
            app_server = FakeAppServer(missing_thread=False)
            app_server.responses["thread/turns/list"] = {
                "data": [{"id": "turn-recover", "status": "completed", "items": []}],
                "nextCursor": None,
            }
            app_server.responses["thread/items/list"] = {
                "data": [
                    {
                        "turnId": "turn-recover",
                        "item": {
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": "restored response",
                        },
                    }
                ],
                "nextCursor": None,
            }
            finished: list[dict[str, Any]] = []

            async def no_status(_key: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
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
                session = CodexSession(config, store, no_status, no_status, on_finished, no_status)

            self.assertFalse(await session.start())

            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0]["status"], "completed")
            self.assertEqual(finished[0]["text"], "restored response")
            self.assertTrue(finished[0]["recovered"])
            self.assertEqual(store.load().pending_turn.phase, "delivery_pending")  # type: ignore[union-attr]

    async def test_recovered_result_is_not_delivered_twice_when_recovery_paths_race(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            finished: list[dict[str, Any]] = []
            callback_started = asyncio.Event()
            allow_callback_to_finish = asyncio.Event()

            async def no_status(_key: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)
                session.clear_pending_turn()
                callback_started.set()
                await allow_callback_to_finish.wait()

            session, app_server, store = self._build_session(
                workspace,
                "thread-recover-race",
                on_status=no_status,
                on_finished=on_finished,
            )
            app_server.responses["thread/items/list"] = {"data": [], "nextCursor": None}
            session.state.pending_turn = PendingTurnState(
                thread_id="thread-recover-race",
                turn_id="turn-recover-race",
                prompt_message_id=90,
                started_at=123,
                phase="delivery_pending",
                result_status="completed",
            )
            store.save(session.state)
            pending = session.pending_turn
            self.assertIsNotNone(pending)

            first = asyncio.create_task(session._deliver_pending_turn(pending))  # type: ignore[arg-type]
            await callback_started.wait()
            second = asyncio.create_task(session._deliver_pending_turn(pending))  # type: ignore[arg-type]
            await asyncio.sleep(0)
            allow_callback_to_finish.set()
            await asyncio.gather(first, second)

            self.assertEqual(len(finished), 1)
            self.assertIsNone(session.pending_turn)

    async def test_startup_reports_unconfirmed_start_without_replaying_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            store = RuntimeStateStore(workspace / "state.sqlite3").for_user(101)
            store.save(
                RuntimeState(
                    thread_id="thread-starting",
                    cwd=str(workspace),
                    pending_turn=PendingTurnState(
                        thread_id="thread-starting",
                        turn_id=None,
                        prompt_message_id=89,
                        started_at=123,
                        phase="starting",
                    ),
                )
            )
            app_server = FakeAppServer(missing_thread=False)
            finished: list[dict[str, Any]] = []

            async def no_status(_key: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
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
                session = CodexSession(config, store, no_status, no_status, on_finished, no_status)

            await session.start()

            self.assertFalse(any(method == "turn/start" for method, _params in app_server.requests))
            self.assertEqual(finished[0]["status"], "unknown")
            self.assertTrue(finished[0]["recovered"])

    async def test_startup_reconnects_confirmed_active_turn_and_recovers_its_final_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            store = RuntimeStateStore(workspace / "state.sqlite3").for_user(101)
            store.save(
                RuntimeState(
                    thread_id="thread-running",
                    cwd=str(workspace),
                    pending_turn=PendingTurnState(
                        thread_id="thread-running",
                        turn_id="turn-running",
                        prompt_message_id=90,
                        started_at=123,
                        phase="running",
                    ),
                )
            )
            app_server = FakeAppServer(missing_thread=False)
            app_server.responses["thread/turns/list"] = {
                "data": [{"id": "turn-running", "status": "inProgress", "items": []}],
                "nextCursor": None,
            }
            app_server.responses["thread/read"] = {
                "thread": {"id": "thread-running", "status": {"type": "active", "activeFlags": []}}
            }
            app_server.responses["thread/items/list"] = {
                "data": [
                    {
                        "turnId": "turn-running",
                        "item": {
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": "finished after reconnect",
                        },
                    }
                ],
                "nextCursor": None,
            }
            finished: list[dict[str, Any]] = []

            async def no_status(_key: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
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
                session = CodexSession(config, store, no_status, no_status, on_finished, no_status)

            self.assertTrue(await session.start())
            self.assertTrue(session.is_busy)
            self.assertIn(("thread/resume", {"threadId": "thread-running", "excludeTurns": True}), app_server.requests)

            session._on_event(
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": "turn-running", "status": "completed"}},
                }
            )
            await session._wait_for_turn_completion()

            self.assertEqual(finished[0]["text"], "finished after reconnect")
            self.assertEqual(finished[0]["status"], "completed")

    async def test_startup_interrupts_turn_waiting_for_old_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            store = RuntimeStateStore(workspace / "state.sqlite3").for_user(101)
            store.save(
                RuntimeState(
                    thread_id="thread-approval",
                    cwd=str(workspace),
                    pending_turn=PendingTurnState(
                        thread_id="thread-approval",
                        turn_id="turn-approval",
                        prompt_message_id=90,
                        started_at=123,
                        phase="running",
                        prompt_message_ids=[91],
                    ),
                )
            )
            app_server = FakeAppServer(missing_thread=False)
            app_server.responses["thread/turns/list"] = {
                "data": [{"id": "turn-approval", "status": "inProgress", "items": []}],
                "nextCursor": None,
            }
            app_server.responses["thread/read"] = {
                "thread": {
                    "id": "thread-approval",
                    "status": {"type": "active", "activeFlags": ["waitingOnApproval"]},
                }
            }
            app_server.responses["thread/items/list"] = {"data": [], "nextCursor": None}
            finished: list[dict[str, Any]] = []

            async def no_status(_key: str, _command: object = None, _item_id: str | None = None) -> None:
                return None

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)

            with patch("telecodex.codex.session.CodexAppServer", return_value=app_server):
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
                session = CodexSession(config, store, no_status, no_status, on_finished, no_status)

            self.assertFalse(await session.start())

            self.assertIn(
                (
                    "turn/interrupt",
                    {"threadId": "thread-approval", "turnId": "turn-approval"},
                ),
                app_server.requests,
            )
            self.assertEqual(finished[0]["status"], "interrupted")
            self.assertEqual(finished[0]["error_key"], "recovery.label.prompt_expired")
            self.assertFalse(any(method == "turn/start" for method, _params in app_server.requests))


    async def test_recover_idle_thread_restarts_only_app_server_and_preserves_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, store = self._build_session(
                workspace,
                "thread-recover-idle",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            session.state.selected_model = "gpt-6-luna"
            session.state.selected_effort = "high"
            session.state.selected_service_tier = "default"
            session.state.output_mode = "answers"
            store.save(session.state)
            store.save_thread(
                "thread-recover-idle",
                ThreadRuntimeState(
                    permission_mode="ask",
                    memory_mode="enabled",
                    managed_by_telecodex=True,
                    token_usage={"total": 123},
                ),
            )
            app_server.responses["thread/read"] = {
                "thread": {
                    "id": "thread-recover-idle",
                    "cwd": str(workspace),
                    "status": {"type": "idle"},
                },
                "model": "gpt-6-luna",
                "reasoningEffort": "high",
                "serviceTier": "default",
            }

            result = await session.recover_current_thread("thread-recover-idle")

            self.assertIsNone(result)
            self.assertEqual(session.state.thread_id, "thread-recover-idle")
            self.assertEqual(session.state.cwd, str(workspace))
            self.assertEqual(session.state.selected_model, "gpt-6-luna")
            self.assertEqual(session.state.selected_effort, "high")
            self.assertEqual(session.state.output_mode, "answers")
            thread_state = store.load_thread("thread-recover-idle")
            self.assertEqual(thread_state.permission_mode, "ask")
            self.assertEqual(thread_state.memory_mode, "enabled")
            self.assertEqual(thread_state.token_usage, {"total": 123})
            self.assertEqual(app_server.start_calls, 2)
            self.assertEqual(app_server.stop_calls, 1)
            self.assertFalse(session.recovery_blocked)
            self.assertFalse(session.is_busy)

    async def test_recover_requires_confirmation_before_terminating_terminals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, _store = self._build_session(
                workspace,
                "thread-recover-terminal",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            terminal = {
                "processId": "terminal-recover",
                "command": "python worker.py",
                "cwd": str(workspace),
            }
            empty = {"data": [], "nextCursor": None}
            app_server.response_sequences["thread/backgroundTerminals/list"] = [
                {"data": [terminal], "nextCursor": None},
                {"data": [terminal], "nextCursor": None},
                {"data": [terminal], "nextCursor": None},
                empty,
                empty,
                empty,
            ]
            app_server.responses["thread/read"] = {
                "thread": {
                    "id": "thread-recover-terminal",
                    "cwd": str(workspace),
                    "status": {"type": "idle"},
                }
            }
            app_server.responses["thread/backgroundTerminals/terminate"] = {"terminated": True}

            snapshot = await session.inspect_current_thread_for_recovery()
            self.assertEqual([item.process_id for item in snapshot.terminals], ["terminal-recover"])

            needs_confirmation = await session.recover_current_thread("thread-recover-terminal")
            self.assertIsNotNone(needs_confirmation)
            self.assertEqual(app_server.stop_calls, 0)

            result = await session.recover_current_thread(
                "thread-recover-terminal",
                confirmed_terminal_ids={"terminal-recover"},
            )

            self.assertIsNone(result)
            self.assertIn(
                (
                    "thread/backgroundTerminals/terminate",
                    {"threadId": "thread-recover-terminal", "processId": "terminal-recover"},
                ),
                app_server.requests,
            )
            self.assertEqual(app_server.stop_calls, 1)

    async def test_recover_clears_a_stale_compaction_only_after_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, _store = self._build_session(
                workspace,
                "thread-recover-stale-compact",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            app_server.responses["thread/read"] = {
                "thread": {
                    "id": "thread-recover-stale-compact",
                    "cwd": str(workspace),
                    "status": {"type": "idle"},
                }
            }
            compaction = asyncio.get_running_loop().create_future()
            session._manual_compaction_future = compaction

            snapshot = await session.inspect_current_thread_for_recovery()
            self.assertTrue(snapshot.compaction_pending)
            self.assertTrue(snapshot.requires_confirmation)

            needs_confirmation = await session.recover_current_thread(
                "thread-recover-stale-compact"
            )
            self.assertIsNotNone(needs_confirmation)
            self.assertEqual(app_server.stop_calls, 0)

            result = await session.recover_current_thread(
                "thread-recover-stale-compact",
                confirm_interrupt=True,
            )

            self.assertIsNone(result)
            self.assertTrue(compaction.done())
            self.assertFalse(session.compaction_barrier_pending)
            self.assertEqual(session.state.thread_id, "thread-recover-stale-compact")

    async def test_recover_interrupts_active_turn_and_delivers_confirmed_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            finished: list[dict[str, Any]] = []
            session_ref: dict[str, CodexSession] = {}

            async def on_finished(result: dict[str, Any]) -> None:
                finished.append(result)
                session_ref["session"].clear_pending_turn()

            session, app_server, _store = self._build_session(
                workspace,
                "thread-recover-active",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=on_finished,
            )
            session_ref["session"] = session
            session.state.pending_turn = PendingTurnState(
                thread_id="thread-recover-active",
                turn_id="turn-recover-active",
                prompt_message_id=77,
                started_at=123,
                phase="running",
            )
            session.state_store.save(session.state)
            session.active_turn_id = "turn-recover-active"
            session._turn_completed_event = asyncio.Event()
            idle = {
                "thread": {
                    "id": "thread-recover-active",
                    "cwd": str(workspace),
                    "status": {"type": "idle"},
                }
            }
            app_server.responses["thread/read"] = idle
            app_server.responses["thread/items/list"] = {"data": [], "nextCursor": None}
            app_server.response_sequences["thread/turns/list"] = [
                {"data": [{"id": "turn-recover-active", "status": "inProgress"}]},
                {"data": [{"id": "turn-recover-active", "status": "interrupted"}]},
            ]

            result = await session.recover_current_thread(
                "thread-recover-active",
                confirm_interrupt=True,
            )

            self.assertIsNone(result)
            self.assertIn(
                (
                    "turn/interrupt",
                    {"threadId": "thread-recover-active", "turnId": "turn-recover-active"},
                ),
                app_server.requests,
            )
            self.assertEqual(finished[0]["status"], "interrupted")
            self.assertIsNone(session.state.pending_turn)
            self.assertIsNone(session.active_turn_id)
            self.assertEqual(session.state.thread_id, "thread-recover-active")

    async def test_recover_resume_failure_keeps_thread_selected_and_blocks_new_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            session, app_server, _store = self._build_session(
                workspace,
                "thread-recover-resume-failure",
                on_status=lambda *_args: asyncio.sleep(0),
                on_finished=lambda *_args: asyncio.sleep(0),
            )
            session._loaded_thread_id = "thread-recover-resume-failure"
            app_server.is_running = True
            app_server.active_writer_on_resume = True
            app_server.responses["thread/read"] = {
                "thread": {
                    "id": "thread-recover-resume-failure",
                    "cwd": str(workspace),
                    "status": {"type": "idle"},
                }
            }

            with self.assertRaises(ActiveThreadWriterError):
                await session.recover_current_thread("thread-recover-resume-failure")

            self.assertEqual(session.state.thread_id, "thread-recover-resume-failure")
            self.assertTrue(session.recovery_blocked)
            self.assertFalse(app_server.is_running)
            self.assertEqual(app_server.stop_calls, 2)
