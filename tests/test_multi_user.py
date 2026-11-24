from __future__ import annotations

import asyncio
import html
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from telecodex.config import AppConfig, ProxySettings, load_config
from telecodex.codex.command_prompts import GOAL_CONTINUATION_PROMPT
from telecodex.codex.rpc import RpcError
from telecodex.codex.session import (
    ActiveThreadWriterError,
    CodexChatBusyError,
    CodexMemorySettings,
    CodexBackgroundTerminal,
    CodexGoal,
    CodexModel,
    CodexRecoverySnapshot,
    CodexServiceTier,
    CodexSession,
    CodexSkill,
    CodexThread,
    FastTierUnavailableError,
    GoalNotFoundError,
    GoalUnavailableError,
    MemoryFeatureUnavailableError,
    SelectedSkillUnavailableError,
)
from telecodex.audio.transcriber import AudioTranscriptionError, LocalAudioTranscriber
from telecodex.documents.models import TelegramAttachment
from telecodex.group_triggers import parse_trigger_words, strip_group_trigger
from telecodex.i18n import TextCatalog
from telecodex.shell.runner import ShellResult
from telecodex.state import PendingTurnState, RuntimeState, RuntimeStateStore, ThreadRuntimeState
from telecodex.telegram.controller import TelegramController
from telecodex.telegram.updates import IncomingCallback, IncomingMessage, parse_update
from telecodex.telegram.user_controller import (
    PendingApproval,
    RunningCommand,
    TelegramUserController,
)
from telecodex.telegram.activity import (
    explored_action_summaries,
    file_change_groups,
    image_saved_path,
    web_search_detail,
)
from telecodex.telegram.api import TelegramAPI, TelegramApiError
from telecodex.telegram.formatting import TELEGRAM_TEXT_LIMIT, split_plain_text
from telecodex.telegram.pagination import navigation_row, page_count, two_column_rows

_TEXT_CATALOG = TextCatalog()


def _catalog_text(key: str, **values: Any) -> str:
    return _TEXT_CATALOG.render("ru", key, **values)


_HTML_FORMAT_TAG = re.compile(r"</?(?:strong|code|pre)>")


def _plain_html(text: str) -> str:
    return html.unescape(_HTML_FORMAT_TAG.sub("", text))


CODEX_TRIGGER_CYRILLIC = "\u043a\u043e\u0434\u0435\u043a\u0441"
KAWAII_TRIGGER_CYRILLIC = "\u043a\u0430\u0432\u0430\u0438"
BOT_TRIGGER_CYRILLIC = "\u0431\u043e\u0442"
HELPER_TRIGGER_CYRILLIC = "\u043f\u043e\u043c\u043e\u0449\u043d\u0438\u043a"


class FakeTelegram:
    def __init__(self) -> None:
        self.bot_user_id = 999
        self.messages: list[tuple[int, str]] = []
        self.message_entities: list[list[dict[str, Any]]] = []
        self.edits: list[tuple[int, int, str]] = []
        self.reply_markup_edits: list[tuple[int, int, Any]] = []
        self.callbacks: list[tuple[str, str]] = []
        self.actions: list[tuple[int, str]] = []
        self.reply_markups: list[Any] = []
        self.edit_markups: list[Any] = []
        self.replies: list[int | None] = []
        self.notification_flags: list[bool] = []
        self.rich_messages: list[tuple[int, str, bool]] = []
        self.documents: list[tuple[int, Path, str, int | None]] = []
        self.next_message_id = 1

    async def send_message(
        self,
        chat_id: int,
        text: str,
        _reply_markup: Any = None,
        reply_to_message_id: int | None = None,
        entities: list[dict[str, Any]] | None = None,
        disable_notification: bool = False,
    ) -> dict[str, int]:
        self.messages.append((chat_id, text))
        self.message_entities.append(entities or [])
        self.reply_markups.append(_reply_markup)
        self.replies.append(reply_to_message_id)
        self.notification_flags.append(disable_notification)
        message_id = self.next_message_id
        self.next_message_id += 1
        return {"message_id": message_id}

    async def send_rich_message(
        self,
        chat_id: int,
        markdown: str,
        *,
        reply_to_message_id: int | None = None,
        disable_notification: bool = False,
    ) -> dict[str, int]:
        self.rich_messages.append((chat_id, markdown, disable_notification))
        return await self.send_message(
            chat_id,
            markdown,
            reply_to_message_id=reply_to_message_id,
            disable_notification=disable_notification,
        )

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.actions.append((chat_id, action))

    async def answer_callback(self, callback_id: str, text: str | None = None) -> None:
        self.callbacks.append((callback_id, text or ""))

    async def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        _reply_markup: Any = None,
    ) -> None:
        self.edits.append((chat_id, message_id, text))
        self.edit_markups.append(_reply_markup)
        return None

    async def edit_message_reply_markup(
        self,
        chat_id: int,
        message_id: int,
        reply_markup: Any,
    ) -> None:
        self.reply_markup_edits.append((chat_id, message_id, reply_markup))

    async def download_file(self, _file_id: str, destination: Path, max_bytes: int) -> int:
        destination.write_text("test attachment", encoding="utf-8")
        return len("test attachment")

    async def send_document(
        self,
        chat_id: int,
        path: Path,
        filename: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> dict[str, int]:
        self.documents.append((chat_id, path, filename, reply_to_message_id))
        return {"message_id": self.next_message_id}


class FakeAudioTranscriber:
    status = "ready"

    def __init__(self, transcript: str = "recognized transcript") -> None:
        self.transcript = transcript
        self.calls: list[tuple[Path, int | None]] = []

    async def transcribe(self, audio_path: Path, *, duration_hint: int | None = None) -> str:
        self.calls.append((audio_path, duration_hint))
        return self.transcript


class FakeCodex:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.input_items: list[list[dict[str, Any]]] = []
        self.is_busy = False
        self.compaction_barrier_pending = False
        self.compaction_barrier_event = asyncio.Event()
        self.compaction_barrier_event.set()
        self.has_active_command_execution = False
        self.state = RuntimeState()
        self.deleted_thread: str | None = None
        self.renamed_chat_name: str | None = None
        self.forked = False
        self.compacted = False
        self.started = False
        self.closed = False
        self.models: list[CodexModel] = []
        self.working_directory = Path("/tmp")
        self.threads: list[CodexThread] = []
        self.select_calls: list[tuple[str, Path]] = []
        self.select_error: Exception | None = None
        self.fork_by_id_calls: list[tuple[str, Path]] = []
        self.fork_by_id_error: Exception | None = None
        self.background_terminals: list[CodexBackgroundTerminal] = []
        self.terminated_background_terminals: list[str] = []
        self.background_terminal_list_calls = 0
        self.collaboration_mode = "default"
        self.collaboration_mode_calls: list[tuple[str, bool]] = []
        self.fast_enabled = False
        self.fast_error: Exception | None = None
        self.skills: list[CodexSkill] = []
        self.skill_enabled_calls: list[tuple[str, bool]] = []
        self.memory_mode: str | None = None
        self.memory_settings = CodexMemorySettings(True, False, False, "config-v1")
        self.memories_reset = False
        self.goal: CodexGoal | None = None
        self.recovery_blocked = False
        self.recovery_held = False
        self.recovery_snapshot: CodexRecoverySnapshot | None = None
        self.recovery_inspections = 0
        self.recovery_calls: list[tuple[str, set[str], bool]] = []
        self.recovery_result: CodexRecoverySnapshot | None = None

    async def send_message(
        self,
        text: str,
        _prompt_message_id: int | None = None,
        *,
        additional_input_items: list[dict[str, Any]] | None = None,
    ) -> str:
        self.messages.append(text)
        self.input_items.append(list(additional_input_items or []))
        return "started"

    @property
    def pending_turn(self) -> PendingTurnState | None:
        return self.state.pending_turn

    @property
    def manual_compaction_waiting_for_idle(self) -> bool:
        return self.compaction_barrier_pending

    def add_pending_prompt(self, message_id: int) -> None:
        if self.state.pending_turn and message_id not in self.state.pending_turn.prompt_message_ids:
            self.state.pending_turn.prompt_message_ids.append(message_id)

    def remove_pending_prompt(self, message_id: int) -> None:
        if self.state.pending_turn and message_id in self.state.pending_turn.prompt_message_ids:
            self.state.pending_turn.prompt_message_ids.remove(message_id)

    def clear_pending_turn(self) -> None:
        self.state.pending_turn = None

    def select_output_mode(self, mode: str) -> None:
        self.state.quiet_progress = mode == "quiet_answers"
        self.state.output_mode = "answers" if mode == "quiet_answers" else mode

    async def status_text(self, *, include_account: bool = True) -> str:
        return "Codex: ready"

    async def new_thread(self, _cwd: Path | None = None, *, force: bool = False) -> None:
        return None

    async def rename_thread(self, name: str) -> None:
        self.renamed_chat_name = name

    async def interrupt(self) -> bool:
        return False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def delete_thread(self, thread_id: str) -> None:
        self.deleted_thread = thread_id
        self.state.thread_id = None

    async def fork_thread(self) -> str:
        self.forked = True
        self.state.thread_id = "thread-forked"
        return "thread-forked"

    async def compact_thread(self) -> None:
        self.compacted = True

    async def wait_for_manual_compaction_barrier(self) -> None:
        await self.compaction_barrier_event.wait()

    async def wait_for_current_turn_completion(self) -> None:
        return None

    async def cancel_pending_manual_compaction(self) -> bool:
        was_pending = self.compaction_barrier_pending
        self.compaction_barrier_pending = False
        self.compaction_barrier_event.set()
        return was_pending

    def hold_recovery(self) -> None:
        self.recovery_held = True

    def release_recovery_hold(self) -> None:
        if not self.recovery_blocked:
            self.recovery_held = False

    async def inspect_current_thread_for_recovery(self) -> CodexRecoverySnapshot:
        self.recovery_inspections += 1
        if self.recovery_snapshot is not None:
            return self.recovery_snapshot
        assert self.state.thread_id is not None
        return CodexRecoverySnapshot(self.state.thread_id, "idle", None, ())

    async def recover_current_thread(
        self,
        thread_id: str,
        *,
        confirmed_terminal_ids: set[str] | frozenset[str] = frozenset(),
        confirm_interrupt: bool = False,
    ) -> CodexRecoverySnapshot | None:
        self.recovery_calls.append(
            (thread_id, set(confirmed_terminal_ids), confirm_interrupt)
        )
        return self.recovery_result

    async def list_background_terminals(self) -> list[CodexBackgroundTerminal]:
        self.background_terminal_list_calls += 1
        return list(self.background_terminals)

    async def terminate_background_terminal(self, process_id: str) -> bool:
        if not any(item.process_id == process_id for item in self.background_terminals):
            return False
        self.terminated_background_terminals.append(process_id)
        self.background_terminals = [
            item for item in self.background_terminals if item.process_id != process_id
        ]
        return True

    async def list_threads(self) -> list[CodexThread]:
        return self.threads

    async def list_mcp_servers(self) -> list[Any]:
        return []

    async def last_assistant_message(self, _thread_id: str) -> str:
        return "latest assistant response"

    async def select_thread(self, thread_id: str, cwd: Path, *, force: bool = False) -> None:
        self.select_calls.append((thread_id, cwd))
        if self.select_error is not None:
            raise self.select_error
        self.state.thread_id = thread_id
        self.state.cwd = str(cwd)
        self.working_directory = cwd

    async def fork_thread_by_id(self, thread_id: str, cwd: Path) -> str:
        self.fork_by_id_calls.append((thread_id, cwd))
        if self.fork_by_id_error is not None:
            raise self.fork_by_id_error
        self.state.thread_id = "thread-forked-from-picker"
        self.state.cwd = str(cwd)
        self.working_directory = cwd
        return self.state.thread_id

    @property
    def selected_model(self) -> str | None:
        return self.state.selected_model

    @property
    def selected_effort(self) -> str | None:
        return self.state.selected_effort

    async def list_models(self) -> list[CodexModel]:
        return self.models

    def default_model(self, models: list[CodexModel]) -> str | None:
        return next((model.model for model in models if model.is_default), None)

    async def select_model(self, model: str | None, effort: str | None = None) -> None:
        self.state.selected_model = model
        self.state.selected_effort = effort

    async def toggle_fast_mode(self) -> bool:
        if self.fast_error is not None:
            raise self.fast_error
        self.fast_enabled = not self.fast_enabled
        return self.fast_enabled

    async def list_skills(self, *, force_reload: bool = False) -> list[CodexSkill]:
        return list(self.skills)

    async def set_skill_enabled(self, skill: CodexSkill, enabled: bool) -> None:
        self.skill_enabled_calls.append((skill.name, enabled))
        self.skills = [
            CodexSkill(item.name, item.description, item.path, enabled, item.display_name)
            if item.path == skill.path
            else item
            for item in self.skills
        ]

    def select_skill_for_next_turn(self, skill: CodexSkill) -> None:
        self.state.pending_skill_name = skill.name
        self.state.pending_skill_path = skill.path

    def clear_selected_skill(self) -> None:
        self.state.pending_skill_name = None
        self.state.pending_skill_path = None

    def current_memory_mode(self) -> str | None:
        if self.state.thread_id:
            return self.memory_mode
        return self.state.selected_memory_mode

    async def read_memory_settings(self) -> CodexMemorySettings:
        return self.memory_settings

    async def set_memory_mode(self, mode: str) -> None:
        self.memory_mode = mode
        self.state.selected_memory_mode = mode

    async def toggle_memory_generation(self) -> bool:
        self.memory_settings = CodexMemorySettings(
            self.memory_settings.feature_enabled,
            self.memory_settings.use_memories,
            not self.memory_settings.generate_memories,
            self.memory_settings.user_config_version,
        )
        return self.memory_settings.generate_memories

    async def enable_memories_feature(self) -> None:
        self.memory_settings = CodexMemorySettings(
            True,
            self.memory_settings.use_memories,
            self.memory_settings.generate_memories,
            self.memory_settings.user_config_version,
        )

    async def reset_memories(self) -> None:
        self.memories_reset = True

    async def get_goal(self) -> CodexGoal | None:
        return self.goal

    async def set_goal(self, objective: str, prompt_message_id: int | None = None) -> CodexGoal:
        self.goal = CodexGoal(objective, "active", None, 0, 0)
        self.state.thread_id = self.state.thread_id or "thread-goal"
        return self.goal

    async def set_goal_status(
        self,
        status: str,
        prompt_message_id: int | None = None,
    ) -> CodexGoal:
        if self.goal is None:
            raise RuntimeError("No Goal")
        self.goal = CodexGoal(
            self.goal.objective,
            status,
            self.goal.token_budget,
            self.goal.tokens_used,
            self.goal.time_used_seconds,
        )
        return self.goal

    async def clear_goal(self) -> bool:
        had_goal = self.goal is not None
        self.goal = None
        return had_goal

    async def toggle_plan_mode(self) -> str:
        self.collaboration_mode = "default" if self.collaboration_mode == "plan" else "plan"
        self.collaboration_mode_calls.append((self.collaboration_mode, False))
        return self.collaboration_mode

    async def set_collaboration_mode(
        self,
        mode: str,
        *,
        create_if_missing: bool = False,
    ) -> None:
        self.collaboration_mode = mode
        self.collaboration_mode_calls.append((mode, create_if_missing))

class FailingCodex(FakeCodex):
    async def send_message(
        self,
        _text: str,
        _prompt_message_id: int | None = None,
        *,
        additional_input_items: list[dict[str, Any]] | None = None,
    ) -> str:
        raise RpcError(-32602, "Invalid request: unsupported request field")


class FakeAppServer:
    def __init__(
        self,
        *,
        closing_resume_failures: int = 0,
        missing_thread: bool = True,
        active_writer_on_resume: bool = False,
    ) -> None:
        self.is_running = False
        self.start_calls = 0
        self.stop_calls = 0
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.closing_resume_failures = closing_resume_failures
        self.missing_thread = missing_thread
        self.active_writer_on_resume = active_writer_on_resume
        self.account = {"email": "codex@example.com", "planType": "plus"}
        self.responses: dict[str, dict[str, Any]] = {}
        self.response_sequences: dict[str, list[dict[str, Any]]] = {}
        self.response_errors: dict[str, Exception] = {}
        self.response_error_sequences: dict[str, list[Exception]] = {}
        self.codex_config = {
            "features": {"memories": True},
            "memories": {"use_memories": False, "generate_memories": False},
        }
        self.config_version = "config-v1"
        self.skills: list[dict[str, Any]] = []
        self.goals: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        self.start_calls += 1
        self.is_running = True

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_seconds: int = 90,
    ) -> dict[str, Any]:
        params = params or {}
        self.requests.append((method, params))
        error_sequence = self.response_error_sequences.get(method)
        if error_sequence:
            raise error_sequence.pop(0)
        if method in self.response_errors:
            raise self.response_errors[method]
        sequence = self.response_sequences.get(method)
        if sequence:
            return sequence.pop(0)
        if method in self.responses:
            return self.responses[method]
        if method == "config/read":
            return {
                "config": self.codex_config,
                "layers": [{"name": {"type": "user", "file": "/tmp/config.toml"}, "version": self.config_version}],
                "origins": {},
            }
        if method == "config/batchWrite":
            for edit in params.get("edits", []):
                if not isinstance(edit, dict):
                    continue
                key_path = edit.get("keyPath")
                if not isinstance(key_path, str):
                    continue
                parts = key_path.split(".")
                target = self.codex_config
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = edit.get("value")
            self.config_version = f"config-v{len(self.requests)}"
            return {}
        if method == "skills/list":
            cwd = next(iter(params.get("cwds", [])), None)
            return {"data": [{"cwd": cwd, "skills": list(self.skills), "errors": []}]}
        if method == "skills/config/write":
            path = params.get("path")
            enabled = params.get("enabled") is True
            for skill in self.skills:
                if skill.get("path") == path:
                    skill["enabled"] = enabled
            return {"effectiveEnabled": enabled}
        if method == "thread/memoryMode/set":
            return {}
        if method == "thread/goal/get":
            return {"goal": self.goals.get(params.get("threadId"))}
        if method == "thread/goal/set":
            thread_id = params.get("threadId")
            goal = dict(self.goals.get(thread_id, {}))
            if isinstance(params.get("objective"), str):
                goal["objective"] = params["objective"]
                goal.setdefault("tokensUsed", 0)
                goal.setdefault("timeUsedSeconds", 0)
            if isinstance(params.get("status"), str):
                goal["status"] = params["status"]
            if type(params.get("tokenBudget")) is int:
                goal["tokenBudget"] = params["tokenBudget"]
            goal["threadId"] = thread_id
            self.goals[thread_id] = goal
            return {"goal": goal}
        if method == "thread/goal/clear":
            thread_id = params.get("threadId")
            cleared = self.goals.pop(thread_id, None) is not None
            return {"cleared": cleared}
        if method == "model/list":
            return {
                "data": [
                    {
                        "model": "gpt-6-luna",
                        "displayName": "GPT-6-Luna",
                        "isDefault": True,
                        "supportedReasoningEfforts": [{"reasoningEffort": "high"}],
                        "defaultReasoningEffort": "high",
                        "serviceTiers": [{"id": "priority", "name": "Fast", "description": "fast inference"}],
                    }
                ],
                "nextCursor": None,
            }
        if method == "thread/resume":
            if self.closing_resume_failures:
                self.closing_resume_failures -= 1
                raise RpcError(-32600, "thread thread-id is closing; retry thread/resume after the thread is closed")
            if self.active_writer_on_resume:
                raise RpcError(-32600, "thread thread-id already has an active writer")
            if self.missing_thread:
                raise RpcError(-32600, "no rollout found for thread id thread-gone")
            return {
                "thread": {"id": params["threadId"]},
                "model": "gpt-6-luna",
                "reasoningEffort": "high",
                "serviceTier": "default",
                "collaborationMode": {
                    "mode": "default",
                    "settings": {
                        "model": "gpt-6-luna",
                        "reasoning_effort": "high",
                        "developer_instructions": None,
                    },
                },
            }
        if method == "collaborationMode/list":
            return {
                "data": [
                    {"name": "Plan", "mode": "plan", "model": "gpt-6-luna"},
                    {"name": "Default", "mode": "default", "model": "gpt-6-luna"},
                ]
            }
        if method == "thread/unsubscribe":
            return {"status": "unsubscribed"}
        if method == "thread/backgroundTerminals/list":
            return {"data": [], "nextCursor": None}
        if method == "thread/backgroundTerminals/terminate":
            return {"terminated": False}
        return {"thread": {"id": "thread-fresh"}}

    async def stop(self) -> None:
        self.stop_calls += 1
        self.is_running = False


class CodexActivityFormattingTests(unittest.TestCase):
    def test_plain_text_chunks_respect_limit_without_markdown_fences(self) -> None:
        source = ("word " * 80).strip()
        chunks = split_plain_text(source, limit=64)

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 64 for chunk in chunks))
        self.assertEqual(" ".join(chunks), source)

    def test_file_change_groups_include_added_edited_and_deleted_line_counts(self) -> None:
        groups = file_change_groups(
            {
                "changes": [
                    {"path": "src/new.py", "kind": {"type": "add"}, "diff": "+one\n+two"},
                    {
                        "path": "src/changed.py",
                        "kind": {"type": "update"},
                        "diff": "--- old\n+++ new\n-old\n+new\n+extra",
                    },
                    {"path": "src/old.py", "kind": {"type": "delete"}, "diff": "-one\n-two"},
                ]
            }
        )

        self.assertEqual(groups["add"], ["src/new.py (+2 -0)"])
        self.assertEqual(groups["update"], ["src/changed.py (+2 -1)"])
        self.assertEqual(groups["delete"], ["src/old.py (+0 -2)"])

    def test_explored_actions_report_read_paths_and_search_targets(self) -> None:
        actions = [
            {"type": "read", "path": "src/main.py", "name": "read_file", "command": "cat"},
            {"type": "listFiles", "path": "tests", "command": "ls"},
            {"type": "search", "path": "src", "query": "turn/start", "command": "rg"},
        ]

        self.assertEqual(
            explored_action_summaries(actions, render=lambda key, **values: _catalog_text(key, **values)),
            [
                "src/main.py",
                _catalog_text("detailed_log.values.explored_action.files_in", target="tests"),
                _catalog_text("detailed_log.values.explored_action.search_in", query="turn/start", target="src"),
            ],
        )

    def test_web_search_detail_uses_action_specific_query_or_url(self) -> None:
        self.assertEqual(
            web_search_detail(
                {"query": "fallback", "action": {"type": "search", "queries": ["one", "two"]}}
            ),
            "one; two",
        )
        self.assertEqual(
            web_search_detail({"query": "open", "action": {"type": "openPage", "url": "https://example.com"}}),
            "https://example.com",
        )
        self.assertEqual(
            web_search_detail(
                {"query": "find", "action": {"type": "findInPage", "pattern": "release", "url": "https://example.com"}}
            ),
            "release — https://example.com",
        )

    def test_image_saved_path_is_relative_and_hides_unrelated_host_paths(self) -> None:
        self.assertEqual(
            image_saved_path(
                {"savedPath": "/workspace/project/assets/generated.png"},
                "/workspace/project",
            ),
            "assets/generated.png",
        )
        self.assertEqual(
            image_saved_path({"savedPath": "/tmp/generated.png"}, "/workspace/project"),
            "generated.png",
        )
        self.assertEqual(
            image_saved_path({"savedPath": "../../private/generated.png"}, "/workspace/project"),
            "generated.png",
        )


class CodexActivityHandlerTests(unittest.IsolatedAsyncioTestCase):
    def _controller(self, workspace: Path) -> tuple[TelegramUserController, FakeTelegram]:
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
        controller = TelegramUserController(config, telegram, object(), 101)  # type: ignore[arg-type]
        controller.bind_codex(FakeCodex())  # type: ignore[arg-type]
        return controller, telegram

    async def test_recap_uses_a_normal_turn_without_compacting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/recap")
            )

            self.assertEqual(len(codex.messages), 1)
            self.assertIn("without compacting", codex.messages[0])
            self.assertFalse(codex.compacted)
            self.assertEqual(telegram.messages, [])
            await controller._finish_typing()

    async def test_recap_requires_thread_and_steers_an_active_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/recap")
            )
            self.assertEqual(codex.messages, [])
            self.assertIn("Нет активного диалога", telegram.messages[-1][1])

            codex.state.thread_id = "active-thread"
            codex.is_busy = True
            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", "/recap")
            )
            self.assertEqual(len(codex.messages), 1)
            self.assertIn("Summarize this Codex conversation", codex.messages[0])
            self.assertFalse(
                any("Дождись завершения" in text for _, text in telegram.messages)
            )
            await controller._finish_typing()

    async def test_plan_command_toggles_the_active_thread_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/plan")
            )
            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", "/plan")
            )

            self.assertEqual(codex.collaboration_mode_calls, [("plan", False), ("default", False)])
            self.assertIn("режим планирования", telegram.messages[0][1].lower())
            self.assertIn("обычный режим", telegram.messages[1][1].lower())

    async def test_plan_inline_task_starts_and_busy_plan_task_steers_active_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            task = "Спланируй обновление интерфейса"

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", f"/plan {task}")
            )

            self.assertEqual(codex.collaboration_mode_calls, [("plan", True)])
            self.assertEqual(codex.messages, [task])

            codex.is_busy = True
            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", "/plan another task")
            )
            self.assertEqual(codex.messages, [task, "another task"])
            self.assertEqual(
                codex.collaboration_mode_calls,
                [("plan", True), ("plan", True)],
            )
            self.assertFalse(
                any("Дождись завершения" in text for _, text in telegram.messages)
            )
            await controller._finish_typing()

    async def test_messages_steer_active_turn_and_queue_after_compaction_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"

            codex.is_busy = True
            await controller._send_to_codex(101, "steer this turn", 77)
            self.assertEqual(codex.messages, ["steer this turn"])

            codex.compaction_barrier_pending = True
            codex.compaction_barrier_event.clear()
            await controller._send_to_codex(101, "after compact one", 78)
            await controller._send_to_codex(101, "after compact two", 79)

            self.assertEqual(codex.messages, ["steer this turn"])
            self.assertIn("позиция 1", telegram.messages[-2][1])
            self.assertIn("позиция 2", telegram.messages[-1][1])

            codex.is_busy = False
            codex.compaction_barrier_pending = False
            codex.compaction_barrier_event.set()
            task = controller._compaction_input_queue_task
            self.assertIsNotNone(task)
            await task  # type: ignore[arg-type]

            self.assertEqual(
                codex.messages,
                ["steer this turn", "after compact one", "after compact two"],
            )
            self.assertEqual(len(controller._compaction_input_queue), 0)

    async def test_compact_command_is_accepted_while_a_turn_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.is_busy = True

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/compact")
            )

            self.assertTrue(codex.compacted)
            self.assertEqual(telegram.messages, [])

    async def test_compact_queue_status_is_sent_when_the_barrier_reports_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "waiting-thread"
            codex.compaction_barrier_pending = True

            await controller._on_codex_status("status.compaction_queued")

            self.assertEqual(len(telegram.messages), 1)
            self.assertIn("сжатие поставлено в очередь", telegram.messages[0][1].lower())

    async def test_stop_cancels_compaction_barrier_and_discards_queued_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.compaction_barrier_pending = True
            codex.compaction_barrier_event.clear()
            controller._start_typing()

            await controller._send_to_codex(101, "queued prompt", 78)
            self.assertEqual(len(controller._compaction_input_queue), 1)

            await controller._stop_codex(IncomingMessage(1, 79, 101, 101, "private", "/stop"))

            self.assertFalse(codex.compaction_barrier_pending)
            self.assertEqual(len(controller._compaction_input_queue), 0)
            self.assertIn("удалено сообщений — 1", telegram.messages[-1][1].lower())
            self.assertFalse(controller._typing_enabled)
            task = controller._compaction_input_queue_task
            if task is not None:
                await task

    async def test_goal_command_reads_sets_pauses_resumes_and_clears_native_goal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            objective = "Исправь обработку повторных Telegram updates и проверь тесты"

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", f"/goal {objective}")
            )
            self.assertIsNotNone(codex.goal)
            self.assertEqual(codex.goal.objective, objective)
            self.assertEqual(codex.goal.status, "active")
            self.assertIn("Цель установлена", _plain_html(telegram.messages[-1][1]))

            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", "/goal")
            )
            self.assertIn(objective, telegram.messages[-1][1])
            self.assertIn("выполняется", telegram.messages[-1][1])

            codex.is_busy = True
            await controller._handle_message(
                IncomingMessage(3, 79, 101, 101, "private", "/goal pause")
            )
            self.assertEqual(codex.goal.status, "paused")
            self.assertIn(
                "цель поставлена на паузу",
                _plain_html(telegram.messages[-1][1]).lower(),
            )

            await controller._handle_message(
                IncomingMessage(4, 80, 101, 101, "private", "/goal resume")
            )
            self.assertEqual(codex.goal.status, "active")
            self.assertIn("Цель возобновлена", telegram.messages[-1][1])

            await controller._handle_message(
                IncomingMessage(5, 81, 101, 101, "private", "/goal clear")
            )
            self.assertIsNone(codex.goal)
            self.assertIn("Цель очищена", telegram.messages[-1][1])
            await controller._finish_typing()

    async def test_goal_objective_starts_a_chat_when_none_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/goal Подготовь проект и проверь сборку")
            )

            self.assertEqual(codex.state.thread_id, "thread-goal")
            self.assertIsNotNone(codex.goal)
            self.assertIn("Цель установлена", _plain_html(telegram.messages[-1][1]))

    async def test_goal_edit_collects_a_replacement_and_supports_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.goal = CodexGoal("Old objective", "paused", None, 20, 4)

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/goal edit")
            )
            self.assertIn("новым текстом цели", telegram.messages[-1][1].lower())
            self.assertIsNotNone(controller.pending_goal_edit)

            new_objective = "Finish the new objective"
            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", new_objective)
            )
            self.assertEqual(codex.goal.objective, new_objective)
            self.assertEqual(codex.goal.status, "active")
            self.assertIsNone(controller.pending_goal_edit)

            await controller._handle_message(
                IncomingMessage(3, 79, 101, 101, "private", "/goal edit")
            )
            await controller._handle_message(
                IncomingMessage(4, 80, 101, 101, "private", "/cancel")
            )
            self.assertIsNone(controller.pending_goal_edit)
            self.assertEqual(codex.goal.objective, new_objective)
            self.assertIn("изменение цели отменено", telegram.messages[-1][1].lower())
            await controller._finish_typing()

    async def test_fast_command_only_toggles_without_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()

            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/fast"))
            self.assertTrue(codex.fast_enabled)
            self.assertIn("Быстрый режим включён", _plain_html(telegram.messages[-1][1]))

            await controller._handle_message(IncomingMessage(2, 78, 101, 101, "private", "/fast off"))
            self.assertTrue(codex.fast_enabled)
            self.assertIn("не принимает аргументы", telegram.messages[-1][1])

            await controller._handle_message(IncomingMessage(3, 79, 101, 101, "private", "/fast"))
            self.assertFalse(codex.fast_enabled)
            self.assertIn("стандартная скорость", telegram.messages[-1][1])

    async def test_skills_command_selects_skill_for_the_next_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.skills = [
                CodexSkill("reviewer", "Review changes", "/tmp/skills/reviewer/SKILL.md", True, "Reviewer")
            ]

            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/skills"))
            picker = controller.skill_picker
            self.assertIsNotNone(picker)
            assert picker is not None

            await controller._handle_callback(
                IncomingCallback(2, "callback-use", 101, 101, picker.message_id or 0,
                                f"sk:{picker.token}:a:use")
            )
            await controller._handle_callback(
                IncomingCallback(3, "callback-skill", 101, 101, picker.message_id or 0,
                                f"sk:{picker.token}:use:0")
            )

            self.assertEqual(codex.state.pending_skill_name, "reviewer")
            self.assertEqual(codex.state.pending_skill_path, "/tmp/skills/reviewer/SKILL.md")
            self.assertIn("Reviewer", telegram.edits[-1][2])

    async def test_skills_command_can_disable_a_codex_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.skills = [CodexSkill("reviewer", "Review changes", "/skills/reviewer/SKILL.md", True)]

            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/skills"))
            picker = controller.skill_picker
            self.assertIsNotNone(picker)
            assert picker is not None
            await controller._handle_callback(
                IncomingCallback(2, "callback-manage", 101, 101, picker.message_id or 0,
                                f"sk:{picker.token}:a:manage")
            )
            await controller._handle_callback(
                IncomingCallback(3, "callback-disable", 101, 101, picker.message_id or 0,
                                f"sk:{picker.token}:toggle:0")
            )

            self.assertEqual(codex.skill_enabled_calls, [("reviewer", False)])
            self.assertFalse(codex.skills[0].enabled)
            self.assertIn("общей", telegram.edits[-1][2].lower())

    async def test_memories_command_scopes_use_to_thread_and_confirms_global_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()

            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/memories"))
            picker = controller.memory_picker
            self.assertIsNotNone(picker)
            assert picker is not None

            await controller._handle_callback(
                IncomingCallback(2, "callback-use-memory", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:use:enabled")
            )
            self.assertEqual(codex.state.selected_memory_mode, "enabled")
            memory_markup = telegram.edit_markups[-1]
            memory_labels = [
                button["text"]
                for row in memory_markup["inline_keyboard"]
                for button in row
            ]
            self.assertEqual(
                memory_markup["inline_keyboard"][0:2],
                [
                    [{"text": "› Использовать в этом диалоге ‹", "callback_data": f"mm:{picker.token}:use:enabled"}],
                    [{"text": "Не использовать в этом диалоге", "callback_data": f"mm:{picker.token}:use:disabled"}],
                ],
            )
            self.assertTrue(
                all(len(row) == 1 for row in memory_markup["inline_keyboard"])
            )
            self.assertIn("› Использовать в этом диалоге ‹", memory_labels)

            await controller._handle_callback(
                IncomingCallback(3, "callback-generate", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:generate")
            )
            self.assertFalse(codex.memory_settings.generate_memories)
            await controller._handle_callback(
                IncomingCallback(4, "callback-confirm-generate", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:confirm")
            )
            self.assertTrue(codex.memory_settings.generate_memories)
            self.assertTrue(any("общую настройку" in text.lower() for _chat, _message_id, text in telegram.edits))

    async def test_memories_can_enable_the_codex_feature_for_future_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.memory_settings = CodexMemorySettings(False, False, False, "config-v1")

            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/memories"))
            picker = controller.memory_picker
            self.assertIsNotNone(picker)
            assert picker is not None
            await controller._handle_callback(
                IncomingCallback(2, "callback-enable", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:enable")
            )

            self.assertTrue(codex.memory_settings.feature_enabled)
            self.assertIn("новых диалогов", telegram.edits[-1][2].lower())

    async def test_memories_reset_requires_a_second_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/memories"))
            picker = controller.memory_picker
            self.assertIsNotNone(picker)
            assert picker is not None

            await controller._handle_callback(
                IncomingCallback(2, "reset-start", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:reset")
            )
            self.assertFalse(codex.memories_reset)
            self.assertIn("отменить действие нельзя", _plain_html(telegram.edits[-1][2]).lower())

            await controller._handle_callback(
                IncomingCallback(3, "reset-confirm", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:reset_confirm")
            )
            self.assertTrue(codex.memories_reset)

    async def test_init_prompts_codex_to_create_only_a_missing_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            controller, _telegram = self._controller(project)
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.working_directory = project

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/init")
            )

            self.assertEqual(len(codex.messages), 1)
            self.assertIn(str(project / "AGENTS.md"), codex.messages[0])
            self.assertIn("Create only this AGENTS.md", codex.messages[0])
            self.assertFalse((project / "AGENTS.md").exists())
            await controller._finish_typing()

    async def test_init_does_not_overwrite_an_existing_agents_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            agents_file = project / "AGENTS.md"
            agents_file.write_text("keep this file", encoding="utf-8")
            controller, telegram = self._controller(project)
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.working_directory = project

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/init")
            )

            self.assertEqual(codex.messages, [])
            self.assertIn("AGENTS.md", telegram.messages[-1][1])
            self.assertEqual(agents_file.read_text(encoding="utf-8"), "keep this file")
            await controller._finish_typing()

    async def test_init_refuses_a_symlink_target_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            outside_file = Path(temp_dir) / "outside.md"
            outside_file.write_text("keep this content", encoding="utf-8")
            (project / "AGENTS.md").symlink_to(outside_file)
            controller, telegram = self._controller(project)
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.working_directory = project

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/init")
            )

            self.assertEqual(codex.messages, [])
            self.assertIn("уже существует", telegram.messages[-1][1])
            self.assertEqual(outside_file.read_text(encoding="utf-8"), "keep this content")
            await controller._finish_typing()

    async def test_ps_lists_background_terminals_and_paginates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.background_terminals = [
                CodexBackgroundTerminal(
                    process_id=f"terminal-{index}",
                    command=f"python3 server_{index}.py",
                    cwd=temp_dir,
                    os_pid=2000 + index,
                    cpu_percent=1.25,
                    rss_kb=2048,
                )
                for index in range(11)
            ]

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/ps")
            )

            self.assertIn("python3 server_0.py", telegram.messages[-1][1])
            rows = telegram.reply_markups[-1]["inline_keyboard"]
            self.assertEqual(len(rows), 11)
            self.assertTrue(all(len(row) == 1 for row in rows[:10]))
            navigation = rows[-1]
            next_callback = navigation[2]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(2, "next", 101, 101, 1, next_callback)
            )
            self.assertIn("2/2", telegram.edits[-1][2])
            self.assertIn("server_10.py", telegram.edits[-1][2])
            self.assertNotIn("server_0.py", telegram.edits[-1][2])
            await controller._finish_typing()

    async def test_ps_stops_only_the_selected_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.background_terminals = [
                CodexBackgroundTerminal(
                    process_id="terminal-1",
                    command="python3 -m http.server --password private-test-value 8080",
                    cwd=temp_dir,
                    os_pid=4242,
                    cpu_percent=0.5,
                    rss_kb=1024,
                )
            ]

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/ps")
            )
            view_callback = telegram.reply_markups[-1]["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(2, "view", 101, 101, 1, view_callback)
            )
            self.assertIn("PID 4242", telegram.edits[-1][2])
            self.assertIn("[REDACTED]", telegram.edits[-1][2])
            self.assertNotIn("private-test-value", telegram.edits[-1][2])
            stop_callback = telegram.edit_markups[-1]["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(3, "stop", 101, 101, 1, stop_callback)
            )

            self.assertEqual(codex.terminated_background_terminals, ["terminal-1"])
            self.assertEqual(codex.background_terminals, [])
            self.assertEqual(telegram.edits[-1][2], _catalog_text("ps.label.empty"))
            await controller._finish_typing()

    async def test_ps_requires_a_thread_and_rejects_stale_stop_buttons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/ps")
            )
            self.assertEqual(codex.background_terminal_list_calls, 0)
            self.assertIn("выбери диалог", telegram.messages[-1][1].lower())

            codex.state.thread_id = "thread-before-switch"
            codex.background_terminals = [
                CodexBackgroundTerminal(
                    process_id="terminal-stale",
                    command="sleep 60",
                    cwd=temp_dir,
                    os_pid=9090,
                    cpu_percent=0,
                    rss_kb=128,
                )
            ]
            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", "/ps")
            )
            view_callback = telegram.reply_markups[-1]["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(3, "view", 101, 101, 2, view_callback)
            )
            stop_callback = telegram.edit_markups[-1]["inline_keyboard"][0][0]["callback_data"]
            codex.state.thread_id = "thread-after-switch"

            await controller._handle_callback(
                IncomingCallback(4, "stale-stop", 101, 101, 2, stop_callback)
            )

            self.assertEqual(codex.terminated_background_terminals, [])
            self.assertIn("устарел", telegram.callbacks[-1][1])
            await controller._finish_typing()

    async def test_assistant_commentary_is_sent_in_both_output_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for mode in ("answers", "detailed"):
                with self.subTest(mode=mode):
                    controller, telegram = self._controller(Path(temp_dir) / mode)
                    controller._codex().state.output_mode = mode
                    controller._turn_prompt_message_id = 55

                    await controller._on_codex_assistant_message(
                        "commentary-item",
                        "I will check the task now.",
                    )

                    self.assertEqual(
                        telegram.messages,
                        [(101, "I will check the task now.")],
                    )
                    self.assertEqual(
                        telegram.rich_messages,
                        [(101, "I will check the task now.", False)],
                    )
                    self.assertEqual(telegram.replies, [55])
            self.assertEqual(telegram.notification_flags, [False])

    async def test_quiet_answers_silences_progress_and_notifies_on_final_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            state = controller._codex().state
            state.output_mode = "answers"
            state.quiet_progress = True

            await controller._on_codex_assistant_message(
                "commentary-item",
                "Working through the task.",
            )
            await controller._on_codex_finished(
                {"status": "completed", "text": "Final answer. " * TELEGRAM_TEXT_LIMIT}
            )

            self.assertGreater(len(telegram.messages), 2)
            self.assertTrue(telegram.notification_flags[0])
            self.assertTrue(all(telegram.notification_flags[1:-1]))
            self.assertFalse(telegram.notification_flags[-1])

    async def test_quiet_answers_keeps_terminal_errors_audible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            state = controller._codex().state
            state.output_mode = "answers"
            state.quiet_progress = True

            await controller._on_codex_finished(
                {"status": "failed", "text": "Codex failed", "error": "request failed"}
            )

            self.assertEqual(len(telegram.messages), 1)
            self.assertEqual(telegram.notification_flags, [False])

    async def test_quiet_answers_silences_compaction_progress_but_not_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            state = controller._codex().state
            state.output_mode = "answers"
            state.quiet_progress = True

            await controller._on_codex_status("status.label.compaction_started")
            await controller._on_codex_status("status.label.compaction_failed")

            self.assertEqual(telegram.notification_flags, [True, False])

    async def test_review_result_is_delivered_as_final_text_in_answers_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._codex().state.output_mode = "answers"

            await controller._on_codex_finished(
                {
                    "status": "completed",
                    "text": "Review finding: non-owners can delete the project.",
                }
            )

            self.assertEqual(
                telegram.messages,
                [(101, "Review finding: non-owners can delete the project.")],
            )
            self.assertEqual(
                telegram.rich_messages,
                [(101, "Review finding: non-owners can delete the project.", False)],
            )

    async def test_long_assistant_commentary_is_split_at_telegram_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._turn_prompt_message_id = 77
            long_text = "Start.\n\n```python\n" + "print('progress')\n" * 700 + "```\nDone."

            await controller._on_codex_assistant_message("long-commentary", long_text)

            self.assertGreater(len(telegram.messages), 1)
            self.assertTrue(all(len(text) <= TELEGRAM_TEXT_LIMIT for _, text in telegram.messages))
            self.assertTrue(all(reply == 77 for reply in telegram.replies))
            self.assertTrue(telegram.messages[0][1].startswith("Start."))
            self.assertTrue(telegram.messages[-1][1].endswith("Done."))

    async def test_group_commentary_stays_ordered_and_separate_from_activity_batch(self) -> None:
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
            controller = TelegramUserController(
                config, telegram, object(), -1001, is_group=True
            )  # type: ignore[arg-type]
            controller.bind_codex(FakeCodex())  # type: ignore[arg-type]
            controller._turn_prompt_message_id = 88

            with patch("telecodex.telegram.user_controller.GROUP_ACTIVITY_INTERVAL_SECONDS", 0):
                await controller._send_codex_activity_lines(
                    [("Ran", "pwd")],
                    reply_to_message_id=88,
                )
                await controller._on_codex_assistant_message(
                    "group-commentary",
                    "Found the requested directory.",
                )
                await controller._on_codex_finished(
                    {"status": "completed", "text": "Done."}
                )

            self.assertEqual(len(telegram.messages), 3)
            self.assertIn("• Ran pwd", telegram.messages[0][1])
            self.assertEqual(telegram.messages[1][1], "Found the requested directory.")
            self.assertIn("Done.", telegram.messages[2][1])
            self.assertEqual(telegram.replies, [88, 88, 88])
            self.assertEqual(
                telegram.rich_messages,
                [
                    (-1001, "Found the requested directory.", False),
                    (-1001, "Done.", False),
                ],
            )
            await controller._stop_group_activity()

    async def test_file_change_statuses_separate_added_edited_and_deleted_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))

            await controller._on_codex_status(
                "status.file_change",
                {
                    "status": "completed",
                    "changes": [
                        {"path": "added.py", "kind": {"type": "add"}, "diff": "+line"},
                        {"path": "edited.py", "kind": {"type": "update"}, "diff": "-old\n+new"},
                        {"path": "deleted.py", "kind": {"type": "delete"}, "diff": "-line"},
                    ],
                },
            )

            self.assertIn("• Added added.py (+1 -0)", telegram.messages[0][1])
            self.assertIn("• Edited edited.py (+1 -1)", telegram.messages[0][1])
            self.assertIn("• Deleted deleted.py (+0 -1)", telegram.messages[0][1])

    async def test_web_search_emits_started_and_completed_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            item = {"query": "default", "action": {"type": "search", "query": "Codex updates"}}

            await controller._on_codex_status("detailed_log.values.web_search_started", item)
            await controller._on_codex_status("detailed_log.values.web_search_completed", item)

            self.assertIn("• Browsing the web Codex updates", telegram.messages[0][1])
            self.assertIn("• Searched the web Codex updates", telegram.messages[1][1])

    async def test_web_search_start_without_query_still_emits_activity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))

            await controller._on_codex_status(
                "detailed_log.values.web_search_started",
                {"id": "web-1", "type": "webSearch", "query": "", "action": None},
            )

            self.assertEqual("• Browsing the web...", telegram.messages[0][1])

    async def test_answers_mode_shows_automatic_compaction_start_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._codex().state.output_mode = "answers"

            await controller._on_codex_status("status.label.compaction_started")
            await controller._on_codex_status("status.label.compaction_completed")

            self.assertEqual(
                [text for _, text in telegram.messages],
                [
                    "Codex сжимает контекст...",
                    "Контекст сжат. Codex продолжает задачу.",
                ],
            )

    async def test_typing_action_is_refreshed_during_long_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))

            with patch("telecodex.telegram.user_controller.TYPING_ACTION_INTERVAL_SECONDS", 0.01):
                controller._start_typing()
                await asyncio.sleep(0.04)
                await controller._finish_typing()

            self.assertGreaterEqual(len(telegram.actions), 2)

    async def test_skill_review_image_and_subagent_activity_obey_output_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for mode in ("detailed", "answers"):
                with self.subTest(mode=mode):
                    controller, telegram = self._controller(Path(temp_dir) / mode)
                    controller._codex().state.output_mode = mode
                    controller._codex().working_directory = Path(temp_dir) / mode

                    updates = [
                        ("detailed_log.values.skill_attached", "OpenAI Docs"),
                        ("detailed_log.values.review_started", None),
                        ("detailed_log.values.review_completed", None),
                        ("detailed_log.values.image_generation_started", None),
                        (
                            "detailed_log.values.image_generation_completed",
                            {"savedPath": str(Path(temp_dir) / mode / "assets/generated.png")},
                        ),
                        (
                            "detailed_log.values.image_generation_failed",
                            {"failureType": "usageLimitExceeded"},
                        ),
                        ("detailed_log.values.subagent_started", {"agentPath": "root/researcher"}),
                        ("detailed_log.values.subagent_completed", {"agentPath": "root/researcher"}),
                        ("detailed_log.values.subagent_interrupted", None),
                        ("detailed_log.values.subagent_failed", None),
                    ]
                    for key, details in updates:
                        await controller._on_codex_status(key, details)

                    if mode == "detailed":
                        rendered = [text for _, text in telegram.messages]
                        self.assertEqual(
                            rendered,
                            [
                                "• Skill attached: OpenAI Docs",
                                "• Code review started",
                                "• Code review finished",
                                "• Generating image",
                                "• Image generated assets/generated.png",
                                "• Image generation failed usage limit exceeded",
                                "• Subagent started root/researcher",
                                "• Subagent completed root/researcher",
                                "• Subagent interrupted",
                                "• Subagent failed",
                            ],
                        )
                        self.assertEqual(telegram.message_entities[0][0]["type"], "bold")
                        self.assertEqual(telegram.message_entities[0][0]["offset"], 2)
                        self.assertEqual(
                            telegram.message_entities[0][0]["length"],
                            len("Skill attached:"),
                        )
                    else:
                        self.assertEqual(telegram.messages, [])

    async def test_completed_command_reports_explored_file_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._command_messages["command-1"] = RunningCommand(
                message_id=55,
                started_at=0,
                command="cat src/main.py",
            )

            await controller._on_codex_command_completed(
                {
                    "id": "command-1",
                    "command": "cat src/main.py",
                    "status": "completed",
                    "exitCode": 0,
                    "durationMs": 50,
                    "aggregatedOutput": "source",
                    "commandActions": [
                        {"type": "read", "path": "src/main.py", "name": "read_file", "command": "cat"}
                    ],
                }
            )

            self.assertEqual(len(telegram.messages), 2)
            expected_header = _catalog_text(
                "detailed_log.label.command_result_header",
                state=_catalog_text("detailed_log.values.command_succeeded"),
                duration=_catalog_text("detailed_log.values.duration_seconds", seconds="0.1"),
                exit_code="0",
            )
            self.assertIn(expected_header, telegram.messages[0][1])
            self.assertIn("• Explored src/main.py", telegram.messages[1][1])
            self.assertEqual(telegram.replies[1], 55)

    async def test_answers_mode_hides_codex_activity_but_keeps_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._codex().state.output_mode = "answers"

            await controller._on_codex_status(
                "detailed_log.label.command_running",
                "pwd",
                item_id="command-1",
            )
            await controller._on_codex_status(
                "detailed_log.values.web_search_started",
                {"query": "Codex"},
            )
            await controller._on_codex_status(
                "status.file_change",
                {"status": "completed", "changes": [{"path": "main.py", "kind": {"type": "add"}}]},
            )
            self.assertIsNone(controller._command_messages["command-1"].message_id)

            await controller._on_codex_command_completed(
                {
                    "id": "command-1",
                    "command": "pwd",
                    "status": "completed",
                    "exitCode": 0,
                    "durationMs": 1,
                    "aggregatedOutput": "/project",
                }
            )
            await controller._on_codex_finished({"status": "completed", "text": "Done."})

            self.assertNotIn("command-1", controller._command_messages)
            self.assertEqual([text for _, text in telegram.messages], ["Done."])

    async def test_mode_picker_updates_same_message_and_persists_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))

            await controller._handle_message(
                IncomingMessage(1, 1, 101, 101, "private", "/mode")
            )

            self.assertEqual(len(telegram.messages), 1)
            markup = telegram.reply_markups[0]
            self.assertEqual([len(row) for row in markup["inline_keyboard"]], [1, 1, 1])
            callback_data = markup["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(2, "callback-1", 101, 101, 1, callback_data)
            )

            self.assertEqual(controller._codex().state.output_mode, "answers")
            self.assertEqual(len(telegram.messages), 1)
            self.assertEqual(telegram.edits[0][1], 1)
            self.assertIn(_catalog_text("mode.buttons.names.answers"), telegram.edits[0][2])
            self.assertEqual(
                telegram.edit_markups[-1]["inline_keyboard"][0][0]["text"],
                "› Только ответы ‹",
            )

    async def test_mode_picker_selects_quiet_answers_without_changing_content_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            await controller._handle_message(
                IncomingMessage(1, 1, 101, 101, "private", "/mode")
            )
            callback_data = telegram.reply_markups[0]["inline_keyboard"][1][0]["callback_data"]

            await controller._handle_callback(
                IncomingCallback(2, "quiet-callback", 101, 101, 1, callback_data)
            )

            state = controller._codex().state
            self.assertEqual(state.output_mode, "answers")
            self.assertTrue(state.quiet_progress)
            self.assertIn(_catalog_text("mode.buttons.names.quiet_answers"), telegram.edits[-1][2])

    async def test_quiet_answers_keeps_codex_question_audible_and_hides_delivery_ack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._codex().state.output_mode = "answers"
            controller._codex().state.quiet_progress = True
            task = asyncio.create_task(
                controller._ask_user_input(
                    {"questions": [{"id": "q1", "question": "Which format?"}]}
                )
            )
            await asyncio.sleep(0)

            self.assertEqual(len(telegram.messages), 1)
            self.assertIn("Which format?", telegram.messages[0][1])
            await controller._resolve_user_input("JSON")

            self.assertEqual(await task, {"answers": {"q1": {"answers": ["JSON"]}}})
            self.assertEqual(len(telegram.messages), 1)
            self.assertEqual(telegram.notification_flags, [False])

    async def test_quiet_answers_keeps_permission_confirmation_audible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._codex().state.output_mode = "answers"
            controller._codex().state.quiet_progress = True
            task = asyncio.create_task(
                controller._ask_approval(
                    "item/commandExecution/requestApproval",
                    {"command": "npm test"},
                )
            )
            await asyncio.sleep(0)

            self.assertEqual(len(telegram.messages), 1)
            markup = telegram.reply_markups[0]
            self.assertEqual([len(row) for row in markup["inline_keyboard"]], [1, 1, 1])
            callback_data = markup["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(1, "approval-callback", 101, 101, 1, callback_data)
            )

            self.assertEqual(await task, "accept")
            self.assertIn("npm test", telegram.messages[0][1])
            self.assertEqual(telegram.notification_flags, [False])
            self.assertEqual(telegram.edits, [])
            self.assertEqual(
                telegram.reply_markup_edits,
                [(101, 1, {"inline_keyboard": []})],
            )


class TelegramAudioParsingTests(unittest.TestCase):
    def test_voice_message_is_parsed_with_audio_metadata(self) -> None:
        incoming = parse_update(
            {
                "update_id": 1,
                "message": {
                    "message_id": 2,
                    "from": {"id": 101},
                    "chat": {"id": 101, "type": "private"},
                    "voice": {
                        "file_id": "voice-file",
                        "file_unique_id": "voice-unique",
                        "duration": 12,
                        "mime_type": "audio/ogg",
                        "file_size": 2048,
                    },
                },
            }
        )

        self.assertIsInstance(incoming, IncomingMessage)
        self.assertEqual(incoming.attachments[0].kind, "voice")  # type: ignore[union-attr]
        self.assertEqual(incoming.attachments[0].duration_seconds, 12)  # type: ignore[union-attr]
        self.assertEqual(incoming.attachments[0].file_name, "voice.ogg")  # type: ignore[union-attr]

    def test_audio_document_is_classified_as_audio(self) -> None:
        incoming = parse_update(
            {
                "update_id": 1,
                "message": {
                    "message_id": 2,
                    "from": {"id": 101},
                    "chat": {"id": 101, "type": "private"},
                    "document": {
                        "file_id": "audio-file",
                        "file_name": "request.mp3",
                        "mime_type": "audio/mpeg",
                        "file_size": 4096,
                    },
                },
            }
        )

        self.assertIsInstance(incoming, IncomingMessage)
        self.assertEqual(incoming.attachments[0].kind, "audio")  # type: ignore[union-attr]


class LocalAudioTranscriberTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcriber_joins_segments_and_reuses_loaded_model(self) -> None:
        class Segment:
            def __init__(self, text: str) -> None:
                self.text = text

        class Model:
            calls = 0

            def transcribe(self, *_args: Any, **_kwargs: Any) -> tuple[list[Segment], object]:
                self.calls += 1
                return [Segment(" Hello"), Segment(" world ")], object()

        transcriber = LocalAudioTranscriber(max_duration_seconds=30)
        model = Model()
        transcriber._model = model
        with patch("telecodex.audio.transcriber._audio_duration_seconds", return_value=5):
            first = await transcriber.transcribe(Path("voice.ogg"), duration_hint=5)
            second = await transcriber.transcribe(Path("voice.ogg"), duration_hint=5)

        self.assertEqual(first, "Hello world")
        self.assertEqual(second, "Hello world")
        self.assertEqual(model.calls, 2)

    async def test_transcriber_rejects_overlong_audio_before_loading_model(self) -> None:
        transcriber = LocalAudioTranscriber(max_duration_seconds=30)

        with self.assertRaises(AudioTranscriptionError) as raised:
            await transcriber.transcribe(Path("voice.ogg"), duration_hint=31)

        self.assertEqual(raised.exception.text_key, "audio.label.too_long")
        self.assertIsNone(transcriber._model)


class GroupTriggerParsingTests(unittest.TestCase):
    def test_words_are_trimmed_casefolded_and_deduplicated(self) -> None:
        self.assertEqual(
            parse_trigger_words(
                f" {CODEX_TRIGGER_CYRILLIC}, {KAWAII_TRIGGER_CYRILLIC}, {CODEX_TRIGGER_CYRILLIC.upper()} ,,"
            ),
            (CODEX_TRIGGER_CYRILLIC, KAWAII_TRIGGER_CYRILLIC),
        )

    def test_trigger_matches_only_as_the_first_complete_word(self) -> None:
        words = parse_trigger_words(f"{CODEX_TRIGGER_CYRILLIC}, {KAWAII_TRIGGER_CYRILLIC}")
        self.assertEqual(
            strip_group_trigger(f"{KAWAII_TRIGGER_CYRILLIC.capitalize()}, help with a project", words),
            "help with a project",
        )
        self.assertEqual(
            strip_group_trigger(f"{CODEX_TRIGGER_CYRILLIC}: run the tests", words),
            "run the tests",
        )
        self.assertIsNone(strip_group_trigger(f"hello, {KAWAII_TRIGGER_CYRILLIC} check", words))
        self.assertIsNone(strip_group_trigger(f"{CODEX_TRIGGER_CYRILLIC}project", words))

    def test_empty_or_invalid_trigger_lists_are_rejected(self) -> None:
        for value in ("", " , , ", "two words", "word, bad!"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_trigger_words(value)


class MultiUserConfigTests(unittest.TestCase):
    def test_load_config_reads_local_audio_transcription_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "101",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
                "TELECODEX_TRANSCRIPTION_MODEL": "base",
                "TELECODEX_TRANSCRIPTION_DEVICE": "cpu",
                "TELECODEX_TRANSCRIPTION_COMPUTE_TYPE": "int8",
                "TELECODEX_AUDIO_MAX_DURATION_SECONDS": "120",
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                config = load_config()

        self.assertEqual(config.transcription_model, "base")
        self.assertEqual(config.transcription_device, "cpu")
        self.assertEqual(config.transcription_compute_type, "int8")
        self.assertEqual(config.max_audio_duration_seconds, 120)

    def test_load_config_reads_telegram_socks_proxy_and_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "101",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
                "TELEGRAM_PROXY_URL": "socks5://127.0.0.1:2080",
                "TELEGRAM_PROXY_USERNAME": "proxy-user",
                "TELEGRAM_PROXY_PASSWORD": "proxy-secret",
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                config = load_config()

        self.assertEqual(
            config.telegram_proxy,
            ProxySettings("socks5://127.0.0.1:2080", "proxy-user", "proxy-secret"),
        )
        self.assertNotIn("proxy-secret", repr(config))

    def test_load_config_leaves_telegram_proxy_unset_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "101",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                config = load_config()

        self.assertIsNone(config.telegram_proxy)

    def test_load_config_rejects_proxy_uri_with_embedded_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "101",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
                "TELEGRAM_PROXY_URL": "http://proxy-user:proxy-secret@127.0.0.1:2080",
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                with self.assertRaises(ValueError) as raised:
                    load_config()

        self.assertIn("TELEGRAM_PROXY_URL", str(raised.exception))
        self.assertNotIn("proxy-user", str(raised.exception))
        self.assertNotIn("proxy-secret", str(raised.exception))

    def test_load_config_parses_multiple_ids_and_removes_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "202, 101,303",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                config = load_config()
        self.assertEqual(config.telegram_allowed_user_ids, (202, 101, 303))

    def test_load_config_rejects_empty_id_in_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "101,,202",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                with self.assertRaisesRegex(ValueError, "TELEGRAM_ALLOWED_USER_IDS"):
                    load_config()


class TelegramRichMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_rich_message_passes_markdown_and_reply_metadata(self) -> None:
        api = TelegramAPI("bot-token")
        try:
            with patch.object(api, "call", new_callable=AsyncMock) as call:
                call.return_value = {"message_id": 12}
                result = await api.send_rich_message(
                    101,
                    "# Result\n\n**Done**",
                    reply_to_message_id=55,
                    disable_notification=True,
                )

            self.assertEqual(result, {"message_id": 12})
            method, payload = call.await_args.args
            self.assertEqual(method, "sendRichMessage")
            self.assertEqual(payload["rich_message"], {"markdown": "# Result\n\n**Done**"})
            self.assertEqual(
                payload["reply_parameters"],
                {"message_id": 55, "allow_sending_without_reply": True},
            )
            self.assertTrue(payload["disable_notification"])
        finally:
            await api.close()

    async def test_rich_format_rejection_falls_back_to_plain_send_message(self) -> None:
        api = TelegramAPI("bot-token")
        rejection = TelegramApiError(
            "Telegram API sendRichMessage failed: Bad Request: can't parse markdown",
            method="sendRichMessage",
            error_code=400,
            http_status=400,
        )
        try:
            with patch.object(api, "call", new_callable=AsyncMock) as call:
                call.side_effect = [rejection, {"message_id": 13}]
                result = await api.send_rich_message(101, "Unparsed **text**")

            self.assertEqual(result, {"message_id": 13})
            self.assertEqual(
                [call.await_args_list[index].args[0] for index in range(2)],
                ["sendRichMessage", "sendMessage"],
            )
            self.assertEqual(call.await_args_list[1].args[1]["text"], "Unparsed **text**")
            self.assertNotIn("parse_mode", call.await_args_list[1].args[1])
        finally:
            await api.close()

    async def test_non_format_rich_rejection_does_not_retry(self) -> None:
        api = TelegramAPI("bot-token")
        rejection = TelegramApiError(
            "Telegram API sendRichMessage failed: Bad Request: chat not found",
            method="sendRichMessage",
            error_code=400,
            http_status=400,
        )
        try:
            with patch.object(api, "call", new_callable=AsyncMock) as call:
                call.side_effect = rejection
                with self.assertRaises(TelegramApiError):
                    await api.send_rich_message(101, "No chat")
            call.assert_awaited_once()
        finally:
            await api.close()


class TelegramHtmlMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_message_uses_html_and_keeps_explicit_entities(self) -> None:
        api = TelegramAPI("bot-token")
        try:
            with patch.object(api, "call", new_callable=AsyncMock) as call:
                call.return_value = {"message_id": 21}
                await api.send_message(101, "<strong>Ready</strong>")
                await api.send_message(
                    101,
                    "output",
                    entities=[{"type": "pre", "offset": 0, "length": 6}],
                )

            self.assertEqual(call.await_args_list[0].args[0], "sendMessage")
            self.assertEqual(call.await_args_list[0].args[1]["parse_mode"], "HTML")
            entity_payload = call.await_args_list[1].args[1]
            self.assertEqual(entity_payload["entities"], [{"type": "pre", "offset": 0, "length": 6}])
            self.assertNotIn("parse_mode", entity_payload)
        finally:
            await api.close()

    async def test_html_format_rejection_retries_plain_text_for_send_and_edit(self) -> None:
        api = TelegramAPI("bot-token")
        send_rejection = TelegramApiError(
            "Telegram API sendMessage failed: Bad Request: can't parse entities",
            method="sendMessage",
            error_code=400,
        )
        edit_rejection = TelegramApiError(
            "Telegram API editMessageText failed: Bad Request: can't parse entities: Unsupported start tag",
            method="editMessageText",
            error_code=400,
        )
        try:
            with patch.object(api, "call", new_callable=AsyncMock) as call:
                call.side_effect = [send_rejection, {"message_id": 22}, edit_rejection, True]
                await api.send_message(101, "<strong>bad markup</strong>")
                await api.edit_message(101, 22, "<strong>bad markup</strong>")

            self.assertEqual(call.await_args_list[0].args[1]["parse_mode"], "HTML")
            self.assertNotIn("parse_mode", call.await_args_list[1].args[1])
            self.assertEqual(call.await_args_list[2].args[1]["parse_mode"], "HTML")
            self.assertNotIn("parse_mode", call.await_args_list[3].args[1])
        finally:
            await api.close()


class TelegramButtonLayoutTests(unittest.TestCase):
    def test_two_column_rows_keep_short_labels_paired_and_stack_long_labels(self) -> None:
        buttons = [
            {"text": "Русский", "callback_data": "l:ru"},
            {"text": "English", "callback_data": "l:en"},
            {"text": "Локальный процесс (STDIO)", "callback_data": "m:stdio"},
            {"text": "Streamable HTTP", "callback_data": "m:http"},
        ]

        rows = two_column_rows(buttons, 0)

        self.assertEqual(
            [[button["text"] for button in row] for row in rows],
            [["Русский", "English"], ["Локальный процесс (STDIO)"], ["Streamable HTTP"]],
        )
        self.assertEqual(
            [button["callback_data"] for row in rows for button in row],
            ["l:ru", "l:en", "m:stdio", "m:http"],
        )

    def test_selected_button_wraps_the_label_and_respects_telegram_limit(self) -> None:
        controller = object.__new__(TelegramUserController)
        controller.texts = TextCatalog()

        self.assertEqual(controller._selected_button_label("Русский", True), "› Русский ‹")
        long_selected = controller._selected_button_label("Название" * 20, True)
        long_unselected = controller._selected_button_label("Название" * 20, False)
        self.assertLessEqual(len(long_selected), 64)
        self.assertLessEqual(len(long_unselected), 64)
        self.assertTrue(long_selected.startswith("› "))
        self.assertTrue(long_selected.endswith(" ‹"))


class TelegramProxyClientTests(unittest.TestCase):
    def test_api_client_uses_configured_proxy_auth_without_leaking_password(self) -> None:
        proxy = ProxySettings("socks5h://127.0.0.1:2080", "proxy-user", "proxy-secret")
        with patch("telecodex.telegram.api.httpx.AsyncClient") as async_client:
            TelegramAPI("bot-token", proxy)

        client_options = async_client.call_args.kwargs
        configured_proxy = client_options["proxy"]
        self.assertEqual(str(configured_proxy.url), "socks5h://127.0.0.1:2080")
        self.assertEqual(configured_proxy.auth, ("proxy-user", "proxy-secret"))
        self.assertNotIn("proxy-secret", repr(configured_proxy))

    def test_api_client_without_proxy_preserves_httpx_default(self) -> None:
        with patch("telecodex.telegram.api.httpx.AsyncClient") as async_client:
            TelegramAPI("bot-token")

        self.assertNotIn("proxy", async_client.call_args.kwargs)


class MultiUserStateTests(unittest.TestCase):
    def test_migrates_legacy_state_and_keeps_user_threads_and_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            legacy_path = Path(temp_dir) / "state.json"
            legacy_path.write_text(json.dumps({"thread_id": "thread-old", "last_update_id": 55}))
            store = RuntimeStateStore(path, migration_user_id=101, migration_path=legacy_path)

            self.assertEqual(store.load_last_update_id(), 55)
            self.assertEqual(store.for_user(101).load(), RuntimeState("thread-old"))
            store.for_user(202).save(RuntimeState("thread-new", cwd="/project/child"))
            store.save_last_update_id(56)

            reloaded = RuntimeStateStore(path)
            self.assertEqual(reloaded.load_last_update_id(), 56)
            self.assertEqual(reloaded.for_user(101).load().thread_id, "thread-old")
            self.assertEqual(reloaded.for_user(202).load().thread_id, "thread-new")
            self.assertEqual(reloaded.for_user(202).load().cwd, "/project/child")
            self.assertEqual(json.loads(legacy_path.read_text()), {"thread_id": "thread-old", "last_update_id": 55})

    def test_deleting_threads_clears_metadata_and_all_user_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            user_one = root_store.for_user(101)
            user_two = root_store.for_user(202)
            user_one.save(RuntimeState("thread-shared", cwd="/project/one"))
            user_two.save(RuntimeState("thread-shared", cwd="/project/two"))
            user_one.save_thread(
                "thread-shared",
                ThreadRuntimeState(permission_mode="ask", token_usage={"context_tokens": 10}),
            )

            user_one.delete_threads({"thread-shared"})

            self.assertIsNone(user_one.load().thread_id)
            self.assertIsNone(user_two.load().thread_id)
            self.assertEqual(user_one.load().cwd, "/project/one")
            self.assertEqual(user_two.load().cwd, "/project/two")
            self.assertEqual(user_one.load_thread("thread-shared"), ThreadRuntimeState())

    def test_selected_effort_survives_state_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3").for_user(101)
            store.save(RuntimeState(selected_model="gpt-6-luna", selected_effort="xhigh"))

            self.assertEqual(store.load().selected_model, "gpt-6-luna")
            self.assertEqual(store.load().selected_effort, "xhigh")

    def test_output_mode_is_persisted_per_user_and_old_states_default_to_detailed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            legacy_path = Path(temp_dir) / "state.json"
            legacy_path.write_text(
                json.dumps(
                    {
                        "last_update_id": 0,
                        "users": {
                            "101": {"thread_id": "one"},
                            "202": {"thread_id": "two", "output_mode": "answers"},
                        },
                        "threads": {},
                    }
                )
            )
            store = RuntimeStateStore(path, migration_path=legacy_path)

            self.assertEqual(store.for_user(101).load().output_mode, "detailed")
            self.assertEqual(store.for_user(202).load().output_mode, "answers")
            store.for_user(101).save(RuntimeState(output_mode="answers"))

            reloaded = RuntimeStateStore(path)
            self.assertEqual(reloaded.for_user(101).load().output_mode, "answers")
            self.assertEqual(reloaded.for_user(202).load().output_mode, "answers")

    def test_pending_turn_survives_reload_and_is_isolated_by_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            user_one = root.for_user(101)
            user_two = root.for_user(202)
            pending = PendingTurnState(
                thread_id="thread-one",
                turn_id="turn-one",
                prompt_message_id=41,
                started_at=123,
                phase="running",
                prompt_message_ids=[42],
            )
            user_one.save(RuntimeState(pending_turn=pending))
            user_two.save(RuntimeState(pending_turn=PendingTurnState(
                thread_id="thread-two",
                turn_id="turn-two",
                prompt_message_id=51,
                started_at=124,
            )))

            self.assertEqual(user_one.load().pending_turn, pending)
            self.assertEqual(user_two.load().pending_turn.thread_id, "thread-two")  # type: ignore[union-attr]

            user_one.delete_threads({"thread-one"})

            self.assertIsNone(user_one.load().pending_turn)
            self.assertEqual(user_two.load().pending_turn.thread_id, "thread-two")  # type: ignore[union-attr]

    def test_thread_permissions_and_token_usage_are_shared(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            thread_state = ThreadRuntimeState(
                permission_mode="read_only",
                managed_by_telecodex=True,
                token_usage={"context_tokens": 12000, "model_context_window": 200000},
            )
            store.for_user(101).save_thread("shared-thread", thread_state)

            self.assertEqual(store.for_user(202).load_thread("shared-thread"), thread_state)

    def test_group_context_has_independent_state_and_answers_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            store = RuntimeStateStore(path)
            self.assertFalse(store.attach_group(-100123, "Private team"))

            group = store.for_group(-100123)
            self.assertEqual(group.load().output_mode, "answers")
            group.save(RuntimeState(thread_id="group-thread", cwd="/project/team", output_mode="detailed"))
            store.for_user(101).save(RuntimeState(thread_id="private-thread", cwd="/project/personal"))

            reloaded = RuntimeStateStore(path)
            self.assertEqual(reloaded.list_attached_groups(), {-100123: "Private team"})
            self.assertEqual(reloaded.for_group(-100123).load().thread_id, "group-thread")
            self.assertEqual(reloaded.for_group(-100123).load().output_mode, "detailed")
            self.assertEqual(reloaded.for_user(101).load().thread_id, "private-thread")

            self.assertTrue(reloaded.detach_group(-100123))
            self.assertFalse(reloaded.is_group_attached(-100123))
            self.assertEqual(reloaded.for_group(-100123).load().thread_id, "group-thread")

    def test_group_trigger_setup_persists_and_updates_without_replacing_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            store = RuntimeStateStore(path)
            self.assertEqual(store.begin_group_setup(-100124, "Team", 101), "started")
            self.assertFalse(store.is_group_attached(-100124))
            self.assertEqual(store.pending_group_setup_user(-100124), 101)

            reloaded = RuntimeStateStore(path)
            self.assertEqual(reloaded.pending_group_setup_user(-100124), 101)
            self.assertIsNone(reloaded.complete_group_setup(-100124, 202, ("codex",)))
            self.assertEqual(
                reloaded.complete_group_setup(
                    -100124, 101, (CODEX_TRIGGER_CYRILLIC, KAWAII_TRIGGER_CYRILLIC)
                ),
                "attached",
            )
            self.assertTrue(reloaded.is_group_attached(-100124))
            russian_triggers = (CODEX_TRIGGER_CYRILLIC, KAWAII_TRIGGER_CYRILLIC)
            self.assertEqual(reloaded.group_trigger_words(-100124), russian_triggers)

            reloaded.for_group(-100124).save(RuntimeState(thread_id="shared-thread"))
            self.assertEqual(reloaded.begin_group_setup(-100124, "Team", 202), "started")
            self.assertEqual(reloaded.group_trigger_words(-100124), russian_triggers)
            self.assertTrue(reloaded.cancel_group_setup(-100124))
            self.assertTrue(reloaded.is_group_attached(-100124))
            self.assertEqual(reloaded.for_group(-100124).load().thread_id, "shared-thread")
            self.assertEqual(reloaded.group_trigger_words(-100124), russian_triggers)

            self.assertEqual(reloaded.begin_group_setup(-100124, "Team", 202), "started")
            self.assertEqual(
                reloaded.complete_group_setup(-100124, 202, ("help",)),
                "updated",
            )
            self.assertEqual(reloaded.group_trigger_words(-100124), ("help",))
            self.assertEqual(reloaded.for_group(-100124).load().thread_id, "shared-thread")

    def test_old_group_state_loads_without_trigger_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            legacy_path = Path(temp_dir) / "state.json"
            legacy_path.write_text(
                json.dumps(
                    {
                        "last_update_id": 8,
                        "users": {},
                        "groups": {
                            "-100125": {
                                "title": "Older group",
                                "attached": True,
                                "thread_id": "old-group-thread",
                            }
                        },
                        "threads": {},
                        "pending_media_groups": {},
                    }
                ),
                encoding="utf-8",
            )
            store = RuntimeStateStore(path, migration_path=legacy_path)
            self.assertTrue(store.is_group_attached(-100125))
            self.assertEqual(store.group_trigger_words(-100125), ())
            self.assertEqual(store.for_group(-100125).load().thread_id, "old-group-thread")
            self.assertEqual(store.begin_group_setup(-100125, "Older group", 101), "started")
            self.assertEqual(
                store.complete_group_setup(-100125, 101, ("codex",)),
                "updated",
            )
            self.assertEqual(store.group_trigger_words(-100125), ("codex",))
            self.assertEqual(store.for_group(-100125).load().thread_id, "old-group-thread")

    def test_group_state_rejects_non_group_chat_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            with self.assertRaisesRegex(ValueError, "group chat_id"):
                store.attach_group(101, "not a group")


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


class AudioMessageHandlerTests(unittest.IsolatedAsyncioTestCase):
    def _controller(
        self,
        workspace: Path,
        transcriber: FakeAudioTranscriber,
        *,
        is_group: bool = False,
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
        codex.working_directory = workspace
        controller = TelegramUserController(
            config,
            telegram,
            object(),
            -100900 if is_group else 101,
            is_group=is_group,
            group_trigger_words=lambda: (CODEX_TRIGGER_CYRILLIC,),
            audio_transcriber=transcriber,  # type: ignore[arg-type]
        )
        controller.bind_codex(codex)  # type: ignore[arg-type]
        return controller, telegram, codex

    def _voice(self, *, duration: int = 5) -> TelegramAttachment:
        return TelegramAttachment(
            file_id="voice-file",
            kind="voice",
            file_size=2048,
            file_name="voice.ogg",
            mime_type="audio/ogg",
            duration_seconds=duration,
        )

    async def test_voice_is_transcribed_echoed_and_sent_as_codex_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcriber = FakeAudioTranscriber("check configuration")
            controller, telegram, codex = self._controller(workspace, transcriber)

            await controller._handle_message(
                IncomingMessage(
                    1,
                    22,
                    101,
                    101,
                    "private",
                    "",
                    attachments=(self._voice(),),
                )
            )
            await controller._finish_typing()

            self.assertEqual(codex.messages, ["check configuration"])
            self.assertEqual(transcriber.calls[0][1], 5)
            self.assertIn("check configuration", telegram.messages[0][1])
            self.assertEqual(telegram.replies[0], 22)
            self.assertFalse(list((workspace / ".telecodex" / "attachments" / "101").glob("*voice.ogg")))

    async def test_audio_over_duration_limit_is_rejected_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcriber = FakeAudioTranscriber("must not be transcribed")
            controller, telegram, codex = self._controller(workspace, transcriber)

            await controller._handle_message(
                IncomingMessage(
                    1,
                    22,
                    101,
                    101,
                    "private",
                    "",
                    attachments=(self._voice(duration=301),),
                )
            )

            self.assertEqual(transcriber.calls, [])
            self.assertEqual(codex.messages, [])
            self.assertIn(_catalog_text("audio.label.too_long", seconds=300), telegram.messages[0][1])

    async def test_group_voice_uses_transcribed_trigger_and_ignores_unaddressed_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcript = f"{CODEX_TRIGGER_CYRILLIC.capitalize()}, check configuration"
            transcriber = FakeAudioTranscriber(transcript)
            controller, telegram, codex = self._controller(workspace, transcriber, is_group=True)

            await controller._handle_message(
                IncomingMessage(
                    1,
                    23,
                    101,
                    -100900,
                    "supergroup",
                    "",
                    attachments=(self._voice(),),
                )
            )
            await controller._finish_typing()

            self.assertEqual(codex.messages, ["check configuration"])
            self.assertEqual(len(transcriber.calls), 1)
            self.assertIn(transcript, telegram.messages[0][1])

            quiet_transcriber = FakeAudioTranscriber("check configuration")
            quiet_workspace = workspace / "quiet"
            quiet_workspace.mkdir()
            quiet_controller, quiet_telegram, quiet_codex = self._controller(
                quiet_workspace,
                quiet_transcriber,
                is_group=True,
            )
            await quiet_controller._handle_message(
                IncomingMessage(
                    2,
                    24,
                    101,
                    -100900,
                    "supergroup",
                    "",
                    attachments=(self._voice(),),
                )
            )
            await quiet_controller._finish_typing()

            self.assertEqual(quiet_codex.messages, [])
            self.assertEqual(quiet_telegram.messages, [])

    async def test_group_reply_to_bot_bypasses_trigger_word(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, _telegram, codex = self._controller(
                Path(temp_dir), FakeAudioTranscriber(), is_group=True
            )
            await controller._handle_message(
                IncomingMessage(
                    1,
                    22,
                    101,
                    -100900,
                    "supergroup",
                    "without a trigger word",
                    reply_to_user_id=999,
                )
            )
            self.assertEqual(codex.messages, ["without a trigger word"])

    async def test_group_message_without_trigger_or_bot_reply_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, _telegram, codex = self._controller(
                Path(temp_dir), FakeAudioTranscriber(), is_group=True
            )
            await controller._handle_message(
                IncomingMessage(1, 22, 101, -100900, "supergroup", "ordinary text")
            )
            self.assertEqual(codex.messages, [])


class ModelEffortSelectionTests(unittest.IsolatedAsyncioTestCase):
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

            await controller._handle_message(IncomingMessage(1, 1, 101, 101, "private", "/files"))

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
                IncomingCallback(
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
                IncomingCallback(
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
                IncomingCallback(
                    4,
                    "download-file",
                    101,
                    101,
                    picker.message_id or 0,
                    f"fb:{picker.token}:d",
                )
            )
            self.assertEqual(telegram.documents[-1][2], "note.txt")

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


class MultiUserRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_messages_and_approval_callbacks_are_isolated_by_user(self) -> None:
        state_directory = tempfile.TemporaryDirectory(prefix="telecodex-routing-")
        config = AppConfig(
            telegram_bot_token="token",
            telegram_allowed_user_ids=(101, 202),
            workspace_root=Path("/project"),
            codex_bin="codex",
            codex_model=None,
            network_access=False,
            approval_policy="on-request",
            shell_timeout_seconds=60,
            database_file=Path(state_directory.name) / "state.sqlite3",
        )
        telegram = FakeTelegram()
        sessions: dict[int, FakeCodex] = {}

        def make_session(user_id: int, _controller: Any) -> FakeCodex:
            sessions[user_id] = FakeCodex()
            return sessions[user_id]

        controller = TelegramController(config, telegram, object(), make_session)  # type: ignore[arg-type]
        await controller._handle_update(self._message_update(1, 101, "first task"))
        await controller._handle_update(self._message_update(2, 202, "second task"))
        await controller._handle_update(self._message_update(3, 303, "ignored"))

        self.assertEqual(sessions[101].messages, ["first task"])
        self.assertEqual(sessions[202].messages, ["second task"])
        self.assertEqual(set(sessions), {101, 202})
        await asyncio.sleep(0)
        self.assertEqual({chat_id for chat_id, _ in telegram.actions}, {101, 202})

        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        controller.users[101].pending_approvals["approval-token"] = PendingApproval(
            future=future,
            message_id=9,
            chat_id=101,
        )
        await controller._handle_update(self._callback_update(4, 202, "approval-token", "once"))
        self.assertFalse(future.done())
        await controller._handle_update(self._callback_update(5, 101, "approval-token", "once"))
        self.assertEqual(await future, "accept")

        await controller.close()
        state_directory.cleanup()

    async def test_rpc_failure_detail_is_reported_in_chat(self) -> None:
        state_directory = tempfile.TemporaryDirectory(prefix="telecodex-routing-")
        config = AppConfig(
            telegram_bot_token="token",
            telegram_allowed_user_ids=(101,),
            workspace_root=Path("/project"),
            codex_bin="codex",
            codex_model=None,
            network_access=False,
            approval_policy="on-request",
            shell_timeout_seconds=60,
            database_file=Path(state_directory.name) / "state.sqlite3",
        )
        telegram = FakeTelegram()
        controller = TelegramController(
            config,
            telegram,
            object(),
            lambda _user_id, _controller: FailingCodex(),
        )  # type: ignore[arg-type]

        await controller._handle_update(self._message_update(1, 101, "task"))

        self.assertTrue(
            any(
                "RPC -32602: Invalid request: unsupported request field" in text
                for _chat_id, text in telegram.messages
            )
        )
        await controller.close()
        state_directory.cleanup()

    async def test_group_is_one_shared_context_and_mode_is_available_to_members(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101, 202),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=Path(temp_dir) / "state.sqlite3",
            )
            telegram = FakeTelegram()
            sessions: dict[int, FakeCodex] = {}

            def make_session(context_id: int, _controller: Any) -> FakeCodex:
                codex = FakeCodex()
                if context_id < 0:
                    codex.state.output_mode = state_store.for_group(context_id).load().output_mode
                sessions[context_id] = codex
                return codex

            controller = TelegramController(
                config,
                telegram,
                object(),
                make_session,
                state_store=state_store,
            )  # type: ignore[arg-type]
            await controller._handle_update(self._group_message_update(1, -100900, 303, "/group_add"))
            self.assertFalse(state_store.is_group_attached(-100900))
            self.assertEqual(telegram.messages, [])

            await controller._handle_update(self._group_message_update(2, -100900, 101, "/group_add", "Dev team"))
            self.assertFalse(state_store.is_group_attached(-100900))
            self.assertEqual(state_store.pending_group_setup_user(-100900), 101)
            self.assertEqual(telegram.messages[-1][1], _catalog_text("groups.label.setup_prompt"))
            await controller._handle_update(self._group_message_update(25, -100900, 101, "several words, bad!"))
            self.assertFalse(state_store.is_group_attached(-100900))
            self.assertEqual(state_store.pending_group_setup_user(-100900), 101)
            await controller._handle_update(self._group_message_update(3, -100900, 202, "other, list"))
            self.assertFalse(state_store.is_group_attached(-100900))
            await controller._handle_update(
                self._group_message_update(
                    4,
                    -100900,
                    101,
                    f"{CODEX_TRIGGER_CYRILLIC.capitalize()}, {KAWAII_TRIGGER_CYRILLIC}",
                )
            )
            self.assertTrue(state_store.is_group_attached(-100900))
            codex_triggers = (CODEX_TRIGGER_CYRILLIC, KAWAII_TRIGGER_CYRILLIC)
            self.assertEqual(state_store.group_trigger_words(-100900), codex_triggers)
            await controller._handle_update(self._group_message_update(5, -100900, 101, "plain text is ignored"))
            await controller._handle_update(
                self._group_message_update(6, -100900, 101, f"{CODEX_TRIGGER_CYRILLIC.upper()}, first task")
            )
            await controller._handle_update(
                self._group_message_update(7, -100900, 202, f"{KAWAII_TRIGGER_CYRILLIC} second task")
            )
            await controller._handle_update(
                self._group_message_update(8, -100900, 202, f"{KAWAII_TRIGGER_CYRILLIC}ny third task")
            )
            await controller._handle_update(
                self._group_message_update(9, -100900, 303, f"{CODEX_TRIGGER_CYRILLIC} ignored")
            )
            await controller._handle_update(self._message_update(10, 101, "private task"))
            await controller._handle_update(
                self._group_message_update(26, -100900, 101, CODEX_TRIGGER_CYRILLIC.upper())
            )
            self.assertEqual(telegram.messages[-1][1], _catalog_text("groups.label.trigger_empty"))

            self.assertEqual(sessions[-100900].messages, ["first task", "second task"])
            self.assertEqual(sessions[101].messages, ["private task"])
            self.assertNotIn(303, sessions)

            await controller._handle_update(self._group_message_update(11, -100900, 202, "/group_add"))
            self.assertTrue(state_store.is_group_attached(-100900))
            self.assertEqual(state_store.group_trigger_words(-100900), codex_triggers)
            await controller._handle_update(
                self._group_message_update(12, -100900, 101, f"{KAWAII_TRIGGER_CYRILLIC} old list still works")
            )
            helper_triggers = (BOT_TRIGGER_CYRILLIC, HELPER_TRIGGER_CYRILLIC)
            await controller._handle_update(
                self._group_message_update(13, -100900, 202, ", ".join(helper_triggers))
            )
            self.assertEqual(state_store.group_trigger_words(-100900), helper_triggers)
            await controller._handle_update(
                self._group_message_update(14, -100900, 101, f"{CODEX_TRIGGER_CYRILLIC} old list stopped")
            )
            await controller._handle_update(
                self._group_message_update(15, -100900, 202, f"{BOT_TRIGGER_CYRILLIC.capitalize()}: next task")
            )
            self.assertEqual(
                sessions[-100900].messages,
                ["first task", "second task", "old list still works", "next task"],
            )

            await controller._handle_update(self._group_message_update(16, -100900, 202, "/mode"))
            group_controller = controller.groups[-100900]
            self.assertEqual(group_controller._output_mode(), "answers")
            picker = group_controller.mode_picker
            self.assertIsNotNone(picker)
            await controller._handle_update(
                self._group_callback_update(
                    17,
                    -100900,
                    303,
                    picker.message_id,  # type: ignore[union-attr]
                    f"om:{picker.token}:detailed",  # type: ignore[union-attr]
                )
            )
            self.assertEqual(group_controller._output_mode(), "answers")
            await controller._handle_update(
                self._group_callback_update(
                    18,
                    -100900,
                    202,
                    picker.message_id,  # type: ignore[union-attr]
                    f"om:{picker.token}:detailed",  # type: ignore[union-attr]
                )
            )
            self.assertEqual(group_controller._output_mode(), "detailed")

            sessions[-100900].working_directory = Path(temp_dir)
            await controller._handle_media_group(
                [
                    self._group_attachment_update(19, -100900, 101, "album-1", "one.txt"),
                    self._group_attachment_update(
                        20,
                        -100900,
                        101,
                        "album-2",
                        "two.txt",
                        caption=f"{HELPER_TRIGGER_CYRILLIC.upper()}, inspect these files",
                    ),
                ]
            )
            self.assertEqual(len(sessions[-100900].messages), 5)
            self.assertIn("one.txt", sessions[-100900].messages[-1])
            self.assertIn("two.txt", sessions[-100900].messages[-1])

            await controller._handle_update(self._group_message_update(21, -100900, 101, "/group_remove"))
            self.assertFalse(state_store.is_group_attached(-100900))
            self.assertTrue(sessions[-100900].closed)
            await controller._handle_update(self._group_message_update(22, -100900, 202, "after detach"))
            self.assertEqual(len(sessions[-100900].messages), 5)
            await controller._handle_update(self._group_message_update(23, -100900, 202, "/group_add"))
            self.assertEqual(state_store.pending_group_setup_user(-100900), 202)
            await controller._handle_update(self._group_message_update(24, -100900, 101, "/group_remove"))
            self.assertIsNone(state_store.pending_group_setup_user(-100900))
            self.assertFalse(state_store.is_group_attached(-100900))
            await controller.close()

    async def test_group_all_messages_mode_is_opt_in_and_keeps_the_group_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "state.sqlite3"
            state_store = RuntimeStateStore(database)
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101, 202),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=database,
            )
            telegram = FakeTelegram()
            sessions: dict[int, FakeCodex] = {}

            def make_session(context_id: int, _controller: Any) -> FakeCodex:
                codex = FakeCodex()
                sessions[context_id] = codex
                return codex

            controller = TelegramController(
                config,
                telegram,
                object(),
                make_session,
                state_store=state_store,
            )  # type: ignore[arg-type]
            await controller._handle_update(
                self._group_message_update(1, -100920, 101, "/group_add", "Work chat")
            )
            await controller._handle_update(
                self._group_message_update(2, -100920, 101, CODEX_TRIGGER_CYRILLIC)
            )

            self.assertFalse(state_store.group_all_messages_to_codex(-100920))
            await controller._handle_update(
                self._group_message_update(3, -100920, 101, "small talk stays in Telegram")
            )
            self.assertEqual(sessions[-100920].messages, [])

            await controller._handle_update(
                self._group_message_update(4, -100920, 101, "/group_mode")
            )
            group_controller = controller.groups[-100920]
            picker = group_controller.group_mode_picker
            self.assertIsNotNone(picker)
            assert picker is not None and picker.message_id is not None
            mode_markup = telegram.reply_markups[-1]
            all_callback = mode_markup["inline_keyboard"][1][0]["callback_data"]

            await controller._handle_update(
                self._group_callback_update(5, -100920, 303, picker.message_id, all_callback)
            )
            self.assertFalse(state_store.group_all_messages_to_codex(-100920))
            await controller._handle_update(
                self._group_callback_update(6, -100920, 202, picker.message_id, all_callback)
            )
            self.assertTrue(state_store.group_all_messages_to_codex(-100920))
            self.assertTrue(RuntimeStateStore(database).group_all_messages_to_codex(-100920))

            await controller._handle_update(
                self._group_message_update(7, -100920, 101, "first ordinary task")
            )
            await controller._handle_update(
                self._group_message_update(8, -100920, 202, "second ordinary task")
            )
            await controller._handle_update(
                self._group_message_update(9, -100920, 303, "not on the allowlist")
            )
            await controller._handle_update(
                self._group_message_update(10, -100920, 101, f"{CODEX_TRIGGER_CYRILLIC}, stripped trigger")
            )
            await controller._handle_update(
                self._group_message_update(11, -100920, 101, "/status")
            )

            self.assertEqual(
                sessions[-100920].messages,
                ["first ordinary task", "second ordinary task", "stripped trigger"],
            )
            self.assertIn("Codex: ready", telegram.messages[-1][1])

            reply_callback = telegram.edit_markups[-1]
            stop_all_callback = reply_callback["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_update(
                self._group_callback_update(12, -100920, 101, picker.message_id, stop_all_callback)
            )
            self.assertFalse(state_store.group_all_messages_to_codex(-100920))
            await controller._handle_update(
                self._group_message_update(13, -100920, 101, "ordinary message again ignored")
            )
            await controller._handle_update(
                self._group_message_update(14, -100920, 101, f"{CODEX_TRIGGER_CYRILLIC} trigger still works")
            )
            self.assertEqual(
                sessions[-100920].messages,
                [
                    "first ordinary task",
                    "second ordinary task",
                    "stripped trigger",
                    "trigger still works",
                ],
            )
            await controller.close()

    async def test_group_all_messages_mode_forwards_media_without_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "state.sqlite3"
            state_store = RuntimeStateStore(database)
            state_store.attach_group(-100921, "Work media", ("codex",))
            self.assertTrue(state_store.set_group_all_messages_to_codex(-100921, True))
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101,),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=database,
            )
            telegram = FakeTelegram()
            codex = FakeCodex()
            codex.working_directory = Path(temp_dir)
            user_controller = TelegramUserController(
                config,
                telegram,
                object(),
                -100921,
                is_group=True,
                group_trigger_words=lambda: state_store.group_trigger_words(-100921),
                group_all_messages_to_codex=lambda: state_store.group_all_messages_to_codex(-100921),
            )  # type: ignore[arg-type]
            user_controller.attached = True
            user_controller.bind_codex(codex)  # type: ignore[arg-type]

            await user_controller._handle_media_group(
                [
                    self._group_attachment_update(1, -100921, 101, "media-1", "one.txt"),
                    self._group_attachment_update(2, -100921, 101, "media-2", "two.txt"),
                ]
            )

            self.assertEqual(len(codex.messages), 1)
            self.assertIn("one.txt", codex.messages[0])
            self.assertIn("two.txt", codex.messages[0])
            await user_controller._finish_typing()

    async def test_group_access_is_limited_to_global_operators(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            state_store.attach_group(-100910, "Operator group", ("codex",))
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
            sessions: dict[int, FakeCodex] = {}

            def make_session(context_id: int, _controller: Any) -> FakeCodex:
                session = FakeCodex()
                sessions[context_id] = session
                return session

            controller = TelegramController(
                config,
                telegram,
                object(),
                make_session,
                state_store=state_store,
            )  # type: ignore[arg-type]

            await controller._handle_update(
                self._group_message_update(1, -100910, 101, "codex allowed task")
            )
            await controller._handle_update(
                self._group_message_update(2, -100910, 303, "codex ignored task")
            )
            await controller._handle_update(
                self._group_message_update(3, -100910, 303, "/add_user 303")
            )
            await controller._handle_update(self._message_update(4, 303, "private attempt"))
            self.assertEqual(sessions[-100910].messages, ["allowed task"])
            self.assertEqual(set(sessions), {-100910})
            self.assertFalse(state_store.is_group_attached(-100911))
            await controller.close()

    async def test_start_restores_attached_group_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            state_store = RuntimeStateStore(path)
            state_store.attach_group(-100901, "Persistent group")
            state_store.for_group(-100901).save(
                RuntimeState(thread_id="persisted-group-thread", output_mode="detailed")
            )
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101,),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=path,
            )
            telegram = FakeTelegram()
            sessions: dict[int, FakeCodex] = {}

            def make_session(context_id: int, _controller: Any) -> FakeCodex:
                codex = FakeCodex()
                codex.state = (
                    state_store.for_group(context_id).load()
                    if context_id < 0
                    else state_store.for_user(context_id).load()
                )
                sessions[context_id] = codex
                return codex

            controller = TelegramController(
                config,
                telegram,
                object(),
                make_session,
                state_store=state_store,
            )  # type: ignore[arg-type]
            await controller.start()

            self.assertIn(-100901, controller.groups)
            self.assertTrue(sessions[-100901].started)
            self.assertEqual(sessions[-100901].state.thread_id, "persisted-group-thread")
            self.assertEqual(sessions[-100901].state.output_mode, "detailed")
            await controller.close()

    @staticmethod
    def _message_update(update_id: int, user_id: int, text: str) -> dict[str, Any]:
        return {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "from": {"id": user_id},
                "chat": {"id": user_id, "type": "private"},
                "text": text,
            },
        }

    @staticmethod
    def _callback_update(update_id: int, user_id: int, token: str, choice: str) -> dict[str, Any]:
        return {
            "update_id": update_id,
            "callback_query": {
                "id": f"callback-{update_id}",
                "from": {"id": user_id},
                "message": {
                    "message_id": 9,
                    "chat": {"id": user_id, "type": "private"},
                },
                "data": f"ap:{token}:{choice}",
            },
        }

    @staticmethod
    def _group_message_update(
        update_id: int,
        chat_id: int,
        user_id: int,
        text: str,
        title: str = "Dev team",
    ) -> dict[str, Any]:
        return {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "from": {"id": user_id},
                "chat": {"id": chat_id, "type": "supergroup", "title": title},
                "text": text,
            },
        }

    @staticmethod
    def _group_callback_update(
        update_id: int,
        chat_id: int,
        user_id: int,
        message_id: int,
        data: str,
    ) -> dict[str, Any]:
        return {
            "update_id": update_id,
            "callback_query": {
                "id": f"callback-{update_id}",
                "from": {"id": user_id},
                "message": {
                    "message_id": message_id,
                    "chat": {"id": chat_id, "type": "supergroup"},
                },
                "data": data,
            },
        }

    @staticmethod
    def _group_attachment_update(
        update_id: int,
        chat_id: int,
        user_id: int,
        file_id: str,
        filename: str,
        caption: str = "",
    ) -> dict[str, Any]:
        update = MultiUserRoutingTests._group_message_update(
            update_id,
            chat_id,
            user_id,
            "",
        )
        message = update["message"]
        message.pop("text")
        message["media_group_id"] = "test-album"
        message["document"] = {
            "file_id": file_id,
            "file_size": 12,
            "file_name": filename,
            "mime_type": "text/plain",
        }
        if caption:
            message["caption"] = caption
        return update


if __name__ == "__main__":
    unittest.main()
