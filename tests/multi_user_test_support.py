from __future__ import annotations

import asyncio
import html
import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from aiogram import Bot
from aiogram.types import Update

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
from telecodex.telegram.routers import TelegramUpdateRouters
from telecodex.telegram.updates import (
    TelegramCallbackContext,
    TelegramMessageContext,
    parse_aiogram_update,
    parse_update,
)
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
from telecodex.telegram.formatting import TELEGRAM_TEXT_LIMIT, split_plain_text
from telecodex.telegram.pagination import navigation_row, page_count, two_column_rows

_TEXT_CATALOG = TextCatalog()


def message_event(
    update_id: int,
    message_id: int,
    user_id: int,
    chat_id: int,
    chat_type: str,
    text: str,
    *,
    attachments: tuple[TelegramAttachment, ...] = (),
    media_group_id: str | None = None,
    group_trigger_checked: bool = False,
    group_trigger_required: bool = False,
    reply_to_user_id: int | None = None,
    model_text: str | None = None,
    entities: list[dict[str, Any]] | None = None,
):
    raw_message: dict[str, Any] = {
        "message_id": message_id,
        "date": 1_700_000_000,
        "from": {"id": user_id, "is_bot": False, "first_name": "Test"},
        "chat": {"id": chat_id, "type": chat_type},
        "text": text,
    }
    if entities:
        raw_message["entities"] = entities
    if media_group_id is not None:
        raw_message["media_group_id"] = media_group_id
    if reply_to_user_id is not None:
        raw_message["reply_to_message"] = {
            "message_id": message_id - 1,
            "date": 1_700_000_000,
            "from": {"id": reply_to_user_id, "is_bot": False, "first_name": "Reply"},
            "chat": {"id": chat_id, "type": chat_type},
            "text": "reply",
        }
    for attachment in attachments:
        media = {
            "file_id": attachment.file_id,
            "file_unique_id": f"unique-{attachment.file_id}",
            "file_size": attachment.file_size,
        }
        if attachment.file_name is not None:
            media["file_name"] = attachment.file_name
        if attachment.mime_type is not None:
            media["mime_type"] = attachment.mime_type
        if attachment.duration_seconds is not None:
            media["duration"] = attachment.duration_seconds
        if attachment.kind == "voice":
            raw_message["voice"] = media
        elif attachment.kind == "audio":
            raw_message["audio"] = media
        elif attachment.kind == "photo":
            raw_message["photo"] = [media]
        else:
            raw_message["document"] = media
    update = Update.model_validate({"update_id": update_id, "message": raw_message})
    message = parse_aiogram_update(update, model_text=model_text)
    if message is None:
        raise ValueError("Test update did not produce a Telegram message context")
    return replace(
        message,
        attachments=attachments,
        group_trigger_checked=group_trigger_checked,
        group_trigger_required=group_trigger_required,
    )


def callback_event(
    update_id: int,
    callback_id: str,
    user_id: int,
    chat_id: int,
    message_id: int,
    data: str,
    *,
    chat_type: str = "private",
):
    update = Update.model_validate(
        {
            "update_id": update_id,
            "callback_query": {
                "id": callback_id,
                "from": {"id": user_id, "is_bot": False, "first_name": "Test"},
                "chat_instance": "test-chat-instance",
                "message": {
                    "message_id": message_id,
                    "date": 1_700_000_000,
                    "chat": {"id": chat_id, "type": chat_type},
                    "text": "test callback",
                },
                "data": data,
            },
        }
    )
    callback = parse_aiogram_update(update)
    if callback is None:
        raise ValueError("Test update did not produce a Telegram callback context")
    return callback


def _catalog_text(key: str, **values: Any) -> str:
    return _TEXT_CATALOG.render("ru", key, **values)


_HTML_FORMAT_TAG = re.compile(r"</?(?:strong|code|pre)>")
_TEST_BOT_TOKEN = "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"


class _NoopMediaGroups:
    async def flush_before(self, _update: dict[str, Any]) -> None:
        return None

    async def add(self, _update: dict[str, Any]) -> bool:
        return True


def _typed_update(context: TelegramMessageContext | TelegramCallbackContext) -> Update:
    if isinstance(context, TelegramMessageContext):
        return Update.model_validate(
            {
                "update_id": context.update_id,
                "message": context.event.model_dump(mode="json", by_alias=True, exclude_none=True),
            }
        )
    return Update.model_validate(
        {
            "update_id": context.update_id,
            "callback_query": context.event.model_dump(mode="json", by_alias=True, exclude_none=True),
        }
    )


async def dispatch_controller_update(
    controller: TelegramController,
    update: dict[str, Any],
) -> None:
    context = parse_update(update)
    if context is None:
        return
    typed_update = _typed_update(context)
    bot = Bot(_TEST_BOT_TOKEN)
    try:
        routes = TelegramUpdateRouters(controller, _NoopMediaGroups())  # type: ignore[arg-type]
        await routes.dispatcher.feed_update(
            bot,
            typed_update,
            telecodex_update=typed_update,
        )
    finally:
        await bot.session.close()


class _UserControllerRouterAdapter:
    def __init__(self, controller: TelegramUserController) -> None:
        self.user_controller = controller
        self.config = controller.config
        self.state_store = type(
            "TestGroupState",
            (),
            {"is_group_attached": lambda _self, _chat_id: controller.is_group and controller.attached},
        )()

    def can_user_access_context(self, chat_id: int, user_id: int) -> bool:
        return (
            self.user_controller._is_authorized_user(user_id)
            and (chat_id == user_id or (chat_id < 0 and self.user_controller.is_group and self.user_controller.attached))
        )

    async def _prepare_incoming(
        self,
        context: TelegramMessageContext | TelegramCallbackContext,
    ):
        prepared = await self.user_controller._prepare_incoming(context)
        if prepared is None:
            return None
        return self.user_controller, prepared


async def _feed_aiogram_context(controller: Any, context: TelegramMessageContext | TelegramCallbackContext) -> None:
    bot = Bot(_TEST_BOT_TOKEN)
    try:
        routes = TelegramUpdateRouters(controller, _NoopMediaGroups())  # type: ignore[arg-type]
        update = _typed_update(context)
        await routes.dispatcher.feed_update(bot, update, telecodex_update=update)
    finally:
        await bot.session.close()


async def dispatch_controller_update(
    controller: TelegramController,
    update: dict[str, Any],
) -> None:
    context = parse_update(update)
    if context is not None:
        await _feed_aiogram_context(controller, context)


async def dispatch_user_update(
    controller: TelegramUserController,
    context: TelegramMessageContext | TelegramCallbackContext,
) -> None:
    await _feed_aiogram_context(_UserControllerRouterAdapter(controller), context)


async def dispatch_user_update(
    controller: TelegramUserController,
    context: TelegramMessageContext | TelegramCallbackContext,
) -> None:
    typed_update = _typed_update(context)
    bot = Bot(_TEST_BOT_TOKEN)
    try:
        routes = TelegramUpdateRouters(
            _UserControllerRouterAdapter(controller),  # type: ignore[arg-type]
            _NoopMediaGroups(),  # type: ignore[arg-type]
        )
        await routes.dispatcher.feed_update(
            bot,
            typed_update,
            telecodex_update=typed_update,
        )
    finally:
        await bot.session.close()


def _plain_html(text: str) -> str:
    return html.unescape(_HTML_FORMAT_TAG.sub("", text))


CODEX_TRIGGER_CYRILLIC = "\u043a\u043e\u0434\u0435\u043a\u0441"
KAWAII_TRIGGER_CYRILLIC = "\u043a\u0430\u0432\u0430\u0438"
BOT_TRIGGER_CYRILLIC = "\u0431\u043e\u0442"
HELPER_TRIGGER_CYRILLIC = "\u043f\u043e\u043c\u043e\u0449\u043d\u0438\u043a"


class FakeSentMessage(dict[str, int]):
    @property
    def message_id(self) -> int:
        return self["message_id"]


class FakeTelegram:
    def __init__(self) -> None:
        self.bot_user_id = 999
        self.id = self.bot_user_id
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
        reply_markup: Any = None,
        reply_to_message_id: int | None = None,
        entities: list[dict[str, Any]] | None = None,
        disable_notification: bool = False,
    ) -> FakeSentMessage:
        self.messages.append((chat_id, text))
        self.message_entities.append(entities or [])
        self.reply_markups.append(reply_markup)
        self.replies.append(reply_to_message_id)
        self.notification_flags.append(disable_notification)
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
        reply_to_message_id = (
            reply_parameters.message_id
            if reply_parameters is not None
            else None
        )
        self.rich_messages.append((chat_id, markdown, disable_notification))
        return await self.send_message(
            chat_id,
            markdown,
            reply_to_message_id=reply_to_message_id,
            disable_notification=disable_notification,
        )

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.actions.append((chat_id, action))

    async def answer_callback_query(self, callback_query_id: str, text: str | None = None) -> None:
        self.callbacks.append((callback_query_id, text or ""))

    async def answer_callback(self, callback_id: str, text: str | None = None) -> None:
        self.callbacks.append((callback_id, text or ""))

    async def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: Any = None,
    ) -> None:
        self.edits.append((chat_id, message_id, text))
        self.edit_markups.append(reply_markup)
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
        document: Any,
        *,
        reply_to_message_id: int | None = None,
    ) -> FakeSentMessage:
        path = Path(document.path)
        filename = document.filename
        self.documents.append((chat_id, path, filename, reply_to_message_id))
        return FakeSentMessage(message_id=self.next_message_id)


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
        self.thread_model: str | None = None
        self.thread_effort: str | None = None
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

    @property
    def settings_manager(self) -> FakeCodex:
        return self

    @property
    def pending_skill_name(self) -> str | None:
        return self.state.pending_skill_name

    @property
    def pending_skill_path(self) -> str | None:
        return self.state.pending_skill_path

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

    async def current_model_settings(
        self,
        models: list[CodexModel],
    ) -> tuple[str | None, str | None]:
        model = (
            self.thread_model
            if self.state.thread_id
            else None
        ) or self.state.selected_model or self.default_model(models)
        effort = (
            self.thread_effort
            if self.state.thread_id
            else None
        ) or self.state.selected_effort
        return model, effort

    async def select_model(self, model: str | None, effort: str | None = None) -> None:
        self.state.selected_model = model
        self.state.selected_effort = effort
        if self.state.thread_id:
            self.thread_model = model
            self.thread_effort = effort

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
