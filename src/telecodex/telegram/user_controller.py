from __future__ import annotations

import asyncio
import difflib
import html
import logging
import re
import secrets
import tempfile
import uuid
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from telecodex.codex.app_server import CodexAuthenticationError
from telecodex.codex.permissions import PERMISSION_MODES
from telecodex.codex.command_prompts import RECAP_PROMPT, init_prompt
from telecodex.codex.rpc import RpcError
from telecodex.telegram.mcp import McpTelegramMixin, PendingMcpElicitation
from telecodex.codex.session import (
    ActiveThreadWriterError,
    CollaborationModeUnavailableError,
    CodexChatBusyError,
    CodexBackgroundTerminal,
    CodexRecoverySnapshot,
    GoalNotFoundError,
    GoalUnavailableError,
    FastTierUnavailableError,
    MemoryFeatureUnavailableError,
    PendingTurnDeliveryError,
    SelectedSkillUnavailableError,
)
from telecodex.config import AppConfig
from telecodex.doctor import DoctorReport, collect_doctor_report
from telecodex.documents.models import PreparedAttachment
from telecodex.documents.processor import (
    AttachmentProcessingError,
    prepare_attachments,
)
from telecodex.audio.transcriber import AudioTranscriptionError, LocalAudioTranscriber
from telecodex.group_triggers import strip_group_trigger
from telecodex.i18n import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, TextCatalog, load_text_catalog
from telecodex.logging_config import (
    bind_log_context,
    isolated_log_context,
    log_exception,
    log_identifier,
    redact_text,
)
from telecodex.shell.runner import EmptyShellCommandError, ShellRunner
from telecodex.shell.session import ShellOutput, ShellSession
from telecodex.state import OUTPUT_MODE_CHOICES
from telecodex.telegram.activity import (
    explored_action_summaries,
    file_change_groups,
    format_activity_lines,
    image_saved_path,
    utf16_length,
    web_search_detail,
)
from telecodex.telegram.api import TelegramAPI, TelegramApiError
from telecodex.telegram.formatting import (
    TELEGRAM_TEXT_LIMIT,
    format_command,
    split_plain_text,
    truncate,
)
from telecodex.telegram.pagination import (
    INLINE_BUTTON_TEXT_LIMIT,
    PAGE_SIZE,
    keyboard,
    navigation_row,
    page_count,
    two_column_rows,
)
from telecodex.telegram.pickers import (
    ChatsPicker,
    DeleteConfirmation,
    DirectoryPicker,
    FilesPicker,
    GroupModePicker,
    LanguagePicker,
    MemoryPicker,
    McpAddWizard,
    McpPicker,
    ModelPicker,
    ModePicker,
    PermissionsPicker,
    ProcessesPicker,
    SkillPicker,
)
from telecodex.telegram.reply_format import format_long_text
from telecodex.telegram.updates import IncomingCallback, IncomingMessage, parse_command, parse_update
from telecodex.workspace_files import (
    WorkspaceEntry,
    WorkspaceFileBrowser,
    WorkspaceFileTooLarge,
    WorkspaceFilesError,
    WorkspacePreviewUnavailable,
)

if TYPE_CHECKING:
    from telecodex.codex.session import CodexModel, CodexSession, CodexThread

logger = logging.getLogger(__name__)
TYPING_ACTION_INTERVAL_SECONDS = 2.9
GROUP_ACTIVITY_INTERVAL_SECONDS = 3.1
GROUP_ACTIVITY_MESSAGE_LIMIT = 3900
GROUP_ACTIVITY_QUEUE_LIMIT = 300
_SENSITIVE_COMMAND_ARGUMENT = re.compile(
    r"(?i)(--?(?:password|passwd|token|secret|api[-_]?key|credential|auth)(?:=|\s+))"
    r"(\"[^\"]*\"|'[^']*'|[^\s]+)"
)
_TERMINAL_ESCAPE_SEQUENCE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))"
)
TELEGRAM_COMMAND_NAMES = (
    "start", "help", "doctor", "status", "permissions", "rename", "delete",
    "fork", "compact", "plan", "goal", "recap", "init", "ps", "new", "files",
    "chats", "stop", "recover", "model", "fast", "skills", "memories", "mcp", "language",
    "mode", "shell", "exit", "group_add", "group_remove", "group_mode",
)
# Keep chat tools and next-turn settings usable during a Codex turn. Deletion
# remains gated; thread switches use confirmation and /fork waits for idle.
BUSY_ALLOWED_COMMANDS = frozenset({
    "start", "help", "doctor", "status", "permissions", "rename", "fork",
    "compact", "ps", "new", "chats", "files", "stop", "recover", "model",
    "fast", "skills", "memories", "mcp", "language", "mode", "shell", "plan",
    "goal", "recap", "init", "exit", "group_mode",
})
BUSY_ALLOWED_CALLBACKS = frozenset({
    "ap", "me", "ps", "rc", "rr", "l", "om", "sk", "mm", "m", "np",
    "fb", "cp", "pm", "mc", "sw",
    "gm",
})


def _clean_terminal_output(text: str) -> str:
    text = _TERMINAL_ESCAPE_SEQUENCE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        character
        for character in text
        if character in "\n\t" or (ord(character) >= 32 and not 0x7F <= ord(character) <= 0x9F)
    )


@dataclass
class PendingApproval:
    future: asyncio.Future[str]
    message_id: int
    chat_id: int


@dataclass
class PendingUserInput:
    future: asyncio.Future[dict[str, Any]]
    questions: list[dict[str, Any]]


@dataclass(frozen=True)
class RunningCommand:
    message_id: int | None
    started_at: float
    command: str


@dataclass(frozen=True)
class PendingGoalEdit:
    thread_id: str


@dataclass(frozen=True)
class PendingThreadSwitch:
    token: str
    message_id: int
    action: str
    path: Path | None = None
    thread_id: str | None = None
    label: str | None = None
    picker_message_id: int | None = None
    picker_token: str | None = None
    picker_index: int | None = None
    source_thread_id: str | None = None


@dataclass(frozen=True)
class PendingStaleTurnAction:
    message_id: int
    thread_id: str
    turn_id: str | None


@dataclass
class PendingChatRecovery:
    token: str
    message_id: int
    thread_id: str
    stage: str
    terminal_ids: tuple[str, ...] = ()
    confirm_interrupt: bool = False


@dataclass
class GroupActivityItem:
    text: str
    entities: list[dict[str, Any]]
    reply_to_message_id: int | None
    batchable: bool = True
    delivery: asyncio.Future[dict[str, Any]] | None = None
    disable_notification: bool = False
    rich_markdown: bool = False


@dataclass(slots=True)
class QueuedCompactionInput:
    chat_id: int
    text: str
    prompt_message_id: int
    additional_input_items: list[dict[str, Any]] | None = None


class TelegramUserController(McpTelegramMixin):
    """Handle one user's messages and translate their Codex events into Telegram UI."""

    def __init__(
        self,
        config: AppConfig,
        telegram: TelegramAPI,
        shell: ShellRunner,
        user_id: int,
        texts: TextCatalog | None = None,
        *,
        is_group: bool = False,
        group_trigger_words: Callable[[], tuple[str, ...]] | None = None,
        group_all_messages_to_codex: Callable[[], bool] | None = None,
        set_group_all_messages_to_codex: Callable[[bool], bool] | None = None,
        audio_transcriber: LocalAudioTranscriber | None = None,
        authorize_user: Callable[[int], bool] | None = None,
        poll_health: Callable[[], dict[str, int | None]] | None = None,
    ) -> None:
        self.config = config
        self.telegram = telegram
        self.shell = shell
        self.shell_session: ShellSession | None = None
        self.user_id = user_id
        self.chat_id = user_id
        self.is_group = is_group
        self._group_trigger_words = group_trigger_words or (lambda: ())
        self._group_all_messages_to_codex = group_all_messages_to_codex or (lambda: False)
        self._set_group_all_messages_to_codex = set_group_all_messages_to_codex
        self._authorize_user = authorize_user
        self._poll_health = poll_health
        self.attached = True
        self.texts = texts or load_text_catalog()
        self.audio_transcriber = audio_transcriber
        self.codex: CodexSession | None = None
        self.pending_approvals: dict[str, PendingApproval] = {}
        self.pending_user_input: PendingUserInput | None = None
        self._codex_activity_lock = asyncio.Lock()
        self._typing_task: asyncio.Task[None] | None = None
        self._typing_enabled = False
        self._typing_refresh = asyncio.Event()
        self._command_messages: dict[str, RunningCommand] = {}
        self.pending_thread_switch: PendingThreadSwitch | None = None
        self.pending_stale_turn_actions: dict[str, PendingStaleTurnAction] = {}
        self.pending_chat_recovery: PendingChatRecovery | None = None
        self._chat_recovery_in_progress = False
        self._recovery_ready = asyncio.Event()
        self._recovery_ready.set()
        self._fork_after_turn_task: asyncio.Task[None] | None = None
        self._fork_after_turn_thread_id: str | None = None
        self.pending_goal_edit: PendingGoalEdit | None = None
        self._turn_prompt_message_id: int | None = None
        self._shutdown_in_progress = False
        self.model_picker: ModelPicker | None = None
        self.language_picker: LanguagePicker | None = None
        self.mode_picker: ModePicker | None = None
        self.group_mode_picker: GroupModePicker | None = None
        self.skill_picker: SkillPicker | None = None
        self.memory_picker: MemoryPicker | None = None
        self.directory_picker: DirectoryPicker | None = None
        self.files_picker: FilesPicker | None = None
        self.chats_picker: ChatsPicker | None = None
        self.permissions_picker: PermissionsPicker | None = None
        self.delete_confirmation: DeleteConfirmation | None = None
        self.processes_picker: ProcessesPicker | None = None
        self.mcp_picker: McpPicker | None = None
        self.mcp_add_wizard: McpAddWizard | None = None
        self.pending_mcp_elicitations: dict[str, PendingMcpElicitation] = {}
        self._mcp_form_token: str | None = None
        self._reload_mcp_config: Callable[[], Awaitable[tuple[int, int]]] | None = None
        self._group_activity_items: list[GroupActivityItem] = []
        self._group_activity_event = asyncio.Event()
        self._group_activity_task: asyncio.Task[None] | None = None
        self._group_activity_last_sent = 0.0
        self._group_activity_dropped = 0
        self._compaction_input_queue: deque[QueuedCompactionInput] = deque()
        self._compaction_input_queue_task: asyncio.Task[None] | None = None

    def bind_codex(self, codex: CodexSession) -> None:
        self.codex = codex

    def bind_mcp_reloader(
        self,
        callback: Callable[[], Awaitable[tuple[int, int]]],
    ) -> None:
        self._reload_mcp_config = callback

    def _codex(self) -> CodexSession:
        if self.codex is None:
            raise RuntimeError("Codex session has not been bound")
        return self.codex

    def _log_context_fields(self) -> dict[str, str | None]:
        fields: dict[str, str | None] = {
            "context_kind": "group" if self.is_group else "private",
            "context_hash": log_identifier(self.chat_id, namespace="telegram-chat"),
            "actor_hash": log_identifier(self.user_id, namespace="telegram-user"),
        }
        if self.codex is not None and self.codex.state.thread_id:
            fields["thread_hash"] = log_identifier(
                self.codex.state.thread_id, namespace="codex-thread"
            )
        return fields


    def _codex_model_log_fields(self) -> dict[str, str]:
        codex = self.codex
        state = getattr(codex, "state", None)
        model_resolver = getattr(codex, "_effective_model", None)
        model = model_resolver() if callable(model_resolver) else None
        model = (
            model
            or getattr(state, "selected_model", None)
            or self.config.codex_model
            or "default"
        )
        effort = getattr(state, "selected_effort", None) or "default"
        return {"model": str(model), "effort": str(effort)}

    def _log_failure(
        self,
        message: str,
        exc: BaseException,
        *,
        event: str,
        level: int = logging.ERROR,
        **fields: Any,
    ) -> None:
        context = self._log_context_fields()
        context.update(fields)
        if isinstance(exc, RpcError) and isinstance(exc.code, (str, int)):
            context.setdefault("rpc_code", exc.code)
        if isinstance(exc, TelegramApiError):
            if exc.method:
                context.setdefault("operation", exc.method)
            if exc.error_code is not None:
                context.setdefault("error_code", exc.error_code)
            if exc.http_status is not None:
                context.setdefault("http_status", exc.http_status)
        log_exception(
            logger,
            message,
            exc,
            event=event,
            level=level,
            **context,
        )

    def _language(self) -> str:
        if self.codex is None:
            return DEFAULT_LANGUAGE
        return getattr(self.codex, "language", DEFAULT_LANGUAGE)

    def _is_authorized_user(self, user_id: int) -> bool:
        if self._authorize_user is not None:
            return self._authorize_user(user_id)
        return user_id in self.config.telegram_allowed_user_ids

    def _output_mode(self) -> str:
        if self.codex is None:
            return "detailed"
        return getattr(self.codex.state, "output_mode", "detailed")

    def _selected_output_mode(self) -> str:
        if self._output_mode() == "answers" and getattr(
            self._codex().state,
            "quiet_progress",
            False,
        ):
            return "quiet_answers"
        return self._output_mode()

    def _quiet_progress_notifications(self) -> bool:
        return self._output_mode() == "answers" and getattr(
            self._codex().state,
            "quiet_progress",
            False,
        )

    def _shows_detailed_activity(self) -> bool:
        return self._output_mode() == "detailed"

    def _text(self, key: str, **values: Any) -> str:
        return self.texts.render(self._language(), key, **values)

    def _selected_button_label(self, label: str, selected: bool) -> str:
        ellipsis = self.texts.system("values.ellipsis")
        if not selected:
            if len(label) > INLINE_BUTTON_TEXT_LIMIT:
                return truncate(label, INLINE_BUTTON_TEXT_LIMIT, ellipsis)
            return label
        marker = self.texts.system("values.selected_label", selected="")
        max_label_length = INLINE_BUTTON_TEXT_LIMIT - len(marker)
        if len(label) > max_label_length:
            label = truncate(label, max_label_length, ellipsis)
        return self.texts.system("values.selected_label", selected=label)

    def _start_typing(self) -> None:
        self._typing_enabled = True
        self._resume_typing()

    def _resume_typing(self) -> None:
        if not self._typing_enabled:
            return
        if self._typing_task is None or self._typing_task.done():
            with isolated_log_context(
                correlation_id=f"typing-{uuid.uuid4().hex[:10]}",
                **self._log_context_fields(),
            ):
                self._typing_task = asyncio.create_task(self._typing_loop())

    def _notify_typing_cleared(self) -> None:
        if self._typing_enabled:
            self._typing_refresh.set()

    async def _pause_typing(self) -> None:
        task = self._typing_task
        self._typing_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _finish_typing(self) -> None:
        self._typing_enabled = False
        await self._pause_typing()

    async def _queue_group_activity(
        self,
        text: str,
        entities: list[dict[str, Any]],
        reply_to_message_id: int | None,
        *,
        batchable: bool = True,
        wait_for_delivery: bool = False,
        disable_notification: bool = False,
        rich_markdown: bool = False,
    ) -> dict[str, Any]:
        if not self.attached:
            return {}
        if len(self._group_activity_items) >= GROUP_ACTIVITY_QUEUE_LIMIT:
            if batchable:
                self._group_activity_dropped += 1
                return {}
            removable = next(
                (
                    index
                    for index, item in enumerate(self._group_activity_items)
                    if item.batchable
                ),
                None,
            )
            if removable is not None:
                del self._group_activity_items[removable]
                self._group_activity_dropped += 1
        delivery = (
            asyncio.get_running_loop().create_future()
            if wait_for_delivery
            else None
        )
        self._group_activity_items.append(
            GroupActivityItem(
                text,
                entities,
                reply_to_message_id,
                batchable=batchable,
                delivery=delivery,
                disable_notification=disable_notification,
                rich_markdown=rich_markdown,
            )
        )
        self._group_activity_event.set()
        if self._group_activity_task is None or self._group_activity_task.done():
            with isolated_log_context(
                correlation_id=f"group-activity-{uuid.uuid4().hex[:10]}",
                **self._log_context_fields(),
            ):
                self._group_activity_task = asyncio.create_task(self._group_activity_loop())
        if delivery is not None:
            try:
                return await delivery
            except asyncio.CancelledError:
                if not delivery.done():
                    delivery.cancel()
                raise
        return {}

    async def _group_activity_loop(self) -> None:
        while self.attached:
            await self._group_activity_event.wait()
            self._group_activity_event.clear()
            while self.attached and self._group_activity_items:
                delay = GROUP_ACTIVITY_INTERVAL_SECONDS - (
                    monotonic() - self._group_activity_last_sent
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                (
                    text,
                    entities,
                    reply_to,
                    delivery,
                    disable_notification,
                    rich_markdown,
                ) = self._take_group_activity_batch()
                try:
                    if rich_markdown:
                        result = await self.telegram.send_rich_message(
                            self.chat_id,
                            text,
                            reply_to_message_id=reply_to,
                            disable_notification=disable_notification,
                        )
                    else:
                        notification_options = (
                            {"disable_notification": True} if disable_notification else {}
                        )
                        result = await self.telegram.send_message(
                            self.chat_id,
                            text,
                            reply_to_message_id=reply_to,
                            entities=entities,
                            **notification_options,
                        )
                    self._group_activity_last_sent = monotonic()
                    if delivery is not None and not delivery.done():
                        delivery.set_result(result)
                except asyncio.CancelledError:
                    if delivery is not None and not delivery.done():
                        delivery.set_exception(
                            RuntimeError("Group activity delivery was stopped")
                        )
                    raise
                except Exception as exc:
                    if delivery is not None:
                        if not delivery.done():
                            delivery.set_exception(exc)
                    else:
                        self._log_failure(
                            "Could not send group activity update",
                            exc,
                            event="telegram.group_activity.send_failed",
                            level=logging.WARNING,
                            bytes=utf16_length(text),
                        )
            if self._group_activity_dropped and len(self._group_activity_items) < GROUP_ACTIVITY_QUEUE_LIMIT:
                dropped = self._group_activity_dropped
                self._group_activity_dropped = 0
                self._group_activity_items.append(
                    GroupActivityItem(
                        self._text("groups.label.activity_skipped", count=dropped), [], None
                    )
                )
                self._group_activity_event.set()

    def _take_group_activity_batch(
        self,
    ) -> tuple[
        str,
        list[dict[str, Any]],
        int | None,
        asyncio.Future[dict[str, Any]] | None,
        bool,
        bool,
    ]:
        first = self._group_activity_items.pop(0)
        if not first.batchable:
            return (
                first.text,
                first.entities,
                first.reply_to_message_id,
                first.delivery,
                first.disable_notification,
                first.rich_markdown,
            )
        parts = [first.text]
        entities = [dict(entity) for entity in first.entities]
        combined_length = utf16_length(first.text)
        reply_to = first.reply_to_message_id
        while self._group_activity_items:
            next_item = self._group_activity_items[0]
            if not next_item.batchable:
                break
            separator_length = 1
            next_length = utf16_length(next_item.text)
            if combined_length + separator_length + next_length > GROUP_ACTIVITY_MESSAGE_LIMIT:
                break
            if next_item.reply_to_message_id != reply_to:
                break
            if next_item.disable_notification != first.disable_notification:
                break
            if next_item.rich_markdown != first.rich_markdown:
                break
            self._group_activity_items.pop(0)
            offset = combined_length + separator_length
            entities.extend(
                {**entity, "offset": entity["offset"] + offset}
                for entity in next_item.entities
            )
            parts.append(next_item.text)
            combined_length += separator_length + next_length
        return (
            "\n".join(parts),
            entities,
            reply_to,
            None,
            first.disable_notification,
            first.rich_markdown,
        )

    async def _stop_group_activity(self) -> None:
        task = self._group_activity_task
        self._group_activity_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for item in self._group_activity_items:
            if item.delivery is not None and not item.delivery.done():
                item.delivery.set_exception(
                    RuntimeError("Group activity delivery was stopped")
                )
        self._group_activity_items.clear()
        self._group_activity_dropped = 0

    async def _typing_loop(self) -> None:
        warned = False
        while self._typing_enabled:
            started_at = monotonic()
            self._typing_refresh.clear()
            try:
                if not self.is_group or self.attached:
                    await self.telegram.send_chat_action(self.chat_id, "typing")
                warned = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not warned:
                    self._log_failure(
                        "Telegram typing action failed",
                        exc,
                        event="telegram.chat_action.typing_failed",
                        level=logging.WARNING,
                    )
                    warned = True
            delay = TYPING_ACTION_INTERVAL_SECONDS - (monotonic() - started_at)
            if delay > 0:
                try:
                    await asyncio.wait_for(self._typing_refresh.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass

    async def _handle_update(self, update: dict[str, Any]) -> None:
        incoming = parse_update(update)
        if incoming is None:
            logger.debug(
                "Ignored unsupported Telegram update shape",
                extra={
                    "event": "telegram.context.update_ignored",
                    "access_decision": "unrecognized",
                    "update_id": update.get("update_id") if isinstance(update.get("update_id"), int) else None,
                },
            )
            return
        accepted = (
            incoming.chat_id == self.chat_id
            and self._is_authorized_user(incoming.user_id)
            and (not self.is_group or self.attached)
        )
        if not accepted:
            logger.info(
                "Telegram update rejected by context access rules",
                extra={
                    "event": "telegram.context.update_rejected",
                    "access_decision": "denied_or_detached",
                    "context_kind": "group" if self.is_group else "private",
                    "context_hash": log_identifier(incoming.chat_id, namespace="telegram-chat"),
                    "actor_hash": log_identifier(incoming.user_id, namespace="telegram-user"),
                    "update_id": incoming.update_id,
                },
            )
            return
        if isinstance(incoming, IncomingCallback):
            if not self.is_group and incoming.chat_id != incoming.user_id:
                await self.telegram.answer_callback(
                    incoming.callback_id,
                    self._text("commands.label.open_private_chat"),
                )
                return
            logger.debug(
                "Telegram callback routed to context",
                extra={
                    "event": "telegram.context.callback_routed",
                    "context_kind": "group" if self.is_group else "private",
                    **self._log_context_fields(),
                },
            )
            await self._handle_callback(incoming)
            return
        if self.is_group:
            if incoming.chat_type not in {"group", "supergroup"}:
                return
        elif incoming.chat_type != "private" or incoming.chat_id != incoming.user_id:
            await self.telegram.send_message(incoming.chat_id, self._text("commands.label.private_only"))
            return
        await self._handle_message(incoming)

    async def _handle_media_group(self, updates: list[dict[str, Any]]) -> None:
        messages = [parse_update(update) for update in updates]
        incoming_messages = [item for item in messages if isinstance(item, IncomingMessage)]
        if not incoming_messages:
            return
        first = incoming_messages[0]
        if (
            first.chat_id != self.chat_id
            or not self._is_authorized_user(first.user_id)
            or (self.is_group and not self.attached)
            or (self.is_group and first.chat_type not in {"group", "supergroup"})
            or (not self.is_group and first.chat_id != first.user_id)
        ):
            return
        if any(
            item.user_id != first.user_id
            or item.chat_id != first.chat_id
            or not self._is_authorized_user(item.user_id)
            for item in incoming_messages
        ):
            logger.warning(
                "Ignoring inconsistent Telegram media group",
                extra={
                    "event": "telegram.media_group.inconsistent",
                    "context_kind": "group" if self.is_group else "private",
                    "context_hash": log_identifier(first.chat_id, namespace="telegram-chat"),
                    "item_count": len(incoming_messages),
                },
            )
            return
        attachments = tuple(
            attachment
            for item in incoming_messages
            for attachment in item.attachments
        )
        if not attachments:
            return
        captions: list[str] = []
        trigger_found = (
            self._is_reply_to_bot(first)
            or (self.is_group and self._group_all_messages_to_codex())
        )
        for item in incoming_messages:
            caption = item.text.strip()
            if not caption:
                continue
            if self.is_group:
                without_trigger = self._consume_group_trigger(caption)
                if without_trigger is not None:
                    trigger_found = True
                    caption = without_trigger
            captions.append(caption)
        has_audio = any(attachment.kind in {"voice", "audio"} for attachment in attachments)
        if self.is_group and not trigger_found and not has_audio:
            return
        captions = list(dict.fromkeys(captions))
        grouped = IncomingMessage(
            update_id=first.update_id,
            message_id=first.message_id,
            user_id=first.user_id,
            chat_id=first.chat_id,
            chat_type=first.chat_type,
            text="\n\n".join(captions),
            attachments=attachments,
            chat_title=first.chat_title,
            group_trigger_checked=self.is_group,
            group_trigger_required=self.is_group and not trigger_found,
            reply_to_user_id=first.reply_to_user_id,
        )
        await self._handle_message(grouped)

    async def _handle_message(self, incoming: IncomingMessage) -> None:
        parsed = None if incoming.attachments else parse_command(incoming.text)
        command_name = parsed[0] if parsed else "message"
        logger.info(
            "Telegram message routed to Codex context",
            extra={
                "event": "telegram.context.message_routed",
                "command_name": command_name,
                "update_id": incoming.update_id,
                "has_text": bool(incoming.text.strip()),
                "has_attachments": bool(incoming.attachments),
                "attachment_count": len(incoming.attachments),
                "attachment_kinds": sorted({item.kind for item in incoming.attachments}),
                "is_reply": incoming.reply_to_user_id is not None,
                "group_trigger_checked": incoming.group_trigger_checked,
                "group_trigger_required": incoming.group_trigger_required,
                **self._log_context_fields(),
            },
        )
        if parsed is not None and parsed[0] == "recover":
            await self._request_chat_recovery(incoming)
            return
        if self.shell_session is not None:
            if incoming.attachments:
                await self.telegram.send_message(
                    incoming.chat_id,
                    self._text("shell.label.attachments_unsupported"),
                    reply_to_message_id=incoming.message_id,
                )
                return
            if parsed is not None and parsed[0] == "exit":
                await self._exit_shell(incoming)
                return
            if parsed is not None and parsed[0] == "stop":
                await self._stop_shell(incoming)
                return
            if parsed is None:
                await self._send_shell_input(incoming)
                return
        command = parsed[0] if parsed else None
        if self._chat_recovery_in_progress:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("recovery.label.in_progress"),
                reply_to_message_id=incoming.message_id,
            )
            return
        if self.pending_chat_recovery is not None and command == "status":
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("recovery.label.confirm_pending"),
                reply_to_message_id=incoming.message_id,
            )
            return
        if self.pending_chat_recovery is not None and command not in {
            "status", "stop", "ps", "help", "start", "recover"
        }:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("recovery.label.confirm_pending"),
                reply_to_message_id=incoming.message_id,
            )
            return
        if getattr(self._codex(), "recovery_blocked", False) and command not in {
            "status", "stop", "ps", "help", "start", "doctor", "recover", "new", "chats"
        }:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("recovery.label.blocked"),
                reply_to_message_id=incoming.message_id,
            )
            return
        if await self._consume_mcp_dialog_input(incoming):
            return
        if await self._consume_goal_edit_input(incoming):
            return
        if parsed:
            command, argument = parsed
            busy_allowed = command in BUSY_ALLOWED_COMMANDS or (
                getattr(self._codex(), "recovery_blocked", False)
                and command in {"new", "chats"}
            )
            if (
                self._codex().is_busy
                and not busy_allowed
            ):
                logger.info(
                    "Blocked Telegram command while Codex is busy",
                    extra={
                        "event": "telegram.command.blocked_while_busy",
                        "command_name": command,
                        **self._log_context_fields(),
                    },
                )
                await self.telegram.send_message(
                    incoming.chat_id,
                    self._text("thread_commands.label.busy"),
                    reply_to_message_id=incoming.message_id,
                )
                return
            if command == "start" or command == "help":
                await self.telegram.send_message(incoming.chat_id, self._help_text())
            elif command == "doctor":
                if self.is_group and incoming.user_id not in self.config.telegram_allowed_user_ids:
                    await self.telegram.send_message(
                        incoming.chat_id,
                        self._text("doctor.label.group_owner_only"),
                    )
                    return
                try:
                    report = await collect_doctor_report(
                        self.telegram,
                        self._codex(),
                        self.config.telegram_proxy is not None,
                        self.audio_transcriber,
                        self._poll_health() if self._poll_health else None,
                    )
                except Exception as exc:
                    self._log_failure(
                        "Could not collect TeleCodex diagnostics",
                        exc,
                        event="telegram.command.doctor_failed",
                    )
                    await self.telegram.send_message(incoming.chat_id, self._text("doctor.label.failed"))
                else:
                    await self.telegram.send_message(incoming.chat_id, self._format_doctor_report(report))
            elif command == "status":
                if getattr(self._codex(), "recovery_blocked", False):
                    await self.telegram.send_message(
                        incoming.chat_id,
                        self._text("recovery.label.blocked"),
                    )
                    return
                try:
                    status = await self._codex().status_text(include_account=not self.is_group)
                    await self.telegram.send_message(incoming.chat_id, status)
                except Exception as exc:
                    self._log_failure(
                        "Could not read Codex status",
                        exc,
                        event="telegram.command.status_failed",
                    )
                    await self._send_error("errors.label.read_status")
            elif command == "permissions":
                await self._show_permissions_picker(incoming.chat_id)
            elif command == "rename":
                await self._rename_chat(incoming.chat_id, argument)
            elif command == "delete":
                await self._show_delete_confirmation(incoming.chat_id)
            elif command == "fork":
                await self._fork_chat(incoming.chat_id)
            elif command == "compact":
                await self._compact_chat(incoming.chat_id)
            elif command == "plan":
                await self._handle_plan_command(incoming, argument)
            elif command == "goal":
                await self._handle_goal_command(incoming, argument)
            elif command == "recap":
                await self._recap_chat(incoming)
            elif command == "init":
                await self._init_project(incoming)
            elif command == "ps":
                if getattr(self._codex(), "recovery_blocked", False):
                    await self.telegram.send_message(
                        incoming.chat_id,
                        self._text("recovery.label.blocked"),
                    )
                    return
                await self._show_processes(incoming.chat_id)
            elif command == "new":
                await self._show_directory_picker(incoming.chat_id)
            elif command == "files":
                await self._show_files_picker(incoming.chat_id, incoming.user_id)
            elif command == "chats":
                await self._show_chats_picker(incoming.chat_id)
            elif command == "stop":
                await self._stop_codex(incoming)
            elif command == "model":
                await self._show_model_picker(incoming.chat_id)
            elif command == "fast":
                await self._handle_fast_command(incoming, argument)
            elif command == "skills":
                await self._show_skills_picker(incoming.chat_id)
            elif command == "memories":
                await self._show_memory_picker(incoming.chat_id)
            elif command == "mcp":
                await self._show_mcp_picker(incoming.chat_id, argument, incoming.user_id)
            elif command == "language":
                await self._show_language_picker(incoming.chat_id)
            elif command == "mode":
                await self._show_mode_picker(incoming.chat_id)
            elif command == "group_mode":
                await self._show_group_mode_picker(incoming)
            elif command == "shell":
                if argument.strip():
                    await self._run_shell(incoming.chat_id, argument)
                else:
                    await self._enter_shell(incoming)
            elif command == "exit":
                await self.telegram.send_message(
                    incoming.chat_id,
                    self._text("shell.label.not_open"),
                    reply_to_message_id=incoming.message_id,
                )
            else:
                if self.shell_session is not None:
                    await self._send_shell_input(incoming)
                else:
                    await self.telegram.send_message(
                        incoming.chat_id,
                        self._unknown_command_text(command),
                    )
            return

        if self.is_group and not incoming.group_trigger_checked:
            if self._group_all_messages_to_codex():
                without_trigger = self._consume_group_trigger(incoming.text)
                if without_trigger is not None:
                    if not without_trigger and not incoming.attachments:
                        await self.telegram.send_message(
                            incoming.chat_id,
                            self._text("groups.label.trigger_empty"),
                            reply_to_message_id=incoming.message_id,
                        )
                        return
                    incoming = replace(incoming, text=without_trigger)
                incoming = replace(
                    incoming,
                    group_trigger_checked=True,
                    group_trigger_required=False,
                )
            elif self._is_reply_to_bot(incoming):
                incoming = replace(
                    incoming,
                    group_trigger_checked=True,
                    group_trigger_required=False,
                )
            else:
                without_trigger = self._consume_group_trigger(incoming.text)
                has_audio = any(attachment.kind in {"voice", "audio"} for attachment in incoming.attachments)
                if without_trigger is None and not has_audio:
                    return
                if without_trigger is None:
                    incoming = replace(
                        incoming,
                        group_trigger_checked=True,
                        group_trigger_required=True,
                    )
                elif not without_trigger and not incoming.attachments:
                    await self.telegram.send_message(
                        incoming.chat_id,
                        self._text("groups.label.trigger_empty"),
                        reply_to_message_id=incoming.message_id,
                    )
                    return
                else:
                    incoming = replace(
                        incoming,
                        text=without_trigger,
                        group_trigger_checked=True,
                        group_trigger_required=False,
                    )

        if self.pending_user_input is not None:
            audio_only = bool(incoming.attachments) and all(
                attachment.kind in {"voice", "audio"} for attachment in incoming.attachments
            )
            if incoming.attachments and not audio_only:
                await self.telegram.send_message(
                    incoming.chat_id,
                    self._text("documents.label.pending_user_input"),
                    reply_to_message_id=incoming.message_id,
                )
                return
            if audio_only:
                await self._send_attachments(incoming)
                return
            await self._resolve_user_input(incoming.text)
            return
        if incoming.attachments:
            await self._send_attachments(incoming)
            return
        await self._send_to_codex(incoming.chat_id, incoming.text, incoming.message_id)

    def _consume_group_trigger(self, text: str) -> str | None:
        return strip_group_trigger(text, self._group_trigger_words())

    def _is_reply_to_bot(self, incoming: IncomingMessage) -> bool:
        bot_user_id = getattr(self.telegram, "bot_user_id", None)
        return isinstance(bot_user_id, int) and incoming.reply_to_user_id == bot_user_id

    async def _send_attachments(self, incoming: IncomingMessage) -> None:
        await self._send_attachments_in_directory(
            incoming, self._codex().working_directory
        )

    async def _send_attachments_in_directory(
        self,
        incoming: IncomingMessage,
        attachment_working_directory: Path,
    ) -> None:
        audio_attachments = [
            attachment
            for attachment in incoming.attachments
            if attachment.kind in {"voice", "audio"}
        ]
        if any(
            attachment.duration_seconds is not None
            and attachment.duration_seconds > self.config.max_audio_duration_seconds
            for attachment in audio_attachments
        ):
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("audio.label.too_long", seconds=self.config.max_audio_duration_seconds),
                reply_to_message_id=incoming.message_id,
            )
            return

        self._start_typing()
        prepared: list[PreparedAttachment] = []
        try:
            prepared = await prepare_attachments(
                self.telegram,
                incoming.attachments,
                attachment_working_directory,
                self.chat_id,
            )
            audio_transcripts: list[str] = []
            audio_items = [item for item in prepared if item.is_audio]
            if audio_items and self.audio_transcriber is None:
                raise AudioTranscriptionError("audio.label.transcriber_missing")
            for source, item in zip(incoming.attachments, prepared, strict=True):
                if not item.is_audio:
                    continue
                assert self.audio_transcriber is not None
                audio_transcripts.append(
                    await self.audio_transcriber.transcribe(
                        Path(item.original_path),
                        duration_hint=source.duration_seconds,
                    )
                )
        except AttachmentProcessingError as exc:
            await self._finish_typing()
            await self.telegram.send_message(
                incoming.chat_id,
                self._text(exc.text_key, filename=exc.filename),
                reply_to_message_id=incoming.message_id,
            )
            return
        except AudioTranscriptionError as exc:
            await self._finish_typing()
            await self.telegram.send_message(
                incoming.chat_id,
                self._text(exc.text_key, seconds=self.config.max_audio_duration_seconds),
                reply_to_message_id=incoming.message_id,
            )
            return
        except Exception as exc:
            await self._finish_typing()
            self._log_failure(
                "Could not process Telegram attachment",
                exc,
                event="telegram.attachment.process_failed",
                attachment_count=len(incoming.attachments),
            )
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("documents.label.processing_failed"),
                reply_to_message_id=incoming.message_id,
            )
            return
        finally:
            for item in prepared:
                if item.is_audio:
                    Path(item.original_path).unlink(missing_ok=True)

        transcript_text = "\n".join(audio_transcripts) if audio_transcripts else ""
        prompt_parts = [incoming.text.strip(), transcript_text]
        prompt_parts = [part for part in prompt_parts if part]
        if incoming.group_trigger_required:
            trigger_found = False
            for index, part in enumerate(prompt_parts):
                without_trigger = self._consume_group_trigger(part)
                if without_trigger is not None:
                    prompt_parts[index] = without_trigger
                    trigger_found = True
                    break
            if not trigger_found:
                await self._finish_typing()
                return
            prompt_parts = [part for part in prompt_parts if part]
            if not prompt_parts and not any(not item.is_audio for item in prepared):
                await self._finish_typing()
                await self.telegram.send_message(
                    incoming.chat_id,
                    self._text("groups.label.trigger_empty"),
                    reply_to_message_id=incoming.message_id,
                )
                return

        if transcript_text:
            await self._send_audio_transcript(incoming.chat_id, incoming.message_id, transcript_text)

        if self.pending_user_input is not None:
            answer = "\n\n".join(prompt_parts)
            await self._finish_typing()
            await self._resolve_user_input(answer)
            return

        task = "\n\n".join(prompt_parts) or self._text("documents.values.default_prompt")
        references: list[str] = []
        input_items: list[dict[str, Any]] = []
        for attachment in prepared:
            if attachment.is_audio:
                continue
            original_path = attachment.original_path
            context_path = attachment.context_path
            if attachment.is_image:
                references.append(f"- {attachment.name}: image attached to this request")
                input_items.append({"type": "localImage", "path": original_path})
            elif context_path != original_path:
                references.append(
                    f"- {attachment.name}: original file {original_path}; "
                    f"extracted Markdown {context_path}"
                )
            else:
                references.append(f"- {attachment.name}: {original_path}")
        prompt = task
        if references:
            prompt += (
                f"\n\n{self._text('documents.values.paths_heading')}\n"
                + "\n".join(references)
                + f"\n\n{self._text('documents.values.read_instruction')}"
            )
        await self._send_to_codex(
            incoming.chat_id,
            prompt,
            incoming.message_id,
            additional_input_items=input_items,
        )

    async def _send_audio_transcript(self, chat_id: int, message_id: int, transcript: str) -> None:
        text = self._text("audio.label.transcript", transcript=transcript)
        for part in split_plain_text(text, limit=3900):
            try:
                await self.telegram.send_message(
                    chat_id,
                    part,
                    reply_to_message_id=message_id,
                )
            except TelegramApiError as exc:
                self._log_failure(
                    "Could not send recognized Telegram audio text",
                    exc,
                    event="telegram.audio.transcript_send_failed",
                    level=logging.WARNING,
                    text_part_bytes=len(part.encode("utf-8")),
                )
                return

    async def _send_to_codex(
        self,
        chat_id: int,
        text: str,
        prompt_message_id: int,
        *,
        additional_input_items: list[dict[str, Any]] | None = None,
        from_compaction_queue: bool = False,
    ) -> None:
        if not text.strip():
            return
        codex = self._codex()
        if self._chat_recovery_in_progress or self.pending_chat_recovery is not None:
            await self.telegram.send_message(
                chat_id,
                self._text("recovery.label.confirm_pending"),
                reply_to_message_id=prompt_message_id,
            )
            return
        if getattr(codex, "recovery_blocked", False):
            await self.telegram.send_message(
                chat_id,
                self._text("recovery.label.blocked"),
                reply_to_message_id=prompt_message_id,
            )
            return
        if not from_compaction_queue and (
            codex.compaction_barrier_pending
            or self._compaction_input_queue_task is not None
            or self._compaction_input_queue
        ):
            if len(self._compaction_input_queue) >= 20:
                await self.telegram.send_message(
                    chat_id,
                    self._text("thread_commands.label.compact_queue_full"),
                    reply_to_message_id=prompt_message_id,
                )
                return
            self._compaction_input_queue.append(
                QueuedCompactionInput(
                    chat_id=chat_id,
                    text=text,
                    prompt_message_id=prompt_message_id,
                    additional_input_items=(
                        [dict(item) for item in additional_input_items]
                        if additional_input_items
                        else None
                    ),
                )
            )
            position = len(self._compaction_input_queue)
            self._start_compaction_input_queue_worker()
            await self.telegram.send_message(
                chat_id,
                self._text("thread_commands.label.input_queued_after_compact", position=position),
                reply_to_message_id=prompt_message_id,
            )
            return
        if not codex.is_busy:
            self._turn_prompt_message_id = prompt_message_id
        self._start_typing()
        try:
            outcome = await codex.send_message(
                text,
                prompt_message_id,
                additional_input_items=additional_input_items,
            )
        except CodexChatBusyError:
            await self._finish_typing()
            await self.telegram.send_message(
                chat_id,
                self._text("thread_commands.label.busy"),
                reply_to_message_id=prompt_message_id,
            )
            return
        except SelectedSkillUnavailableError:
            await self._finish_typing()
            await self.telegram.send_message(chat_id, self._text("skills.label.selected_stale"))
            return
        except PendingTurnDeliveryError:
            logger.warning(
                "New Codex turn blocked by an undelivered result",
                extra={
                    "event": "codex.turn.start.blocked_by_pending_delivery",
                    **self._log_context_fields(),
                },
            )
            await self._finish_typing()
            await self._send_error("recovery.label.delivery_pending")
            return
        except CodexAuthenticationError:
            logger.warning(
                "Codex authentication is required",
                extra={
                    "event": "codex.auth.required",
                    **self._log_context_fields(),
                },
            )
            await self._on_codex_finished({"status": "failed", "text": self._text("errors.label.codex_auth")})
            return
        except RpcError as exc:
            detail = self._truncate(str(exc), 500)
            logger.error(
                "Codex turn request returned an RPC error",
                extra={
                    "event": "codex.turn.start.rpc_error",
                    "rpc_code": exc.code,
                    **self._log_context_fields(),
                },
            )
            await self._on_codex_finished(
                {
                    "status": "failed",
                    "text": self._text("errors.label.rpc_start", code=exc.code, detail=detail),
                }
            )
            return
        except Exception as exc:
            log_exception(
                logger,
                "Could not start Codex turn",
                exc,
                event="codex.turn.start.failed",
                **self._log_context_fields(),
            )
            await self._on_codex_finished(
                {"status": "failed", "text": self._text("errors.label.codex_start")}
            )
            return
        logger.info(
            "Codex accepted user input",
            extra={
                "event": "codex.turn.input.accepted",
                "outcome": outcome,
                **self._codex_model_log_fields(),
                **self._log_context_fields(),
            },
        )
        if outcome == "steered":
            self._turn_prompt_message_id = prompt_message_id
            if self._shows_detailed_activity():
                await self.telegram.send_message(
                    chat_id,
                    self._text("commands.label.steered"),
                    reply_to_message_id=prompt_message_id,
                )
            self._notify_typing_cleared()
        elif outcome == "started":
            self._turn_prompt_message_id = prompt_message_id

    def _start_compaction_input_queue_worker(self) -> None:
        task = self._compaction_input_queue_task
        if task is not None and not task.done():
            return
        self._compaction_input_queue_task = asyncio.create_task(
            self._drain_compaction_input_queue()
        )

    async def _drain_compaction_input_queue(self) -> None:
        current_task = asyncio.current_task()
        wait_failure_reported = False
        try:
            codex = self._codex()
            while self._compaction_input_queue and not self._shutdown_in_progress:
                await self._recovery_ready.wait()
                if self._chat_recovery_in_progress or self.pending_chat_recovery is not None:
                    continue
                try:
                    await codex.wait_for_manual_compaction_barrier()
                    await codex.wait_for_current_turn_completion()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if not wait_failure_reported:
                        log_exception(
                            logger,
                            "Could not advance queued Telegram input after Codex compaction",
                            exc,
                            event="telegram.compaction_queue.wait_failed",
                            level=logging.WARNING,
                            queued_count=len(self._compaction_input_queue),
                            **self._log_context_fields(),
                        )
                        await self.telegram.send_message(
                            self.chat_id,
                            self._text("thread_commands.label.compact_queue_wait_failed"),
                        )
                        wait_failure_reported = True
                    await asyncio.sleep(2)
                    continue
                wait_failure_reported = False
                if getattr(codex, "recovery_blocked", False):
                    await self._recovery_ready.wait()
                    continue
                queued = self._compaction_input_queue.popleft()
                await self._send_to_codex(
                    queued.chat_id,
                    queued.text,
                    queued.prompt_message_id,
                    additional_input_items=queued.additional_input_items,
                    from_compaction_queue=True,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception(
                logger,
                "Could not drain Telegram messages queued after Codex compaction",
                exc,
                event="telegram.compaction_queue.drain_failed",
                level=logging.WARNING,
                queued_count=len(self._compaction_input_queue),
                **self._log_context_fields(),
            )
            if self._compaction_input_queue and not self._shutdown_in_progress:
                await self.telegram.send_message(
                    self.chat_id,
                    self._text("thread_commands.label.compact_queue_wait_failed"),
                )
        finally:
            if self._compaction_input_queue_task is current_task:
                self._compaction_input_queue_task = None

    def _discard_compaction_input_queue(self) -> int:
        count = len(self._compaction_input_queue)
        self._compaction_input_queue.clear()
        return count

    async def _close_compaction_input_queue(self) -> None:
        self._discard_compaction_input_queue()
        task = self._compaction_input_queue_task
        self._compaction_input_queue_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        fork_task = self._fork_after_turn_task
        self._fork_after_turn_task = None
        self._fork_after_turn_thread_id = None
        if fork_task is not None and not fork_task.done():
            fork_task.cancel()
            await asyncio.gather(fork_task, return_exceptions=True)

    async def close_shell_session(self) -> None:
        session = self.shell_session
        self.shell_session = None
        if session is not None:
            await session.close()

    async def _enter_shell(self, incoming: IncomingMessage) -> None:
        if self.shell_session is not None and self.shell_session.is_open:
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
        self.shell_session = session
        try:
            await session.start()
        except Exception as exc:
            if self.shell_session is session:
                self.shell_session = None
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

    async def _exit_shell(self, incoming: IncomingMessage) -> None:
        session = self.shell_session
        if session is None:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("shell.label.not_open"),
                reply_to_message_id=incoming.message_id,
            )
            return
        self.shell_session = None
        await session.close()
        await self.telegram.send_message(
            incoming.chat_id,
            self._text("shell.label.closed"),
            reply_to_message_id=incoming.message_id,
        )

    async def _send_shell_input(self, incoming: IncomingMessage) -> None:
        session = self.shell_session
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
            if self.shell_session is session:
                self.shell_session = None
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

    async def _stop_shell(self, incoming: IncomingMessage) -> None:
        session = self.shell_session
        stopped = await session.interrupt() if session is not None else False
        await self.telegram.send_message(
            incoming.chat_id,
            self._text("shell.label.stop_requested" if stopped else "shell.label.stop_idle"),
            reply_to_message_id=incoming.message_id,
        )

    async def _request_chat_recovery(self, incoming: IncomingMessage) -> None:
        codex = self._codex()
        if self._chat_recovery_in_progress:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("recovery.label.in_progress"),
                reply_to_message_id=incoming.message_id,
            )
            return
        if self.pending_chat_recovery is not None:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("recovery.label.confirm_pending"),
                reply_to_message_id=incoming.message_id,
            )
            return
        thread_id = codex.state.thread_id
        if not thread_id:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("common.label.no_chat"),
                reply_to_message_id=incoming.message_id,
            )
            return

        token = secrets.token_urlsafe(5)
        self._recovery_ready.clear()
        hold = getattr(codex, "hold_recovery", None)
        if callable(hold):
            hold()
        try:
            message = await self.telegram.send_message(
                incoming.chat_id,
                self._text("recovery.label.reset_confirm"),
                self._chat_recovery_keyboard(token, "reset"),
                reply_to_message_id=incoming.message_id,
            )
        except Exception:
            self._release_chat_recovery_hold()
            if not getattr(codex, "recovery_blocked", False):
                self._recovery_ready.set()
            raise
        message_id = message.get("message_id")
        if not isinstance(message_id, int):
            self._release_chat_recovery_hold()
            if not getattr(codex, "recovery_blocked", False):
                self._recovery_ready.set()
            return
        self.pending_chat_recovery = PendingChatRecovery(
            token=token,
            message_id=message_id,
            thread_id=thread_id,
            stage="reset",
        )

    def _chat_recovery_keyboard(self, token: str, action: str) -> dict[str, Any]:
        confirm_key = "recovery.buttons.reset_button" if action == "reset" else "recovery.buttons.apply_button"
        confirm_action = action
        return {
            "inline_keyboard": [
                [
                    {
                        "text": self._text(confirm_key),
                        "callback_data": f"rr:{token}:{confirm_action}",
                    }
                ],
                [
                    {
                        "text": self._text("common.buttons.cancel"),
                        "callback_data": f"rr:{token}:cancel",
                    }
                ],
            ]
        }

    def _recovery_impact_text(self, snapshot: CodexRecoverySnapshot) -> str:
        parts = [self._text("recovery.label.impact_title")]
        if snapshot.thread_status != "idle" or snapshot.active_turn_id is not None:
            parts.append(self._text("recovery.label.active_turn_warning"))
        if snapshot.compaction_pending:
            parts.append(self._text("recovery.label.compaction_warning"))
        if snapshot.thread_status is None:
            parts.append(self._text("recovery.label.status_unknown_warning"))
        if snapshot.terminals:
            lines = [self._text("recovery.label.terminals_heading")]
            for terminal in snapshot.terminals[:8]:
                lines.append(f"• {self._terminal_command_text(terminal)}")
            if len(snapshot.terminals) > 8:
                lines.append(
                    self._text("recovery.label.more_terminals", count=len(snapshot.terminals) - 8)
                )
            parts.append("\n".join(lines))
        parts.append(self._text("recovery.label.impact_confirm_note"))
        return "\n\n".join(parts)

    async def _handle_chat_recovery_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        pending = self.pending_chat_recovery
        if (
            pending is None
            or len(parts) != 3
            or parts[1] != pending.token
            or incoming.message_id != pending.message_id
        ):
            await self.telegram.answer_callback(
                incoming.callback_id,
                self._text("recovery.label.stale"),
            )
            return
        action = parts[2]
        if self._chat_recovery_in_progress:
            await self.telegram.answer_callback(
                incoming.callback_id,
                self._text("recovery.label.in_progress"),
            )
            return
        if action == "cancel":
            self.pending_chat_recovery = None
            self._release_chat_recovery_hold()
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("recovery.label.cancelled"),
                {"inline_keyboard": []},
            )
            self._resume_compaction_queue_after_recovery()
            return
        if (
            action == "reset" and pending.stage != "reset"
        ) or (
            action == "apply" and pending.stage != "impact"
        ) or action not in {"reset", "apply"}:
            await self.telegram.answer_callback(
                incoming.callback_id,
                self._text("recovery.label.stale"),
            )
            return

        codex = self._codex()
        if codex.state.thread_id != pending.thread_id:
            self.pending_chat_recovery = None
            self._release_chat_recovery_hold()
            await self.telegram.answer_callback(
                incoming.callback_id,
                self._text("recovery.label.stale"),
            )
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("recovery.label.stale"),
                {"inline_keyboard": []},
            )
            self._resume_compaction_queue_after_recovery()
            return

        self._chat_recovery_in_progress = True
        pending.stage = "working"
        try:
            await self.telegram.answer_callback(incoming.callback_id)
            if action == "reset":
                snapshot = await codex.inspect_current_thread_for_recovery()
                if snapshot.thread_id != pending.thread_id:
                    raise RuntimeError("The selected chat changed during recovery inspection")
                if snapshot.requires_confirmation:
                    await self._ask_recovery_impact_confirmation(
                        pending,
                        snapshot,
                        incoming.chat_id,
                    )
                    return
                result = await codex.recover_current_thread(
                    pending.thread_id,
                    confirmed_terminal_ids=set(),
                    confirm_interrupt=False,
                )
            else:
                result = await codex.recover_current_thread(
                    pending.thread_id,
                    confirmed_terminal_ids=set(pending.terminal_ids),
                    confirm_interrupt=pending.confirm_interrupt,
                )
            if result is not None:
                await self._ask_recovery_impact_confirmation(
                    pending,
                    result,
                    incoming.chat_id,
                )
                return

            await self._close_codex_interactions_for_recovery()
            self.pending_chat_recovery = None
            self._release_chat_recovery_hold()
            await self._finish_typing()
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("recovery.label.chat_recovered"),
                {"inline_keyboard": []},
            )
            self._resume_compaction_queue_after_recovery()
        except Exception as exc:
            self._log_failure(
                "Could not recover the selected Codex chat",
                exc,
                event="telegram.recovery.chat_failed",
            )
            self.pending_chat_recovery = None
            if not getattr(codex, "recovery_blocked", False):
                self._release_chat_recovery_hold()
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("recovery.label.chat_recovery_failed"),
                {"inline_keyboard": []},
            )
            if not getattr(codex, "recovery_blocked", False):
                self._resume_compaction_queue_after_recovery()
        finally:
            self._chat_recovery_in_progress = False

    async def _ask_recovery_impact_confirmation(
        self,
        pending: PendingChatRecovery,
        snapshot: CodexRecoverySnapshot,
        chat_id: int,
    ) -> None:
        pending.stage = "impact"
        pending.terminal_ids = tuple(item.process_id for item in snapshot.terminals)
        pending.confirm_interrupt = snapshot.has_active_work
        await self._edit_message(
            chat_id,
            pending.message_id,
            self._recovery_impact_text(snapshot),
            self._chat_recovery_keyboard(pending.token, "apply"),
        )

    async def _close_codex_interactions_for_recovery(self) -> None:
        if self.pending_user_input is not None and not self.pending_user_input.future.done():
            self.pending_user_input.future.set_result({"answers": {}})
        self.pending_user_input = None
        for pending in self.pending_approvals.values():
            if not pending.future.done():
                pending.future.set_result("cancel")
        self.pending_approvals.clear()
        for pending in self.pending_mcp_elicitations.values():
            if not pending.future.done():
                pending.future.set_result({"action": "cancel"})
            await self._edit_message(
                self.chat_id,
                pending.message_id,
                self._text("mcp.label.elicitation_closed"),
                {"inline_keyboard": []},
            )
        self.pending_mcp_elicitations.clear()
        self._mcp_form_token = None

    def _release_chat_recovery_hold(self) -> None:
        codex = self._codex()
        release = getattr(codex, "release_recovery_hold", None)
        if callable(release):
            release()

    def _resume_compaction_queue_after_recovery(self) -> None:
        if getattr(self._codex(), "recovery_blocked", False):
            return
        self._recovery_ready.set()
        if self._compaction_input_queue:
            self._start_compaction_input_queue_worker()

    async def _stop_codex(self, incoming: IncomingMessage) -> None:
        codex = self._codex()
        discarded = self._discard_compaction_input_queue()
        cancelled_compaction = await codex.cancel_pending_manual_compaction()
        was_busy = codex.is_busy
        try:
            with bind_log_context(interrupt_source="telegram_stop"):
                stopped = await codex.interrupt()
        except Exception as exc:
            self._log_failure(
                "Could not interrupt Codex turn",
                exc,
                event="telegram.command.stop_failed",
                level=logging.WARNING,
            )
            await self.telegram.send_message(incoming.chat_id, self._text("commands.label.stop_failed"))
        else:
            if stopped and was_busy:
                messages = []
            elif stopped:
                messages = [self._text("goal.label.paused")]
            else:
                messages = [self._text("commands.label.stop_idle")]
            if cancelled_compaction:
                messages.append(self._text("thread_commands.label.compact_cancelled"))
            if discarded:
                messages.append(
                    self._text("thread_commands.label.compact_queue_discarded", count=discarded)
                )
            if messages:
                await self.telegram.send_message(incoming.chat_id, "\n".join(messages))
        if cancelled_compaction and not codex.is_busy:
            await self._finish_typing()

    async def _on_shell_output(self, output: ShellOutput) -> None:
        if self.is_group and not self.attached:
            return
        text = _clean_terminal_output(output.text)
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
        limit = GROUP_ACTIVITY_MESSAGE_LIMIT if self.is_group else TELEGRAM_TEXT_LIMIT
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
        limit = GROUP_ACTIVITY_MESSAGE_LIMIT if self.is_group else TELEGRAM_TEXT_LIMIT
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
        if self.shell_session is not session:
            return
        self.shell_session = None
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

    async def _handle_callback(self, incoming: IncomingCallback) -> None:
        parts = incoming.data.split(":")
        if self._chat_recovery_in_progress and parts and parts[0] not in {"ap", "me", "rr"}:
            await self.telegram.answer_callback(
                incoming.callback_id,
                self._text("recovery.label.in_progress"),
            )
            return
        if (
            self._codex().is_busy
            and parts
            and parts[0] not in BUSY_ALLOWED_CALLBACKS
        ):
            logger.info(
                "Blocked Telegram action while Codex is busy",
                extra={
                    "event": "telegram.callback.blocked_while_busy",
                    "callback_kind": parts[0],
                    **self._log_context_fields(),
                },
            )
            await self.telegram.answer_callback(
                incoming.callback_id,
                self._text("thread_commands.label.busy"),
            )
            return
        if parts and parts[0] == "l":
            await self._handle_language_callback(incoming, parts)
            return
        if parts and parts[0] == "om":
            await self._handle_mode_callback(incoming, parts)
            return
        if parts and parts[0] == "gm":
            await self._handle_group_mode_callback(incoming, parts)
            return
        if parts and parts[0] == "sk":
            await self._handle_skills_callback(incoming, parts)
            return
        if parts and parts[0] == "mm":
            await self._handle_memory_callback(incoming, parts)
            return
        if parts and parts[0] == "m":
            await self._handle_model_callback(incoming, parts)
            return
        if parts and parts[0] == "np":
            await self._handle_directory_callback(incoming, parts)
            return
        if parts and parts[0] == "fb":
            await self._handle_files_callback(incoming, parts)
            return
        if parts and parts[0] == "cp":
            await self._handle_chats_callback(incoming, parts)
            return
        if parts and parts[0] == "pm":
            await self._handle_permissions_callback(incoming, parts)
            return
        if parts and parts[0] == "td":
            await self._handle_delete_callback(incoming, parts)
            return
        if parts and parts[0] == "ps":
            await self._handle_processes_callback(incoming, parts)
            return
        if parts and parts[0] == "mc":
            await self._handle_mcp_callback(incoming, parts)
            return
        if parts and parts[0] == "sw":
            await self._handle_thread_switch_callback(incoming, parts)
            return
        if parts and parts[0] == "rc":
            await self._handle_stale_turn_callback(incoming, parts)
            return
        if parts and parts[0] == "rr":
            await self._handle_chat_recovery_callback(incoming, parts)
            return
        if parts and parts[0] == "me":
            await self._handle_mcp_elicitation_callback(incoming, parts)
            return
        if len(parts) == 3 and parts[0] == "ap":
            token, choice = parts[1], parts[2]
            pending = self.pending_approvals.pop(token, None)
            if pending is None or pending.future.done():
                await self.telegram.answer_callback(incoming.callback_id, self._text("approval.label.closed"))
                return
            decisions = {"once": "accept", "session": "acceptForSession", "deny": "decline"}
            decision = decisions.get(choice)
            if not decision:
                await self.telegram.answer_callback(incoming.callback_id, self._text("approval.label.unknown_decision"))
                return
            pending.future.set_result(decision)
            await self.telegram.answer_callback(incoming.callback_id, self._text("approval.label.sent"))
            await self._edit_message_reply_markup(
                incoming.chat_id,
                incoming.message_id,
                {"inline_keyboard": []},
            )
            return
        await self.telegram.answer_callback(incoming.callback_id, self._text("common.label.stale_button"))


















    async def _show_model_picker(self, chat_id: int) -> None:
        try:
            codex = self._codex()
            models = await codex.list_models()
        except Exception as exc:
            self._log_failure(
                "Could not list Codex models",
                exc,
                event="telegram.models.list_failed",
            )
            await self.telegram.send_message(chat_id, self._text("models.label.list_failure"))
            return
        if not models:
            await self.telegram.send_message(chat_id, self._text("models.label.empty"))
            return

        try:
            current_model, current_effort = await codex.current_model_settings(models)
        except Exception as exc:
            self._log_failure(
                "Could not read the current model settings for the Codex thread",
                exc,
                event="telegram.models.current_settings_failed",
                level=logging.WARNING,
            )
            current_model = codex.selected_model or codex.default_model(models)
            current_effort = codex.selected_effort
        picker = ModelPicker(
            token=secrets.token_urlsafe(6),
            models=models,
            page=0,
            current_model=current_model,
            current_effort=current_effort,
        )
        message = await self.telegram.send_message(
            chat_id,
            self._model_picker_text(picker),
            self._model_picker_keyboard(picker),
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.model_picker = picker

    async def _show_language_picker(self, chat_id: int) -> None:
        picker = LanguagePicker(token=secrets.token_urlsafe(6))
        keyboard = self._language_picker_keyboard(picker)
        message = await self.telegram.send_message(
            chat_id,
            self.texts.system("label.language_picker_prompt"),
            keyboard,
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.language_picker = picker

    async def _show_mode_picker(self, chat_id: int) -> None:
        picker = ModePicker(token=secrets.token_urlsafe(6))
        message = await self.telegram.send_message(
            chat_id,
            self._mode_picker_text(),
            self._mode_picker_keyboard(picker),
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.mode_picker = picker

    async def _show_skills_picker(self, chat_id: int) -> None:
        try:
            skills = await self._codex().list_skills()
        except Exception as exc:
            self._log_failure(
                "Could not list Codex skills",
                exc,
                event="telegram.skills.list_failed",
            )
            await self.telegram.send_message(chat_id, self._text("skills.label.list_failed"))
            return
        if not skills:
            await self.telegram.send_message(chat_id, self._text("skills.label.empty"))
            return
        picker = SkillPicker(token=secrets.token_urlsafe(6), skills=skills)
        message = await self.telegram.send_message(
            chat_id,
            self._skills_picker_text(picker),
            self._skills_picker_keyboard(picker),
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.skill_picker = picker

    def _skills_picker_text(self, picker: SkillPicker) -> str:
        if picker.action == "menu":
            selected = self._codex().state.pending_skill_name
            selected_text = (
                self._text("skills.values.selected", name=selected)
                if selected
                else self._text("skills.values.none_selected")
            )
            return self._text("skills.label.menu", selected=selected_text)
        title = self._text(
            "skills.label.list_title",
            mode=self._text(f"skills.values.actions.{picker.action}"),
            current=picker.page + 1,
            total=page_count(len(picker.skills)),
        )
        if picker.action == "manage":
            return f"{title}\n{self._text('skills.label.manage_notice')}"
        return title

    def _skills_picker_keyboard(self, picker: SkillPicker) -> dict[str, Any]:
        if picker.action == "menu":
            return {
                "inline_keyboard": [
                    [{"text": self._text("skills.buttons.use_button"), "callback_data": f"sk:{picker.token}:a:use"}],
                    [{"text": self._text("skills.buttons.manage_button"), "callback_data": f"sk:{picker.token}:a:manage"}],
                ]
            }
        start = picker.page * PAGE_SIZE
        skills = picker.skills[start : start + PAGE_SIZE]
        buttons = []
        for offset, skill in enumerate(skills):
            absolute_index = start + offset
            label = skill.display_name or skill.name
            if picker.action == "manage":
                label = self._text(
                    "skills.buttons.enabled_prefix" if skill.enabled else "skills.buttons.disabled_prefix",
                    name=label,
                )
                action = "toggle"
            else:
                action = "use"
            buttons.append({
                "text": truncate(
                    label,
                    INLINE_BUTTON_TEXT_LIMIT,
                    self.texts.system("values.ellipsis"),
                ),
                "callback_data": f"sk:{picker.token}:{action}:{absolute_index}",
            })
        rows = two_column_rows(buttons, 0)
        total_pages = page_count(len(picker.skills))
        if total_pages > 1:
            nav: list[dict[str, str]] = []
            if picker.page > 0:
                nav.append({"text": "‹", "callback_data": f"sk:{picker.token}:p:{picker.page - 1}"})
            nav.append({
                "text": self._text("skills.buttons.page", current=picker.page + 1, total=total_pages),
                "callback_data": f"sk:{picker.token}:n",
            })
            if picker.page + 1 < total_pages:
                nav.append({"text": "›", "callback_data": f"sk:{picker.token}:p:{picker.page + 1}"})
            rows.append(nav)
        rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"sk:{picker.token}:b"}])
        return {"inline_keyboard": rows}

    async def _show_memory_picker(self, chat_id: int) -> None:
        codex = self._codex()
        try:
            settings = await codex.read_memory_settings()
        except Exception as exc:
            self._log_failure(
                "Could not read Codex memory settings",
                exc,
                event="telegram.memories.read_failed",
            )
            await self.telegram.send_message(chat_id, self._text("memories.label.read_failed"))
            return
        current_mode = codex.current_memory_mode()
        current_use_memories = (
            current_mode == "enabled"
            if current_mode is not None
            else (settings.use_memories if codex.state.thread_id is None else None)
        )
        picker = MemoryPicker(
            token=secrets.token_urlsafe(6),
            thread_id=codex.state.thread_id,
            settings=settings,
            use_memories=current_use_memories,
        )
        message = await self.telegram.send_message(
            chat_id,
            self._memory_picker_text(picker),
            self._memory_picker_keyboard(picker),
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.memory_picker = picker

    def _memory_picker_text(self, picker: MemoryPicker) -> str:
        if picker.confirming_generation:
            return self._text("memories.label.generate_confirmation")
        if picker.confirming_reset:
            return self._text("memories.label.reset_confirmation")
        if not picker.settings.feature_enabled:
            return self._text("memories.label.feature_disabled")
        return self._text(
            "memories.label.prompt",
            scope=self._text("memories.values.current_scope" if picker.thread_id else "memories.values.new_scope"),
            use_state=self._text(
                "memories.values.enabled"
                if picker.use_memories is True
                else "memories.values.disabled"
                if picker.use_memories is False
                else "memories.values.unknown"
            ),
            generate_state=self._text(
                "memories.values.enabled" if picker.settings.generate_memories else "memories.values.disabled"
            ),
        )

    def _memory_picker_keyboard(self, picker: MemoryPicker) -> dict[str, Any]:
        if picker.confirming_generation:
            return {
                "inline_keyboard": [
                    [{"text": self._text("memories.buttons.confirm_button"), "callback_data": f"mm:{picker.token}:confirm"}],
                    [{"text": self._text("common.buttons.cancel"), "callback_data": f"mm:{picker.token}:cancel"}],
                ]
            }
        if picker.confirming_reset:
            return {
                "inline_keyboard": [
                    [{"text": self._text("memories.buttons.reset_confirm_button"), "callback_data": f"mm:{picker.token}:reset_confirm"}],
                    [{"text": self._text("common.buttons.cancel"), "callback_data": f"mm:{picker.token}:cancel"}],
                ]
            }
        if not picker.settings.feature_enabled:
            return {
                "inline_keyboard": [[{
                    "text": self._text("memories.buttons.enable_button"),
                    "callback_data": f"mm:{picker.token}:enable",
                }]]
            }
        memory_mode_buttons = two_column_rows(
            [
                {
                    "text": self._selected_button_label(
                        self._text("memories.buttons.enable_use_button"),
                        picker.use_memories is True,
                    ),
                    "callback_data": f"mm:{picker.token}:use:enabled",
                },
                {
                    "text": self._selected_button_label(
                        self._text("memories.buttons.disable_use_button"),
                        picker.use_memories is False,
                    ),
                    "callback_data": f"mm:{picker.token}:use:disabled",
                },
            ],
            0,
        )
        return {
            "inline_keyboard": memory_mode_buttons + [
                [{
                    "text": self._text(
                        "memories.buttons.generate_button",
                        state=self._text("memories.values.enabled" if picker.settings.generate_memories else "memories.values.disabled"),
                    ),
                    "callback_data": f"mm:{picker.token}:generate",
                }],
                [{
                    "text": self._text("memories.buttons.reset_button"),
                    "callback_data": f"mm:{picker.token}:reset",
                }],
            ]
        }

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
            message = await self.telegram.send_message(chat_id, text, markup)
            message_id = message.get("message_id")
            if isinstance(message_id, int):
                picker.message_id = message_id
                self.files_picker = picker
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
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        picker = self.files_picker
        if (
            picker is None
            or len(parts) < 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
            or incoming.user_id != picker.owner_user_id
        ):
            await self.telegram.answer_callback(incoming.callback_id, self._text("files.label.stale"))
            return
        action = parts[2]
        if action == "p" and len(parts) == 4:
            try:
                page = int(parts[3])
            except ValueError:
                page = -1
            if not 0 <= page < page_count(len(picker.entries)):
                await self.telegram.answer_callback(incoming.callback_id, self._text("files.label.stale"))
                return
            picker.page = page
            picker.selected_index = None
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._files_picker_text(picker),
                self._files_picker_keyboard(picker),
            )
            return
        if action == "i" and len(parts) == 3:
            await self.telegram.answer_callback(incoming.callback_id)
            return
        if action == "u" and len(parts) == 3:
            if picker.current_path == self._workspace_root():
                await self.telegram.answer_callback(incoming.callback_id)
                return
            picker.current_path = picker.current_path.parent
            picker.page = 0
            await self.telegram.answer_callback(incoming.callback_id)
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
                await self.telegram.answer_callback(incoming.callback_id, self._text("files.label.stale"))
                return
            if action == "o" and entry.kind == "directory":
                picker.current_path = entry.path
                picker.page = 0
                await self.telegram.answer_callback(incoming.callback_id)
                if not await self._load_files_picker(incoming.chat_id, picker):
                    await self.telegram.send_message(incoming.chat_id, self._text("files.label.read_failed"))
                return
            if action == "f" and entry.kind == "file":
                picker.selected_index = index
                await self.telegram.answer_callback(incoming.callback_id)
                await self._edit_message(
                    incoming.chat_id,
                    incoming.message_id,
                    self._files_detail_text(entry),
                    self._files_detail_keyboard(picker),
                )
                return
        if action == "b" and len(parts) == 3 and self._selected_workspace_file(picker):
            picker.selected_index = None
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._files_picker_text(picker),
                self._files_picker_keyboard(picker),
            )
            return
        entry = self._selected_workspace_file(picker)
        if entry is None or action not in {"d", "v"} or len(parts) != 3:
            await self.telegram.answer_callback(incoming.callback_id, self._text("files.label.stale"))
            return
        await self.telegram.answer_callback(incoming.callback_id)
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
                await self.telegram.send_document(
                    chat_id,
                    path,
                    filename,
                    reply_to_message_id=picker.message_id,
                )
            except WorkspaceFileTooLarge:
                await self.telegram.send_message(chat_id, self._text("files.label.too_large"))
            except TelegramApiError as exc:
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

    async def _show_directory_picker(self, chat_id: int) -> None:
        root = self._workspace_root()
        try:
            directories = await self._list_workspace_directories(root)
        except Exception:
            await self.telegram.send_message(chat_id, self._text("directories.label.read_failed"))
            return
        picker = DirectoryPicker(
            token=secrets.token_urlsafe(6),
            current_path=root,
            directories=directories,
        )
        message = await self.telegram.send_message(
            chat_id,
            self._directory_picker_text(picker),
            self._directory_picker_keyboard(picker),
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.directory_picker = picker

    async def _show_chats_picker(self, chat_id: int) -> None:
        try:
            threads = await self._codex().list_threads()
        except Exception as exc:
            self._log_failure(
                "Could not list Codex chats",
                exc,
                event="telegram.chats.list_failed",
            )
            await self.telegram.send_message(chat_id, self._text("chats.label.list_failed"))
            return
        if not threads:
            await self.telegram.send_message(chat_id, self._text("chats.label.empty"))
            return
        picker = ChatsPicker(token=secrets.token_urlsafe(6), threads=threads)
        message = await self.telegram.send_message(
            chat_id,
            self._chats_picker_text(picker),
            self._chats_picker_keyboard(picker),
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.chats_picker = picker

    async def _show_permissions_picker(self, chat_id: int) -> None:
        picker = PermissionsPicker(token=secrets.token_urlsafe(6))
        message = await self.telegram.send_message(
            chat_id,
            self._permissions_picker_text(),
            self._permissions_picker_keyboard(picker),
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.permissions_picker = picker

    async def _rename_chat(self, chat_id: int, name: str) -> None:
        normalized_name = " ".join(name.split())
        if not normalized_name:
            await self.telegram.send_message(chat_id, self._text("rename.label.usage"))
            return
        if len(normalized_name) > 100:
            await self.telegram.send_message(chat_id, self._text("rename.label.too_long"))
            return
        try:
            await self._codex().rename_thread(normalized_name)
        except RuntimeError as exc:
            if "no active" in str(exc).lower():
                await self.telegram.send_message(chat_id, self._text("rename.label.no_chat"))
            else:
                self._log_failure(
                    "Could not rename Codex chat",
                    exc,
                    event="telegram.chat.rename_failed",
                    chat_name_length=len(normalized_name),
                )
                await self.telegram.send_message(chat_id, self._text("rename.label.failed"))
        except Exception as exc:
            self._log_failure(
                "Could not rename Codex chat",
                exc,
                event="telegram.chat.rename_failed",
                chat_name_length=len(normalized_name),
            )
            await self.telegram.send_message(chat_id, self._text("rename.label.failed"))
        else:
            await self.telegram.send_message(
                chat_id,
                self._text("rename.label.saved", name=normalized_name),
            )

    async def _show_delete_confirmation(self, chat_id: int) -> None:
        codex = self._codex()
        if not codex.state.thread_id:
            await self.telegram.send_message(chat_id, self._text("common.label.no_chat"))
            return
        if codex.is_busy:
            await self.telegram.send_message(chat_id, self._text("thread_commands.label.busy"))
            return
        confirmation = DeleteConfirmation(
            token=secrets.token_urlsafe(6),
            thread_id=codex.state.thread_id,
        )
        keyboard = {
            "inline_keyboard": [
                [
                    {
                        "text": self._text("thread_commands.buttons.delete_confirm_button"),
                        "callback_data": f"td:{confirmation.token}:confirm",
                    },
                    {
                        "text": self._text("common.buttons.cancel"),
                        "callback_data": f"td:{confirmation.token}:cancel",
                    },
                ]
            ]
        }
        message = await self.telegram.send_message(
            chat_id,
            self._text("thread_commands.label.delete_confirmation"),
            keyboard,
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            confirmation.message_id = message_id
            self.delete_confirmation = confirmation

    async def _handle_delete_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        confirmation = self.delete_confirmation
        if (
            confirmation is None
            or len(parts) != 3
            or parts[1] != confirmation.token
            or incoming.message_id != confirmation.message_id
        ):
            await self.telegram.answer_callback(incoming.callback_id, self._text("thread_commands.label.delete_stale"))
            return
        action = parts[2]
        if action == "cancel":
            self.delete_confirmation = None
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("thread_commands.label.delete_cancelled"),
                {"inline_keyboard": []},
            )
            return
        if action != "confirm":
            await self.telegram.answer_callback(incoming.callback_id, self._text("thread_commands.label.delete_stale"))
            return

        self.delete_confirmation = None
        await self.telegram.answer_callback(incoming.callback_id)
        try:
            await self._codex().delete_thread(confirmation.thread_id)
        except RuntimeError as exc:
            message_key = "thread_commands.label.busy" if "busy" in str(exc).lower() else "thread_commands.label.delete_stale"
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text(message_key),
                {"inline_keyboard": []},
            )
        except Exception as exc:
            self._log_failure(
                "Could not delete Codex chat",
                exc,
                event="telegram.chat.delete_failed",
                thread_hash=log_identifier(confirmation.thread_id, namespace="codex-thread"),
            )
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("thread_commands.label.delete_failed"),
                {"inline_keyboard": []},
            )
        else:
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("thread_commands.label.deleted"),
                {"inline_keyboard": []},
            )

    async def _fork_chat(self, chat_id: int) -> None:
        codex = self._codex()
        if not codex.state.thread_id:
            await self.telegram.send_message(chat_id, self._text("common.label.no_chat"))
            return
        if codex.is_busy:
            task = self._fork_after_turn_task
            if task is not None and not task.done():
                await self.telegram.send_message(
                    chat_id,
                    self._text("thread_commands.label.fork_already_waiting"),
                )
                return
            thread_id = codex.state.thread_id
            self._fork_after_turn_thread_id = thread_id
            self._fork_after_turn_task = asyncio.create_task(
                self._fork_chat_after_turn(chat_id, thread_id)
            )
            await self.telegram.send_message(
                chat_id,
                self._text("thread_commands.label.fork_waiting"),
            )
            return
        await self._create_fork(chat_id)

    async def _fork_chat_after_turn(self, chat_id: int, source_thread_id: str) -> None:
        current_task = asyncio.current_task()
        codex = self._codex()
        try:
            while not self._shutdown_in_progress and (not self.is_group or self.attached):
                if codex.state.thread_id != source_thread_id:
                    raise RuntimeError("The selected Codex chat changed before the queued fork")
                try:
                    await codex.wait_for_manual_compaction_barrier()
                    await codex.wait_for_current_turn_completion()
                except CodexChatBusyError:
                    await asyncio.sleep(0.5)
                    continue

                queue_task = self._compaction_input_queue_task
                if queue_task is not None and queue_task is not current_task and not queue_task.done():
                    await asyncio.shield(queue_task)
                    continue
                if codex.is_busy:
                    await asyncio.sleep(0.2)
                    continue
                terminals = await codex.list_background_terminals()
                if terminals:
                    await asyncio.sleep(2)
                    continue
                if await self._create_fork(
                    chat_id,
                    retry_if_busy_for_thread=source_thread_id,
                ):
                    return
                await asyncio.sleep(0.2)
        except Exception as exc:
            self._log_failure(
                "Could not complete the Codex fork after the active turn",
                exc,
                event="telegram.chat.deferred_fork_failed",
            )
            await self.telegram.send_message(chat_id, self._text("thread_commands.label.fork_failed"))
        finally:
            if self._fork_after_turn_task is current_task:
                self._fork_after_turn_task = None
                self._fork_after_turn_thread_id = None

    async def _create_fork(
        self,
        chat_id: int,
        *,
        retry_if_busy_for_thread: str | None = None,
    ) -> bool:
        try:
            await self._codex().fork_thread()
        except Exception as exc:
            if (
                retry_if_busy_for_thread is not None
                and self._codex().state.thread_id == retry_if_busy_for_thread
                and self._codex().is_busy
                and isinstance(exc, (CodexChatBusyError, RuntimeError))
                and "busy" in str(exc).casefold()
            ):
                return False
            self._log_failure(
                "Could not fork Codex chat",
                exc,
                event="telegram.chat.fork_failed",
            )
            await self.telegram.send_message(chat_id, self._text("thread_commands.label.fork_failed"))
            return True
        shell_was_open = self.shell_session is not None
        await self.close_shell_session()
        key = "thread_commands.label.forked_shell_closed" if shell_was_open else "thread_commands.label.forked"
        await self.telegram.send_message(chat_id, self._text(key))
        return True

    async def _compact_chat(self, chat_id: int) -> None:
        codex = self._codex()
        if not codex.state.thread_id:
            await self.telegram.send_message(chat_id, self._text("common.label.no_chat"))
            return
        self._start_typing()
        try:
            await codex.compact_thread()
        except CodexChatBusyError:
            await self._finish_typing()
            await self.telegram.send_message(chat_id, self._text("thread_commands.label.busy"))
            return
        except Exception as exc:
            await self._finish_typing()
            self._log_failure(
                "Could not compact Codex chat",
                exc,
                event="telegram.chat.compact_failed",
            )
            await self.telegram.send_message(chat_id, self._text("thread_commands.label.compact_failed"))
            return
        self._notify_typing_cleared()

    async def _handle_plan_command(self, incoming: IncomingMessage, argument: str) -> None:
        codex = self._codex()
        value = argument.strip()
        explicit_mode = value.casefold()
        if not value or explicit_mode in {"on", "off"}:
            if not codex.state.thread_id:
                await self.telegram.send_message(
                    incoming.chat_id,
                    self._text("common.label.no_chat"),
                )
                return
            try:
                if not value:
                    mode = await codex.toggle_plan_mode()
                else:
                    mode = "plan" if explicit_mode == "on" else "default"
                    await codex.set_collaboration_mode(mode)
            except CollaborationModeUnavailableError as exc:
                logger.info(
                    "Codex plan mode is unavailable",
                    extra={"event": "telegram.command.plan_unavailable"},
                )
                await self.telegram.send_message(incoming.chat_id, self._text("plan.label.unavailable"))
            except Exception as exc:
                self._log_failure(
                    "Could not change Codex collaboration mode",
                    exc,
                    event="telegram.command.plan_failed",
                    requested_mode=(
                        "plan" if explicit_mode == "on" else "default"
                    ) if explicit_mode in {"on", "off"} else None,
                )
                await self.telegram.send_message(incoming.chat_id, self._text("plan.label.failed"))
            else:
                text_key = "plan.label.enabled" if mode == "plan" else "plan.label.disabled"
                await self.telegram.send_message(incoming.chat_id, self._text(text_key))
            return

        try:
            await codex.set_collaboration_mode("plan", create_if_missing=True)
        except CollaborationModeUnavailableError:
            await self.telegram.send_message(incoming.chat_id, self._text("plan.label.unavailable"))
        except Exception as exc:
            self._log_failure(
                "Could not start a Codex planning turn",
                exc,
                event="telegram.command.plan_start_failed",
            )
            await self.telegram.send_message(incoming.chat_id, self._text("plan.label.failed"))
        else:
            await self._send_to_codex(incoming.chat_id, value, incoming.message_id)

    async def _consume_goal_edit_input(self, incoming: IncomingMessage) -> bool:
        pending = self.pending_goal_edit
        if pending is None:
            return False
        codex = self._codex()
        if codex.state.thread_id != pending.thread_id:
            self.pending_goal_edit = None
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("goal.label.edit_stale"),
                reply_to_message_id=incoming.message_id,
            )
            return True

        parsed = None if incoming.attachments else parse_command(incoming.text)
        if parsed is not None:
            self.pending_goal_edit = None
            if parsed[0] == "cancel":
                await self.telegram.send_message(
                    incoming.chat_id,
                    self._text("goal.label.edit_cancelled"),
                    reply_to_message_id=incoming.message_id,
                )
                return True
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("goal.label.edit_cancelled"),
                reply_to_message_id=incoming.message_id,
            )
            return False

        if incoming.attachments:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("goal.label.edit_text_only"),
                reply_to_message_id=incoming.message_id,
            )
            return True
        if not incoming.text.strip():
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("goal.label.edit_text_only"),
                reply_to_message_id=incoming.message_id,
            )
            return True

        self.pending_goal_edit = None
        self._turn_prompt_message_id = incoming.message_id
        self._start_typing()
        try:
            goal = await codex.set_goal(incoming.text, prompt_message_id=incoming.message_id)
        except GoalUnavailableError:
            await self._finish_typing()
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("goal.label.unavailable"),
                reply_to_message_id=incoming.message_id,
            )
        except Exception as exc:
            await self._finish_typing()
            self._log_failure(
                "Could not edit Codex Goal",
                exc,
                event="telegram.command.goal_edit_failed",
            )
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("goal.label.failed"),
                reply_to_message_id=incoming.message_id,
            )
        else:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("goal.label.started" if goal.status == "active" else "goal.label.updated"),
                reply_to_message_id=incoming.message_id,
            )
            if not codex.is_busy:
                await self._finish_typing()
        return True

    async def _handle_goal_command(self, incoming: IncomingMessage, argument: str) -> None:
        codex = self._codex()
        value = argument.strip()
        action = value.casefold()
        try:
            if not value:
                goal = await codex.get_goal()
                if goal is None:
                    text = self._text(
                        "common.label.no_chat" if not codex.state.thread_id else "goal.label.none"
                    )
                else:
                    status_key = goal.status if goal.status in {
                        "active", "paused", "blocked", "usageLimited", "budgetLimited", "complete"
                    } else "unknown"
                    budget = (
                        str(goal.token_budget)
                        if goal.token_budget is not None
                        else self._text("goal.values.no_budget")
                    )
                    text = self._text(
                        "goal.label.current",
                        objective=goal.objective,
                        status=self._text(f"goal.values.status.{status_key}"),
                        tokens=goal.tokens_used,
                        seconds=goal.time_used_seconds,
                        budget=budget,
                    )
                for part in split_plain_text(text, limit=3900):
                    await self.telegram.send_message(
                        incoming.chat_id,
                        part,
                        reply_to_message_id=incoming.message_id,
                    )
                return

            if action == "edit":
                if not codex.state.thread_id:
                    await self.telegram.send_message(
                        incoming.chat_id,
                        self._text("common.label.no_chat"),
                        reply_to_message_id=incoming.message_id,
                    )
                    return
                goal = await codex.get_goal()
                if goal is None:
                    await self.telegram.send_message(
                        incoming.chat_id,
                        self._text("goal.label.none"),
                        reply_to_message_id=incoming.message_id,
                    )
                    return
                self.pending_goal_edit = PendingGoalEdit(codex.state.thread_id)
                await self.telegram.send_message(
                    incoming.chat_id,
                    self._text("goal.label.edit_prompt"),
                    reply_to_message_id=incoming.message_id,
                )
                return

            if action in {"pause", "resume", "clear"}:
                if not codex.state.thread_id:
                    await self.telegram.send_message(
                        incoming.chat_id,
                        self._text("common.label.no_chat"),
                        reply_to_message_id=incoming.message_id,
                    )
                    return
                if action == "resume":
                    self._turn_prompt_message_id = incoming.message_id
                    self._start_typing()
                if action == "clear":
                    cleared = await codex.clear_goal()
                    await self.telegram.send_message(
                        incoming.chat_id,
                        self._text("goal.label.cleared" if cleared else "goal.label.none"),
                        reply_to_message_id=incoming.message_id,
                    )
                    return
                goal = await codex.set_goal_status(
                    "paused" if action == "pause" else "active",
                    prompt_message_id=incoming.message_id if action == "resume" else None,
                )
                text_key = "goal.label.paused" if action == "pause" else "goal.label.resumed"
                await self.telegram.send_message(
                    incoming.chat_id,
                    self._text(text_key),
                    reply_to_message_id=incoming.message_id,
                )
                if action == "resume" and not codex.is_busy:
                    await self._finish_typing()
                return

            self._turn_prompt_message_id = incoming.message_id
            self._start_typing()
            goal = await codex.set_goal(value, prompt_message_id=incoming.message_id)
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("goal.label.started" if goal.status == "active" else "goal.label.updated"),
                reply_to_message_id=incoming.message_id,
            )
            if not codex.is_busy:
                await self._finish_typing()
        except GoalNotFoundError:
            await self._finish_typing()
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("goal.label.none"),
                reply_to_message_id=incoming.message_id,
            )
        except GoalUnavailableError:
            await self._finish_typing()
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("goal.label.unavailable"),
                reply_to_message_id=incoming.message_id,
            )
        except Exception as exc:
            await self._finish_typing()
            self._log_failure(
                "Could not manage Codex Goal",
                exc,
                event="telegram.command.goal_failed",
                action=(action if action in {"pause", "resume", "clear"} else "set_or_read"),
            )
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("goal.label.failed"),
                reply_to_message_id=incoming.message_id,
            )

    async def _handle_fast_command(self, incoming: IncomingMessage, argument: str) -> None:
        requested = argument.strip().casefold()
        if requested:
            await self.telegram.send_message(incoming.chat_id, self._text("fast.label.usage"))
            return
        try:
            current = await self._codex().toggle_fast_mode()
        except FastTierUnavailableError:
            await self.telegram.send_message(incoming.chat_id, self._text("fast.label.unavailable"))
        except Exception as exc:
            self._log_failure(
                "Could not change the Codex Fast service tier",
                exc,
                event="telegram.command.fast_failed",
            )
            await self.telegram.send_message(incoming.chat_id, self._text("fast.label.failed"))
        else:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("fast.label.enabled" if current else "fast.label.disabled"),
            )

    async def _recap_chat(self, incoming: IncomingMessage) -> None:
        codex = self._codex()
        if not codex.state.thread_id:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("common.label.no_chat"),
            )
            return
        await self._send_to_codex(incoming.chat_id, RECAP_PROMPT, incoming.message_id)

    async def _init_project(self, incoming: IncomingMessage) -> None:
        codex = self._codex()
        if not codex.state.thread_id:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("common.label.no_chat"),
            )
            return
        project_root = codex.working_directory.resolve()
        if not project_root.is_dir() or not project_root.is_relative_to(self.config.workspace_root):
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("init.label.path_unavailable"),
            )
            return
        target = project_root / "AGENTS.md"
        if target.exists() or target.is_symlink():
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("init.label.already_exists", path=target),
            )
            return
        await self._send_to_codex(
            incoming.chat_id,
            init_prompt(str(project_root)),
            incoming.message_id,
        )

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
            self.processes_picker = picker

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
            self.processes_picker = None
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
        picker = self.processes_picker
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

    def _permissions_picker_text(self) -> str:
        codex = self._codex()
        mode = codex.permission_mode
        managed = True
        if mode is None and codex.state.thread_id:
            managed = codex.state_store.load_thread(codex.state.thread_id).managed_by_telecodex
        if mode is None and codex.state.thread_id and not managed:
            description = self._text("permissions.values.current_unknown")
        elif mode is None and codex.config.approval_policy != "on-request":
            description = self._text(
                "permissions.values.current_default",
                policy=codex.config.approval_policy,
            )
        else:
            description = self._text(
                "permissions.values.current",
                mode=self._text(f"permissions.buttons.mode_names.{mode or 'ask'}"),
            )
        if codex.is_busy:
            description += self._text("permissions.values.next_turn")
        scope_key = "permissions.values.current_thread" if codex.state.thread_id else "permissions.values.next_thread"
        return self._text("permissions.label.prompt", scope=self._text(scope_key), current=description)

    def _permissions_picker_keyboard(self, picker: PermissionsPicker) -> dict[str, Any]:
        codex = self._codex()
        selected = codex.permission_mode
        managed = True
        if selected is None and codex.state.thread_id:
            managed = codex.state_store.load_thread(codex.state.thread_id).managed_by_telecodex
        if selected is None and managed and codex.config.approval_policy == "on-request":
            selected = "ask"
        modes = PERMISSION_MODES
        rows = []
        for mode in modes:
            label = self._text(f"permissions.buttons.mode_names.{mode}")
            label = self._selected_button_label(label, mode == selected)
            rows.append(
                [{"text": label, "callback_data": f"pm:{picker.token}:{mode}"}]
            )
        return {"inline_keyboard": rows}

    async def _handle_permissions_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        picker = self.permissions_picker
        if (
            picker is None
            or len(parts) != 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback(incoming.callback_id, self._text("permissions.label.stale"))
            return
        mode = parts[2]
        if mode not in PERMISSION_MODES:
            await self.telegram.answer_callback(incoming.callback_id, self._text("permissions.label.stale"))
            return
        try:
            self._codex().select_permission_mode(mode)
        except Exception as exc:
            self._log_failure(
                "Could not select Codex permission mode",
                exc,
                event="telegram.permissions.select_failed",
                permission_mode=mode,
            )
            await self.telegram.answer_callback(incoming.callback_id, self._text("permissions.label.save_failed"))
            return
        await self.telegram.answer_callback(
            incoming.callback_id,
            self._text("permissions.label.selection_saved"),
        )
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self._permissions_picker_text(),
            self._permissions_picker_keyboard(picker),
        )

    def _workspace_root(self) -> Path:
        return self.config.workspace_root

    async def _list_workspace_directories(self, parent: Path) -> list[Path]:
        return await self._codex().list_workspace_directories(parent)

    def _directory_picker_text(self, picker: DirectoryPicker) -> str:
        return self._text(
            "directories.label.prompt",
            path=picker.current_path,
        )

    def _directory_picker_keyboard(self, picker: DirectoryPicker) -> dict[str, Any]:
        count = page_count(len(picker.directories))
        buttons = [
            {
                "text": truncate(path.name, 64, self.texts.system("values.ellipsis")),
                "callback_data": f"np:{picker.token}:d:{index}",
            }
            for index, path in enumerate(picker.directories)
        ]
        rows = two_column_rows(buttons, picker.page)
        if picker.current_path != self._workspace_root():
            rows.append(
                [{"text": self._text("directories.buttons.up"), "callback_data": f"np:{picker.token}:u"}]
            )
        if count > 1:
            rows.append(
                navigation_row(
                    picker.page,
                    count,
                    self.texts.system("buttons.pager.previous"),
                    self.texts.system("buttons.pager.page", current="{current}", total="{total}"),
                    self.texts.system("buttons.pager.next"),
                    f"np:{picker.token}",
                )
            )
        rows.append(
            [{"text": self._text("directories.buttons.choose"), "callback_data": f"np:{picker.token}:s"}]
        )
        return keyboard(rows)

    def _chats_picker_text(self, picker: ChatsPicker) -> str:
        return self._text(
            "chats.label.prompt",
            current=self._codex().working_directory,
        )

    def _chats_picker_keyboard(self, picker: ChatsPicker) -> dict[str, Any]:
        count = page_count(len(picker.threads))
        active_thread_id = self._codex().state.thread_id
        buttons = []
        for index, thread in enumerate(picker.threads):
            title = thread.label
            suffix = thread.cwd.name or str(thread.cwd)
            if isinstance(thread.updated_at, (int, float)):
                suffix += " · " + datetime.fromtimestamp(thread.updated_at).astimezone().strftime("%d.%m %H:%M")
            label = self._selected_button_label(
                f"{title} · {suffix}",
                thread.thread_id == active_thread_id,
            )
            buttons.append(
                {
                    "text": truncate(label, 64, self.texts.system("values.ellipsis")),
                    "callback_data": f"cp:{picker.token}:v:{index}",
                }
            )
        rows = two_column_rows(buttons, picker.page)
        if count > 1:
            rows.append(
                navigation_row(
                    picker.page,
                    count,
                    self.texts.system("buttons.pager.previous"),
                    self.texts.system("buttons.pager.page", current="{current}", total="{total}"),
                    self.texts.system("buttons.pager.next"),
                    f"cp:{picker.token}",
                )
            )
        return keyboard(rows)

    def _chats_detail_text(self, thread: CodexThread, last_message: str | None) -> str:
        name = truncate(thread.name or thread.preview or self._text("chats.buttons.untitled"), 120, "…")
        directory = truncate(str(thread.cwd), 400, "…")
        if last_message is None:
            message = self._text("chats.label.preview_unavailable")
        elif last_message:
            message = truncate(last_message, TELEGRAM_TEXT_LIMIT - 900, "…")
        else:
            message = self._text("chats.label.no_ai_message")
        return self._text(
            "chats.label.detail",
            name=name,
            path=directory,
            message=message,
        )

    def _chats_detail_keyboard(self, picker: ChatsPicker) -> dict[str, Any]:
        index = picker.selected_index
        if index is None:
            return self._chats_picker_keyboard(picker)
        return keyboard(
            [
                [
                    {
                        "text": self._text("chats.buttons.choose"),
                        "callback_data": f"cp:{picker.token}:s:{index}",
                    },
                    {
                        "text": self._text("chats.buttons.copy"),
                        "callback_data": f"cp:{picker.token}:f:{index}",
                    },
                ],
                [
                    {
                        "text": self._text("common.buttons.back_to_list"),
                        "callback_data": f"cp:{picker.token}:b",
                    }
                ],
            ]
        )

    async def _show_chat_details(
        self,
        incoming: IncomingCallback,
        picker: ChatsPicker,
        index: int,
    ) -> None:
        thread = picker.threads[index]
        await self.telegram.answer_callback(incoming.callback_id, self._text("chats.label.loading"))
        try:
            last_message: str | None = await self._codex().last_assistant_message(
                thread.thread_id
            )
        except Exception as exc:
            self._log_failure(
                "Could not load selected Codex chat preview",
                exc,
                event="telegram.chats.preview_failed",
                level=logging.WARNING,
                operation="preview",
                thread_hash=log_identifier(thread.thread_id, namespace="codex-thread"),
            )
            last_message = None
        picker.selected_index = index
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self._chats_detail_text(thread, last_message),
            self._chats_detail_keyboard(picker),
        )

    def _chats_recovery_keyboard(self, picker: ChatsPicker) -> dict[str, Any]:
        index = picker.blocked_thread_index
        if index is None:
            return self._chats_picker_keyboard(picker)
        return keyboard(
            [
                [
                    {
                        "text": self._text("chats.buttons.retry"),
                        "callback_data": f"cp:{picker.token}:r:{index}",
                    },
                    {
                        "text": self._text("chats.buttons.fork"),
                        "callback_data": f"cp:{picker.token}:f:{index}",
                    },
                ],
                [
                    {
                        "text": self._text("common.buttons.back_to_list"),
                        "callback_data": f"cp:{picker.token}:b",
                    }
                ],
            ]
        )

    async def _show_active_writer_recovery(
        self,
        incoming: IncomingCallback,
        picker: ChatsPicker,
        index: int,
        *,
        fork_failed: bool = False,
    ) -> None:
        picker.blocked_thread_index = index
        picker.selected_index = None
        thread = picker.threads[index]
        text_key = "chats.label.fork_failed" if fork_failed else "chats.label.active_writer"
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self._text(text_key, label=thread.label),
            self._chats_recovery_keyboard(picker),
        )

    async def _open_chat_from_picker(
        self,
        incoming: IncomingCallback,
        picker: ChatsPicker,
        index: int,
        *,
        fork_copy: bool,
    ) -> None:
        thread = picker.threads[index]
        codex = self._codex()
        pending_turn = codex.pending_turn
        turn_active = codex.is_busy or (
            pending_turn is not None and pending_turn.phase != "delivery_pending"
        )
        if not fork_copy and thread.thread_id == codex.state.thread_id:
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("chats.label.selected", label=thread.label, path=thread.cwd),
                {"inline_keyboard": []},
            )
            picker.message_id = None
            picker.selected_index = None
            return
        if not fork_copy and turn_active:
            await self._request_thread_switch(
                incoming,
                action="select",
                path=thread.cwd,
                thread_id=thread.thread_id,
                label=thread.label,
                picker_message_id=picker.message_id,
                picker_token=picker.token,
                picker_index=index,
            )
            return
        try:
            if fork_copy:
                await self._codex().fork_thread_by_id(thread.thread_id, thread.cwd)
            else:
                await self._codex().select_thread(thread.thread_id, thread.cwd)
        except ActiveThreadWriterError:
            logger.info(
                "Selected Codex chat is still active in another session",
                extra={
                    "event": "codex.thread.active_elsewhere",
                    "thread_hash": log_identifier(thread.thread_id, namespace="codex-thread"),
                    **self._log_context_fields(),
                },
            )
            await self._show_active_writer_recovery(incoming, picker, index, fork_failed=fork_copy)
            return
        except Exception as exc:
            self._log_failure(
                "Could not open selected Codex chat",
                exc,
                event="telegram.chat.open_failed",
                operation="fork" if fork_copy else "resume",
                thread_hash=log_identifier(thread.thread_id, namespace="codex-thread"),
            )
            key = "chats.label.fork_failed" if fork_copy else "chats.label.resume_failed"
            if fork_copy:
                await self._show_active_writer_recovery(incoming, picker, index, fork_failed=True)
            else:
                await self.telegram.send_message(incoming.chat_id, self._text(key))
            return

        picker.blocked_thread_index = None
        picker.selected_index = None
        picker.message_id = None
        key = "chats.label.forked" if fork_copy else "chats.label.selected"
        shell_was_open = self.shell_session is not None
        await self.close_shell_session()
        confirmation_text = self._text(key, label=thread.label, path=thread.cwd)
        if shell_was_open:
            confirmation_text = f"{confirmation_text}\n\n{self._text('shell.label.closed_on_chat_switch')}"
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            confirmation_text,
            {"inline_keyboard": []},
        )

    async def _handle_directory_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        picker = self.directory_picker
        if (
            picker is None
            or len(parts) < 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback(incoming.callback_id, self._text("directories.label.stale"))
            return
        action = parts[2]
        if action == "p" and len(parts) == 4:
            try:
                page = int(parts[3])
            except ValueError:
                page = -1
            if page < 0 or page >= page_count(len(picker.directories)):
                await self.telegram.answer_callback(incoming.callback_id)
                return
            picker.page = page
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._directory_picker_text(picker),
                self._directory_picker_keyboard(picker),
            )
            return
        if action == "i" and len(parts) == 3:
            await self.telegram.answer_callback(incoming.callback_id)
            return
        if action == "u" and len(parts) == 3:
            root = self._workspace_root()
            if picker.current_path == root:
                await self.telegram.answer_callback(incoming.callback_id)
                return
            target = picker.current_path.parent
            try:
                target.relative_to(root)
                picker.directories = await self._list_workspace_directories(target)
            except Exception:
                await self.telegram.answer_callback(incoming.callback_id, self._text("directories.label.stale"))
                return
            picker.current_path = target
            picker.page = 0
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._directory_picker_text(picker),
                self._directory_picker_keyboard(picker),
            )
            return
        if action == "d" and len(parts) == 4:
            try:
                index = int(parts[3])
                first_index = picker.page * PAGE_SIZE
                if not first_index <= index < min(first_index + PAGE_SIZE, len(picker.directories)):
                    raise ValueError("Directory button is not on the current page")
                target = picker.directories[index]
                target.relative_to(self._workspace_root())
                directories = await self._list_workspace_directories(target)
            except (IndexError, OSError, ValueError):
                await self.telegram.answer_callback(incoming.callback_id, self._text("directories.label.stale"))
                return
            picker.current_path = target
            picker.directories = directories
            picker.page = 0
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._directory_picker_text(picker),
                self._directory_picker_keyboard(picker),
            )
            return
        if action == "s" and len(parts) == 3:
            await self.telegram.answer_callback(incoming.callback_id)
            codex = self._codex()
            pending_turn = codex.pending_turn
            if codex.is_busy or (
                pending_turn is not None and pending_turn.phase != "delivery_pending"
            ):
                await self._request_thread_switch(
                    incoming,
                    action="new",
                    path=picker.current_path,
                    picker_message_id=picker.message_id,
                    picker_token=picker.token,
                )
                return
            try:
                await codex.new_thread(picker.current_path)
            except Exception as exc:
                self._log_failure(
                    "Could not create Codex thread for selected directory",
                    exc,
                    event="telegram.directory.new_thread_failed",
                )
                await self.telegram.send_message(incoming.chat_id, self._text("directories.label.create_failed"))
                return
            shell_was_open = self.shell_session is not None
            await self.close_shell_session()
            picker.message_id = None
            text = self._text("directories.label.created", path=picker.current_path)
            if shell_was_open:
                text = f"{text}\n\n{self._text('shell.label.closed_on_chat_switch')}"
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                text,
                {"inline_keyboard": []},
            )
            return
        await self.telegram.answer_callback(incoming.callback_id, self._text("directories.label.stale"))

    async def _handle_chats_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        picker = self.chats_picker
        if (
            picker is None
            or len(parts) < 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback(incoming.callback_id, self._text("chats.label.stale"))
            return
        action = parts[2]
        if action == "p" and len(parts) == 4:
            try:
                page = int(parts[3])
            except ValueError:
                page = -1
            if page < 0 or page >= page_count(len(picker.threads)):
                await self.telegram.answer_callback(incoming.callback_id)
                return
            picker.page = page
            picker.selected_index = None
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._chats_picker_text(picker),
                self._chats_picker_keyboard(picker),
            )
            return
        if action == "i" and len(parts) == 3:
            await self.telegram.answer_callback(incoming.callback_id)
            return
        if action == "b" and len(parts) == 3:
            if picker.blocked_thread_index is not None:
                picker.blocked_thread_index = None
            elif picker.selected_index is not None:
                picker.selected_index = None
            else:
                await self.telegram.answer_callback(incoming.callback_id, self._text("chats.label.stale"))
                return
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._chats_picker_text(picker),
                self._chats_picker_keyboard(picker),
            )
            return
        if action in {"r", "f"} and len(parts) == 4:
            try:
                index = int(parts[3])
            except ValueError:
                index = -1
            if index < 0 or index >= len(picker.threads):
                await self.telegram.answer_callback(incoming.callback_id, self._text("chats.label.stale"))
                return
            if action == "r" and index != picker.blocked_thread_index:
                await self.telegram.answer_callback(incoming.callback_id, self._text("chats.label.stale"))
                return
            if action == "f" and index not in {
                picker.blocked_thread_index,
                picker.selected_index,
            }:
                await self.telegram.answer_callback(incoming.callback_id, self._text("chats.label.stale"))
                return
            await self.telegram.answer_callback(incoming.callback_id)
            await self._open_chat_from_picker(
                incoming,
                picker,
                index,
                fork_copy=action == "f",
            )
            return
        if action == "v" and len(parts) == 4:
            try:
                index = int(parts[3])
                first_index = picker.page * PAGE_SIZE
                if not first_index <= index < min(first_index + PAGE_SIZE, len(picker.threads)):
                    raise ValueError("Chat button is not on the current page")
            except (IndexError, ValueError):
                await self.telegram.answer_callback(incoming.callback_id, self._text("chats.label.stale"))
                return
            await self._show_chat_details(incoming, picker, index)
            return
        if action == "s" and len(parts) == 4:
            try:
                index = int(parts[3])
                first_index = picker.page * PAGE_SIZE
                if not first_index <= index < min(first_index + PAGE_SIZE, len(picker.threads)):
                    raise ValueError("Chat button is not on the current page")
                thread = picker.threads[index]
            except (IndexError, ValueError):
                await self.telegram.answer_callback(incoming.callback_id, self._text("chats.label.stale"))
                return
            await self.telegram.answer_callback(incoming.callback_id)
            await self._open_chat_from_picker(incoming, picker, index, fork_copy=False)
            return
        await self.telegram.answer_callback(incoming.callback_id, self._text("chats.label.stale"))

    def _language_picker_keyboard(self, picker: LanguagePicker) -> dict[str, Any]:
        selected = self._language()
        rows = [
            [
                {
                    "text": self._selected_button_label(
                        self.texts.system(f"values.language_names.{language}"),
                        language == selected,
                    ),
                    "callback_data": f"l:{picker.token}:{language}",
                }
            ]
            for language in SUPPORTED_LANGUAGES
        ]
        return {"inline_keyboard": rows}

    def _mode_picker_text(self) -> str:
        return self._text(
            "mode.label.prompt",
            current=self._text(f"mode.buttons.names.{self._selected_output_mode()}"),
        )

    def _mode_picker_keyboard(self, picker: ModePicker) -> dict[str, Any]:
        selected = self._selected_output_mode()
        rows = [
            [
                {
                    "text": self._selected_button_label(
                        self._text(f"mode.buttons.names.{mode}"),
                        mode == selected,
                    ),
                    "callback_data": f"om:{picker.token}:{mode}",
                }
            ]
            for mode in ("answers", "quiet_answers", "detailed")
        ]
        return {"inline_keyboard": rows}

    def _group_mode_name(self, mode: str) -> str:
        key = "all_messages" if mode == "all" else "reply_or_trigger"
        return self._text(f"groups.buttons.mode_names.{key}")

    def _group_mode_picker_text(self) -> str:
        mode = "all" if self._group_all_messages_to_codex() else "triggered"
        return self._text(
            "groups.label.mode_prompt",
            current=self._group_mode_name(mode),
        )

    def _group_mode_picker_keyboard(self, picker: GroupModePicker) -> dict[str, Any]:
        selected = "all" if self._group_all_messages_to_codex() else "triggered"
        rows = []
        for mode, name in (
            ("triggered", "reply_or_trigger"),
            ("all", "all_messages"),
        ):
            label = self._text(f"groups.buttons.mode_names.{name}")
            label = self._selected_button_label(label, mode == selected)
            rows.append(
                [{"text": label, "callback_data": f"gm:{picker.token}:{mode}"}]
            )
        return {"inline_keyboard": rows}

    async def _show_group_mode_picker(self, incoming: IncomingMessage) -> None:
        if not self.is_group:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("groups.label.only_group"),
                reply_to_message_id=incoming.message_id,
            )
            return
        if not self.attached:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("groups.label.not_attached"),
                reply_to_message_id=incoming.message_id,
            )
            return
        picker = GroupModePicker(token=secrets.token_urlsafe(5))
        message = await self.telegram.send_message(
            incoming.chat_id,
            self._group_mode_picker_text(),
            self._group_mode_picker_keyboard(picker),
            reply_to_message_id=incoming.message_id,
        )
        picker.message_id = message.get("message_id")
        self.group_mode_picker = picker

    async def _handle_group_mode_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        picker = self.group_mode_picker
        if (
            not self.is_group
            or not self.attached
            or picker is None
            or len(parts) != 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
            or parts[2] not in {"triggered", "all"}
            or self._set_group_all_messages_to_codex is None
        ):
            await self.telegram.answer_callback(
                incoming.callback_id,
                self._text("groups.label.mode_stale"),
            )
            return

        enabled = parts[2] == "all"
        try:
            saved = self._set_group_all_messages_to_codex(enabled)
        except Exception as exc:
            self._log_failure(
                "Could not save group message routing mode",
                exc,
                event="telegram.group.mode_save_failed",
                enabled=enabled,
            )
            await self.telegram.answer_callback(
                incoming.callback_id,
                self._text("groups.label.mode_save_failed"),
            )
            return
        if not saved:
            await self.telegram.answer_callback(
                incoming.callback_id,
                self._text("groups.label.mode_stale"),
            )
            return
        await self.telegram.answer_callback(
            incoming.callback_id,
            self._text("groups.label.mode_saved", mode=self._group_mode_name(parts[2])),
        )
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self._group_mode_picker_text(),
            self._group_mode_picker_keyboard(picker),
        )

    async def _handle_mode_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        picker = self.mode_picker
        if (
            picker is None
            or len(parts) != 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback(incoming.callback_id, self._text("mode.label.stale"))
            return
        mode = parts[2]
        if mode not in OUTPUT_MODE_CHOICES:
            await self.telegram.answer_callback(incoming.callback_id, self._text("mode.label.stale"))
            return

        try:
            self._codex().select_output_mode(mode)
        except Exception as exc:
            self._log_failure(
                "Could not select output mode",
                exc,
                event="telegram.output_mode.select_failed",
                outcome=mode,
            )
            await self.telegram.answer_callback(incoming.callback_id, self._text("mode.label.save_failed"))
            return
        await self.telegram.answer_callback(incoming.callback_id)
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self._mode_picker_text(),
            self._mode_picker_keyboard(picker),
        )

    async def _handle_skills_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        picker = self.skill_picker
        if (
            picker is None
            or len(parts) < 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback(incoming.callback_id, self._text("skills.label.stale"))
            return
        action = parts[2]
        if action == "a" and len(parts) == 4 and parts[3] in {"use", "manage"}:
            picker.action = parts[3]
            picker.page = 0
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._skills_picker_text(picker),
                self._skills_picker_keyboard(picker),
            )
            return
        if action == "b" and len(parts) == 3:
            picker.action = "menu"
            picker.page = 0
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._skills_picker_text(picker),
                self._skills_picker_keyboard(picker),
            )
            return
        if action == "n" and len(parts) == 3:
            await self.telegram.answer_callback(incoming.callback_id)
            return
        if action == "p" and len(parts) == 4:
            try:
                page = int(parts[3])
            except ValueError:
                page = -1
            if page < 0 or page >= page_count(len(picker.skills)):
                await self.telegram.answer_callback(incoming.callback_id, self._text("skills.label.stale"))
                return
            picker.page = page
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._skills_picker_text(picker),
                self._skills_picker_keyboard(picker),
            )
            return
        if action in {"use", "toggle"} and len(parts) == 4:
            try:
                index = int(parts[3])
                first_index = picker.page * PAGE_SIZE
                if not first_index <= index < min(first_index + PAGE_SIZE, len(picker.skills)):
                    raise ValueError("Skill button is not on the current page")
                skill = picker.skills[index]
            except (IndexError, ValueError):
                await self.telegram.answer_callback(incoming.callback_id, self._text("skills.label.stale"))
                return
            if action == "use":
                if not skill.enabled:
                    await self.telegram.answer_callback(incoming.callback_id, self._text("skills.label.disabled"))
                    return
                try:
                    self._codex().select_skill_for_next_turn(skill)
                except Exception as exc:
                    self._log_failure(
                        "Could not select a Codex skill",
                        exc,
                        event="telegram.skills.select_failed",
                    )
                    await self.telegram.answer_callback(incoming.callback_id, self._text("skills.label.select_failed"))
                    return
                await self.telegram.answer_callback(incoming.callback_id, self._text("skills.label.selected_toast"))
                self.skill_picker = None
                await self._edit_message(
                    incoming.chat_id,
                    incoming.message_id,
                    self._text("skills.label.selected_next", name=skill.display_name or skill.name),
                    {"inline_keyboard": []},
                )
                return
            try:
                await self._codex().set_skill_enabled(skill, not skill.enabled)
                if skill.enabled and (
                    self._codex().state.pending_skill_name == skill.name
                    and self._codex().state.pending_skill_path == skill.path
                ):
                    self._codex().clear_selected_skill()
                picker.skills = await self._codex().list_skills(force_reload=True)
            except Exception as exc:
                self._log_failure(
                    "Could not change Codex skill configuration",
                    exc,
                    event="telegram.skills.toggle_failed",
                    skill_name=skill.name,
                )
                await self.telegram.answer_callback(incoming.callback_id, self._text("skills.label.update_failed"))
                return
            await self.telegram.answer_callback(incoming.callback_id, self._text("skills.label.updated"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._skills_picker_text(picker),
                self._skills_picker_keyboard(picker),
            )
            return
        await self.telegram.answer_callback(incoming.callback_id, self._text("skills.label.stale"))

    async def _handle_memory_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        picker = self.memory_picker
        if (
            picker is None
            or len(parts) not in {3, 4}
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
            or picker.thread_id != self._codex().state.thread_id
        ):
            await self.telegram.answer_callback(incoming.callback_id, self._text("memories.label.stale"))
            return
        codex = self._codex()
        action = parts[2]
        if action == "cancel" and (picker.confirming_generation or picker.confirming_reset):
            picker.confirming_generation = False
            picker.confirming_reset = False
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if action == "reset":
            picker.confirming_reset = True
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if action == "reset_confirm" and picker.confirming_reset:
            try:
                await codex.reset_memories()
                picker.confirming_reset = False
            except Exception as exc:
                self._log_failure(
                    "Could not reset Codex memories",
                    exc,
                    event="telegram.memories.reset_failed",
                )
                await self.telegram.answer_callback(incoming.callback_id, self._text("memories.label.reset_failed"))
                return
            await self.telegram.answer_callback(incoming.callback_id, self._text("memories.label.reset_done"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if action == "generate":
            picker.confirming_generation = True
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if action == "confirm" and picker.confirming_generation:
            try:
                await codex.toggle_memory_generation()
                picker.settings = await codex.read_memory_settings()
                picker.confirming_generation = False
            except Exception as exc:
                self._log_failure(
                    "Could not change the Codex memory generation setting",
                    exc,
                    event="telegram.memories.generation_write_failed",
                )
                await self.telegram.answer_callback(incoming.callback_id, self._text("memories.label.save_failed"))
                return
            await self.telegram.answer_callback(incoming.callback_id, self._text("memories.label.saved"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if action == "enable" and not picker.settings.feature_enabled:
            try:
                await codex.enable_memories_feature()
                picker.settings = await codex.read_memory_settings()
                picker.use_memories = picker.settings.use_memories
            except Exception as exc:
                self._log_failure(
                    "Could not enable Codex memories for new chats",
                    exc,
                    event="telegram.memories.feature_enable_failed",
                )
                await self.telegram.answer_callback(incoming.callback_id, self._text("memories.label.save_failed"))
                return
            await self.telegram.answer_callback(incoming.callback_id, self._text("memories.label.feature_enabled"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if (
            action == "use"
            and len(parts) == 4
            and parts[3] in {"enabled", "disabled"}
            and picker.settings.feature_enabled
        ):
            target_mode = parts[3]
            try:
                await codex.set_memory_mode(target_mode)
                picker.use_memories = target_mode == "enabled"
            except MemoryFeatureUnavailableError:
                picker.settings = await codex.read_memory_settings()
                await self.telegram.answer_callback(incoming.callback_id, self._text("memories.label.feature_disabled"))
                await self._edit_message(
                    incoming.chat_id,
                    incoming.message_id,
                    self._memory_picker_text(picker),
                    self._memory_picker_keyboard(picker),
                )
                return
            except Exception as exc:
                self._log_failure(
                    "Could not change thread memory mode",
                    exc,
                    event="telegram.memories.thread_mode_failed",
                )
                await self.telegram.answer_callback(incoming.callback_id, self._text("memories.label.save_failed"))
                return
            await self.telegram.answer_callback(incoming.callback_id, self._text("memories.label.saved"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        await self.telegram.answer_callback(incoming.callback_id, self._text("memories.label.stale"))

    async def _handle_language_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        picker = self.language_picker
        if (
            picker is None
            or len(parts) != 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback(
                incoming.callback_id,
                self.texts.system("label.language_picker_stale"),
            )
            return
        language = parts[2]
        if language not in SUPPORTED_LANGUAGES:
            await self.telegram.answer_callback(
                incoming.callback_id,
                self.texts.system("label.language_picker_invalid"),
            )
            return

        self._codex().select_language(language)
        await self.telegram.answer_callback(
            incoming.callback_id,
            self.texts.system(
                "label.language_picker_saved",
                language=self.texts.system(f"values.language_names.{language}"),
            ),
        )
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self.texts.system("label.language_picker_prompt"),
            self._language_picker_keyboard(picker),
        )

    async def _handle_model_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        picker = self.model_picker
        if (
            picker is None
            or len(parts) < 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback(incoming.callback_id, self._text("models.label.expired"))
            return

        action = parts[2]
        if picker.effort_model_index is not None:
            model = picker.models[picker.effort_model_index]
            if action == "b" and len(parts) == 3:
                picker.effort_model_index = None
                await self.telegram.answer_callback(incoming.callback_id)
                await self._edit_message(
                    incoming.chat_id,
                    incoming.message_id,
                    self._model_picker_text(picker),
                    self._model_picker_keyboard(picker),
                )
                return
            if action == "d" and len(parts) == 3:
                effort = model.default_reasoning_effort
            elif action == "e" and len(parts) == 4:
                try:
                    effort_index = int(parts[3])
                    efforts = self._model_effort_options(model)
                    if not 0 <= effort_index < len(efforts):
                        raise ValueError("Effort button is not valid")
                    effort = efforts[effort_index]
                except ValueError:
                    await self.telegram.answer_callback(incoming.callback_id, self._text("common.label.stale_button"))
                    return
            else:
                await self.telegram.answer_callback(incoming.callback_id, self._text("common.label.stale_button"))
                return

            try:
                await self._codex().select_model(model.model, effort)
            except Exception as exc:
                self._log_failure(
                    "Could not select Codex model and effort",
                    exc,
                    event="telegram.model.select_failed",
                    model=model.model,
                    effort=effort,
                )
                await self.telegram.answer_callback(
                    incoming.callback_id,
                    self._text("models.label.selection_failed"),
                )
                return
            picker.current_model = model.model
            picker.current_effort = effort
            picker.effort_model_index = None
            await self.telegram.answer_callback(
                incoming.callback_id,
                self._text("models.label.selection_saved"),
            )
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._model_picker_text(picker),
                self._model_picker_keyboard(picker),
            )
            return

        if action == "i" and len(parts) == 3:
            await self.telegram.answer_callback(incoming.callback_id)
            return

        if action == "p" and len(parts) == 4:
            try:
                page = int(parts[3])
            except ValueError:
                page = -1
            page_count = self._model_page_count(picker)
            if page < 0 or page >= page_count:
                await self.telegram.answer_callback(incoming.callback_id, self._text("models.label.invalid_page"))
                return
            picker.page = page
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._model_picker_text(picker),
                self._model_picker_keyboard(picker),
            )
            return

        if action == "r" and len(parts) == 3:
            try:
                await self._codex().select_model(None, None)
                picker.current_model, picker.current_effort = await self._codex().current_model_settings(
                    picker.models
                )
            except Exception as exc:
                self._log_failure(
                    "Could not reset Codex model and effort",
                    exc,
                    event="telegram.model.reset_failed",
                )
                await self.telegram.answer_callback(
                    incoming.callback_id,
                    self._text("models.label.selection_failed"),
                )
                return
            await self.telegram.answer_callback(
                incoming.callback_id,
                self._text("models.label.selection_saved"),
            )
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._model_picker_text(picker),
                self._model_picker_keyboard(picker),
            )
            return
        if action == "s" and len(parts) == 4:
            try:
                model_index = int(parts[3])
            except ValueError:
                model_index = -1
            if model_index < 0 or model_index >= len(picker.models):
                await self.telegram.answer_callback(incoming.callback_id, self._text("models.label.unknown"))
                return
            picker.effort_model_index = model_index
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._model_picker_text(picker),
                self._model_picker_keyboard(picker),
            )
            return
        else:
            await self.telegram.answer_callback(incoming.callback_id, self._text("common.label.stale_button"))
            return

    @staticmethod
    def _model_page_count(picker: ModelPicker) -> int:
        return max(1, (len(picker.models) + 7) // 8)

    def _model_picker_text(self, picker: ModelPicker) -> str:
        codex = self._codex()
        if picker.effort_model_index is not None:
            model = picker.models[picker.effort_model_index]
            default_effort = (
                self._effort_label(model.default_reasoning_effort)
                if model.default_reasoning_effort
                else self._text("models.values.codex_default_effort")
            )
            return self._text(
                "models.label.effort_prompt",
                model=model.display_name,
                default_effort=default_effort,
            )
        current_model = picker.current_model
        model_info = next((model for model in picker.models if model.model == current_model), None)
        current = model_info.display_name if model_info else current_model or self._text("common.values.default_codex_model")
        if picker.current_effort and (
            model_info is None
            or picker.current_effort != model_info.default_reasoning_effort
        ):
            current += self._text(
                "models.values.current_effort",
                effort=self._effort_label(picker.current_effort),
            )
        elif model_info and model_info.default_reasoning_effort:
            current += self._text(
                "models.values.current_effort_default",
                effort=self._effort_label(model_info.default_reasoning_effort),
            )
        elif picker.current_effort:
            current += self._text(
                "models.values.current_effort",
                effort=self._effort_label(picker.current_effort),
            )
        message = self._text("models.label.prompt", current=current)
        if codex.is_busy:
            message += self._text("models.values.busy_notice")
        return message

    def _model_picker_keyboard(self, picker: ModelPicker) -> dict[str, Any]:
        selected_model = picker.current_model
        if picker.effort_model_index is not None:
            model = picker.models[picker.effort_model_index]
            rows: list[list[dict[str, str]]] = []
            default_effort = model.default_reasoning_effort
            current_effort = picker.current_effort if selected_model == model.model else None
            if default_effort:
                default_label = self._text(
                    "models.buttons.default_effort",
                    effort=self._effort_label(default_effort),
                )
                default_label = self._selected_button_label(
                    default_label,
                    selected_model == model.model and current_effort in {None, default_effort},
                )
                rows.append(
                    [{"text": default_label, "callback_data": f"m:{picker.token}:d"}]
                )
            efforts = self._model_effort_options(model)
            effort_buttons = []
            for effort_index, effort in enumerate(efforts):
                label = self._selected_button_label(
                    self._effort_label(effort),
                    selected_model == model.model and current_effort == effort,
                )
                effort_buttons.append(
                    {"text": label, "callback_data": f"m:{picker.token}:e:{effort_index}"}
                )
            rows.extend(two_column_rows(effort_buttons, 0))
            rows.append([{"text": self._text("models.buttons.back"), "callback_data": f"m:{picker.token}:b"}])
            return {"inline_keyboard": rows}

        start = picker.page * 8
        end = min(start + 8, len(picker.models))
        model_buttons = []
        for model_index in range(start, end):
            model = picker.models[model_index]
            label = self._selected_button_label(
                model.display_name,
                model.model == selected_model,
            )
            model_buttons.append(
                {"text": label, "callback_data": f"m:{picker.token}:s:{model_index}"}
            )
        rows = two_column_rows(model_buttons, 0)

        rows.append([{"text": self._text("models.buttons.reset"), "callback_data": f"m:{picker.token}:r"}])
        page_count = self._model_page_count(picker)
        if page_count > 1:
            rows.append(
                navigation_row(
                    picker.page,
                    page_count,
                    self.texts.system("buttons.pager.previous"),
                    self.texts.system("buttons.pager.page", current="{current}", total="{total}"),
                    self.texts.system("buttons.pager.next"),
                    f"m:{picker.token}",
                )
            )
        return {"inline_keyboard": rows}

    @staticmethod
    def _model_effort_options(model: CodexModel) -> tuple[str, ...]:
        return tuple(
            effort
            for effort in model.supported_reasoning_efforts
            if effort != model.default_reasoning_effort
        )

    def _effort_label(self, effort: str) -> str:
        labels = {
            "none": "models.buttons.effort_names.none",
            "minimal": "models.buttons.effort_names.minimal",
            "low": "models.buttons.effort_names.low",
            "medium": "models.buttons.effort_names.medium",
            "high": "models.buttons.effort_names.high",
            "xhigh": "models.buttons.effort_names.xhigh",
            "max": "models.buttons.effort_names.max",
        }
        key = labels.get(effort)
        return self._text(key) if key else effort

    async def _on_codex_prompt(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.is_group and not self.attached:
            if method == "item/tool/requestUserInput":
                return {"answers": {}}
            if method == "item/permissions/requestApproval":
                return {"permissions": {}}
            if method == "mcpServer/elicitation/request":
                return {"action": "decline"}
            return {"decision": "decline"}
        if method == "mcpServer/elicitation/request":
            return await self._ask_mcp_elicitation(params)
        if method == "item/tool/requestUserInput":
            return await self._ask_user_input(params)
        if method == "item/permissions/requestApproval":
            return await self._ask_permission(params)
        decision = await self._ask_approval(method, params)
        return {"decision": decision}







    async def _ask_permission(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.is_group and not self.attached:
            return {"permissions": {}}
        requested = params.get("permissions", {})
        if not isinstance(requested, dict):
            return {"permissions": {}}
        network = requested.get("network")
        filesystem = requested.get("fileSystem")
        network_requested = isinstance(network, dict) and network.get("enabled")
        filesystem_requested = isinstance(filesystem, dict) and any(
            filesystem.get(key) for key in ("read", "write", "entries")
        )
        if filesystem_requested and not network_requested:
            await self.telegram.send_message(
                self.chat_id,
                self._text("permissions.label.filesystem_denied"),
            )
            return {"permissions": {}}
        if not network_requested:
            return {"permissions": {}}

        token = secrets.token_urlsafe(7)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        reason = self._truncate(
            str(params.get("reason") or self._text("permissions.values.default_reason")),
            500,
        )
        text_key = (
            "permissions.label.network_request_with_filesystem"
            if filesystem_requested
            else "permissions.label.network_request"
        )
        await self._pause_typing()
        message = await self.telegram.send_message(
            self.chat_id,
            self._text(
                text_key,
                filesystem_note=self._text("permissions.values.filesystem_note"),
                reason=reason,
            ),
            {
                "inline_keyboard": two_column_rows(
                    [
                        {"text": self._text("approval.label.allow_task"), "callback_data": f"ap:{token}:once"},
                        {"text": self._text("approval.label.allow_session"), "callback_data": f"ap:{token}:session"},
                    ],
                    0,
                )
                + [[{"text": self._text("common.buttons.decline"), "callback_data": f"ap:{token}:deny"}]]
            },
        )
        if self.is_group and not self.attached:
            return {"permissions": {}}
        self._codex().add_pending_prompt(message["message_id"])
        self.pending_approvals[token] = PendingApproval(
            future=future,
            message_id=message["message_id"],
            chat_id=self.chat_id,
        )
        try:
            choice = await asyncio.wait_for(future, timeout=900)
        except asyncio.TimeoutError:
            choice = "decline"
        finally:
            self.pending_approvals.pop(token, None)
            if not self._shutdown_in_progress:
                self._codex().remove_pending_prompt(message["message_id"])
            self._resume_typing()

        grants: dict[str, Any] = {}
        if choice in {"accept", "acceptForSession"}:
            if isinstance(network, dict) and network.get("enabled"):
                grants["network"] = {"enabled": True}
        result: dict[str, Any] = {"permissions": grants}
        if choice == "acceptForSession":
            result["scope"] = "session"
        return result

    async def _ask_approval(self, method: str, params: dict[str, Any]) -> str:
        token = secrets.token_urlsafe(7)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        reason = params.get("reason")
        network = params.get("networkApprovalContext")
        command = params.get("command")
        if isinstance(network, dict):
            kind = self._text("approval.values.network_kind")
            detail = self._text(
                "approval.values.network_detail",
                protocol=network.get("protocol") or "network",
                host=network.get("host") or "unknown host",
            )
        else:
            kind = self._text(
                "approval.values.filesystem_kind"
                if method == "item/fileChange/requestApproval"
                else "approval.values.command_kind"
            )
            if isinstance(command, str) and command:
                detail = self._text("approval.values.command_detail", command=self._truncate(command, 500))
            else:
                detail = str(reason or self._text("approval.values.default_reason"))
        text = self._text("approval.label.request", kind=kind, detail=self._truncate(detail, 700))
        keyboard = {
            "inline_keyboard": two_column_rows(
                [
                    {"text": self._text("approval.label.allow_once"), "callback_data": f"ap:{token}:once"},
                    {"text": self._text("approval.label.allow_session"), "callback_data": f"ap:{token}:session"},
                ],
                0,
            )
            + [[{"text": self._text("common.buttons.decline"), "callback_data": f"ap:{token}:deny"}]]
        }
        await self._pause_typing()
        message = await self.telegram.send_message(self.chat_id, text, keyboard)
        if self.is_group and not self.attached:
            return "cancel"
        self._codex().add_pending_prompt(message["message_id"])
        pending = PendingApproval(
            future=future,
            message_id=message["message_id"],
            chat_id=self.chat_id,
        )
        self.pending_approvals[token] = pending
        try:
            return await asyncio.wait_for(future, timeout=900)
        except asyncio.TimeoutError:
            return "cancel"
        finally:
            self.pending_approvals.pop(token, None)
            if not self._shutdown_in_progress:
                self._codex().remove_pending_prompt(message["message_id"])
            self._resume_typing()

    async def _ask_user_input(self, params: dict[str, Any]) -> dict[str, Any]:
        questions = params.get("questions", [])
        if not isinstance(questions, list) or not questions:
            return {"answers": {}}
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        prompt_lines = [self._text("user_input.label.title")]
        for index, question in enumerate(questions, start=1):
            if isinstance(question, dict):
                question_text = question.get("question")
                if not isinstance(question_text, str) or not question_text:
                    question_text = self._text("user_input.values.question_fallback")
                prompt_lines.append(
                    self._text("user_input.values.numbered_question", index=index, question=question_text)
                )
                options = question.get("options")
                if isinstance(options, list):
                    labels = [str(item.get("label", "")) for item in options if isinstance(item, dict)]
                    if labels:
                        prompt_lines.append(self._text("user_input.values.options", options=", ".join(labels)))
        prompt_lines.append(self._text("user_input.label.instructions"))
        await self._pause_typing()
        message = await self.telegram.send_message(
            self.chat_id,
            "\n".join(prompt_lines),
        )
        if self.is_group and not self.attached:
            return {"answers": {}}
        self._codex().add_pending_prompt(message["message_id"])
        self.pending_user_input = PendingUserInput(future=future, questions=questions)
        try:
            return await asyncio.wait_for(future, timeout=900)
        except asyncio.TimeoutError:
            await self._edit_message(
                self.chat_id,
                message["message_id"],
                self._text("user_input.label.timeout"),
            )
            return {"answers": {}}
        finally:
            if self.pending_user_input and self.pending_user_input.future is future:
                self.pending_user_input = None
            if not self._shutdown_in_progress:
                self._codex().remove_pending_prompt(message["message_id"])
            self._resume_typing()

    async def _resolve_user_input(self, text: str) -> None:
        pending = self.pending_user_input
        if pending is None or pending.future.done():
            return
        lines = [line.strip() for line in text.splitlines()]
        answers: dict[str, dict[str, list[str]]] = {}
        for index, question in enumerate(pending.questions):
            if not isinstance(question, dict) or not isinstance(question.get("id"), str):
                continue
            if len(pending.questions) == 1:
                answer = text.strip()
            else:
                answer = lines[index] if index < len(lines) else ""
            answers[question["id"]] = {"answers": [answer] if answer else []}
        pending.future.set_result({"answers": answers})
        self.pending_user_input = None
        if self._shows_detailed_activity():
            await self.telegram.send_message(self.chat_id, self._text("user_input.label.delivered"))

    async def _on_codex_status(
        self,
        text_key: str,
        raw_command: object | None = None,
        item_id: str | None = None,
    ) -> None:
        if self.is_group and not self.attached:
            return
        if text_key == "status.goal_continuing":
            if type(raw_command) is int and raw_command > 0:
                self._turn_prompt_message_id = raw_command
            self._start_typing()
            return
        if text_key == "status.compaction_queued":
            if not self._codex().manual_compaction_waiting_for_idle:
                return
            await self.telegram.send_message(
                self.chat_id,
                self._text("thread_commands.label.compact_queued"),
            )
            return
        if text_key in {"status.label.compaction_started", "status.label.compaction_completed"}:
            await self._send_codex_activity(
                self._text(text_key),
                disable_notification=self._quiet_progress_notifications(),
            )
            return
        if text_key in {"status.label.manual_compaction_completed", "status.label.compaction_failed"}:
            await self._finish_typing()
            await self._send_codex_activity(self._text(text_key))
            return
        mcp_activity_keys = {
            "detailed_log.values.mcp_started",
            "detailed_log.values.mcp_completed",
            "detailed_log.values.mcp_failed",
            "detailed_log.values.mcp_server_status",
        }
        if text_key in mcp_activity_keys:
            if not self._shows_detailed_activity():
                return
            details = raw_command if isinstance(raw_command, dict) else {}
            server = details.get("server") or details.get("name")
            tool = details.get("tool")
            if not isinstance(server, str):
                return
            if text_key == "detailed_log.values.mcp_server_status":
                status = details.get("status")
                translated_status = self._text(
                    f"mcp.values.runtime.{status}"
                    if isinstance(status, str) and status in {
                        "notStarted", "starting", "connected", "authenticationRequired",
                        "failed", "cancelled", "disabled",
                    }
                    else "common.values.no_data"
                )
                detail = f"{server}: {translated_status}"
            else:
                detail = f"{server}/{tool}" if isinstance(tool, str) else server
                duration = details.get("durationMs")
                if text_key == "detailed_log.values.mcp_completed" and isinstance(duration, (int, float)):
                    detail += self._text(
                        "detailed_log.values.mcp_duration",
                        duration=self._format_command_duration(max(0, duration)),
                    )
            await self._send_codex_activity_lines([(self._text(text_key), detail)])
            return
        if text_key in {"status.label.mcp_oauth_success", "status.label.mcp_oauth_failed"}:
            name = raw_command if isinstance(raw_command, str) else ""
            await self._send_codex_activity(self._text(text_key, name=name))
            if text_key == "status.label.mcp_oauth_success":
                await self._reload_mcp_after_change()
                if self.mcp_picker is not None:
                    await self._refresh_mcp_picker(self.mcp_picker)
            return
        activity_keys = {
            "detailed_log.values.skill_attached",
            "detailed_log.values.review_started",
            "detailed_log.values.review_completed",
            "detailed_log.values.image_generation_started",
            "detailed_log.values.image_generation_completed",
            "detailed_log.values.image_generation_failed",
            "detailed_log.values.subagent_started",
            "detailed_log.values.subagent_completed",
            "detailed_log.values.subagent_interrupted",
            "detailed_log.values.subagent_failed",
        }
        if text_key in activity_keys:
            if not self._shows_detailed_activity():
                return
            detail = ""
            if text_key == "detailed_log.values.skill_attached" and isinstance(raw_command, str):
                detail = " ".join(raw_command.split())[:240]
            elif text_key.startswith("detailed_log.values.image_generation_") and isinstance(raw_command, dict):
                if text_key == "detailed_log.values.image_generation_completed":
                    detail = image_saved_path(raw_command, self._codex().working_directory)
                elif (
                    text_key == "detailed_log.values.image_generation_failed"
                    and raw_command.get("failureType") == "usageLimitExceeded"
                ):
                    detail = self._text("detailed_log.values.image_generation_limit")
            elif text_key.startswith("detailed_log.values.subagent_") and isinstance(raw_command, dict):
                agent_path = raw_command.get("agentPath")
                if isinstance(agent_path, str):
                    detail = " ".join(agent_path.split())[:240]
            await self._send_codex_activity_lines([(self._text(text_key), detail)])
            return
        if text_key in {"detailed_log.values.web_search_started", "detailed_log.values.web_search_completed"}:
            if not self._shows_detailed_activity():
                return
            detail = web_search_detail(raw_command)
            if detail:
                await self._send_codex_activity_lines([(self._text(text_key), detail)])
            elif text_key == "detailed_log.values.web_search_started":
                await self._send_codex_activity_lines(
                    [(self._text("detailed_log.label.web_search_started_generic"), "")]
                )
            return
        if text_key == "status.file_change":
            if not self._shows_detailed_activity():
                return
            groups = file_change_groups(raw_command)
            item_status = raw_command.get("status") if isinstance(raw_command, dict) else None
            if item_status == "completed":
                lines = [
                    (self._text(key), "; ".join(groups[kind]))
                    for kind, key in (
                        ("add", "detailed_log.values.file_added"),
                        ("update", "detailed_log.values.file_edited"),
                        ("delete", "detailed_log.values.file_deleted"),
                    )
                    if groups[kind]
                ]
            else:
                files = "; ".join(
                    path
                    for kind in ("add", "update", "delete")
                    for path in groups[kind]
                )
                if not files:
                    return
                key = "detailed_log.values.file_edit_declined" if item_status == "declined" else "detailed_log.values.file_edit_failed"
                lines = [(self._text(key), files)]
            if lines:
                await self._send_codex_activity_lines(lines)
            return

        command = format_command(raw_command)
        if command:
            async with self._codex_activity_lock:
                message_id = None
                if self._shows_detailed_activity():
                    message = await self._send_codex_activity_lines_locked(
                        [(self._text("detailed_log.values.command_started"), command)],
                        suffix_key="detailed_log.values.command_truncated",
                    )
                    result_message_id = message.get("message_id")
                    if isinstance(result_message_id, int):
                        message_id = result_message_id
                if isinstance(item_id, str):
                    self._command_messages[item_id] = RunningCommand(
                        message_id=message_id,
                        started_at=monotonic(),
                        command=command,
                    )
            return
        if not self._shows_detailed_activity():
            return
        await self._send_codex_activity(self._text(text_key))

    async def _on_codex_command_completed(self, item: dict[str, Any]) -> None:
        if self.is_group and not self.attached:
            return
        item_id = item.get("id")
        async with self._codex_activity_lock:
            running = self._command_messages.pop(item_id, None) if isinstance(item_id, str) else None
            if not self._shows_detailed_activity():
                return
            command = format_command(item.get("command")) or (
                running.command if running else self._text("detailed_log.values.command_unknown")
            )
            status = item.get("status")
            exit_code = item.get("exitCode")
            valid_exit_code = isinstance(exit_code, int) and not isinstance(exit_code, bool)
            if status == "declined":
                state = self._text("detailed_log.values.command_declined")
            elif status == "failed" or (valid_exit_code and exit_code != 0):
                state = self._text("detailed_log.values.command_failed")
            else:
                state = self._text("detailed_log.values.command_succeeded")

            duration_ms = item.get("durationMs")
            if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
                duration = self._format_command_duration(max(0, duration_ms))
            elif running:
                elapsed_ms = max(0, int((monotonic() - running.started_at) * 1000))
                duration = self._format_command_duration(elapsed_ms)
            else:
                duration = self._text("detailed_log.values.command_duration_unknown")

            output = item.get("aggregatedOutput")
            if not isinstance(output, str) or not output.strip():
                output = self._text("detailed_log.values.command_no_output")
            is_linked = running is not None and running.message_id is not None
            header_key = (
                "detailed_log.label.command_result_header"
                if is_linked
                else "detailed_log.label.command_result_unlinked_header"
            )
            header = self._text(
                header_key,
                state=state,
                command=command,
                duration=duration,
                exit_code=(str(exit_code) if valid_exit_code else self._text("detailed_log.values.command_exit_code_unknown")),
            )
            await self._send_command_result_locked(
                header,
                output.strip(),
                reply_to_message_id=running.message_id if is_linked else None,
            )
            actions = explored_action_summaries(
                item.get("commandActions"),
                item.get("cwd"),
                render=self._text,
            )
            if status == "completed" and actions:
                await self._send_codex_activity_lines_locked(
                    [(self._text("detailed_log.values.explored"), "; ".join(actions))],
                    reply_to_message_id=running.message_id if is_linked else None,
                )

    async def _send_command_result_locked(
        self,
        header: str,
        output: str,
        reply_to_message_id: int | None,
    ) -> None:
        truncated_suffix = ""
        message_limit = GROUP_ACTIVITY_MESSAGE_LIMIT if self.is_group else TELEGRAM_TEXT_LIMIT
        if len(header) + 1 + len(output) > message_limit:
            truncated_suffix = self._text("format.values.truncated")
            header_limit = max(0, message_limit - len(truncated_suffix) - 2)
            if len(header) > header_limit:
                header = truncate(
                    header,
                    header_limit,
                    self._text("format.values.truncated").lstrip("\n"),
                )
            output_limit = max(0, message_limit - len(header) - 1 - len(truncated_suffix))
            output = output[:output_limit].rstrip()
        text = f"{header}\n{output}{truncated_suffix}"
        entities = []
        if output:
            entities.append(
                {
                    "type": "pre",
                    "offset": utf16_length(header + "\n"),
                    "length": utf16_length(output),
                }
            )
        if self.is_group:
            await self._queue_group_activity(text, entities, reply_to_message_id)
        elif self.attached:
            await self.telegram.send_message(
                self.chat_id,
                text,
                reply_to_message_id=reply_to_message_id,
                entities=entities,
            )
        self._notify_typing_cleared()

    async def _send_codex_activity(
        self,
        text: str,
        suffix_key: str = "format.values.truncated",
        reply_to_message_id: int | None = None,
        *,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        async with self._codex_activity_lock:
            return await self._send_codex_activity_locked(
                text,
                suffix_key=suffix_key,
                reply_to_message_id=reply_to_message_id,
                disable_notification=disable_notification,
            )

    async def send_recovery_notice(self, key: str) -> None:
        """Deliver a one-off health notice without exposing transport internals."""
        if self.is_group and not self.attached:
            return
        await self._send_codex_activity(
            self._text(key),
            reply_to_message_id=self._turn_prompt_message_id,
            disable_notification=self._quiet_progress_notifications(),
        )

    async def send_recovery_switch_offer(self) -> None:
        """Offer a manual switch after a quiet turn has been verified active."""
        if self.is_group and not self.attached:
            return
        codex = self._codex()
        pending = codex.pending_turn
        thread_id = pending.thread_id if pending is not None else codex.state.thread_id
        if pending is None or not thread_id:
            return
        self.pending_stale_turn_actions.clear()
        token = secrets.token_urlsafe(5)
        message = await self.telegram.send_message(
            self.chat_id,
            self._text("recovery.label.quiet_turn"),
            {
                "inline_keyboard": [
                    [
                        {
                            "text": self._text("recovery.buttons.wait_button"),
                            "callback_data": f"rc:{token}:wait",
                        }
                    ],
                    [
                        {
                            "text": self._text("recovery.buttons.switch_button"),
                            "callback_data": f"rc:{token}:switch",
                        }
                    ],
                ]
            },
            reply_to_message_id=self._turn_prompt_message_id,
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            self.pending_stale_turn_actions[token] = PendingStaleTurnAction(
                message_id=message_id,
                thread_id=thread_id,
                turn_id=pending.turn_id,
            )

    async def _request_thread_switch(
        self,
        incoming: IncomingCallback,
        *,
        action: str,
        path: Path | None = None,
        thread_id: str | None = None,
        label: str | None = None,
        picker_message_id: int | None = None,
        picker_token: str | None = None,
        picker_index: int | None = None,
    ) -> None:
        token = secrets.token_urlsafe(5)
        if action == "new":
            text = self._text("thread_switch.label.confirm_new", path=path)
        else:
            text = self._text("thread_switch.label.confirm_chat", label=label)
        message = await self.telegram.send_message(
            incoming.chat_id,
            text,
            {
                "inline_keyboard": [
                    [
                        {
                            "text": self._text("thread_switch.buttons.confirm_button"),
                            "callback_data": f"sw:{token}:yes",
                        }
                    ],
                    [
                        {
                            "text": self._text("thread_switch.buttons.cancel_button"),
                            "callback_data": f"sw:{token}:no",
                        }
                    ],
                ]
            },
            reply_to_message_id=incoming.message_id,
        )
        message_id = message.get("message_id")
        if not isinstance(message_id, int):
            return
        self.pending_thread_switch = PendingThreadSwitch(
            token=token,
            message_id=message_id,
            action=action,
            path=path,
            thread_id=thread_id,
            label=label,
            picker_message_id=picker_message_id,
            picker_token=picker_token,
            picker_index=picker_index,
            source_thread_id=self._codex().state.thread_id,
        )

    async def _handle_thread_switch_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        pending = self.pending_thread_switch
        if (
            pending is None
            or len(parts) != 3
            or parts[1] != pending.token
            or incoming.message_id != pending.message_id
        ):
            await self.telegram.answer_callback(incoming.callback_id, self._text("thread_switch.label.stale"))
            return
        if parts[2] == "no":
            self.pending_thread_switch = None
            await self.telegram.answer_callback(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("thread_switch.label.cancelled"),
                {"inline_keyboard": []},
            )
            return
        if parts[2] != "yes":
            await self.telegram.answer_callback(incoming.callback_id, self._text("thread_switch.label.stale"))
            return

        codex = self._codex()
        if codex.state.thread_id != pending.source_thread_id:
            self.pending_thread_switch = None
            await self.telegram.answer_callback(incoming.callback_id, self._text("thread_switch.label.stale"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("thread_switch.label.stale"),
                {"inline_keyboard": []},
            )
            return

        self.pending_thread_switch = None
        await self.telegram.answer_callback(incoming.callback_id)
        try:
            if codex.is_busy or codex.pending_turn is not None:
                await codex.interrupt_and_abandon_for_switch()
            if pending.action == "new":
                if pending.path is None:
                    raise RuntimeError("A new-thread switch is missing its target directory")
                await codex.new_thread(pending.path, force=True)
            else:
                if pending.thread_id is None:
                    raise RuntimeError("A chat switch is missing its target thread")
                target_path = pending.path or codex.working_directory
                await codex.select_thread(pending.thread_id, target_path, force=True)
        except ActiveThreadWriterError:
            if (
                pending.action == "select"
                and self.chats_picker is not None
                and pending.picker_token == self.chats_picker.token
                and pending.picker_index is not None
            ):
                recovery_incoming = replace(
                    incoming,
                    message_id=pending.picker_message_id or incoming.message_id,
                )
                await self._show_active_writer_recovery(
                    recovery_incoming,
                    self.chats_picker,
                    pending.picker_index,
                )
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("thread_switch.label.failed"),
                {"inline_keyboard": []},
            )
            return
        except Exception as exc:
            self._log_failure(
                "Could not complete a confirmed Codex chat switch",
                exc,
                event="telegram.chat.confirmed_switch_failed",
                operation=pending.action,
            )
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("thread_switch.label.failed"),
                {"inline_keyboard": []},
            )
            return

        shell_was_open = self.shell_session is not None
        await self.close_shell_session()
        if pending.action == "new":
            if (
                self.directory_picker is not None
                and self.directory_picker.token == pending.picker_token
            ):
                picker = self.directory_picker
                picker.message_id = None
                await self._edit_message(
                    incoming.chat_id,
                    pending.picker_message_id or incoming.message_id,
                    self._text("directories.label.created", path=pending.path),
                    {"inline_keyboard": []},
                )
            completion = self._text("thread_switch.label.new_completed", path=pending.path)
        else:
            thread = next(
                (
                    item
                    for item in (self.chats_picker.threads if self.chats_picker else [])
                    if item.thread_id == pending.thread_id
                ),
                None,
            )
            path = pending.path or (thread.cwd if thread is not None else codex.working_directory)
            label = thread.label if thread is not None else (pending.label or "")
            if (
                self.chats_picker is not None
                and self.chats_picker.token == pending.picker_token
            ):
                self.chats_picker.message_id = None
                await self._edit_message(
                    incoming.chat_id,
                    pending.picker_message_id or incoming.message_id,
                    self._text("chats.label.selected", label=label, path=path),
                    {"inline_keyboard": []},
                )
            completion = self._text("thread_switch.label.chat_completed", label=label, path=path)
        if shell_was_open:
            completion = f"{completion}\n\n{self._text('shell.label.closed_on_chat_switch')}"
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            completion,
            {"inline_keyboard": []},
        )

    async def _handle_stale_turn_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        if len(parts) != 3:
            await self.telegram.answer_callback(incoming.callback_id, self._text("thread_switch.label.stale"))
            return
        token, action = parts[1], parts[2]
        notice = self.pending_stale_turn_actions.get(token)
        codex = self._codex()
        pending = codex.pending_turn
        if (
            notice is None
            or notice.message_id != incoming.message_id
            or pending is None
            or pending.thread_id != notice.thread_id
            or pending.turn_id != notice.turn_id
        ):
            self.pending_stale_turn_actions.pop(token, None)
            await self.telegram.answer_callback(incoming.callback_id, self._text("thread_switch.label.stale"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("thread_switch.label.stale"),
                {"inline_keyboard": []},
            )
            return
        if action not in {"wait", "switch"}:
            await self.telegram.answer_callback(incoming.callback_id, self._text("thread_switch.label.stale"))
            return
        self.pending_stale_turn_actions.pop(token, None)
        await self.telegram.answer_callback(incoming.callback_id)
        if action == "wait":
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("recovery.label.waiting"),
                {"inline_keyboard": []},
            )
            return
        try:
            await codex.interrupt_and_abandon_for_switch()
            target = codex.working_directory
            await codex.new_thread(target, force=True)
        except Exception as exc:
            self._log_failure(
                "Could not switch away from a quiet Codex turn",
                exc,
                event="telegram.recovery.switch_failed",
            )
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._text("thread_switch.label.failed"),
                {"inline_keyboard": []},
            )
            return
        shell_was_open = self.shell_session is not None
        await self.close_shell_session()
        text = self._text("recovery.label.switched", path=target)
        if shell_was_open:
            text = f"{text}\n\n{self._text('shell.label.closed_on_chat_switch')}"
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            text,
            {"inline_keyboard": []},
        )

    async def _send_codex_activity_locked(
        self,
        text: str,
        suffix_key: str = "format.values.truncated",
        reply_to_message_id: int | None = None,
        *,
        priority: bool = False,
        batchable: bool = True,
        wait_for_delivery: bool = False,
        disable_notification: bool = False,
        rich_markdown: bool = False,
    ) -> dict[str, Any]:
        message_limit = GROUP_ACTIVITY_MESSAGE_LIMIT if self.is_group else TELEGRAM_TEXT_LIMIT
        body = self._truncate(text, message_limit, suffix_key)
        if self.is_group and not self.attached:
            return {}
        if self.is_group:
            return await self._queue_group_activity(
                body,
                [],
                reply_to_message_id,
                batchable=batchable and not priority,
                wait_for_delivery=wait_for_delivery or priority,
                disable_notification=disable_notification,
                rich_markdown=rich_markdown,
            )
        if rich_markdown:
            message = await self.telegram.send_rich_message(
                self.chat_id,
                body,
                reply_to_message_id=reply_to_message_id,
                disable_notification=disable_notification,
            )
            self._notify_typing_cleared()
            return message
        notification_options = (
            {"disable_notification": True} if disable_notification else {}
        )
        message = await self.telegram.send_message(
            self.chat_id,
            body,
            reply_to_message_id=reply_to_message_id,
            **notification_options,
        )
        self._notify_typing_cleared()
        return message

    async def _send_codex_activity_lines(
        self,
        lines: list[tuple[str, str]],
        suffix_key: str = "format.values.truncated",
        reply_to_message_id: int | None = None,
        *,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        async with self._codex_activity_lock:
            return await self._send_codex_activity_lines_locked(
                lines,
                suffix_key=suffix_key,
                reply_to_message_id=reply_to_message_id,
                disable_notification=disable_notification,
            )

    async def _send_codex_activity_lines_locked(
        self,
        lines: list[tuple[str, str]],
        suffix_key: str = "format.values.truncated",
        reply_to_message_id: int | None = None,
        *,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        text, entities = format_activity_lines(lines)
        message_limit = GROUP_ACTIVITY_MESSAGE_LIMIT if self.is_group else TELEGRAM_TEXT_LIMIT
        body = self._truncate(text, message_limit, suffix_key)
        body_length = utf16_length(body)
        entities = [
            entity
            for entity in entities
            if entity["offset"] + entity["length"] <= body_length
        ]
        if self.is_group:
            return await self._queue_group_activity(
                body,
                entities,
                reply_to_message_id,
                disable_notification=disable_notification,
            )
        notification_options = (
            {"disable_notification": True} if disable_notification else {}
        )
        message = await self.telegram.send_message(
            self.chat_id,
            body,
            reply_to_message_id=reply_to_message_id,
            entities=entities,
            **notification_options,
        )
        self._notify_typing_cleared()
        return message

    def _format_command_duration(self, duration_ms: int | float) -> str:
        total_seconds = duration_ms / 1000
        if total_seconds < 60:
            return self._text("detailed_log.values.duration_seconds", seconds=f"{total_seconds:.1f}")
        minutes, seconds = divmod(int(round(total_seconds)), 60)
        return self._text("detailed_log.values.duration_minutes", minutes=minutes, seconds=seconds)

    async def _on_codex_assistant_message(self, item_id: str, text: str) -> None:
        if self.is_group and not self.attached:
            return
        message_text = text.strip()
        if not message_text:
            return
        limit = GROUP_ACTIVITY_MESSAGE_LIMIT if self.is_group else TELEGRAM_TEXT_LIMIT
        chunks = format_long_text(message_text, limit)
        started_at = monotonic()
        try:
            async with self._codex_activity_lock:
                for chunk in chunks:
                    await self._send_codex_activity_locked(
                        chunk,
                        reply_to_message_id=self._turn_prompt_message_id,
                        batchable=False,
                        wait_for_delivery=self.is_group,
                        disable_notification=self._quiet_progress_notifications(),
                        rich_markdown=True,
                    )
        except Exception as exc:
            self._log_failure(
                "Could not deliver Codex commentary message",
                exc,
                event="telegram.codex.commentary.delivery_failed",
                level=logging.WARNING,
                operation="commentary",
                item_hash=log_identifier(item_id, namespace="codex-item"),
                text_length=len(message_text),
                item_count=len(chunks),
                duration_ms=round((monotonic() - started_at) * 1000),
            )
            return
        logger.info(
            "Delivered Codex commentary message",
            extra={
                "event": "telegram.codex.commentary.delivered",
                "operation": "commentary",
                "outcome": "delivered",
                "item_hash": log_identifier(item_id, namespace="codex-item"),
                "text_length": len(message_text),
                "item_count": len(chunks),
                "duration_ms": round((monotonic() - started_at) * 1000),
            },
        )

    async def _on_codex_finished(self, result: dict[str, Any]) -> None:
        if self.pending_user_input and not self.pending_user_input.future.done():
            self.pending_user_input.future.set_result({"answers": {}})
        self.pending_user_input = None
        for pending in self.pending_approvals.values():
            if not pending.future.done():
                pending.future.set_result("cancel")
        self.pending_approvals.clear()
        for pending in self.pending_mcp_elicitations.values():
            if not pending.future.done():
                pending.future.set_result({"action": "cancel"})
            await self._edit_message(
                self.chat_id,
                pending.message_id,
                self._text("mcp.label.elicitation_closed"),
                {"inline_keyboard": []},
            )
        self.pending_mcp_elicitations.clear()
        self._mcp_form_token = None
        status = result.get("status")
        text = str(result.get("text") or "").strip()
        error = result.get("error")
        error_key = result.get("error_key")
        if not error and isinstance(error_key, str):
            error = self._text(error_key)
        if result.get("recovered") is True:
            if status == "completed":
                message_text = self._text(
                    "recovery.label.turn_completed",
                    message=text or self._text("chat.label.no_result"),
                )
            elif status == "interrupted":
                if error_key == "recovery.label.prompt_expired":
                    message_text = self._text(
                        "recovery.label.prompt_interrupted",
                        message=text or self._text("chat.label.interrupted"),
                    )
                else:
                    message_text = self._text(
                        "recovery.label.interrupted",
                        message=text or self._text("chat.label.interrupted"),
                    )
            elif status == "unknown":
                message_text = self._text("recovery.label.unknown")
            else:
                message_text = self._text(
                    "recovery.label.turn_failed",
                    message=error or text or self._text("chat.label.failed"),
                )
            error = None
        elif status == "completed":
            message_text = text or self._text("chat.label.no_result")
        elif status == "interrupted":
            message_text = text or self._text("chat.label.interrupted")
        else:
            message_text = text or self._text("chat.label.failed")
        if error:
            message_text = self._text(
                "chat.label.with_error",
                message=message_text,
                error=self._truncate(str(error), 500),
            )
        if self.is_group and not self.attached:
            await self._finish_typing()
            self._command_messages.clear()
            self._turn_prompt_message_id = None
            pending_turn = self._codex().pending_turn
            if pending_turn and pending_turn.phase == "delivery_pending":
                self._codex().clear_pending_turn()
            return
        chunks = format_long_text(
            message_text,
            GROUP_ACTIVITY_MESSAGE_LIMIT if self.is_group else TELEGRAM_TEXT_LIMIT,
        )
        reply_to_message_id = self._turn_prompt_message_id
        async with self._codex_activity_lock:
            pending_turn = self._codex().pending_turn
            if pending_turn:
                for message_id in pending_turn.prompt_message_ids:
                    await self._edit_message(
                        self.chat_id,
                        message_id,
                        self._text("recovery.label.prompt_expired"),
                        {"inline_keyboard": []},
                    )
            quiet_final_delivery = self._quiet_progress_notifications()
            for index, chunk in enumerate(chunks):
                await self._send_codex_activity_locked(
                    self._text("chat.label.final_answer", message=chunk),
                    reply_to_message_id=reply_to_message_id,
                    priority=True,
                    disable_notification=quiet_final_delivery and index < len(chunks) - 1,
                    rich_markdown=True,
                )
            self._command_messages.clear()
            if self._turn_prompt_message_id == reply_to_message_id and not self._codex().is_busy:
                self._turn_prompt_message_id = None
            pending_turn = self._codex().pending_turn
            if pending_turn and pending_turn.phase == "delivery_pending":
                self._codex().clear_pending_turn()
        if self._codex().is_busy:
            self._start_typing()
        else:
            await self._finish_typing()

    async def _edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        if self.is_group and not self.attached:
            return False
        try:
            if reply_markup is None:
                await self.telegram.edit_message(chat_id, message_id, text)
            else:
                await self.telegram.edit_message(chat_id, message_id, text, reply_markup)
            return True
        except TelegramApiError as exc:
            if "message is not modified" not in str(exc).lower():
                self._log_failure(
                    "Could not edit Telegram message",
                    exc,
                    event="telegram.message.edit_failed",
                    level=logging.WARNING,
                )
                return False
            return True

    async def _edit_message_reply_markup(
        self,
        chat_id: int,
        message_id: int,
        reply_markup: dict[str, Any],
    ) -> bool:
        if self.is_group and not self.attached:
            return False
        try:
            await self.telegram.edit_message_reply_markup(
                chat_id,
                message_id,
                reply_markup,
            )
            return True
        except TelegramApiError as exc:
            if "message is not modified" not in str(exc).lower():
                self._log_failure(
                    "Could not clear Telegram inline keyboard",
                    exc,
                    event="telegram.message.reply_markup_edit_failed",
                    level=logging.WARNING,
                )
                return False
            return True

    async def _send_error(self, key: str) -> None:
        try:
            await self.telegram.send_message(self.chat_id, self._text(key))
        except TelegramApiError as exc:
            self._log_failure(
                "Could not send a Telegram error message",
                exc,
                event="telegram.error_message.send_failed",
                level=logging.WARNING,
            )

    def _truncate(self, text: str, limit: int, suffix_key: str = "format.values.truncated") -> str:
        return truncate(text, limit, self._text(suffix_key))

    async def _cancel_pending_prompts(self) -> None:
        for pending in self.pending_approvals.values():
            if not pending.future.done():
                pending.future.set_result("cancel")
        self.pending_approvals.clear()
        if self.pending_user_input and not self.pending_user_input.future.done():
            self.pending_user_input.future.set_result({"answers": {}})
        self.pending_user_input = None
        for pending in self.pending_mcp_elicitations.values():
            if not pending.future.done():
                pending.future.set_result({"action": "cancel"})
            await self._edit_message(
                self.chat_id,
                pending.message_id,
                self._text("mcp.label.elicitation_closed"),
                {"inline_keyboard": []},
            )
        self.pending_mcp_elicitations.clear()
        self._mcp_form_token = None

    def _help_text(self) -> str:
        return self._text("commands.label.help")

    def _unknown_command_text(self, command: str) -> str:
        matches = difflib.get_close_matches(
            command,
            TELEGRAM_COMMAND_NAMES,
            n=3,
            cutoff=0.45,
        )
        if matches:
            suggestions = ", ".join(f"/{name}" for name in matches)
            return self._text(
                "commands.label.unknown_suggested",
                command=command,
                suggestions=suggestions,
            )
        return self._text("commands.label.unknown")

    def _format_doctor_report(self, report: DoctorReport) -> str:
        if report.telegram_ok:
            telegram = self._text("doctor.values.telegram_ok")
        else:
            telegram = self._text("doctor.values.telegram_failed", detail=report.telegram_error or "unknown")

        if not report.codex_process_running:
            codex = self._text("doctor.values.codex_stopped")
        elif report.codex_rpc_ok:
            codex = self._text("doctor.values.codex_ok")
        else:
            codex = self._text("doctor.values.codex_failed", detail=report.codex_error or "unknown")

        auth_key = {
            "ok": "doctor.values.auth_ok",
            "api_key": "doctor.values.auth_api_key",
            "missing": "doctor.values.auth_missing",
            "unverified": "doctor.values.auth_unverified",
        }.get(report.chatgpt_auth, "doctor.values.auth_unknown")
        chat_key = "doctor.values.chat_selected" if report.thread_selected else "doctor.values.chat_empty"
        task_key = "doctor.values.task_busy" if report.task_busy else "doctor.values.task_idle"
        proxy_key = "doctor.values.proxy_on" if report.telegram_proxy_configured else "doctor.values.proxy_off"
        audio_key = {
            "ready": "doctor.values.audio_ready",
            "configured": "doctor.values.audio_configured",
            "missing": "doctor.values.audio_missing",
        }.get(report.audio_transcription_status, "doctor.values.audio_missing")
        lines = [self._text("doctor.label.title"), telegram, codex, self._text(auth_key)]

        lines.extend(
            (
                self._text(chat_key),
                self._text(task_key),
                self._text(proxy_key),
                self._text(audio_key, model=self.config.transcription_model),
            )
        )
        if report.poll_error_count:
            lines.append(self._text("doctor.values.poll_recovering", count=report.poll_error_count))
        elif report.poll_age_seconds is not None:
            lines.append(self._text("doctor.values.poll_healthy", seconds=report.poll_age_seconds))
        return "\n".join(lines)
