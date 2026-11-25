from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from time import monotonic, time as epoch_time
from typing import Any
from zoneinfo import ZoneInfo

from telecodex.codex.command_prompts import GOAL_CONTINUATION_PROMPT
from telecodex.codex.app_server import CodexAppServer
from telecodex.codex.mcp import McpServer
from telecodex.codex.mcp_session import CodexMcpSession
from telecodex.codex.settings import CodexSettings
from telecodex.codex.turn_history import CodexTurnHistory
from telecodex.codex.permissions import (
    PERMISSION_MODES,
    WORKSPACE_PERMISSION_PROFILE,
    thread_permission_params,
    turn_permission_params,
)
from telecodex.codex.rpc import RpcError
from telecodex.codex.types import (
    ActiveThreadWriterError,
    AssistantMessageCallback,
    CodexBackgroundTerminal,
    CodexChatBusyError,
    CodexGoal,
    CodexMemorySettings,
    CodexModel,
    CodexRecoverySnapshot,
    CodexServiceTier,
    CodexSkill,
    CodexThread,
    CollaborationModeUnavailableError,
    CommandFinishedCallback,
    FastTierUnavailableError,
    FinishedCallback,
    GoalNotFoundError,
    GoalUnavailableError,
    MemoryFeatureUnavailableError,
    PendingTurnDeliveryError,
    PromptCallback,
    SelectedSkillUnavailableError,
    StatusCallback,
)
from telecodex.config import AppConfig
from telecodex.i18n import SUPPORTED_LANGUAGES, TextCatalog, load_text_catalog
from telecodex.logging_config import (
    bind_log_context,
    current_log_context,
    log_exception,
    log_identifier,
)
from telecodex.state import (
    OUTPUT_MODE_CHOICES,
    GroupRuntimeStateStore,
    NewChatDefaults,
    PendingTurnState,
    ThreadRuntimeState,
    UserRuntimeStateStore,
)
from telecodex.codex.rpc import EventHandler, ServerRequestHandler

logger = logging.getLogger(__name__)
GOAL_TURN_START_TIMEOUT_SECONDS = 1.0
TURN_COMPLETION_WAIT_SECONDS = 30
MANUAL_COMPACTION_IDLE_POLL_SECONDS = 2.0


def _is_active_thread_writer_error(error: BaseException) -> bool:
    return "already has an active writer" in str(error).casefold()

class CodexSession:
    def __init__(
        self,
        config: AppConfig,
        state_store: UserRuntimeStateStore | GroupRuntimeStateStore,
        on_status: StatusCallback,
        on_command_completed: CommandFinishedCallback,
        on_finished: FinishedCallback,
        on_prompt: PromptCallback,
        texts: TextCatalog | None = None,
        app_server_factory: Callable[[EventHandler, ServerRequestHandler], Any] | None = None,
        on_assistant_message: AssistantMessageCallback | None = None,
    ) -> None:
        self.config = config
        self.state_store = state_store
        self.state = state_store.load()
        self.on_status = on_status
        self.on_command_completed = on_command_completed
        self.on_finished = on_finished
        self.on_prompt = on_prompt
        self.on_assistant_message = on_assistant_message
        self.texts = texts or load_text_catalog()
        self.active_turn_id: str | None = None
        self.active_command_executions: set[str] = set()
        self._active_command_without_id = False
        self._loaded_thread_id: str | None = None
        self._messages: dict[str, str] = {}
        self._message_phases: dict[str, str | None] = {}
        self._message_order: list[str] = []
        self._plan_text: str | None = None
        self._plan_deltas: dict[str, str] = {}
        self._review_text: str | None = None
        self._reported_activity_items: set[tuple[str, str]] = set()
        self._agent_activity_states: dict[str, str] = {}
        self._agent_activity_labels: dict[str, str] = {}
        self._mcp_reported_startup: dict[str, str] = {}
        self._assistant_message_ids: set[str] = set()
        self._assistant_message_tasks: list[asyncio.Task[None]] = []
        self._ready_lock = asyncio.Lock()
        self._reconciliation_lock = asyncio.Lock()
        self._pending_delivery_lock = asyncio.Lock()
        self._manual_compaction_recovery_lock = asyncio.Lock()
        self._recovery_lock = asyncio.Lock()
        self._thread_unsubscribe_lock = asyncio.Lock()
        self._thread_use_lock = asyncio.Lock()
        self._recovery_in_progress = False
        self._recovery_blocked = False
        self._recovery_hold = False
        self._turn_completion_task: asyncio.Task[None] | None = None
        self._switch_interrupt_pending = False
        self._turn_completed_event: asyncio.Event | None = None
        self._recovered_turn_id: str | None = None
        self._manual_compaction_future: asyncio.Future[None] | None = None
        self._manual_compaction_turn_id: str | None = None
        self._manual_compaction_item_completed = False
        self._manual_compaction_started_at: float | None = None
        self._manual_compaction_requested = False
        self._manual_compaction_barrier_future: asyncio.Future[None] | None = None
        self._manual_compaction_task: asyncio.Task[None] | None = None
        self._background_terminal_api_supported: bool | None = None
        self._goal: CodexGoal | None = None
        self._goal_thread_id: str | None = None
        self._goal_prompt_message_id: int | None = None
        self._goal_turn_started_event = asyncio.Event()
        self._last_event_at = monotonic()
        self._last_event_method: str | None = None
        if app_server_factory is None:
            self.app_server = CodexAppServer(
                codex_bin=config.codex_bin,
                network_access=config.network_access,
                on_event=self._on_event,
                on_server_request=self._on_server_request,
                mcp_env_vars=config.mcp_env_vars,
            )
        else:
            self.app_server = app_server_factory(self._on_event, self._on_server_request)
        self.mcp_manager = CodexMcpSession(
            config=config,
            app_server=self.app_server,
            ensure_app_server=self._ensure_app_server,
            state=lambda: self.state,
            loaded_thread_id=lambda: self._loaded_thread_id,
            thread_log_fields=self._thread_log_fields,
            schedule_status=self._schedule_status,
        )
        self.turn_history = CodexTurnHistory(self.app_server)
        self.settings_manager = CodexSettings(
            config=self.config,
            app_server=self.app_server,
            state=self.state,
            state_store=self.state_store,
            ensure_app_server=self._ensure_app_server,
            ensure_thread=self._ensure_thread,
            unsubscribe_thread=self._unsubscribe_thread,
            is_busy=lambda: self.is_busy,
            thread_log_fields=self._thread_log_fields,
            working_directory=lambda: self.working_directory,
            sync_persisted_thread_selection=self._sync_persisted_thread_selection,
            is_config_version_conflict=self._is_config_version_conflict,
        )

    def _thread_log_fields(self, thread_id: str | None = None) -> dict[str, str | None]:
        selected_thread_id = thread_id or self.state.thread_id
        return {
            "thread_hash": log_identifier(selected_thread_id, namespace="codex-thread")
            if selected_thread_id
            else None
        }

    def _turn_log_fields(self, turn_id: str | None = None) -> dict[str, str | None]:
        return {
            "turn_hash": log_identifier(turn_id, namespace="codex-turn") if turn_id else None
        }

    def _remember_goal(self, thread_id: str, value: Any) -> CodexGoal | None:
        if value is None:
            if self.state.thread_id == thread_id:
                self._goal = None
                self._goal_thread_id = thread_id
            return None
        if not isinstance(value, dict):
            raise ValueError("Codex app-server returned an invalid Goal")
        objective = value.get("objective")
        status = value.get("status")
        if not isinstance(objective, str) or not isinstance(status, str):
            raise ValueError("Codex app-server returned an invalid Goal")

        def non_negative_integer(field: str) -> int:
            field_value = value.get(field, 0)
            return field_value if type(field_value) is int and field_value >= 0 else 0

        token_budget = value.get("tokenBudget")
        goal = CodexGoal(
            objective=objective,
            status=status,
            token_budget=(token_budget if type(token_budget) is int and token_budget >= 0 else None),
            tokens_used=non_negative_integer("tokensUsed"),
            time_used_seconds=non_negative_integer("timeUsedSeconds"),
        )
        if self.state.thread_id == thread_id:
            self._goal = goal
            self._goal_thread_id = thread_id
        return goal

    async def _read_goal(self, thread_id: str) -> CodexGoal | None:
        result = await self._goal_request("thread/goal/get", {"threadId": thread_id})
        return self._remember_goal(thread_id, result.get("goal"))

    async def _ensure_goal_continuation_started(
        self,
        thread_id: str,
        prompt_message_id: int | None,
    ) -> None:
        if self.is_busy or not self._goal_is_active(thread_id):
            return
        try:
            await asyncio.wait_for(
                self._goal_turn_started_event.wait(),
                timeout=GOAL_TURN_START_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            pass
        if self.is_busy or not self._goal_is_active(thread_id):
            return
        pending = self.state.pending_turn
        if (
            pending is not None
            and pending.thread_id == thread_id
            and pending.phase in {"starting", "running"}
        ):
            return
        logger.info(
            "Starting explicit turn to resume an active Codex Goal",
            extra={"event": "codex.goal.continuation.kickoff", **self._thread_log_fields(thread_id)},
        )
        await self.send_message(GOAL_CONTINUATION_PROMPT, prompt_message_id)

    async def _goal_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self.app_server.request(method, params, timeout_seconds=30)
        except RpcError as exc:
            message = str(exc).casefold()
            if "unknown method" in message or "method not found" in message:
                raise GoalUnavailableError(
                    "The connected Codex app-server does not support native Goals"
                ) from exc
            raise

    def _goal_is_active(self, thread_id: str | None = None) -> bool:
        selected_thread = thread_id or self.state.thread_id
        return (
            selected_thread is not None
            and self._goal_thread_id == selected_thread
            and self._goal is not None
            and self._goal.status == "active"
        )

    @property
    def is_busy(self) -> bool:
        return (
            self.active_turn_id is not None
            or self._manual_compaction_future is not None
            or self._manual_compaction_requested
        )

    @property
    def recovery_in_progress(self) -> bool:
        return self._recovery_in_progress

    @property
    def recovery_blocked(self) -> bool:
        return self._recovery_blocked

    def hold_recovery(self) -> None:
        self._recovery_hold = True

    def release_recovery_hold(self) -> None:
        if not self._recovery_in_progress and not self._recovery_blocked:
            self._recovery_hold = False

    @property
    def compaction_barrier_pending(self) -> bool:
        return self._manual_compaction_requested or self._manual_compaction_future is not None

    @property
    def manual_compaction_waiting_for_idle(self) -> bool:
        return self._manual_compaction_requested and self._manual_compaction_future is None

    @property
    def has_active_command_execution(self) -> bool:
        return bool(self.active_command_executions) or self._active_command_without_id

    async def start(self) -> bool:
        await self._ensure_app_server()
        if self.state.pending_turn:
            if await self.reconcile_pending_turn() == "active":
                return True
        if self.state.thread_id:
            await self._ensure_thread()
            if self.is_busy:
                return True
            await self._unsubscribe_thread(source="session_start")
        return False

    async def close(self) -> None:
        await self.cancel_pending_manual_compaction()
        if self.active_turn_id and self.app_server.is_running:
            try:
                with bind_log_context(interrupt_source="service_shutdown"):
                    await self._request_turn_interrupt(
                        self.state.thread_id,
                        self.active_turn_id,
                        timeout_seconds=10,
                    )
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not interrupt the active Codex turn during shutdown",
                    exc,
                    event="codex.turn.shutdown_interrupt.failed",
                    level=logging.WARNING,
                    timeout_seconds=10,
                    **self._thread_log_fields(),
                    **self._turn_log_fields(self.active_turn_id),
                )
            if self._turn_completed_event is not None:
                try:
                    await asyncio.wait_for(self._turn_completed_event.wait(), timeout=5)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Codex turn did not report interruption before shutdown",
                        extra={
                            "event": "codex.turn.shutdown_completion.timeout",
                            "timeout_seconds": 5,
                            **self._thread_log_fields(),
                            **self._turn_log_fields(self.active_turn_id),
                        },
                    )
            if self._turn_completion_task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(self._turn_completion_task), timeout=5)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Codex completion notification is pending recovery after shutdown",
                        extra={
                            "event": "codex.turn.shutdown_delivery.pending",
                            "timeout_seconds": 5,
                            **self._thread_log_fields(),
                            **self._turn_log_fields(self.active_turn_id),
                        },
                    )
        await self.app_server.stop()

    async def send_message(
        self,
        text: str,
        prompt_message_id: int | None = None,
        *,
        additional_input_items: list[dict[str, Any]] | None = None,
    ) -> str:
        if self._recovery_in_progress or self._recovery_blocked or self._recovery_hold:
            raise CodexChatBusyError("The Codex chat is being recovered")
        input_items = [{"type": "text", "text": text}]
        if additional_input_items:
            input_items.extend(additional_input_items)
        selected_skill_name = self.settings_manager.pending_skill_name
        selected_skill_path = self.settings_manager.pending_skill_path
        selected_skill_display_name: str | None = None
        if selected_skill_name and selected_skill_path:
            skills = await self.list_skills(force_reload=True)
            skill = next(
                (
                    item
                    for item in skills
                    if item.name == selected_skill_name
                    and item.path == selected_skill_path
                    and item.enabled
                ),
                None,
            )
            if skill is None:
                self.clear_selected_skill()
                raise SelectedSkillUnavailableError(
                    "The selected Codex skill is no longer enabled for the active project"
                )
            selected_skill_display_name = skill.display_name or skill.name
            input_items[0]["text"] = f"${skill.name}\n\n{text}"
            input_items.append({"type": "skill", "name": skill.name, "path": skill.path})
        await self._wait_for_turn_completion()
        pending = self.state.pending_turn
        if pending and pending.phase == "delivery_pending":
            await self._deliver_pending_turn(pending)
            if self.state.pending_turn is not None:
                raise PendingTurnDeliveryError

        async with self._thread_use_lock:
            return await self._send_message_locked(
                text,
                input_items,
                prompt_message_id,
                selected_skill_name,
                selected_skill_display_name,
            )

    async def _send_message_locked(
        self,
        text: str,
        input_items: list[dict[str, Any]],
        prompt_message_id: int | None,
        selected_skill_name: str | None,
        selected_skill_display_name: str | None,
    ) -> str:
        if self._manual_compaction_future is not None or self._manual_compaction_requested:
            raise CodexChatBusyError("The Codex chat is busy while compacting its context")
        if self.is_busy and self.app_server.is_running:
            try:
                await self.app_server.request(
                    "turn/steer",
                    {
                        "threadId": self.state.thread_id,
                        "expectedTurnId": self.active_turn_id,
                        "input": input_items,
                    },
                )
                self._update_pending_prompt(prompt_message_id)
                if selected_skill_name:
                    self.clear_selected_skill()
                    if selected_skill_display_name:
                        self._schedule_status(
                            "detailed_log.values.skill_attached",
                            selected_skill_display_name,
                        )
                logger.info(
                    "Codex accepted input for the active turn",
                    extra={
                        "event": "codex.turn.input.steered",
                        "text_length": len(text),
                        "input_item_count": len(input_items),
                        "model": self._effective_model() or "default",
                        "effort": self.settings_manager.selected_effort or "default",
                        **self._thread_log_fields(),
                        **self._turn_log_fields(self.active_turn_id),
                    },
                )
                return "steered"
            except RpcError:
                # The turn may have completed between the busy check and the steer call.
                self.active_turn_id = None

        pending = self.state.pending_turn
        if pending and pending.phase == "starting" and self.active_turn_id is None:
            self.state.pending_turn = None
            self.state_store.save(self.state)

        await self._ensure_thread()
        assert self.state.thread_id
        self._clear_turn_messages()
        params: dict[str, Any] = {
            "threadId": self.state.thread_id,
            "input": input_items,
            "cwd": str(self.working_directory),
            "summary": "concise",
        }
        params.update(turn_permission_params(self.permission_mode))
        model = self._effective_model()
        if model:
            params["model"] = model
        if self.settings_manager.selected_effort is not None:
            params["effort"] = self.settings_manager.selected_effort
        service_tier = self.selected_service_tier
        if service_tier:
            params["serviceTier"] = service_tier
        if self.settings_manager.collaboration_mode == "plan":
            plan_model = self.settings_manager.thread_model or model
            if plan_model:
                plan_settings: dict[str, Any] = {
                    "model": plan_model,
                    "developer_instructions": None,
                }
                effort = self.settings_manager.thread_effort or self.settings_manager.selected_effort
                if effort:
                    plan_settings["reasoning_effort"] = effort
                params["collaborationMode"] = {"mode": "plan", "settings": plan_settings}
        self.state.pending_turn = PendingTurnState(
            thread_id=self.state.thread_id,
            turn_id=None,
            prompt_message_id=prompt_message_id,
            started_at=int(epoch_time()),
        )
        self._clear_active_command_executions()
        self.state_store.save(self.state)
        self._turn_completed_event = asyncio.Event()
        request_started_at = monotonic()
        logger.info(
            "Starting Codex turn",
            extra={
                "event": "codex.turn.start.requested",
                "text_length": len(text),
                "input_item_count": len(input_items),
                "has_attachments": len(input_items) > 1,
                "model": model or "default",
                "effort": self.settings_manager.selected_effort or "default",
                "permission_mode": self.permission_mode or self.config.approval_policy,
                **self._thread_log_fields(),
            },
        )
        try:
            result = await self.app_server.request("turn/start", params, timeout_seconds=60)
        except RpcError as exc:
            logger.error(
                "Codex rejected turn/start",
                extra={
                    "event": "codex.turn.start.rpc_error",
                    "rpc_method": "turn/start",
                    "rpc_code": exc.code,
                    "duration_ms": round((monotonic() - request_started_at) * 1000),
                    **self._thread_log_fields(),
                },
            )
            self._persist_turn_result(
                {
                    "status": "failed",
                    "text": "",
                    "error": str(exc),
                }
            )
            raise
        turn = result.get("turn", {})
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        if not isinstance(turn_id, str) or not turn_id:
            raise RuntimeError("Codex app-server did not return a turn ID")
        turn_status = turn.get("status") if isinstance(turn, dict) else None
        logger.info(
            "Codex turn started",
            extra={
                "event": "codex.turn.started",
                "turn_hash": log_identifier(turn_id, namespace="codex-turn"),
                "turn_status": turn_status if isinstance(turn_status, str) else None,
                "duration_ms": round((monotonic() - request_started_at) * 1000),
                "model": model or "default",
                "effort": self.settings_manager.selected_effort or "default",
                **self._thread_log_fields(),
            },
        )
        if self._turn_completed_event is not None and self._turn_completed_event.is_set():
            self.active_turn_id = None
        else:
            self._set_active_turn_id(turn_id)
        if selected_skill_name:
            self.clear_selected_skill()
            if selected_skill_display_name:
                self._schedule_status("detailed_log.values.skill_attached", selected_skill_display_name)
        return "started"

    async def interrupt(self) -> bool:
        goal_was_active = self._goal_is_active()
        interrupt_source = current_log_context().get("interrupt_source", "unspecified")
        pause_error: Exception | None = None
        if goal_was_active:
            try:
                await self.set_goal_status("paused")
            except Exception as exc:
                pause_error = exc
                log_exception(
                    logger,
                    "Could not pause Codex Goal before interrupting its turn",
                    exc,
                    event="codex.goal.interrupt_pause_failed",
                    level=logging.WARNING,
                    **self._thread_log_fields(),
                )
        if (
            self.is_busy
            and self.state.thread_id
            and self.active_turn_id
            and self.app_server.is_running
        ):
            logger.info(
                "Requesting interruption of active Codex turn",
                extra={
                    "event": "codex.turn.interrupt.requested",
                    "interrupt_source": interrupt_source,
                    **self._thread_log_fields(),
                    **self._turn_log_fields(self.active_turn_id),
                },
            )
            try:
                await self._request_turn_interrupt(
                    self.state.thread_id,
                    self.active_turn_id,
                    timeout_seconds=15,
                )
            except Exception as exc:
                log_exception(
                    logger,
                    "Codex rejected or lost the turn interruption request",
                    exc,
                    event="codex.turn.interrupt.failed",
                    level=logging.WARNING,
                    interrupt_source=interrupt_source,
                    **self._thread_log_fields(),
                    **self._turn_log_fields(self.active_turn_id),
                )
                raise
            logger.info(
                "Codex accepted turn interruption request",
                extra={
                    "event": "codex.turn.interrupt.accepted",
                    "interrupt_source": interrupt_source,
                    **self._thread_log_fields(),
                    **self._turn_log_fields(self.active_turn_id),
                },
            )
            return True
        logger.info(
            "No active Codex turn was available to interrupt",
            extra={
                "event": "codex.turn.interrupt.skipped",
                "interrupt_source": interrupt_source,
                "outcome": "goal_paused" if goal_was_active and pause_error is None else "idle",
                "transport_running": self.app_server.is_running,
                **self._thread_log_fields(),
                **self._turn_log_fields(self.active_turn_id),
            },
        )
        if goal_was_active and pause_error is None:
            return True
        if pause_error is not None:
            raise pause_error
        return False

    async def interrupt_and_abandon_for_switch(self) -> str:
        if self._switch_interrupt_pending:
            raise RuntimeError("A Codex chat switch is already being processed")
        self._switch_interrupt_pending = True
        try:
            return await self._interrupt_and_abandon_for_switch()
        finally:
            self._switch_interrupt_pending = False

    async def _interrupt_and_abandon_for_switch(self) -> str:
        """Interrupt the current turn, then release this context if it stays active.

        A forced app-server stop is only used after the caller has received an
        explicit user confirmation. The old prompt is never replayed.
        """
        pending = self.state.pending_turn
        if pending is not None and pending.phase == "delivery_pending":
            await self._deliver_pending_turn(pending)
            pending = self.state.pending_turn
            if not self.is_busy and pending is None:
                return "delivered"
        if not self.is_busy and pending is None:
            return "idle"

        thread_id = pending.thread_id if pending is not None else self.state.thread_id
        turn_id = pending.turn_id if pending is not None else self.active_turn_id
        interrupt_error: BaseException | None = None
        if self.app_server.is_running and thread_id and turn_id:
            try:
                with bind_log_context(interrupt_source="confirmed_thread_switch"):
                    await self.interrupt()
            except Exception as exc:
                interrupt_error = exc
                log_exception(
                    logger,
                    "Could not interrupt Codex before a confirmed chat switch",
                    exc,
                    event="codex.turn.switch.interrupt_failed",
                    level=logging.WARNING,
                    **self._thread_log_fields(thread_id),
                    **self._turn_log_fields(turn_id),
                )

        # Allow a normally responsive server to report its final state before
        # taking the context-local app-server down.
        for _ in range(3):
            if pending is not None and pending.turn_id and self.app_server.is_running:
                try:
                    turn = await self._read_persisted_turn(
                        pending.thread_id,
                        pending.turn_id,
                    )
                except Exception:
                    turn = None
                status = turn.get("status") if isinstance(turn, dict) else None
                if status in {"completed", "interrupted", "failed"}:
                    self._persist_turn_result(
                        {"status": status, "error": self._turn_error_message(turn)}
                    )
                    self.active_turn_id = None
                    self._clear_active_command_executions()
                    if self._turn_completed_event is not None:
                        self._turn_completed_event.set()
                    await self._deliver_pending_turn(pending)
                    return status
            if not self.is_busy:
                return "interrupted"
            await asyncio.sleep(0.5)

        # Persist the unknown outcome before closing the transport. That keeps
        # recovery truthful if delivery itself is interrupted by shutdown.
        if pending is not None:
            self._persist_turn_result(
                {"status": "unknown", "error_key": "recovery.label.unknown"}
            )
        self.active_turn_id = None
        self._clear_active_command_executions()
        if self._turn_completed_event is not None:
            self._turn_completed_event.set()
        compaction = self._manual_compaction_future
        self._manual_compaction_future = None
        self._manual_compaction_turn_id = None
        self._manual_compaction_item_completed = False
        self._manual_compaction_started_at = None
        if compaction is not None and not compaction.done():
            compaction.set_result(None)

        if self.app_server.is_running:
            try:
                await self.app_server.stop()
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not stop the Codex app-server after confirmed chat switch",
                    exc,
                    event="codex.turn.switch.app_server_stop_failed",
                    level=logging.ERROR,
                    **self._thread_log_fields(thread_id),
                    **self._turn_log_fields(turn_id),
                )
                raise
        if pending is not None:
            await self._deliver_pending_turn(pending)
        logger.warning(
            "Released Codex context after an unconfirmed turn interruption",
            extra={
                "event": "codex.turn.switch.forced_release",
                "outcome": "unknown",
                "interrupt_error_type": type(interrupt_error).__name__ if interrupt_error else None,
                **self._thread_log_fields(thread_id),
                **self._turn_log_fields(turn_id),
            },
        )
        return "unknown"

    def _clear_active_command_executions(self) -> None:
        self.active_command_executions.clear()
        self._active_command_without_id = False

    async def _request_turn_interrupt(
        self,
        thread_id: str | None,
        turn_id: str | None,
        *,
        timeout_seconds: float,
    ) -> None:
        await self.app_server.request(
            "turn/interrupt",
            {"threadId": thread_id, "turnId": turn_id},
            timeout_seconds=timeout_seconds,
        )

    async def _interrupt_before_thread_switch(self, source: str) -> None:
        with bind_log_context(interrupt_source=source):
            try:
                await self.interrupt()
            except RpcError as exc:
                pending = self.state.pending_turn
                if exc.code != -32600:
                    raise
                if pending is not None:
                    pending_thread_id = pending.thread_id
                    pending_turn_id = pending.turn_id
                    outcome = await self.reconcile_pending_turn()
                    reconciled = self.state.pending_turn is None and not self.is_busy
                else:
                    pending_thread_id = self.state.thread_id
                    pending_turn_id = self.active_turn_id
                    if not pending_thread_id or not pending_turn_id:
                        raise
                    turn = await self._read_persisted_turn(
                        pending_thread_id,
                        pending_turn_id,
                    )
                    status = turn.get("status") if isinstance(turn, dict) else None
                    outcome = status if isinstance(status, str) else "not_found"
                    reconciled = (
                        status in {"completed", "interrupted", "failed"}
                        or turn is None
                    )
                    if reconciled:
                        self.active_turn_id = None
                        if self._turn_completed_event is not None:
                            self._turn_completed_event.set()
                        self._clear_turn_messages()
                if reconciled:
                    logger.info(
                        "Reconciled a terminal Codex turn after interrupt was rejected",
                        extra={
                            "event": "codex.turn.switch.reconciled",
                            "outcome": outcome,
                            "interrupt_source": source,
                            **self._thread_log_fields(pending_thread_id),
                            **self._turn_log_fields(pending_turn_id),
                        },
                    )
                    return
                logger.warning(
                    "Could not switch Codex threads because the rejected interrupt did not resolve the active turn",
                    extra={
                        "event": "codex.turn.switch.blocked",
                        "outcome": outcome,
                        "interrupt_source": source,
                        **self._thread_log_fields(pending_thread_id),
                        **self._turn_log_fields(pending_turn_id),
                    },
                )
                raise



















    async def list_mcp_servers(self) -> list[McpServer]:
        return await self.mcp_manager.list_mcp_servers()

    async def add_mcp_stdio_server(
        self,
        name: str,
        command: str,
        args: tuple[str, ...],
        cwd: str | None,
        env_vars: tuple[str, ...],
    ) -> None:
        await self.mcp_manager.add_mcp_stdio_server(name, command, args, cwd, env_vars)

    async def add_mcp_http_server(
        self,
        name: str,
        url: str,
        auth: str,
        bearer_token_env_var: str | None = None,
    ) -> None:
        await self.mcp_manager.add_mcp_http_server(name, url, auth, bearer_token_env_var)

    async def set_mcp_server_enabled(self, name: str, enabled: bool) -> None:
        await self.mcp_manager.set_mcp_server_enabled(name, enabled)

    async def remove_mcp_server(self, name: str) -> None:
        await self.mcp_manager.remove_mcp_server(name)

    async def logout_mcp_server(self, name: str) -> None:
        await self.mcp_manager.logout_mcp_server(name)

    async def reload_mcp_config(self) -> bool:
        return await self.mcp_manager.reload_mcp_config()

    async def start_mcp_oauth(self, name: str) -> tuple[str, str | None]:
        return await self.mcp_manager.start_mcp_oauth(name)

    async def mcp_oauth_uses_remote_callback(self, name: str) -> bool:
        return await self.mcp_manager.mcp_oauth_uses_remote_callback(name)

    def _handle_mcp_oauth_completed(self, params: dict[str, Any]) -> None:
        self.mcp_manager._handle_mcp_oauth_completed(params)

    @staticmethod
    def _is_config_version_conflict(error: RpcError) -> bool:
        return CodexMcpSession._is_config_version_conflict(error)

    async def _read_persisted_turn(self, thread_id: str, turn_id: str) -> dict[str, Any] | None:
        return await self.turn_history._read_persisted_turn(thread_id, turn_id)

    async def last_assistant_message(self, thread_id: str) -> str:
        async with asyncio.timeout(20):
            await self._ensure_app_server()
            return await self.turn_history.last_assistant_message(thread_id)

    async def _read_turn_text(
        self,
        thread_id: str,
        turn_id: str | None,
        *,
        timeout_seconds: int = 60,
    ) -> str:
        return await self.turn_history._read_turn_text(
            thread_id,
            turn_id,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _turn_error_message(turn: dict[str, Any]) -> str | None:
        return CodexTurnHistory._turn_error_message(turn)

















    async def list_models(self) -> list[CodexModel]:
        return await self.settings_manager.list_models()

    async def current_model_settings(self, models: list[CodexModel]) -> tuple[str | None, str | None]:
        return await self.settings_manager.current_model_settings(models)

    @property
    def selected_model(self) -> str | None:
        return self.settings_manager.selected_model

    @property
    def selected_effort(self) -> str | None:
        return self.settings_manager.selected_effort

    @property
    def selected_service_tier(self) -> str | None:
        return self.settings_manager.selected_service_tier

    async def toggle_fast_mode(self) -> bool:
        return await self.settings_manager.toggle_fast_mode()

    async def list_skills(self, *, force_reload: bool=False) -> list[CodexSkill]:
        return await self.settings_manager.list_skills(force_reload=force_reload)

    async def set_skill_enabled(self, skill: CodexSkill, enabled: bool) -> None:
        await self.settings_manager.set_skill_enabled(skill, enabled)

    def select_skill_for_next_turn(self, skill: CodexSkill) -> None:
        self.settings_manager.select_skill_for_next_turn(skill)

    def clear_selected_skill(self) -> None:
        self.settings_manager.clear_selected_skill()

    def current_memory_mode(self) -> str | None:
        return self.settings_manager.current_memory_mode()

    async def read_memory_settings(self) -> CodexMemorySettings:
        return await self.settings_manager.read_memory_settings()

    async def set_memory_mode(self, mode: str) -> None:
        await self.settings_manager.set_memory_mode(mode)

    async def set_memory_generation(self, enabled: bool) -> None:
        await self.settings_manager.set_memory_generation(enabled)

    async def toggle_memory_generation(self) -> bool:
        return await self.settings_manager.toggle_memory_generation()

    async def enable_memories_feature(self) -> None:
        await self.settings_manager.enable_memories_feature()

    async def reset_memories(self) -> None:
        await self.settings_manager.reset_memories()

    def default_model(self, models: list[CodexModel]) -> str | None:
        return self.settings_manager.default_model(models)

    async def select_model(self, model: str | None, effort: str | None=None) -> None:
        await self.settings_manager.select_model(model, effort)

    def _effective_model(self) -> str | None:
        return self.settings_manager._effective_model()

    @property
    def collaboration_mode(self) -> str:
        return self.settings_manager.collaboration_mode

    async def toggle_plan_mode(self) -> str:
        return await self.settings_manager.toggle_plan_mode()

    async def set_collaboration_mode(self, mode: str, *, create_if_missing: bool=False) -> None:
        await self.settings_manager.set_collaboration_mode(mode, create_if_missing=create_if_missing)

    async def _update_collaboration_mode(self, mode: str, *, unsubscribe: bool=True) -> None:
        await self.settings_manager._update_collaboration_mode(mode, unsubscribe=unsubscribe)

    async def _list_collaboration_mode_presets(self) -> dict[str, dict[str, Any]]:
        return await self.settings_manager._list_collaboration_mode_presets()

    def _capture_thread_settings(self, response: dict[str, Any]) -> None:
        self.settings_manager._capture_thread_settings(response)

    @property
    def pending_turn(self) -> PendingTurnState | None:
        return self.state.pending_turn

    @property
    def seconds_since_last_event(self) -> float:
        return max(0.0, monotonic() - self._last_event_at)

    @property
    def last_event_method(self) -> str | None:
        return self._last_event_method

    @property
    def transport_running(self) -> bool:
        return bool(self.app_server.is_running)

    def pending_turn_age_seconds(self) -> float | None:
        pending = self.state.pending_turn
        if pending is None:
            return None
        return max(0.0, epoch_time() - pending.started_at)

    async def reconcile_pending_turn(self) -> str:
        """Reconcile persisted turn state without replaying its prompt.

        Returns ``none``, ``starting``, ``active``, ``delivered`` or ``retry``.
        ``retry`` leaves persisted state intact for a later watchdog attempt.
        """
        async with self._reconciliation_lock:
            pending = self.state.pending_turn
            if pending is None:
                return "none"
            if pending.phase == "delivery_pending":
                await self._deliver_pending_turn(pending)
                return "delivered" if self.state.pending_turn is None else "retry"
            if not pending.turn_id:
                if self.pending_turn_age_seconds() is not None and self.pending_turn_age_seconds() < 90:
                    return "starting"
                self._persist_turn_result({"status": "unknown", "error_key": "recovery.label.unknown"})
                await self._deliver_pending_turn(pending)
                return "delivered" if self.state.pending_turn is None else "retry"
            try:
                if not self.app_server.is_running:
                    await self._ensure_app_server()
                turn = await self._read_persisted_turn(pending.thread_id, pending.turn_id)
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not inspect pending Codex turn",
                    exc,
                    event="codex.recovery.pending_turn.inspect_failed",
                    level=logging.WARNING,
                    pending_phase=pending.phase,
                    **self._thread_log_fields(pending.thread_id),
                    **self._turn_log_fields(pending.turn_id),
                )
                return "retry"
            if turn is None:
                logger.warning(
                    "Codex turn was not found during recovery",
                    extra={
                        "event": "codex.recovery.turn_status",
                        "outcome": "not_found",
                        "pending_phase": pending.phase,
                        "recovery_age_seconds": self.pending_turn_age_seconds(),
                        **self._thread_log_fields(pending.thread_id),
                        **self._turn_log_fields(pending.turn_id),
                    },
                )
                self._persist_turn_result({"status": "unknown", "error_key": "recovery.label.unknown"})
                await self._deliver_pending_turn(pending)
                return "delivered" if self.state.pending_turn is None else "retry"
            status = turn.get("status")
            logger.info(
                "Read Codex turn status during recovery",
                extra={
                    "event": "codex.recovery.turn_status",
                    "turn_status": status if isinstance(status, str) else "unknown",
                    "pending_phase": pending.phase,
                    "recovery_age_seconds": self.pending_turn_age_seconds(),
                    **self._thread_log_fields(pending.thread_id),
                    **self._turn_log_fields(pending.turn_id),
                },
            )
            if status in {"completed", "interrupted", "failed"}:
                self._persist_turn_result({"status": status, "error": self._turn_error_message(turn)})
                await self._deliver_pending_turn(pending)
                return "delivered" if self.state.pending_turn is None else "retry"
            if status != "inProgress":
                self._persist_turn_result({"status": "unknown", "error_key": "recovery.label.unknown"})
                await self._deliver_pending_turn(pending)
                return "delivered" if self.state.pending_turn is None else "retry"
            if self.is_busy and self.app_server.is_running:
                return "active"
            try:
                if await self._recover_pending_turn(resolve_error=False):
                    return "active"
                return "delivered" if self.state.pending_turn is None else "retry"
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not resume pending Codex turn",
                    exc,
                    event="codex.recovery.pending_turn.resume_failed",
                    level=logging.WARNING,
                    pending_phase=pending.phase,
                    **self._thread_log_fields(pending.thread_id),
                    **self._turn_log_fields(pending.turn_id),
                )
                return "retry"

    def add_pending_prompt(self, message_id: int) -> None:
        pending = self.state.pending_turn
        if pending and message_id not in pending.prompt_message_ids:
            pending.prompt_message_ids.append(message_id)
            self.state_store.save(self.state)

    def remove_pending_prompt(self, message_id: int) -> None:
        pending = self.state.pending_turn
        if pending and message_id in pending.prompt_message_ids:
            pending.prompt_message_ids.remove(message_id)
            self.state_store.save(self.state)

    def clear_pending_turn(self) -> None:
        if self.state.pending_turn is not None:
            self.state.pending_turn = None
            self.state_store.save(self.state)

    def _release_reconciled_turn_state(self, pending: PendingTurnState) -> bool:
        if self.state.thread_id != pending.thread_id:
            logger.warning(
                "Kept local Codex turn state because the selected thread changed",
                extra={
                    "event": "codex.recovery.local_state_release_skipped",
                    "outcome": "thread_changed",
                    "pending_phase": pending.phase,
                    **self._thread_log_fields(pending.thread_id),
                    **self._turn_log_fields(pending.turn_id),
                },
            )
            return False
        if (
            self.active_turn_id
            and pending.turn_id
            and self.active_turn_id != pending.turn_id
        ):
            logger.warning(
                "Kept local Codex turn state because another turn is active",
                extra={
                    "event": "codex.recovery.local_state_release_skipped",
                    "outcome": "different_turn_active",
                    "pending_phase": pending.phase,
                    **self._thread_log_fields(pending.thread_id),
                    **self._turn_log_fields(self.active_turn_id),
                    "expected_turn_hash": log_identifier(
                        pending.turn_id,
                        namespace="codex-turn",
                    ) if pending.turn_id else None,
                },
            )
            return False
        was_busy = self.is_busy
        self.active_turn_id = None
        self._clear_active_command_executions()
        if self._turn_completed_event is not None:
            self._turn_completed_event.set()
        self._clear_turn_messages()
        logger.info(
            "Released local Codex turn state after reconciliation",
            extra={
                "event": "codex.recovery.local_state_released",
                "outcome": "released",
                "was_busy": was_busy,
                "pending_phase": pending.phase,
                **self._thread_log_fields(pending.thread_id),
                **self._turn_log_fields(pending.turn_id),
            },
        )
        return True










    def _update_pending_prompt(self, message_id: int | None) -> None:
        pending = self.state.pending_turn
        if pending and message_id is not None and pending.phase != "delivery_pending":
            pending.prompt_message_id = message_id
            self.state_store.save(self.state)

    def _set_active_turn_id(self, turn_id: str) -> None:
        pending = self.state.pending_turn
        if pending and pending.phase == "delivery_pending" and self._goal_is_active():
            self._prepare_goal_turn(pending.thread_id)
            pending = self.state.pending_turn
        elif pending and pending.phase == "delivery_pending" and pending.turn_id == turn_id:
            self.active_turn_id = None
            return
        self.active_turn_id = turn_id
        if pending and pending.phase != "delivery_pending":
            pending.turn_id = turn_id
            pending.phase = "running"
            self.state_store.save(self.state)

    def _prepare_goal_turn(self, thread_id: str) -> None:
        pending = self.state.pending_turn
        if pending is None or pending.thread_id != thread_id:
            self.state.pending_turn = PendingTurnState(
                thread_id=thread_id,
                turn_id=None,
                prompt_message_id=self._goal_prompt_message_id,
                started_at=int(epoch_time()),
                phase="starting",
            )
            self.state_store.save(self.state)
            return
        if pending.phase == "delivery_pending":
            pending.turn_id = None
            pending.prompt_message_id = self._goal_prompt_message_id or pending.prompt_message_id
            pending.started_at = int(epoch_time())
            pending.phase = "starting"
            pending.result_status = None
            pending.error = None
            pending.error_key = None
            pending.prompt_message_ids.clear()
            self.state_store.save(self.state)

    def _persist_turn_result(self, result: dict[str, Any]) -> None:
        pending = self.state.pending_turn
        if pending is None:
            return
        status = result.get("status")
        pending.phase = "delivery_pending"
        pending.result_status = (
            status
            if isinstance(status, str) and status in {"completed", "interrupted", "failed", "unknown"}
            else "failed"
        )
        error = result.get("error")
        error_key = result.get("error_key")
        pending.error = error if isinstance(error, str) else None
        pending.error_key = error_key if isinstance(error_key, str) else None
        self.state_store.save(self.state)

    async def _deliver_pending_turn(self, pending: PendingTurnState) -> None:
        async with self._pending_delivery_lock:
            # A live completion event and startup reconciliation can race. The
            # first successful callback clears this record; the second path
            # must not send the same result again.
            if self.state.pending_turn is not pending:
                return
            self._release_reconciled_turn_state(pending)
            if pending.result_status == "unknown" and not self.app_server.is_running:
                text = ""
            else:
                try:
                    text = await self._read_turn_text(pending.thread_id, pending.turn_id)
                except Exception as exc:
                    log_exception(
                        logger,
                        "Could not recover Codex turn output",
                        exc,
                        event="codex.recovery.turn_output.read_failed",
                        level=logging.WARNING,
                        **self._thread_log_fields(pending.thread_id),
                        **self._turn_log_fields(pending.turn_id),
                    )
                    text = ""
            if not text and pending.result_status == "completed":
                text = self._text("recovery.label.result_unavailable")
            result = {
                "status": pending.result_status or "unknown",
                "text": text,
                "error": pending.error,
                "error_key": pending.error_key,
                "recovered": True,
            }
            try:
                await self.on_finished(result)
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not deliver recovered Codex result",
                    exc,
                    event="codex.recovery.result.delivery_failed",
                    level=logging.WARNING,
                    turn_status=pending.result_status,
                    **self._thread_log_fields(pending.thread_id),
                    **self._turn_log_fields(pending.turn_id),
                )

    async def _recover_pending_turn(self, *, resolve_error: bool = True) -> bool:
        pending = self.state.pending_turn
        if pending is None:
            return False
        if pending.phase == "delivery_pending":
            await self._deliver_pending_turn(pending)
            return False

        if not pending.turn_id:
            self._persist_turn_result(
                {"status": "unknown", "error_key": "recovery.label.unknown"}
            )
            await self._deliver_pending_turn(pending)
            return False

        try:
            turn = await self._read_persisted_turn(pending.thread_id, pending.turn_id)
            if turn is None:
                self._persist_turn_result(
                    {"status": "unknown", "error_key": "recovery.label.unknown"}
                )
                await self._deliver_pending_turn(pending)
                return False

            turn_status = turn.get("status")
            if turn_status in {"completed", "interrupted", "failed"}:
                self._persist_turn_result(
                    {
                        "status": turn_status,
                        "error": self._turn_error_message(turn),
                    }
                )
                await self._deliver_pending_turn(pending)
                return False
            if turn_status != "inProgress":
                self._persist_turn_result(
                    {"status": "unknown", "error_key": "recovery.label.unknown"}
                )
                await self._deliver_pending_turn(pending)
                return False

            self._turn_completed_event = asyncio.Event()
            await self._resume_thread(pending.thread_id)
            self._loaded_thread_id = pending.thread_id
            runtime = await self.app_server.request(
                "thread/read", {"threadId": pending.thread_id}, timeout_seconds=30
            )
            if self._turn_completed_event.is_set():
                if pending.phase == "delivery_pending":
                    await self._deliver_pending_turn(pending)
                await self._unsubscribe_thread()
                return False
            thread = runtime.get("thread")
            status = thread.get("status") if isinstance(thread, dict) else None
            status_type = status.get("type") if isinstance(status, dict) else None
            if status_type != "active":
                self._persist_turn_result(
                    {"status": "unknown", "error_key": "recovery.label.unknown"}
                )
                await self._deliver_pending_turn(pending)
                await self._unsubscribe_thread()
                return False

            active_flags = status.get("activeFlags", []) if isinstance(status, dict) else []
            waiting_for_user = isinstance(active_flags, list) and any(
                flag in {"waitingOnApproval", "waitingOnUserInput"}
                for flag in active_flags
                if isinstance(flag, str)
            )
            if pending.prompt_message_ids or waiting_for_user:
                with bind_log_context(interrupt_source="recovery_prompt_expired"):
                    await self._request_turn_interrupt(
                        pending.thread_id,
                        pending.turn_id,
                        timeout_seconds=15,
                    )
                if pending.phase != "delivery_pending":
                    self._persist_turn_result(
                        {
                            "status": "interrupted",
                            "error_key": "recovery.label.prompt_expired",
                        }
                    )
                await self._deliver_pending_turn(pending)
                await self._unsubscribe_thread()
                return False

            if self.state.thread_id != pending.thread_id:
                self._persist_turn_result(
                    {"status": "unknown", "error_key": "recovery.label.unknown"}
                )
                await self._deliver_pending_turn(pending)
                await self._unsubscribe_thread()
                return False

            self._set_active_turn_id(pending.turn_id)
            self._recovered_turn_id = pending.turn_id
            return True
        except Exception as exc:
            log_exception(
                logger,
                "Could not reconcile Codex turn after restart",
                exc,
                event="codex.recovery.restart.reconcile_failed",
                level=logging.WARNING,
                resolve_error=resolve_error,
                pending_phase=pending.phase,
                **self._thread_log_fields(pending.thread_id),
                **self._turn_log_fields(pending.turn_id),
            )
            if not resolve_error:
                raise
            if pending.phase != "delivery_pending":
                self._persist_turn_result(
                    {"status": "unknown", "error_key": "recovery.label.unknown"}
                )
            await self._deliver_pending_turn(pending)
            await self._unsubscribe_thread(source="turn_recovery")
            return False





    @property
    def language(self) -> str:
        return self.state.language

    def _text(self, key: str, **values: Any) -> str:
        return self.texts.render(self.language, key, **values)

    def select_language(self, language: str) -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError("Unsupported language")
        self.state.language = language
        self.state_store.save(self.state)

    def select_output_mode(self, mode: str) -> None:
        if mode not in OUTPUT_MODE_CHOICES:
            raise ValueError("Unsupported output mode")
        self.state.quiet_progress = mode == "quiet_answers"
        self.state.output_mode = "answers" if mode == "quiet_answers" else mode
        self.state_store.save(self.state)

    def select_permission_mode(self, mode: str) -> None:
        if mode not in PERMISSION_MODES:
            raise ValueError("Unsupported permission mode")
        if self.state.thread_id:
            thread_state = self.state_store.load_thread(self.state.thread_id)
            thread_state.permission_mode = mode
            thread_state.managed_by_telecodex = True
            self.state_store.save_thread(self.state.thread_id, thread_state)
            return
        self.state.default_permission_mode = mode
        self.state_store.save(self.state)

    async def get_goal(self) -> CodexGoal | None:
        if not self.is_busy:
            await self._wait_for_turn_completion()
        thread_id = self.state.thread_id
        if not thread_id:
            return None
        await self._ensure_thread(create_if_missing=False)
        goal = await self._read_goal(thread_id)
        if not self.is_busy:
            await self._unsubscribe_thread(source="get_goal")
        return goal

    async def set_goal(self, objective: str, prompt_message_id: int | None = None) -> CodexGoal:
        objective = objective.strip()
        if not objective:
            raise ValueError("A Goal objective is required")
        if not self.is_busy:
            await self._wait_for_turn_completion()
        await self._ensure_thread(create_if_missing=True)
        thread_id = self.state.thread_id
        if not thread_id:
            raise RuntimeError("Codex app-server did not create a thread for the Goal")

        previous_goal = self._goal if self._goal_thread_id == thread_id else None
        previous_prompt_message_id = self._goal_prompt_message_id
        self._goal_prompt_message_id = prompt_message_id
        self._goal = CodexGoal(objective, "active", None, 0, 0)
        self._goal_thread_id = thread_id
        if not self.is_busy:
            self._goal_turn_started_event.clear()
        try:
            result = await self._goal_request(
                "thread/goal/set",
                {
                    "threadId": thread_id,
                    "objective": objective,
                    "status": "active",
                    "origin": "user",
                },
            )
        except Exception:
            self._goal = previous_goal
            self._goal_thread_id = thread_id if previous_goal else None
            self._goal_prompt_message_id = previous_prompt_message_id
            raise
        goal = self._remember_goal(thread_id, result.get("goal"))
        if goal is None:
            goal = await self._read_goal(thread_id)
        if goal is None:
            raise RuntimeError("Codex app-server did not save the Goal")
        if goal.status == "active":
            await self._ensure_goal_continuation_started(thread_id, prompt_message_id)
        logger.info(
            "Codex Goal updated",
            extra={
                "event": "codex.goal.updated",
                "goal_status": goal.status,
                "objective_length": len(goal.objective),
                **self._thread_log_fields(thread_id),
            },
        )
        return goal

    async def set_goal_status(
        self,
        status: str,
        prompt_message_id: int | None = None,
    ) -> CodexGoal:
        if status not in {"active", "paused"}:
            raise ValueError("Unsupported user Goal status")
        thread_id = self.state.thread_id
        if not thread_id:
            raise RuntimeError("There is no active Codex chat")
        if not self.is_busy:
            await self._wait_for_turn_completion()
        await self._ensure_thread(create_if_missing=False)
        current_goal = (
            self._goal
            if self._goal_thread_id == thread_id
            else await self._read_goal(thread_id)
        )
        if current_goal is None:
            raise GoalNotFoundError("There is no Goal in the active Codex chat")

        previous_prompt_message_id = self._goal_prompt_message_id
        self._goal = CodexGoal(
            current_goal.objective,
            status,
            current_goal.token_budget,
            current_goal.tokens_used,
            current_goal.time_used_seconds,
        )
        self._goal_thread_id = thread_id
        if status == "active" and prompt_message_id is not None:
            self._goal_prompt_message_id = prompt_message_id
        if status == "active" and not self.is_busy:
            self._goal_turn_started_event.clear()
        try:
            result = await self._goal_request(
                "thread/goal/set",
                {"threadId": thread_id, "status": status, "origin": "user"},
            )
        except Exception:
            self._goal = current_goal
            self._goal_prompt_message_id = previous_prompt_message_id
            raise
        goal = self._remember_goal(thread_id, result.get("goal"))
        if goal is None:
            goal = await self._read_goal(thread_id)
        if goal is None:
            raise RuntimeError("Codex app-server did not return the updated Goal")
        logger.info(
            "Codex Goal status changed",
            extra={
                "event": "codex.goal.status_changed",
                "goal_status": goal.status,
                **self._thread_log_fields(thread_id),
            },
        )
        if status == "active" and goal.status == "active":
            await self._ensure_goal_continuation_started(
                thread_id,
                prompt_message_id,
            )
        if not self.is_busy and goal.status != "active":
            await self._unsubscribe_thread()
        return goal

    async def clear_goal(self) -> bool:
        thread_id = self.state.thread_id
        if not thread_id:
            return False
        if not self.is_busy:
            await self._wait_for_turn_completion()
        await self._ensure_thread(create_if_missing=False)
        current_goal = (
            self._goal
            if self._goal_thread_id == thread_id
            else await self._read_goal(thread_id)
        )
        if current_goal is None:
            return False
        cleared = False
        try:
            result = await self._goal_request(
                "thread/goal/clear",
                {"threadId": thread_id, "origin": "user"},
            )
        except RpcError as exc:
            message = str(exc).casefold()
            if "thread" not in message or "not found" not in message:
                raise
            self._loaded_thread_id = None
            await self._ensure_thread(create_if_missing=False)
            if self.state.thread_id != thread_id:
                raise RuntimeError("The active Codex chat changed while clearing its Goal") from exc
            current_goal = (
                self._goal
                if self._goal_thread_id == thread_id
                else await self._read_goal(thread_id)
            )
            if current_goal is None:
                cleared = True
                result = {"cleared": True}
            else:
                result = await self._goal_request(
                    "thread/goal/clear",
                    {"threadId": thread_id, "origin": "user"},
                )
        self._remember_goal(thread_id, None)
        if not self.is_busy:
            self._goal_prompt_message_id = None
        pending = self.state.pending_turn
        if (
            not self.is_busy
            and pending is not None
            and pending.thread_id == thread_id
            and pending.phase == "starting"
            and pending.turn_id is None
        ):
            self.clear_pending_turn()
        logger.info(
            "Codex Goal cleared",
            extra={"event": "codex.goal.cleared", **self._thread_log_fields(thread_id)},
        )
        if not self.is_busy:
            await self._unsubscribe_thread()
        return cleared or result.get("cleared") is True

    async def rename_thread(self, name: str) -> None:
        if not self.state.thread_id:
            raise RuntimeError("There is no active Codex chat to rename")
        await self._ensure_thread(create_if_missing=False)
        await self.app_server.request(
            "thread/name/set",
            {"threadId": self.state.thread_id, "name": name},
            timeout_seconds=30,
        )
        if not self.is_busy:
            await self._unsubscribe_thread()

    async def delete_thread(self, thread_id: str) -> None:
        await self._wait_for_turn_completion()
        if not thread_id or self.state.thread_id != thread_id:
            raise RuntimeError("The active Codex chat changed; run /delete again")
        if self.is_busy:
            raise RuntimeError("The Codex chat is busy")
        await self._ensure_thread(create_if_missing=False)
        if self.state.thread_id != thread_id or self.is_busy:
            raise RuntimeError("The active Codex chat changed or became busy")
        try:
            await self.app_server.request("thread/delete", {"threadId": thread_id}, timeout_seconds=60)
        except Exception:
            await self._unsubscribe_thread()
            raise
        self.state_store.delete_threads({thread_id})
        self.state.thread_id = None
        self._loaded_thread_id = None
        self._goal = None
        self._goal_thread_id = None
        self._goal_prompt_message_id = None
        self.settings_manager.reset_thread_settings()
        self._clear_turn_messages()
        self.state_store.save(self.state)

    async def fork_thread(self) -> str:
        await self._wait_for_turn_completion()
        if not self.state.thread_id:
            raise RuntimeError("There is no active Codex chat to fork")
        if self.is_busy:
            raise RuntimeError("The Codex chat is busy")
        await self._ensure_thread(create_if_missing=False)
        source_thread_id = self.state.thread_id
        if not source_thread_id or self.is_busy:
            raise RuntimeError("The active Codex chat changed or became busy")

        source_state = self.state_store.load_thread(source_thread_id)
        source_goal: CodexGoal | None = None
        try:
            source_goal = await self._read_goal(source_thread_id)
        except Exception as exc:
            logger.debug(
                "Could not read Goal state before forking a Codex thread",
                extra={
                    "event": "codex.goal.fork_source_read_failed",
                    "error_type": type(exc).__name__,
                    **self._thread_log_fields(source_thread_id),
                },
            )
        fork_params: dict[str, Any] = {"threadId": source_thread_id, "excludeTurns": True}
        if source_goal is not None and source_goal.status == "active":
            fork_params["deferGoalContinuation"] = True
        try:
            result = await self.app_server.request(
                "thread/fork",
                fork_params,
                timeout_seconds=90,
            )
        except Exception:
            await self._unsubscribe_thread()
            raise
        thread = result.get("thread")
        if not isinstance(thread, dict):
            raise RuntimeError("Codex app-server did not return the forked thread")
        forked_thread_id = thread.get("id")
        if not isinstance(forked_thread_id, str) or not forked_thread_id:
            raise RuntimeError("Codex app-server did not return the forked thread ID")

        cwd_value = thread.get("cwd") or result.get("cwd") or self.state.cwd
        cwd = self._validated_workspace_path(Path(cwd_value)) if isinstance(cwd_value, str) else self.working_directory
        self.state.thread_id = forked_thread_id
        self._goal = None
        self._goal_thread_id = None
        self._goal_prompt_message_id = None
        self.state.cwd = str(cwd)
        self.state_store.save_thread(
            forked_thread_id,
            ThreadRuntimeState(
                permission_mode=source_state.permission_mode,
                memory_mode=source_state.memory_mode,
                managed_by_telecodex=source_state.managed_by_telecodex,
                token_usage=(dict(source_state.token_usage) if source_state.token_usage else None),
            ),
        )
        self.state_store.save(self.state)
        self._loaded_thread_id = forked_thread_id
        self._clear_turn_messages()
        try:
            forked_goal = await self._read_goal(forked_thread_id)
        except Exception as exc:
            forked_goal = None
            logger.debug(
                "Could not read Goal state on a forked Codex thread",
                extra={
                    "event": "codex.goal.fork_read_failed",
                    "error_type": type(exc).__name__,
                    **self._thread_log_fields(forked_thread_id),
                },
            )
        if (
            source_goal is not None
            and source_goal.status == "active"
            and forked_goal is not None
            and forked_goal.status == "active"
        ):
            await self.set_goal_status("active")
        await self._unsubscribe_thread()
        return forked_thread_id

    async def fork_thread_by_id(self, source_thread_id: str, cwd: Path) -> str:
        """Fork a listed thread directly, without first resuming its active writer."""
        await self._wait_for_turn_completion()
        if self.is_busy:
            raise RuntimeError("The Codex chat is busy")
        cwd = self._validated_workspace_path(cwd)
        goal_state_known = self._goal_thread_id == source_thread_id
        source_goal = self._goal if goal_state_known else None

        await self.app_server.stop()
        self._loaded_thread_id = None
        self._clear_turn_messages()
        await self._ensure_app_server()

        if not goal_state_known:
            try:
                source_goal = await self._read_goal(source_thread_id)
            except Exception as exc:
                logger.debug(
                    "Could not read Goal state before forking a listed Codex thread",
                    extra={
                        "event": "codex.goal.fork_source_read_failed",
                        "error_type": type(exc).__name__,
                        **self._thread_log_fields(source_thread_id),
                    },
                )

        source_state = self.state_store.load_thread(source_thread_id)
        fork_params: dict[str, Any] = {"threadId": source_thread_id, "excludeTurns": True}
        if source_goal is not None and source_goal.status == "active":
            fork_params["deferGoalContinuation"] = True
        try:
            result = await self.app_server.request(
                "thread/fork",
                fork_params,
                timeout_seconds=90,
            )
        except RpcError as exc:
            if _is_active_thread_writer_error(exc):
                raise ActiveThreadWriterError(
                    "The selected Codex chat is still active in another session"
                ) from exc
            raise

        thread = result.get("thread")
        if not isinstance(thread, dict):
            raise RuntimeError("Codex app-server did not return the forked thread")
        forked_thread_id = thread.get("id")
        if not isinstance(forked_thread_id, str) or not forked_thread_id:
            raise RuntimeError("Codex app-server did not return the forked thread ID")

        cwd_value = thread.get("cwd") or result.get("cwd")
        forked_cwd = (
            self._validated_workspace_path(Path(cwd_value))
            if isinstance(cwd_value, str)
            else cwd
        )
        self.state.thread_id = forked_thread_id
        self._goal = None
        self._goal_thread_id = None
        self._goal_prompt_message_id = None
        self.state.cwd = str(forked_cwd)
        self.state_store.save_thread(
            forked_thread_id,
            ThreadRuntimeState(
                permission_mode=source_state.permission_mode,
                memory_mode=source_state.memory_mode,
                managed_by_telecodex=source_state.managed_by_telecodex,
                token_usage=(dict(source_state.token_usage) if source_state.token_usage else None),
            ),
        )
        self.state_store.save(self.state)
        self._loaded_thread_id = forked_thread_id
        try:
            forked_goal = await self._read_goal(forked_thread_id)
        except Exception as exc:
            forked_goal = None
            logger.debug(
                "Could not read Goal state on a forked Codex thread",
                extra={
                    "event": "codex.goal.fork_read_failed",
                    "error_type": type(exc).__name__,
                    **self._thread_log_fields(forked_thread_id),
                },
            )
        if (
            source_goal is not None
            and source_goal.status == "active"
            and forked_goal is not None
            and forked_goal.status == "active"
        ):
            await self.set_goal_status("active")
        await self._unsubscribe_thread()
        return forked_thread_id

    async def compact_thread(self) -> None:
        if not self.state.thread_id:
            raise RuntimeError("There is no active Codex chat to compact")
        if self._manual_compaction_requested or self._manual_compaction_future is not None:
            raise CodexChatBusyError("A Codex compaction is already pending")
        await self._ensure_thread(create_if_missing=False)
        thread_id = self.state.thread_id
        if not thread_id:
            raise RuntimeError("The active Codex chat changed before compaction")

        async with self._thread_use_lock:
            if self.state.thread_id != thread_id:
                raise RuntimeError("The active Codex chat changed before compaction")
            if self._manual_compaction_requested or self._manual_compaction_future is not None:
                raise CodexChatBusyError("A Codex compaction is already pending")
            self._manual_compaction_requested = True
            self._manual_compaction_barrier_future = asyncio.get_running_loop().create_future()
            self._manual_compaction_task = asyncio.create_task(
                self._run_pending_manual_compaction(self._manual_compaction_barrier_future)
            )

    async def wait_for_manual_compaction_barrier(self) -> None:
        barrier = self._manual_compaction_barrier_future
        if barrier is not None:
            await asyncio.shield(barrier)

    async def wait_for_current_turn_completion(self) -> None:
        while True:
            if self.active_turn_id is not None:
                event = self._turn_completed_event
                if event is None or event.is_set():
                    await asyncio.sleep(0.1)
                else:
                    await event.wait()
                continue

            await self._wait_for_turn_completion()
            if self.active_turn_id is None:
                return

    async def cancel_pending_manual_compaction(self) -> bool:
        if not self._manual_compaction_requested or self._manual_compaction_future is not None:
            return False
        self._manual_compaction_requested = False
        task = self._manual_compaction_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        barrier = self._manual_compaction_barrier_future
        if barrier is not None and not barrier.done():
            barrier.set_result(None)
        return True

    async def _run_pending_manual_compaction(
        self,
        barrier: asyncio.Future[None],
    ) -> None:
        cancelled = False
        queued_notice_sent = False

        def notify_queued() -> None:
            nonlocal queued_notice_sent
            if not queued_notice_sent:
                queued_notice_sent = True
                self._schedule_status("status.compaction_queued")

        try:
            while self._manual_compaction_requested:
                if self._recovery_in_progress or self._recovery_blocked:
                    notify_queued()
                    await asyncio.sleep(0.1)
                    continue
                if self.active_turn_id is not None or self.has_active_command_execution:
                    notify_queued()
                    event = self._turn_completed_event
                    if event is None or event.is_set():
                        await asyncio.sleep(MANUAL_COMPACTION_IDLE_POLL_SECONDS)
                    else:
                        try:
                            await asyncio.wait_for(
                                event.wait(),
                                timeout=MANUAL_COMPACTION_IDLE_POLL_SECONDS,
                            )
                        except asyncio.TimeoutError:
                            pass
                    continue

                completion_task = self._turn_completion_task
                if completion_task is not None and not completion_task.done():
                    notify_queued()
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(completion_task),
                            timeout=MANUAL_COMPACTION_IDLE_POLL_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        continue
                elif completion_task is not None and self._turn_completion_task is completion_task:
                    self._turn_completion_task = None

                pending_turn = self.state.pending_turn
                if pending_turn is not None and pending_turn.phase in {"starting", "running"}:
                    notify_queued()
                    await asyncio.sleep(MANUAL_COMPACTION_IDLE_POLL_SECONDS)
                    continue
                if pending_turn is not None and pending_turn.phase == "delivery_pending":
                    notify_queued()
                    await self._deliver_pending_turn(pending_turn)
                    if self.state.pending_turn is not None:
                        await asyncio.sleep(MANUAL_COMPACTION_IDLE_POLL_SECONDS)
                        continue

                thread_id = self.state.thread_id
                if not thread_id:
                    raise RuntimeError("The active Codex chat disappeared before compaction")
                try:
                    await self._ensure_thread(create_if_missing=False)
                    thread_response = await self.app_server.request(
                        "thread/read",
                        {"threadId": thread_id},
                        timeout_seconds=30,
                    )
                    thread = thread_response.get("thread")
                    status = thread.get("status") if isinstance(thread, dict) else None
                    status_type = status.get("type") if isinstance(status, dict) else None
                    if status_type != "idle":
                        notify_queued()
                        await asyncio.sleep(MANUAL_COMPACTION_IDLE_POLL_SECONDS)
                        continue
                    terminals = await self.list_background_terminals()
                except RpcError as exc:
                    if "unknown method" in str(exc).casefold() or "method not found" in str(exc).casefold():
                        raise RuntimeError(
                            "Codex cannot confirm whether background terminals are still running"
                        ) from exc
                    logger.warning(
                        "Could not check whether Codex is idle before manual compaction",
                        extra={
                            "event": "codex.compaction.idle_check_failed",
                            "error_type": type(exc).__name__,
                            **self._thread_log_fields(thread_id),
                        },
                    )
                    notify_queued()
                    await asyncio.sleep(MANUAL_COMPACTION_IDLE_POLL_SECONDS)
                    continue
                except Exception as exc:
                    logger.warning(
                        "Could not confirm Codex idle state before manual compaction",
                        extra={
                            "event": "codex.compaction.idle_check_failed",
                            "error_type": type(exc).__name__,
                            **self._thread_log_fields(thread_id),
                        },
                    )
                    notify_queued()
                    await asyncio.sleep(MANUAL_COMPACTION_IDLE_POLL_SECONDS)
                    continue

                if terminals:
                    notify_queued()
                    await asyncio.sleep(MANUAL_COMPACTION_IDLE_POLL_SECONDS)
                    continue

                retry_after_lock = False
                future: asyncio.Future[None] | None = None
                async with self._thread_use_lock:
                    if (
                        not self._manual_compaction_requested
                        or self.state.thread_id != thread_id
                        or self.active_turn_id is not None
                        or self.has_active_command_execution
                        or (
                            self.state.pending_turn is not None
                            and self.state.pending_turn.phase in {"starting", "running"}
                        )
                    ):
                        retry_after_lock = True
                    else:
                        try:
                            terminals = await self.list_background_terminals()
                        except RpcError as exc:
                            if "unknown method" in str(exc).casefold() or "method not found" in str(exc).casefold():
                                raise RuntimeError(
                                    "Codex cannot confirm whether background terminals are still running"
                                ) from exc
                            logger.warning(
                                "Could not recheck Codex background terminals before compaction",
                                extra={
                                    "event": "codex.compaction.background_terminal_recheck_failed",
                                    "error_type": type(exc).__name__,
                                    **self._thread_log_fields(thread_id),
                                },
                            )
                            retry_after_lock = True
                        except Exception as exc:
                            logger.warning(
                                "Could not recheck Codex background terminals before compaction",
                                extra={
                                    "event": "codex.compaction.background_terminal_recheck_failed",
                                    "error_type": type(exc).__name__,
                                    **self._thread_log_fields(thread_id),
                                },
                            )
                            retry_after_lock = True
                        else:
                            if terminals:
                                retry_after_lock = True
                            elif (
                                self.active_turn_id is not None
                                or self.has_active_command_execution
                                or (
                                    self.state.pending_turn is not None
                                    and self.state.pending_turn.phase in {"starting", "running"}
                                )
                            ):
                                retry_after_lock = True
                            else:
                                try:
                                    thread_response = await self.app_server.request(
                                        "thread/read",
                                        {"threadId": thread_id},
                                        timeout_seconds=30,
                                    )
                                    thread = thread_response.get("thread")
                                    status = thread.get("status") if isinstance(thread, dict) else None
                                    status_type = status.get("type") if isinstance(status, dict) else None
                                except Exception as exc:
                                    logger.warning(
                                        "Could not recheck the Codex thread before compaction",
                                        extra={
                                            "event": "codex.compaction.thread_recheck_failed",
                                            "error_type": type(exc).__name__,
                                            **self._thread_log_fields(thread_id),
                                        },
                                    )
                                    retry_after_lock = True
                                else:
                                    if status_type != "idle":
                                        retry_after_lock = True
                                    elif (
                                        self.active_turn_id is not None
                                        or self.has_active_command_execution
                                        or not self._manual_compaction_requested
                                        or (
                                            self.state.pending_turn is not None
                                            and self.state.pending_turn.phase
                                            in {"starting", "running"}
                                        )
                                    ):
                                        retry_after_lock = True
                                    else:
                                        future = asyncio.get_running_loop().create_future()
                                        self._manual_compaction_future = future
                                        self._manual_compaction_turn_id = None
                                        self._manual_compaction_item_completed = False
                                        self._manual_compaction_started_at = monotonic()
                                        self._turn_completed_event = asyncio.Event()
                                        try:
                                            await self.app_server.request(
                                                "thread/compact/start",
                                                {"threadId": thread_id},
                                                timeout_seconds=30,
                                            )
                                        except Exception as exc:
                                            logger.warning(
                                                "Codex rejected or did not confirm the manual compaction request",
                                                extra={
                                                    "event": "codex.compaction.start_unconfirmed",
                                                    "error_type": type(exc).__name__,
                                                    **self._thread_log_fields(thread_id),
                                                },
                                            )

                if retry_after_lock:
                    notify_queued()
                    await asyncio.sleep(MANUAL_COMPACTION_IDLE_POLL_SECONDS)
                    continue
                if future is None:
                    continue

                await self._wait_for_manual_compaction_terminal(future)
                while self._turn_completion_task is not None and not self._turn_completion_task.done():
                    await asyncio.sleep(0.1)
                if self._turn_completion_task is not None:
                    self._turn_completion_task = None
                break
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            log_exception(
                logger,
                "Could not complete queued Codex context compaction",
                exc,
                event="codex.compaction.deferred.failed",
                level=logging.WARNING,
                **self._thread_log_fields(),
            )
            self._schedule_status("status.label.compaction_failed")
        finally:
            if self._manual_compaction_task is asyncio.current_task():
                self._manual_compaction_requested = False
                self._manual_compaction_task = None
                self._manual_compaction_barrier_future = None
                if not barrier.done():
                    barrier.set_result(None)
                if not cancelled and self._goal_is_active() and self.state.thread_id:
                    self._schedule_callback(
                        self._ensure_goal_continuation_started(
                            self.state.thread_id,
                            self._goal_prompt_message_id,
                        ),
                        "Codex Goal continuation after manual compaction",
                    )

    async def _wait_for_manual_compaction_terminal(
        self,
        future: asyncio.Future[None],
    ) -> None:
        while self._manual_compaction_future is future:
            try:
                await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=TURN_COMPLETION_WAIT_SECONDS,
                )
            except asyncio.TimeoutError:
                recovered = await self._recover_manual_compaction_if_idle(future)
                if not recovered:
                    continue

    async def inspect_current_thread_for_recovery(self) -> CodexRecoverySnapshot:
        """Read the current thread and its background processes before recovery."""
        if self._recovery_in_progress or self._recovery_lock.locked():
            raise CodexChatBusyError("The Codex chat is already being recovered")
        try:
            return await self._read_recovery_snapshot()
        except Exception:
            self._recovery_blocked = True
            self._recovery_hold = True
            raise

    async def recover_current_thread(
        self,
        expected_thread_id: str,
        *,
        confirmed_terminal_ids: set[str] | frozenset[str] = frozenset(),
        confirm_interrupt: bool = False,
    ) -> CodexRecoverySnapshot | None:
        """Restart this context's app-server and resume the same idle thread.

        Returns a fresh snapshot when new work or terminals appeared and need
        separate confirmation. Raises on an unconfirmed stop/resume so callers
        can keep the chat blocked instead of sending into an uncertain writer.
        """
        if self._recovery_lock.locked() or self._recovery_in_progress:
            raise CodexChatBusyError("The Codex chat is already being recovered")
        async with self._recovery_lock:
            if self._recovery_in_progress:
                raise CodexChatBusyError("The Codex chat is already being recovered")
            was_blocked = self._recovery_blocked
            self._recovery_in_progress = True
            self._recovery_hold = True
            mutated = False
            old_server_stopped = False
            try:
                if self.state.thread_id != expected_thread_id:
                    raise RuntimeError("The selected Codex chat changed before recovery")

                try:
                    snapshot = await self._read_recovery_snapshot()
                except Exception:
                    self._recovery_blocked = True
                    mutated = True
                    raise
                if snapshot.thread_id != expected_thread_id:
                    raise RuntimeError("The selected Codex chat changed before recovery")
                unconfirmed_terminals = {
                    item.process_id for item in snapshot.terminals
                } - set(confirmed_terminal_ids)
                if (snapshot.has_active_work and not confirm_interrupt) or unconfirmed_terminals:
                    return snapshot
                self._recovery_blocked = True
                mutated = True
                await self._wait_for_recovery_completion_callback()

                if snapshot.terminals:
                    self._recovery_blocked = True
                    mutated = True
                    for terminal in snapshot.terminals:
                        if terminal.process_id not in confirmed_terminal_ids:
                            continue
                        if not await self.terminate_background_terminal(terminal.process_id):
                            raise RuntimeError("Codex did not confirm stopping a background terminal")
                    remaining: list[CodexBackgroundTerminal] = []
                    for attempt in range(4):
                        remaining = await self.list_background_terminals()
                        if not remaining:
                            break
                        if attempt < 3:
                            await asyncio.sleep(0.2)
                    if remaining:
                        unconfirmed = {
                            item.process_id for item in remaining
                        } - set(confirmed_terminal_ids)
                        if unconfirmed:
                            return await self._read_recovery_snapshot()
                        raise RuntimeError("Codex still reports a background terminal after stopping it")

                snapshot = await self._read_recovery_snapshot()
                if snapshot.has_active_work and not confirm_interrupt:
                    return snapshot
                if snapshot.terminals:
                    unconfirmed = {
                        item.process_id for item in snapshot.terminals
                    } - set(confirmed_terminal_ids)
                    if unconfirmed:
                        return snapshot
                    raise RuntimeError("A background terminal appeared during recovery")

                pending = self.state.pending_turn
                if pending is not None and pending.phase == "delivery_pending":
                    await self._deliver_pending_turn(pending)
                    pending = self.state.pending_turn
                    if pending is not None:
                        self._recovery_blocked = True
                        mutated = True
                        raise RuntimeError("The previous Codex result could not be delivered")

                persisted_status: str | None = None
                if snapshot.active_turn_id:
                    persisted_turn: dict[str, Any] | None = None
                    try:
                        persisted_turn = await self._read_persisted_turn(
                            expected_thread_id,
                            snapshot.active_turn_id,
                        )
                    except Exception as exc:
                        log_exception(
                            logger,
                            "Could not reconcile Codex turn before requesting interruption",
                            exc,
                            event="codex.recovery.turn_preflight_failed",
                            level=logging.WARNING,
                            **self._thread_log_fields(expected_thread_id),
                            **self._turn_log_fields(snapshot.active_turn_id),
                        )
                    persisted_status = (
                        persisted_turn.get("status")
                        if isinstance(persisted_turn, dict)
                        else None
                    )
                    if persisted_status in {"completed", "interrupted", "failed"}:
                        await self._wait_for_recovery_completion_callback()
                        pending = self.state.pending_turn
                        if pending is not None and pending.phase != "delivery_pending":
                            self._persist_turn_result(
                                {
                                    "status": persisted_status,
                                    "error": self._turn_error_message(persisted_turn),
                                }
                            )
                        self.active_turn_id = None
                        self._clear_active_command_executions()
                        if self._turn_completed_event is not None:
                            self._turn_completed_event.set()
                        pending = self.state.pending_turn
                        if pending is not None and pending.phase == "delivery_pending":
                            await self._deliver_pending_turn(pending)
                        snapshot = await self._read_recovery_snapshot()
                    elif snapshot.thread_status == "idle" and persisted_status != "inProgress":
                        pending = self.state.pending_turn
                        if pending is not None and pending.phase != "delivery_pending":
                            self._persist_turn_result(
                                {"status": "unknown", "error_key": "recovery.label.unknown"}
                            )
                        self.active_turn_id = None
                        self._clear_active_command_executions()
                        if self._turn_completed_event is not None:
                            self._turn_completed_event.set()
                        pending = self.state.pending_turn
                        if pending is not None and pending.phase == "delivery_pending":
                            await self._deliver_pending_turn(pending)
                        if self.state.pending_turn is not None:
                            self._recovery_blocked = True
                            mutated = True
                            raise RuntimeError("The previous Codex result could not be delivered")
                        snapshot = await self._read_recovery_snapshot()

                turn_id = snapshot.active_turn_id
                needs_interrupt = (
                    snapshot.thread_status != "idle"
                    or persisted_status == "inProgress"
                )
                if needs_interrupt:
                    if not turn_id:
                        self._recovery_blocked = True
                        mutated = True
                        raise RuntimeError(
                            "Codex reports active work but its turn ID could not be confirmed"
                        )
                    self._recovery_blocked = True
                    mutated = True
                    if self._goal_is_active(expected_thread_id):
                        await self.set_goal_status("paused")
                    await self._request_turn_interrupt(
                        expected_thread_id,
                        turn_id,
                        timeout_seconds=15,
                    )
                    logger.info(
                        "Accepted Codex turn interruption for chat recovery",
                        extra={
                            "event": "codex.recovery.interrupt.accepted",
                            **self._thread_log_fields(expected_thread_id),
                            **self._turn_log_fields(turn_id),
                        },
                    )
                    await self._reconcile_recovery_turn(
                        expected_thread_id,
                        turn_id,
                    )

                if self.state.pending_turn is not None:
                    pending = self.state.pending_turn
                    if pending.phase != "delivery_pending":
                        self._persist_turn_result(
                            {"status": "unknown", "error_key": "recovery.label.unknown"}
                        )

                self._recovery_blocked = True
                mutated = True
                logger.warning(
                    "Restarting the Codex app-server for the selected chat",
                    extra={
                        "event": "codex.recovery.app_server.restart_started",
                        **self._thread_log_fields(expected_thread_id),
                    },
                )
                if self.app_server.is_running:
                    await self.app_server.stop()
                    if self.app_server.is_running:
                        raise RuntimeError("Codex app-server shutdown was not confirmed")
                    old_server_stopped = True
                elif getattr(self.app_server, "is_running", False):
                    raise RuntimeError("Codex app-server shutdown was not confirmed")
                else:
                    old_server_stopped = True

                await self._clear_recovery_transient_state()
                pending = self.state.pending_turn
                if pending is not None and pending.phase == "delivery_pending":
                    await self._deliver_pending_turn(pending)
                    if self.state.pending_turn is not None:
                        raise RuntimeError("The previous Codex result could not be delivered")

                await self._ensure_app_server()
                await self._ensure_thread(create_if_missing=False)
                if self.state.thread_id != expected_thread_id:
                    raise RuntimeError("Codex changed the selected chat while resuming it")
                response = await self.app_server.request(
                    "thread/read",
                    {"threadId": expected_thread_id},
                    timeout_seconds=30,
                )
                self._capture_thread_settings(response)
                thread = response.get("thread")
                thread = thread if isinstance(thread, dict) else {}
                if thread.get("id") != expected_thread_id:
                    raise RuntimeError("Codex did not return the selected chat after recovery")
                status = thread.get("status")
                status_type = status.get("type") if isinstance(status, dict) else None
                if status_type != "idle":
                    snapshot = await self._read_recovery_snapshot()
                    self._recovery_blocked = True
                    return snapshot
                terminals = await self.list_background_terminals()
                if terminals:
                    self._recovery_blocked = True
                    return CodexRecoverySnapshot(
                        thread_id=expected_thread_id,
                        thread_status="idle",
                        active_turn_id=None,
                        terminals=tuple(terminals),
                    )

                self._recovery_blocked = False
                self._recovery_hold = False
                logger.info(
                    "Recovered Codex chat and verified its thread is idle",
                    extra={
                        "event": "codex.recovery.completed",
                        "outcome": "same_thread_idle",
                        **self._thread_log_fields(expected_thread_id),
                    },
                )
                return None
            except Exception:
                if mutated or was_blocked:
                    self._recovery_blocked = True
                if old_server_stopped and self.app_server.is_running:
                    try:
                        await self.app_server.stop()
                    except Exception as stop_error:
                        log_exception(
                            logger,
                            "Could not stop Codex app-server after recovery resume failed",
                            stop_error,
                            event="codex.recovery.resume_failure_stop_failed",
                            level=logging.ERROR,
                            **self._thread_log_fields(expected_thread_id),
                        )
                    self._loaded_thread_id = None
                raise
            finally:
                self._recovery_in_progress = False
                if not mutated and not was_blocked:
                    self._recovery_blocked = False

    async def _read_recovery_snapshot(self) -> CodexRecoverySnapshot:
        thread_id = self.state.thread_id
        if not thread_id:
            raise RuntimeError("There is no selected Codex chat to recover")
        await self._ensure_thread(create_if_missing=False)
        if self.state.thread_id != thread_id:
            raise RuntimeError("The selected Codex chat changed during recovery inspection")
        response = await self.app_server.request(
            "thread/read",
            {"threadId": thread_id},
            timeout_seconds=30,
        )
        thread = response.get("thread")
        if not isinstance(thread, dict) or thread.get("id") != thread_id:
            raise RuntimeError("Codex did not confirm the selected chat during recovery inspection")
        status = thread.get("status")
        status_type = status.get("type") if isinstance(status, dict) else None
        local_turn_id = self.active_turn_id or self._manual_compaction_turn_id
        pending = self.state.pending_turn
        if (
            not local_turn_id
            and pending is not None
            and pending.thread_id == thread_id
            and pending.phase in {"starting", "running"}
        ):
            local_turn_id = pending.turn_id
        if status_type == "active" and not local_turn_id:
            local_turn_id = await self._find_active_turn_id(thread_id)
        terminals = await self.list_background_terminals()
        return CodexRecoverySnapshot(
            thread_id=thread_id,
            thread_status=status_type if isinstance(status_type, str) else None,
            active_turn_id=local_turn_id,
            terminals=tuple(terminals),
            compaction_pending=(
                self._manual_compaction_future is not None
                or self._manual_compaction_requested
            ),
        )

    async def _find_active_turn_id(self, thread_id: str) -> str | None:
        result = await self.app_server.request(
            "thread/turns/list",
            {
                "threadId": thread_id,
                "limit": 20,
                "sortDirection": "desc",
                "itemsView": "notLoaded",
            },
            timeout_seconds=30,
        )
        turns = result.get("data")
        if not isinstance(turns, list):
            return None
        for turn in turns:
            if (
                isinstance(turn, dict)
                and turn.get("status") == "inProgress"
                and isinstance(turn.get("id"), str)
                and turn["id"]
            ):
                return turn["id"]
        return None

    async def _reconcile_recovery_turn(self, thread_id: str, turn_id: str) -> None:
        deadline = monotonic() + 5
        while True:
            pending = self.state.pending_turn
            if pending is not None and pending.phase == "delivery_pending":
                await self._deliver_pending_turn(pending)
            turn_status: str | None = None
            try:
                turn = await self._read_persisted_turn(thread_id, turn_id)
                candidate = turn.get("status") if isinstance(turn, dict) else None
                turn_status = candidate if isinstance(candidate, str) else None
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not confirm Codex turn outcome during chat recovery",
                    exc,
                    event="codex.recovery.turn_status_failed",
                    level=logging.WARNING,
                    **self._thread_log_fields(thread_id),
                    **self._turn_log_fields(turn_id),
                )
            if turn_status in {"completed", "interrupted", "failed"}:
                await self._wait_for_recovery_completion_callback()
                pending = self.state.pending_turn
                if pending is not None and pending.phase != "delivery_pending":
                    self._persist_turn_result(
                        {
                            "status": turn_status,
                            "error": self._turn_error_message(turn or {}),
                        }
                    )
                self.active_turn_id = None
                self._clear_active_command_executions()
                if self._turn_completed_event is not None:
                    self._turn_completed_event.set()
                pending = self.state.pending_turn
                if pending is not None and pending.phase == "delivery_pending":
                    await self._deliver_pending_turn(pending)
                logger.info(
                    "Reconciled Codex turn before restarting its app-server",
                    extra={
                        "event": "codex.recovery.turn_reconciled",
                        "outcome": turn_status,
                        **self._thread_log_fields(thread_id),
                        **self._turn_log_fields(turn_id),
                    },
                )
                return
            try:
                response = await self.app_server.request(
                    "thread/read",
                    {"threadId": thread_id},
                    timeout_seconds=15,
                )
                thread = response.get("thread")
                status = thread.get("status") if isinstance(thread, dict) else None
                status_type = status.get("type") if isinstance(status, dict) else None
            except Exception:
                status_type = None
            if status_type == "idle":
                pending = self.state.pending_turn
                if pending is not None and pending.phase != "delivery_pending":
                    self._persist_turn_result(
                        {"status": "unknown", "error_key": "recovery.label.unknown"}
                    )
                self.active_turn_id = None
                self._clear_active_command_executions()
                if self._turn_completed_event is not None:
                    self._turn_completed_event.set()
                pending = self.state.pending_turn
                if pending is not None and pending.phase == "delivery_pending":
                    await self._deliver_pending_turn(pending)
                return
            if monotonic() >= deadline:
                pending = self.state.pending_turn
                if pending is not None and pending.phase != "delivery_pending":
                    self._persist_turn_result(
                        {"status": "unknown", "error_key": "recovery.label.unknown"}
                    )
                return
            await asyncio.sleep(0.2)

    async def _wait_for_recovery_completion_callback(self) -> None:
        task = self._turn_completion_task
        if task is None:
            return
        if not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except asyncio.TimeoutError as exc:
                logger.warning(
                    "Codex completion delivery is still pending during chat recovery",
                    extra={
                        "event": "codex.recovery.completion_delivery.timeout",
                        "timeout_seconds": 5,
                        **self._thread_log_fields(),
                    },
                )
                raise RuntimeError("Codex completion delivery is still pending") from exc
        if self._turn_completion_task is task:
            self._turn_completion_task = None

    async def _clear_recovery_transient_state(self) -> None:
        task = self._manual_compaction_task
        self._manual_compaction_requested = False
        self._manual_compaction_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        barrier = self._manual_compaction_barrier_future
        self._manual_compaction_barrier_future = None
        if barrier is not None and not barrier.done():
            barrier.set_result(None)
        compaction = self._manual_compaction_future
        self._manual_compaction_future = None
        self._manual_compaction_turn_id = None
        self._manual_compaction_item_completed = False
        self._manual_compaction_started_at = None
        if compaction is not None and not compaction.done():
            compaction.set_result(None)
        completion_task = self._turn_completion_task
        self._turn_completion_task = None
        if completion_task is not None and completion_task is not asyncio.current_task() and not completion_task.done():
            completion_task.cancel()
            await asyncio.gather(completion_task, return_exceptions=True)
        for task in self._assistant_message_tasks:
            if not task.done():
                task.cancel()
        if self._assistant_message_tasks:
            await asyncio.gather(*self._assistant_message_tasks, return_exceptions=True)
        self._assistant_message_tasks.clear()
        self.active_turn_id = None
        self._clear_active_command_executions()
        self._turn_completed_event = None
        self._recovered_turn_id = None
        self._loaded_thread_id = None
        self._background_terminal_api_supported = None
        self.settings_manager.reset_thread_settings(clear_presets=True)
        self._goal = None
        self._goal_thread_id = None
        self._goal_prompt_message_id = None
        self._clear_turn_messages()

    async def list_background_terminals(self) -> list[CodexBackgroundTerminal]:
        thread_id = self.state.thread_id
        if not thread_id:
            return []
        await self._ensure_thread(create_if_missing=False)
        if self.state.thread_id != thread_id:
            raise RuntimeError("The active Codex chat changed while listing background terminals")

        terminals: list[CodexBackgroundTerminal] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _ in range(100):
            params: dict[str, Any] = {"threadId": thread_id, "limit": 100}
            if cursor is not None:
                params["cursor"] = cursor
            result = await self.app_server.request(
                "thread/backgroundTerminals/list",
                params,
                timeout_seconds=30,
            )
            data = result.get("data")
            if not isinstance(data, list):
                raise RuntimeError("Codex app-server returned an invalid background terminal list")
            for item in data:
                if not isinstance(item, dict):
                    continue
                process_id = item.get("processId")
                if not isinstance(process_id, str) or not process_id:
                    continue
                command = item.get("command")
                cwd = item.get("cwd")
                os_pid = item.get("osPid")
                cpu_percent = item.get("cpuPercent")
                rss_kb = item.get("rssKb")
                terminals.append(
                    CodexBackgroundTerminal(
                        process_id=process_id,
                        command=command if isinstance(command, str) else "(unknown command)",
                        cwd=cwd if isinstance(cwd, str) else None,
                        os_pid=os_pid if type(os_pid) is int and os_pid > 0 else None,
                        cpu_percent=(
                            float(cpu_percent)
                            if isinstance(cpu_percent, (int, float)) and not isinstance(cpu_percent, bool)
                            else None
                        ),
                        rss_kb=rss_kb if type(rss_kb) is int and rss_kb >= 0 else None,
                    )
                )
            next_cursor = result.get("nextCursor")
            if next_cursor is None:
                break
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
                raise RuntimeError("Codex app-server returned an invalid background terminal cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        else:
            raise RuntimeError("Codex app-server returned too many background terminal pages")

        self._background_terminal_api_supported = True
        if (
            not terminals
            and not self.is_busy
            and not self._recovery_in_progress
            and not self._recovery_blocked
            and not self._recovery_hold
        ):
            await self._unsubscribe_thread(check_background_terminals=False)
        return terminals

    async def terminate_background_terminal(self, process_id: str) -> bool:
        thread_id = self.state.thread_id
        if not thread_id or not process_id:
            return False
        await self._ensure_thread(create_if_missing=False)
        if self.state.thread_id != thread_id:
            raise RuntimeError("The active Codex chat changed while stopping a background terminal")
        result = await self.app_server.request(
            "thread/backgroundTerminals/terminate",
            {"threadId": thread_id, "processId": process_id},
            timeout_seconds=30,
        )
        return result.get("terminated") is True

    @property
    def permission_mode(self) -> str | None:
        if self.state.thread_id:
            return self.state_store.load_thread(self.state.thread_id).permission_mode
        return self.state.default_permission_mode

    async def new_thread(
        self,
        cwd: Path,
        *,
        force: bool = False,
        new_chat_defaults: NewChatDefaults | None = None,
    ) -> None:
        if not force:
            await self._wait_for_turn_completion()
        cwd = self._validated_workspace_path(cwd)
        if self.is_busy:
            await self._interrupt_before_thread_switch("new_thread")
            self.active_turn_id = None
        await self.app_server.stop()
        self._loaded_thread_id = None
        self._clear_turn_messages()
        self._clear_active_command_executions()
        self.settings_manager.reset_thread_settings()
        self.state.thread_id = None
        if new_chat_defaults is not None:
            self.state.selected_model = new_chat_defaults.selected_model
            self.state.selected_effort = new_chat_defaults.selected_effort
            self.state.default_permission_mode = new_chat_defaults.default_permission_mode
            self.state.selected_memory_mode = new_chat_defaults.selected_memory_mode
            if new_chat_defaults.output_mode is not None:
                self.state.quiet_progress = new_chat_defaults.output_mode == "quiet_answers"
                self.state.output_mode = (
                    "answers"
                    if new_chat_defaults.output_mode == "quiet_answers"
                    else new_chat_defaults.output_mode
                )
        self._goal = None
        self._goal_thread_id = None
        self._goal_prompt_message_id = None
        self.state.cwd = str(cwd)
        self.state_store.save(self.state)

    async def select_thread(self, thread_id: str, cwd: Path, *, force: bool = False) -> None:
        if not force:
            await self._wait_for_turn_completion()
        cwd = self._validated_workspace_path(cwd)
        if self.is_busy:
            await self._interrupt_before_thread_switch("select_thread")
            self.active_turn_id = None
        await self.app_server.stop()
        self._loaded_thread_id = None
        self._clear_turn_messages()
        self._clear_active_command_executions()
        previous_thread_id = self.state.thread_id
        previous_cwd = self.state.cwd
        previous_goal = self._goal
        previous_goal_thread_id = self._goal_thread_id
        previous_goal_prompt_message_id = self._goal_prompt_message_id
        self._goal = None
        self._goal_thread_id = None
        self._goal_prompt_message_id = None
        previous_thread_settings = self.settings_manager.snapshot_thread_settings()
        self.settings_manager.reset_thread_settings()
        self.state.thread_id = thread_id
        self.state.cwd = str(cwd)
        self.state_store.save(self.state)
        try:
            await self._ensure_thread(create_if_missing=False)
            await self._unsubscribe_thread(source="thread_selected")
        except Exception:
            await self.app_server.stop()
            self._loaded_thread_id = None
            self.state.thread_id = previous_thread_id
            self.state.cwd = previous_cwd
            self.settings_manager.restore_thread_settings(previous_thread_settings)
            self._goal = previous_goal
            self._goal_thread_id = previous_goal_thread_id
            self._goal_prompt_message_id = previous_goal_prompt_message_id
            self.state_store.save(self.state)
            raise

    async def list_threads(self) -> list[CodexThread]:
        await self._ensure_app_server()
        threads: list[CodexThread] = []
        seen_cursors: set[str] = set()
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "limit": 100,
                "sortKey": "updated_at",
                "sourceKinds": ["cli", "vscode", "appServer"],
                "archived": False,
            }
            if cursor is not None:
                params["cursor"] = cursor
            result = await self.app_server.request("thread/list", params, timeout_seconds=90)
            data = result.get("data", [])
            if not isinstance(data, list):
                raise RuntimeError("Codex app-server returned an invalid thread list")
            for raw_thread in data:
                if not isinstance(raw_thread, dict):
                    continue
                thread_id = raw_thread.get("id")
                cwd_value = raw_thread.get("cwd")
                if not isinstance(thread_id, str) or not thread_id or not isinstance(cwd_value, str):
                    continue
                try:
                    cwd = self._validated_workspace_path(Path(cwd_value))
                except ValueError:
                    continue
                name = raw_thread.get("name")
                preview = raw_thread.get("preview")
                source = raw_thread.get("source")
                threads.append(
                    CodexThread(
                        thread_id=thread_id,
                        name=name.strip() if isinstance(name, str) and name.strip() else None,
                        preview=preview.strip() if isinstance(preview, str) and preview.strip() else None,
                        cwd=cwd,
                        updated_at=(
                            raw_thread.get("updatedAt")
                            if isinstance(raw_thread.get("updatedAt"), (int, float))
                            else None
                        ),
                        source=source if isinstance(source, str) else None,
                    )
                )
            next_cursor = result.get("nextCursor")
            if next_cursor is None:
                return threads
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
                raise RuntimeError("Codex app-server returned an invalid thread-list cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor

    @property
    def working_directory(self) -> Path:
        if self.state.cwd:
            try:
                return self._validated_workspace_path(Path(self.state.cwd))
            except ValueError:
                pass
        return self.config.workspace_root

    async def status_text(self, *, include_account: bool = True) -> str:
        self._sync_persisted_thread_selection()
        await self._ensure_app_server()
        thread: dict[str, Any] = {}
        if self.state.thread_id:
            status_thread_id = self.state.thread_id
            try:
                await self._ensure_thread(create_if_missing=False)
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not refresh collaboration mode from the Codex thread",
                    exc,
                    event="codex.status.collaboration_mode_read_failed",
                    level=logging.WARNING,
                    rpc_method="thread/resume",
                    **self._thread_log_fields(),
                )
            try:
                result = await self.app_server.request(
                    "thread/read",
                    {"threadId": status_thread_id},
                    timeout_seconds=30,
                )
                self._capture_thread_settings(result)
                candidate = result.get("thread")
                if isinstance(candidate, dict):
                    thread = candidate
                await self._recover_manual_compaction_if_idle(thread_id=status_thread_id)
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not read Codex thread metadata",
                    exc,
                    event="codex.status.thread_read_failed",
                    level=logging.WARNING,
                    rpc_method="thread/read",
                    timeout_seconds=30,
                    **self._thread_log_fields(),
                )
            if not self.is_busy:
                await self._unsubscribe_thread()

        rate_limits = await self._read_rate_limits()
        account = self.app_server.account or {}
        email = account.get("email")
        plan = account.get("planType")
        if include_account:
            account_name = email if isinstance(email, str) and email else self._text("status.values.chatgpt")
            if isinstance(plan, str) and plan:
                account_name = self._text("status.values.account_plan", account=account_name, plan=plan.title())
        else:
            account_name = self._text("status.values.account_hidden")
        activity_key = "status.values.busy" if self.is_busy else "status.values.ready"
        title = thread.get("name") or thread.get("preview")
        if not isinstance(title, str) or not title.strip():
            title = self._text("common.values.no_active_chat") if not self.state.thread_id else self._text("status.values.untitled")
        title = " ".join(title.split())
        if len(title) > 120:
            title = title[:117].rstrip() + "…"
        model = thread.get("model")
        if not isinstance(model, str) or not model:
            model = self.settings_manager.selected_model or self.config.codex_model or self._text("common.values.default_codex_model")
        effort = thread.get("reasoningEffort")
        models_for_status: list[CodexModel] | None = None
        if not isinstance(effort, str) or not effort:
            effort = self.settings_manager.selected_effort
        if not effort:
            try:
                models_for_status = await self.list_models()
                model_info = next((item for item in models_for_status if item.model == model), None)
                effort = model_info.default_reasoning_effort if model_info else None
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not determine model reasoning effort",
                    exc,
                    event="codex.status.reasoning_effort_read_failed",
                    level=logging.WARNING,
                    model=model,
                    **self._thread_log_fields(),
                )
        model_id = model
        model = f"{model} ({effort or self._text('status.values.effort_unknown')})"
        service_tier_id = thread.get("serviceTier")
        if not isinstance(service_tier_id, str) or not service_tier_id:
            service_tier_id = self.settings_manager.thread_service_tier or self.settings_manager.selected_service_tier
        if not service_tier_id:
            try:
                if models_for_status is None:
                    models_for_status = await self.list_models()
                model_info = next((item for item in models_for_status if item.model == model_id), None)
                service_tier_id = model_info.default_service_tier if model_info else None
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not determine the default Codex service tier",
                    exc,
                    event="codex.status.service_tier_default_read_failed",
                    level=logging.WARNING,
                    model=model_id,
                    **self._thread_log_fields(),
                )
        service_tier_name = self._text("status.values.speed_standard")
        if service_tier_id and service_tier_id != "default":
            try:
                if models_for_status is None:
                    models_for_status = await self.list_models()
                model_info = next((item for item in models_for_status if item.model == model_id), None)
                tier = (
                    next((item for item in model_info.service_tiers if item.id == service_tier_id), None)
                    if model_info
                    else None
                )
                service_tier_name = tier.name if tier else service_tier_id
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not determine Codex service tier",
                    exc,
                    event="codex.status.service_tier_read_failed",
                    level=logging.WARNING,
                    model=model,
                    **self._thread_log_fields(),
                )
                service_tier_name = service_tier_id
        cwd = thread.get("cwd")
        if not isinstance(cwd, str):
            cwd = str(self.working_directory)
        permissions = self._status_permission_mode()
        context = self._status_context_usage()
        short_limit = self._status_rate_limit(rate_limits, 300)
        weekly_limit = self._status_rate_limit(rate_limits, 10080)
        collaboration_mode_key = (
            f"status.values.collaboration_mode_{self.settings_manager.collaboration_mode}"
            if self.state.thread_id
            else "common.values.no_active_chat"
        )
        return self._text(
            "status.label.summary",
            activity=self._text(activity_key),
            title=title,
            model=model,
            service_tier=service_tier_name,
            collaboration_mode=self._text(collaboration_mode_key),
            project=cwd,
            permissions=permissions,
            account=account_name,
            context=context,
            short_limit=short_limit,
            weekly_limit=weekly_limit,
        )

    async def _read_rate_limits(self) -> dict[str, Any]:
        try:
            result = await self.app_server.request("account/rateLimits/read", timeout_seconds=30)
        except Exception as exc:
            log_exception(
                logger,
                "Could not read Codex rate limits",
                exc,
                event="codex.status.rate_limits_read_failed",
                level=logging.WARNING,
                rpc_method="account/rateLimits/read",
                timeout_seconds=30,
                **self._thread_log_fields(),
            )
            return {}
        return result

    def _status_permission_mode(self) -> str:
        mode = self.permission_mode
        if mode:
            return self._text(f"permissions.buttons.mode_names.{mode}")
        if self.state.thread_id:
            thread_state = self.state_store.load_thread(self.state.thread_id)
            if not thread_state.managed_by_telecodex:
                return self._text("status.values.permissions_unknown")
        if self.config.approval_policy == "on-request":
            return self._text("permissions.buttons.mode_names.ask")
        return self._text(
            "status.values.permissions_default",
            policy=self.config.approval_policy,
        )

    def _status_context_usage(self) -> str:
        if not self.state.thread_id:
            return self._text("common.values.no_data")
        usage = self.state_store.load_thread(self.state.thread_id).token_usage
        if not usage:
            return self._text("common.values.no_data")
        used = usage.get("context_tokens")
        window = usage.get("model_context_window")
        if not isinstance(used, int):
            return self._text("common.values.no_data")
        if isinstance(window, int) and window > 0:
            percent = max(0, min(100, round(used * 100 / window)))
            return self._text(
                "status.values.context_usage",
                percent=percent,
                used=self._format_token_count(used),
                window=self._format_token_count(window),
            )
        return self._text("status.values.context_unknown_window", used=self._format_token_count(used))

    def _status_rate_limit(self, result: dict[str, Any], duration_minutes: int) -> str:
        bucket: dict[str, Any] | None = None
        limits = result.get("rateLimitsByLimitId")
        if isinstance(limits, dict):
            candidates = [value for value in limits.values() if isinstance(value, dict)]
        else:
            candidates = []
        legacy = result.get("rateLimits")
        if isinstance(legacy, dict):
            candidates.append(legacy)
        for candidate in candidates:
            for window_name in ("primary", "secondary"):
                value = candidate.get(window_name)
                if isinstance(value, dict) and value.get("windowDurationMins") == duration_minutes:
                    bucket = value
                    break
            if bucket:
                break
        if not bucket:
            return self._text("common.values.no_data")
        used = bucket.get("usedPercent")
        if not isinstance(used, (int, float)):
            return self._text("common.values.no_data")
        remaining = max(0, min(100, round(100 - used)))
        resets_at = bucket.get("resetsAt")
        if isinstance(resets_at, (int, float)):
            reset = datetime.fromtimestamp(resets_at, ZoneInfo("Europe/Moscow")).strftime("%d.%m %H:%M")
            return self._text("status.values.rate_limit", remaining=remaining, reset=reset)
        return self._text("status.values.rate_limit_no_reset", remaining=remaining)

    def _format_token_count(self, value: int) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.0f}K"
        return str(value)

    async def _ensure_thread(self, *, create_if_missing: bool = True) -> None:
        self._sync_persisted_thread_selection()
        async with self._ready_lock:
            if not self.app_server.is_running:
                await self.app_server.start()
                self._loaded_thread_id = None

            if self.state.thread_id:
                if self._loaded_thread_id == self.state.thread_id:
                    return
                thread_id = self.state.thread_id
                try:
                    result = await self._resume_thread(thread_id)
                    self._capture_thread_settings(result)
                    thread = result.get("thread", {})
                    cwd = thread.get("cwd") if isinstance(thread, dict) else None
                    if not isinstance(cwd, str):
                        detail = await self.app_server.request(
                            "thread/read", {"threadId": thread_id}
                        )
                        self._capture_thread_settings(detail)
                        thread = detail.get("thread", {})
                        cwd = thread.get("cwd") if isinstance(thread, dict) else None
                    if isinstance(cwd, str):
                        self.state.cwd = str(self._validated_workspace_path(Path(cwd)))
                    elif self.state.cwd:
                        self.state.cwd = str(self._validated_workspace_path(Path(self.state.cwd)))
                    else:
                        self.state.cwd = str(self.working_directory)
                    if self.settings_manager.collaboration_mode not in {"plan", "default"}:
                        self.settings_manager.reset_collaboration_mode()
                    self._loaded_thread_id = thread_id
                    self.state_store.save(self.state)
                    try:
                        await self._read_goal(thread_id)
                    except Exception as exc:
                        logger.debug(
                            "Could not refresh Codex Goal after resuming its thread",
                            extra={
                                "event": "codex.goal.resume_refresh_failed",
                                "error_type": type(exc).__name__,
                                **self._thread_log_fields(thread_id),
                            },
                        )
                    return
                except RpcError as exc:
                    if "no rollout found for thread id" in str(exc).lower():
                        if not create_if_missing:
                            raise RuntimeError("The selected Codex chat no longer exists") from exc
                        logger.warning("Saved Codex thread is missing; starting a new thread")
                        self.state.thread_id = None
                        self._goal = None
                        self._goal_thread_id = None
                        self._goal_prompt_message_id = None
                        self.settings_manager.reset_thread_settings()
                        self.state_store.save(self.state)
                    elif _is_active_thread_writer_error(exc):
                        logger.info(
                            "Codex thread is active in another app-server session",
                            extra={
                                "event": "codex.thread.active_elsewhere",
                                **self._thread_log_fields(),
                            },
                        )
                        raise ActiveThreadWriterError(
                            "The selected Codex chat is active in another Codex session"
                        ) from exc
                    else:
                        logger.error(
                            "Could not resume saved Codex thread",
                            extra={
                                "event": "codex.thread.resume_failed",
                                "rpc_code": exc.code,
                                "error_type": type(exc).__name__,
                                **self._thread_log_fields(),
                            },
                        )
                        raise RuntimeError(
                            "Could not resume the saved Codex thread; start a new thread or check the local Codex data."
                        ) from exc
                except ValueError as exc:
                    if not create_if_missing:
                        raise RuntimeError(
                            "The selected Codex chat is outside the configured workspace"
                        ) from exc
                    logger.warning(
                        "Saved Codex thread is outside the configured workspace; starting a new thread",
                        extra={
                            "event": "codex.thread.workspace_rejected",
                            **self._thread_log_fields(),
                        },
                    )
                    self.state.thread_id = None
                    self.state.cwd = None
                    self._goal = None
                    self._goal_thread_id = None
                    self._goal_prompt_message_id = None
                    self.settings_manager.reset_thread_settings()
                    self._loaded_thread_id = None
                    self.state_store.save(self.state)

            if not create_if_missing:
                raise RuntimeError("The selected Codex chat could not be resumed")

            params: dict[str, Any] = {
                "cwd": str(self.working_directory),
                "serviceName": "telecodex",
            }
            mode = self.state.default_permission_mode
            if mode:
                params.update(thread_permission_params(mode))
            else:
                params.update(
                    {
                        "approvalPolicy": self.config.approval_policy,
                        "permissions": WORKSPACE_PERMISSION_PROFILE,
                    }
                )
            model = self._effective_model()
            if model:
                params["model"] = model
            if self.settings_manager.selected_service_tier:
                params["serviceTier"] = self.settings_manager.selected_service_tier
            result = await self.app_server.request("thread/start", params, timeout_seconds=90)
            self._capture_thread_settings(result)
            thread = result.get("thread", {})
            thread_id = thread.get("id") if isinstance(thread, dict) else None
            if not isinstance(thread_id, str) or not thread_id:
                raise RuntimeError("Codex app-server did not return a thread ID")
            self.state.thread_id = thread_id
            self._goal = None
            self._goal_thread_id = thread_id
            self._goal_prompt_message_id = None
            self.state.cwd = str(self.working_directory)
            memory_mode = None
            if self.settings_manager.selected_memory_mode is not None:
                try:
                    await self.app_server.request(
                        "thread/memoryMode/set",
                        {"threadId": thread_id, "mode": self.settings_manager.selected_memory_mode},
                        timeout_seconds=30,
                    )
                    memory_mode = self.settings_manager.selected_memory_mode
                except Exception as exc:
                    log_exception(
                        logger,
                        "Could not apply the selected memory mode to a new Codex thread",
                        exc,
                        event="codex.thread.memory_mode.apply_failed",
                        level=logging.WARNING,
                        **self._thread_log_fields(thread_id),
                    )
            if self.settings_manager.collaboration_mode not in {"plan", "default"}:
                self.settings_manager.reset_collaboration_mode()
            self._loaded_thread_id = thread_id
            self.state_store.save_thread(
                thread_id,
                ThreadRuntimeState(
                    permission_mode=mode,
                    memory_mode=memory_mode,
                    managed_by_telecodex=True,
                ),
            )
            self.state_store.save(self.state)

    async def _ensure_app_server(self) -> None:
        async with self._ready_lock:
            if not self.app_server.is_running:
                await self.app_server.start()
                self._loaded_thread_id = None
                self._goal = None
                self._goal_thread_id = None

    async def _resume_thread(self, thread_id: str) -> dict[str, Any]:
        closing_deadline = monotonic() + 5
        while True:
            try:
                return await self.app_server.request(
                    "thread/resume",
                    {"threadId": thread_id, "excludeTurns": True},
                    timeout_seconds=90,
                )
            except RpcError as exc:
                if "is closing" not in str(exc).lower() or monotonic() >= closing_deadline:
                    raise
                logger.debug(
                    "Codex thread resume is waiting for previous session to close",
                    extra={
                        "event": "codex.thread.resume.waiting_for_close",
                        "rpc_code": exc.code,
                        **self._thread_log_fields(thread_id),
                    },
                )
                await asyncio.sleep(0.1)

    async def _finish_turn(self, result: dict[str, Any]) -> None:
        try:
            while self._assistant_message_tasks:
                tasks, self._assistant_message_tasks = self._assistant_message_tasks, []
                await asyncio.gather(*tasks, return_exceptions=True)
            await self.on_finished(result)
        finally:
            thread_id = self.state.thread_id
            if thread_id and (
                self._goal_thread_id != thread_id or self._goal_is_active(thread_id)
            ):
                try:
                    await self._read_goal(thread_id)
                except Exception as exc:
                    logger.debug(
                        "Could not refresh Goal before releasing the completed thread",
                        extra={
                            "event": "codex.goal.completion_refresh_failed",
                            "error_type": type(exc).__name__,
                            **self._thread_log_fields(thread_id),
                        },
                    )
            if self._goal_is_active():
                logger.debug(
                    "Keeping Codex thread loaded for Goal continuation",
                    extra={"event": "codex.goal.thread_retained", **self._thread_log_fields()},
                )
            else:
                self._goal_prompt_message_id = None
                await self._unsubscribe_thread(source="turn_finished")

    async def _finish_manual_compaction(self, turn: dict[str, Any]) -> None:
        future = self._manual_compaction_future
        try:
            if turn.get("status") != "completed":
                await self.on_status("status.label.compaction_failed", None, None)
            elif not self._manual_compaction_item_completed:
                await self.on_status("status.label.manual_compaction_completed", None, None)
            await self._unsubscribe_thread(source="manual_compaction_finished")
        finally:
            self._manual_compaction_future = None
            self._manual_compaction_turn_id = None
            self._manual_compaction_item_completed = False
            self._manual_compaction_started_at = None
            if future is not None and not future.done():
                future.set_result(None)

    async def _recover_manual_compaction_if_idle(
        self,
        future: asyncio.Future[None] | None = None,
        thread_id: str | None = None,
    ) -> bool:
        expected_future = self._manual_compaction_future if future is None else future
        if expected_future is None:
            return True

        async with self._manual_compaction_recovery_lock:
            current_future = self._manual_compaction_future
            if current_future is None:
                return True
            if current_future is not expected_future:
                return False
            started_at = self._manual_compaction_started_at
            if (
                started_at is None
                or monotonic() - started_at < TURN_COMPLETION_WAIT_SECONDS
                or not self.app_server.is_running
            ):
                return False

            completion_task = self._turn_completion_task
            if completion_task is not None and not completion_task.done():
                return False

            expected_thread_id = self.state.thread_id if thread_id is None else thread_id
            if not expected_thread_id or self.state.thread_id != expected_thread_id:
                return False
            expected_turn_id = self._manual_compaction_turn_id
            try:
                response = await self.app_server.request(
                    "thread/read",
                    {"threadId": expected_thread_id},
                    timeout_seconds=30,
                )
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not reconcile pending Codex compaction",
                    exc,
                    event="codex.compaction.recovery.thread_read_failed",
                    level=logging.WARNING,
                    **self._thread_log_fields(expected_thread_id),
                )
                return False
            candidate = response.get("thread")
            thread = candidate if isinstance(candidate, dict) else None
            status = thread.get("status") if isinstance(thread, dict) else None
            status_type = status.get("type") if isinstance(status, dict) else None
            if status_type != "idle":
                return False

            if (
                self._manual_compaction_future is not expected_future
                or self.state.thread_id != expected_thread_id
                or self._manual_compaction_turn_id != expected_turn_id
                or (
                    self.active_turn_id is not None
                    and self.active_turn_id != expected_turn_id
                )
            ):
                return False
            completion_task = self._turn_completion_task
            if completion_task is not None and not completion_task.done():
                return False

            compaction_item_completed = self._manual_compaction_item_completed
            self.active_turn_id = None
            self._clear_active_command_executions()
            if self._turn_completed_event is not None:
                self._turn_completed_event.set()
            self._manual_compaction_future = None
            self._manual_compaction_turn_id = None
            self._manual_compaction_item_completed = False
            self._manual_compaction_started_at = None
            if not expected_future.done():
                expected_future.set_result(None)
            logger.warning(
                "Recovered stale Codex compaction state after the server reported an idle thread",
                extra={
                    "event": "codex.compaction.state_recovered",
                    "outcome": "idle_without_completion_event",
                    **self._thread_log_fields(expected_thread_id),
                },
            )
            if not compaction_item_completed:
                self._schedule_status("status.label.compaction_failed")
            return True

    async def _wait_for_turn_completion(self) -> None:
        compaction = self._manual_compaction_future
        if compaction is not None:
            recovered = await self._recover_manual_compaction_if_idle(compaction)
            if not recovered:
                task = self._turn_completion_task
                if task is not None and not task.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(compaction),
                            timeout=TURN_COMPLETION_WAIT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        recovered = await self._recover_manual_compaction_if_idle(compaction)
                    else:
                        recovered = True
                if recovered:
                    compaction = None
            if compaction is not None and self._manual_compaction_future is compaction:
                logger.warning(
                    "Codex compaction is still active or its state could not be confirmed",
                    extra={
                        "event": "codex.turn.completion_wait.pending",
                        "operation": "manual_compaction",
                        "timeout_seconds": TURN_COMPLETION_WAIT_SECONDS,
                        **self._thread_log_fields(),
                    },
                )
                raise CodexChatBusyError("The Codex chat is busy while compacting its context")
        task = self._turn_completion_task
        if task is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=TURN_COMPLETION_WAIT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                logger.warning(
                    "Timed out waiting for Codex completion delivery",
                    extra={
                        "event": "codex.turn.completion_wait.timeout",
                        "operation": "completion_callback",
                        "timeout_seconds": TURN_COMPLETION_WAIT_SECONDS,
                        **self._thread_log_fields(),
                    },
                )
                raise RuntimeError("Codex completion delivery is still pending") from exc
            if self._turn_completion_task is task:
                self._turn_completion_task = None

    async def _release_goal_thread_if_idle(self, thread_id: str) -> None:
        compaction = self._manual_compaction_future
        if self._manual_compaction_requested or (
            compaction is not None
            and not await self._recover_manual_compaction_if_idle(compaction)
        ):
            logger.debug(
                "Keeping Codex thread loaded while manual compaction is active",
                extra={
                    "event": "codex.goal.thread_release.skipped",
                    "outcome": "manual_compaction_active",
                    **self._thread_log_fields(thread_id),
                },
            )
            return
        await self._wait_for_turn_completion()
        with bind_log_context(thread_unsubscribe_source="goal_idle_callback"):
            async with self._thread_unsubscribe_lock:
                async with self._thread_use_lock:
                    if (
                        self.state.thread_id == thread_id
                        and not self.is_busy
                        and not self._goal_is_active(thread_id)
                    ):
                        await self._unsubscribe_thread_serialized(
                            check_background_terminals=True
                        )

    async def _unsubscribe_thread(
        self,
        *,
        check_background_terminals: bool = True,
        source: str = "unspecified",
    ) -> None:
        with bind_log_context(thread_unsubscribe_source=source):
            async with self._thread_unsubscribe_lock:
                await self._unsubscribe_thread_locked(
                    check_background_terminals=check_background_terminals
                )

    async def _unsubscribe_thread_locked(self, *, check_background_terminals: bool = True) -> None:
        async with self._thread_use_lock:
            await self._unsubscribe_thread_serialized(
                check_background_terminals=check_background_terminals
            )

    async def _unsubscribe_thread_serialized(self, *, check_background_terminals: bool) -> None:
        thread_id = self.state.thread_id
        if self._recovery_in_progress or self._recovery_blocked or self._recovery_hold:
            logger.debug(
                "Skipped thread unsubscribe while Codex recovery owns the context",
                extra={
                    "event": "codex.thread.unsubscribe.skipped",
                    "outcome": "recovery_active",
                    **self._thread_log_fields(thread_id),
                },
            )
            return
        if (
            not thread_id
            or self._loaded_thread_id != thread_id
            or not self.app_server.is_running
        ):
            return
        source = current_log_context().get("thread_unsubscribe_source", "unspecified")
        manual_compaction_finished = source == "manual_compaction_finished"
        if self.active_turn_id is not None or (
            (
                self._manual_compaction_future is not None
                or self._manual_compaction_requested
            )
            and not manual_compaction_finished
        ):
            logger.info(
                "Skipped thread unsubscribe while a Codex turn is active",
                extra={
                    "event": "codex.thread.unsubscribe.skipped",
                    "outcome": "turn_active",
                    "thread_unsubscribe_source": source,
                    **self._thread_log_fields(thread_id),
                    **self._turn_log_fields(self.active_turn_id),
                },
            )
            return
        if self._goal_is_active(thread_id):
            logger.debug(
                "Keeping Codex thread loaded for Goal continuation",
                extra={"event": "codex.goal.thread_retained", **self._thread_log_fields(thread_id)},
            )
            return
        if check_background_terminals and self._background_terminal_api_supported is not False:
            try:
                terminals = await self.app_server.request(
                    "thread/backgroundTerminals/list",
                    {"threadId": thread_id, "limit": 1},
                    timeout_seconds=10,
                )
            except RpcError as exc:
                if "unknown method" in str(exc).lower() or "method not found" in str(exc).lower():
                    self._background_terminal_api_supported = False
                else:
                    log_exception(
                        logger,
                        "Could not check Codex background terminals before releasing a thread",
                        exc,
                        event="codex.thread.background_terminal_check_failed",
                        level=logging.WARNING,
                        rpc_method="thread/backgroundTerminals/list",
                        timeout_seconds=10,
                        **self._thread_log_fields(thread_id),
                    )
                    return
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not check Codex background terminals before releasing a thread",
                    exc,
                    event="codex.thread.background_terminal_check_failed",
                    level=logging.WARNING,
                    rpc_method="thread/backgroundTerminals/list",
                    timeout_seconds=10,
                    **self._thread_log_fields(thread_id),
                )
                return
            else:
                self._background_terminal_api_supported = True
                if not isinstance(terminals.get("data"), list):
                    return
                if terminals["data"]:
                    return
        try:
            logger.info(
                "Requesting Codex thread unsubscribe",
                extra={
                    "event": "codex.thread.unsubscribe.requested",
                    "thread_unsubscribe_source": source,
                    **self._thread_log_fields(thread_id),
                },
            )
            result = await self.app_server.request(
                "thread/unsubscribe",
                {"threadId": thread_id},
                timeout_seconds=15,
            )
        except Exception as exc:
            log_exception(
                logger,
                "Could not release idle Codex thread",
                exc,
                event="codex.thread.unsubscribe_failed",
                level=logging.WARNING,
                rpc_method="thread/unsubscribe",
                timeout_seconds=15,
                **self._thread_log_fields(thread_id),
            )
            return
        logger.info(
            "Codex thread unsubscribe completed",
            extra={
                "event": "codex.thread.unsubscribe.completed",
                "thread_unsubscribe_source": source,
                "response_status": result.get("status")
                if isinstance(result.get("status"), str)
                else "unknown",
                **self._thread_log_fields(thread_id),
            },
        )
        if result.get("status") in {"unsubscribed", "notSubscribed", "notLoaded"}:
            self._loaded_thread_id = None

    def _validated_workspace_path(
        self,
        path: Path,
    ) -> Path:
        resolved = path.expanduser().resolve()
        try:
            resolved.relative_to(self.config.workspace_root)
        except ValueError as exc:
            raise ValueError("Working directory is outside the configured workspace root") from exc
        if not resolved.is_dir():
            raise ValueError(f"Working directory is not available: {resolved}")
        return resolved

    async def list_workspace_directories(self, path: Path) -> list[Path]:
        validated = self._validated_workspace_path(path)
        return sorted(
            (item for item in validated.iterdir() if item.is_dir() and not item.is_symlink()),
            key=lambda item: item.name.casefold(),
        )

    def _clear_turn_messages(self) -> None:
        self._messages = {}
        self._message_phases = {}
        self._message_order = []
        self._plan_text = None
        self._plan_deltas = {}
        self._review_text = None
        self._reported_activity_items.clear()
        self._agent_activity_states.clear()
        self._agent_activity_labels.clear()
        self._assistant_message_ids.clear()

    def _sync_persisted_thread_selection(self) -> None:
        saved = self.state_store.load()
        if saved.thread_id == self.state.thread_id:
            return
        self.state.thread_id = saved.thread_id
        self._goal = None
        self._goal_thread_id = None
        self._goal_prompt_message_id = None
        if saved.cwd is not None:
            self.state.cwd = saved.cwd
        self.settings_manager.reset_thread_settings()
        self._loaded_thread_id = None
        self._clear_turn_messages()

    async def _on_server_request(
        self,
        method: str,
        _request_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            decision = await self.on_prompt(method, params)
            if isinstance(decision, dict):
                return decision
            return {"decision": decision}
        if method == "item/tool/requestUserInput":
            return await self.on_prompt(method, params)
        if method == "mcpServer/elicitation/request":
            return await self.on_prompt(method, params)
        if method == "item/permissions/requestApproval":
            return await self.on_prompt(method, params)
        raise RuntimeError(f"Unsupported Codex app-server request: {method}")

    def _on_event(self, message: dict[str, Any]) -> None:
        self._last_event_at = monotonic()
        method = message.get("method")
        self._last_event_method = method if isinstance(method, str) else None
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return

        if method == "thread/goal/updated":
            thread_id = params.get("threadId")
            if isinstance(thread_id, str):
                goal = self._remember_goal(thread_id, params.get("goal"))
                if goal is None or goal.status != "active":
                    self._schedule_callback(
                        self._release_goal_thread_if_idle(thread_id),
                        "Codex Goal thread release callback",
                    )
            return

        if method == "thread/goal/cleared":
            thread_id = params.get("threadId")
            if isinstance(thread_id, str):
                self._remember_goal(thread_id, None)
                if not self.is_busy:
                    self._goal_prompt_message_id = None
                self._schedule_callback(
                    self._release_goal_thread_if_idle(thread_id),
                    "Cleared Codex Goal thread release callback",
                )
            return

        if method == "mcpServer/oauthLogin/completed":
            self._handle_mcp_oauth_completed(params)
            return

        if method == "mcpServer/startupStatus/updated":
            name = params.get("name")
            status = params.get("status")
            if isinstance(name, str) and isinstance(status, str):
                if self._mcp_reported_startup.get(name) != status:
                    self._mcp_reported_startup[name] = status
                    self._schedule_status(
                        "detailed_log.values.mcp_server_status",
                        {"name": name, "status": status},
                    )
            return

        if method in {"turn/started", "turn/completed"}:
            event_thread_id = params.get("threadId")
            pending_turn = self.state.pending_turn
            target_thread_id = (
                pending_turn.thread_id if pending_turn else self.state.thread_id
            )
            turn_event = params.get("turn")
            event_turn_id = turn_event.get("id") if isinstance(turn_event, dict) else None
            if (
                isinstance(event_thread_id, str)
                and target_thread_id
                and event_thread_id != target_thread_id
            ):
                logger.warning(
                    "Ignored Codex turn event for a different thread",
                    extra={
                        "event": "codex.turn.event.ignored",
                        "rpc_method": method,
                        "outcome": "thread_mismatch",
                        "received_thread_hash": log_identifier(
                            event_thread_id,
                            namespace="codex-thread",
                        ),
                        "expected_thread_hash": log_identifier(
                            target_thread_id,
                            namespace="codex-thread",
                        ),
                        **self._turn_log_fields(
                            event_turn_id if isinstance(event_turn_id, str) else None
                        ),
                    },
                )
                return
            if method == "turn/started":
                logger.info(
                    "Codex turn event started",
                    extra={
                        "event": "codex.turn.started_event",
                        "turn_hash": log_identifier(event_turn_id, namespace="codex-turn")
                        if isinstance(event_turn_id, str)
                        else None,
                        "model": turn_event.get("model")
                        if isinstance(turn_event, dict) and isinstance(turn_event.get("model"), str)
                        else None,
                        **self._thread_log_fields(
                            event_thread_id if isinstance(event_thread_id, str) else None
                        ),
                    },
                )

        if method == "thread/tokenUsage/updated":
            thread_id = params.get("threadId")
            usage = params.get("tokenUsage")
            if isinstance(thread_id, str) and isinstance(usage, dict):
                last = usage.get("last")
                context_tokens = last.get("totalTokens") if isinstance(last, dict) else None
                context_window = usage.get("modelContextWindow")
                if type(context_tokens) is int and context_tokens >= 0:
                    thread_state = self.state_store.load_thread(thread_id)
                    thread_state.token_usage = {
                        "context_tokens": context_tokens,
                        "model_context_window": (
                            context_window
                            if type(context_window) is int and context_window > 0
                            else None
                        ),
                    }
                    self.state_store.save_thread(thread_id, thread_state)
            return

        if method == "thread/deleted":
            thread_id = params.get("threadId")
            if isinstance(thread_id, str) and thread_id:
                self.state_store.delete_threads({thread_id})
                if self.state.thread_id == thread_id:
                    self.state.thread_id = None
                    self._loaded_thread_id = None
                    self._goal = None
                    self._goal_thread_id = None
                    self._goal_prompt_message_id = None
                    self._clear_turn_messages()
                    self.state_store.save(self.state)
            return

        if method == "thread/settings/updated":
            thread_id = params.get("threadId")
            if thread_id == self.state.thread_id:
                self._capture_thread_settings(params)
            return

        if method == "turn/started":
            turn = params.get("turn", {})
            if isinstance(turn, dict):
                turn_id = turn.get("rootTurnId")
                if not isinstance(turn_id, str) or not turn_id:
                    turn_id = turn.get("id")
                if isinstance(turn_id, str) and turn_id:
                    event_thread_id = params.get("threadId")
                    pending = self.state.pending_turn
                    goal_thread_id = event_thread_id or self.state.thread_id
                    if isinstance(goal_thread_id, str) and goal_thread_id == self.state.thread_id:
                        self._goal_turn_started_event.set()
                    goal_state_unknown = (
                        isinstance(goal_thread_id, str)
                        and self._goal_thread_id != goal_thread_id
                    )
                    unprompted_turn = (
                        self._manual_compaction_future is None
                        and (pending is None or pending.phase == "delivery_pending")
                        and (
                            goal_state_unknown
                            or self._goal_is_active(
                                goal_thread_id if isinstance(goal_thread_id, str) else None
                            )
                        )
                    )
                    if unprompted_turn:
                        if isinstance(goal_thread_id, str) and goal_thread_id:
                            self._prepare_goal_turn(goal_thread_id)
                            self._schedule_status(
                                "status.goal_continuing",
                                self._goal_prompt_message_id,
                            )
                    self._set_active_turn_id(turn_id)
                if self._manual_compaction_future is not None:
                    self._manual_compaction_turn_id = turn_id if isinstance(turn_id, str) else None
            return

        if method == "item/started":
            item = params.get("item", {})
            if not isinstance(item, dict):
                return
            item_type = item.get("type")
            if item_type == "contextCompaction":
                self._schedule_status("status.label.compaction_started")
            elif item_type == "commandExecution":
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id:
                    self.active_command_executions.add(item_id)
                else:
                    self._active_command_without_id = True
                self._schedule_status(
                    "detailed_log.label.command_running",
                    item.get("command"),
                    item_id if isinstance(item_id, str) else None,
                )
            elif item_type == "webSearch":
                self._schedule_status("detailed_log.values.web_search_started", item)
            elif item_type == "enteredReviewMode":
                self._schedule_item_activity("detailed_log.values.review_started", item)
            elif item_type == "imageGeneration":
                self._schedule_item_activity("detailed_log.values.image_generation_started", item)
            elif item_type == "subAgentActivity":
                self._handle_subagent_activity(item)
            elif item_type == "collabAgentToolCall":
                self._handle_collab_agent_states(item)
            elif item_type == "mcpToolCall":
                self._schedule_item_activity("detailed_log.values.mcp_started", item)
            elif item_type == "agentMessage":
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id not in self._messages:
                    self._messages[item_id] = ""
                    phase = item.get("phase")
                    self._message_phases[item_id] = phase if isinstance(phase, str) else None
                    self._message_order.append(item_id)
            return

        if method == "item/agentMessage/delta":
            item_id = params.get("itemId")
            delta = params.get("delta")
            if isinstance(item_id, str) and isinstance(delta, str):
                if item_id not in self._messages:
                    self._messages[item_id] = ""
                    self._message_phases[item_id] = None
                    self._message_order.append(item_id)
                self._messages[item_id] += delta
            return

        if method == "item/plan/delta":
            event_thread_id = params.get("threadId")
            event_turn_id = params.get("turnId")
            pending = self.state.pending_turn
            expected_turn_id = self.active_turn_id or (pending.turn_id if pending else None)
            if (
                isinstance(event_thread_id, str)
                and self.state.thread_id
                and event_thread_id != self.state.thread_id
            ):
                return
            if (
                isinstance(event_turn_id, str)
                and isinstance(expected_turn_id, str)
                and event_turn_id != expected_turn_id
            ):
                return
            item_id = params.get("itemId")
            delta = params.get("delta")
            if isinstance(item_id, str) and isinstance(delta, str):
                self._plan_deltas[item_id] = self._plan_deltas.get(item_id, "") + delta
            return

        if method == "item/completed":
            item = params.get("item", {})
            if isinstance(item, dict):
                if item.get("type") == "contextCompaction":
                    if self._manual_compaction_future is not None:
                        self._manual_compaction_item_completed = True
                        text_key = "status.label.manual_compaction_completed"
                    else:
                        text_key = "status.label.compaction_completed"
                    self._schedule_status(text_key)
                elif item.get("type") == "agentMessage":
                    item_id = item.get("id")
                    text = item.get("text")
                    if isinstance(item_id, str):
                        if item_id not in self._messages:
                            self._message_order.append(item_id)
                            self._messages[item_id] = ""
                        if isinstance(text, str):
                            self._messages[item_id] = text
                        phase = item.get("phase")
                        if isinstance(phase, str):
                            self._message_phases[item_id] = phase
                        else:
                            phase = self._message_phases.get(item_id)
                        completed_text = self._messages[item_id].strip()
                        if phase == "commentary" and completed_text:
                            self._schedule_assistant_message(item_id, completed_text)
                elif item.get("type") == "plan":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        self._plan_text = text.strip()
                    else:
                        item_id = item.get("id")
                        delta_text = self._plan_deltas.get(item_id, "") if isinstance(item_id, str) else ""
                        self._plan_text = delta_text.strip() or None
                elif item.get("type") == "commandExecution":
                    item_id = item.get("id")
                    if isinstance(item_id, str):
                        self.active_command_executions.discard(item_id)
                    else:
                        self._active_command_without_id = False
                    self._schedule_callback(
                        self.on_command_completed(item),
                        "Codex command completion callback",
                    )
                elif item.get("type") == "fileChange":
                    self._schedule_status("status.file_change", item)
                elif item.get("type") == "webSearch":
                    self._schedule_status("detailed_log.values.web_search_completed", item)
                elif item.get("type") == "enteredReviewMode":
                    self._schedule_item_activity("detailed_log.values.review_started", item)
                elif item.get("type") == "exitedReviewMode":
                    review = item.get("review")
                    if isinstance(review, str) and review.strip():
                        self._review_text = review.strip()
                    self._schedule_item_activity("detailed_log.values.review_completed", item)
                elif item.get("type") == "imageGeneration":
                    failure = item.get("failure")
                    failure_type = failure.get("type") if isinstance(failure, dict) else None
                    status = item.get("status")
                    failed = failure is not None or (
                        isinstance(status, str)
                        and status.casefold() in {"failed", "error", "errored"}
                    )
                    self._schedule_item_activity(
                        "detailed_log.values.image_generation_failed" if failed else "detailed_log.values.image_generation_completed",
                        {
                            "id": item.get("id"),
                            "savedPath": item.get("savedPath"),
                            "failureType": failure_type,
                        },
                    )
                elif item.get("type") == "subAgentActivity":
                    self._handle_subagent_activity(item)
                elif item.get("type") == "collabAgentToolCall":
                    self._handle_collab_agent_states(item)
                elif item.get("type") == "mcpToolCall":
                    item_status = item.get("status")
                    text_key = (
                        "detailed_log.values.mcp_completed"
                        if item_status == "completed"
                        else "detailed_log.values.mcp_failed"
                    )
                    self._schedule_item_activity(text_key, item)
            return

        if method == "turn/completed":
            turn = params.get("turn", {})
            if not isinstance(turn, dict):
                return
            turn_id = turn.get("id")
            if self._manual_compaction_future is not None:
                if (
                    self._manual_compaction_turn_id is not None
                    and turn_id is not None
                    and turn_id != self._manual_compaction_turn_id
                ):
                    return
                if self._turn_completed_event is not None:
                    self._turn_completed_event.set()
                self.active_turn_id = None
                self._clear_active_command_executions()
                self._turn_completion_task = self._schedule_callback(
                    self._finish_manual_compaction(turn),
                    "Codex manual compaction callback",
                )
                return
            pending = self.state.pending_turn
            if (
                pending
                and pending.phase == "delivery_pending"
                and pending.turn_id
                and turn_id == pending.turn_id
            ):
                logger.info(
                    "Ignored duplicate Codex turn completion event",
                    extra={
                        "event": "codex.turn.event.ignored",
                        "rpc_method": method,
                        "outcome": "already_delivery_pending",
                        **self._thread_log_fields(),
                        **self._turn_log_fields(turn_id if isinstance(turn_id, str) else None),
                    },
                )
                return
            if self.active_turn_id and turn_id and turn_id != self.active_turn_id:
                logger.warning(
                    "Ignored Codex turn completion for a different active turn",
                    extra={
                        "event": "codex.turn.event.ignored",
                        "rpc_method": method,
                        "outcome": "active_turn_mismatch",
                        "received_turn_hash": log_identifier(turn_id, namespace="codex-turn")
                        if isinstance(turn_id, str)
                        else None,
                        "expected_turn_hash": log_identifier(
                            self.active_turn_id,
                            namespace="codex-turn",
                        ),
                        **self._thread_log_fields(),
                    },
                )
                return
            if pending and pending.turn_id and turn_id and turn_id != pending.turn_id:
                logger.warning(
                    "Ignored Codex turn completion for a different pending turn",
                    extra={
                        "event": "codex.turn.event.ignored",
                        "rpc_method": method,
                        "outcome": "pending_turn_mismatch",
                        "received_turn_hash": log_identifier(turn_id, namespace="codex-turn")
                        if isinstance(turn_id, str)
                        else None,
                        "expected_turn_hash": log_identifier(
                            pending.turn_id,
                            namespace="codex-turn",
                        ),
                        **self._thread_log_fields(),
                    },
                )
                return
            completed_status = turn.get("status", "failed")
            pending_age = self.pending_turn_age_seconds()
            logger.info(
                "Codex turn completed",
                extra={
                    "event": "codex.turn.completed",
                    "turn_hash": log_identifier(turn_id, namespace="codex-turn")
                    if isinstance(turn_id, str)
                    else None,
                    "turn_status": completed_status,
                    "duration_ms": round(pending_age * 1000) if pending_age is not None else None,
                    "error_code": (
                        turn.get("error", {}).get("codexErrorInfo")
                        if isinstance(turn.get("error"), dict)
                        and isinstance(turn.get("error", {}).get("codexErrorInfo"), str)
                        else None
                    ),
                    **self._thread_log_fields(),
                },
            )
            status = turn.get("status", "failed")
            error = turn.get("error")
            error_text = error.get("message") if isinstance(error, dict) else None
            error_key = None
            if isinstance(error, dict) and error.get("codexErrorInfo") == "unauthorized":
                error_text = None
                error_key = "errors.label.auth_expired"
            result = {
                "status": status,
                "text": self._final_message(),
                "error": error_text,
                "error_key": error_key,
            }
            if pending and not pending.turn_id and isinstance(turn_id, str):
                pending.turn_id = turn_id
            self._persist_turn_result(result)
            if self._turn_completed_event is not None:
                self._turn_completed_event.set()
            self.active_turn_id = None
            self._clear_active_command_executions()
            self._messages = {}
            self._message_phases = {}
            self._message_order = []
            if self._switch_interrupt_pending:
                return
            if turn_id == self._recovered_turn_id:
                self._turn_completion_task = self._schedule_callback(
                    self._finish_recovered_turn(result, turn_id),
                    "Recovered Codex completion callback",
                )
            else:
                self._turn_completion_task = self._schedule_callback(
                    self._finish_turn(result),
                    "Codex completion callback",
                )
            return

        if method == "transport/closed" and (
            self._recovery_in_progress or self._recovery_blocked or self._recovery_hold
        ):
            return

        if method == "transport/closed" and self._manual_compaction_future is not None:
            if self._turn_completed_event is not None:
                self._turn_completed_event.set()
            self.active_turn_id = None
            self._clear_active_command_executions()
            self._turn_completion_task = self._schedule_callback(
                self._finish_manual_compaction({"status": "failed"}),
                "Codex manual compaction callback",
            )
            return

        if method == "transport/closed" and self.is_busy:
            result = {
                "status": "interrupted",
                "text": self._final_message(),
                "error_key": "errors.label.transport_closed",
            }
            self._persist_turn_result(result)
            if self._turn_completed_event is not None:
                self._turn_completed_event.set()
            self.active_turn_id = None
            self._clear_active_command_executions()
            self._schedule_callback(
                self._finish_turn(result),
                "Codex transport close callback",
            )

    async def _finish_recovered_turn(self, result: dict[str, Any], turn_id: object) -> None:
        if isinstance(turn_id, str) and self.state.thread_id:
            try:
                recovered_text = await self._read_turn_text(self.state.thread_id, turn_id)
                if recovered_text:
                    result["text"] = recovered_text
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not reload completed Codex turn output",
                    exc,
                    event="codex.recovery.completed_turn_output_failed",
                    level=logging.WARNING,
                    **self._thread_log_fields(),
                    **self._turn_log_fields(turn_id if isinstance(turn_id, str) else None),
                )
        self._recovered_turn_id = None
        await self._finish_turn(result)

    def _final_message(self) -> str:
        if self._plan_text:
            return self._plan_text
        if self._plan_deltas:
            plan_text = next(reversed(self._plan_deltas.values()), "").strip()
            if plan_text:
                return plan_text
        if not self._message_order:
            return self._review_text or ""
        final_ids = [
            item_id
            for item_id in self._message_order
            if self._message_phases.get(item_id) == "final_answer"
        ]
        if final_ids:
            final_text = self._messages.get(final_ids[-1], "").strip()
            return final_text or self._review_text or ""
        last_id = self._message_order[-1]
        if (
            self._message_phases.get(last_id) == "commentary"
            and last_id in self._assistant_message_ids
        ):
            return self._review_text or ""
        return self._messages.get(last_id, "").strip() or self._review_text or ""

    def _schedule_item_activity(self, text_key: str, item: dict[str, Any]) -> None:
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            event_key = (item_id, text_key)
            if event_key in self._reported_activity_items:
                return
            self._reported_activity_items.add(event_key)
        if text_key.startswith("detailed_log.values.image_generation_"):
            details = {
                key: item.get(key)
                for key in ("savedPath", "failureType")
                if isinstance(item.get(key), str)
            }
        elif text_key in {"detailed_log.values.mcp_started", "detailed_log.values.mcp_completed", "detailed_log.values.mcp_failed"}:
            details = {
                key: item.get(key)
                for key in ("server", "tool", "durationMs")
                if isinstance(item.get(key), (str, int, float))
                and not isinstance(item.get(key), bool)
            }
        else:
            details = None
        self._schedule_status(
            text_key,
            details,
            item_id if isinstance(item_id, str) else None,
        )

    def _handle_subagent_activity(self, item: dict[str, Any]) -> None:
        agent_id = item.get("agentThreadId")
        kind = item.get("kind")
        if not isinstance(agent_id, str) or not agent_id or not isinstance(kind, str):
            return
        status = {
            "started": "running",
            "interrupted": "interrupted",
            "completed": "completed",
        }.get(kind)
        if status:
            self._record_agent_state(agent_id, status, item.get("agentPath"))

    def _handle_collab_agent_states(self, item: dict[str, Any]) -> None:
        states = item.get("agentsStates")
        if not isinstance(states, dict) or not states:
            return
        receiver_ids = item.get("receiverThreadIds")
        receivers = (
            [value for value in receiver_ids if isinstance(value, str) and value]
            if isinstance(receiver_ids, list)
            else []
        )
        entries = list(states.items())
        for index, (agent_key, raw_state) in enumerate(entries):
            if not isinstance(raw_state, dict):
                continue
            if agent_key in receivers:
                agent_id = agent_key
            elif len(entries) == len(receivers) and index < len(receivers):
                agent_id = receivers[index]
            elif not receivers and isinstance(agent_key, str) and agent_key:
                agent_id = agent_key
            else:
                continue
            raw_status = raw_state.get("status")
            status = (
                {
                    "pendingInit": "running",
                    "running": "running",
                    "interrupted": "interrupted",
                    "completed": "completed",
                    "errored": "failed",
                    "shutdown": "interrupted",
                }.get(raw_status)
                if isinstance(raw_status, str)
                else None
            )
            if status:
                self._record_agent_state(agent_id, status)

    def _record_agent_state(
        self,
        agent_id: str,
        status: str,
        agent_path: object = None,
    ) -> None:
        if isinstance(agent_path, str) and agent_path.strip():
            self._agent_activity_labels[agent_id] = agent_path.strip()
        if self._agent_activity_states.get(agent_id) == status:
            return
        self._agent_activity_states[agent_id] = status
        text_key = {
            "running": "detailed_log.values.subagent_started",
            "interrupted": "detailed_log.values.subagent_interrupted",
            "completed": "detailed_log.values.subagent_completed",
            "failed": "detailed_log.values.subagent_failed",
        }.get(status)
        if text_key:
            label = self._agent_activity_labels.get(agent_id)
            details = {"agentPath": label} if label else None
            self._schedule_status(text_key, details, agent_id)

    def _schedule_status(
        self,
        text_key: str,
        command: object | None = None,
        item_id: str | None = None,
    ) -> None:
        self._schedule_callback(self.on_status(text_key, command, item_id), "Codex status callback")

    def _schedule_assistant_message(self, item_id: str, text: str) -> None:
        if self.on_assistant_message is None or item_id in self._assistant_message_ids:
            return
        self._assistant_message_ids.add(item_id)
        logger.debug(
            "Codex commentary message completed",
            extra={
                "event": "codex.message.commentary.completed",
                "item_hash": log_identifier(item_id, namespace="codex-item"),
                "text_length": len(text),
            },
        )
        self._assistant_message_tasks.append(
            self._schedule_callback(
                self.on_assistant_message(item_id, text),
                "Codex commentary delivery callback",
            )
        )

    @staticmethod
    def _schedule_callback(callback: Awaitable[None], label: str) -> asyncio.Task[None]:
        async def run() -> None:
            try:
                await callback
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_exception(
                    logger,
                    "Codex callback failed",
                    exc,
                    event="codex.callback.failed",
                    operation=label,
                )

        return asyncio.create_task(run())
