"""Background reconciliation for persisted Codex turns."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import monotonic
from typing import Callable

from telecodex.logging_config import log_exception, log_identifier
from telecodex.telegram.controller import TelegramController

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 20
STALE_EVENT_SECONDS = 90
STALE_TURN_OFFER_SECONDS = 5 * 60
MAX_RETRY_SECONDS = 120


@dataclass
class _RecoveryState:
    thread_id: str | None = None
    turn_id: str | None = None
    next_attempt_at: float = 0.0
    retry_seconds: float = CHECK_INTERVAL_SECONDS
    notice_sent: bool = False
    stale_offer_sent: bool = False


class RecoverySupervisor:
    """Observe pending turns without replaying user prompts or stopping slow work."""

    def __init__(
        self,
        controller: TelegramController,
        on_heartbeat: Callable[[], None],
    ) -> None:
        self.controller = controller
        self.on_heartbeat = on_heartbeat
        self._states: dict[int, _RecoveryState] = {}
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            await self._check_once()
            self.on_heartbeat()
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

    async def _check_once(self) -> None:
        active_context_ids: set[int] = set()
        now = monotonic()
        for controller in self.controller.active_context_controllers():
            context_id = controller.chat_id
            active_context_ids.add(context_id)
            codex = controller._codex()
            pending = codex.pending_turn
            if pending is None:
                self._states.pop(context_id, None)
                continue
            has_pending_interaction = any(
                bool(getattr(controller, attribute, None))
                for attribute in (
                    "pending_approvals",
                    "pending_user_input",
                    "pending_mcp_elicitations",
                )
            )

            should_check = (
                pending.phase == "delivery_pending"
                or not codex.transport_running
                or not codex.is_busy
                or codex.seconds_since_last_event >= STALE_EVENT_SECONDS
            )
            if not should_check:
                continue
            state = self._states.get(context_id)
            if (
                state is None
                or state.thread_id != pending.thread_id
                or state.turn_id != pending.turn_id
            ):
                state = _RecoveryState(
                    thread_id=pending.thread_id,
                    turn_id=pending.turn_id,
                )
                self._states[context_id] = state
            if now < state.next_attempt_at:
                continue
            recovering_connection = (
                not codex.transport_running
                or codex.seconds_since_last_event >= STALE_EVENT_SECONDS
            )
            if (
                recovering_connection
                and not state.notice_sent
                and not has_pending_interaction
            ):
                try:
                    await controller.send_recovery_notice("recovery.label.checking")
                    state.notice_sent = True
                except Exception as exc:
                    log_exception(
                        logger,
                        "Could not report Codex recovery check",
                        exc,
                        event="codex.recovery.notice_failed",
                        level=logging.WARNING,
                        context_hash=log_identifier(context_id, namespace="telegram-chat"),
                        stage="checking",
                    )
            try:
                outcome = await codex.reconcile_pending_turn()
            except Exception as exc:
                log_exception(
                    logger,
                    "Codex watchdog reconciliation failed",
                    exc,
                    event="codex.recovery.reconciliation_failed",
                    level=logging.WARNING,
                    context_hash=log_identifier(context_id, namespace="telegram-chat"),
                    retry_seconds=state.retry_seconds,
                )
                outcome = "retry"

            logger.info(
                "Codex recovery check completed",
                extra={
                    "event": "codex.recovery.check.completed",
                    "outcome": outcome,
                    "pending_phase": pending.phase,
                    "recovery_age_seconds": codex.pending_turn_age_seconds(),
                    "last_event_age_seconds": round(codex.seconds_since_last_event, 1),
                    "last_event_method": codex.last_event_method,
                    "transport_running": codex.transport_running,
                    "context_hash": log_identifier(context_id, namespace="telegram-chat"),
                },
            )

            if outcome == "retry":
                state.next_attempt_at = now + state.retry_seconds
                age_reader = getattr(codex, "pending_turn_age_seconds", None)
                pending_age = age_reader() if callable(age_reader) else None
                logger.warning(
                    "Codex recovery retry scheduled",
                    extra={
                        "event": "codex.recovery.retry_scheduled",
                        "context_hash": log_identifier(context_id, namespace="telegram-chat"),
                        "retry_seconds": state.retry_seconds,
                        "pending_phase": pending.phase,
                        "recovery_age_seconds": pending_age,
                    },
                )
                state.retry_seconds = min(MAX_RETRY_SECONDS, state.retry_seconds * 2)
            elif outcome == "active":
                has_active_command = bool(
                    getattr(codex, "has_active_command_execution", False)
                )
                can_offer_switch = not has_active_command and not has_pending_interaction
                if (
                    can_offer_switch
                    and codex.seconds_since_last_event >= STALE_TURN_OFFER_SECONDS
                    and not state.stale_offer_sent
                ):
                    try:
                        await controller.send_recovery_switch_offer()
                        state.stale_offer_sent = True
                    except Exception as exc:
                        log_exception(
                            logger,
                            "Could not offer a manual switch for a quiet Codex turn",
                            exc,
                            event="codex.recovery.switch_offer_failed",
                            level=logging.WARNING,
                            context_hash=log_identifier(context_id, namespace="telegram-chat"),
                        )
                if state.notice_sent and not has_pending_interaction:
                    try:
                        await controller.send_recovery_notice("recovery.label.running")
                    except Exception as exc:
                        log_exception(
                            logger,
                            "Could not report restored Codex task",
                            exc,
                            event="codex.recovery.notice_failed",
                            level=logging.WARNING,
                            context_hash=log_identifier(context_id, namespace="telegram-chat"),
                            stage="running",
                        )
                state.notice_sent = False
                state.retry_seconds = CHECK_INTERVAL_SECONDS
                seconds_until_offer = max(
                    1.0,
                    STALE_TURN_OFFER_SECONDS - codex.seconds_since_last_event,
                )
                state.next_attempt_at = now + min(
                    STALE_EVENT_SECONDS,
                    seconds_until_offer
                    if can_offer_switch and not state.stale_offer_sent
                    else STALE_EVENT_SECONDS,
                )
            else:
                self._states.pop(context_id, None)
        for context_id in tuple(self._states):
            if context_id not in active_context_ids:
                self._states.pop(context_id, None)
