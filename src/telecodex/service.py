from __future__ import annotations

import asyncio
import logging
import uuid
from time import monotonic
from typing import Any

from telecodex.codex.session import CodexSession
from telecodex.config import AppConfig
from telecodex.logging_config import bind_log_context, log_exception, log_identifier
from telecodex.i18n import TextCatalog
from telecodex.shell.runner import ShellRunner
from telecodex.recovery_supervisor import RecoverySupervisor
from telecodex.state import RuntimeStateStore
from telecodex.systemd import SystemdNotifier
from telecodex.telegram.api import TelegramAPI, TelegramApiError
from telecodex.telegram.controller import TelegramController
from telecodex.telegram.updates import IncomingMessage, parse_update
from telecodex.telegram.user_controller import TelegramUserController
from telecodex.telegram.media_groups import TelegramMediaGroupBuffer

logger = logging.getLogger(__name__)


class TeleCodexService:
    """Compose components and own the Telegram polling lifecycle."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.texts = TextCatalog()
        self.telegram = TelegramAPI(config.telegram_bot_token, config.telegram_proxy)
        self.state_store = RuntimeStateStore(
            config.database_file,
            config.telegram_allowed_user_ids[0],
            config.migration_state_file,
        )
        self.shell = ShellRunner(config.workspace_root, config.shell_timeout_seconds)
        self.controller = TelegramController(
            config,
            self.telegram,
            self.shell,
            self._create_user_session,
            self.texts,
            self.state_store,
            poll_health=self.poll_health,
        )
        self.media_groups = TelegramMediaGroupBuffer(
            self.state_store,
            self.controller._handle_media_group,
            self._report_media_group_error,
        )
        self.last_update_id = self.state_store.load_last_update_id()
        self._running = False
        self._poll_error_count = 0
        self._poll_heartbeat_at = monotonic()
        self._recovery_heartbeat_at = monotonic()
        self._watchdog_degraded = False
        self._notifier = SystemdNotifier()
        self._recovery_supervisor = RecoverySupervisor(
            self.controller,
            self._note_recovery_heartbeat,
        )
        self._watchdog_task: asyncio.Task[None] | None = None

    def _create_user_session(self, user_id: int, controller: TelegramUserController) -> CodexSession:
        state_store = (
            self.state_store.for_group(user_id)
            if controller.is_group
            else self.state_store.for_user(user_id)
        )
        return CodexSession(
            config=self.config,
            state_store=state_store,
            on_status=controller._on_codex_status,
            on_command_completed=controller._on_codex_command_completed,
            on_assistant_message=controller._on_codex_assistant_message,
            on_finished=controller._on_codex_finished,
            on_prompt=controller._on_codex_prompt,
            texts=self.texts,
        )

    async def run(self) -> None:
        try:
            me = await self.telegram.get_me()
            bot_user_id = me.get("id")
            if isinstance(bot_user_id, int) and bot_user_id > 0:
                self.telegram.set_bot_user_id(bot_user_id)
            logger.info(
                "Telegram bot authenticated",
                extra={
                    "event": "telegram.bot.authenticated",
                    "bot_username": me.get("username", "unknown"),
                },
            )
            await self.controller.start()
            self.media_groups.restore()
            self._running = True
            self._recovery_supervisor.start()
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            offset = self.last_update_id or None
            logger.info("TeleCodex service started", extra={"event": "telecodex.service.started"})
            self._notifier.ready("TeleCodex is polling Telegram")
            while self._running:
                try:
                    updates = await self.telegram.get_updates(offset)
                except TelegramApiError as exc:
                    self._poll_error_count += 1
                    logger.warning(
                        "Telegram polling request failed",
                        extra={
                            "event": "telegram.poll.failed",
                            "error_type": type(exc).__name__,
                            "operation": exc.method,
                            "error_code": exc.error_code,
                            "http_status": exc.http_status,
                            "poll_error_count": self._poll_error_count,
                            "timeout_seconds": 60,
                        },
                    )
                    if self._poll_error_count >= 3:
                        try:
                            await self.telegram.reset_connection()
                            logger.warning(
                                "Reset Telegram HTTP connection after repeated failures",
                                extra={
                                    "event": "telegram.poll.connection_reset",
                                    "poll_error_count": self._poll_error_count,
                                },
                            )
                        except Exception as reset_error:
                            log_exception(
                                logger,
                                "Could not reset Telegram HTTP connection",
                                reset_error,
                                event="telegram.poll.connection_reset_failed",
                                level=logging.WARNING,
                                poll_error_count=self._poll_error_count,
                            )
                        self._poll_error_count = 0
                    await asyncio.sleep(3)
                    continue
                self._poll_error_count = 0
                self._poll_heartbeat_at = monotonic()
                for update in updates:
                    update_id = update.get("update_id")
                    if isinstance(update_id, int):
                        offset = update_id + 1
                    update_started_at = monotonic()
                    correlation_id = (
                        f"telegram-{update_id}"
                        if isinstance(update_id, int)
                        else uuid.uuid4().hex[:12]
                    )
                    context_fields: dict[str, Any] = {
                        "correlation_id": correlation_id,
                        "update_id": update_id if isinstance(update_id, int) else None,
                    }
                    outcome = "failed"
                    try:
                        with bind_log_context(**context_fields):
                            incoming = parse_update(update)
                            if isinstance(incoming, IncomingMessage):
                                context_fields = {
                                    **context_fields,
                                    "context_kind": incoming.chat_type or "unknown",
                                    "context_hash": log_identifier(
                                        incoming.chat_id, namespace="telegram-chat"
                                    ),
                                    "actor_hash": log_identifier(
                                        incoming.user_id, namespace="telegram-user"
                                    ),
                                }
                            elif incoming is not None:
                                context_fields = {
                                    **context_fields,
                                    "context_kind": "callback",
                                    "context_hash": log_identifier(
                                        incoming.chat_id, namespace="telegram-chat"
                                    ),
                                    "actor_hash": log_identifier(
                                        incoming.user_id, namespace="telegram-user"
                                    ),
                                }
                            with bind_log_context(**context_fields):
                                logger.info(
                                    "Telegram update received",
                                    extra={
                                        "event": "telegram.update.received",
                                        "update_kind": type(incoming).__name__ if incoming else "ignored",
                                    },
                                )
                                if (
                                    isinstance(incoming, IncomingMessage)
                                    and incoming.media_group_id is not None
                                    and incoming.attachments
                                    and self.controller.can_user_access_context(
                                        incoming.chat_id, incoming.user_id
                                    )
                                    and (
                                        (
                                            incoming.chat_type == "private"
                                            and incoming.chat_id == incoming.user_id
                                        )
                                        or (
                                            incoming.chat_type in {"group", "supergroup"}
                                            and self.state_store.is_group_attached(incoming.chat_id)
                                        )
                                    )
                                ):
                                    await self.media_groups.flush_before(update)
                                    await self.media_groups.add(update)
                                else:
                                    await self.media_groups.flush_before(update)
                                    await self.controller._handle_update(update)
                            outcome = "ok"
                    except Exception as exc:
                        outcome = "error"
                        with bind_log_context(**context_fields):
                            logger.exception(
                                "Telegram update failed",
                                extra={
                                    "event": "telegram.update.failed",
                                    "error_type": type(exc).__name__,
                                },
                            )
                            try:
                                await self.controller.send_error_for_update(
                                    update,
                                    "errors.label.update_failed",
                                )
                            except Exception:
                                logger.exception(
                                    "Could not report Telegram update failure",
                                    extra={"event": "telegram.update.error_report_failed"},
                                )
                    finally:
                        if isinstance(update_id, int):
                            self.last_update_id = update_id + 1
                            try:
                                self.state_store.save_last_update_id(self.last_update_id)
                            except Exception as exc:
                                outcome = "state_persist_error"
                                with bind_log_context(**context_fields):
                                    log_exception(
                                        logger,
                                        "Could not persist Telegram polling offset",
                                        exc,
                                        event="telecodex.state.poll_offset_save_failed",
                                    )
                        elapsed = monotonic() - update_started_at
                        with bind_log_context(**context_fields):
                            logger.info(
                                "Telegram update handled",
                                extra={
                                    "event": "telegram.update.completed",
                                    "duration_ms": round(elapsed * 1000),
                                    "outcome": outcome,
                                },
                            )
                        if elapsed >= 10:
                            logger.warning(
                                "Telegram update handling exceeded the slow-update threshold",
                                extra={
                                    "event": "telegram.update.slow",
                                    "duration_ms": round(elapsed * 1000),
                                    "outcome": outcome,
                                    **context_fields,
                                },
                            )
        finally:
            self._running = False
            logger.info("TeleCodex service stopping", extra={"event": "telecodex.service.stopping"})
            await self._recovery_supervisor.close()
            watchdog_task = self._watchdog_task
            self._watchdog_task = None
            if watchdog_task is not None and not watchdog_task.done():
                watchdog_task.cancel()
                await asyncio.gather(watchdog_task, return_exceptions=True)
            await self.media_groups.close()
            await self.controller.close()
            await self.telegram.close()

    def _note_recovery_heartbeat(self) -> None:
        self._recovery_heartbeat_at = monotonic()

    def poll_health(self) -> dict[str, int | None]:
        return {
            "error_count": self._poll_error_count,
            "age_seconds": max(0, int(monotonic() - self._poll_heartbeat_at)),
        }

    async def _watchdog_loop(self) -> None:
        interval = self._notifier.interval_seconds
        if interval is None:
            return
        next_deadline = monotonic() + interval
        while self._running:
            await asyncio.sleep(max(0, next_deadline - monotonic()))
            now = monotonic()
            event_loop_lag = max(0, now - next_deadline)
            next_deadline = now + interval
            if event_loop_lag >= 5:
                logger.warning(
                    "TeleCodex event loop was delayed",
                    extra={
                        "event": "telecodex.event_loop.lag",
                        "event_loop_lag_ms": round(event_loop_lag * 1000),
                    },
                )
            poll_age = now - self._poll_heartbeat_at
            recovery_age = now - self._recovery_heartbeat_at
            if poll_age <= 80 and recovery_age <= 45:
                self._notifier.watchdog("Telegram polling and recovery supervisor are healthy")
                self._watchdog_degraded = False
            elif not self._watchdog_degraded:
                logger.warning(
                    "TeleCodex health heartbeat is stale",
                    extra={
                        "event": "telecodex.watchdog.health_degraded",
                        "poll_age_seconds": round(poll_age),
                        "recovery_age_seconds": round(recovery_age),
                        "event_loop_lag_ms": round(event_loop_lag * 1000),
                        "poll_error_count": self._poll_error_count,
                    },
                )
                self._watchdog_degraded = True

    async def _report_media_group_error(
        self,
        updates: list[dict[str, Any]],
        error: Exception,
    ) -> None:
        if not updates:
            return
        log_exception(
            logger,
            "Telegram media group failed",
            error,
            event="telegram.media_group.failed",
            update_count=len(updates),
        )
        await self.controller.send_error_for_update(updates[0], "errors.label.update_failed")
