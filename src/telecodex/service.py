from __future__ import annotations

import asyncio
import logging
from time import monotonic
from typing import Any

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from telecodex.codex.session import CodexSession
from telecodex.config import AppConfig
from telecodex.logging_config import log_exception
from telecodex.i18n import TextCatalog
from telecodex.shell.runner import ShellRunner
from telecodex.recovery_supervisor import RecoverySupervisor
from telecodex.state import RuntimeStateStore
from telecodex.systemd import SystemdNotifier
from telecodex.telegram.api import TelegramFormattingFallbackMiddleware, create_telegram_bot
from telecodex.telegram.controller import TelegramController
from telecodex.telegram.routers import TelegramUpdateRouters
from telecodex.telegram.polling import PollingRequestMiddleware, TelegramUpdateMiddleware
from telecodex.telegram.user_controller import TelegramUserController
from telecodex.telegram.media_groups import TelegramMediaGroupBuffer
from telecodex.update_checker import (
    UPDATE_CHECK_INTERVAL_SECONDS,
    GitHubRelease,
    check_for_update,
)

logger = logging.getLogger(__name__)


class TeleCodexService:
    """Compose components and own the Telegram polling lifecycle."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.texts = TextCatalog()
        self.telegram = create_telegram_bot(config.telegram_bot_token, config.telegram_proxy)
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
        self.telegram_updates = TelegramUpdateRouters(self.controller, self.media_groups)
        self.last_update_id = self.state_store.load_last_update_id()
        self.telegram.session.middleware.register(TelegramFormattingFallbackMiddleware())
        self.telegram.session.middleware.register(
            PollingRequestMiddleware(self._poll_succeeded, self._poll_failed)
        )
        self.telegram_updates.dispatcher.update.outer_middleware(
            TelegramUpdateMiddleware(
                self.state_store,
                self.controller,
                initial_offset=self.last_update_id,
                on_offset_saved=self._offset_saved,
            )
        )
        self.telegram_updates.dispatcher.startup.register(self._on_polling_startup)
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
        self._update_check_task: asyncio.Task[None] | None = None

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
            await self.telegram_updates.dispatcher.start_polling(
                self.telegram,
                polling_timeout=50,
                handle_as_tasks=False,
                allowed_updates=["message", "callback_query"],
                handle_signals=True,
                close_bot_session=False,
            )
        finally:
            self._running = False
            logger.info("TeleCodex service stopping", extra={"event": "telecodex.service.stopping"})
            await self._recovery_supervisor.close()
            update_check_task = self._update_check_task
            self._update_check_task = None
            if update_check_task is not None and not update_check_task.done():
                update_check_task.cancel()
                await asyncio.gather(update_check_task, return_exceptions=True)
            watchdog_task = self._watchdog_task
            self._watchdog_task = None
            if watchdog_task is not None and not watchdog_task.done():
                watchdog_task.cancel()
                await asyncio.gather(watchdog_task, return_exceptions=True)
            await self.media_groups.close()
            await self.controller.close()
            await self.telegram.session.close()

    async def _on_polling_startup(self, bot: Bot, **_data: Any) -> None:
        me = await bot.me()
        logger.info(
            "Telegram bot authenticated",
            extra={
                "event": "telegram.bot.authenticated",
                "bot_username": me.username or "unknown",
            },
        )
        await self.controller.start()
        self.media_groups.restore()
        if self.last_update_id > 0:
            try:
                await self.telegram.get_updates(
                    offset=self.last_update_id,
                    timeout=0,
                    allowed_updates=["message", "callback_query"],
                    request_timeout=10,
                )
            except TelegramAPIError as exc:
                logger.warning(
                    "Could not synchronize the persisted Telegram update offset before polling",
                    extra={
                        "event": "telegram.poll.offset_bootstrap_failed",
                        "error_type": type(exc).__name__,
                        "operation": getattr(exc.method, "__api_method__", None),
                        "error_code": getattr(exc, "error_code", None),
                    },
                )
        self._running = True
        self._poll_heartbeat_at = monotonic()
        self._recovery_supervisor.start()
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        self._update_check_task = asyncio.create_task(self._update_check_loop())
        logger.info("TeleCodex service started", extra={"event": "telecodex.service.started"})
        self._notifier.ready("TeleCodex is polling Telegram with Aiogram")

    def _poll_succeeded(self) -> None:
        self._poll_error_count = 0
        self._poll_heartbeat_at = monotonic()

    def _poll_failed(self, exc: Exception) -> None:
        self._poll_error_count += 1
        logger.warning(
            "Aiogram could not fetch Telegram updates; its polling backoff will retry",
            extra={
                "event": "telegram.poll.failed",
                "error_type": type(exc).__name__,
                "poll_error_count": self._poll_error_count,
                "timeout_seconds": 60,
            },
        )

    def _offset_saved(self, next_offset: int) -> None:
        self.last_update_id = next_offset

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

    async def _update_check_loop(self) -> None:
        while self._running:
            try:
                release = await asyncio.to_thread(check_for_update)
                if release is not None:
                    await self._notify_update_available(release)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not check for a newer TeleCodex release",
                    exc,
                    event="telecodex.update_check.failed",
                    level=logging.WARNING,
                )
            await asyncio.sleep(UPDATE_CHECK_INTERVAL_SECONDS)

    async def _notify_update_available(self, release: GitHubRelease) -> None:
        for user_id in self.config.telegram_allowed_user_ids:
            metadata_key = f"update_release_notified:{user_id}"
            try:
                if self.state_store.load_service_metadata(metadata_key) == release.tag_name:
                    continue
                language = self.state_store.for_user(user_id).load().language
                message = self.texts.render(
                    language,
                    "updates.label.available",
                    version=release.version,
                    release_url=release.html_url,
                )
                await self.telegram.send_message(user_id, message)
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not send a TeleCodex update notice to an operator",
                    exc,
                    event="telecodex.update_notice.delivery_failed",
                    level=logging.WARNING,
                    context_hash=log_identifier(user_id, namespace="telegram-user"),
                    version=release.version,
                )
                continue

            try:
                self.state_store.save_service_metadata(metadata_key, release.tag_name)
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not record a delivered TeleCodex update notice",
                    exc,
                    event="telecodex.update_notice.persistence_failed",
                    level=logging.WARNING,
                    context_hash=log_identifier(user_id, namespace="telegram-user"),
                    version=release.version,
                )
            else:
                logger.info(
                    "TeleCodex update notice sent to an operator",
                    extra={
                        "event": "telecodex.update_notice.sent",
                        "context_hash": log_identifier(user_id, namespace="telegram-user"),
                        "version": release.version,
                    },
                )

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
