"""Aiogram middleware for polling health and durable update processing."""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from typing import Any, Callable

from aiogram import BaseMiddleware
from aiogram.client.session.middlewares.base import BaseRequestMiddleware
from aiogram.methods import GetUpdates
from aiogram.types import CallbackQuery, Message, TelegramObject, Update

from telecodex.logging_config import bind_log_context, log_exception, log_identifier
from telecodex.state import RuntimeStateStore
from telecodex.telegram.controller import TelegramController
from telecodex.telegram.updates import (
    aiogram_update_to_dict,
    parse_aiogram_update,
)

logger = logging.getLogger(__name__)


class PollingRequestMiddleware(BaseRequestMiddleware):
    """Track only getUpdates requests; Aiogram owns retry and backoff behavior."""

    def __init__(
        self,
        on_success: Callable[[], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self._on_success = on_success
        self._on_error = on_error

    async def __call__(self, make_request: Any, bot: Any, method: Any) -> Any:
        if not isinstance(method, GetUpdates):
            return await make_request(bot, method)
        try:
            response = await make_request(bot, method)
        except Exception as exc:
            self._on_error(exc)
            raise
        self._on_success()
        return response


class TelegramUpdateMiddleware(BaseMiddleware):
    """Deduplicate, log, and durably record each update around Aiogram routing."""

    def __init__(
        self,
        state_store: RuntimeStateStore,
        controller: TelegramController,
        *,
        initial_offset: int,
        on_offset_saved: Callable[[int], None] | None = None,
    ) -> None:
        self.state_store = state_store
        self.controller = controller
        self.last_update_id = initial_offset
        self._on_offset_saved = on_offset_saved

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Any],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Update):
            return await handler(event, data)

        update_id = event.update_id
        started_at = monotonic()
        correlation_id = f"telegram-{update_id}"
        context_fields: dict[str, Any] = {
            "correlation_id": correlation_id,
            "update_id": update_id,
        }
        if update_id < self.last_update_id:
            logger.info(
                "Telegram update is already covered by the persisted polling offset",
                extra={
                    "event": "telegram.update.duplicate_skipped",
                    **context_fields,
                },
            )
            self._log_completed(started_at, "duplicate", context_fields)
            return None

        outcome = "failed"
        completed = False
        raw_update: dict[str, Any] = {}
        try:
            with bind_log_context(**context_fields):
                raw_update = aiogram_update_to_dict(event)
                incoming = parse_aiogram_update(event)
                if isinstance(incoming, Message):
                    context_fields = {
                        **context_fields,
                        "context_kind": incoming.chat_type or "unknown",
                        "context_hash": log_identifier(
                            incoming.chat_id,
                            namespace="telegram-chat",
                        ),
                        "actor_hash": log_identifier(
                            incoming.user_id,
                            namespace="telegram-user",
                        ),
                    }
                elif isinstance(incoming, CallbackQuery):
                    context_fields = {
                        **context_fields,
                        "context_kind": "callback",
                        "context_hash": log_identifier(
                            incoming.chat_id,
                            namespace="telegram-chat",
                        ),
                        "actor_hash": log_identifier(
                            incoming.user_id,
                            namespace="telegram-user",
                        ),
                    }
                data["telecodex_update"] = event
                data["telecodex_update_data"] = raw_update
                data["telecodex_incoming"] = incoming
                with bind_log_context(**context_fields):
                    logger.info(
                        "Telegram update received",
                        extra={
                            "event": "telegram.update.received",
                            "update_kind": type(incoming).__name__ if incoming else "ignored",
                        },
                    )
                    result = await handler(event, data)
                outcome = "ok"
                completed = True
                return result
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            outcome = "error"
            with bind_log_context(**context_fields):
                log_exception(
                    logger,
                    "Telegram update handling failed",
                    exc,
                    event="telegram.update.failed",
                    error_type=type(exc).__name__,
                )
                try:
                    if raw_update:
                        await self.controller.send_error_for_update(
                            raw_update,
                            "errors.label.update_failed",
                        )
                except Exception as report_exc:
                    log_exception(
                        logger,
                        "Could not report Telegram update failure",
                        report_exc,
                        event="telegram.update.error_report_failed",
                    )
            completed = True
            return None
        finally:
            if completed:
                if not self._save_offset(update_id + 1, context_fields):
                    outcome = "state_persist_error"
            self._log_completed(started_at, outcome, context_fields)

    def _save_offset(self, next_offset: int, context_fields: dict[str, Any]) -> bool:
        next_offset = max(self.last_update_id, next_offset)
        try:
            self.state_store.save_last_update_id(next_offset)
        except Exception as exc:
            log_exception(
                logger,
                "Could not persist Telegram polling offset",
                exc,
                event="telecodex.state.poll_offset_save_failed",
                **context_fields,
            )
            return False
        self.last_update_id = next_offset
        if self._on_offset_saved is not None:
            self._on_offset_saved(next_offset)
        return True

    @staticmethod
    def _log_completed(
        started_at: float,
        outcome: str,
        context_fields: dict[str, Any],
    ) -> None:
        elapsed = monotonic() - started_at
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
