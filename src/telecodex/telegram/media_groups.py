from __future__ import annotations

import asyncio
import json
import logging
from time import monotonic
from collections.abc import Awaitable, Callable
from typing import Any

from telecodex.logging_config import isolated_log_context, log_exception, log_identifier
from telecodex.state import RuntimeStateStore

logger = logging.getLogger(__name__)

MEDIA_GROUP_QUIET_SECONDS = 1.0
MediaGroupCallback = Callable[[list[dict[str, Any]]], Awaitable[None]]
MediaGroupErrorCallback = Callable[[list[dict[str, Any]], Exception], Awaitable[None]]


class TelegramMediaGroupBuffer:
    """Collect Telegram album updates and deliver each group as one user request."""

    def __init__(
        self,
        state_store: RuntimeStateStore,
        on_group: MediaGroupCallback,
        on_error: MediaGroupErrorCallback,
        quiet_seconds: float = MEDIA_GROUP_QUIET_SECONDS,
    ) -> None:
        self.state_store = state_store
        self.on_group = on_group
        self.on_error = on_error
        self.quiet_seconds = quiet_seconds
        self._groups: dict[str, list[dict[str, Any]]] = {}
        self._timers: dict[str, asyncio.Task[None]] = {}
        self._processing: dict[str, asyncio.Task[None]] = {}

    def restore(self) -> None:
        for key, updates in self.state_store.load_pending_media_groups().items():
            self._groups[key] = sorted(updates, key=_update_id)
            self._schedule(key)

    async def add(self, update: dict[str, Any]) -> bool:
        key = _media_group_key(update)
        if key is None:
            return False
        updates = self._groups.setdefault(key, [])
        update_id = _update_id(update)
        if not any(_update_id(existing) == update_id for existing in updates):
            updates.append(update)
            updates.sort(key=_update_id)
            # Persist the group before the polling offset is advanced, so a restart
            # cannot acknowledge an album part that exists only in memory.
            self.state_store.save_pending_media_group(key, updates)
        self._schedule(key)
        return True

    async def flush_before(self, update: dict[str, Any]) -> None:
        """Flush earlier albums from this sender before handling a later message."""
        message = update.get("message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat")
        user = message.get("from")
        chat_id = chat.get("id") if isinstance(chat, dict) else None
        user_id = user.get("id") if isinstance(user, dict) else None
        update_id = _update_id(update)
        incoming_group_key = _media_group_key(update)
        if not isinstance(chat_id, int) or not isinstance(user_id, int):
            return
        earlier = [
            key
            for key, updates in self._groups.items()
            if updates
            and key != incoming_group_key
            and _group_sender(updates[0]) == (chat_id, user_id)
            and _update_id(updates[0]) < update_id
        ]
        for key in sorted(earlier, key=lambda value: _update_id(self._groups[value][0])):
            await self._flush(key)

    async def close(self) -> None:
        # Keep groups that have not reached their quiet window persisted for the
        # next startup; do not dispatch a partial album merely because shutdown
        # began. Groups already being delivered are allowed to finish.
        while self._timers:
            timers = list(self._timers.values())
            for task in timers:
                task.cancel()
            await asyncio.gather(*timers, return_exceptions=True)
        if self._processing:
            await asyncio.gather(*self._processing.values(), return_exceptions=True)

    def _schedule(self, key: str) -> None:
        previous = self._timers.get(key)
        if previous is not None:
            previous.cancel()
        updates = self._groups.get(key, [])
        sender = _group_sender(updates[0]) if updates else None
        context_hash = log_identifier(sender[0], namespace="telegram-chat") if sender else None
        actor_hash = log_identifier(sender[1], namespace="telegram-user") if sender else None
        correlation = log_identifier(key, namespace="telegram-media-group")
        with isolated_log_context(
            correlation_id=f"media-{correlation}" if correlation else "media-group",
            context_kind="group" if sender else "unknown",
            context_hash=context_hash,
            actor_hash=actor_hash,
        ):
            task = asyncio.create_task(self._expire(key))
        self._timers[key] = task

    async def _expire(self, key: str) -> None:
        task = asyncio.current_task()
        try:
            await asyncio.sleep(self.quiet_seconds)
        except asyncio.CancelledError:
            return
        finally:
            if self._timers.get(key) is task:
                self._timers.pop(key, None)
        await self._flush(key)

    async def _flush(self, key: str) -> None:
        processing = self._processing.get(key)
        if processing is not None:
            await processing
            return
        updates = self._groups.get(key)
        if not updates:
            return

        snapshot = list(updates)
        sender = _group_sender(snapshot[0]) if snapshot else None
        context_hash = log_identifier(sender[0], namespace="telegram-chat") if sender else None
        actor_hash = log_identifier(sender[1], namespace="telegram-user") if sender else None
        correlation = log_identifier(key, namespace="telegram-media-group")
        context_fields = {
            "correlation_id": f"media-{correlation}" if correlation else "media-group",
            "context_kind": "group" if sender else "unknown",
            "context_hash": context_hash,
            "actor_hash": actor_hash,
        }
        started_at = monotonic()
        with isolated_log_context(**context_fields):
            logger.info(
                "Telegram media group dispatch started",
                extra={
                    "event": "telegram.media_group.dispatch.started",
                    "update_count": len(snapshot),
                },
            )

        async def deliver() -> None:
            try:
                await self.on_group(snapshot)
                logger.info(
                    "Telegram media group dispatched",
                    extra={
                        "event": "telegram.media_group.dispatch.completed",
                        "update_count": len(snapshot),
                        "duration_ms": round((monotonic() - started_at) * 1000),
                        "outcome": "ok",
                    },
                )
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not dispatch Telegram media group",
                    exc,
                    event="telegram.media_group.dispatch_failed",
                    context_hash=context_hash,
                    actor_hash=actor_hash,
                    update_count=len(snapshot),
                    duration_ms=round((monotonic() - started_at) * 1000),
                )
                try:
                    await self.on_error(snapshot, exc)
                except Exception as report_exc:
                    log_exception(
                        logger,
                        "Could not report Telegram media group failure",
                        report_exc,
                        event="telegram.media_group.error_report_failed",
                        context_hash=context_hash,
                        actor_hash=actor_hash,
                        update_count=len(snapshot),
                    )
            finally:
                snapshot_ids = {_update_id(update) for update in snapshot}
                current = self._groups.get(key, [])
                remaining = [update for update in current if _update_id(update) not in snapshot_ids]
                if remaining:
                    self._groups[key] = remaining
                    self.state_store.save_pending_media_group(key, remaining)
                    self._schedule(key)
                else:
                    self._groups.pop(key, None)
                    self.state_store.delete_pending_media_group(key)

        with isolated_log_context(**context_fields):
            task = asyncio.create_task(deliver())
        self._processing[key] = task
        try:
            await task
        finally:
            if self._processing.get(key) is task:
                self._processing.pop(key, None)


def _media_group_key(update: dict[str, Any]) -> str | None:
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    media_group_id = message.get("media_group_id")
    chat, user = message.get("chat"), message.get("from")
    chat_id = chat.get("id") if isinstance(chat, dict) else None
    user_id = user.get("id") if isinstance(user, dict) else None
    if not isinstance(media_group_id, str) or not media_group_id:
        return None
    if not isinstance(chat_id, int) or not isinstance(user_id, int):
        return None
    return json.dumps((chat_id, user_id, media_group_id), separators=(",", ":"))


def _group_sender(update: dict[str, Any]) -> tuple[int, int] | None:
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    chat, user = message.get("chat"), message.get("from")
    chat_id = chat.get("id") if isinstance(chat, dict) else None
    user_id = user.get("id") if isinstance(user, dict) else None
    if isinstance(chat_id, int) and isinstance(user_id, int):
        return chat_id, user_id
    return None


def _update_id(update: dict[str, Any]) -> int:
    value = update.get("update_id")
    return value if isinstance(value, int) else -1
