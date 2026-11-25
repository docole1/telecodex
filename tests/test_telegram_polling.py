from __future__ import annotations

import asyncio
import unittest
from typing import Any
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiogram.methods import GetUpdates
from aiogram.types import Update

from telecodex.telegram.polling import PollingRequestMiddleware, TelegramUpdateMiddleware
from telecodex.service import TeleCodexService


def _update(update_id: int) -> Update:
    return Update.model_validate(
        {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "date": 1_700_000_000,
                "from": {"id": 101, "is_bot": False, "first_name": "Test"},
                "chat": {"id": 101, "type": "private"},
                "text": f"message-{update_id}",
            },
        }
    )


class FakeStateStore:
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset
        self.saved: list[int] = []
        self.fail_save = False

    def save_last_update_id(self, offset: int) -> None:
        if self.fail_save:
            raise OSError("test state persistence failure")
        self.offset = offset
        self.saved.append(offset)


class FakeController:
    def __init__(self) -> None:
        self.errors: list[tuple[dict[str, Any], str]] = []

    async def send_error_for_update(self, update: dict[str, Any], key: str) -> None:
        self.errors.append((update, key))


class TelegramUpdateOffsetMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_offset_is_saved_after_handler_finishes(self) -> None:
        state = FakeStateStore(offset=100)
        controller = FakeController()
        saved: list[int] = []
        middleware = TelegramUpdateMiddleware(
            state,
            controller,  # type: ignore[arg-type]
            initial_offset=100,
            on_offset_saved=saved.append,
        )
        update = _update(100)
        finished = False

        async def handler(_event: Any, _data: dict[str, Any]) -> str:
            nonlocal finished
            self.assertEqual(state.offset, 100)
            finished = True
            return "handled"

        result = await middleware(handler, update, {})

        self.assertEqual(result, "handled")
        self.assertTrue(finished)
        self.assertEqual(state.offset, 101)
        self.assertEqual(saved, [101])

    async def test_old_update_is_skipped_using_persisted_offset(self) -> None:
        state = FakeStateStore(offset=101)
        middleware = TelegramUpdateMiddleware(
            state,
            FakeController(),  # type: ignore[arg-type]
            initial_offset=101,
        )
        handler = AsyncMock()

        result = await middleware(handler, _update(100), {})

        self.assertIsNone(result)
        handler.assert_not_awaited()
        self.assertEqual(state.saved, [])

    async def test_cancelled_handler_does_not_advance_offset(self) -> None:
        state = FakeStateStore(offset=100)
        middleware = TelegramUpdateMiddleware(
            state,
            FakeController(),  # type: ignore[arg-type]
            initial_offset=100,
        )

        async def cancel(_event: Any, _data: dict[str, Any]) -> None:
            raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await middleware(cancel, _update(100), {})

        self.assertEqual(state.offset, 100)
        self.assertEqual(state.saved, [])

    async def test_handler_error_is_reported_and_offset_is_committed(self) -> None:
        state = FakeStateStore(offset=100)
        controller = FakeController()
        middleware = TelegramUpdateMiddleware(
            state,
            controller,  # type: ignore[arg-type]
            initial_offset=100,
        )

        async def fail(_event: Any, _data: dict[str, Any]) -> None:
            raise RuntimeError("test route failure")

        await middleware(fail, _update(100), {})

        self.assertEqual(state.offset, 101)
        self.assertEqual(len(controller.errors), 1)
        self.assertEqual(controller.errors[0][1], "errors.label.update_failed")

    async def test_offset_persistence_failure_is_reported_without_advancing_memory(self) -> None:
        state = FakeStateStore(offset=100)
        state.fail_save = True
        middleware = TelegramUpdateMiddleware(
            state,
            FakeController(),  # type: ignore[arg-type]
            initial_offset=100,
        )

        await middleware(AsyncMock(return_value=None), _update(100), {})

        self.assertEqual(state.offset, 100)
        self.assertEqual(middleware.last_update_id, 100)


class PollingRequestMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_get_updates_updates_polling_health(self) -> None:
        successes: list[bool] = []
        errors: list[Exception] = []
        middleware = PollingRequestMiddleware(
            lambda: successes.append(True),
            errors.append,
        )
        make_request = AsyncMock(return_value=[])
        method = GetUpdates(timeout=50)

        result = await middleware(make_request, object(), method)

        self.assertEqual(result, [])
        self.assertEqual(successes, [True])
        self.assertEqual(errors, [])

    async def test_get_updates_error_is_counted_and_propagated_to_aiogram(self) -> None:
        error = RuntimeError("network failure")
        errors: list[Exception] = []
        middleware = PollingRequestMiddleware(lambda: None, errors.append)
        make_request = AsyncMock(side_effect=error)

        with self.assertRaises(RuntimeError):
            await middleware(make_request, object(), GetUpdates(timeout=50))

        self.assertEqual(errors, [error])

    async def test_other_bot_methods_are_not_counted_as_poll_requests(self) -> None:
        successes: list[bool] = []
        middleware = PollingRequestMiddleware(lambda: successes.append(True), lambda _exc: None)
        make_request = AsyncMock(return_value=True)

        result = await middleware(make_request, object(), object())

        self.assertTrue(result)
        self.assertEqual(successes, [])


class TeleCodexAiogramPollingTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_delegates_polling_and_serializes_updates_in_dispatcher(self) -> None:
        service = object.__new__(TeleCodexService)
        bot = SimpleNamespace(session=SimpleNamespace(close=AsyncMock()))
        dispatcher = SimpleNamespace(start_polling=AsyncMock())
        service.telegram_updates = SimpleNamespace(dispatcher=dispatcher)
        service.telegram = bot
        service._recovery_supervisor = SimpleNamespace(close=AsyncMock())
        service._watchdog_task = None
        service._update_check_task = None
        service.media_groups = SimpleNamespace(close=AsyncMock())
        service.controller = SimpleNamespace(close=AsyncMock())
        service._running = True

        await service.run()

        dispatcher.start_polling.assert_awaited_once_with(
            bot,
            polling_timeout=50,
            handle_as_tasks=False,
            allowed_updates=["message", "callback_query"],
            handle_signals=True,
            close_bot_session=False,
        )
        service._recovery_supervisor.close.assert_awaited_once()
        service.media_groups.close.assert_awaited_once()
        service.controller.close.assert_awaited_once()
        service.telegram.session.close.assert_awaited_once()
