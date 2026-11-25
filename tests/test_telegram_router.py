from __future__ import annotations

import unittest

from aiogram import Bot
from aiogram.types import Update

from telecodex.telegram.routers import TelegramUpdateRouters
from telecodex.telegram.polling import TelegramUpdateMiddleware
from telecodex.telegram.updates import IncomingCallback, IncomingMessage, parse_aiogram_update


BOT_TOKEN = "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"


class FakeMediaGroups:
    def __init__(self, order: list[str] | None = None) -> None:
        self.flushed: list[dict] = []
        self.added: list[dict] = []
        self.order = order

    async def flush_before(self, update: dict) -> None:
        self.flushed.append(update)

    async def add(self, update: dict) -> bool:
        self.added.append(update)
        if self.order is not None:
            self.order.append("album_saved")
        return True


class FakeStateStore:
    def __init__(self, order: list[str] | None = None) -> None:
        self.order = order
        self.offset = 0

    def is_group_attached(self, _chat_id: int) -> bool:
        return True

    def save_last_update_id(self, offset: int) -> None:
        self.offset = offset
        if self.order is not None:
            self.order.append("offset_saved")


class FakeController:
    def __init__(self) -> None:
        self.state_store = FakeStateStore()
        self.incoming: list[IncomingMessage | IncomingCallback] = []

    def can_user_access_context(self, _chat_id: int, _user_id: int) -> bool:
        return True

    async def _handle_incoming(self, incoming: IncomingMessage | IncomingCallback) -> None:
        self.incoming.append(incoming)


def _message_update(
    *,
    text: str,
    entities: list[dict] | None = None,
    chat_id: int = 101,
    chat_type: str = "private",
    media_group_id: str | None = None,
    document: dict | None = None,
) -> dict:
    message = {
        "message_id": 2,
        "date": 1_700_000_000,
        "from": {"id": 101, "is_bot": False, "first_name": "Test"},
        "chat": {"id": chat_id, "type": chat_type},
        "text": text,
    }
    if entities:
        message["entities"] = entities
    if media_group_id is not None:
        message["media_group_id"] = media_group_id
    if document is not None:
        message["document"] = document
    return {"update_id": 1, "message": message}


class TelegramUpdateRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bot = Bot(BOT_TOKEN)
        self.controller = FakeController()
        self.media_groups = FakeMediaGroups()
        self.routes = TelegramUpdateRouters(self.controller, self.media_groups)  # type: ignore[arg-type]

    async def asyncTearDown(self) -> None:
        await self.bot.session.close()

    async def test_typed_message_reaches_controller_with_middleware_markdown(self) -> None:
        update = Update.model_validate(
            _message_update(
                text="Run this task",
                entities=[{"type": "bold", "offset": 4, "length": 4}],
            )
        )
        incoming = parse_aiogram_update(update)

        await self.routes.dispatcher.feed_update(
            self.bot,
            update,
            telecodex_update=update,
            telecodex_incoming=incoming,
        )

        self.assertEqual(len(self.controller.incoming), 1)
        routed = self.controller.incoming[0]
        self.assertIsInstance(routed, IncomingMessage)
        self.assertEqual(routed.text, "Run this task")  # type: ignore[union-attr]
        self.assertEqual(routed.prompt_text, "Run **this** task")  # type: ignore[union-attr]

    async def test_command_router_keeps_plan_argument_entities(self) -> None:
        update = Update.model_validate(
            _message_update(
                text="/plan Write a plan",
                entities=[{"type": "bold", "offset": 6, "length": 5}],
            )
        )
        incoming = parse_aiogram_update(update)

        await self.routes.dispatcher.feed_update(
            self.bot,
            update,
            telecodex_update=update,
            telecodex_incoming=incoming,
        )

        self.assertEqual(len(self.controller.incoming), 1)
        routed = self.controller.incoming[0]
        self.assertIsInstance(routed, IncomingMessage)
        self.assertEqual(routed.command_argument_markdown(), "**Write** a plan")  # type: ignore[union-attr]

    async def test_media_group_is_persisted_through_router_before_offset_returns(self) -> None:
        update = Update.model_validate(
            _message_update(
                text="caption",
                chat_id=-100123,
                chat_type="supergroup",
                media_group_id="album-1",
                document={
                    "file_id": "file-1",
                    "file_unique_id": "file-unique-1",
                    "file_name": "notes.txt",
                },
            )
        )
        incoming = parse_aiogram_update(update)

        await self.routes.dispatcher.feed_update(
            self.bot,
            update,
            telecodex_update=update,
            telecodex_incoming=incoming,
        )

        self.assertEqual(len(self.media_groups.flushed), 1)
        self.assertEqual(len(self.media_groups.added), 1)
        self.assertEqual(self.media_groups.added[0]["update_id"], 1)
        self.assertEqual(self.controller.incoming, [])

    async def test_album_is_persisted_before_update_offset(self) -> None:
        order: list[str] = []
        state_store = FakeStateStore(order)
        controller = FakeController()
        controller.state_store = state_store
        media_groups = FakeMediaGroups(order)
        routes = TelegramUpdateRouters(controller, media_groups)  # type: ignore[arg-type]
        routes.dispatcher.update.outer_middleware(
            TelegramUpdateMiddleware(
                state_store,  # type: ignore[arg-type]
                controller,  # type: ignore[arg-type]
                initial_offset=0,
            )
        )
        update = Update.model_validate(
            _message_update(
                text="caption",
                chat_id=-100123,
                chat_type="supergroup",
                media_group_id="album-2",
                document={
                    "file_id": "file-2",
                    "file_unique_id": "file-unique-2",
                    "file_name": "notes.txt",
                },
            )
        )

        await routes.dispatcher.feed_update(self.bot, update)

        self.assertEqual(order, ["album_saved", "offset_saved"])
        self.assertEqual(state_store.offset, 2)

    async def test_callback_query_is_adapted_to_existing_callback_dto(self) -> None:
        update = Update.model_validate(
            {
                "update_id": 1,
                "callback_query": {
                    "id": "callback-1",
                    "from": {"id": 101, "is_bot": False, "first_name": "Test"},
                    "chat_instance": "instance",
                    "data": "m:token:d",
                    "message": {
                        "message_id": 3,
                        "date": 1_700_000_000,
                        "chat": {"id": 101, "type": "private"},
                        "from": {"id": 999, "is_bot": True, "first_name": "Bot"},
                        "text": "Select model",
                    },
                },
            }
        )
        incoming = parse_aiogram_update(update)

        await self.routes.dispatcher.feed_update(
            self.bot,
            update,
            telecodex_update=update,
            telecodex_incoming=incoming,
        )

        self.assertEqual(len(self.controller.incoming), 1)
        self.assertEqual(self.controller.incoming[0], incoming)
        self.assertIsInstance(self.controller.incoming[0], IncomingCallback)
