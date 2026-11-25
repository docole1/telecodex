"""Aiogram routers for TeleCodex messages and callback queries."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from aiogram import Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message, TelegramObject, Update
from aiogram.dispatcher.middlewares.base import BaseMiddleware

from telecodex.telegram.controller import TelegramController
from telecodex.telegram.media_groups import TelegramMediaGroupBuffer
from telecodex.telegram.markdown_input import MarkdownInputMiddleware
from telecodex.telegram.updates import (
    TelegramEvent,
    aiogram_update_to_dict,
    parse_aiogram_update,
)
from telecodex.telegram.user_controller import TelegramUserController


MESSAGE_ROUTES: dict[str, frozenset[str]] = {
    "group_commands": frozenset({"group_add", "group_remove", "group_mode"}),
    "conversation_commands": frozenset(
        {"start", "help", "rename", "delete", "fork", "compact", "recap", "init", "new", "files", "chats"}
    ),
    "settings_commands": frozenset(
        {"permissions", "plan", "goal", "model", "fast", "skills", "memories", "mcp", "language", "mode"}
    ),
    "task_and_diagnostics_commands": frozenset(
        {"stop", "ps", "shell", "exit", "status", "doctor", "recover"}
    ),
}

CALLBACK_ROUTES: dict[str, frozenset[str]] = {
    "group_and_interface_buttons": frozenset({"l", "om", "gm"}),
    "codex_settings_buttons": frozenset({"sk", "mm", "m", "pm"}),
    "conversation_and_file_buttons": frozenset({"np", "fb", "cp", "td", "sw", "rc", "rr", "ps"}),
    "mcp_buttons": frozenset({"mc", "me"}),
    "approval_buttons": frozenset({"ap"}),
}

ALL_COMMANDS = frozenset().union(*MESSAGE_ROUTES.values())
ALL_CALLBACK_PREFIXES = frozenset().union(*CALLBACK_ROUTES.values())


class TelegramContextMiddleware(BaseMiddleware):
    """Resolve context, persist albums, and run common stateful gates."""

    def __init__(
        self,
        controller: TelegramController,
        media_groups: TelegramMediaGroupBuffer,
    ) -> None:
        self.controller = controller
        self.media_groups = media_groups

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        update = data.get("telecodex_update")
        if not isinstance(update, Update):
            return await handler(event, data)

        incoming = parse_aiogram_update(
            update,
            model_text=data.get("telecodex_model_text"),
        )
        if isinstance(event, Message) and isinstance(incoming, Message):
            raw_update = aiogram_update_to_dict(update)
            await self.media_groups.flush_before(raw_update)
            if (
                incoming.media_group_id is not None
                and incoming.attachments
                and self.controller.can_user_access_context(incoming.chat_id, incoming.user_id)
                and (
                    (incoming.chat_type == "private" and incoming.chat_id == incoming.user_id)
                    or (
                        incoming.chat_type in {"group", "supergroup"}
                        and self.controller.state_store.is_group_attached(incoming.chat_id)
                    )
                )
            ):
                await self.media_groups.add(raw_update)
                return None
            prepared = await self.controller._prepare_incoming(incoming)
        elif isinstance(event, CallbackQuery) and isinstance(incoming, CallbackQuery):
            prepared = await self.controller._prepare_incoming(incoming)
        else:
            return await handler(event, data)

        if prepared is None:
            return None
        context_controller, context_update = prepared
        data["telecodex_incoming"] = context_update
        data["telecodex_context_controller"] = context_controller
        data["telecodex_preflighted"] = True
        return await handler(event, data)


class TelegramUpdateRouters:
    """Build Aiogram's dispatcher and its typed message/callback handlers."""

    def __init__(
        self,
        controller: TelegramController,
        media_groups: TelegramMediaGroupBuffer,
    ) -> None:
        self.controller = controller
        self.media_groups = media_groups
        self.dispatcher = Dispatcher()
        self.router = Router(name="telecodex")
        self.router.message.outer_middleware(MarkdownInputMiddleware())
        context_middleware = TelegramContextMiddleware(controller, media_groups)
        self.router.message.outer_middleware(context_middleware)
        self.router.callback_query.outer_middleware(context_middleware)
        self._register_message_routes()
        self._register_callback_routes()
        self.dispatcher.include_router(self.router)

    def _register_message_routes(self) -> None:
        for router_name, commands in MESSAGE_ROUTES.items():
            child = Router(name=router_name)
            for command in sorted(commands):
                child.message.register(
                    self._message_handler(command),
                    Command(command, ignore_case=True, ignore_mention=True),
                )
            self.router.include_router(child)

        fallback = Router(name="ordinary_messages_and_unknown_commands")
        fallback.message.register(self._message_handler(None))
        self.router.include_router(fallback)

    def _register_callback_routes(self) -> None:
        for router_name, prefixes in CALLBACK_ROUTES.items():
            child = Router(name=router_name)
            for prefix in sorted(prefixes):
                child.callback_query.register(
                    self._callback_handler(prefix),
                    self._callback_filter({prefix}),
                )
            self.router.include_router(child)

        fallback = Router(name="unknown_buttons")
        fallback.callback_query.register(self._callback_handler(None))
        self.router.include_router(fallback)

    def _message_handler(self, command: str | None) -> Callable[..., Any]:
        async def handle(
            message: Message,
            telecodex_incoming: TelegramEvent | None = None,
            telecodex_context_controller: TelegramUserController | None = None,
            **_data: Any,
        ) -> None:
            if not isinstance(telecodex_incoming, Message) or telecodex_context_controller is None:
                return
            await telecodex_context_controller._handle_message(
                telecodex_incoming,
                message_route=command,
                _preflighted=True,
            )

        handle.__name__ = f"handle_{command or 'ordinary_message'}"
        return handle

    def _callback_handler(self, prefix: str | None) -> Callable[..., Any]:
        async def handle(
            callback_query: CallbackQuery,
            telecodex_incoming: TelegramEvent | None = None,
            telecodex_context_controller: TelegramUserController | None = None,
            **_data: Any,
        ) -> None:
            if isinstance(telecodex_incoming, CallbackQuery) and telecodex_context_controller is not None:
                await telecodex_context_controller._handle_callback(
                    telecodex_incoming,
                    callback_route=prefix,
                )

        handle.__name__ = f"handle_callback_{prefix or 'unknown'}"
        return handle

    @staticmethod
    def _callback_filter(prefixes: set[str] | frozenset[str]):
        def matches(callback_query: CallbackQuery) -> bool:
            data = callback_query.data
            return isinstance(data, str) and data.split(":", 1)[0] in prefixes

        return matches
