"""Aiogram entry routers that adapt Telegram updates to TeleCodex controllers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from aiogram import Dispatcher, Router
from aiogram.types import CallbackQuery, Message, Update

from telecodex.telegram.controller import TelegramController
from telecodex.telegram.media_groups import TelegramMediaGroupBuffer
from telecodex.telegram.markdown_input import MarkdownInputMiddleware
from telecodex.telegram.updates import (
    IncomingCallback,
    IncomingMessage,
    IncomingUpdate,
    aiogram_update_to_dict,
    parse_aiogram_update,
    parse_command,
)


class TelegramUpdateRouters:
    """Bridge typed Aiogram events into existing per-chat application controllers."""

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
        self._register_message_routes()
        self._register_callback_routes()
        self.dispatcher.include_router(self.router)

    def _register_message_routes(self) -> None:
        routes = (
            (
                "group_commands",
                {"group_add", "group_remove", "group_mode"},
            ),
            (
                "conversation_commands",
                {"start", "help", "rename", "delete", "fork", "compact", "recap", "init", "new", "files", "chats"},
            ),
            (
                "settings_commands",
                {"permissions", "plan", "goal", "model", "fast", "skills", "memories", "mcp", "language", "mode"},
            ),
            (
                "task_and_diagnostics_commands",
                {"stop", "ps", "shell", "exit", "status", "doctor", "recover"},
            ),
        )
        for name, commands in routes:
            child = Router(name=name)
            child.message.register(self._handle_message, self._command_filter(commands))
            self.router.include_router(child)

        fallback = Router(name="ordinary_messages_and_unknown_commands")
        fallback.message.register(self._handle_message)
        self.router.include_router(fallback)

    def _register_callback_routes(self) -> None:
        routes = (
            ("group_and_interface_buttons", {"l", "om", "gm"}),
            ("codex_settings_buttons", {"sk", "mm", "m", "pm"}),
            ("conversation_and_file_buttons", {"np", "fb", "cp", "td", "sw", "rc", "rr", "ps"}),
            ("mcp_buttons", {"mc", "me"}),
            ("approval_buttons", {"ap"}),
        )
        for name, prefixes in routes:
            child = Router(name=name)
            child.callback_query.register(self._handle_callback, self._callback_filter(prefixes))
            self.router.include_router(child)

        fallback = Router(name="unknown_buttons")
        fallback.callback_query.register(self._handle_callback)
        self.router.include_router(fallback)

    @staticmethod
    def _command_filter(commands: set[str]):
        def matches(message: Message) -> bool:
            parsed = parse_command(message.text or "")
            return parsed is not None and parsed[0] in commands

        return matches

    @staticmethod
    def _callback_filter(prefixes: set[str]):
        def matches(callback_query: CallbackQuery) -> bool:
            data = callback_query.data
            return isinstance(data, str) and data.split(":", 1)[0] in prefixes

        return matches

    async def _handle_message(
        self,
        message: Message,
        telecodex_update: Update,
        telecodex_incoming: IncomingUpdate | None = None,
        telecodex_model_text: str | None = None,
        **_data: Any,
    ) -> None:
        incoming = telecodex_incoming or parse_aiogram_update(
            telecodex_update,
            model_text=telecodex_model_text,
        )
        if not isinstance(incoming, IncomingMessage):
            return
        if telecodex_model_text is not None:
            incoming = replace(incoming, model_text=telecodex_model_text)

        raw_update = aiogram_update_to_dict(telecodex_update)
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
            return
        await self.controller._handle_incoming(incoming)

    async def _handle_callback(
        self,
        callback_query: CallbackQuery,
        telecodex_update: Update,
        telecodex_incoming: IncomingUpdate | None = None,
        **_data: Any,
    ) -> None:
        incoming = telecodex_incoming or parse_aiogram_update(telecodex_update)
        if isinstance(incoming, IncomingCallback):
            await self.controller._handle_incoming(incoming)
