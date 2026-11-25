"""Extensible Aiogram routers for TeleCodex commands and callback queries."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from aiogram import Dispatcher, F, Router
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message, TelegramObject, Update

from telecodex.telegram.controller import TelegramController
from telecodex.telegram.callback_handlers import TelegramCallbackHandlers
from telecodex.telegram.command_handlers import TelegramCommandHandlers
from telecodex.telegram.media_groups import TelegramMediaGroupBuffer
from telecodex.telegram.markdown_input import MarkdownInputMiddleware
from telecodex.telegram.updates import (
    TelegramCallbackContext,
    TelegramEventContext,
    TelegramMessageContext,
    aiogram_update_to_dict,
    parse_aiogram_update,
    parse_command,
)


GROUP_COMMANDS = frozenset({"group_add", "group_remove", "group_mode"})


class TelegramContextMiddleware(BaseMiddleware):
    """Resolve the per-chat owner and run stateful gates before route handlers."""

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

        context = parse_aiogram_update(
            update,
            model_text=data.get("telecodex_model_text"),
        )
        if isinstance(event, Message) and isinstance(context, TelegramMessageContext):
            raw_update = aiogram_update_to_dict(update)
            await self.media_groups.flush_before(raw_update)
            if self._is_accepted_album(context):
                await self.media_groups.add(raw_update)
                return None

            parsed = None if context.attachments else parse_command(context.text)
            if parsed is not None and parsed[0] in GROUP_COMMANDS:
                if context.user_id not in self.controller.config.telegram_allowed_user_ids:
                    return None
                data["telecodex_context"] = context
                data["telecodex_context_controller"] = None
                return await handler(event, data)

            prepared = await self.controller._prepare_incoming(context)
        elif isinstance(event, CallbackQuery) and isinstance(context, TelegramCallbackContext):
            prepared = await self.controller._prepare_incoming(context)
        else:
            if isinstance(event, (Message, CallbackQuery)):
                return None
            return await handler(event, data)

        if prepared is None:
            return None
        context_controller, context = prepared
        data["telecodex_context"] = context
        data["telecodex_context_controller"] = context_controller
        return await handler(event, data)

    def _is_accepted_album(self, context: TelegramMessageContext) -> bool:
        return (
            context.media_group_id is not None
            and bool(context.attachments)
            and self.controller.can_user_access_context(context.chat_id, context.user_id)
            and (
                (context.chat_type == "private" and context.chat_id == context.user_id)
                or (
                    context.chat_type in {"group", "supergroup"}
                    and self.controller.state_store.is_group_attached(context.chat_id)
                )
            )
        )


class TelegramUpdateRouters:
    """Register direct Aiogram handlers and preserve album storage at the edge."""

    def __init__(
        self,
        controller: TelegramController,
        media_groups: TelegramMediaGroupBuffer,
    ) -> None:
        self.controller = controller
        self.media_groups = media_groups
        self.command_handlers = TelegramCommandHandlers(controller)
        self.callback_handlers = TelegramCallbackHandlers(controller)
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
        route_groups = (
            (
                "group_commands",
                (
                    ("group_add", self.command_handlers.group_add),
                    ("group_remove", self.command_handlers.group_remove),
                    ("group_mode", self.command_handlers.group_mode),
                ),
            ),
            (
                "conversation_commands",
                (
                    ("start", self.command_handlers.start),
                    ("help", self.command_handlers.help),
                    ("rename", self.command_handlers.rename),
                    ("delete", self.command_handlers.delete),
                    ("fork", self.command_handlers.fork),
                    ("compact", self.command_handlers.compact),
                    ("recap", self.command_handlers.recap),
                    ("init", self.command_handlers.init),
                    ("new", self.command_handlers.new),
                    ("files", self.command_handlers.files),
                    ("chats", self.command_handlers.chats),
                ),
            ),
            (
                "settings_commands",
                (
                    ("permissions", self.command_handlers.permissions),
                    ("plan", self.command_handlers.plan),
                    ("goal", self.command_handlers.goal),
                    ("model", self.command_handlers.model),
                    ("fast", self.command_handlers.fast),
                    ("skills", self.command_handlers.skills),
                    ("memories", self.command_handlers.memories),
                    ("mcp", self.command_handlers.mcp),
                    ("language", self.command_handlers.language),
                    ("mode", self.command_handlers.mode),
                ),
            ),
            (
                "task_and_diagnostics_commands",
                (
                    ("stop", self.command_handlers.stop),
                    ("ps", self.command_handlers.processes),
                    ("shell", self.command_handlers.shell),
                    ("exit", self.command_handlers.exit_shell),
                    ("status", self.command_handlers.status),
                    ("doctor", self.command_handlers.doctor),
                    ("recover", self.command_handlers.recover),
                ),
            ),
        )
        for router_name, routes in route_groups:
            child = Router(name=router_name)
            for command_name, handler in routes:
                child.message.register(
                    handler,
                    Command(command_name, ignore_case=True, ignore_mention=True),
                )
            self.router.include_router(child)

        fallback = Router(name="ordinary_messages_and_unknown_commands")
        fallback.message.register(self.command_handlers.ordinary_or_unknown)
        self.router.include_router(fallback)

    def _register_callback_routes(self) -> None:
        route_groups = (
            (
                "group_and_interface_buttons",
                (
                    ("l", self.callback_handlers.language),
                    ("om", self.callback_handlers.output_mode),
                    ("gm", self.callback_handlers.group_mode),
                ),
            ),
            (
                "codex_settings_buttons",
                (
                    ("sk", self.callback_handlers.skill),
                    ("mm", self.callback_handlers.memory),
                    ("m", self.callback_handlers.model),
                    ("pm", self.callback_handlers.permissions),
                ),
            ),
            (
                "conversation_and_file_buttons",
                (
                    ("np", self.callback_handlers.directory),
                    ("fb", self.callback_handlers.files),
                    ("cp", self.callback_handlers.chats),
                    ("td", self.callback_handlers.delete),
                    ("sw", self.callback_handlers.thread_switch),
                    ("rc", self.callback_handlers.stale_turn),
                    ("rr", self.callback_handlers.recovery),
                    ("ps", self.callback_handlers.processes),
                ),
            ),
            (
                "mcp_buttons",
                (
                    ("mc", self.callback_handlers.mcp),
                    ("me", self.callback_handlers.mcp_elicitation),
                ),
            ),
            (
                "approval_buttons",
                (("ap", self.callback_handlers.approval),),
            ),
        )
        for router_name, routes in route_groups:
            child = Router(name=router_name)
            for prefix, handler in routes:
                child.callback_query.register(handler, F.data.startswith(f"{prefix}:"))
            self.router.include_router(child)

        fallback = Router(name="unknown_buttons")
        fallback.callback_query.register(self.callback_handlers.unknown)
        self.router.include_router(fallback)
