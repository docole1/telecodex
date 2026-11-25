"""Aiogram callback handlers for TeleCodex UI actions."""

from __future__ import annotations

from aiogram.types import CallbackQuery

from telecodex.telegram.controller import TelegramController
from telecodex.telegram.updates import TelegramCallbackContext
from telecodex.telegram.user_controller import TelegramUserController


class TelegramCallbackHandlers:
    """Named callback route actions registered directly on Aiogram Routers."""

    def __init__(self, controller: TelegramController) -> None:
        self.controller = controller

    @staticmethod
    def _parts(context: TelegramCallbackContext) -> list[str]:
        return context.data.split(":")

    async def language(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_language_callback(telecodex_context, self._parts(telecodex_context))

    async def output_mode(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_mode_callback(telecodex_context, self._parts(telecodex_context))

    async def group_mode(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_group_mode_callback(telecodex_context, self._parts(telecodex_context))

    async def new_chat_defaults(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_new_chat_defaults_callback(
                telecodex_context,
                self._parts(telecodex_context),
            )

    async def skill(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_skills_callback(telecodex_context, self._parts(telecodex_context))

    async def memory(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_memory_callback(telecodex_context, self._parts(telecodex_context))

    async def model(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_model_callback(telecodex_context, self._parts(telecodex_context))

    async def permissions(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_permissions_callback(telecodex_context, self._parts(telecodex_context))

    async def directory(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_directory_callback(telecodex_context, self._parts(telecodex_context))

    async def files(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_files_callback(telecodex_context, self._parts(telecodex_context))

    async def chats(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_chats_callback(telecodex_context, self._parts(telecodex_context))

    async def delete(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_delete_callback(telecodex_context, self._parts(telecodex_context))

    async def thread_switch(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_thread_switch_callback(telecodex_context, self._parts(telecodex_context))

    async def stale_turn(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_stale_turn_callback(telecodex_context, self._parts(telecodex_context))

    async def recovery(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_chat_recovery_callback(telecodex_context, self._parts(telecodex_context))

    async def processes(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_processes_callback(telecodex_context, self._parts(telecodex_context))

    async def mcp(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_mcp_callback(telecodex_context, self._parts(telecodex_context))

    async def mcp_elicitation(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_mcp_elicitation_callback(telecodex_context, self._parts(telecodex_context))

    async def approval(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, telecodex_context_controller: TelegramUserController | None = None, **_data) -> None:
        if telecodex_context_controller is not None:
            await telecodex_context_controller._handle_approval_callback(telecodex_context, self._parts(telecodex_context))

    async def unknown(self, callback_query: CallbackQuery, telecodex_context: TelegramCallbackContext, **_data) -> None:
        await self.controller.telegram.answer_callback_query(
            telecodex_context.callback_id,
            self.controller.texts.render("ru", "common.label.stale_button"),
        )
