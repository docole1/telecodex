"""Aiogram message handlers for TeleCodex commands and ordinary input."""

from __future__ import annotations

from aiogram.filters.command import CommandObject
from aiogram.types import Message

from telecodex.telegram.controller import TelegramController
from telecodex.telegram.updates import (
    TelegramMessageContext,
    command_argument_markdown,
    parse_command,
)
from telecodex.telegram.user_controller import TelegramUserController


class TelegramCommandHandlers:
    """Named route actions. Add a method and register it on a Router to extend."""

    def __init__(self, controller: TelegramController) -> None:
        self.controller = controller

    @staticmethod
    def _user(controller: TelegramUserController | None) -> TelegramUserController | None:
        return controller

    async def start(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_help_command(telecodex_context)

    async def help(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_help_command(telecodex_context)

    async def group_add(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        **_data,
    ) -> None:
        await self.controller.handle_group_add_command(telecodex_context)

    async def group_remove(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        **_data,
    ) -> None:
        await self.controller.handle_group_remove_command(telecodex_context)

    async def group_mode(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        **_data,
    ) -> None:
        await self.controller.handle_group_mode_command(telecodex_context)

    async def doctor(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_doctor_command(telecodex_context)

    async def status(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_status_command(telecodex_context)

    async def permissions(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._show_permissions_picker(telecodex_context.chat_id)

    async def rename(
        self,
        message: Message,
        command: CommandObject,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._rename_chat(telecodex_context.chat_id, command.args or "")

    async def delete(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._show_delete_confirmation(telecodex_context.chat_id)

    async def fork(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._fork_chat(telecodex_context.chat_id)

    async def compact(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._compact_chat(telecodex_context.chat_id)

    async def plan(
        self,
        message: Message,
        command: CommandObject,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_plan_command(
                telecodex_context,
                command.args or "",
                command_argument_markdown(telecodex_context),
            )

    async def goal(
        self,
        message: Message,
        command: CommandObject,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_goal_command(
                telecodex_context,
                command.args or "",
                command_argument_markdown(telecodex_context),
            )

    async def recap(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._recap_chat(telecodex_context)

    async def init(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._init_project(telecodex_context)

    async def processes(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_processes_command(telecodex_context)

    async def new(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._show_directory_picker(telecodex_context.chat_id)

    async def files(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._show_files_picker(telecodex_context.chat_id, telecodex_context.user_id)

    async def chats(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._show_chats_picker(telecodex_context.chat_id)

    async def stop(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_stop_command(telecodex_context)

    async def model(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._show_model_picker(telecodex_context.chat_id)

    async def fast(
        self,
        message: Message,
        command: CommandObject,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_fast_command(telecodex_context, command.args or "")

    async def skills(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._show_skills_picker(telecodex_context.chat_id)

    async def memories(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._show_memory_picker(telecodex_context.chat_id)

    async def mcp(
        self,
        message: Message,
        command: CommandObject,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._show_mcp_picker(
                telecodex_context.chat_id,
                command.args or "",
                telecodex_context.user_id,
            )

    async def language(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._show_language_picker(telecodex_context.chat_id)

    async def mode(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._show_mode_picker(telecodex_context.chat_id)

    async def shell(
        self,
        message: Message,
        command: CommandObject,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_shell_command(telecodex_context, command.args or "")

    async def exit_shell(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_exit_command(telecodex_context)

    async def recover(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if user := self._user(telecodex_context_controller):
            await user._handle_recover_command(telecodex_context)

    async def ordinary_or_unknown(
        self,
        message: Message,
        telecodex_context: TelegramMessageContext,
        telecodex_context_controller: TelegramUserController | None = None,
        **_data,
    ) -> None:
        if telecodex_context_controller is None:
            return
        parsed = None if telecodex_context.attachments else parse_command(telecodex_context.text)
        if parsed is not None:
            await telecodex_context_controller._handle_unknown_command(telecodex_context, parsed[0])
        else:
            await telecodex_context_controller._handle_plain_message(telecodex_context)
