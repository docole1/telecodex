"""Service-level routing from Telegram updates to isolated dialog contexts."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from telecodex.audio.transcriber import LocalAudioTranscriber
from telecodex.codex.session import ActiveThreadWriterError, CodexSession
from telecodex.config import AppConfig
from telecodex.group_triggers import parse_trigger_words
from telecodex.i18n import TextCatalog, load_text_catalog
from telecodex.logging_config import bind_log_context, log_exception, log_identifier
from telecodex.shell.runner import ShellRunner
from telecodex.state import RuntimeStateStore
from telecodex.telegram.api import TelegramAPI
from telecodex.telegram.updates import IncomingCallback, IncomingMessage, parse_command, parse_update
from telecodex.telegram.user_controller import TelegramUserController

logger = logging.getLogger(__name__)


class TelegramController:
    """Route Telegram updates to isolated private and shared group contexts."""

    def __init__(
        self,
        config: AppConfig,
        telegram: TelegramAPI,
        shell: ShellRunner,
        session_factory: Callable[[int, TelegramUserController], CodexSession],
        texts: TextCatalog | None = None,
        state_store: RuntimeStateStore | None = None,
        audio_transcriber: LocalAudioTranscriber | None = None,
        poll_health: Callable[[], dict[str, int | None]] | None = None,
    ) -> None:
        self.config = config
        self.telegram = telegram
        self.shell = shell
        self.session_factory = session_factory
        self.poll_health = poll_health
        self.texts = texts or load_text_catalog()
        self.audio_transcriber = audio_transcriber or LocalAudioTranscriber(
            config.transcription_model,
            config.transcription_device,
            config.transcription_compute_type,
            config.max_audio_duration_seconds,
        )
        self.state_store = state_store or RuntimeStateStore(
            config.database_file,
            config.telegram_allowed_user_ids[0],
            config.migration_state_file,
        )
        self.users: dict[int, TelegramUserController] = {}
        self.groups: dict[int, TelegramUserController] = {}

    async def start(self) -> None:
        for user_id in self.config.telegram_allowed_user_ids:
            controller = self._get_user_controller(user_id)
            await self._start_controller(controller)
        for chat_id in self.state_store.list_attached_groups():
            controller = self._get_group_controller(chat_id)
            await self._start_controller(controller)

    async def _start_controller(self, controller: TelegramUserController) -> None:
        codex = controller._codex()
        pending_turn = codex.pending_turn
        if pending_turn:
            controller._turn_prompt_message_id = pending_turn.prompt_message_id
        try:
            recovered_active = await codex.start()
        except ActiveThreadWriterError:
            logger.info(
                "Deferring Codex context startup because its thread is active elsewhere",
                extra={
                    "event": "codex.thread.active_elsewhere",
                    "context_hash": log_identifier(controller.chat_id, namespace="telegram-chat"),
                    "context_kind": "group" if controller.is_group else "private",
                },
            )
            await codex.close()
            return
        if recovered_active and codex.is_busy:
            controller._start_typing()
            await controller._send_codex_activity(
                controller._text("recovery.label.running"),
                reply_to_message_id=controller._turn_prompt_message_id,
            )

    def can_user_access_context(self, chat_id: int, user_id: int) -> bool:
        if chat_id == user_id:
            return user_id in self.config.telegram_allowed_user_ids
        if chat_id >= 0 or not self.state_store.is_group_attached(chat_id):
            return False
        return user_id in self.config.telegram_allowed_user_ids

    def active_context_controllers(self) -> tuple[TelegramUserController, ...]:
        """Return instantiated contexts for the recovery supervisor."""
        return tuple(self.users.values()) + tuple(self.groups.values())

    async def reload_mcp_config(self) -> tuple[int, int]:
        """Reload shared MCP configuration in every running Codex app-server."""
        controllers = self.active_context_controllers()
        reloaders = [
            getattr(controller._codex(), "reload_mcp_config", None)
            for controller in controllers
        ]
        active = [reloader for reloader in reloaders if callable(reloader)]
        if not active:
            return 0, 0
        results = await asyncio.gather(
            *(reloader() for reloader in active),
            return_exceptions=True,
        )
        succeeded = sum(result is True for result in results)
        failed = sum(isinstance(result, BaseException) for result in results)
        for result in results:
            if isinstance(result, BaseException):
                logger.warning(
                    "Could not reload MCP configuration in one Codex context",
                    extra={
                        "event": "codex.mcp.broadcast_reload_failed",
                        "error_type": type(result).__name__,
                    },
                )
        return succeeded, failed

    async def _handle_update(self, update: dict[str, Any]) -> None:
        incoming = parse_update(update)
        if incoming is None:
            return
        await self._handle_incoming(incoming)

    async def _handle_incoming(self, incoming: IncomingMessage | IncomingCallback) -> None:
        if isinstance(incoming, IncomingMessage) and await self._handle_group_command(incoming):
            return
        if isinstance(incoming, IncomingMessage) and await self._handle_group_setup_response(incoming):
            return
        if not self.can_user_access_context(incoming.chat_id, incoming.user_id):
            if isinstance(incoming, IncomingCallback):
                await self.telegram.answer_callback(incoming.callback_id)
            return
        controller = self._controller_for_update(incoming)
        if controller is None:
            return
        await controller._handle_incoming(incoming)

    async def _handle_group_setup_response(self, incoming: IncomingMessage) -> bool:
        if (
            incoming.chat_type not in {"group", "supergroup"}
            or incoming.user_id not in self.config.telegram_allowed_user_ids
            or incoming.attachments
        ):
            return False
        setup_user_id = self.state_store.pending_group_setup_user(incoming.chat_id)
        if setup_user_id is None or setup_user_id != incoming.user_id:
            return False
        if parse_command(incoming.text) is not None:
            return False
        try:
            trigger_words = parse_trigger_words(incoming.text)
        except ValueError:
            await self.telegram.send_message(
                incoming.chat_id,
                self.texts.render("ru", "groups.label.setup_invalid"),
                reply_to_message_id=incoming.message_id,
            )
            return True
        result = self.state_store.complete_group_setup(
            incoming.chat_id, incoming.user_id, trigger_words
        )
        if result is None:
            return False
        key = "groups.label.triggers_updated" if result == "updated" else "groups.label.attached"
        await self.telegram.send_message(
            incoming.chat_id,
            self.texts.render("ru", key, words=", ".join(trigger_words)),
            reply_to_message_id=incoming.message_id,
        )
        controller = self.groups.get(incoming.chat_id)
        if controller is not None:
            controller.attached = True
        return True

    async def _handle_group_command(self, incoming: IncomingMessage) -> bool:
        parsed = None if incoming.attachments else parse_command(incoming.text)
        supported = {"group_add", "group_remove", "group_mode"}
        if parsed is None or parsed[0] not in supported:
            return False
        command = parsed[0]
        is_group = incoming.chat_type in {"group", "supergroup"} and incoming.chat_id < 0
        if not is_group:
            if incoming.user_id not in self.config.telegram_allowed_user_ids:
                return True
            await self.telegram.send_message(
                incoming.chat_id,
                self.texts.render("ru", "groups.label.only_group"),
                reply_to_message_id=incoming.message_id,
            )
            return True
        if incoming.user_id not in self.config.telegram_allowed_user_ids:
            return True

        if command == "group_add":
            setup_status = self.state_store.begin_group_setup(
                incoming.chat_id, incoming.chat_title or str(incoming.chat_id), incoming.user_id
            )
            key = "groups.label.setup_pending_other" if setup_status == "pending_other" else "groups.label.setup_prompt"
            await self.telegram.send_message(
                incoming.chat_id,
                self.texts.render("ru", key),
                reply_to_message_id=incoming.message_id,
            )
            controller = self.groups.get(incoming.chat_id)
            if controller is not None:
                controller.attached = self.state_store.is_group_attached(incoming.chat_id)
            return True

        if command == "group_mode":
            if not self.state_store.is_group_attached(incoming.chat_id):
                await self.telegram.send_message(
                    incoming.chat_id,
                    self.texts.render("ru", "groups.label.not_attached"),
                    reply_to_message_id=incoming.message_id,
                )
                return True
            return False

        if not self.state_store.detach_group(incoming.chat_id):
            if self.state_store.cancel_group_setup(incoming.chat_id):
                await self.telegram.send_message(
                    incoming.chat_id,
                    self.texts.render("ru", "groups.label.setup_cancelled"),
                    reply_to_message_id=incoming.message_id,
                )
                return True
            await self.telegram.send_message(
                incoming.chat_id,
                self.texts.render("ru", "groups.label.not_attached"),
                reply_to_message_id=incoming.message_id,
            )
            return True
        await self.telegram.send_message(
            incoming.chat_id,
            self.texts.render("ru", "groups.label.detached"),
            reply_to_message_id=incoming.message_id,
        )
        controller = self.groups.get(incoming.chat_id)
        if controller is not None:
            await self._close_group_controller(controller, interrupt=True)
        return True

    async def _close_group_controller(
        self,
        controller: TelegramUserController,
        *,
        interrupt: bool,
    ) -> None:
        controller.attached = False
        await controller._finish_typing()
        await controller._close_compaction_input_queue()
        await controller._stop_group_activity()
        await controller._cancel_pending_prompts()
        try:
            await controller.close_shell_session()
        except Exception as exc:
            log_exception(
                logger,
                "Could not close group interactive shell",
                exc,
                event="telegram.group.shell_close_failed",
                level=logging.WARNING,
                context_hash=log_identifier(controller.chat_id, namespace="telegram-chat"),
            )
        if interrupt:
            try:
                with bind_log_context(interrupt_source="group_detach"):
                    await controller._codex().interrupt()
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not interrupt group Codex turn",
                    exc,
                    event="telegram.group.turn_interrupt_failed",
                    level=logging.WARNING,
                    context_hash=log_identifier(controller.chat_id, namespace="telegram-chat"),
                )
        try:
            await controller._codex().close()
        except Exception as exc:
            log_exception(
                logger,
                "Could not close group Codex session",
                exc,
                event="telegram.group.session_close_failed",
                level=logging.WARNING,
                context_hash=log_identifier(controller.chat_id, namespace="telegram-chat"),
            )

    def _controller_for_update(
        self, incoming: IncomingMessage | IncomingCallback
    ) -> TelegramUserController | None:
        if isinstance(incoming, IncomingMessage):
            if incoming.chat_type == "private" and incoming.chat_id == incoming.user_id:
                return self._get_user_controller(incoming.user_id)
            if (
                incoming.chat_type in {"group", "supergroup"}
                and self.state_store.is_group_attached(incoming.chat_id)
            ):
                return self._get_group_controller(incoming.chat_id)
            return None
        if incoming.chat_id == incoming.user_id:
            return self._get_user_controller(incoming.user_id)
        if self.state_store.is_group_attached(incoming.chat_id):
            return self._get_group_controller(incoming.chat_id)
        return None


    async def _handle_media_group(self, updates: list[dict[str, Any]]) -> None:
        if not updates:
            return
        incoming = parse_update(updates[0])
        if not isinstance(incoming, IncomingMessage):
            return
        if not self.can_user_access_context(incoming.chat_id, incoming.user_id):
            return
        controller = self._controller_for_update(incoming)
        if controller is None:
            return
        await controller._handle_media_group(updates)

    async def send_error_for_update(self, update: dict[str, Any], key: str) -> None:
        incoming = parse_update(update)
        if incoming is None or not self.can_user_access_context(incoming.chat_id, incoming.user_id):
            return
        controller = self._controller_for_update(incoming)
        if controller is None:
            return
        await controller._send_error(key)

    def _get_user_controller(self, user_id: int) -> TelegramUserController:
        controller = self.users.get(user_id)
        if controller is None:
            controller = TelegramUserController(
                self.config, self.telegram, self.shell, user_id, self.texts,
                audio_transcriber=self.audio_transcriber,
                poll_health=self.poll_health,
            )
            controller.bind_codex(self.session_factory(user_id, controller))
            controller.bind_mcp_reloader(self.reload_mcp_config)
            self.users[user_id] = controller
        return controller

    def _get_group_controller(self, chat_id: int) -> TelegramUserController:
        controller = self.groups.get(chat_id)
        if controller is None:
            controller = TelegramUserController(
                self.config,
                self.telegram,
                self.shell,
                chat_id,
                self.texts,
                is_group=True,
                group_trigger_words=lambda: self.state_store.group_trigger_words(chat_id),
                group_all_messages_to_codex=lambda: self.state_store.group_all_messages_to_codex(chat_id),
                set_group_all_messages_to_codex=lambda enabled: (
                    self.state_store.set_group_all_messages_to_codex(chat_id, enabled)
                ),
                audio_transcriber=self.audio_transcriber,
                authorize_user=lambda user_id: self.can_user_access_context(chat_id, user_id),
                poll_health=self.poll_health,
            )
            controller.bind_codex(self.session_factory(chat_id, controller))
            controller.bind_mcp_reloader(self.reload_mcp_config)
            self.groups[chat_id] = controller
        controller.attached = True
        return controller

    async def close(self) -> None:
        controllers = [*self.users.values(), *self.groups.values()]
        for controller in controllers:
            controller._shutdown_in_progress = True
            await controller._finish_typing()
            await controller._cancel_pending_prompts()
            await controller._close_compaction_input_queue()
        await asyncio.gather(
            *(controller.close_shell_session() for controller in controllers),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(controller._codex().close() for controller in controllers),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(controller._stop_group_activity() for controller in self.groups.values()),
            return_exceptions=True,
        )
