"""Settings pickers and callbacks for a Telegram user context."""

from __future__ import annotations

import logging
import secrets
from typing import Any, Callable

from aiogram import Bot

from telecodex.codex.permissions import PERMISSION_MODES
from telecodex.codex.types import CodexModel, FastTierUnavailableError, MemoryFeatureUnavailableError
from telecodex.i18n import SUPPORTED_LANGUAGES, TextCatalog
from telecodex.state import OUTPUT_MODE_CHOICES
from telecodex.telegram.formatting import truncate
from telecodex.telegram.pagination import (
    INLINE_BUTTON_TEXT_LIMIT,
    PAGE_SIZE,
    keyboard,
    navigation_row,
    page_count,
    two_column_rows,
)
from telecodex.telegram.pickers import (
    GroupModePicker,
    LanguagePicker,
    MemoryPicker,
    ModelPicker,
    ModePicker,
    PermissionsPicker,
    SkillPicker,
)
from telecodex.telegram.updates import TelegramCallbackContext, TelegramMessageContext
class TelegramSettingsController:
    """Own the model, permission, language, mode, skill, and memory pickers."""

    def __init__(
        self,
        *,
        telegram: Bot,
        chat_id: int,
        is_group: bool,
        is_attached: Callable[[], bool],
        codex: Callable[[], Any],
        text: Callable[..., str],
        texts: TextCatalog,
        selected_button_label: Callable[..., str],
        edit_message: Callable[..., Any],
        log_failure: Callable[..., None],
        language: Callable[[], str],
        selected_output_mode: Callable[[], str],
        group_all_messages_to_codex: Callable[[], bool],
        set_group_all_messages_to_codex: Callable[[bool], bool] | None,
    ) -> None:
        self.telegram = telegram
        self.chat_id = chat_id
        self.is_group = is_group
        self._is_attached = is_attached
        self._codex = codex
        self._text = text
        self.texts = texts
        self._selected_button_label = selected_button_label
        self._edit_message = edit_message
        self._log_failure = log_failure
        self._language = language
        self._selected_output_mode = selected_output_mode
        self._group_all_messages_to_codex = group_all_messages_to_codex
        self._set_group_all_messages_to_codex = set_group_all_messages_to_codex
        self.model_picker: ModelPicker | None = None
        self.language_picker: LanguagePicker | None = None
        self.mode_picker: ModePicker | None = None
        self.group_mode_picker: GroupModePicker | None = None
        self.skill_picker: SkillPicker | None = None
        self.memory_picker: MemoryPicker | None = None
        self.permissions_picker: PermissionsPicker | None = None

    @property
    def attached(self) -> bool:
        return self._is_attached()

    async def _show_model_picker(self, chat_id: int) -> None:
        try:
            codex = self._codex()
            models = await codex.list_models()
        except Exception as exc:
            self._log_failure(
                "Could not list Codex models",
                exc,
                event="telegram.models.list_failed",
            )
            await self.telegram.send_message(chat_id, self._text("models.label.list_failure"))
            return
        if not models:
            await self.telegram.send_message(chat_id, self._text("models.label.empty"))
            return

        try:
            current_model, current_effort = await codex.current_model_settings(models)
        except Exception as exc:
            self._log_failure(
                "Could not read the current model settings for the Codex thread",
                exc,
                event="telegram.models.current_settings_failed",
                level=logging.WARNING,
            )
            current_model = codex.selected_model or codex.default_model(models)
            current_effort = codex.selected_effort
        picker = ModelPicker(
            token=secrets.token_urlsafe(6),
            models=models,
            page=0,
            current_model=current_model,
            current_effort=current_effort,
        )
        message = await self.telegram.send_message(
            chat_id,
            self._model_picker_text(picker),
            reply_markup=self._model_picker_keyboard(picker),
        )
        message_id = message.message_id
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.model_picker = picker


    async def _show_language_picker(self, chat_id: int) -> None:
        picker = LanguagePicker(token=secrets.token_urlsafe(6))
        keyboard = self._language_picker_keyboard(picker)
        message = await self.telegram.send_message(
            chat_id,
            self.texts.system("label.language_picker_prompt"),
            reply_markup=keyboard,
        )
        message_id = message.message_id
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.language_picker = picker


    async def _show_mode_picker(self, chat_id: int) -> None:
        picker = ModePicker(token=secrets.token_urlsafe(6))
        message = await self.telegram.send_message(
            chat_id,
            self._mode_picker_text(),
            reply_markup=self._mode_picker_keyboard(picker),
        )
        message_id = message.message_id
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.mode_picker = picker


    async def _show_skills_picker(self, chat_id: int) -> None:
        try:
            skills = await self._codex().list_skills()
        except Exception as exc:
            self._log_failure(
                "Could not list Codex skills",
                exc,
                event="telegram.skills.list_failed",
            )
            await self.telegram.send_message(chat_id, self._text("skills.label.list_failed"))
            return
        if not skills:
            await self.telegram.send_message(chat_id, self._text("skills.label.empty"))
            return
        picker = SkillPicker(token=secrets.token_urlsafe(6), skills=skills)
        message = await self.telegram.send_message(
            chat_id,
            self._skills_picker_text(picker),
            reply_markup=self._skills_picker_keyboard(picker),
        )
        message_id = message.message_id
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.skill_picker = picker


    def _skills_picker_text(self, picker: SkillPicker) -> str:
        if picker.action == "menu":
            selected = self._codex().settings_manager.pending_skill_name
            selected_text = (
                self._text("skills.values.selected", name=selected)
                if selected
                else self._text("skills.values.none_selected")
            )
            return self._text("skills.label.menu", selected=selected_text)
        title = self._text(
            "skills.label.list_title",
            mode=self._text(f"skills.values.actions.{picker.action}"),
            current=picker.page + 1,
            total=page_count(len(picker.skills)),
        )
        if picker.action == "manage":
            return f"{title}\n{self._text('skills.label.manage_notice')}"
        return title


    def _skills_picker_keyboard(self, picker: SkillPicker) -> dict[str, Any]:
        if picker.action == "menu":
            return {
                "inline_keyboard": [
                    [{"text": self._text("skills.buttons.use_button"), "callback_data": f"sk:{picker.token}:a:use"}],
                    [{"text": self._text("skills.buttons.manage_button"), "callback_data": f"sk:{picker.token}:a:manage"}],
                ]
            }
        start = picker.page * PAGE_SIZE
        skills = picker.skills[start : start + PAGE_SIZE]
        buttons = []
        for offset, skill in enumerate(skills):
            absolute_index = start + offset
            label = skill.display_name or skill.name
            if picker.action == "manage":
                label = self._text(
                    "skills.buttons.enabled_prefix" if skill.enabled else "skills.buttons.disabled_prefix",
                    name=label,
                )
                action = "toggle"
            else:
                action = "use"
            buttons.append({
                "text": truncate(
                    label,
                    INLINE_BUTTON_TEXT_LIMIT,
                    self.texts.system("values.ellipsis"),
                ),
                "callback_data": f"sk:{picker.token}:{action}:{absolute_index}",
            })
        rows = two_column_rows(buttons, 0)
        total_pages = page_count(len(picker.skills))
        if total_pages > 1:
            rows.append(
                navigation_row(
                    picker.page,
                    total_pages,
                    "‹",
                    self._text(
                        "skills.buttons.page",
                        current="{current}",
                        total="{total}",
                    ),
                    "›",
                    f"sk:{picker.token}",
                )
            )
        rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"sk:{picker.token}:b"}])
        return {"inline_keyboard": rows}


    async def _show_memory_picker(self, chat_id: int) -> None:
        codex = self._codex()
        try:
            settings = await codex.read_memory_settings()
        except Exception as exc:
            self._log_failure(
                "Could not read Codex memory settings",
                exc,
                event="telegram.memories.read_failed",
            )
            await self.telegram.send_message(chat_id, self._text("memories.label.read_failed"))
            return
        current_mode = codex.current_memory_mode()
        current_use_memories = (
            current_mode == "enabled"
            if current_mode is not None
            else (settings.use_memories if codex.state.thread_id is None else None)
        )
        picker = MemoryPicker(
            token=secrets.token_urlsafe(6),
            thread_id=codex.state.thread_id,
            settings=settings,
            use_memories=current_use_memories,
        )
        message = await self.telegram.send_message(
            chat_id,
            self._memory_picker_text(picker),
            reply_markup=self._memory_picker_keyboard(picker),
        )
        message_id = message.message_id
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.memory_picker = picker


    def _memory_picker_text(self, picker: MemoryPicker) -> str:
        if picker.confirming_generation:
            return self._text("memories.label.generate_confirmation")
        if picker.confirming_reset:
            return self._text("memories.label.reset_confirmation")
        if not picker.settings.feature_enabled:
            return self._text("memories.label.feature_disabled")
        return self._text(
            "memories.label.prompt",
            scope=self._text("memories.values.current_scope" if picker.thread_id else "memories.values.new_scope"),
            use_state=self._text(
                "memories.values.enabled"
                if picker.use_memories is True
                else "memories.values.disabled"
                if picker.use_memories is False
                else "memories.values.unknown"
            ),
            generate_state=self._text(
                "memories.values.enabled" if picker.settings.generate_memories else "memories.values.disabled"
            ),
        )


    def _memory_picker_keyboard(self, picker: MemoryPicker) -> dict[str, Any]:
        if picker.confirming_generation:
            return {
                "inline_keyboard": [
                    [{"text": self._text("memories.buttons.confirm_button"), "callback_data": f"mm:{picker.token}:confirm"}],
                    [{"text": self._text("common.buttons.cancel"), "callback_data": f"mm:{picker.token}:cancel"}],
                ]
            }
        if picker.confirming_reset:
            return {
                "inline_keyboard": [
                    [{"text": self._text("memories.buttons.reset_confirm_button"), "callback_data": f"mm:{picker.token}:reset_confirm"}],
                    [{"text": self._text("common.buttons.cancel"), "callback_data": f"mm:{picker.token}:cancel"}],
                ]
            }
        if not picker.settings.feature_enabled:
            return {
                "inline_keyboard": [[{
                    "text": self._text("memories.buttons.enable_button"),
                    "callback_data": f"mm:{picker.token}:enable",
                }]]
            }
        memory_mode_buttons = two_column_rows(
            [
                {
                    "text": self._selected_button_label(
                        self._text("memories.buttons.enable_use_button"),
                        picker.use_memories is True,
                    ),
                    "callback_data": f"mm:{picker.token}:use:enabled",
                },
                {
                    "text": self._selected_button_label(
                        self._text("memories.buttons.disable_use_button"),
                        picker.use_memories is False,
                    ),
                    "callback_data": f"mm:{picker.token}:use:disabled",
                },
            ],
            0,
        )
        return {
            "inline_keyboard": memory_mode_buttons + [
                [{
                    "text": self._text(
                        "memories.buttons.generate_button",
                        state=self._text("memories.values.enabled" if picker.settings.generate_memories else "memories.values.disabled"),
                    ),
                    "callback_data": f"mm:{picker.token}:generate",
                }],
                [{
                    "text": self._text("memories.buttons.reset_button"),
                    "callback_data": f"mm:{picker.token}:reset",
                }],
            ]
        }


    async def _show_permissions_picker(self, chat_id: int) -> None:
        picker = PermissionsPicker(token=secrets.token_urlsafe(6))
        message = await self.telegram.send_message(
            chat_id,
            self._permissions_picker_text(),
            reply_markup=self._permissions_picker_keyboard(picker),
        )
        message_id = message.message_id
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.permissions_picker = picker


    async def _handle_fast_command(self, incoming: TelegramMessageContext, argument: str) -> None:
        requested = argument.strip().casefold()
        if requested:
            await self.telegram.send_message(incoming.chat_id, self._text("fast.label.usage"))
            return
        try:
            current = await self._codex().toggle_fast_mode()
        except FastTierUnavailableError:
            await self.telegram.send_message(incoming.chat_id, self._text("fast.label.unavailable"))
        except Exception as exc:
            self._log_failure(
                "Could not change the Codex Fast service tier",
                exc,
                event="telegram.command.fast_failed",
            )
            await self.telegram.send_message(incoming.chat_id, self._text("fast.label.failed"))
        else:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("fast.label.enabled" if current else "fast.label.disabled"),
            )


    def _permissions_picker_text(self) -> str:
        codex = self._codex()
        mode = codex.permission_mode
        managed = True
        if mode is None and codex.state.thread_id:
            managed = codex.state_store.load_thread(codex.state.thread_id).managed_by_telecodex
        if mode is None and codex.state.thread_id and not managed:
            description = self._text("permissions.values.current_unknown")
        elif mode is None and codex.config.approval_policy != "on-request":
            description = self._text(
                "permissions.values.current_default",
                policy=codex.config.approval_policy,
            )
        else:
            description = self._text(
                "permissions.values.current",
                mode=self._text(f"permissions.buttons.mode_names.{mode or 'ask'}"),
            )
        if codex.is_busy:
            description += self._text("permissions.values.next_turn")
        scope_key = "permissions.values.current_thread" if codex.state.thread_id else "permissions.values.next_thread"
        return self._text("permissions.label.prompt", scope=self._text(scope_key), current=description)


    def _permissions_picker_keyboard(self, picker: PermissionsPicker) -> dict[str, Any]:
        codex = self._codex()
        selected = codex.permission_mode
        managed = True
        if selected is None and codex.state.thread_id:
            managed = codex.state_store.load_thread(codex.state.thread_id).managed_by_telecodex
        if selected is None and managed and codex.config.approval_policy == "on-request":
            selected = "ask"
        modes = PERMISSION_MODES
        rows = []
        for mode in modes:
            label = self._text(f"permissions.buttons.mode_names.{mode}")
            label = self._selected_button_label(label, mode == selected)
            rows.append(
                [{"text": label, "callback_data": f"pm:{picker.token}:{mode}"}]
            )
        return {"inline_keyboard": rows}


    async def _handle_permissions_callback(
        self,
        incoming: TelegramCallbackContext,
        parts: list[str],
    ) -> None:
        picker = self.permissions_picker
        if (
            picker is None
            or len(parts) != 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("permissions.label.stale"))
            return
        mode = parts[2]
        if mode not in PERMISSION_MODES:
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("permissions.label.stale"))
            return
        try:
            self._codex().select_permission_mode(mode)
        except Exception as exc:
            self._log_failure(
                "Could not select Codex permission mode",
                exc,
                event="telegram.permissions.select_failed",
                permission_mode=mode,
            )
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("permissions.label.save_failed"))
            return
        await self.telegram.answer_callback_query(
            incoming.callback_id,
            self._text("permissions.label.selection_saved"),
        )
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self._permissions_picker_text(),
            self._permissions_picker_keyboard(picker),
        )


    def _language_picker_keyboard(self, picker: LanguagePicker) -> dict[str, Any]:
        selected = self._language()
        rows = [
            [
                {
                    "text": self._selected_button_label(
                        self.texts.system(f"values.language_names.{language}"),
                        language == selected,
                    ),
                    "callback_data": f"l:{picker.token}:{language}",
                }
            ]
            for language in SUPPORTED_LANGUAGES
        ]
        return {"inline_keyboard": rows}


    def _mode_picker_text(self) -> str:
        return self._text(
            "mode.label.prompt",
            current=self._text(f"mode.buttons.names.{self._selected_output_mode()}"),
        )


    def _mode_picker_keyboard(self, picker: ModePicker) -> dict[str, Any]:
        selected = self._selected_output_mode()
        rows = [
            [
                {
                    "text": self._selected_button_label(
                        self._text(f"mode.buttons.names.{mode}"),
                        mode == selected,
                    ),
                    "callback_data": f"om:{picker.token}:{mode}",
                }
            ]
            for mode in ("answers", "quiet_answers", "detailed")
        ]
        return {"inline_keyboard": rows}


    def _group_mode_name(self, mode: str) -> str:
        key = "all_messages" if mode == "all" else "reply_or_trigger"
        return self._text(f"groups.buttons.mode_names.{key}")


    def _group_mode_picker_text(self) -> str:
        mode = "all" if self._group_all_messages_to_codex() else "triggered"
        return self._text(
            "groups.label.mode_prompt",
            current=self._group_mode_name(mode),
        )


    def _group_mode_picker_keyboard(self, picker: GroupModePicker) -> dict[str, Any]:
        selected = "all" if self._group_all_messages_to_codex() else "triggered"
        rows = []
        for mode, name in (
            ("triggered", "reply_or_trigger"),
            ("all", "all_messages"),
        ):
            label = self._text(f"groups.buttons.mode_names.{name}")
            label = self._selected_button_label(label, mode == selected)
            rows.append(
                [{"text": label, "callback_data": f"gm:{picker.token}:{mode}"}]
            )
        return {"inline_keyboard": rows}


    async def _show_group_mode_picker(self, incoming: TelegramMessageContext) -> None:
        if not self.is_group:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("groups.label.only_group"),
                reply_to_message_id=incoming.message_id,
            )
            return
        if not self.attached:
            await self.telegram.send_message(
                incoming.chat_id,
                self._text("groups.label.not_attached"),
                reply_to_message_id=incoming.message_id,
            )
            return
        picker = GroupModePicker(token=secrets.token_urlsafe(5))
        message = await self.telegram.send_message(
            incoming.chat_id,
            self._group_mode_picker_text(),
            reply_markup=self._group_mode_picker_keyboard(picker),
            reply_to_message_id=incoming.message_id,
        )
        picker.message_id = message.message_id
        self.group_mode_picker = picker


    async def _handle_group_mode_callback(
        self,
        incoming: TelegramCallbackContext,
        parts: list[str],
    ) -> None:
        picker = self.group_mode_picker
        if (
            not self.is_group
            or not self.attached
            or picker is None
            or len(parts) != 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
            or parts[2] not in {"triggered", "all"}
            or self._set_group_all_messages_to_codex is None
        ):
            await self.telegram.answer_callback_query(
                incoming.callback_id,
                self._text("groups.label.mode_stale"),
            )
            return

        enabled = parts[2] == "all"
        try:
            saved = self._set_group_all_messages_to_codex(enabled)
        except Exception as exc:
            self._log_failure(
                "Could not save group message routing mode",
                exc,
                event="telegram.group.mode_save_failed",
                enabled=enabled,
            )
            await self.telegram.answer_callback_query(
                incoming.callback_id,
                self._text("groups.label.mode_save_failed"),
            )
            return
        if not saved:
            await self.telegram.answer_callback_query(
                incoming.callback_id,
                self._text("groups.label.mode_stale"),
            )
            return
        await self.telegram.answer_callback_query(
            incoming.callback_id,
            self._text("groups.label.mode_saved", mode=self._group_mode_name(parts[2])),
        )
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self._group_mode_picker_text(),
            self._group_mode_picker_keyboard(picker),
        )


    async def _handle_mode_callback(
        self,
        incoming: TelegramCallbackContext,
        parts: list[str],
    ) -> None:
        picker = self.mode_picker
        if (
            picker is None
            or len(parts) != 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("mode.label.stale"))
            return
        mode = parts[2]
        if mode not in OUTPUT_MODE_CHOICES:
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("mode.label.stale"))
            return

        try:
            self._codex().select_output_mode(mode)
        except Exception as exc:
            self._log_failure(
                "Could not select output mode",
                exc,
                event="telegram.output_mode.select_failed",
                outcome=mode,
            )
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("mode.label.save_failed"))
            return
        await self.telegram.answer_callback_query(incoming.callback_id)
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self._mode_picker_text(),
            self._mode_picker_keyboard(picker),
        )


    async def _handle_skills_callback(
        self,
        incoming: TelegramCallbackContext,
        parts: list[str],
    ) -> None:
        picker = self.skill_picker
        if (
            picker is None
            or len(parts) < 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("skills.label.stale"))
            return
        action = parts[2]
        if action == "a" and len(parts) == 4 and parts[3] in {"use", "manage"}:
            picker.action = parts[3]
            picker.page = 0
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._skills_picker_text(picker),
                self._skills_picker_keyboard(picker),
            )
            return
        if action == "b" and len(parts) == 3:
            picker.action = "menu"
            picker.page = 0
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._skills_picker_text(picker),
                self._skills_picker_keyboard(picker),
            )
            return
        if action in {"n", "i"} and len(parts) == 3:
            await self.telegram.answer_callback_query(incoming.callback_id)
            return
        if action == "p" and len(parts) == 4:
            try:
                page = int(parts[3])
            except ValueError:
                page = -1
            if page < 0 or page >= page_count(len(picker.skills)):
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("skills.label.stale"))
                return
            picker.page = page
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._skills_picker_text(picker),
                self._skills_picker_keyboard(picker),
            )
            return
        if action in {"use", "toggle"} and len(parts) == 4:
            try:
                index = int(parts[3])
                first_index = picker.page * PAGE_SIZE
                if not first_index <= index < min(first_index + PAGE_SIZE, len(picker.skills)):
                    raise ValueError("Skill button is not on the current page")
                skill = picker.skills[index]
            except (IndexError, ValueError):
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("skills.label.stale"))
                return
            if action == "use":
                if not skill.enabled:
                    await self.telegram.answer_callback_query(incoming.callback_id, self._text("skills.label.disabled"))
                    return
                try:
                    self._codex().select_skill_for_next_turn(skill)
                except Exception as exc:
                    self._log_failure(
                        "Could not select a Codex skill",
                        exc,
                        event="telegram.skills.select_failed",
                    )
                    await self.telegram.answer_callback_query(incoming.callback_id, self._text("skills.label.select_failed"))
                    return
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("skills.label.selected_toast"))
                self.skill_picker = None
                await self._edit_message(
                    incoming.chat_id,
                    incoming.message_id,
                    self._text("skills.label.selected_next", name=skill.display_name or skill.name),
                    {"inline_keyboard": []},
                )
                return
            try:
                await self._codex().set_skill_enabled(skill, not skill.enabled)
                if skill.enabled and (
                    self._codex().settings_manager.pending_skill_name == skill.name
                    and self._codex().settings_manager.pending_skill_path == skill.path
                ):
                    self._codex().clear_selected_skill()
                picker.skills = await self._codex().list_skills(force_reload=True)
            except Exception as exc:
                self._log_failure(
                    "Could not change Codex skill configuration",
                    exc,
                    event="telegram.skills.toggle_failed",
                    skill_name=skill.name,
                )
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("skills.label.update_failed"))
                return
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("skills.label.updated"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._skills_picker_text(picker),
                self._skills_picker_keyboard(picker),
            )
            return
        await self.telegram.answer_callback_query(incoming.callback_id, self._text("skills.label.stale"))


    async def _handle_memory_callback(
        self,
        incoming: TelegramCallbackContext,
        parts: list[str],
    ) -> None:
        picker = self.memory_picker
        if (
            picker is None
            or len(parts) not in {3, 4}
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
            or picker.thread_id != self._codex().state.thread_id
        ):
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("memories.label.stale"))
            return
        codex = self._codex()
        action = parts[2]
        if action == "cancel" and (picker.confirming_generation or picker.confirming_reset):
            picker.confirming_generation = False
            picker.confirming_reset = False
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if action == "reset":
            picker.confirming_reset = True
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if action == "reset_confirm" and picker.confirming_reset:
            try:
                await codex.reset_memories()
                picker.confirming_reset = False
            except Exception as exc:
                self._log_failure(
                    "Could not reset Codex memories",
                    exc,
                    event="telegram.memories.reset_failed",
                )
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("memories.label.reset_failed"))
                return
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("memories.label.reset_done"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if action == "generate":
            picker.confirming_generation = True
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if action == "confirm" and picker.confirming_generation:
            try:
                await codex.toggle_memory_generation()
                picker.settings = await codex.read_memory_settings()
                picker.confirming_generation = False
            except Exception as exc:
                self._log_failure(
                    "Could not change the Codex memory generation setting",
                    exc,
                    event="telegram.memories.generation_write_failed",
                )
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("memories.label.save_failed"))
                return
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("memories.label.saved"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if action == "enable" and not picker.settings.feature_enabled:
            try:
                await codex.enable_memories_feature()
                picker.settings = await codex.read_memory_settings()
                picker.use_memories = picker.settings.use_memories
            except Exception as exc:
                self._log_failure(
                    "Could not enable Codex memories for new chats",
                    exc,
                    event="telegram.memories.feature_enable_failed",
                )
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("memories.label.save_failed"))
                return
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("memories.label.feature_enabled"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        if (
            action == "use"
            and len(parts) == 4
            and parts[3] in {"enabled", "disabled"}
            and picker.settings.feature_enabled
        ):
            target_mode = parts[3]
            try:
                await codex.set_memory_mode(target_mode)
                picker.use_memories = target_mode == "enabled"
            except MemoryFeatureUnavailableError:
                picker.settings = await codex.read_memory_settings()
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("memories.label.feature_disabled"))
                await self._edit_message(
                    incoming.chat_id,
                    incoming.message_id,
                    self._memory_picker_text(picker),
                    self._memory_picker_keyboard(picker),
                )
                return
            except Exception as exc:
                self._log_failure(
                    "Could not change thread memory mode",
                    exc,
                    event="telegram.memories.thread_mode_failed",
                )
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("memories.label.save_failed"))
                return
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("memories.label.saved"))
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._memory_picker_text(picker),
                self._memory_picker_keyboard(picker),
            )
            return
        await self.telegram.answer_callback_query(incoming.callback_id, self._text("memories.label.stale"))


    async def _handle_language_callback(
        self,
        incoming: TelegramCallbackContext,
        parts: list[str],
    ) -> None:
        picker = self.language_picker
        if (
            picker is None
            or len(parts) != 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback_query(
                incoming.callback_id,
                self.texts.system("label.language_picker_stale"),
            )
            return
        language = parts[2]
        if language not in SUPPORTED_LANGUAGES:
            await self.telegram.answer_callback_query(
                incoming.callback_id,
                self.texts.system("label.language_picker_invalid"),
            )
            return

        self._codex().select_language(language)
        await self.telegram.answer_callback_query(
            incoming.callback_id,
            self.texts.system(
                "label.language_picker_saved",
                language=self.texts.system(f"values.language_names.{language}"),
            ),
        )
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self.texts.system("label.language_picker_prompt"),
            self._language_picker_keyboard(picker),
        )


    async def _handle_model_callback(
        self,
        incoming: TelegramCallbackContext,
        parts: list[str],
    ) -> None:
        picker = self.model_picker
        if (
            picker is None
            or len(parts) < 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
        ):
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("models.label.expired"))
            return

        action = parts[2]
        if picker.effort_model_index is not None:
            model = picker.models[picker.effort_model_index]
            if action == "b" and len(parts) == 3:
                picker.effort_model_index = None
                await self.telegram.answer_callback_query(incoming.callback_id)
                await self._edit_message(
                    incoming.chat_id,
                    incoming.message_id,
                    self._model_picker_text(picker),
                    self._model_picker_keyboard(picker),
                )
                return
            if action == "d" and len(parts) == 3:
                effort = model.default_reasoning_effort
            elif action == "e" and len(parts) == 4:
                try:
                    effort_index = int(parts[3])
                    efforts = self._model_effort_options(model)
                    if not 0 <= effort_index < len(efforts):
                        raise ValueError("Effort button is not valid")
                    effort = efforts[effort_index]
                except ValueError:
                    await self.telegram.answer_callback_query(incoming.callback_id, self._text("common.label.stale_button"))
                    return
            else:
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("common.label.stale_button"))
                return

            try:
                await self._codex().select_model(model.model, effort)
            except Exception as exc:
                self._log_failure(
                    "Could not select Codex model and effort",
                    exc,
                    event="telegram.model.select_failed",
                    model=model.model,
                    effort=effort,
                )
                await self.telegram.answer_callback_query(
                    incoming.callback_id,
                    self._text("models.label.selection_failed"),
                )
                return
            picker.current_model = model.model
            picker.current_effort = effort
            picker.effort_model_index = None
            await self.telegram.answer_callback_query(
                incoming.callback_id,
                self._text("models.label.selection_saved"),
            )
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._model_picker_text(picker),
                self._model_picker_keyboard(picker),
            )
            return

        if action == "i" and len(parts) == 3:
            await self.telegram.answer_callback_query(incoming.callback_id)
            return

        if action == "p" and len(parts) == 4:
            try:
                page = int(parts[3])
            except ValueError:
                page = -1
            page_count = self._model_page_count(picker)
            if page < 0 or page >= page_count:
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("models.label.invalid_page"))
                return
            picker.page = page
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._model_picker_text(picker),
                self._model_picker_keyboard(picker),
            )
            return

        if action == "r" and len(parts) == 3:
            try:
                await self._codex().select_model(None, None)
                picker.current_model, picker.current_effort = await self._codex().current_model_settings(
                    picker.models
                )
            except Exception as exc:
                self._log_failure(
                    "Could not reset Codex model and effort",
                    exc,
                    event="telegram.model.reset_failed",
                )
                await self.telegram.answer_callback_query(
                    incoming.callback_id,
                    self._text("models.label.selection_failed"),
                )
                return
            await self.telegram.answer_callback_query(
                incoming.callback_id,
                self._text("models.label.selection_saved"),
            )
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._model_picker_text(picker),
                self._model_picker_keyboard(picker),
            )
            return
        if action == "s" and len(parts) == 4:
            try:
                model_index = int(parts[3])
            except ValueError:
                model_index = -1
            if model_index < 0 or model_index >= len(picker.models):
                await self.telegram.answer_callback_query(incoming.callback_id, self._text("models.label.unknown"))
                return
            picker.effort_model_index = model_index
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_message(
                incoming.chat_id,
                incoming.message_id,
                self._model_picker_text(picker),
                self._model_picker_keyboard(picker),
            )
            return
        else:
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("common.label.stale_button"))
            return


    @staticmethod
    def _model_page_count(picker: ModelPicker) -> int:
        return max(1, (len(picker.models) + 7) // 8)


    def _model_picker_text(self, picker: ModelPicker) -> str:
        codex = self._codex()
        if picker.effort_model_index is not None:
            model = picker.models[picker.effort_model_index]
            default_effort = (
                self._effort_label(model.default_reasoning_effort)
                if model.default_reasoning_effort
                else self._text("models.values.codex_default_effort")
            )
            return self._text(
                "models.label.effort_prompt",
                model=model.display_name,
                default_effort=default_effort,
            )
        current_model = picker.current_model
        model_info = next((model for model in picker.models if model.model == current_model), None)
        current = model_info.display_name if model_info else current_model or self._text("common.values.default_codex_model")
        if picker.current_effort and (
            model_info is None
            or picker.current_effort != model_info.default_reasoning_effort
        ):
            current += self._text(
                "models.values.current_effort",
                effort=self._effort_label(picker.current_effort),
            )
        elif model_info and model_info.default_reasoning_effort:
            current += self._text(
                "models.values.current_effort_default",
                effort=self._effort_label(model_info.default_reasoning_effort),
            )
        elif picker.current_effort:
            current += self._text(
                "models.values.current_effort",
                effort=self._effort_label(picker.current_effort),
            )
        message = self._text("models.label.prompt", current=current)
        if codex.is_busy:
            message += self._text("models.values.busy_notice")
        return message


    def _model_picker_keyboard(self, picker: ModelPicker) -> dict[str, Any]:
        selected_model = picker.current_model
        if picker.effort_model_index is not None:
            model = picker.models[picker.effort_model_index]
            rows: list[list[dict[str, str]]] = []
            default_effort = model.default_reasoning_effort
            current_effort = picker.current_effort if selected_model == model.model else None
            if default_effort:
                default_label = self._text(
                    "models.buttons.default_effort",
                    effort=self._effort_label(default_effort),
                )
                default_label = self._selected_button_label(
                    default_label,
                    selected_model == model.model and current_effort in {None, default_effort},
                )
                rows.append(
                    [{"text": default_label, "callback_data": f"m:{picker.token}:d"}]
                )
            efforts = self._model_effort_options(model)
            effort_buttons = []
            for effort_index, effort in enumerate(efforts):
                label = self._selected_button_label(
                    self._effort_label(effort),
                    selected_model == model.model and current_effort == effort,
                )
                effort_buttons.append(
                    {"text": label, "callback_data": f"m:{picker.token}:e:{effort_index}"}
                )
            rows.extend(two_column_rows(effort_buttons, 0))
            rows.append([{"text": self._text("models.buttons.back"), "callback_data": f"m:{picker.token}:b"}])
            return {"inline_keyboard": rows}

        start = picker.page * 8
        end = min(start + 8, len(picker.models))
        model_buttons = []
        for model_index in range(start, end):
            model = picker.models[model_index]
            label = self._selected_button_label(
                model.display_name,
                model.model == selected_model,
            )
            model_buttons.append(
                {"text": label, "callback_data": f"m:{picker.token}:s:{model_index}"}
            )
        rows = two_column_rows(model_buttons, 0)

        rows.append([{"text": self._text("models.buttons.reset"), "callback_data": f"m:{picker.token}:r"}])
        page_count = self._model_page_count(picker)
        if page_count > 1:
            rows.append(
                navigation_row(
                    picker.page,
                    page_count,
                    self.texts.system("buttons.pager.previous"),
                    self.texts.system("buttons.pager.page", current="{current}", total="{total}"),
                    self.texts.system("buttons.pager.next"),
                    f"m:{picker.token}",
                )
            )
        return {"inline_keyboard": rows}


    @staticmethod
    def _model_effort_options(model: CodexModel) -> tuple[str, ...]:
        return tuple(
            effort
            for effort in model.supported_reasoning_efforts
            if effort != model.default_reasoning_effort
        )


    def _effort_label(self, effort: str) -> str:
        labels = {
            "none": "models.buttons.effort_names.none",
            "minimal": "models.buttons.effort_names.minimal",
            "low": "models.buttons.effort_names.low",
            "medium": "models.buttons.effort_names.medium",
            "high": "models.buttons.effort_names.high",
            "xhigh": "models.buttons.effort_names.xhigh",
            "max": "models.buttons.effort_names.max",
        }
        key = labels.get(effort)
        return self._text(key) if key else effort
