"""Operator controls for settings that seed future /new conversations."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace
from typing import Any, Callable

from aiogram import Bot

from telecodex.codex.permissions import PERMISSION_MODES
from telecodex.codex.types import CodexModel
from telecodex.i18n import TextCatalog
from telecodex.state import NewChatDefaults
from telecodex.telegram.pagination import PAGE_SIZE, navigation_row, page_count, two_column_rows
from telecodex.telegram.updates import TelegramCallbackContext


@dataclass
class _DefaultsPicker:
    token: str
    user_id: int
    chat_id: int
    defaults: NewChatDefaults
    message_id: int | None = None
    section: str = "menu"
    page: int = 0
    models: list[CodexModel] | None = None
    effort_model_index: int | None = None


class TelegramNewChatDefaultsController:
    """Edit one operator's defaults without changing the active Codex thread."""

    def __init__(
        self,
        *,
        telegram: Bot,
        codex: Callable[[], Any],
        text: Callable[..., str],
        texts: TextCatalog,
        load_defaults: Callable[[int], NewChatDefaults],
        save_defaults: Callable[[int, NewChatDefaults], None],
        selected_button_label: Callable[[str, bool], str],
        edit_message: Callable[..., Any],
        log_failure: Callable[..., None],
    ) -> None:
        self.telegram = telegram
        self._codex = codex
        self._text = text
        self.texts = texts
        self._load_defaults = load_defaults
        self._save_defaults = save_defaults
        self._selected_button_label = selected_button_label
        self._edit_message = edit_message
        self._log_failure = log_failure
        self.picker: _DefaultsPicker | None = None

    async def show(self, chat_id: int, user_id: int) -> None:
        self.picker = None
        try:
            defaults = self._load_defaults(user_id)
        except Exception as exc:
            self._log_failure(
                "Could not load new-chat defaults",
                exc,
                event="telegram.new_chat_defaults.load_failed",
            )
            await self.telegram.send_message(
                chat_id,
                self._text("new_chat_defaults.label.load_failed"),
            )
            return
        picker = _DefaultsPicker(
            token=secrets.token_urlsafe(6),
            user_id=user_id,
            chat_id=chat_id,
            defaults=defaults,
        )
        message = await self.telegram.send_message(
            chat_id,
            self._menu_text(defaults),
            reply_markup=self._view_keyboard(picker),
        )
        if isinstance(message.message_id, int):
            picker.message_id = message.message_id
            self.picker = picker

    async def handle_callback(
        self,
        incoming: TelegramCallbackContext,
        parts: list[str],
    ) -> None:
        picker = self.picker
        if (
            picker is None
            or len(parts) < 3
            or parts[1] != picker.token
            or incoming.message_id != picker.message_id
            or incoming.chat_id != picker.chat_id
            or incoming.user_id != picker.user_id
        ):
            await self.telegram.answer_callback_query(
                incoming.callback_id,
                self._text("new_chat_defaults.label.stale"),
            )
            return

        action = parts[2]
        if action == "back" and len(parts) == 3:
            picker.section = "menu"
            picker.models = None
            picker.effort_model_index = None
            await self._answer_and_show_menu(incoming, picker)
            return
        if action == "reset" and len(parts) == 3:
            await self._save_and_show_menu(incoming, picker, NewChatDefaults())
            return
        if action == "model" and len(parts) == 3:
            await self._open_models(incoming, picker)
            return
        if action == "model_page" and len(parts) == 5 and parts[3] == "p":
            await self._change_model_page(incoming, picker, parts[4])
            return
        if action == "model_page" and len(parts) == 4 and parts[3] == "i":
            await self.telegram.answer_callback_query(incoming.callback_id)
            return
        if action == "model_reset" and len(parts) == 3:
            await self._save_and_show_menu(
                incoming,
                picker,
                replace(picker.defaults, selected_model=None, selected_effort=None),
            )
            return
        if action == "model_select" and len(parts) == 4:
            await self._select_model(incoming, picker, parts[3])
            return
        if action == "effort_back" and len(parts) == 3:
            picker.section = "model"
            picker.effort_model_index = None
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_current_view(picker)
            return
        if action == "effort_default" and len(parts) == 3:
            model = self._effort_model(picker)
            if model is None:
                await self._stale(incoming)
                return
            await self._save_and_show_menu(
                incoming,
                picker,
                replace(
                    picker.defaults,
                    selected_model=model.model,
                    selected_effort=None,
                ),
            )
            return
        if action == "effort_select" and len(parts) == 4:
            await self._select_effort(incoming, picker, parts[3])
            return
        if action == "permissions" and len(parts) == 4:
            value = None if parts[3] == "default" else parts[3]
            if value is not None and value not in PERMISSION_MODES:
                await self._stale(incoming)
                return
            await self._save_and_show_menu(
                incoming,
                picker,
                replace(picker.defaults, default_permission_mode=value),
            )
            return
        if action == "mode" and len(parts) == 4:
            value = None if parts[3] == "default" else parts[3]
            if value is not None and value not in {"answers", "quiet_answers", "detailed"}:
                await self._stale(incoming)
                return
            await self._save_and_show_menu(
                incoming,
                picker,
                replace(picker.defaults, output_mode=value),
            )
            return
        if action == "memory" and len(parts) == 4:
            value = None if parts[3] == "default" else parts[3]
            if value is not None and value not in {"enabled", "disabled"}:
                await self._stale(incoming)
                return
            await self._save_and_show_menu(
                incoming,
                picker,
                replace(picker.defaults, selected_memory_mode=value),
            )
            return
        if action == "group_mode" and len(parts) == 4:
            value: bool | None
            if parts[3] == "default":
                value = None
            elif parts[3] == "all":
                value = True
            elif parts[3] == "triggered":
                value = False
            else:
                await self._stale(incoming)
                return
            await self._save_and_show_menu(
                incoming,
                picker,
                replace(picker.defaults, group_all_messages_to_codex=value),
            )
            return
        if action in {"permissions_menu", "mode_menu", "memory_menu", "group_mode_menu"} and len(parts) == 3:
            picker.section = action.removesuffix("_menu")
            await self.telegram.answer_callback_query(incoming.callback_id)
            await self._edit_current_view(picker)
            return

        await self._stale(incoming)

    async def _open_models(
        self,
        incoming: TelegramCallbackContext,
        picker: _DefaultsPicker,
    ) -> None:
        try:
            models = await self._codex().list_models()
        except Exception as exc:
            self._log_failure(
                "Could not list models while editing new-chat defaults",
                exc,
                event="telegram.new_chat_defaults.models_failed",
            )
            await self.telegram.answer_callback_query(
                incoming.callback_id,
                self._text("new_chat_defaults.label.models_failed"),
            )
            return
        picker.models = models
        picker.page = 0
        picker.effort_model_index = None
        picker.section = "model"
        await self.telegram.answer_callback_query(incoming.callback_id)
        await self._edit_current_view(picker)

    async def _change_model_page(
        self,
        incoming: TelegramCallbackContext,
        picker: _DefaultsPicker,
        raw_page: str,
    ) -> None:
        try:
            page = int(raw_page)
        except ValueError:
            page = -1
        models = picker.models or []
        if page < 0 or page >= page_count(len(models)):
            await self._stale(incoming)
            return
        picker.page = page
        await self.telegram.answer_callback_query(incoming.callback_id)
        await self._edit_current_view(picker)

    async def _select_model(
        self,
        incoming: TelegramCallbackContext,
        picker: _DefaultsPicker,
        raw_index: str,
    ) -> None:
        try:
            index = int(raw_index)
            model = (picker.models or [])[index]
            first_index = picker.page * PAGE_SIZE
            if not first_index <= index < min(first_index + PAGE_SIZE, len(picker.models or [])):
                raise ValueError("Model is not on the current page")
        except (IndexError, ValueError):
            await self._stale(incoming)
            return
        efforts = self._effort_options(model)
        if model.default_reasoning_effort is None and not efforts:
            await self._save_and_show_menu(
                incoming,
                picker,
                replace(picker.defaults, selected_model=model.model, selected_effort=None),
            )
            return
        picker.effort_model_index = index
        picker.section = "effort"
        await self.telegram.answer_callback_query(incoming.callback_id)
        await self._edit_current_view(picker)

    async def _select_effort(
        self,
        incoming: TelegramCallbackContext,
        picker: _DefaultsPicker,
        raw_index: str,
    ) -> None:
        model = self._effort_model(picker)
        try:
            index = int(raw_index)
            efforts = self._effort_options(model) if model is not None else ()
            if not 0 <= index < len(efforts):
                raise ValueError("Effort button is not valid")
            effort = efforts[index]
        except (IndexError, ValueError):
            await self._stale(incoming)
            return
        assert model is not None
        await self._save_and_show_menu(
            incoming,
            picker,
            replace(picker.defaults, selected_model=model.model, selected_effort=effort),
        )

    async def _save_and_show_menu(
        self,
        incoming: TelegramCallbackContext,
        picker: _DefaultsPicker,
        defaults: NewChatDefaults,
    ) -> None:
        try:
            self._save_defaults(picker.user_id, defaults)
        except Exception as exc:
            self._log_failure(
                "Could not save new-chat defaults",
                exc,
                event="telegram.new_chat_defaults.save_failed",
            )
            await self.telegram.answer_callback_query(
                incoming.callback_id,
                self._text("new_chat_defaults.label.save_failed"),
            )
            return
        picker.defaults = defaults
        picker.section = "menu"
        picker.models = None
        picker.effort_model_index = None
        await self.telegram.answer_callback_query(
            incoming.callback_id,
            self._text("new_chat_defaults.label.saved"),
        )
        await self._edit_current_view(picker)

    async def _answer_and_show_menu(
        self,
        incoming: TelegramCallbackContext,
        picker: _DefaultsPicker,
    ) -> None:
        await self.telegram.answer_callback_query(incoming.callback_id)
        await self._edit_current_view(picker)

    async def _stale(self, incoming: TelegramCallbackContext) -> None:
        await self.telegram.answer_callback_query(
            incoming.callback_id,
            self._text("new_chat_defaults.label.stale"),
        )

    async def _edit_current_view(self, picker: _DefaultsPicker) -> None:
        await self._edit_message(
            picker.chat_id,
            picker.message_id,
            self._view_text(picker),
            self._view_keyboard(picker),
        )

    def _view_text(self, picker: _DefaultsPicker) -> str:
        if picker.section == "menu":
            return self._menu_text(picker.defaults)
        if picker.section == "model":
            models = picker.models or []
            if not models:
                return self._text("new_chat_defaults.label.models_empty")
            return self._text(
                "new_chat_defaults.label.model_prompt",
                current=self._model_summary(picker.defaults),
            )
        if picker.section == "effort":
            model = self._effort_model(picker)
            if model is None:
                return self._text("new_chat_defaults.label.stale")
            default_effort = (
                self._effort_label(model.default_reasoning_effort)
                if model.default_reasoning_effort
                else self._text("new_chat_defaults.values.codex_default")
            )
            return self._text(
                "new_chat_defaults.label.effort_prompt",
                model=model.display_name,
                default_effort=default_effort,
            )
        if picker.section == "permissions":
            return self._text(
                "new_chat_defaults.label.permissions_prompt",
                current=self._permission_summary(picker.defaults),
            )
        if picker.section == "mode":
            return self._text(
                "new_chat_defaults.label.mode_prompt",
                current=self._mode_summary(picker.defaults),
            )
        if picker.section == "memory":
            return self._text(
                "new_chat_defaults.label.memory_prompt",
                current=self._memory_summary(picker.defaults),
            )
        if picker.section == "group_mode":
            return self._text(
                "new_chat_defaults.label.group_mode_prompt",
                current=self._group_mode_summary(picker.defaults),
            )
        return self._text("new_chat_defaults.label.stale")

    def _view_keyboard(self, picker: _DefaultsPicker) -> dict[str, Any]:
        token = picker.token
        if picker.section == "menu":
            return {
                "inline_keyboard": [
                    [{"text": self._text("new_chat_defaults.buttons.model"), "callback_data": f"nd:{token}:model"}],
                    [{"text": self._text("new_chat_defaults.buttons.permissions"), "callback_data": f"nd:{token}:permissions_menu"}],
                    [{"text": self._text("new_chat_defaults.buttons.mode"), "callback_data": f"nd:{token}:mode_menu"}],
                    [{"text": self._text("new_chat_defaults.buttons.memory"), "callback_data": f"nd:{token}:memory_menu"}],
                    [{"text": self._text("new_chat_defaults.buttons.group_mode"), "callback_data": f"nd:{token}:group_mode_menu"}],
                    [{"text": self._text("new_chat_defaults.buttons.reset"), "callback_data": f"nd:{token}:reset"}],
                ]
            }
        if picker.section == "model":
            models = picker.models or []
            buttons = [
                {
                    "text": self._selected_button_label(
                        model.display_name,
                        model.model == picker.defaults.selected_model,
                    ),
                    "callback_data": f"nd:{token}:model_select:{index}",
                }
                for index, model in enumerate(models)
            ]
            rows = two_column_rows(buttons, picker.page)
            rows.append(
                [{
                    "text": self._text("new_chat_defaults.buttons.use_codex_model"),
                    "callback_data": f"nd:{token}:model_reset",
                }]
            )
            count = page_count(len(models))
            if count > 1:
                rows.append(
                    navigation_row(
                        picker.page,
                        count,
                        self.texts.system("buttons.pager.previous"),
                        self.texts.system("buttons.pager.page", current="{current}", total="{total}"),
                        self.texts.system("buttons.pager.next"),
                        f"nd:{token}:model_page",
                    )
                )
            rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"nd:{token}:back"}])
            return {"inline_keyboard": rows}
        if picker.section == "effort":
            model = self._effort_model(picker)
            if model is None:
                return {"inline_keyboard": [[{"text": self._text("common.buttons.back"), "callback_data": f"nd:{token}:back"}]]}
            current_effort = (
                picker.defaults.selected_effort
                if picker.defaults.selected_model == model.model
                else None
            )
            default_label = self._text(
                "new_chat_defaults.buttons.default_effort",
                effort=(
                    self._effort_label(model.default_reasoning_effort)
                    if model.default_reasoning_effort
                    else self._text("new_chat_defaults.values.codex_default")
                ),
            )
            rows = [[{
                "text": self._selected_button_label(
                    default_label,
                    picker.defaults.selected_model == model.model and current_effort is None,
                ),
                "callback_data": f"nd:{token}:effort_default",
            }]]
            effort_buttons = [
                {
                    "text": self._selected_button_label(
                        self._effort_label(effort),
                        picker.defaults.selected_model == model.model and current_effort == effort,
                    ),
                    "callback_data": f"nd:{token}:effort_select:{index}",
                }
                for index, effort in enumerate(self._effort_options(model))
            ]
            rows.extend(two_column_rows(effort_buttons, 0))
            rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"nd:{token}:effort_back"}])
            return {"inline_keyboard": rows}
        if picker.section == "permissions":
            rows = [
                [{
                    "text": self._selected_button_label(
                        self._text(f"permissions.buttons.mode_names.{mode}"),
                        picker.defaults.default_permission_mode == mode,
                    ),
                    "callback_data": f"nd:{token}:permissions:{mode}",
                }]
                for mode in PERMISSION_MODES
            ]
            rows.append([{
                "text": self._selected_button_label(
                    self._text("new_chat_defaults.buttons.use_codex_permissions"),
                    picker.defaults.default_permission_mode is None,
                ),
                "callback_data": f"nd:{token}:permissions:default",
            }])
            rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"nd:{token}:back"}])
            return {"inline_keyboard": rows}
        if picker.section == "mode":
            rows = [
                [{
                    "text": self._selected_button_label(
                        self._text(f"mode.buttons.names.{mode}"),
                        picker.defaults.output_mode == mode,
                    ),
                    "callback_data": f"nd:{token}:mode:{mode}",
                }]
                for mode in ("answers", "quiet_answers", "detailed")
            ]
            rows.append([{
                "text": self._selected_button_label(
                    self._text("new_chat_defaults.buttons.use_chat_mode"),
                    picker.defaults.output_mode is None,
                ),
                "callback_data": f"nd:{token}:mode:default",
            }])
            rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"nd:{token}:back"}])
            return {"inline_keyboard": rows}
        if picker.section == "memory":
            rows = []
            for mode, key in (("enabled", "enable_use_button"), ("disabled", "disable_use_button")):
                rows.append([{
                    "text": self._selected_button_label(
                        self._text(f"memories.buttons.{key}"),
                        picker.defaults.selected_memory_mode == mode,
                    ),
                    "callback_data": f"nd:{token}:memory:{mode}",
                }])
            rows.append([{
                "text": self._selected_button_label(
                    self._text("new_chat_defaults.buttons.use_codex_memories"),
                    picker.defaults.selected_memory_mode is None,
                ),
                "callback_data": f"nd:{token}:memory:default",
            }])
            rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"nd:{token}:back"}])
            return {"inline_keyboard": rows}
        if picker.section == "group_mode":
            options = (
                ("triggered", False, "reply_or_trigger"),
                ("all", True, "all_messages"),
            )
            rows = [
                [{
                    "text": self._selected_button_label(
                        self._text(f"groups.buttons.mode_names.{name}"),
                        picker.defaults.group_all_messages_to_codex == value,
                    ),
                    "callback_data": f"nd:{token}:group_mode:{mode}",
                }]
                for mode, value, name in options
            ]
            rows.append([{
                "text": self._selected_button_label(
                    self._text("new_chat_defaults.buttons.use_group_default"),
                    picker.defaults.group_all_messages_to_codex is None,
                ),
                "callback_data": f"nd:{token}:group_mode:default",
            }])
            rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"nd:{token}:back"}])
            return {"inline_keyboard": rows}
        return {"inline_keyboard": [[{"text": self._text("common.buttons.back"), "callback_data": f"nd:{token}:back"}]]}

    def _menu_text(self, defaults: NewChatDefaults) -> str:
        return self._text(
            "new_chat_defaults.label.menu",
            model=self._model_summary(defaults),
            permissions=self._permission_summary(defaults),
            mode=self._mode_summary(defaults),
            memory=self._memory_summary(defaults),
            group_mode=self._group_mode_summary(defaults),
        )

    def _model_summary(self, defaults: NewChatDefaults) -> str:
        if defaults.selected_model is None:
            return self._text("new_chat_defaults.values.codex_default")
        summary = defaults.selected_model
        if defaults.selected_effort:
            summary += f" · {self._effort_label(defaults.selected_effort)}"
        else:
            summary += f" · {self._text('new_chat_defaults.values.model_default_effort')}"
        return summary

    def _permission_summary(self, defaults: NewChatDefaults) -> str:
        if defaults.default_permission_mode is None:
            return self._text("new_chat_defaults.values.codex_default")
        return self._text(f"permissions.buttons.mode_names.{defaults.default_permission_mode}")

    def _mode_summary(self, defaults: NewChatDefaults) -> str:
        if defaults.output_mode is None:
            return self._text("new_chat_defaults.values.chat_default_mode")
        return self._text(f"mode.buttons.names.{defaults.output_mode}")

    def _memory_summary(self, defaults: NewChatDefaults) -> str:
        if defaults.selected_memory_mode is None:
            return self._text("new_chat_defaults.values.codex_default")
        key = "enabled" if defaults.selected_memory_mode == "enabled" else "disabled"
        return self._text(f"memories.values.{key}")

    def _group_mode_summary(self, defaults: NewChatDefaults) -> str:
        key = "all_messages" if defaults.group_all_messages_to_codex else "reply_or_trigger"
        return self._text(f"groups.buttons.mode_names.{key}")

    def _effort_model(self, picker: _DefaultsPicker) -> CodexModel | None:
        index = picker.effort_model_index
        models = picker.models or []
        if index is None or not 0 <= index < len(models):
            return None
        return models[index]

    @staticmethod
    def _effort_options(model: CodexModel | None) -> tuple[str, ...]:
        if model is None:
            return ()
        return tuple(
            effort
            for effort in model.supported_reasoning_efforts
            if effort != model.default_reasoning_effort
        )

    def _effort_label(self, effort: str) -> str:
        key = {
            "none": "none",
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": "max",
        }.get(effort)
        return self._text(f"models.buttons.effort_names.{key}") if key else effort
