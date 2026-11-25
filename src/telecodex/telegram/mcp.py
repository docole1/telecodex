from __future__ import annotations

import json
import logging
import secrets
from pathlib import Path
from typing import Any

from telecodex.codex.mcp import (
    McpServer,
    validate_env_name,
    validate_http_url,
    validate_requested_env_names,
    validate_server_name,
)
from telecodex.telegram.formatting import truncate
from telecodex.telegram.mcp_elicitation import (
    McpElicitationController,
    PendingMcpElicitation,
)
from telecodex.telegram.pagination import (
    INLINE_BUTTON_TEXT_LIMIT,
    PAGE_SIZE,
    keyboard,
    navigation_row,
    page_count,
    two_column_rows,
)
from telecodex.telegram.pickers import McpAddWizard, McpPicker
from telecodex.telegram.updates import IncomingCallback, IncomingMessage, parse_command

logger = logging.getLogger(__name__)


class McpTelegramMixin:
    """MCP controls and request prompts for a Telegram context controller."""


    async def _show_mcp_picker(
        self,
        chat_id: int,
        argument: str = "",
        owner_user_id: int | None = None,
    ) -> None:
        normalized_argument = argument.strip().casefold()
        argument_parts = argument.strip().split(maxsplit=1)
        if argument_parts and argument_parts[0].casefold() == "login":
            if len(argument_parts) != 2:
                await self.telegram.send_message(chat_id, self._text("mcp.label.usage"))
                return
            server_name = argument_parts[1]
            try:
                server_name = validate_server_name(server_name)
            except ValueError:
                await self.telegram.send_message(chat_id, self._text("mcp.label.usage"))
                return
            await self._start_mcp_oauth_message(chat_id, server_name)
            return
        if normalized_argument not in {"", "verbose", "add"}:
            await self.telegram.send_message(chat_id, self._text("mcp.label.usage"))
            return
        try:
            servers = await self._codex().list_mcp_servers()
        except Exception as exc:
            self._log_mcp_failure(
                "Could not list Codex MCP servers",
                exc,
                event="telegram.mcp.list_failed",
            )
            await self.telegram.send_message(chat_id, self._text("mcp.label.list_failed"))
            return
        picker = McpPicker(
            token=secrets.token_urlsafe(6),
            servers=servers,
            verbose=normalized_argument == "verbose",
        )
        self.mcp_add_wizard = None
        message = await self.telegram.send_message(
            chat_id,
            self._mcp_picker_text(picker),
            self._mcp_picker_keyboard(picker),
        )
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            picker.message_id = message_id
            self.mcp_picker = picker
            if normalized_argument == "add":
                await self._start_mcp_add_wizard(chat_id, picker, owner_user_id)

    async def _start_mcp_oauth_message(
        self,
        chat_id: int,
        server_name: str,
        *,
        picker: McpPicker | None = None,
    ) -> None:
        try:
            remote_callback = await self._codex().mcp_oauth_uses_remote_callback(server_name)
            if not remote_callback:
                message = self._text(
                    "mcp.label.oauth_local_callback",
                    command=f"codex mcp login {server_name} --no-browser",
                )
                rows = []
                if picker is not None:
                    rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"mc:{picker.token}:back"}])
                markup = keyboard(rows) if rows else None
                if picker is not None and picker.message_id is not None:
                    await self._edit_message(self.chat_id, picker.message_id, message, markup)
                else:
                    await self.telegram.send_message(chat_id, message)
                return
            authorization_url, _ = await self._codex().start_mcp_oauth(server_name)
        except Exception as exc:
            self._log_mcp_failure("Could not start MCP OAuth", exc, event="telegram.mcp.oauth_start_failed")
            await self.telegram.send_message(chat_id, self._text("mcp.label.oauth_start_failed"))
            return
        message = self._text("mcp.label.oauth_open_url", name=server_name)
        rows = [[{"text": self._text("mcp.buttons.oauth_open_button"), "url": authorization_url}]]
        if picker is not None:
            rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"mc:{picker.token}:back"}])
        markup = keyboard(rows)
        if picker is not None and picker.message_id is not None:
            await self._edit_message(self.chat_id, picker.message_id, message, markup)
        else:
            await self.telegram.send_message(chat_id, message, markup)

    def _mcp_picker_text(self, picker: McpPicker) -> str:
        wizard = self.mcp_add_wizard
        if wizard is not None and picker.view == "wizard":
            return self._mcp_wizard_prompt(wizard)
        if picker.confirming_remove:
            server = self._selected_mcp_server(picker)
            return self._text(
                "mcp.label.remove_confirmation",
                name=server.name if server else "",
            )
        if picker.confirming_action:
            server = self._selected_mcp_server(picker)
            return self._text(
                f"mcp.label.confirm_{picker.confirming_action}",
                name=server.name if server else "",
            )
        if picker.view == "detail":
            server = self._selected_mcp_server(picker)
            if server is None:
                return self._text("mcp.label.stale")
            runtime = self._text(
                f"mcp.values.runtime.{server.runtime_status}"
                if server.runtime_status in {
                    "notStarted", "starting", "connected", "authenticationRequired",
                    "failed", "cancelled", "disabled",
                }
                else "common.values.no_data"
            )
            if server.auth_status == "oAuth":
                auth_key = "common.values.oauth_name"
            elif server.auth_status in {
                "unknown", "unsupported", "notLoggedIn", "bearerToken",
            }:
                auth_key = f"mcp.values.auth.{server.auth_status}"
            else:
                auth_key = "mcp.values.auth.unknown"
            auth = self._text(auth_key)
            transport = self._text("mcp.values.transport.http" if server.http_origin else "mcp.values.transport.stdio")
            lines = [
                self._text("mcp.values.detail_title", name=server.name),
                self._text("mcp.values.detail_runtime", status=runtime),
                self._text("mcp.values.detail_auth", status=auth),
                self._text("mcp.values.detail_transport", transport=transport),
                self._text("mcp.values.detail_tools", count=len(server.tools)),
            ]
            if server.http_origin:
                lines.append(self._text("mcp.values.detail_origin", origin=server.http_origin))
            if server.server_name:
                version = f" {server.server_version}" if server.server_version else ""
                lines.append(self._text("mcp.values.detail_server_info", info=server.server_name + version))
            if server.tools_error:
                lines.append(self._text("mcp.label.tools_unavailable"))
            if server.is_plugin:
                lines.append(self._text("mcp.values.plugin_managed", plugin=server.plugin_id or ""))
            return "\n".join(lines)
        if picker.view == "tools":
            server = self._selected_mcp_server(picker)
            if server is None:
                return self._text("mcp.label.stale")
            lines = [
                self._text(
                    "mcp.label.tools_title",
                    name=server.name,
                    current=picker.tools_page + 1,
                    total=page_count(len(server.tools)),
                )
            ]
            start = picker.tools_page * PAGE_SIZE
            for tool in server.tools[start : start + PAGE_SIZE]:
                description = " ".join((tool.description or "").split())
                if len(description) > 180:
                    description = truncate(description, 180, self.texts.system("values.ellipsis"))
                marker = self._text("mcp.values.tool_read_only") if tool.read_only else ""
                suffix = f" — {description}" if description else ""
                lines.append(f"• {tool.name}{marker}{suffix}")
            if not server.tools:
                lines.append(self._text("mcp.label.tools_empty"))
            return self._truncate("\n".join(lines), 3800)

        count = page_count(len(picker.servers))
        text = self._text("mcp.label.list_title", current=picker.page + 1, total=count)
        if not picker.servers:
            return f"{text}\n{self._text('mcp.label.empty')}"
        if picker.verbose:
            details: list[str] = []
            start = picker.page * PAGE_SIZE
            for server in picker.servers[start : start + PAGE_SIZE]:
                runtime = self._text(
                    f"mcp.values.runtime.{server.runtime_status}"
                    if server.runtime_status in {
                        "notStarted", "starting", "connected", "authenticationRequired",
                        "failed", "cancelled", "disabled",
                    }
                    else "common.values.no_data"
                )
                if server.auth_status == "oAuth":
                    auth_key = "common.values.oauth_name"
                elif server.auth_status in {
                    "unknown", "unsupported", "notLoggedIn", "bearerToken",
                }:
                    auth_key = f"mcp.values.auth.{server.auth_status}"
                else:
                    auth_key = "mcp.values.auth.unknown"
                auth = self._text(auth_key)
                details.append(
                    self._text(
                        "mcp.values.verbose_entry",
                        name=server.name,
                        runtime=runtime,
                        auth=auth,
                        tools=len(server.tools),
                        origin=server.http_origin or self._text("mcp.values.transport.stdio"),
                    )
                )
                if server.tools:
                    details.append(self._text("mcp.values.verbose_tools", tools=", ".join(t.name for t in server.tools[:20])))
                if server.tools_error:
                    details.append(self._text("mcp.label.tools_unavailable"))
            text += "\n\n" + "\n".join(details)
        return self._truncate(text, 3800)

    def _mcp_picker_keyboard(self, picker: McpPicker) -> dict[str, Any]:
        wizard = self.mcp_add_wizard
        if wizard is not None and picker.view == "wizard":
            return self._mcp_wizard_keyboard(picker, wizard)
        if picker.confirming_remove:
            return keyboard([[
                {"text": self._text("mcp.buttons.confirm_remove_button"), "callback_data": f"mc:{picker.token}:remove_confirm"},
                {"text": self._text("common.buttons.cancel"), "callback_data": f"mc:{picker.token}:cancel"},
            ]])
        if picker.confirming_action:
            return keyboard([[
                {"text": self._text("common.buttons.confirm"), "callback_data": f"mc:{picker.token}:action_confirm"},
                {"text": self._text("common.buttons.cancel"), "callback_data": f"mc:{picker.token}:cancel"},
            ]])
        if picker.view == "detail":
            server = self._selected_mcp_server(picker)
            if server is None:
                return keyboard([[{"text": self._text("common.buttons.back"), "callback_data": f"mc:{picker.token}:back"}]])
            rows: list[list[dict[str, str]]] = [
                [{"text": self._text("mcp.buttons.tools_button"), "callback_data": f"mc:{picker.token}:tools"}],
            ]
            if server.auth_status == "notLoggedIn" or server.runtime_status == "authenticationRequired":
                rows.append([{"text": self._text("mcp.buttons.login_button"), "callback_data": f"mc:{picker.token}:login"}])
            if server.auth_status == "oAuth":
                rows.append([{"text": self._text("mcp.buttons.logout_button"), "callback_data": f"mc:{picker.token}:logout"}])
            if not server.is_plugin:
                action = "enable" if server.runtime_status == "disabled" else "disable"
                rows.append([{
                    "text": self._text(f"mcp.buttons.{action}_button"),
                    "callback_data": f"mc:{picker.token}:{action}",
                }])
                rows.append([{"text": self._text("mcp.buttons.remove_button"), "callback_data": f"mc:{picker.token}:remove"}])
            rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"mc:{picker.token}:back"}])
            return keyboard(rows)
        if picker.view == "tools":
            server = self._selected_mcp_server(picker)
            total = page_count(len(server.tools)) if server else 1
            rows: list[list[dict[str, str]]] = []
            if total > 1:
                rows.append(navigation_row(
                    picker.tools_page,
                    total,
                    self.texts.system("buttons.pager.previous"),
                    self.texts.system("buttons.pager.page", current="{current}", total="{total}"),
                    self.texts.system("buttons.pager.next"),
                    f"mc:{picker.token}",
                ))
            rows.append([{"text": self._text("common.buttons.back"), "callback_data": f"mc:{picker.token}:back"}])
            return keyboard(rows)

        start = picker.page * PAGE_SIZE
        buttons: list[dict[str, str]] = []
        for index, server in enumerate(picker.servers[start : start + PAGE_SIZE], start):
            runtime = server.runtime_status or "?"
            buttons.append({
                "text": truncate(
                    f"{server.name} · {runtime} · {len(server.tools)}",
                    INLINE_BUTTON_TEXT_LIMIT,
                    self.texts.system("values.ellipsis"),
                ),
                "callback_data": f"mc:{picker.token}:server:{index}",
            })
        rows = two_column_rows(buttons, 0)
        total = page_count(len(picker.servers))
        if total > 1:
            rows.append(navigation_row(
                picker.page,
                total,
                self.texts.system("buttons.pager.previous"),
                self.texts.system("buttons.pager.page", current="{current}", total="{total}"),
                self.texts.system("buttons.pager.next"),
                f"mc:{picker.token}",
            ))
        rows.extend([
            [{"text": self._text("mcp.buttons.add_button"), "callback_data": f"mc:{picker.token}:add"}],
            [{"text": self._text("mcp.buttons.refresh_button"), "callback_data": f"mc:{picker.token}:refresh"}],
        ])
        return keyboard(rows)

    @staticmethod

    def _selected_mcp_server(picker: McpPicker) -> McpServer | None:
        return next((server for server in picker.servers if server.name == picker.selected_name), None)

    async def _render_mcp_picker(self, picker: McpPicker) -> None:
        if picker.message_id is None:
            return
        await self._edit_message(
            self.chat_id,
            picker.message_id,
            self._mcp_picker_text(picker),
            self._mcp_picker_keyboard(picker),
        )

    async def _refresh_mcp_picker(self, picker: McpPicker) -> None:
        try:
            picker.servers = await self._codex().list_mcp_servers()
            picker.page = min(picker.page, page_count(len(picker.servers)) - 1)
            selected = self._selected_mcp_server(picker)
            if picker.selected_name is not None and selected is None:
                picker.selected_name = None
                picker.view = "servers"
            elif selected is not None:
                picker.tools_page = min(picker.tools_page, page_count(len(selected.tools)) - 1)
        except Exception as exc:
            self._log_mcp_failure("Could not refresh MCP server list", exc, event="telegram.mcp.refresh_failed")
            await self.telegram.send_message(self.chat_id, self._text("mcp.label.list_failed"))
            return
        await self._render_mcp_picker(picker)

    async def _start_mcp_add_wizard(
        self,
        chat_id: int,
        picker: McpPicker,
        owner_user_id: int | None = None,
    ) -> None:
        picker.view = "wizard"
        self.mcp_add_wizard = McpAddWizard(
            owner_user_id=owner_user_id if owner_user_id is not None else self.user_id,
            stage="transport",
        )
        self.mcp_picker = picker
        if picker.message_id is None:
            message = await self.telegram.send_message(
                chat_id,
                self._mcp_picker_text(picker),
                self._mcp_picker_keyboard(picker),
            )
            message_id = message.get("message_id")
            if not isinstance(message_id, int):
                self.mcp_add_wizard = None
                return
            picker.message_id = message_id
        await self._render_mcp_picker(picker)

    def _mcp_wizard_prompt(self, wizard: McpAddWizard) -> str:
        prompts = {
            "transport": "mcp.label.choose_transport",
            "name": "mcp.label.ask_name",
            "command": "mcp.label.ask_command",
            "args": "mcp.label.ask_args",
            "cwd": "mcp.label.ask_cwd",
            "env_vars": "mcp.label.ask_env_vars",
            "url": "mcp.label.ask_url",
            "http_auth": "mcp.label.choose_http_auth",
            "bearer_env": "mcp.label.ask_bearer_env",
        }
        if wizard.stage == "confirm":
            return self._mcp_add_summary(wizard)
        key = prompts.get(wizard.stage, "mcp.label.wizard_error")
        return self._text(key)

    def _mcp_wizard_keyboard(self, picker: McpPicker, wizard: McpAddWizard) -> dict[str, Any]:
        if wizard.stage == "transport":
            rows = two_column_rows(
                [
                    {"text": self._text("mcp.buttons.stdio_button"), "callback_data": f"mc:{picker.token}:stdio"},
                    {"text": self._text("mcp.buttons.http_button"), "callback_data": f"mc:{picker.token}:http"},
                ],
                0,
            )
        elif wizard.stage == "http_auth":
            rows = [
                [{"text": self._text("mcp.buttons.auth_none_button"), "callback_data": f"mc:{picker.token}:auth:none"}],
                [{"text": self._text("mcp.buttons.auth_oauth_button"), "callback_data": f"mc:{picker.token}:auth:oauth"}],
                [{"text": self._text("mcp.buttons.auth_bearer_button"), "callback_data": f"mc:{picker.token}:auth:bearer"}],
            ]
        elif wizard.stage == "confirm":
            rows = [[
                {"text": self._text("mcp.buttons.confirm_add_button"), "callback_data": f"mc:{picker.token}:add_confirm"},
                {"text": self._text("common.buttons.cancel"), "callback_data": f"mc:{picker.token}:cancel"},
            ]]
        else:
            rows = []
        rows.append([{"text": self._text("common.buttons.cancel"), "callback_data": f"mc:{picker.token}:cancel"}])
        return keyboard(rows)

    def _mcp_add_summary(self, wizard: McpAddWizard) -> str:
        assert wizard.name
        if wizard.transport == "stdio":
            import json

            command = json.dumps([wizard.command, *wizard.args], ensure_ascii=False)
            detail = self._text(
                "mcp.label.summary_stdio",
                command=command,
                cwd=wizard.cwd or self._text("mcp.values.default_cwd"),
                env_vars=", ".join(wizard.env_vars) or self._text("mcp.values.none"),
            )
        else:
            auth = self._text("common.values.oauth_name" if wizard.auth == "oauth" else f"mcp.values.auth_mode.{wizard.auth}")
            token_name = wizard.bearer_token_env_var or self._text("mcp.values.none")
            detail = self._text(
                "mcp.label.summary_http",
                url=wizard.url or "",
                auth=auth,
                env_var=token_name,
            )
        return self._text("mcp.label.confirm_add", name=wizard.name, detail=detail)

    async def _consume_mcp_dialog_input(self, incoming: IncomingMessage) -> bool:
        parsed = parse_command(incoming.text) if not incoming.attachments else None
        if parsed is not None and parsed[0] in {"stop", "status"}:
            return False
        wizard = self.mcp_add_wizard
        if wizard is not None and wizard.owner_user_id == incoming.user_id and wizard.stage not in {"transport", "http_auth", "confirm"}:
            await self._handle_mcp_wizard_input(incoming, wizard)
            return True
        active_elicitation = next(
            (
                pending
                for pending in self.pending_mcp_elicitations.values()
                if not pending.future.done()
            ),
            None,
        )
        if active_elicitation is not None:
            if active_elicitation.mode == "form":
                await self._handle_mcp_form_input(incoming, active_elicitation)
            else:
                await self.telegram.send_message(
                    self.chat_id,
                    self._text("mcp.label.elicitation_use_buttons"),
                )
            return True
        return False

    async def _handle_mcp_wizard_input(self, incoming: IncomingMessage, wizard: McpAddWizard) -> None:
        value = incoming.text.strip()
        if value.casefold() in {"/cancel", "cancel", "отмена"}:
            await self._cancel_mcp_wizard()
            return
        try:
            if wizard.stage == "name":
                wizard.name = validate_server_name(value)
                wizard.stage = "command" if wizard.transport == "stdio" else "url"
            elif wizard.stage == "command":
                if not value or len(value) > 1024 or "\x00" in value:
                    raise ValueError("Executable path is empty or invalid")
                wizard.command = value
                wizard.stage = "args"
            elif wizard.stage == "args":
                parsed = json.loads(value)
                if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
                    raise ValueError("Arguments must be a JSON array of strings")
                assert wizard.command is not None
                from telecodex.codex.mcp import stdio_config

                stdio_config(wizard.command, tuple(parsed), None, ())
                wizard.args = tuple(parsed)
                wizard.stage = "cwd"
            elif wizard.stage == "cwd":
                if value and value != "-":
                    if len(value) > 800:
                        raise ValueError("Working directory exceeds the size limit")
                    path = Path(value).expanduser()
                    if not path.is_absolute() or not path.is_dir():
                        raise ValueError("Working directory must be an existing absolute directory")
                    wizard.cwd = str(path.resolve())
                wizard.stage = "env_vars"
            elif wizard.stage == "env_vars":
                names = tuple(part.strip() for part in value.split(",") if part.strip() and part.strip() != "-")
                wizard.env_vars = validate_requested_env_names(names, self.config.mcp_env_vars)
                wizard.stage = "confirm"
            elif wizard.stage == "url":
                wizard.url = validate_http_url(value)
                wizard.stage = "http_auth"
            elif wizard.stage == "bearer_env":
                name = validate_env_name(value)
                wizard.bearer_token_env_var = validate_requested_env_names((name,), self.config.mcp_env_vars)[0]
                wizard.stage = "confirm"
            else:
                raise ValueError("MCP setup is not waiting for text input")
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            logger.info(
                "MCP setup input was rejected",
                extra={
                    "event": "telegram.mcp.wizard_input_rejected",
                    "input_stage": wizard.stage,
                    "error_type": type(exc).__name__,
                    **self._log_context_fields(),
                },
            )
            if self.mcp_picker is not None:
                await self._edit_message(
                    self.chat_id,
                    self.mcp_picker.message_id or 0,
                    f"{self._mcp_wizard_prompt(wizard)}\n\n{self._text('mcp.label.input_invalid')}",
                    self._mcp_picker_keyboard(self.mcp_picker),
                )
            return
        if self.mcp_picker is not None:
            await self._render_mcp_picker(self.mcp_picker)

    async def _cancel_mcp_wizard(self) -> None:
        self.mcp_add_wizard = None
        picker = self.mcp_picker
        if picker is not None:
            picker.view = "servers"
            picker.confirming_remove = False
            await self._refresh_mcp_picker(picker)

    async def _handle_mcp_callback(self, incoming: IncomingCallback, parts: list[str]) -> None:
        picker = self.mcp_picker
        if picker is None or len(parts) < 3 or parts[1] != picker.token or incoming.message_id != picker.message_id:
            await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.stale"))
            return
        action = parts[2]
        wizard = self.mcp_add_wizard
        if wizard is not None and wizard.owner_user_id != incoming.user_id:
            await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.owner_only"))
            return
        if wizard is not None and action in {"stdio", "http"} and wizard.stage == "transport":
            wizard.transport = action
            wizard.stage = "name"
            await self.telegram.answer_callback(incoming.callback_id)
            await self._render_mcp_picker(picker)
            return
        if wizard is not None and action == "auth" and len(parts) == 4 and wizard.stage == "http_auth":
            auth = parts[3]
            if auth not in {"none", "oauth", "bearer"}:
                await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.stale"))
                return
            wizard.auth = auth
            wizard.stage = "bearer_env" if auth == "bearer" else "confirm"
            await self.telegram.answer_callback(incoming.callback_id)
            await self._render_mcp_picker(picker)
            return
        if action == "cancel":
            await self.telegram.answer_callback(incoming.callback_id)
            if wizard is not None:
                await self._cancel_mcp_wizard()
            else:
                picker.confirming_remove = False
                picker.confirming_action = None
                picker.view = "detail" if picker.selected_name else "servers"
                await self._render_mcp_picker(picker)
            return
        if wizard is not None and action == "add_confirm" and wizard.stage == "confirm":
            await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.saving"))
            await self._save_mcp_wizard(wizard, picker)
            return
        if wizard is not None:
            await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.stale"))
            return
        if action == "p" and len(parts) == 4 and picker.view in {"servers", "tools"}:
            try:
                page = int(parts[3])
            except ValueError:
                page = -1
            total_pages = (
                page_count(len(picker.servers))
                if picker.view == "servers"
                else page_count(len(self._selected_mcp_server(picker).tools))
                if self._selected_mcp_server(picker) is not None
                else 1
            )
            if not 0 <= page < total_pages:
                await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.stale"))
                return
            if picker.view == "servers":
                picker.page = page
            else:
                picker.tools_page = page
            await self.telegram.answer_callback(incoming.callback_id)
            await self._render_mcp_picker(picker)
            return
        if action == "server" and len(parts) == 4 and picker.view == "servers":
            try:
                index = int(parts[3])
                if not picker.page * PAGE_SIZE <= index < min((picker.page + 1) * PAGE_SIZE, len(picker.servers)):
                    raise ValueError
                server = picker.servers[index]
            except (ValueError, IndexError):
                await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.stale"))
                return
            picker.selected_name = server.name
            picker.view = "detail"
            await self.telegram.answer_callback(incoming.callback_id)
            await self._render_mcp_picker(picker)
            return
        if action == "refresh" and picker.view == "servers":
            await self.telegram.answer_callback(incoming.callback_id)
            await self._refresh_mcp_picker(picker)
            return
        if action == "add" and picker.view == "servers":
            await self.telegram.answer_callback(incoming.callback_id)
            wizard = McpAddWizard(
                owner_user_id=incoming.user_id,
                stage="transport",
            )
            self.mcp_add_wizard = wizard
            picker.view = "wizard"
            await self._render_mcp_picker(picker)
            return
        if action == "tools" and picker.view == "detail":
            picker.view = "tools"
            picker.tools_page = 0
            await self.telegram.answer_callback(incoming.callback_id)
            await self._render_mcp_picker(picker)
            return
        if action == "back":
            picker.view = "servers" if picker.view == "detail" else "detail"
            picker.confirming_remove = False
            await self.telegram.answer_callback(incoming.callback_id)
            await self._render_mcp_picker(picker)
            return
        server = self._selected_mcp_server(picker)
        if server is None or picker.view != "detail":
            await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.stale"))
            return
        if action == "remove":
            if server.is_plugin:
                await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.plugin_managed_short"))
                return
            picker.confirming_remove = True
            await self.telegram.answer_callback(incoming.callback_id)
            await self._render_mcp_picker(picker)
            return
        if action in {"enable", "disable", "logout"}:
            if action == "logout" and server.auth_status != "oAuth":
                await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.stale"))
                return
            if action != "logout" and server.is_plugin:
                await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.plugin_managed_short"))
                return
            picker.confirming_action = action
            await self.telegram.answer_callback(incoming.callback_id)
            await self._render_mcp_picker(picker)
            return
        if action == "action_confirm" and picker.confirming_action:
            confirmed_action = picker.confirming_action
            picker.confirming_action = None
            await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.saving"))
            try:
                if confirmed_action == "logout":
                    await self._codex().logout_mcp_server(server.name)
                else:
                    await self._codex().set_mcp_server_enabled(
                        server.name,
                        confirmed_action == "enable",
                    )
                await self._reload_mcp_after_change()
                if confirmed_action == "logout":
                    await self.telegram.send_message(self.chat_id, self._text("mcp.label.logged_out", name=server.name))
            except Exception as exc:
                self._log_mcp_failure(
                    "Could not update Codex MCP server",
                    exc,
                    event="telegram.mcp.action_failed",
                )
                await self.telegram.send_message(self.chat_id, self._text("mcp.label.operation_failed"))
            await self._refresh_mcp_picker(picker)
            return
        if action == "remove_confirm" and picker.confirming_remove:
            await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.removing"))
            try:
                await self._codex().remove_mcp_server(server.name)
                await self._reload_mcp_after_change()
            except Exception as exc:
                self._log_mcp_failure("Could not remove Codex MCP server", exc, event="telegram.mcp.remove_failed")
                picker.confirming_remove = False
                await self.telegram.send_message(self.chat_id, self._text("mcp.label.operation_failed"))
                await self._refresh_mcp_picker(picker)
                return
            picker.selected_name = None
            picker.view = "servers"
            picker.confirming_remove = False
            await self.telegram.send_message(self.chat_id, self._text("mcp.label.removed", name=server.name))
            await self._refresh_mcp_picker(picker)
            return
        if action == "login":
            await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.oauth_starting"))
            await self._start_mcp_oauth_message(self.chat_id, server.name, picker=picker)
            return
        await self.telegram.answer_callback(incoming.callback_id, self._text("mcp.label.stale"))

    async def _save_mcp_wizard(self, wizard: McpAddWizard, picker: McpPicker) -> None:
        try:
            assert wizard.name
            if wizard.transport == "stdio":
                assert wizard.command is not None
                await self._codex().add_mcp_stdio_server(
                    wizard.name,
                    wizard.command,
                    wizard.args,
                    wizard.cwd,
                    wizard.env_vars,
                )
            else:
                assert wizard.url is not None
                await self._codex().add_mcp_http_server(
                    wizard.name,
                    wizard.url,
                    wizard.auth,
                    wizard.bearer_token_env_var,
                )
            await self._reload_mcp_after_change()
        except ValueError as exc:
            self._log_mcp_failure("MCP server configuration was rejected", exc, event="telegram.mcp.add_rejected", level=logging.INFO)
            self.mcp_add_wizard = None
            picker.view = "servers"
            await self.telegram.send_message(self.chat_id, self._text("mcp.label.add_rejected"))
            await self._refresh_mcp_picker(picker)
            return
        except Exception as exc:
            self._log_mcp_failure("Could not add Codex MCP server", exc, event="telegram.mcp.add_failed")
            await self.telegram.send_message(self.chat_id, self._text("mcp.label.operation_failed"))
            return
        name = wizard.name
        self.mcp_add_wizard = None
        picker.view = "servers"
        picker.selected_name = None
        await self.telegram.send_message(self.chat_id, self._text("mcp.label.added", name=name))
        await self._refresh_mcp_picker(picker)

    async def _reload_mcp_after_change(self) -> None:
        if self._reload_mcp_config is not None:
            reloaded, failed = await self._reload_mcp_config()
            logger.info(
                "Reloaded shared MCP config across running Codex contexts",
                extra={
                    "event": "codex.mcp.broadcast_reload.completed",
                    "reloaded_context_count": reloaded,
                    "failed_context_count": failed,
                    **self._log_context_fields(),
                },
            )
            if failed:
                await self.telegram.send_message(
                    self.chat_id,
                    self._text("mcp.label.reload_partial", count=failed),
                )


    def _log_mcp_failure(
        self,
        message: str,
        error: BaseException,
        *,
        event: str,
        level: int = logging.ERROR,
    ) -> None:
        logger.log(
            level,
            message,
            extra={
                "event": event,
                "error_type": type(error).__name__,
                **self._log_context_fields(),
            },
        )
    async def _ask_mcp_elicitation(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self.mcp_elicitation._ask_mcp_elicitation(params)

    def _mcp_form_prompt(
        self,
        server_name: str,
        params: dict[str, Any],
        schema: dict[str, Any],
    ) -> str:
        return self.mcp_elicitation._mcp_form_prompt(server_name, params, schema)

    async def _handle_mcp_form_input(
        self,
        incoming: IncomingMessage,
        pending: PendingMcpElicitation,
    ) -> None:
        await self.mcp_elicitation._handle_mcp_form_input(incoming, pending)

    @staticmethod
    def _validate_mcp_form_content(schema: dict[str, Any], content: dict[str, Any]) -> None:
        McpElicitationController._validate_mcp_form_content(schema, content)

    @staticmethod
    def _validate_mcp_scalar(definition: dict[str, Any], value: Any) -> None:
        McpElicitationController._validate_mcp_scalar(definition, value)

    async def _handle_mcp_elicitation_callback(
        self,
        incoming: IncomingCallback,
        parts: list[str],
    ) -> None:
        await self.mcp_elicitation._handle_mcp_elicitation_callback(incoming, parts)
