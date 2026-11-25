"""Runtime handling for MCP elicitation requests in Telegram chats."""

from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from aiogram import Bot

from telecodex.telegram.pagination import keyboard
from telecodex.telegram.updates import TelegramCallbackContext, TelegramMessageContext


@dataclass
class PendingMcpElicitation:
    token: str
    future: asyncio.Future[dict[str, Any]]
    mode: str
    server_name: str
    message_id: int
    requested_schema: dict[str, Any] | None = None


class McpElicitationController:
    """Own pending MCP requests and render their Telegram prompts."""

    def __init__(
        self,
        *,
        telegram: Bot,
        chat_id: int,
        text: Callable[..., str],
        truncate: Callable[..., str],
        edit_message: Callable[..., Awaitable[None]],
        send_activity: Callable[..., Awaitable[Any]],
    ) -> None:
        self.telegram = telegram
        self.chat_id = chat_id
        self._text = text
        self._truncate = truncate
        self._edit_message = edit_message
        self._send_codex_activity = send_activity
        self.pending: dict[str, PendingMcpElicitation] = {}
        self._form_token: str | None = None

    async def _ask_mcp_elicitation(self, params: dict[str, Any]) -> dict[str, Any]:
        mode = params.get("mode")
        server_name = params.get("serverName")
        form_confirmation = False
        if not isinstance(server_name, str):
            return {"action": "decline"}
        if mode not in {"url", "form", "openai/form", "openaiForm"}:
            await self._send_codex_activity(self._text("mcp.label.elicitation_unsupported", name=server_name))
            return {"action": "decline"}
        if mode == "url":
            url = params.get("url")
            if not isinstance(url, str):
                return {"action": "decline"}
            parsed = urlsplit(url)
            try:
                _ = parsed.port
            except ValueError:
                return {"action": "decline"}
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
            ):
                return {"action": "decline"}
        else:
            schema = params.get("requestedSchema")
            if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict):
                return {"action": "decline"}
            required = schema.get("required")
            if not schema["properties"] and not required:
                form_confirmation = True
            elif not schema["properties"] or len(schema["properties"]) > 10:
                await self._send_codex_activity(self._text("mcp.label.elicitation_unsupported", name=server_name))
                return {"action": "decline"}
        if self.pending:
            return {"action": "cancel"}

        token = secrets.token_urlsafe(7)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        schema = params.get("requestedSchema") if isinstance(params.get("requestedSchema"), dict) else None
        pending = PendingMcpElicitation(
            token=token,
            future=future,
            mode="url" if mode == "url" else "form_confirmation" if form_confirmation else "form",
            server_name=server_name,
            message_id=0,
            requested_schema=schema,
        )
        if mode == "url":
            prompt = self._text(
                "mcp.label.elicitation_url",
                name=server_name,
                message=self._truncate(str(params.get("message") or ""), 1200),
                url=url,
            )
            markup = keyboard([
                [{"text": self._text("mcp.label.elicitation_open_url"), "url": url}],
                [
                    {"text": self._text("mcp.label.elicitation_accept"), "callback_data": f"me:{token}:accept"},
                    {"text": self._text("common.buttons.decline"), "callback_data": f"me:{token}:decline"},
                ],
            ])
        elif form_confirmation:
            prompt = self._text(
                "mcp.label.elicitation_form_confirmation",
                name=server_name,
                message=self._truncate(str(params.get("message") or ""), 1200),
            )
            markup = keyboard([[
                {"text": self._text("common.buttons.confirm"), "callback_data": f"me:{token}:accept"},
                {"text": self._text("common.buttons.decline"), "callback_data": f"me:{token}:decline"},
            ]])
        else:
            prompt = self._mcp_form_prompt(server_name, params, schema or {})
            markup = keyboard([[
                {"text": self._text("common.buttons.decline"), "callback_data": f"me:{token}:decline"},
                {"text": self._text("mcp.label.elicitation_cancel"), "callback_data": f"me:{token}:cancel"},
            ]])
        message = await self.telegram.send_message(self.chat_id, prompt, reply_markup=markup)
        message_id = message.message_id
        if not isinstance(message_id, int):
            return {"action": "cancel"}
        pending.message_id = message_id
        self.pending[token] = pending
        if pending.mode == "form":
            self._form_token = token
        try:
            return await asyncio.wait_for(future, timeout=900)
        except asyncio.TimeoutError:
            await self._edit_message(
                self.chat_id,
                message_id,
                self._text("mcp.label.elicitation_expired"),
                {"inline_keyboard": []},
            )
            return {"action": "cancel"}
        finally:
            self.pending.pop(token, None)
            if self._form_token == token:
                self._form_token = None


    def _mcp_form_prompt(
        self,
        server_name: str,
        params: dict[str, Any],
        schema: dict[str, Any],
    ) -> str:
        properties = schema.get("properties")
        required = set(schema.get("required") or [])
        lines: list[str] = []
        if isinstance(properties, dict):
            for name, definition in properties.items():
                if not isinstance(name, str):
                    continue
                definition = definition if isinstance(definition, dict) else {}
                title = definition.get("title")
                description = definition.get("description")
                field_type = definition.get("type", "value")
                choices = definition.get("enum")
                if not isinstance(choices, list):
                    choices = [
                        option.get("const")
                        for option in definition.get("oneOf", [])
                        if isinstance(option, dict) and "const" in option
                    ]
                items = definition.get("items")
                if not choices and isinstance(items, dict):
                    choices = items.get("enum")
                label = str(title) if isinstance(title, str) and title else name
                suffix = " (required)" if name in required else ""
                if isinstance(description, str) and description.strip():
                    suffix += f" — {description.strip()[:160]}"
                if isinstance(choices, list) and choices:
                    suffix += f"; варианты: {', '.join(str(value) for value in choices[:12])}"
                lines.append(f"• {name}: {label} [{field_type}]{suffix}")
        return self._truncate(
            self._text(
                "mcp.label.elicitation_form",
                name=server_name,
                message=self._truncate(str(params.get("message") or ""), 1000),
                fields="\n".join(lines) or self._text("mcp.values.none"),
            ),
            3500,
        )


    async def _handle_mcp_form_input(
        self,
        incoming: TelegramMessageContext,
        pending: PendingMcpElicitation,
    ) -> None:
        if incoming.attachments:
            await self.telegram.send_message(self.chat_id, self._text("mcp.label.elicitation_json_required"))
            return
        try:
            content = json.loads(incoming.text)
            if not isinstance(content, dict):
                raise ValueError
            self._validate_mcp_form_content(pending.requested_schema or {}, content)
        except (json.JSONDecodeError, ValueError, TypeError):
            await self.telegram.send_message(self.chat_id, self._text("mcp.label.elicitation_json_invalid"))
            return
        if not pending.future.done():
            pending.future.set_result({"action": "accept", "content": content})
        await self._edit_message(
            self.chat_id,
            pending.message_id,
            self._text("mcp.label.elicitation_submitted"),
            {"inline_keyboard": []},
        )


    @staticmethod
    def _validate_mcp_form_content(schema: dict[str, Any], content: dict[str, Any]) -> None:
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("Invalid MCP form schema")
        if set(content).difference(properties):
            raise ValueError("MCP form contains unknown fields")
        required = schema.get("required")
        if isinstance(required, list) and any(name not in content for name in required):
            raise ValueError("MCP form is missing a required field")
        for name, value in content.items():
            definition = properties.get(name)
            if not isinstance(definition, dict):
                raise ValueError("Invalid MCP form field")
            expected = definition.get("type")
            items = definition.get("items")
            if expected == "array":
                if not isinstance(value, list) or not isinstance(items, dict):
                    raise ValueError("MCP form field has the wrong type")
                minimum_items = definition.get("minItems")
                maximum_items = definition.get("maxItems")
                if isinstance(minimum_items, int) and len(value) < minimum_items:
                    raise ValueError("MCP form field has too few values")
                if isinstance(maximum_items, int) and len(value) > maximum_items:
                    raise ValueError("MCP form field has too many values")
                for item in value:
                    McpElicitationController._validate_mcp_scalar(items, item)
            else:
                McpElicitationController._validate_mcp_scalar(definition, value)


    @staticmethod
    def _validate_mcp_scalar(definition: dict[str, Any], value: Any) -> None:
        expected = definition.get("type")
        choices = definition.get("enum")
        if not isinstance(choices, list):
            choices = [
                option.get("const")
                for option in definition.get("oneOf", [])
                if isinstance(option, dict) and "const" in option
            ]
        if expected == "string" or (expected is None and choices):
            if not isinstance(value, str):
                raise ValueError("MCP form field has the wrong type")
        elif expected == "boolean" and not isinstance(value, bool):
            raise ValueError("MCP form field has the wrong type")
        elif expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError("MCP form field has the wrong type")
        elif expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise ValueError("MCP form field has the wrong type")
        if isinstance(value, str):
            minimum = definition.get("minLength")
            maximum = definition.get("maxLength")
            if isinstance(minimum, int) and len(value) < minimum:
                raise ValueError("MCP form field is too short")
            if isinstance(maximum, int) and len(value) > maximum:
                raise ValueError("MCP form field is too long")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum = definition.get("minimum")
            maximum = definition.get("maximum")
            if isinstance(minimum, (int, float)) and value < minimum:
                raise ValueError("MCP form field is too small")
            if isinstance(maximum, (int, float)) and value > maximum:
                raise ValueError("MCP form field is too large")
        if isinstance(choices, list) and choices and value not in choices:
            raise ValueError("MCP form field is not an allowed choice")


    async def _handle_mcp_elicitation_callback(
        self,
        incoming: TelegramCallbackContext,
        parts: list[str],
    ) -> None:
        if len(parts) != 3:
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("mcp.label.elicitation_closed"))
            return
        pending = self.pending.get(parts[1])
        if pending is None or pending.message_id != incoming.message_id or pending.future.done():
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("mcp.label.elicitation_closed"))
            return
        action = parts[2]
        if action not in {"accept", "decline", "cancel"} or (
            action == "accept" and pending.mode not in {"url", "form_confirmation"}
        ):
            await self.telegram.answer_callback_query(incoming.callback_id, self._text("mcp.label.elicitation_closed"))
            return
        response = {"action": action}
        if action == "accept" and pending.mode == "form_confirmation":
            response["content"] = {}
        pending.future.set_result(response)
        await self.telegram.answer_callback_query(incoming.callback_id)
        await self._edit_message(
            incoming.chat_id,
            incoming.message_id,
            self._text("mcp.label.elicitation_submitted" if action == "accept" else "mcp.label.elicitation_declined"),
            {"inline_keyboard": []},
        )
