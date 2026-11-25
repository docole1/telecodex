"""Telegram prompts for Codex approvals, questions, and permissions."""

from __future__ import annotations

import asyncio
import secrets
from typing import Any, Awaitable, Callable

from aiogram import Bot
from telecodex.telegram.pagination import two_column_rows
from telecodex.telegram.user_state import PendingApproval, PendingUserInput


class TelegramCodexPromptController:
    """Own approval and user-input prompts for one Telegram context."""

    def __init__(
        self,
        *,
        telegram: Bot,
        chat_id: int,
        is_group: bool,
        is_attached: Callable[[], bool],
        codex: Callable[[], Any],
        text: Callable[..., str],
        truncate: Callable[..., str],
        edit_message: Callable[..., Awaitable[None]],
        pause_typing: Callable[[], Awaitable[None]],
        resume_typing: Callable[[], None],
        is_shutting_down: Callable[[], bool],
        shows_detailed_activity: Callable[[], bool],
        ask_mcp_elicitation: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> None:
        self.telegram = telegram
        self.chat_id = chat_id
        self.is_group = is_group
        self._is_attached = is_attached
        self._codex = codex
        self._text = text
        self._truncate = truncate
        self._edit_message = edit_message
        self._pause_typing = pause_typing
        self._resume_typing = resume_typing
        self._is_shutting_down = is_shutting_down
        self._shows_detailed_activity = shows_detailed_activity
        self._ask_mcp_elicitation = ask_mcp_elicitation
        self.pending_approvals: dict[str, PendingApproval] = {}
        self.pending_user_input: PendingUserInput | None = None

    @property
    def attached(self) -> bool:
        return self._is_attached()

    async def _on_codex_prompt(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.is_group and not self.attached:
            if method == "item/tool/requestUserInput":
                return {"answers": {}}
            if method == "item/permissions/requestApproval":
                return {"permissions": {}}
            if method == "mcpServer/elicitation/request":
                return {"action": "decline"}
            return {"decision": "decline"}
        if method == "mcpServer/elicitation/request":
            return await self._ask_mcp_elicitation(params)
        if method == "item/tool/requestUserInput":
            return await self._ask_user_input(params)
        if method == "item/permissions/requestApproval":
            return await self._ask_permission(params)
        decision = await self._ask_approval(method, params)
        return {"decision": decision}


    async def _ask_permission(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.is_group and not self.attached:
            return {"permissions": {}}
        requested = params.get("permissions", {})
        if not isinstance(requested, dict):
            return {"permissions": {}}
        network = requested.get("network")
        filesystem = requested.get("fileSystem")
        network_requested = isinstance(network, dict) and network.get("enabled")
        filesystem_requested = isinstance(filesystem, dict) and any(
            filesystem.get(key) for key in ("read", "write", "entries")
        )
        if filesystem_requested and not network_requested:
            await self.telegram.send_message(
                self.chat_id,
                self._text("permissions.label.filesystem_denied"),
            )
            return {"permissions": {}}
        if not network_requested:
            return {"permissions": {}}

        token = secrets.token_urlsafe(7)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        reason = self._truncate(
            str(params.get("reason") or self._text("permissions.values.default_reason")),
            500,
        )
        text_key = (
            "permissions.label.network_request_with_filesystem"
            if filesystem_requested
            else "permissions.label.network_request"
        )
        await self._pause_typing()
        message = await self.telegram.send_message(
            self.chat_id,
            self._text(
                text_key,
                filesystem_note=self._text("permissions.values.filesystem_note"),
                reason=reason,
            ),
            reply_markup={
                "inline_keyboard": two_column_rows(
                    [
                        {"text": self._text("approval.label.allow_task"), "callback_data": f"ap:{token}:once"},
                        {"text": self._text("approval.label.allow_session"), "callback_data": f"ap:{token}:session"},
                    ],
                    0,
                )
                + [[{"text": self._text("common.buttons.decline"), "callback_data": f"ap:{token}:deny"}]]
            },
        )
        if self.is_group and not self.attached:
            return {"permissions": {}}
        self._codex().add_pending_prompt(message["message_id"])
        self.pending_approvals[token] = PendingApproval(
            future=future,
            message_id=message["message_id"],
            chat_id=self.chat_id,
        )
        try:
            choice = await asyncio.wait_for(future, timeout=900)
        except asyncio.TimeoutError:
            choice = "decline"
        finally:
            self.pending_approvals.pop(token, None)
            if not self._is_shutting_down():
                self._codex().remove_pending_prompt(message["message_id"])
            self._resume_typing()

        grants: dict[str, Any] = {}
        if choice in {"accept", "acceptForSession"}:
            if isinstance(network, dict) and network.get("enabled"):
                grants["network"] = {"enabled": True}
        result: dict[str, Any] = {"permissions": grants}
        if choice == "acceptForSession":
            result["scope"] = "session"
        return result


    async def _ask_approval(self, method: str, params: dict[str, Any]) -> str:
        token = secrets.token_urlsafe(7)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        reason = params.get("reason")
        network = params.get("networkApprovalContext")
        command = params.get("command")
        if isinstance(network, dict):
            kind = self._text("approval.values.network_kind")
            detail = self._text(
                "approval.values.network_detail",
                protocol=network.get("protocol") or "network",
                host=network.get("host") or "unknown host",
            )
        else:
            kind = self._text(
                "approval.values.filesystem_kind"
                if method == "item/fileChange/requestApproval"
                else "approval.values.command_kind"
            )
            if isinstance(command, str) and command:
                detail = self._text("approval.values.command_detail", command=self._truncate(command, 500))
            else:
                detail = str(reason or self._text("approval.values.default_reason"))
        text = self._text("approval.label.request", kind=kind, detail=self._truncate(detail, 700))
        keyboard = {
            "inline_keyboard": two_column_rows(
                [
                    {"text": self._text("approval.label.allow_once"), "callback_data": f"ap:{token}:once"},
                    {"text": self._text("approval.label.allow_session"), "callback_data": f"ap:{token}:session"},
                ],
                0,
            )
            + [[{"text": self._text("common.buttons.decline"), "callback_data": f"ap:{token}:deny"}]]
        }
        await self._pause_typing()
        message = await self.telegram.send_message(self.chat_id, text, reply_markup=keyboard)
        if self.is_group and not self.attached:
            return "cancel"
        self._codex().add_pending_prompt(message["message_id"])
        pending = PendingApproval(
            future=future,
            message_id=message["message_id"],
            chat_id=self.chat_id,
        )
        self.pending_approvals[token] = pending
        try:
            return await asyncio.wait_for(future, timeout=900)
        except asyncio.TimeoutError:
            return "cancel"
        finally:
            self.pending_approvals.pop(token, None)
            if not self._is_shutting_down():
                self._codex().remove_pending_prompt(message["message_id"])
            self._resume_typing()


    async def _ask_user_input(self, params: dict[str, Any]) -> dict[str, Any]:
        questions = params.get("questions", [])
        if not isinstance(questions, list) or not questions:
            return {"answers": {}}
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        prompt_lines = [self._text("user_input.label.title")]
        for index, question in enumerate(questions, start=1):
            if isinstance(question, dict):
                question_text = question.get("question")
                if not isinstance(question_text, str) or not question_text:
                    question_text = self._text("user_input.values.question_fallback")
                prompt_lines.append(
                    self._text("user_input.values.numbered_question", index=index, question=question_text)
                )
                options = question.get("options")
                if isinstance(options, list):
                    labels = [str(item.get("label", "")) for item in options if isinstance(item, dict)]
                    if labels:
                        prompt_lines.append(self._text("user_input.values.options", options=", ".join(labels)))
        prompt_lines.append(self._text("user_input.label.instructions"))
        await self._pause_typing()
        message = await self.telegram.send_message(
            self.chat_id,
            "\n".join(prompt_lines),
        )
        if self.is_group and not self.attached:
            return {"answers": {}}
        self._codex().add_pending_prompt(message["message_id"])
        self.pending_user_input = PendingUserInput(future=future, questions=questions)
        try:
            return await asyncio.wait_for(future, timeout=900)
        except asyncio.TimeoutError:
            await self._edit_message(
                self.chat_id,
                message["message_id"],
                self._text("user_input.label.timeout"),
            )
            return {"answers": {}}
        finally:
            if self.pending_user_input and self.pending_user_input.future is future:
                self.pending_user_input = None
            if not self._is_shutting_down():
                self._codex().remove_pending_prompt(message["message_id"])
            self._resume_typing()


    async def _resolve_user_input(self, text: str) -> None:
        pending = self.pending_user_input
        if pending is None or pending.future.done():
            return
        lines = [line.strip() for line in text.splitlines()]
        answers: dict[str, dict[str, list[str]]] = {}
        for index, question in enumerate(pending.questions):
            if not isinstance(question, dict) or not isinstance(question.get("id"), str):
                continue
            if len(pending.questions) == 1:
                answer = text.strip()
            else:
                answer = lines[index] if index < len(lines) else ""
            answers[question["id"]] = {"answers": [answer] if answer else []}
        pending.future.set_result({"answers": answers})
        self.pending_user_input = None
        if self._shows_detailed_activity():
            await self.telegram.send_message(self.chat_id, self._text("user_input.label.delivered"))
