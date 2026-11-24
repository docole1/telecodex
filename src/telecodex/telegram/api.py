from __future__ import annotations

import json
import logging
import os
from typing import Any
from pathlib import Path
from time import monotonic
from urllib.parse import quote

import httpx

from telecodex.config import ProxySettings
from telecodex.logging_config import log_exception

logger = logging.getLogger(__name__)

class TelegramApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        method: str | None = None,
        error_code: int | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.error_code = error_code
        self.http_status = http_status


def _is_formatting_rejection(exc: TelegramApiError, parse_mode: str | None) -> bool:
    if parse_mode is None or parse_mode.casefold() != "html" or exc.error_code != 400:
        return False
    description = str(exc).partition(" failed: ")[2].casefold()
    return any(
        marker in description
        for marker in ("html", "parse", "format", "entity")
    )


class TelegramFileTooLargeError(TelegramApiError):
    pass


class TelegramFileStorageError(TelegramApiError):
    pass


class TelegramAPI:
    def __init__(self, bot_token: str, proxy: ProxySettings | None = None) -> None:
        self._base_url = f"https://api.telegram.org/bot{bot_token}"
        self._file_base_url = f"https://api.telegram.org/file/bot{bot_token}"
        self._proxy = proxy
        self._client = self._new_client()
        self.bot_user_id: int | None = None

    def set_bot_user_id(self, user_id: int) -> None:
        if user_id <= 0:
            raise ValueError("Telegram bot user ID must be positive")
        self.bot_user_id = user_id

    def _new_client(self) -> httpx.AsyncClient:
        # getUpdates long-polls for 50 seconds; keep a 10-second network margin.
        client_options: dict[str, Any] = {
            "timeout": httpx.Timeout(60.0, connect=10.0),
        }
        if self._proxy is not None:
            auth = (
                (self._proxy.username, self._proxy.password or "")
                if self._proxy.username
                else None
            )
            client_options["proxy"] = httpx.Proxy(self._proxy.url, auth=auth)
        return httpx.AsyncClient(**client_options)

    async def close(self) -> None:
        await self._client.aclose()

    async def reset_connection(self) -> None:
        """Replace the HTTP pool while keeping this API object referenced by controllers."""
        previous = self._client
        self._client = self._new_client()
        await previous.aclose()

    async def call(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        started_at = monotonic()
        try:
            if timeout is None:
                response = await self._client.post(f"{self._base_url}/{method}", json=payload or {})
            else:
                response = await self._client.post(
                    f"{self._base_url}/{method}",
                    json=payload or {},
                    timeout=timeout,
                )
        except httpx.HTTPError as exc:
            # The Bot API token is part of the request URL; log only the exception class.
            log_exception(
                logger,
                "Telegram Bot API transport request failed",
                exc,
                event="telegram.api.transport_failed",
                level=logging.WARNING,
                operation=method,
                timeout_seconds=timeout or 60,
                duration_ms=round((monotonic() - started_at) * 1000),
            )
            raise TelegramApiError(
                f"Telegram request failed: {method} ({type(exc).__name__})",
                method=method,
            ) from exc
        try:
            result = self._response_result(method, response)
        except TelegramApiError as exc:
            logger.warning(
                "Telegram Bot API returned an error",
                extra={
                    "event": "telegram.api.response_error",
                    "operation": method,
                    "http_status": response.status_code,
                    "error_code": exc.error_code,
                    "duration_ms": round((monotonic() - started_at) * 1000),
                },
            )
            raise
        logger.debug(
            "Telegram Bot API request completed",
            extra={
                "event": "telegram.api.request_completed",
                "operation": method,
                "http_status": response.status_code,
                "duration_ms": round((monotonic() - started_at) * 1000),
            },
        )
        return result

    @staticmethod
    def _response_result(method: str, response: httpx.Response) -> Any:
        try:
            result = response.json()
        except ValueError as exc:
            raise TelegramApiError(
                f"Telegram returned invalid JSON for {method} (HTTP {response.status_code})"
            ) from exc
        if response.is_error or not isinstance(result, dict) or not result.get("ok"):
            description = result.get("description", "unknown error") if isinstance(result, dict) else "unknown error"
            code = result.get("error_code") if isinstance(result, dict) else None
            raise TelegramApiError(
                f"Telegram API {method} failed: {description}",
                method=method,
                error_code=code if isinstance(code, int) else None,
                http_status=response.status_code,
            )
        return result.get("result")

    async def get_me(self, timeout: float | None = None) -> dict[str, Any]:
        return await self.call("getMe", timeout=timeout)

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        await self.call(
            "sendChatAction",
            {"chat_id": chat_id, "action": action},
            timeout=2.0,
        )

    async def get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": 50, "allowed_updates": ["message", "callback_query"]}
        if offset:
            payload["offset"] = offset
        result = await self.call("getUpdates", payload)
        return result if isinstance(result, list) else []

    async def download_file(
        self,
        file_id: str,
        destination: Path,
        *,
        max_bytes: int,
    ) -> int:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        info = await self.call("getFile", {"file_id": file_id})
        if not isinstance(info, dict) or not isinstance(info.get("file_path"), str):
            raise TelegramApiError("Telegram did not return a file path")
        reported_size = info.get("file_size")
        if isinstance(reported_size, int) and reported_size > max_bytes:
            raise TelegramFileTooLargeError("Telegram file exceeds maximum size")

        file_path = quote(info["file_path"], safe="/")
        url = f"{self._file_base_url}/{file_path}"
        total = 0
        started_at = monotonic()
        try:
            async with self._client.stream("GET", url) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                    raise TelegramFileTooLargeError("Telegram file exceeds maximum size")
                file_descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(file_descriptor, "wb") as output:
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise TelegramFileTooLargeError("Telegram file exceeds maximum size")
                        output.write(chunk)
        except TelegramApiError as exc:
            destination.unlink(missing_ok=True)
            logger.warning(
                "Telegram file download was rejected",
                extra={
                    "event": "telegram.file_download.rejected",
                    "error_type": type(exc).__name__,
                    "file_size": total,
                    "duration_ms": round((monotonic() - started_at) * 1000),
                },
            )
            raise
        except OSError as exc:
            destination.unlink(missing_ok=True)
            log_exception(
                logger,
                "Could not store Telegram file download",
                exc,
                event="telegram.file_download.storage_failed",
                level=logging.WARNING,
                file_size=total,
                duration_ms=round((monotonic() - started_at) * 1000),
            )
            raise TelegramFileStorageError(
                f"Could not save downloaded Telegram file ({type(exc).__name__})"
            ) from exc
        except httpx.HTTPError as exc:
            destination.unlink(missing_ok=True)
            log_exception(
                logger,
                "Telegram file download transport failed",
                exc,
                event="telegram.file_download.transport_failed",
                level=logging.WARNING,
                file_size=total,
                duration_ms=round((monotonic() - started_at) * 1000),
            )
            raise TelegramApiError(
                f"Telegram file download failed ({type(exc).__name__})",
                method="getFileDownload",
                http_status=exc.response.status_code
                if isinstance(exc, httpx.HTTPStatusError)
                else None,
            ) from exc
        logger.info(
            "Telegram file download completed",
            extra={
                "event": "telegram.file_download.completed",
                "file_size": total,
                "file_type": Path(destination).suffix.lower().lstrip("."),
                "duration_ms": round((monotonic() - started_at) * 1000),
            },
        )
        return total

    async def send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
        reply_to_message_id: int | None = None,
        entities: list[dict[str, Any]] | None = None,
        disable_notification: bool = False,
        parse_mode: str | None = "HTML",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
            "disable_notification": disable_notification,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        if reply_to_message_id is not None:
            payload["reply_parameters"] = {
                "message_id": reply_to_message_id,
                "allow_sending_without_reply": True,
            }
        if entities:
            payload["entities"] = entities
        elif parse_mode is not None:
            payload["parse_mode"] = parse_mode
        try:
            return await self.call("sendMessage", payload)
        except TelegramApiError as exc:
            if entities or not _is_formatting_rejection(exc, parse_mode):
                raise
            logger.warning(
                "Telegram rejected HTML formatting; retrying as plain text",
                extra={
                    "event": "telegram.api.html_message.fallback",
                    "http_status": exc.http_status,
                    "error_code": exc.error_code,
                },
            )
            return await self.send_message(
                chat_id,
                text,
                reply_markup=reply_markup,
                reply_to_message_id=reply_to_message_id,
                disable_notification=disable_notification,
                parse_mode=None,
            )

    async def send_rich_message(
        self,
        chat_id: int,
        markdown: str,
        *,
        reply_markup: dict[str, Any] | None = None,
        reply_to_message_id: int | None = None,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "rich_message": {"markdown": markdown},
            "disable_notification": disable_notification,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        if reply_to_message_id is not None:
            payload["reply_parameters"] = {
                "message_id": reply_to_message_id,
                "allow_sending_without_reply": True,
            }
        try:
            return await self.call("sendRichMessage", payload)
        except TelegramApiError as exc:
            description = str(exc).partition(" failed: ")[2].casefold()
            format_rejected = any(
                marker in description
                for marker in ("rich", "markdown", "html", "parse", "format", "entity")
            )
            if exc.error_code != 400 or not format_rejected:
                raise
            logger.warning(
                "Telegram rejected Rich Message formatting; retrying as plain text",
                extra={
                    "event": "telegram.api.rich_message.fallback",
                    "http_status": exc.http_status,
                    "error_code": exc.error_code,
                },
            )
            return await self.send_message(
                chat_id,
                markdown,
                reply_markup=reply_markup,
                reply_to_message_id=reply_to_message_id,
                disable_notification=disable_notification,
                parse_mode=None,
            )

    async def send_document(
        self,
        chat_id: int,
        path: Path,
        filename: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> dict[str, Any]:
        if not filename or "/" in filename or "\\" in filename or "\0" in filename:
            raise TelegramApiError("Telegram document file name is invalid")
        payload: dict[str, str] = {"chat_id": str(chat_id)}
        if reply_to_message_id is not None:
            payload["reply_parameters"] = json.dumps(
                {
                    "message_id": reply_to_message_id,
                    "allow_sending_without_reply": True,
                }
            )
        try:
            with path.open("rb") as document:
                response = await self._client.post(
                    f"{self._base_url}/sendDocument",
                    data=payload,
                    files={"document": (filename, document, "application/octet-stream")},
                    timeout=httpx.Timeout(180.0, connect=10.0),
                )
        except (OSError, httpx.HTTPError) as exc:
            raise TelegramApiError(
                f"Telegram document upload failed ({type(exc).__name__})"
            ) from exc
        return self._response_result("sendDocument", response)

    async def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
        parse_mode: str | None = "HTML",
    ) -> dict[str, Any] | bool:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            return await self.call("editMessageText", payload)
        except TelegramApiError as exc:
            if not _is_formatting_rejection(exc, parse_mode):
                raise
            logger.warning(
                "Telegram rejected HTML formatting while editing; retrying as plain text",
                extra={
                    "event": "telegram.api.html_edit.fallback",
                    "http_status": exc.http_status,
                    "error_code": exc.error_code,
                },
            )
            return await self.edit_message(
                chat_id,
                message_id,
                text,
                reply_markup,
                parse_mode=None,
            )

    async def edit_message_reply_markup(
        self,
        chat_id: int,
        message_id: int,
        reply_markup: dict[str, Any],
    ) -> dict[str, Any] | bool:
        return await self.call(
            "editMessageReplyMarkup",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": reply_markup,
            },
        )

    async def answer_callback(self, callback_query_id: str, text: str | None = None) -> None:
        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        await self.call("answerCallbackQuery", payload)
