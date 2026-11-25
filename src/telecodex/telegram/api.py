from __future__ import annotations

import logging
import os
import ssl
import warnings
from urllib.parse import urlsplit
from pathlib import Path
from time import monotonic
from typing import Any, AsyncIterator

import aiohttp
from aiohttp import BasicAuth
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import (
    TelegramAPIError as AiogramTelegramAPIError,
    TelegramBadRequest,
    TelegramConflictError,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramNotFound,
    TelegramRetryAfter,
    TelegramServerError,
    TelegramUnauthorizedError,
)
from aiogram.methods import GetUpdates
from aiogram.types import LinkPreviewOptions, ReplyParameters
from aiohttp_socks import ProxyConnector
from python_socks import ProxyType
import certifi

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
    return any(marker in description for marker in ("html", "parse", "format", "entity"))


class TelegramFileTooLargeError(TelegramApiError):
    pass


class TelegramFileStorageError(TelegramApiError):
    pass


def _camel_to_api_method(name: str) -> str:
    return name[:1].lower() + name[1:]


def _translate_api_error(exc: AiogramTelegramAPIError, fallback_method: str) -> TelegramApiError:
    method_object = getattr(exc, "method", None)
    method_name = getattr(method_object, "__api_method__", None)
    if not isinstance(method_name, str):
        method_name = _camel_to_api_method(type(method_object).__name__) if method_object else fallback_method

    if isinstance(exc, TelegramBadRequest):
        code = 400
    elif isinstance(exc, TelegramUnauthorizedError):
        code = 401
    elif isinstance(exc, TelegramForbiddenError):
        code = 403
    elif isinstance(exc, TelegramNotFound):
        code = 404
    elif isinstance(exc, TelegramConflictError):
        code = 409
    elif isinstance(exc, TelegramRetryAfter):
        code = 429
    elif isinstance(exc, TelegramServerError):
        code = 500
    else:
        code = None

    if isinstance(exc, TelegramNetworkError):
        original = getattr(exc, "original", None)
        error_type = type(original).__name__ if original is not None else type(exc).__name__
        message = f"Telegram request failed: {method_name} ({error_type})"
    else:
        description = getattr(exc, "message", None)
        if not isinstance(description, str) or not description:
            description = type(exc).__name__
        message = f"Telegram API {method_name} failed: {description}"
    return TelegramApiError(
        message,
        method=method_name,
        error_code=code,
        http_status=code,
    )


def _model_result(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="python", by_alias=True, exclude_none=True)
    return value


def _reply_parameters(message_id: int | None) -> ReplyParameters | None:
    if message_id is None:
        return None
    return ReplyParameters(message_id=message_id, allow_sending_without_reply=True)


class _TelegramAiohttpSession(AiohttpSession):
    """Aiogram session that preserves TeleCodex's HTTPS and remote-DNS proxy modes."""

    def __init__(self, *, proxy: str | tuple[str, BasicAuth] | None, timeout: int) -> None:
        super().__init__(proxy=None, timeout=timeout)
        if proxy is None:
            return
        proxy_url = proxy[0] if isinstance(proxy, tuple) else proxy
        parsed = urlsplit(proxy_url)
        if parsed.scheme == "https":
            auth = proxy[1] if isinstance(proxy, tuple) else None
            tls_context = ssl.create_default_context(cafile=certifi.where())
            self._connector_type = ProxyConnector
            self._connector_init = {
                "host": parsed.hostname,
                "port": parsed.port,
                "proxy_type": ProxyType.HTTP,
                "username": auth.login if auth is not None else None,
                "password": auth.password if auth is not None else None,
                "rdns": True,
                "proxy_ssl": tls_context,
                "ssl": ssl.create_default_context(cafile=certifi.where()),
                "limit": 100,
                "ttl_dns_cache": 3600,
            }
            self._proxy = proxy
            self._should_reset_connector = True
            return
        if parsed.scheme == "socks5h":
            normalized_url = proxy_url.replace("socks5h://", "socks5://", 1)
            proxy = (normalized_url, proxy[1]) if isinstance(proxy, tuple) else normalized_url
        self._setup_proxy_connector(proxy)


class TelegramAPI:
    """Aiogram-backed Telegram gateway with TeleCodex's established interface."""

    def __init__(self, bot_token: str, proxy: ProxySettings | None = None) -> None:
        self._bot_token = bot_token
        self._proxy = proxy
        self.bot = self._new_bot()
        self.bot_user_id: int | None = None

    def _new_session(self) -> AiohttpSession:
        proxy: str | tuple[str, BasicAuth] | None = None
        if self._proxy is not None:
            if self._proxy.username:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="BasicAuth is deprecated.*",
                        category=DeprecationWarning,
                    )
                    proxy_auth = BasicAuth(
                        self._proxy.username,
                        self._proxy.password or "",
                    )
                proxy = (self._proxy.url, proxy_auth)
            else:
                proxy = self._proxy.url
        # Aiogram adds the long-poll timeout to this session timeout when it builds
        # getUpdates requests. Keeping this at 10 seconds gives a 60-second poll timeout.
        return _TelegramAiohttpSession(proxy=proxy, timeout=10)

    def _new_bot(self) -> Bot:
        return Bot(
            self._bot_token,
            session=self._new_session(),
            default=DefaultBotProperties(
                parse_mode=ParseMode.HTML,
                link_preview_is_disabled=True,
            ),
        )

    def set_bot_user_id(self, user_id: int) -> None:
        if user_id <= 0:
            raise ValueError("Telegram bot user ID must be positive")
        self.bot_user_id = user_id

    async def close(self) -> None:
        await self.bot.session.close()

    def add_request_middleware(self, middleware: Any) -> None:
        self.bot.session.middleware.register(middleware)

    async def _request(self, operation: str, request: Any, *, timeout: float | None = None) -> Any:
        started_at = monotonic()
        try:
            result = await request
        except AiogramTelegramAPIError as exc:
            translated = _translate_api_error(exc, operation)
            logger.warning(
                "Telegram Bot API request failed",
                extra={
                    "event": "telegram.api.response_error",
                    "operation": translated.method or operation,
                    "http_status": translated.http_status,
                    "error_code": translated.error_code,
                    "duration_ms": round((monotonic() - started_at) * 1000),
                },
            )
            raise translated from exc
        logger.debug(
            "Telegram Bot API request completed",
            extra={
                "event": "telegram.api.request_completed",
                "operation": operation,
                "duration_ms": round((monotonic() - started_at) * 1000),
                "timeout_seconds": timeout,
            },
        )
        return _model_result(result)

    async def get_me(self, timeout: float | None = None) -> dict[str, Any]:
        result = await self._request(
            "getMe",
            self.bot.get_me(request_timeout=timeout if timeout is not None else 60),
            timeout=timeout if timeout is not None else 60,
        )
        return result if isinstance(result, dict) else {}

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        await self._request(
            "sendChatAction",
            self.bot.send_chat_action(chat_id=chat_id, action=action, request_timeout=2),
            timeout=2,
        )

    async def confirm_updates_before(self, offset: int) -> None:
        """Confirm updates below the persisted offset before Aiogram starts polling."""
        if offset <= 0:
            return
        await self._request(
            "getUpdates",
            self.bot.get_updates(
                offset=offset,
                timeout=0,
                allowed_updates=["message", "callback_query"],
                request_timeout=10,
            ),
            timeout=10,
        )

    async def download_file(
        self,
        file_id: str,
        destination: Path,
        *,
        max_bytes: int,
    ) -> int:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        info = await self._request(
            "getFile",
            self.bot.get_file(file_id=file_id, request_timeout=60),
            timeout=60,
        )
        file_path = info.get("file_path") if isinstance(info, dict) else None
        reported_size = info.get("file_size") if isinstance(info, dict) else None
        if not isinstance(file_path, str) or not file_path:
            raise TelegramApiError("Telegram did not return a file path", method="getFile")
        if isinstance(reported_size, int) and reported_size > max_bytes:
            raise TelegramFileTooLargeError(
                "Telegram file exceeds maximum size",
                method="getFile",
            )

        url = self.bot.session.api.file_url(self._bot_token, file_path)
        stream: AsyncIterator[bytes] | None = None
        file_created = False
        total = 0
        started_at = monotonic()
        try:
            stream = self.bot.session.stream_content(
                url=url,
                timeout=60,
                chunk_size=64 * 1024,
                raise_for_status=True,
            )
            file_descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            file_created = True
            with os.fdopen(file_descriptor, "wb") as output:
                async for chunk in stream:
                    total += len(chunk)
                    if total > max_bytes:
                        raise TelegramFileTooLargeError(
                            "Telegram file exceeds maximum size",
                            method="getFileDownload",
                        )
                    output.write(chunk)
        except TelegramApiError:
            if file_created:
                destination.unlink(missing_ok=True)
            logger.warning(
                "Telegram file download was rejected",
                extra={
                    "event": "telegram.file_download.rejected",
                    "file_size": total,
                    "duration_ms": round((monotonic() - started_at) * 1000),
                },
            )
            raise
        except AiogramTelegramAPIError as exc:
            if file_created:
                destination.unlink(missing_ok=True)
            translated = _translate_api_error(exc, "getFileDownload")
            log_exception(
                logger,
                "Telegram file download failed",
                translated,
                event="telegram.file_download.transport_failed",
                level=logging.WARNING,
                file_size=total,
                duration_ms=round((monotonic() - started_at) * 1000),
            )
            raise translated from exc
        except (aiohttp.ClientError, OSError) as exc:
            if file_created:
                destination.unlink(missing_ok=True)
            if isinstance(exc, OSError) and not isinstance(exc, aiohttp.ClientError):
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
            ) from exc
        finally:
            if stream is not None:
                await stream.aclose()

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
        try:
            result = await self._request(
                "sendMessage",
                self.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup,
                    reply_parameters=_reply_parameters(reply_to_message_id),
                    entities=entities,
                    disable_notification=disable_notification,
                    parse_mode=None if entities else parse_mode,
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                    request_timeout=60,
                ),
                timeout=60,
            )
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
        return result if isinstance(result, dict) else {}

    async def send_rich_message(
        self,
        chat_id: int,
        markdown: str,
        *,
        reply_markup: dict[str, Any] | None = None,
        reply_to_message_id: int | None = None,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        try:
            result = await self._request(
                "sendRichMessage",
                self.bot.send_rich_message(
                    chat_id=chat_id,
                    rich_message={"markdown": markdown},
                    reply_markup=reply_markup,
                    reply_parameters=_reply_parameters(reply_to_message_id),
                    disable_notification=disable_notification,
                    request_timeout=60,
                ),
                timeout=60,
            )
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
        return result if isinstance(result, dict) else {}

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
        from aiogram.types import FSInputFile

        result = await self._request(
            "sendDocument",
            self.bot.send_document(
                chat_id=chat_id,
                document=FSInputFile(path, filename=filename),
                reply_parameters=_reply_parameters(reply_to_message_id),
                request_timeout=180,
            ),
            timeout=180,
        )
        return result if isinstance(result, dict) else {}

    async def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
        parse_mode: str | None = "HTML",
    ) -> dict[str, Any] | bool:
        try:
            return await self._request(
                "editMessageText",
                self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode,
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                    request_timeout=60,
                ),
                timeout=60,
            )
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
        return await self._request(
            "editMessageReplyMarkup",
            self.bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=message_id,
                reply_markup=reply_markup,
                request_timeout=60,
            ),
            timeout=60,
        )

    async def answer_callback(self, callback_query_id: str, text: str | None = None) -> None:
        await self._request(
            "answerCallbackQuery",
            self.bot.answer_callback_query(
                callback_query_id=callback_query_id,
                text=text,
                request_timeout=60,
            ),
            timeout=60,
        )
