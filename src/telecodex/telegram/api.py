"""Aiogram Bot construction and narrow Telegram-specific safety helpers."""

from __future__ import annotations

import logging
import os
import ssl
import warnings
from pathlib import Path
from time import monotonic
from typing import AsyncIterator
from urllib.parse import urlsplit

import aiohttp
import certifi
from aiohttp import BasicAuth
from aiohttp_socks import ProxyConnector
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.session.middlewares.base import BaseRequestMiddleware
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.methods import EditMessageText, SendMessage, SendRichMessage
from aiogram.methods.base import TelegramMethod
from aiogram.types import InputRichMessage, LinkPreviewOptions, ReplyParameters
from python_socks import ProxyType

from telecodex.config import ProxySettings
from telecodex.logging_config import log_exception

logger = logging.getLogger(__name__)


class TelegramFileTooLargeError(RuntimeError):
    """Telegram reported a file larger than TeleCodex permits."""


class TelegramFileStorageError(RuntimeError):
    """A downloaded Telegram file could not be safely stored locally."""


class TelegramFileDownloadError(RuntimeError):
    """A Telegram file download failed before a complete file was stored."""


class _TelegramAiohttpSession(AiohttpSession):
    """Aiogram session preserving HTTPS and remote-DNS proxy modes."""

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

    async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None):
        if timeout is None:
            timeout = {
                "sendChatAction": 2,
                "sendDocument": 180,
            }.get(method.__api_method__, 60)
        return await super().make_request(bot, method, timeout=timeout)


def create_telegram_bot(bot_token: str, proxy: ProxySettings | None = None) -> Bot:
    proxy_value: str | tuple[str, BasicAuth] | None = None
    if proxy is not None:
        if proxy.username:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="BasicAuth is deprecated.*",
                    category=DeprecationWarning,
                )
                proxy_auth = BasicAuth(proxy.username, proxy.password or "")
            proxy_value = (proxy.url, proxy_auth)
        else:
            proxy_value = proxy.url
    # Aiogram adds the long-poll timeout to this session timeout when it builds
    # getUpdates requests. This keeps the long-poll request within 60 seconds.
    session = _TelegramAiohttpSession(proxy=proxy_value, timeout=10)
    return Bot(
        bot_token,
        session=session,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML,
            link_preview_is_disabled=True,
        ),
    )


class TelegramFormattingFallbackMiddleware(BaseRequestMiddleware):
    """Retry rejected HTML/Rich formatting through Aiogram's request pipeline."""

    async def __call__(self, make_request, bot: Bot, method: TelegramMethod):
        method = self._normalize_method(method)
        started_at = monotonic()
        try:
            result = await make_request(bot, method)
        except TelegramBadRequest as exc:
            fallback = self._fallback_method(bot, method, exc)
            if fallback is None:
                self._log_error(method, exc, started_at)
                raise
            logger.warning(
                "Telegram rejected message formatting; retrying without formatting",
                extra={
                    "event": "telegram.api.formatting.fallback",
                    "operation": method.__api_method__,
                },
            )
            return await make_request(bot, fallback)
        except TelegramAPIError as exc:
            self._log_error(method, exc, started_at)
            raise
        logger.debug(
            "Telegram Bot API request completed",
            extra={
                "event": "telegram.api.request_completed",
                "operation": method.__api_method__,
                "duration_ms": round((monotonic() - started_at) * 1000),
            },
        )
        return result

    @staticmethod
    def _normalize_method(method: TelegramMethod) -> TelegramMethod:
        updates = {}
        if isinstance(method, SendMessage) and method.entities:
            updates["parse_mode"] = None
        reply_to_message_id = getattr(method, "reply_to_message_id", None)
        if isinstance(reply_to_message_id, int):
            updates["reply_to_message_id"] = None
            if getattr(method, "reply_parameters", None) is None:
                updates["reply_parameters"] = ReplyParameters(
                    message_id=reply_to_message_id,
                    allow_sending_without_reply=True,
                )
            if "allow_sending_without_reply" in type(method).model_fields:
                updates["allow_sending_without_reply"] = None
        return method.model_copy(update=updates) if updates else method

    @staticmethod
    def _fallback_method(
        bot: Bot,
        method: TelegramMethod,
        exc: TelegramBadRequest,
    ) -> TelegramMethod | None:
        description = exc.message.casefold()
        if not any(marker in description for marker in ("html", "rich", "markdown", "parse", "format", "entity")):
            return None

        if isinstance(method, (SendMessage, EditMessageText)):
            parse_mode = method.parse_mode
            if parse_mode is None:
                return None
            return method.model_copy(update={"parse_mode": None})

        if isinstance(method, SendRichMessage):
            rich_message = method.rich_message
            markdown = rich_message.markdown if isinstance(rich_message, InputRichMessage) else None
            if not isinstance(markdown, str):
                return None
            return SendMessage(
                chat_id=method.chat_id,
                text=markdown,
                parse_mode=None,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
                disable_notification=method.disable_notification,
                reply_parameters=method.reply_parameters,
                reply_markup=method.reply_markup,
            )
        return None

    @staticmethod
    def _log_error(method: TelegramMethod, exc: TelegramAPIError, started_at: float) -> None:
        log_exception(
            logger,
            "Telegram Bot API request failed",
            exc,
            event="telegram.api.response_error",
            level=logging.WARNING,
            operation=method.__api_method__,
            duration_ms=round((monotonic() - started_at) * 1000),
        )


async def download_telegram_file(
    bot: Bot,
    file_id: str,
    destination: Path,
    *,
    max_bytes: int,
) -> int:
    """Stream one Telegram file to a private, bounded local file."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    test_download = getattr(bot, "download_file", None)
    if not isinstance(bot, Bot) and callable(test_download):
        return await test_download(file_id, destination, max_bytes=max_bytes)
    info = await bot.get_file(file_id=file_id, request_timeout=60)
    file_path = info.file_path
    if not isinstance(file_path, str) or not file_path:
        raise TelegramFileDownloadError("Telegram did not return a file path")
    if info.file_size is not None and info.file_size > max_bytes:
        raise TelegramFileTooLargeError("Telegram file exceeds maximum size")

    url = bot.session.api.file_url(bot.token, file_path)
    stream: AsyncIterator[bytes] | None = None
    file_created = False
    total = 0
    started_at = monotonic()
    try:
        stream = bot.session.stream_content(
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
                    raise TelegramFileTooLargeError("Telegram file exceeds maximum size")
                output.write(chunk)
    except TelegramFileTooLargeError:
        if file_created:
            destination.unlink(missing_ok=True)
        logger.warning(
            "Telegram file download exceeded its size limit",
            extra={
                "event": "telegram.file_download.rejected",
                "file_size": total,
                "duration_ms": round((monotonic() - started_at) * 1000),
            },
        )
        raise
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
        raise TelegramFileDownloadError(
            f"Telegram file download failed ({type(exc).__name__})"
        ) from exc
    except Exception:
        if file_created:
            destination.unlink(missing_ok=True)
        raise
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
