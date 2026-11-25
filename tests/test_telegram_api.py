from __future__ import annotations

import asyncio
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageText, SendChatAction, SendDocument, SendMessage, SendRichMessage
from aiogram.types import File, InputRichMessage, LinkPreviewOptions
from python_socks import ProxyType

from telecodex.config import ProxySettings
from telecodex.telegram.api import (
    TelegramFileStorageError,
    TelegramFileTooLargeError,
    TelegramFormattingFallbackMiddleware,
    create_telegram_bot,
    download_telegram_file,
)

BOT_TOKEN = "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"


class TelegramFormattingFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bot = create_telegram_bot(BOT_TOKEN)
        self.middleware = TelegramFormattingFallbackMiddleware()

    async def asyncTearDown(self) -> None:
        await self.bot.session.close()

    async def test_html_send_and_edit_retry_without_parse_mode(self) -> None:
        send = SendMessage(chat_id=101, text="<strong>bad</strong>", parse_mode="HTML")
        edit = EditMessageText(chat_id=101, message_id=22, text="<strong>bad</strong>", parse_mode="HTML")
        send_rejection = TelegramBadRequest(
            method=send,
            message="can't parse entities",
        )
        edit_rejection = TelegramBadRequest(
            method=edit,
            message="can't parse entities: Unsupported start tag",
        )
        calls: list[SendMessage | EditMessageText] = []

        async def make_request(_bot: Bot, method):
            calls.append(method)
            if isinstance(method, SendMessage) and method.parse_mode is not None:
                raise send_rejection
            if isinstance(method, EditMessageText) and method.parse_mode is not None:
                raise edit_rejection
            return True

        await self.middleware(make_request, self.bot, send)
        await self.middleware(make_request, self.bot, edit)

        self.assertEqual(len(calls), 4)
        self.assertIsNone(calls[1].parse_mode)
        self.assertIsNone(calls[3].parse_mode)

    async def test_rich_rejection_retries_as_plain_message_and_keeps_reply_markup(self) -> None:
        markup = {"inline_keyboard": [[{"text": "Retry", "callback_data": "retry"}]]}
        rich = SendRichMessage(
            chat_id=101,
            rich_message=InputRichMessage(markdown="**unparsed**"),
            reply_markup=markup,
        )
        rejection = TelegramBadRequest(method=rich, message="can't parse rich message markdown")
        calls = []

        async def make_request(_bot: Bot, method):
            calls.append(method)
            if len(calls) == 1:
                raise rejection
            return True

        await self.middleware(make_request, self.bot, rich)

        self.assertIsInstance(calls[1], SendMessage)
        self.assertEqual(calls[1].text, "**unparsed**")
        self.assertIsNone(calls[1].parse_mode)
        self.assertEqual(calls[1].reply_markup.inline_keyboard[0][0].callback_data, "retry")
        self.assertEqual(calls[1].link_preview_options, LinkPreviewOptions(is_disabled=True))

    async def test_non_formatting_bad_request_does_not_retry(self) -> None:
        send = SendMessage(chat_id=101, text="hello", parse_mode="HTML")
        rejection = TelegramBadRequest(method=send, message="chat not found")
        calls = []

        async def make_request(_bot: Bot, method):
            calls.append(method)
            raise rejection

        with self.assertRaises(TelegramBadRequest):
            await self.middleware(make_request, self.bot, send)
        self.assertEqual(calls, [send])

    async def test_explicit_entities_suppress_default_html_and_legacy_reply_fields(self) -> None:
        send = SendMessage(
            chat_id=101,
            text="already formatted",
            parse_mode="HTML",
            entities=[{"type": "bold", "offset": 0, "length": 7}],
            reply_to_message_id=55,
        )
        normalized = self.middleware._normalize_method(send)

        self.assertIsNone(normalized.parse_mode)
        self.assertIsNone(normalized.reply_to_message_id)
        self.assertEqual(normalized.reply_parameters.message_id, 55)
        self.assertTrue(normalized.reply_parameters.allow_sending_without_reply)


class TelegramFileTransferTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bot = create_telegram_bot(BOT_TOKEN)
        self.temp_dir = tempfile.TemporaryDirectory()

    async def asyncTearDown(self) -> None:
        await self.bot.session.close()
        self.temp_dir.cleanup()

    def _file_info(self, size: int) -> File:
        return File(
            file_id="file-id",
            file_unique_id="unique-file-id",
            file_path="documents/file.txt",
            file_size=size,
        )

    async def test_download_is_size_limited_and_creates_private_file(self) -> None:
        destination = Path(self.temp_dir.name) / "download.txt"

        async def chunks(**_kwargs):
            yield b"safe"

        with (
            patch.object(self.bot, "get_file", new_callable=AsyncMock, return_value=self._file_info(4)),
            patch.object(self.bot.session, "stream_content", new=chunks),
        ):
            size = await download_telegram_file(self.bot, "file-id", destination, max_bytes=4)

        self.assertEqual(size, 4)
        self.assertEqual(destination.read_bytes(), b"safe")
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)

    async def test_download_removes_partial_file_when_stream_exceeds_limit(self) -> None:
        destination = Path(self.temp_dir.name) / "too-large.txt"

        async def chunks(**_kwargs):
            yield b"123"

        with (
            patch.object(self.bot, "get_file", new_callable=AsyncMock, return_value=self._file_info(3)),
            patch.object(self.bot.session, "stream_content", new=chunks),
        ):
            with self.assertRaises(TelegramFileTooLargeError):
                await download_telegram_file(self.bot, "file-id", destination, max_bytes=2)

        self.assertFalse(destination.exists())

    async def test_download_does_not_remove_existing_destination(self) -> None:
        destination = Path(self.temp_dir.name) / "existing.txt"
        destination.write_text("keep me", encoding="utf-8")
        with patch.object(
            self.bot,
            "get_file",
            new_callable=AsyncMock,
            return_value=self._file_info(4),
        ):
            with self.assertRaises(TelegramFileStorageError):
                await download_telegram_file(self.bot, "file-id", destination, max_bytes=10)
        self.assertEqual(destination.read_text(encoding="utf-8"), "keep me")


class TelegramBotConfigurationTests(unittest.TestCase):
    def test_bot_defaults_to_html_and_disables_link_previews(self) -> None:
        bot = create_telegram_bot(BOT_TOKEN)
        try:
            self.assertEqual(bot.default.parse_mode, ParseMode.HTML)
            self.assertTrue(bot.default.link_preview_is_disabled)
        finally:
            asyncio.run(bot.session.close())

    def test_socks5h_proxy_keeps_remote_dns_and_separate_credentials(self) -> None:
        proxy = ProxySettings("socks5h://127.0.0.1:2080", "proxy-user", "proxy-secret")
        bot = create_telegram_bot(BOT_TOKEN, proxy)
        try:
            session = bot.session
            self.assertEqual(session.proxy[0], "socks5://127.0.0.1:2080")
            self.assertEqual(session._connector_init["rdns"], True)
            self.assertEqual(session._connector_init["username"], "proxy-user")
            self.assertEqual(session._connector_init["password"], "proxy-secret")
            self.assertNotIn("proxy-secret", repr(bot))
        finally:
            asyncio.run(bot.session.close())

    def test_https_proxy_uses_tls_to_the_proxy(self) -> None:
        bot = create_telegram_bot(BOT_TOKEN, ProxySettings("https://127.0.0.1:2080"))
        try:
            self.assertEqual(bot.session._connector_init["proxy_type"], ProxyType.HTTP)
            self.assertIsNotNone(bot.session._connector_init["proxy_ssl"])
        finally:
            asyncio.run(bot.session.close())

    def test_no_proxy_uses_aiogram_default_connector(self) -> None:
        bot = create_telegram_bot(BOT_TOKEN)
        try:
            self.assertIsNone(bot.session.proxy)
        finally:
            asyncio.run(bot.session.close())


class TelegramRequestTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_message_action_and_document_timeouts_are_preserved(self) -> None:
        bot = create_telegram_bot(BOT_TOKEN)
        try:
            with patch.object(
                AiohttpSession,
                "make_request",
                new_callable=AsyncMock,
                return_value=True,
            ) as make_request:
                await bot.session.make_request(bot, SendMessage(chat_id=101, text="message"))
                await bot.session.make_request(bot, SendChatAction(chat_id=101, action="typing"))
                await bot.session.make_request(bot, SendDocument(chat_id=101, document="file-id"))

            self.assertEqual(
                [call.kwargs["timeout"] for call in make_request.await_args_list],
                [60, 2, 180],
            )
        finally:
            await bot.session.close()
