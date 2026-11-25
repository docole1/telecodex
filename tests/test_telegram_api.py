from __future__ import annotations

import asyncio
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageText, SendMessage, SendRichMessage
from aiogram.types import ReplyParameters
from python_socks import ProxyType

from telecodex.config import ProxySettings
from telecodex.i18n import TextCatalog
from telecodex.telegram.api import TelegramAPI, TelegramApiError
from telecodex.telegram.pagination import two_column_rows
from telecodex.telegram.user_controller import TelegramUserController


BOT_TOKEN = "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"


class TelegramRichMessageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.api = TelegramAPI(BOT_TOKEN)

    async def asyncTearDown(self) -> None:
        await self.api.close()

    async def test_send_rich_message_passes_markdown_and_reply_metadata(self) -> None:
        with patch.object(self.api.bot, "send_rich_message", new_callable=AsyncMock) as send:
            send.return_value = {"message_id": 12}
            result = await self.api.send_rich_message(
                101,
                "# Result\n\n**Done**",
                reply_to_message_id=55,
                disable_notification=True,
            )

        self.assertEqual(result, {"message_id": 12})
        payload = send.await_args.kwargs
        self.assertEqual(payload["rich_message"], {"markdown": "# Result\n\n**Done**"})
        self.assertEqual(payload["reply_parameters"].message_id, 55)
        self.assertTrue(payload["reply_parameters"].allow_sending_without_reply)
        self.assertTrue(payload["disable_notification"])

    async def test_rich_format_rejection_falls_back_to_plain_send_message(self) -> None:
        rejection = TelegramBadRequest(
            method=SendRichMessage(chat_id=101, rich_message={"markdown": "invalid"}),
            message="can't parse markdown",
        )
        with (
            patch.object(self.api.bot, "send_rich_message", new_callable=AsyncMock) as rich,
            patch.object(self.api.bot, "send_message", new_callable=AsyncMock) as plain,
        ):
            rich.side_effect = rejection
            plain.return_value = {"message_id": 13}
            result = await self.api.send_rich_message(101, "Unparsed **text**")

        self.assertEqual(result, {"message_id": 13})
        plain_payload = plain.await_args.kwargs
        self.assertEqual(plain_payload["text"], "Unparsed **text**")
        self.assertIsNone(plain_payload["parse_mode"])
        self.assertEqual(rich.await_count, 1)

    async def test_non_format_rich_rejection_does_not_retry(self) -> None:
        rejection = TelegramBadRequest(
            method=SendRichMessage(chat_id=101, rich_message={"markdown": "text"}),
            message="chat not found",
        )
        with (
            patch.object(self.api.bot, "send_rich_message", new_callable=AsyncMock) as rich,
            patch.object(self.api.bot, "send_message", new_callable=AsyncMock) as plain,
        ):
            rich.side_effect = rejection
            with self.assertRaises(TelegramApiError):
                await self.api.send_rich_message(101, "No chat")
            plain.assert_not_awaited()


class TelegramHtmlMessageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.api = TelegramAPI(BOT_TOKEN)

    async def asyncTearDown(self) -> None:
        await self.api.close()

    async def test_send_message_uses_html_and_keeps_explicit_entities(self) -> None:
        with patch.object(self.api.bot, "send_message", new_callable=AsyncMock) as send:
            send.return_value = {"message_id": 21}
            await self.api.send_message(101, "<strong>Ready</strong>")
            await self.api.send_message(
                101,
                "output",
                entities=[{"type": "pre", "offset": 0, "length": 6}],
            )

        first, second = [call.kwargs for call in send.await_args_list]
        self.assertEqual(first["parse_mode"], "HTML")
        self.assertEqual(second["entities"], [{"type": "pre", "offset": 0, "length": 6}])
        self.assertIsNone(second["parse_mode"])

    async def test_html_format_rejection_retries_plain_text_for_send_and_edit(self) -> None:
        send_rejection = TelegramBadRequest(
            method=SendMessage(chat_id=101, text="<strong>bad markup</strong>"),
            message="can't parse entities",
        )
        edit_rejection = TelegramBadRequest(
            method=EditMessageText(chat_id=101, message_id=22, text="<strong>bad markup</strong>"),
            message="can't parse entities: Unsupported start tag",
        )
        with (
            patch.object(self.api.bot, "send_message", new_callable=AsyncMock) as send,
            patch.object(self.api.bot, "edit_message_text", new_callable=AsyncMock) as edit,
        ):
            send.side_effect = [send_rejection, {"message_id": 22}]
            edit.side_effect = [edit_rejection, True]
            await self.api.send_message(101, "<strong>bad markup</strong>")
            await self.api.edit_message(101, 22, "<strong>bad markup</strong>")

        self.assertEqual(send.await_args_list[0].kwargs["parse_mode"], "HTML")
        self.assertIsNone(send.await_args_list[1].kwargs["parse_mode"])
        self.assertEqual(edit.await_args_list[0].kwargs["parse_mode"], "HTML")
        self.assertIsNone(edit.await_args_list[1].kwargs["parse_mode"])


class TelegramFileTransferTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.api = TelegramAPI(BOT_TOKEN)
        self.temp_dir = tempfile.TemporaryDirectory()

    async def asyncTearDown(self) -> None:
        await self.api.close()
        self.temp_dir.cleanup()

    async def test_download_is_stream_limited_and_creates_private_file(self) -> None:
        destination = Path(self.temp_dir.name) / "download.txt"

        async def chunks(**_kwargs):
            yield b"safe"

        with (
            patch.object(self.api.bot, "get_file", new_callable=AsyncMock) as get_file,
            patch.object(self.api.bot.session, "stream_content", new=chunks),
        ):
            get_file.return_value = {"file_path": "documents/file.txt", "file_size": 4}
            size = await self.api.download_file("file-id", destination, max_bytes=4)

        self.assertEqual(size, 4)
        self.assertEqual(destination.read_bytes(), b"safe")
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)

    async def test_download_removes_partial_file_when_stream_exceeds_limit(self) -> None:
        destination = Path(self.temp_dir.name) / "too-large.txt"

        async def chunks(**_kwargs):
            yield b"123"

        with (
            patch.object(self.api.bot, "get_file", new_callable=AsyncMock) as get_file,
            patch.object(self.api.bot.session, "stream_content", new=chunks),
        ):
            get_file.return_value = {"file_path": "documents/file.txt", "file_size": 3}
            with self.assertRaises(TelegramApiError):
                await self.api.download_file("file-id", destination, max_bytes=2)

        self.assertFalse(destination.exists())

    async def test_failed_exclusive_create_does_not_remove_existing_file(self) -> None:
        destination = Path(self.temp_dir.name) / "existing.txt"
        destination.write_text("keep me", encoding="utf-8")
        with patch.object(self.api.bot, "get_file", new_callable=AsyncMock) as get_file:
            get_file.return_value = {"file_path": "documents/file.txt", "file_size": 4}
            with self.assertRaises(TelegramApiError):
                await self.api.download_file("file-id", destination, max_bytes=10)

        self.assertEqual(destination.read_text(encoding="utf-8"), "keep me")


class TelegramOffsetBootstrapTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.api = TelegramAPI(BOT_TOKEN)

    async def asyncTearDown(self) -> None:
        await self.api.close()

    async def test_bootstrap_confirms_only_updates_before_saved_offset(self) -> None:
        with patch.object(self.api.bot, "get_updates", new_callable=AsyncMock) as get_updates:
            get_updates.return_value = []
            await self.api.confirm_updates_before(812)

        payload = get_updates.await_args.kwargs
        self.assertEqual(payload["offset"], 812)
        self.assertEqual(payload["timeout"], 0)
        self.assertEqual(payload["allowed_updates"], ["message", "callback_query"])
        self.assertEqual(payload["request_timeout"], 10)

    async def test_bootstrap_skips_empty_initial_offset(self) -> None:
        with patch.object(self.api.bot, "get_updates", new_callable=AsyncMock) as get_updates:
            await self.api.confirm_updates_before(0)
        get_updates.assert_not_awaited()


class TelegramButtonLayoutTests(unittest.TestCase):
    def test_two_column_rows_keep_short_labels_paired_and_stack_long_labels(self) -> None:
        buttons = [
            {"text": "Русский", "callback_data": "l:ru"},
            {"text": "English", "callback_data": "l:en"},
            {"text": "Локальный процесс (STDIO)", "callback_data": "m:stdio"},
            {"text": "Streamable HTTP", "callback_data": "m:http"},
        ]

        rows = two_column_rows(buttons, 0)

        self.assertEqual(
            [[button["text"] for button in row] for row in rows],
            [["Русский", "English"], ["Локальный процесс (STDIO)"], ["Streamable HTTP"]],
        )
        self.assertEqual(
            [button["callback_data"] for row in rows for button in row],
            ["l:ru", "l:en", "m:stdio", "m:http"],
        )

    def test_selected_button_wraps_the_label_and_respects_telegram_limit(self) -> None:
        controller = object.__new__(TelegramUserController)
        controller.texts = TextCatalog()

        self.assertEqual(controller._selected_button_label("Русский", True), "› Русский ‹")
        long_selected = controller._selected_button_label("Название" * 20, True)
        long_unselected = controller._selected_button_label("Название" * 20, False)
        self.assertLessEqual(len(long_selected), 64)
        self.assertLessEqual(len(long_unselected), 64)
        self.assertTrue(long_selected.startswith("› "))
        self.assertTrue(long_selected.endswith(" ‹"))


class TelegramProxyClientTests(unittest.TestCase):
    def test_socks5h_proxy_keeps_remote_dns_and_separate_credentials(self) -> None:
        proxy = ProxySettings("socks5h://127.0.0.1:2080", "proxy-user", "proxy-secret")
        api = TelegramAPI(BOT_TOKEN, proxy)
        try:
            session = api.bot.session
            self.assertEqual(session.proxy[0], "socks5://127.0.0.1:2080")
            self.assertEqual(session._connector_init["rdns"], True)
            self.assertEqual(session._connector_init["username"], "proxy-user")
            self.assertEqual(session._connector_init["password"], "proxy-secret")
            self.assertNotIn("proxy-secret", repr(api))
        finally:
            asyncio.run(api.close())

    def test_https_proxy_uses_tls_to_the_proxy(self) -> None:
        api = TelegramAPI(BOT_TOKEN, ProxySettings("https://127.0.0.1:2080"))
        try:
            session = api.bot.session
            self.assertEqual(session._connector_init["proxy_type"], ProxyType.HTTP)
            self.assertIsNotNone(session._connector_init["proxy_ssl"])
        finally:
            asyncio.run(api.close())

    def test_no_proxy_uses_aiogram_default_connector(self) -> None:
        api = TelegramAPI(BOT_TOKEN)
        try:
            self.assertIsNone(api.bot.session.proxy)
        finally:
            asyncio.run(api.close())
