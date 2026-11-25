from __future__ import annotations
from unittest.mock import AsyncMock, patch
import unittest
from telecodex.config import ProxySettings
from telecodex.i18n import TextCatalog
from telecodex.telegram.api import TelegramAPI, TelegramApiError
from telecodex.telegram.pagination import two_column_rows
from telecodex.telegram.user_controller import TelegramUserController

class TelegramRichMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_rich_message_passes_markdown_and_reply_metadata(self) -> None:
        api = TelegramAPI("bot-token")
        try:
            with patch.object(api, "call", new_callable=AsyncMock) as call:
                call.return_value = {"message_id": 12}
                result = await api.send_rich_message(
                    101,
                    "# Result\n\n**Done**",
                    reply_to_message_id=55,
                    disable_notification=True,
                )

            self.assertEqual(result, {"message_id": 12})
            method, payload = call.await_args.args
            self.assertEqual(method, "sendRichMessage")
            self.assertEqual(payload["rich_message"], {"markdown": "# Result\n\n**Done**"})
            self.assertEqual(
                payload["reply_parameters"],
                {"message_id": 55, "allow_sending_without_reply": True},
            )
            self.assertTrue(payload["disable_notification"])
        finally:
            await api.close()

    async def test_rich_format_rejection_falls_back_to_plain_send_message(self) -> None:
        api = TelegramAPI("bot-token")
        rejection = TelegramApiError(
            "Telegram API sendRichMessage failed: Bad Request: can't parse markdown",
            method="sendRichMessage",
            error_code=400,
            http_status=400,
        )
        try:
            with patch.object(api, "call", new_callable=AsyncMock) as call:
                call.side_effect = [rejection, {"message_id": 13}]
                result = await api.send_rich_message(101, "Unparsed **text**")

            self.assertEqual(result, {"message_id": 13})
            self.assertEqual(
                [call.await_args_list[index].args[0] for index in range(2)],
                ["sendRichMessage", "sendMessage"],
            )
            self.assertEqual(call.await_args_list[1].args[1]["text"], "Unparsed **text**")
            self.assertNotIn("parse_mode", call.await_args_list[1].args[1])
        finally:
            await api.close()

    async def test_non_format_rich_rejection_does_not_retry(self) -> None:
        api = TelegramAPI("bot-token")
        rejection = TelegramApiError(
            "Telegram API sendRichMessage failed: Bad Request: chat not found",
            method="sendRichMessage",
            error_code=400,
            http_status=400,
        )
        try:
            with patch.object(api, "call", new_callable=AsyncMock) as call:
                call.side_effect = rejection
                with self.assertRaises(TelegramApiError):
                    await api.send_rich_message(101, "No chat")
            call.assert_awaited_once()
        finally:
            await api.close()


class TelegramHtmlMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_message_uses_html_and_keeps_explicit_entities(self) -> None:
        api = TelegramAPI("bot-token")
        try:
            with patch.object(api, "call", new_callable=AsyncMock) as call:
                call.return_value = {"message_id": 21}
                await api.send_message(101, "<strong>Ready</strong>")
                await api.send_message(
                    101,
                    "output",
                    entities=[{"type": "pre", "offset": 0, "length": 6}],
                )

            self.assertEqual(call.await_args_list[0].args[0], "sendMessage")
            self.assertEqual(call.await_args_list[0].args[1]["parse_mode"], "HTML")
            entity_payload = call.await_args_list[1].args[1]
            self.assertEqual(entity_payload["entities"], [{"type": "pre", "offset": 0, "length": 6}])
            self.assertNotIn("parse_mode", entity_payload)
        finally:
            await api.close()

    async def test_html_format_rejection_retries_plain_text_for_send_and_edit(self) -> None:
        api = TelegramAPI("bot-token")
        send_rejection = TelegramApiError(
            "Telegram API sendMessage failed: Bad Request: can't parse entities",
            method="sendMessage",
            error_code=400,
        )
        edit_rejection = TelegramApiError(
            "Telegram API editMessageText failed: Bad Request: can't parse entities: Unsupported start tag",
            method="editMessageText",
            error_code=400,
        )
        try:
            with patch.object(api, "call", new_callable=AsyncMock) as call:
                call.side_effect = [send_rejection, {"message_id": 22}, edit_rejection, True]
                await api.send_message(101, "<strong>bad markup</strong>")
                await api.edit_message(101, 22, "<strong>bad markup</strong>")

            self.assertEqual(call.await_args_list[0].args[1]["parse_mode"], "HTML")
            self.assertNotIn("parse_mode", call.await_args_list[1].args[1])
            self.assertEqual(call.await_args_list[2].args[1]["parse_mode"], "HTML")
            self.assertNotIn("parse_mode", call.await_args_list[3].args[1])
        finally:
            await api.close()


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
    def test_api_client_uses_configured_proxy_auth_without_leaking_password(self) -> None:
        proxy = ProxySettings("socks5h://127.0.0.1:2080", "proxy-user", "proxy-secret")
        with patch("telecodex.telegram.api.httpx.AsyncClient") as async_client:
            TelegramAPI("bot-token", proxy)

        client_options = async_client.call_args.kwargs
        configured_proxy = client_options["proxy"]
        self.assertEqual(str(configured_proxy.url), "socks5h://127.0.0.1:2080")
        self.assertEqual(configured_proxy.auth, ("proxy-user", "proxy-secret"))
        self.assertNotIn("proxy-secret", repr(configured_proxy))

    def test_api_client_without_proxy_preserves_httpx_default(self) -> None:
        with patch("telecodex.telegram.api.httpx.AsyncClient") as async_client:
            TelegramAPI("bot-token")

        self.assertNotIn("proxy", async_client.call_args.kwargs)
