from __future__ import annotations

import unittest

from aiogram.types import Message, MessageEntity

from multi_user_test_support import message_event

from telecodex.group_triggers import parse_trigger_words, strip_group_trigger_with_offset
from telecodex.telegram.markdown_input import to_model_markdown
from telecodex.telegram.updates import command_argument_markdown, parse_update, without_message_prefix


class TelegramMarkdownConversionTests(unittest.TestCase):
    def test_plain_text_without_entities_is_returned_unchanged(self) -> None:
        self.assertEqual(to_model_markdown("Keep **literal** Markdown"), "Keep **literal** Markdown")

    def test_bold_offsets_after_emoji_use_utf16_units(self) -> None:
        text = "😀 Bold after emoji"
        entity = MessageEntity(type="bold", offset=3, length=4)

        self.assertEqual(to_model_markdown(text, [entity]), "😀 **Bold** after emoji")

    def test_inline_and_preformatted_code_keep_fences(self) -> None:
        inline = to_model_markdown(
            "use a`b now",
            [MessageEntity(type="code", offset=4, length=3)],
        )
        block = to_model_markdown(
            'print("ok")',
            [MessageEntity(type="pre", offset=0, length=11, language="python")],
        )

        self.assertEqual(inline, "use ``a`b`` now")
        self.assertEqual(block, '```python\nprint("ok")\n```')

    def test_text_link_preserves_visible_label_and_target(self) -> None:
        text = "Read this page"
        entity = MessageEntity(
            type="text_link",
            offset=5,
            length=4,
            url="https://example.test/a(b)",
        )

        self.assertEqual(
            to_model_markdown(text, [entity]),
            "Read [this](https://example.test/a(b\\)) page",
        )


class TelegramIncomingMarkdownTests(unittest.TestCase):
    def test_parse_update_reconstructs_message_entities_and_keeps_raw_text(self) -> None:
        incoming = parse_update(
            {
                "update_id": 1,
                "message": {
                    "message_id": 2,
                    "from": {"id": 101},
                    "chat": {"id": 101, "type": "private"},
                    "text": "Start the task",
                    "entities": [{"type": "bold", "offset": 10, "length": 3}],
                },
            }
        )

        self.assertIsInstance(incoming, Message)
        self.assertEqual(incoming.text, "Start the task")  # type: ignore[union-attr]
        self.assertEqual(incoming.prompt_text, "Start the **tas**k")  # type: ignore[union-attr]

    def test_caption_entities_are_used_when_message_has_media(self) -> None:
        incoming = parse_update(
            {
                "update_id": 1,
                "message": {
                    "message_id": 2,
                    "from": {"id": 101},
                    "chat": {"id": 101, "type": "private"},
                    "caption": "Review this",
                    "caption_entities": [{"type": "italic", "offset": 7, "length": 4}],
                    "document": {"file_id": "file", "file_name": "notes.txt"},
                },
            }
        )

        self.assertIsInstance(incoming, Message)
        self.assertEqual(incoming.prompt_text, "Review *this*")  # type: ignore[union-attr]

    def test_command_argument_keeps_formatting_without_changing_command_parse(self) -> None:
        incoming = parse_update(
            {
                "update_id": 1,
                "message": {
                    "message_id": 2,
                    "from": {"id": 101},
                    "chat": {"id": 101, "type": "private"},
                    "text": "/plan Write a careful plan",
                    "entities": [{"type": "bold", "offset": 6, "length": 5}],
                },
            }
        )

        self.assertIsInstance(incoming, Message)
        self.assertEqual(incoming.text.partition(" ")[0], "/plan")  # type: ignore[union-attr]
        self.assertEqual(command_argument_markdown(incoming), "**Write** a careful plan")  # type: ignore[arg-type]

    def test_group_trigger_removal_rebases_entities(self) -> None:
        trigger = "codex"
        text = "codex build this"
        stripped = strip_group_trigger_with_offset(text, parse_trigger_words(trigger))
        self.assertEqual(stripped, ("build this", 6))

        incoming = message_event(
            update_id=1,
            message_id=2,
            user_id=101,
            chat_id=-100123,
            chat_type="supergroup",
            text=text,
            entities=({"type": "bold", "offset": 6, "length": 5},),
        )
        without_trigger = without_message_prefix(incoming, stripped[1])  # type: ignore[arg-type,index]

        self.assertEqual(without_trigger.text, "build this")
        self.assertEqual(without_trigger.prompt_text, "**build** this")
