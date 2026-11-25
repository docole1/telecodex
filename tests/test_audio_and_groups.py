from __future__ import annotations
from pathlib import Path
from typing import Any
from unittest.mock import patch
import tempfile
import unittest
from aiogram.types import Message
from telecodex.audio.transcriber import AudioTranscriptionError, LocalAudioTranscriber
from telecodex.config import AppConfig
from telecodex.documents.models import TelegramAttachment
from telecodex.group_triggers import parse_trigger_words, strip_group_trigger
from telecodex.telegram.updates import parse_update
from telecodex.telegram.user_controller import TelegramUserController

from multi_user_test_support import (
    dispatch_user_update,
    callback_event,
    message_event,
    CODEX_TRIGGER_CYRILLIC,
    FakeAudioTranscriber,
    FakeCodex,
    FakeTelegram,
    KAWAII_TRIGGER_CYRILLIC,
    _catalog_text,
)

class TelegramAudioParsingTests(unittest.TestCase):
    def test_voice_message_is_parsed_with_audio_metadata(self) -> None:
        incoming = parse_update(
            {
                "update_id": 1,
                "message": {
                    "message_id": 2,
                    "from": {"id": 101},
                    "chat": {"id": 101, "type": "private"},
                    "voice": {
                        "file_id": "voice-file",
                        "file_unique_id": "voice-unique",
                        "duration": 12,
                        "mime_type": "audio/ogg",
                        "file_size": 2048,
                    },
                },
            }
        )

        self.assertIsInstance(incoming.event, Message)  # type: ignore[union-attr]
        self.assertEqual(incoming.attachments[0].kind, "voice")  # type: ignore[union-attr]
        self.assertEqual(incoming.attachments[0].duration_seconds, 12)  # type: ignore[union-attr]
        self.assertEqual(incoming.attachments[0].file_name, "voice.ogg")  # type: ignore[union-attr]

    def test_audio_document_is_classified_as_audio(self) -> None:
        incoming = parse_update(
            {
                "update_id": 1,
                "message": {
                    "message_id": 2,
                    "from": {"id": 101},
                    "chat": {"id": 101, "type": "private"},
                    "document": {
                        "file_id": "audio-file",
                        "file_name": "request.mp3",
                        "mime_type": "audio/mpeg",
                        "file_size": 4096,
                    },
                },
            }
        )

        self.assertIsInstance(incoming.event, Message)  # type: ignore[union-attr]
        self.assertEqual(incoming.attachments[0].kind, "audio")  # type: ignore[union-attr]


class LocalAudioTranscriberTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcriber_joins_segments_and_reuses_loaded_model(self) -> None:
        class Segment:
            def __init__(self, text: str) -> None:
                self.text = text

        class Model:
            calls = 0

            def transcribe(self, *_args: Any, **_kwargs: Any) -> tuple[list[Segment], object]:
                self.calls += 1
                return [Segment(" Hello"), Segment(" world ")], object()

        transcriber = LocalAudioTranscriber(max_duration_seconds=30)
        model = Model()
        transcriber._model = model
        with patch("telecodex.audio.transcriber._audio_duration_seconds", return_value=5):
            first = await transcriber.transcribe(Path("voice.ogg"), duration_hint=5)
            second = await transcriber.transcribe(Path("voice.ogg"), duration_hint=5)

        self.assertEqual(first, "Hello world")
        self.assertEqual(second, "Hello world")
        self.assertEqual(model.calls, 2)

    async def test_transcriber_rejects_overlong_audio_before_loading_model(self) -> None:
        transcriber = LocalAudioTranscriber(max_duration_seconds=30)

        with self.assertRaises(AudioTranscriptionError) as raised:
            await transcriber.transcribe(Path("voice.ogg"), duration_hint=31)

        self.assertEqual(raised.exception.text_key, "audio.label.too_long")
        self.assertIsNone(transcriber._model)


class GroupTriggerParsingTests(unittest.TestCase):
    def test_words_are_trimmed_casefolded_and_deduplicated(self) -> None:
        self.assertEqual(
            parse_trigger_words(
                f" {CODEX_TRIGGER_CYRILLIC}, {KAWAII_TRIGGER_CYRILLIC}, {CODEX_TRIGGER_CYRILLIC.upper()} ,,"
            ),
            (CODEX_TRIGGER_CYRILLIC, KAWAII_TRIGGER_CYRILLIC),
        )

    def test_trigger_matches_only_as_the_first_complete_word(self) -> None:
        words = parse_trigger_words(f"{CODEX_TRIGGER_CYRILLIC}, {KAWAII_TRIGGER_CYRILLIC}")
        self.assertEqual(
            strip_group_trigger(f"{KAWAII_TRIGGER_CYRILLIC.capitalize()}, help with a project", words),
            "help with a project",
        )
        self.assertEqual(
            strip_group_trigger(f"{CODEX_TRIGGER_CYRILLIC}: run the tests", words),
            "run the tests",
        )
        self.assertIsNone(strip_group_trigger(f"hello, {KAWAII_TRIGGER_CYRILLIC} check", words))
        self.assertIsNone(strip_group_trigger(f"{CODEX_TRIGGER_CYRILLIC}project", words))

    def test_empty_or_invalid_trigger_lists_are_rejected(self) -> None:
        for value in ("", " , , ", "two words", "word, bad!"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_trigger_words(value)


class AudioMessageHandlerTests(unittest.IsolatedAsyncioTestCase):
    def _controller(
        self,
        workspace: Path,
        transcriber: FakeAudioTranscriber,
        *,
        is_group: bool = False,
    ) -> tuple[TelegramUserController, FakeTelegram, FakeCodex]:
        config = AppConfig(
            telegram_bot_token="token",
            telegram_allowed_user_ids=(101,),
            workspace_root=workspace,
            codex_bin="codex",
            codex_model=None,
            network_access=False,
            approval_policy="on-request",
            shell_timeout_seconds=60,
            database_file=workspace / "state.sqlite3",
        )
        telegram = FakeTelegram()
        codex = FakeCodex()
        codex.working_directory = workspace
        controller = TelegramUserController(
            config,
            telegram,
            object(),
            -100900 if is_group else 101,
            is_group=is_group,
            group_trigger_words=lambda: (CODEX_TRIGGER_CYRILLIC,),
            audio_transcriber=transcriber,  # type: ignore[arg-type]
        )
        controller.bind_codex(codex)  # type: ignore[arg-type]
        return controller, telegram, codex

    def _voice(self, *, duration: int = 5) -> TelegramAttachment:
        return TelegramAttachment(
            file_id="voice-file",
            kind="voice",
            file_size=2048,
            file_name="voice.ogg",
            mime_type="audio/ogg",
            duration_seconds=duration,
        )

    async def test_voice_is_transcribed_echoed_and_sent_as_codex_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcriber = FakeAudioTranscriber("check configuration")
            controller, telegram, codex = self._controller(workspace, transcriber)

            await dispatch_user_update(controller,
                message_event(
                    1,
                    22,
                    101,
                    101,
                    "private",
                    "",
                    attachments=(self._voice(),),
                )
            )
            await controller._finish_typing()

            self.assertEqual(codex.messages, ["check configuration"])
            self.assertEqual(transcriber.calls[0][1], 5)
            self.assertIn("check configuration", telegram.messages[0][1])
            self.assertEqual(telegram.replies[0], 22)
            self.assertFalse(list((workspace / ".telecodex" / "attachments" / "101").glob("*voice.ogg")))

    async def test_audio_over_duration_limit_is_rejected_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcriber = FakeAudioTranscriber("must not be transcribed")
            controller, telegram, codex = self._controller(workspace, transcriber)

            await dispatch_user_update(controller,
                message_event(
                    1,
                    22,
                    101,
                    101,
                    "private",
                    "",
                    attachments=(self._voice(duration=301),),
                )
            )

            self.assertEqual(transcriber.calls, [])
            self.assertEqual(codex.messages, [])
            self.assertIn(_catalog_text("audio.label.too_long", seconds=300), telegram.messages[0][1])

    async def test_group_voice_uses_transcribed_trigger_and_ignores_unaddressed_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcript = f"{CODEX_TRIGGER_CYRILLIC.capitalize()}, check configuration"
            transcriber = FakeAudioTranscriber(transcript)
            controller, telegram, codex = self._controller(workspace, transcriber, is_group=True)

            await dispatch_user_update(controller,
                message_event(
                    1,
                    23,
                    101,
                    -100900,
                    "supergroup",
                    "",
                    attachments=(self._voice(),),
                )
            )
            await controller._finish_typing()

            self.assertEqual(codex.messages, ["check configuration"])
            self.assertEqual(len(transcriber.calls), 1)
            self.assertIn(transcript, telegram.messages[0][1])

            quiet_transcriber = FakeAudioTranscriber("check configuration")
            quiet_workspace = workspace / "quiet"
            quiet_workspace.mkdir()
            quiet_controller, quiet_telegram, quiet_codex = self._controller(
                quiet_workspace,
                quiet_transcriber,
                is_group=True,
            )
            await dispatch_user_update(quiet_controller,
                message_event(
                    2,
                    24,
                    101,
                    -100900,
                    "supergroup",
                    "",
                    attachments=(self._voice(),),
                )
            )
            await quiet_controller._finish_typing()

            self.assertEqual(quiet_codex.messages, [])
            self.assertEqual(quiet_telegram.messages, [])

    async def test_group_reply_to_bot_bypasses_trigger_word(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, _telegram, codex = self._controller(
                Path(temp_dir), FakeAudioTranscriber(), is_group=True
            )
            await dispatch_user_update(controller,
                message_event(
                    1,
                    22,
                    101,
                    -100900,
                    "supergroup",
                    "without a trigger word",
                    reply_to_user_id=999,
                )
            )
            self.assertEqual(codex.messages, ["without a trigger word"])

    async def test_group_message_without_trigger_or_bot_reply_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, _telegram, codex = self._controller(
                Path(temp_dir), FakeAudioTranscriber(), is_group=True
            )
            await dispatch_user_update(controller,
                message_event(1, 22, 101, -100900, "supergroup", "ordinary text")
            )
            self.assertEqual(codex.messages, [])
