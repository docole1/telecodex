from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path
from typing import Any

from telecodex.logging_config import log_exception

logger = logging.getLogger(__name__)


class AudioTranscriptionError(RuntimeError):
    def __init__(self, text_key: str) -> None:
        super().__init__(text_key)
        self.text_key = text_key


class LocalAudioTranscriber:
    """Load faster-whisper lazily and serialize CPU inference across all chats."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        max_duration_seconds: int = 300,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.max_duration_seconds = max_duration_seconds
        self._model: Any = None
        self._model_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    @property
    def status(self) -> str:
        if self._model is not None:
            return "ready"
        try:
            available = importlib.util.find_spec("faster_whisper") is not None
        except (ImportError, ValueError):
            available = False
        return "configured" if available else "missing"

    async def transcribe(
        self,
        audio_path: Path,
        *,
        duration_hint: int | None = None,
    ) -> str:
        if duration_hint is not None and duration_hint > self.max_duration_seconds:
            raise AudioTranscriptionError("audio.label.too_long")

        try:
            duration = await asyncio.to_thread(_audio_duration_seconds, audio_path)
        except Exception as exc:
            log_exception(
                logger,
                "Could not inspect Telegram audio duration",
                exc,
                event="audio.duration.inspect_failed",
                level=logging.INFO,
                file_type=audio_path.suffix.lower().lstrip("."),
            )
            raise AudioTranscriptionError("audio.label.invalid_file") from exc
        if duration is not None and duration > self.max_duration_seconds:
            raise AudioTranscriptionError("audio.label.too_long")

        model = await self._get_model()
        try:
            async with self._inference_lock:
                transcript = await asyncio.to_thread(_transcribe_sync, model, audio_path)
        except Exception as exc:
            log_exception(
                logger,
                "Could not transcribe Telegram audio",
                exc,
                event="audio.transcription.failed",
                level=logging.WARNING,
                backend="faster-whisper",
                model=self.model_name,
                duration_seconds=round(duration) if duration is not None else duration_hint,
                file_type=audio_path.suffix.lower().lstrip("."),
            )
            raise AudioTranscriptionError("audio.label.transcription_failed") from exc
        if not transcript:
            raise AudioTranscriptionError("audio.label.empty")
        return transcript

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is None:
                try:
                    self._model = await asyncio.to_thread(self._load_model_sync)
                except ModuleNotFoundError as exc:
                    logger.warning(
                        "Local audio transcription dependency is missing",
                        extra={
                            "event": "audio.model_load.missing_dependency",
                            "error_type": type(exc).__name__,
                            "backend": "faster-whisper",
                            "model": self.model_name,
                        },
                    )
                    raise AudioTranscriptionError("audio.label.transcriber_missing") from exc
                except Exception as exc:
                    log_exception(
                        logger,
                        "Could not load local transcription model",
                        exc,
                        event="audio.model_load.failed",
                        level=logging.WARNING,
                        backend="faster-whisper",
                        model=self.model_name,
                        device=self.device,
                        compute_type=self.compute_type,
                    )
                    raise AudioTranscriptionError("audio.label.model_load_failed") from exc
        return self._model

    def _load_model_sync(self) -> Any:
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )


def _audio_duration_seconds(audio_path: Path) -> float | None:
    import av

    with av.open(str(audio_path)) as container:
        if container.duration is not None:
            return container.duration / av.time_base
        durations = [
            stream.duration * stream.time_base
            for stream in container.streams
            if stream.duration is not None and stream.time_base is not None
        ]
        return max(durations) if durations else None


def _transcribe_sync(model: Any, audio_path: Path) -> str:
    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
