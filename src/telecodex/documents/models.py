from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TelegramAttachment:
    file_id: str
    kind: str
    file_size: int | None = None
    file_name: str | None = None
    mime_type: str | None = None
    duration_seconds: int | None = None


@dataclass(frozen=True)
class PreparedAttachment:
    name: str
    original_path: str
    context_path: str
    is_image: bool
    is_audio: bool = False
