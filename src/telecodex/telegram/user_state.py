"""Per-user pending interactions and queued Telegram activity state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from telecodex.state.models import NewChatDefaults


@dataclass
class PendingApproval:
    future: asyncio.Future[str]
    message_id: int
    chat_id: int


@dataclass
class PendingUserInput:
    future: asyncio.Future[dict[str, Any]]
    questions: list[dict[str, Any]]


@dataclass(frozen=True)
class RunningCommand:
    message_id: int | None
    started_at: float
    command: str


@dataclass(frozen=True)
class PendingGoalEdit:
    thread_id: str


@dataclass(frozen=True)
class PendingThreadSwitch:
    token: str
    message_id: int
    action: str
    path: Path | None = None
    thread_id: str | None = None
    label: str | None = None
    picker_message_id: int | None = None
    picker_token: str | None = None
    picker_index: int | None = None
    source_thread_id: str | None = None
    new_chat_defaults: NewChatDefaults | None = None


@dataclass(frozen=True)
class PendingStaleTurnAction:
    message_id: int
    thread_id: str
    turn_id: str | None


@dataclass
class PendingChatRecovery:
    token: str
    message_id: int
    thread_id: str
    stage: str
    terminal_ids: tuple[str, ...] = ()
    confirm_interrupt: bool = False


@dataclass
class GroupActivityItem:
    text: str
    entities: list[dict[str, Any]]
    reply_to_message_id: int | None
    batchable: bool = True
    delivery: asyncio.Future[dict[str, Any]] | None = None
    disable_notification: bool = False
    rich_markdown: bool = False


@dataclass(slots=True)
class QueuedCompactionInput:
    chat_id: int
    text: str
    prompt_message_id: int
    additional_input_items: list[dict[str, Any]] | None = None
