"""Shared value objects, callback types, and errors for Codex sessions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


StatusCallback = Callable[[str, object | None, str | None], Awaitable[None]]
CommandFinishedCallback = Callable[[dict[str, Any]], Awaitable[None]]
FinishedCallback = Callable[[dict[str, Any]], Awaitable[None]]
PromptCallback = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
AssistantMessageCallback = Callable[[str, str], Awaitable[None]]


@dataclass(frozen=True)
class CodexServiceTier:
    id: str
    name: str
    description: str


@dataclass(frozen=True)
class CodexModel:
    model: str
    display_name: str
    is_default: bool
    supported_reasoning_efforts: tuple[str, ...]
    default_reasoning_effort: str | None
    service_tiers: tuple[CodexServiceTier, ...] = ()
    default_service_tier: str | None = None


@dataclass(frozen=True)
class CodexSkill:
    name: str
    description: str
    path: str
    enabled: bool
    display_name: str | None = None


@dataclass(frozen=True)
class CodexMemorySettings:
    feature_enabled: bool
    use_memories: bool
    generate_memories: bool
    user_config_version: str | None


class PendingTurnDeliveryError(RuntimeError):
    """A previous Codex result could not be delivered, so a new turn was not started."""


class ActiveThreadWriterError(RuntimeError):
    """A Codex thread is currently owned by another app-server client."""


class CodexChatBusyError(RuntimeError):
    """The active Codex chat is busy with an operation that cannot be steered."""


class CollaborationModeUnavailableError(RuntimeError):
    """The connected app-server does not expose a requested collaboration mode."""


class SelectedSkillUnavailableError(RuntimeError):
    """A previously selected skill no longer appears in the active workspace."""


class FastTierUnavailableError(RuntimeError):
    """The current Codex model does not advertise a Fast service tier."""


class MemoryFeatureUnavailableError(RuntimeError):
    """Local Codex memories are disabled in the user's Codex Home configuration."""


class GoalUnavailableError(RuntimeError):
    """The connected Codex app-server does not expose native Goals."""


class GoalNotFoundError(RuntimeError):
    """The selected Codex thread does not have a Goal to control."""


@dataclass(frozen=True)
class CodexGoal:
    objective: str
    status: str
    token_budget: int | None
    tokens_used: int
    time_used_seconds: int


@dataclass(frozen=True)
class CodexThread:
    thread_id: str
    name: str | None
    preview: str | None
    cwd: Path
    updated_at: int | float | None
    source: str | None

    @property
    def label(self) -> str:
        title = self.name or self.preview or self.cwd.name or str(self.cwd)
        return " ".join(title.split())


@dataclass(frozen=True)
class CodexBackgroundTerminal:
    process_id: str
    command: str
    cwd: str | None
    os_pid: int | None
    cpu_percent: float | None
    rss_kb: int | None


@dataclass(frozen=True)
class CodexRecoverySnapshot:
    thread_id: str
    thread_status: str | None
    active_turn_id: str | None
    terminals: tuple[CodexBackgroundTerminal, ...]
    compaction_pending: bool = False

    @property
    def has_active_work(self) -> bool:
        return (
            self.thread_status != "idle"
            or self.active_turn_id is not None
            or self.compaction_pending
        )

    @property
    def requires_confirmation(self) -> bool:
        return self.has_active_work or bool(self.terminals)
