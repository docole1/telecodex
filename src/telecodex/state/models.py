from __future__ import annotations

from dataclasses import dataclass, field

from telecodex.i18n import DEFAULT_LANGUAGE


@dataclass
class PendingTurnState:
    """Persisted metadata for a Codex turn that has not finished delivery."""

    thread_id: str
    turn_id: str | None
    prompt_message_id: int | None
    started_at: int
    phase: str = "starting"
    result_status: str | None = None
    error: str | None = None
    error_key: str | None = None
    prompt_message_ids: list[int] = field(default_factory=list)


@dataclass
class RuntimeState:
    """Per-user or per-group selection state for one Codex context."""

    thread_id: str | None = None
    cwd: str | None = None
    selected_model: str | None = None
    default_permission_mode: str | None = None
    language: str = DEFAULT_LANGUAGE
    selected_effort: str | None = None
    selected_service_tier: str | None = None
    selected_memory_mode: str | None = None
    pending_skill_name: str | None = None
    pending_skill_path: str | None = None
    pending_turn: PendingTurnState | None = None
    output_mode: str = "detailed"
    quiet_progress: bool = False


@dataclass
class ThreadRuntimeState:
    """State shared by every Telegram context that selects one Codex thread."""

    permission_mode: str | None = None
    memory_mode: str | None = None
    managed_by_telecodex: bool = False
    token_usage: dict[str, int | None] | None = None
