"""In-memory state for Telegram inline-picker messages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telecodex.codex.session import (
        CodexBackgroundTerminal,
        CodexMemorySettings,
        McpServer,
        CodexModel,
        CodexSkill,
        CodexThread,
    )
    from telecodex.workspace_files import WorkspaceEntry


@dataclass
class ModelPicker:
    token: str
    models: list[CodexModel]
    page: int
    current_model: str | None = None
    current_effort: str | None = None
    message_id: int | None = None
    effort_model_index: int | None = None


@dataclass
class LanguagePicker:
    token: str
    message_id: int | None = None


@dataclass
class ModePicker:
    token: str
    message_id: int | None = None


@dataclass
class GroupModePicker:
    token: str
    message_id: int | None = None


@dataclass
class SkillPicker:
    token: str
    skills: list[CodexSkill]
    action: str = "menu"
    page: int = 0
    message_id: int | None = None


@dataclass
class MemoryPicker:
    token: str
    thread_id: str | None
    settings: CodexMemorySettings
    use_memories: bool | None
    confirming_generation: bool = False
    confirming_reset: bool = False
    message_id: int | None = None


@dataclass
class DirectoryPicker:
    token: str
    current_path: Path
    directories: list[Path]
    page: int = 0
    message_id: int | None = None


@dataclass
class FilesPicker:
    token: str
    owner_user_id: int
    current_path: Path
    entries: list[WorkspaceEntry]
    page: int = 0
    message_id: int | None = None
    selected_index: int | None = None


@dataclass
class ChatsPicker:
    token: str
    threads: list[CodexThread]
    page: int = 0
    message_id: int | None = None
    selected_index: int | None = None
    blocked_thread_index: int | None = None


@dataclass
class PermissionsPicker:
    token: str
    message_id: int | None = None


@dataclass
class DeleteConfirmation:
    token: str
    thread_id: str
    message_id: int | None = None


@dataclass
class ProcessesPicker:
    token: str
    thread_id: str
    terminals: list[CodexBackgroundTerminal]
    page: int = 0
    message_id: int | None = None
    selected_process_id: str | None = None


@dataclass
class McpPicker:
    token: str
    servers: list[McpServer]
    page: int = 0
    message_id: int | None = None
    selected_name: str | None = None
    view: str = "servers"
    tools_page: int = 0
    verbose: bool = False
    confirming_remove: bool = False
    confirming_action: str | None = None


@dataclass
class McpAddWizard:
    owner_user_id: int
    stage: str
    transport: str | None = None
    name: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env_vars: tuple[str, ...] = ()
    url: str | None = None
    auth: str = "none"
    bearer_token_env_var: str | None = None
