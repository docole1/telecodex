from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from telecodex.codex.mcp import validate_env_name


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def default_state_dir() -> Path:
    return _path_from_env("XDG_STATE_HOME", "~/.local/state") / "telecodex"


@dataclass(frozen=True)
class ProxySettings:
    url: str
    username: str | None = field(default=None, repr=False)
    password: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class AppConfig:
    telegram_bot_token: str
    telegram_allowed_user_ids: tuple[int, ...]
    workspace_root: Path
    codex_bin: str
    codex_model: str | None
    network_access: bool
    approval_policy: str
    shell_timeout_seconds: int
    database_file: Path
    migration_state_file: Path | None = None
    telegram_proxy: ProxySettings | None = None
    transcription_model: str = "small"
    transcription_device: str = "cpu"
    transcription_compute_type: str = "int8"
    max_audio_duration_seconds: int = 300
    mcp_env_vars: tuple[str, ...] = ()


def state_paths_from_env() -> tuple[Path, Path]:
    """Return SQLite destination and legacy JSON import source paths."""
    state_dir = default_state_dir()
    configured_legacy_state = os.environ.get("TELECODEX_STATE_FILE", "").strip()
    migration_state_file = (
        Path(configured_legacy_state).expanduser().resolve()
        if configured_legacy_state
        else state_dir / "state.json"
    )
    configured_database = os.environ.get("TELECODEX_DATABASE_FILE", "").strip()
    if configured_database:
        database_file = Path(configured_database).expanduser().resolve()
    elif configured_legacy_state:
        database_file = migration_state_file.with_suffix(".sqlite3")
    else:
        database_file = state_dir / "state.sqlite3"
    if database_file == migration_state_file:
        raise ValueError("TELECODEX_DATABASE_FILE must differ from TELECODEX_STATE_FILE")
    return database_file, migration_state_file


def load_config() -> AppConfig:
    """Load and validate settings required to run the Telegram bot."""
    load_dotenv()

    required = ("TELEGRAM_BOT_TOKEN",)
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}")

    configured_user_ids = _parse_telegram_user_ids(
        os.environ.get("TELEGRAM_ALLOWED_USER_IDS", ""), "TELEGRAM_ALLOWED_USER_IDS"
    )
    if not configured_user_ids:
        raise ValueError("Set TELEGRAM_ALLOWED_USER_IDS to one or more positive Telegram user IDs")
    allowed_user_ids = tuple(dict.fromkeys(configured_user_ids))

    workspace_root = _path_from_env("CODEX_WORKSPACE_ROOT", "/root")
    if not workspace_root.is_dir():
        raise ValueError(f"CODEX_WORKSPACE_ROOT is not a directory: {workspace_root}")

    codex_bin = os.environ.get("CODEX_BIN") or shutil.which("codex")
    if not codex_bin:
        raise ValueError("Codex CLI was not found; set CODEX_BIN to its executable path")

    try:
        shell_timeout = int(os.environ.get("SHELL_TIMEOUT_SECONDS", "60"))
    except ValueError as exc:
        raise ValueError("SHELL_TIMEOUT_SECONDS must be an integer") from exc
    if shell_timeout < 1:
        raise ValueError("SHELL_TIMEOUT_SECONDS must be greater than zero")

    approval_policy = os.environ.get("CODEX_APPROVAL_POLICY", "on-request")
    if approval_policy not in {"never", "on-request", "on-failure", "untrusted"}:
        raise ValueError("CODEX_APPROVAL_POLICY must be never, on-request, on-failure, or untrusted")

    network_access = os.environ.get("CODEX_NETWORK_ACCESS", "false").strip().lower()
    if network_access not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError("CODEX_NETWORK_ACCESS must be true or false")

    database_file, migration_state_file = state_paths_from_env()

    model = os.environ.get("CODEX_MODEL", "").strip() or None
    mcp_env_vars = _parse_mcp_env_vars(os.environ.get("TELECODEX_MCP_ENV_VARS", ""))
    telegram_proxy = _proxy_settings_from_env("TELEGRAM_PROXY")
    transcription_model = os.environ.get("TELECODEX_TRANSCRIPTION_MODEL", "small").strip()
    if not transcription_model:
        raise ValueError("TELECODEX_TRANSCRIPTION_MODEL must not be empty")
    transcription_device = os.environ.get("TELECODEX_TRANSCRIPTION_DEVICE", "cpu").strip().lower()
    if transcription_device not in {"cpu", "cuda"}:
        raise ValueError("TELECODEX_TRANSCRIPTION_DEVICE must be cpu or cuda")
    transcription_compute_type = os.environ.get(
        "TELECODEX_TRANSCRIPTION_COMPUTE_TYPE", "int8"
    ).strip().lower()
    if transcription_compute_type not in {
        "int8",
        "int8_float16",
        "int8_bfloat16",
        "float16",
        "bfloat16",
        "float32",
    }:
        raise ValueError("TELECODEX_TRANSCRIPTION_COMPUTE_TYPE is invalid")
    try:
        max_audio_duration_seconds = int(
            os.environ.get("TELECODEX_AUDIO_MAX_DURATION_SECONDS", "300")
        )
    except ValueError as exc:
        raise ValueError("TELECODEX_AUDIO_MAX_DURATION_SECONDS must be an integer") from exc
    if not 1 <= max_audio_duration_seconds <= 3600:
        raise ValueError("TELECODEX_AUDIO_MAX_DURATION_SECONDS must be between 1 and 3600")
    return AppConfig(
        telegram_bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
        telegram_allowed_user_ids=allowed_user_ids,
        workspace_root=workspace_root,
        codex_bin=codex_bin,
        codex_model=model,
        network_access=network_access in {"true", "1", "yes"},
        approval_policy=approval_policy,
        shell_timeout_seconds=shell_timeout,
        database_file=database_file,
        migration_state_file=migration_state_file,
        telegram_proxy=telegram_proxy,
        transcription_model=transcription_model,
        transcription_device=transcription_device,
        transcription_compute_type=transcription_compute_type,
        max_audio_duration_seconds=max_audio_duration_seconds,
        mcp_env_vars=mcp_env_vars,
    )


def _parse_mcp_env_vars(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    parsed: list[str] = []
    for value in raw.split(","):
        name = value.strip()
        if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(
                "TELECODEX_MCP_ENV_VARS must contain comma-separated environment variable names"
            )
        try:
            validate_env_name(name)
        except ValueError as exc:
            raise ValueError(
                "TELECODEX_MCP_ENV_VARS cannot include TeleCodex, Telegram, or Codex variables"
            ) from exc
        if name not in parsed:
            parsed.append(name)
        if len(parsed) > 32:
            raise ValueError("TELECODEX_MCP_ENV_VARS can contain at most 32 names")
    return tuple(parsed)


def _proxy_settings_from_env(prefix: str) -> ProxySettings | None:
    url_name = f"{prefix}_URL"
    username_name = f"{prefix}_USERNAME"
    password_name = f"{prefix}_PASSWORD"
    url = os.environ.get(url_name, "").strip()
    username = os.environ.get(username_name, "").strip()
    password = os.environ.get(password_name, "")

    if not url:
        if username or password:
            raise ValueError(f"{username_name} and {password_name} require {url_name}")
        return None

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(
            f"{url_name} must be a valid proxy URL with a host and port"
        ) from exc
    if (
        parsed.scheme.lower() not in {"http", "https", "socks5", "socks5h"}
        or not parsed.hostname
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"{url_name} must use http://, https://, socks5://, or socks5h:// "
            "with a host and port; put credentials in separate variables"
        )
    if password and not username:
        raise ValueError(f"{password_name} requires {username_name}")

    return ProxySettings(
        url=url,
        username=username or None,
        password=password if username else None,
    )


def _parse_telegram_user_ids(raw: str, setting: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    parsed: list[int] = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            raise ValueError(f"{setting} must contain comma-separated positive integers")
        try:
            user_id = int(value)
        except ValueError as exc:
            raise ValueError(f"{setting} must contain comma-separated positive integers") from exc
        if user_id <= 0:
            raise ValueError(f"{setting} must contain positive Telegram user IDs")
        parsed.append(user_id)
    return tuple(parsed)
