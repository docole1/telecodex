from __future__ import annotations

import shlex

TELEGRAM_TEXT_LIMIT = 3900


def truncate(text: str, limit: int = 1200, suffix: str = "") -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    if not suffix:
        return text[:limit]
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def split_plain_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Split plain text without adding formatting markers or changing its contents."""
    if limit < 1:
        raise ValueError("limit must be positive")
    remaining = text.strip()
    chunks: list[str] = []
    while len(remaining) > limit:
        split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def format_command(command: object) -> str | None:
    if isinstance(command, str):
        value = command.strip()
        if not value:
            return None
        try:
            parts = shlex.split(value)
        except ValueError:
            return value
        return _unwrap_shell_command(parts) or value
    if isinstance(command, list) and all(isinstance(part, str) for part in command):
        if not command:
            return None
        return _unwrap_shell_command(command) or shlex.join(command)
    return None


def _unwrap_shell_command(parts: list[str]) -> str | None:
    if len(parts) < 3:
        return None
    shell = parts[0].rsplit("/", 1)[-1]
    if shell not in {"bash", "sh", "zsh", "dash", "ksh", "fish"}:
        return None
    options = parts[1]
    if options != "--command" and not (options.startswith("-") and "c" in options[1:]):
        return None
    command = parts[-1].strip()
    return command or None
