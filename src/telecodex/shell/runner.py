from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from telecodex.logging_config import log_exception

MAX_CAPTURE_BYTES = 64 * 1024
logger = logging.getLogger(__name__)


class EmptyShellCommandError(ValueError):
    """Raised when a shell request has no command text."""


@dataclass(frozen=True)
class ShellResult:
    command: str
    return_code: int | None
    output: str
    timed_out: bool
    output_truncated: bool


class ShellRunner:
    def __init__(self, working_directory: Path, timeout_seconds: int = 60) -> None:
        self.working_directory = working_directory
        self.timeout_seconds = timeout_seconds

    def resolve_working_directory(self, working_directory: Path | None = None) -> Path:
        root = self.working_directory.expanduser().resolve()
        cwd = (working_directory or root).expanduser().resolve()
        try:
            cwd.relative_to(root)
        except ValueError as exc:
            raise OSError("Shell working directory is outside the configured project root") from exc
        if not cwd.is_dir():
            raise OSError("Shell working directory is unavailable")
        return cwd

    async def run(
        self,
        command: str,
        *,
        working_directory: Path | None = None,
    ) -> ShellResult:
        if not command.strip():
            raise EmptyShellCommandError
        cwd = self.resolve_working_directory(working_directory)
        started_at = monotonic()
        env = build_shell_environment()
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except Exception as exc:
            log_exception(
                logger,
                "Could not start host shell command",
                exc,
                event="shell.host.start_failed",
                timeout_seconds=self.timeout_seconds,
            )
            raise
        timed_out = False
        captured = bytearray()
        total_bytes = 0

        async def drain_output() -> None:
            nonlocal total_bytes
            assert process.stdout is not None
            while chunk := await process.stdout.read(8192):
                total_bytes += len(chunk)
                remaining = MAX_CAPTURE_BYTES - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])

        reader_task = asyncio.create_task(drain_output())
        wait_task = asyncio.create_task(process.wait())
        _done, pending = await asyncio.wait(
            {reader_task, wait_task}, timeout=self.timeout_seconds
        )
        if pending:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await asyncio.gather(reader_task, wait_task, return_exceptions=True)
        result = ShellResult(
            command=command,
            return_code=process.returncode,
            output=captured.decode("utf-8", errors="replace"),
            timed_out=timed_out,
            output_truncated=total_bytes > len(captured),
        )
        logger.info(
            "Host shell command completed",
            extra={
                "event": "shell.host.completed",
                "return_code": process.returncode,
                "timeout_seconds": self.timeout_seconds,
                "duration_ms": round((monotonic() - started_at) * 1000),
                "bytes": total_bytes,
                "output_truncated": result.output_truncated,
                "outcome": "timeout" if timed_out else "completed",
            },
        )
        return result


def build_shell_environment() -> dict[str, str]:
    """Return an inherited shell environment without TeleCodex credentials."""
    env = os.environ.copy()
    for variable in (
        "TELEGRAM_BOT_TOKEN",
        "ACCESS_TOKEN",
        "CODEX_ACCESS_TOKEN",
        "OPENAI_API_KEY",
    ):
        env.pop(variable, None)
    return env
