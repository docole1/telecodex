from __future__ import annotations

import asyncio
import codecs
import errno
import fcntl
import logging
import os
import pty
import shutil
import signal
import struct
import sys
import termios
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Awaitable, Callable

from telecodex.logging_config import log_exception
from telecodex.shell.runner import MAX_CAPTURE_BYTES, ShellRunner, build_shell_environment

logger = logging.getLogger(__name__)

_READ_CHUNK_BYTES = 8192
_OUTPUT_FLUSH_INTERVAL_SECONDS = 0.8
_OUTPUT_QUEUE_LIMIT = 64
_SESSION_IDLE_TIMEOUT_SECONDS = 30 * 60
_SESSION_START_TIMEOUT_SECONDS = 10
_INTERRUPT_GRACE_SECONDS = 5
_MAX_INPUT_BYTES = 16 * 1024


@dataclass(frozen=True)
class ShellOutput:
    text: str
    return_code: int | None = None
    timed_out: bool = False
    truncated: bool = False
    partial: bool = False
    previous_output_sent: bool = False
    reply_to_message_id: int | None = None


OutputHandler = Callable[[ShellOutput], Awaitable[None]]
ClosedHandler = Callable[["ShellSession", str], Awaitable[None]]


class ShellSession:
    """A persistent Bash process attached to a PTY for one Telegram context."""

    def __init__(
        self,
        runner: ShellRunner,
        working_directory: Path,
        timeout_seconds: int,
        on_output: OutputHandler,
        on_closed: ClosedHandler,
        log_fields: dict[str, Any] | None = None,
    ) -> None:
        self.runner = runner
        self.working_directory = working_directory
        self.timeout_seconds = timeout_seconds
        self.on_output = on_output
        self.on_closed = on_closed
        self.log_fields = log_fields or {}

        self._master_fd: int | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._delivery_task: asyncio.Task[None] | None = None
        self._maintenance_task: asyncio.Task[None] | None = None
        self._startup_future: asyncio.Future[None] | None = None
        self._output_queue: asyncio.Queue[ShellOutput | None] = asyncio.Queue(
            maxsize=_OUTPUT_QUEUE_LIMIT
        )
        self._write_lock = asyncio.Lock()
        self._capture_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._raw_buffer = bytearray()
        self._output_buffer = bytearray()
        self._output_bytes = 0
        self._output_truncated = False
        self._partial_output_sent = False
        self._command_reply_to: int | None = None
        self._command_started_at: float | None = None
        self._command_timed_out = False
        self._interrupt_sent_at: float | None = None
        self._at_prompt = False
        self._initialized = False
        self._closing = False
        self._closed = False
        self._cleanup_complete = False
        self._last_activity_at = monotonic()
        self._close_notified = False
        self._marker_prefix = b""
        self._decoder: codecs.IncrementalDecoder | None = None

    @property
    def is_open(self) -> bool:
        return not self._closing and not self._closed and self._process is not None

    async def start(self) -> None:
        cwd = self.runner.resolve_working_directory(self.working_directory)
        bash = shutil.which("bash")
        if bash is None:
            raise OSError("Bash is not installed")

        master_fd, slave_fd = pty.openpty()
        self._master_fd = master_fd
        rc_read_fd: int | None = None
        rc_write_fd: int | None = None
        try:
            attributes = termios.tcgetattr(slave_fd)
            attributes[3] &= ~(termios.ECHO | termios.ECHONL)
            termios.tcsetattr(slave_fd, termios.TCSANOW, attributes)
            fcntl.ioctl(
                slave_fd,
                termios.TIOCSWINSZ,
                struct.pack("HHHH", 24, 160, 0, 0),
            )
            os.set_blocking(master_fd, False)

            token = uuid.uuid4().hex
            marker = f"TC_PROMPT_{token}:".encode("ascii")
            self._marker_prefix = b"\x1e" + marker
            env = build_shell_environment()
            env.pop("BASH_ENV", None)
            env["TERM"] = "dumb"
            env["NO_COLOR"] = "1"
            env["PAGER"] = "cat"
            env["GIT_PAGER"] = "cat"
            env["HISTFILE"] = "/dev/null"
            rc_read_fd, rc_write_fd = os.pipe()
            rc_source = _bash_rc_source(token).encode("utf-8")
            os.write(rc_write_fd, rc_source)
            os.close(rc_write_fd)
            rc_write_fd = None

            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "telecodex.shell.pty_helper",
                "--",
                bash,
                "--noprofile",
                "--rcfile",
                f"/dev/fd/{rc_read_fd}",
                "-i",
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=cwd,
                env=env,
                start_new_session=True,
                pass_fds=(rc_read_fd,),
            )
        except Exception:
            os.close(master_fd)
            self._master_fd = None
            raise
        finally:
            os.close(slave_fd)
            if rc_read_fd is not None:
                os.close(rc_read_fd)
            if rc_write_fd is not None:
                os.close(rc_write_fd)

        loop = asyncio.get_running_loop()
        self._startup_future = loop.create_future()
        self._decoder = _new_decoder()
        self._reader_task = asyncio.create_task(self._reader_loop())
        self._delivery_task = asyncio.create_task(self._delivery_loop())
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        try:
            await asyncio.wait_for(
                asyncio.shield(self._startup_future),
                timeout=_SESSION_START_TIMEOUT_SECONDS,
            )
        except Exception:
            await self.close()
            raise

        logger.info(
            "Interactive host shell started",
            extra={
                "event": "shell.session.started",
                "outcome": "started",
                **self.log_fields,
            },
        )

    async def send_input(self, text: str, message_id: int | None = None) -> None:
        if not self.is_open:
            raise RuntimeError("Interactive shell session is closed")
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) > _MAX_INPUT_BYTES:
            raise ValueError("Interactive shell input is too long")

        async with self._write_lock:
            if self._at_prompt:
                async with self._capture_lock:
                    self._reset_command_capture()
                self._command_reply_to = message_id
                self._command_started_at = monotonic()
                self._command_timed_out = False
                self._interrupt_sent_at = None
                self._at_prompt = False
            self._last_activity_at = monotonic()
            await self._write_to_pty(encoded + b"\n")

        logger.info(
            "Interactive shell input accepted",
            extra={
                "event": "shell.session.input.accepted",
                "input_bytes": len(encoded),
                **self.log_fields,
            },
        )

    async def interrupt(self) -> bool:
        if not self.is_open or self._at_prompt:
            return False
        async with self._write_lock:
            try:
                await self._write_to_pty(b"\x03")
            except (OSError, RuntimeError):
                return False
        logger.info(
            "Interactive shell foreground process interrupted",
            extra={"event": "shell.session.interrupted", **self.log_fields},
        )
        return True

    async def close(self) -> None:
        async with self._close_lock:
            if self._cleanup_complete:
                return
            self._closing = True
            current_task = asyncio.current_task()
            if self._maintenance_task is not None and self._maintenance_task is not current_task:
                self._maintenance_task.cancel()
                await asyncio.gather(self._maintenance_task, return_exceptions=True)

            process = self._process
            if process is not None and process.returncode is None:
                self._signal_session(signal.SIGHUP)
                try:
                    await asyncio.wait_for(process.wait(), timeout=1.5)
                except asyncio.TimeoutError:
                    self._signal_session(signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=1.5)
                    except asyncio.TimeoutError:
                        self._signal_session(signal.SIGKILL)
                        await asyncio.gather(process.wait(), return_exceptions=True)

            self._close_master()
            reader_task = self._reader_task
            if reader_task is not None and reader_task is not current_task and not reader_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(reader_task), timeout=3)
                except asyncio.TimeoutError:
                    reader_task.cancel()
                    await asyncio.gather(reader_task, return_exceptions=True)

            if reader_task is None:
                await self._queue_delivery_stop()
            if self._delivery_task is not None and self._delivery_task is not current_task:
                if not self._delivery_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(self._delivery_task), timeout=5)
                    except asyncio.TimeoutError:
                        self._delivery_task.cancel()
                        await asyncio.gather(self._delivery_task, return_exceptions=True)

            self._closed = True
            self._cleanup_complete = True
            logger.info(
                "Interactive host shell stopped",
                extra={
                    "event": "shell.session.stopped",
                    "return_code": process.returncode if process is not None else None,
                    **self.log_fields,
                },
            )

    async def _write_to_pty(self, data: bytes) -> None:
        fd = self._master_fd
        if fd is None:
            raise RuntimeError("Interactive shell PTY is closed")
        view = memoryview(data)
        loop = asyncio.get_running_loop()
        while view:
            try:
                written = os.write(fd, view)
            except BlockingIOError:
                await _wait_for_fd(loop, fd, writable=True)
                continue
            if written <= 0:
                raise OSError("Could not write to interactive shell PTY")
            view = view[written:]

    async def _read_from_pty(self) -> bytes:
        fd = self._master_fd
        if fd is None:
            return b""
        loop = asyncio.get_running_loop()
        await _wait_for_fd(loop, fd, writable=False)
        try:
            return os.read(fd, _READ_CHUNK_BYTES)
        except OSError as exc:
            if exc.errno == errno.EIO:
                return b""
            raise

    async def _reader_loop(self) -> None:
        reason = "process_exit"
        try:
            while not self._closing:
                chunk = await self._read_from_pty()
                if not chunk:
                    break
                self._last_activity_at = monotonic()
                self._raw_buffer.extend(chunk)
                await self._parse_raw_buffer()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = "reader_error"
            log_exception(
                logger,
                "Interactive shell PTY reader failed",
                exc,
                event="shell.session.reader_failed",
                **self.log_fields,
            )
        finally:
            self._close_master()
            self._closed = True
            if self._startup_future is not None and not self._startup_future.done():
                self._startup_future.set_exception(
                    RuntimeError("Interactive shell exited before becoming ready")
                )
            if not self._closing:
                await self._queue_final_output(None, timed_out=self._command_timed_out)
            await self._queue_delivery_stop()
            if not self._closing:
                if self._initialized:
                    asyncio.create_task(self._notify_closed(reason))

    async def _parse_raw_buffer(self) -> None:
        while True:
            marker_start = self._raw_buffer.find(self._marker_prefix)
            if marker_start < 0:
                safe_length = max(0, len(self._raw_buffer) - len(self._marker_prefix) + 1)
                if safe_length:
                    output = bytes(self._raw_buffer[:safe_length])
                    del self._raw_buffer[:safe_length]
                    await self._record_output(output)
                return

            if marker_start:
                output = bytes(self._raw_buffer[:marker_start])
                del self._raw_buffer[:marker_start]
                await self._record_output(output)

            marker_end = self._raw_buffer.find(b"\x1f", len(self._marker_prefix))
            if marker_end < 0:
                return
            status_bytes = bytes(self._raw_buffer[len(self._marker_prefix) : marker_end])
            del self._raw_buffer[: marker_end + 1]
            try:
                return_code = int(status_bytes.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                await self._record_output(self._marker_prefix + status_bytes + b"\x1f")
                continue
            await self._handle_prompt(return_code)

    async def _record_output(self, data: bytes) -> None:
        if not data:
            return
        async with self._capture_lock:
            self._output_bytes += len(data)
            remaining = max(0, MAX_CAPTURE_BYTES - (self._output_bytes - len(data)))
            if remaining:
                self._output_buffer.extend(data[:remaining])
            if len(data) > remaining:
                self._output_truncated = True
                self._last_activity_at = monotonic()

    async def _handle_prompt(self, return_code: int) -> None:
        self._last_activity_at = monotonic()
        if not self._initialized:
            self._initialized = True
            self._at_prompt = True
            if self._startup_future is not None and not self._startup_future.done():
                self._startup_future.set_result(None)
            async with self._capture_lock:
                self._reset_command_capture()
            return

        await self._queue_final_output(return_code, timed_out=self._command_timed_out)
        self._at_prompt = True
        self._command_started_at = None
        self._interrupt_sent_at = None
        self._command_timed_out = False

    async def _queue_final_output(self, return_code: int | None, *, timed_out: bool) -> None:
        async with self._capture_lock:
            decoder = self._decoder
            text = decoder.decode(bytes(self._output_buffer), final=True) if decoder else ""
            self._output_buffer.clear()
            event = ShellOutput(
                text=text,
                return_code=return_code,
                timed_out=timed_out,
                truncated=self._output_truncated,
                partial=False,
                previous_output_sent=self._partial_output_sent,
                reply_to_message_id=self._command_reply_to,
            )
            self._decoder = _new_decoder()
            self._output_bytes = 0
            self._output_truncated = False
            self._partial_output_sent = False
            self._command_reply_to = None
        if self._initialized:
            await self._output_queue.put(event)

    async def _flush_partial_output(self) -> None:
        async with self._capture_lock:
            if not self._output_buffer:
                return
            decoder = self._decoder
            text = decoder.decode(bytes(self._output_buffer), final=False) if decoder else ""
            self._output_buffer.clear()
            if not text:
                return
            self._partial_output_sent = True
            event = ShellOutput(
                text=text,
                partial=True,
                reply_to_message_id=self._command_reply_to,
            )
        await self._output_queue.put(event)

    async def _delivery_loop(self) -> None:
        while True:
            event = await self._output_queue.get()
            if event is None:
                return
            try:
                await self.on_output(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not deliver interactive shell output",
                    exc,
                    event="shell.session.output_delivery_failed",
                    **self.log_fields,
                )

    async def _maintenance_loop(self) -> None:
        while not self._closing and not self._closed:
            await asyncio.sleep(_OUTPUT_FLUSH_INTERVAL_SECONDS)
            await self._flush_partial_output()
            now = monotonic()
            if self._at_prompt and now - self._last_activity_at >= _SESSION_IDLE_TIMEOUT_SECONDS:
                self._closing = True
                await self.close()
                await self._notify_closed("idle_timeout")
                return
            if (
                not self._at_prompt
                and self._command_started_at is not None
                and self.timeout_seconds > 0
                and now - self._command_started_at >= self.timeout_seconds
                and not self._command_timed_out
            ):
                self._command_timed_out = True
                self._interrupt_sent_at = now
                await self.interrupt()
            elif (
                self._command_timed_out
                and self._interrupt_sent_at is not None
                and now - self._interrupt_sent_at >= _INTERRUPT_GRACE_SECONDS
            ):
                self._signal_foreground(signal.SIGKILL)
                self._interrupt_sent_at = now

    async def _notify_closed(self, reason: str) -> None:
        if self._close_notified:
            return
        self._close_notified = True
        try:
            await self.on_closed(self, reason)
        except Exception as exc:
            log_exception(
                logger,
                "Could not notify Telegram context about shell closure",
                exc,
                event="shell.session.close_notification_failed",
                reason=reason,
                **self.log_fields,
            )

    async def _queue_delivery_stop(self) -> None:
        try:
            self._output_queue.put_nowait(None)
        except asyncio.QueueFull:
            await self._output_queue.put(None)

    def _reset_command_capture(self) -> None:
        self._output_buffer.clear()
        self._output_bytes = 0
        self._output_truncated = False
        self._partial_output_sent = False
        self._decoder = _new_decoder()

    def _signal_foreground(self, sig: int) -> None:
        fd = self._master_fd
        if fd is None:
            return
        try:
            foreground_pgid = os.tcgetpgrp(fd)
            if self._process is not None and foreground_pgid != self._process.pid:
                os.killpg(foreground_pgid, sig)
        except OSError:
            return

    def _signal_session(self, sig: int) -> None:
        process = self._process
        if process is None:
            return
        session_id = process.pid
        process_groups: set[int] = set()
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                stat = stat_path.read_text()
                command_end = stat.rfind(")")
                fields = stat[command_end + 1 :].split()
                process_group = int(fields[2])
                current_session = int(fields[3])
            except (OSError, ValueError, IndexError):
                continue
            if current_session == session_id:
                process_groups.add(process_group)
        process_groups.add(session_id)
        for process_group in process_groups:
            self._signal_group(process_group, sig)

    @staticmethod
    def _signal_group(process_id: int, sig: int) -> None:
        try:
            os.killpg(process_id, sig)
        except OSError:
            pass

    def _close_master(self) -> None:
        fd = self._master_fd
        self._master_fd = None
        if fd is not None:
            try:
                asyncio.get_running_loop().remove_reader(fd)
            except RuntimeError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass


async def _wait_for_fd(
    loop: asyncio.AbstractEventLoop,
    fd: int,
    *,
    writable: bool,
) -> None:
    future: asyncio.Future[None] = loop.create_future()

    def ready() -> None:
        if not future.done():
            future.set_result(None)

    add_reader = loop.add_writer if writable else loop.add_reader
    remove_reader = loop.remove_writer if writable else loop.remove_reader
    add_reader(fd, ready)
    try:
        await future
    finally:
        remove_reader(fd)


def _new_decoder() -> codecs.IncrementalDecoder:
    return codecs.getincrementaldecoder("utf-8")("replace")


def _bash_rc_source(token: str) -> str:
    return """if [[ -r \"$HOME/.bashrc\" ]]; then
    source \"$HOME/.bashrc\"
fi
__telecodex_prompt_declaration=\"$(declare -p PROMPT_COMMAND 2>/dev/null || true)\"
if [[ $__telecodex_prompt_declaration == \"declare -a\"* ]]; then
    __telecodex_user_prompt_command=(\"${PROMPT_COMMAND[@]}\")
elif [[ -n ${PROMPT_COMMAND-} ]]; then
    __telecodex_user_prompt_command=(\"${PROMPT_COMMAND}\")
else
    __telecodex_user_prompt_command=()
fi
unset __telecodex_prompt_declaration
HISTFILE=/dev/null
PS1=\"\"
PROMPT_COMMAND='__telecodex_status=$?; for __telecodex_prompt in \"${__telecodex_user_prompt_command[@]}\"; do eval \"$__telecodex_prompt\"; done; printf \"\\036TC_PROMPT___TC_TOKEN__:%s\\037\" \"$__telecodex_status\"; unset __telecodex_status __telecodex_prompt'
""".replace("__TC_TOKEN__", token)
