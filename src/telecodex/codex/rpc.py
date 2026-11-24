from __future__ import annotations

import asyncio
import itertools
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any

from telecodex.logging_config import isolated_log_context, log_identifier, redact_text

logger = logging.getLogger(__name__)
RPC_READ_CHUNK_BYTES = 64 * 1024
RPC_MAX_LINE_BYTES = 64 * 1024 * 1024
STDERR_TAIL_BYTES = 8192
STREAM_DELTA_METHODS = {"item/agentMessage/delta", "item/commandExecution/outputDelta"}
LIFECYCLE_RPC_METHODS = {
    "thread/read",
    "thread/resume",
    "thread/start",
    "thread/turns/list",
    "thread/unsubscribe",
    "turn/interrupt",
    "turn/start",
    "turn/steer",
}
LIFECYCLE_EVENT_METHODS = {
    "transport/closed",
    "turn/aborted",
    "turn/completed",
    "turn/started",
}

EventHandler = Callable[[dict[str, Any]], None]
ServerRequestHandler = Callable[[str, Any, dict[str, Any]], Awaitable[dict[str, Any]]]


class RpcError(RuntimeError):
    def __init__(self, code: int | str, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class JsonRpcProcess:
    """Bidirectional JSONL RPC transport for `codex app-server --listen stdio://`."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        on_event: EventHandler,
        on_server_request: ServerRequestHandler,
    ) -> None:
        self.process = process
        self.on_event = on_event
        self.on_server_request = on_server_request
        self._request_ids = itertools.count(1)
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._write_lock = asyncio.Lock()
        self._closed = False
        self._stderr_tail = bytearray()
        self._stderr_total_bytes = 0
        with isolated_log_context(correlation_id=f"codex-process-{uuid.uuid4().hex[:10]}"):
            self._reader_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._drain_stderr())

    @classmethod
    async def start(
        cls,
        command: list[str],
        env: dict[str, str],
        on_event: EventHandler,
        on_server_request: ServerRequestHandler,
    ) -> "JsonRpcProcess":
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                limit=1024 * 1024,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start Codex app-server: {exc}") from exc
        return cls(process, on_event, on_server_request)

    @property
    def is_running(self) -> bool:
        return (
            not self._closed
            and self.process.returncode is None
            and not self._reader_task.done()
        )

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._write({"method": method, "params": params or {}})

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_seconds: float = 120,
        *,
        params_as_null: bool = False,
    ) -> dict[str, Any]:
        if not self.is_running:
            raise RuntimeError("Codex app-server is not running")
        request_id = f"telecodex-{next(self._request_ids)}"
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        started_at = monotonic()
        lifecycle_request = method in LIFECYCLE_RPC_METHODS
        log_request = logger.info if lifecycle_request else logger.debug
        log_request(
            "Codex RPC request started",
            extra={
                "event": "codex.rpc.request.started",
                "rpc_method": method,
                "request_id": request_id,
                "timeout_seconds": timeout_seconds,
                "pending_count": len(self._pending),
                **self._lifecycle_request_fields(method, params),
            },
        )
        try:
            await self._write({
                "id": request_id,
                "method": method,
                "params": None if params_as_null else params or {},
            })
            response = await asyncio.wait_for(future, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(
                "Codex RPC request timed out",
                extra={
                    "event": "codex.rpc.request.timeout",
                    "rpc_method": method,
                    "request_id": request_id,
                    "timeout_seconds": timeout_seconds,
                    "duration_ms": round((monotonic() - started_at) * 1000),
                    "pending_count": len(self._pending),
                    "outcome": "timeout",
                    **self._lifecycle_request_fields(method, params),
                },
            )
            raise
        except Exception as exc:
            logger.exception(
                "Codex RPC transport request failed",
                extra={
                    "event": "codex.rpc.request.failed",
                    "rpc_method": method,
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "duration_ms": round((monotonic() - started_at) * 1000),
                    "outcome": "transport_error",
                    **self._lifecycle_request_fields(method, params),
                },
            )
            raise
        finally:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()
        if "error" in response:
            error = response["error"]
            logger.warning(
                "Codex RPC request returned an error",
                extra={
                    "event": "codex.rpc.request.remote_error",
                    "rpc_method": method,
                    "request_id": request_id,
                    "rpc_code": error.get("code"),
                    "error_message": redact_text(str(error.get("message", "")), max_length=300),
                    "duration_ms": round((monotonic() - started_at) * 1000),
                    "outcome": "remote_error",
                    **self._lifecycle_request_fields(method, params),
                },
            )
            raise RpcError(
                error.get("code", "unknown"),
                error.get("message", "Codex app-server request failed"),
                error.get("data"),
            )
        result = response.get("result", {})
        result = result if isinstance(result, dict) else {"value": result}
        log_request(
            "Codex RPC request completed",
            extra={
                "event": "codex.rpc.request.completed",
                "rpc_method": method,
                "request_id": request_id,
                "duration_ms": round((monotonic() - started_at) * 1000),
                "outcome": "ok",
                "response_status": self._lifecycle_response_status(result),
                **self._lifecycle_request_fields(method, params),
            },
        )
        return result

    @staticmethod
    def _lifecycle_request_fields(
        method: str,
        params: dict[str, Any] | None,
    ) -> dict[str, str | None]:
        if method not in LIFECYCLE_RPC_METHODS:
            return {}
        values = params if isinstance(params, dict) else {}
        thread_id = values.get("threadId")
        turn_id = values.get("turnId") or values.get("expectedTurnId")
        return {
            "thread_hash": log_identifier(thread_id, namespace="codex-thread")
            if isinstance(thread_id, str)
            else None,
            "turn_hash": log_identifier(turn_id, namespace="codex-turn")
            if isinstance(turn_id, str)
            else None,
        }

    @staticmethod
    def _lifecycle_response_status(result: dict[str, Any]) -> str | None:
        turn = result.get("turn")
        if isinstance(turn, dict) and isinstance(turn.get("status"), str):
            return turn["status"]
        thread = result.get("thread")
        status = thread.get("status") if isinstance(thread, dict) else None
        if isinstance(status, dict) and isinstance(status.get("type"), str):
            return status["type"]
        return None

    async def _write(self, message: dict[str, Any]) -> None:
        if not self.is_running or self.process.stdin is None:
            raise RuntimeError("Codex app-server transport is closed")
        encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        async with self._write_lock:
            self.process.stdin.write(encoded + b"\n")
            await self.process.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        buffer = bytearray()
        try:
            while chunk := await self.process.stdout.read(RPC_READ_CHUNK_BYTES):
                buffer.extend(chunk)
                while True:
                    separator = buffer.find(b"\n")
                    if separator < 0:
                        break
                    if separator > RPC_MAX_LINE_BYTES:
                        raise ValueError("Codex app-server output line exceeded the configured size limit")
                    line = bytes(buffer[:separator])
                    del buffer[: separator + 1]
                    self._handle_stdout_line(line)
                if len(buffer) > RPC_MAX_LINE_BYTES:
                    raise ValueError("Codex app-server output line exceeded the configured size limit")
            if buffer:
                self._handle_stdout_line(bytes(buffer))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Codex app-server output stream failed",
                extra={
                    "event": "codex.rpc.reader.failed",
                    "error_type": type(exc).__name__,
                },
            )
        finally:
            self._fail_pending("Codex app-server output stream closed")
            try:
                self.on_event({"method": "transport/closed", "params": {}})
            except Exception as exc:
                logger.exception(
                    "Codex transport close handler failed",
                    extra={
                        "event": "codex.transport.close_handler.failed",
                        "error_type": type(exc).__name__,
                    },
                )

    def _handle_stdout_line(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Ignoring a non-JSON line from Codex app-server",
                extra={
                    "event": "codex.rpc.line.invalid_json",
                    "error_type": type(exc).__name__,
                    "bytes": len(line),
                },
            )
            return
        if not isinstance(message, dict):
            logger.warning(
                "Ignoring a non-object Codex app-server line",
                extra={"event": "codex.rpc.line.invalid_shape", "bytes": len(line)},
            )
            return
        if "method" in message and "id" in message:
            asyncio.create_task(self._answer_server_request(message))
        elif "method" in message:
            method = message.get("method")
            params = message.get("params")
            if (
                isinstance(method, str)
                and method in LIFECYCLE_EVENT_METHODS
                and isinstance(params, dict)
            ):
                turn = params.get("turn")
                turn = turn if isinstance(turn, dict) else {}
                thread_id = params.get("threadId")
                turn_id = turn.get("id") or turn.get("rootTurnId") or params.get("turnId")
                status = turn.get("status")
                reason = params.get("reason")
                logger.info(
                    "Codex app-server lifecycle event received",
                    extra={
                        "event": "codex.rpc.event.lifecycle",
                        "rpc_method": method,
                        "thread_hash": log_identifier(thread_id, namespace="codex-thread")
                        if isinstance(thread_id, str)
                        else None,
                        "turn_hash": log_identifier(turn_id, namespace="codex-turn")
                        if isinstance(turn_id, str)
                        else None,
                        "turn_status": status if isinstance(status, str) else None,
                        "completion_reason": reason if isinstance(reason, str) else None,
                    },
                )
            if isinstance(method, str) and method not in STREAM_DELTA_METHODS:
                logger.debug(
                    "Codex app-server event received",
                    extra={"event": "codex.rpc.event", "rpc_method": method},
                )
            try:
                self.on_event(message)
            except Exception as exc:
                logger.exception(
                    "Codex event handler failed",
                    extra={
                        "event": "codex.rpc.event_handler.failed",
                        "rpc_method": method if isinstance(method, str) else "unknown",
                        "error_type": type(exc).__name__,
                    },
                )
        elif "id" in message:
            request_id = str(message["id"])
            future = self._pending.pop(request_id, None)
            if future is not None and not future.done():
                future.set_result(message)

    async def _answer_server_request(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params")
        try:
            if not isinstance(method, str) or not isinstance(params, dict):
                raise ValueError("Malformed server request")
            logger.debug(
                "Codex app-server request received",
                extra={
                    "event": "codex.rpc.server_request.received",
                    "server_request_method": method,
                    "rpc_id_type": type(request_id).__name__,
                },
            )
            result = await self.on_server_request(method, request_id, params)
            await self._write({"id": request_id, "result": result})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Handler failures can chain Telegram HTTP errors whose URL contains the bot token.
            logger.exception(
                "Could not handle a Codex app-server request",
                extra={
                    "event": "codex.rpc.server_request.failed",
                    "server_request_method": method if isinstance(method, str) else "unknown",
                    "rpc_id_type": type(request_id).__name__,
                    "error_type": type(exc).__name__,
                },
            )
            try:
                await self._write(
                    {
                        "id": request_id,
                        "error": {"code": -32603, "message": "Request could not be handled"},
                    }
                )
            except Exception as exc:
                logger.exception(
                    "Could not return an error to Codex app-server",
                    extra={
                        "event": "codex.rpc.server_request.reply_failed",
                        "server_request_method": method if isinstance(method, str) else "unknown",
                        "error_type": type(exc).__name__,
                    },
                )

    async def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        first_chunk = True
        try:
            while chunk := await self.process.stderr.read(8192):
                self._stderr_total_bytes += len(chunk)
                self._stderr_tail.extend(chunk)
                if len(self._stderr_tail) > STDERR_TAIL_BYTES:
                    del self._stderr_tail[:-STDERR_TAIL_BYTES]
                if first_chunk:
                    first_chunk = False
                    logger.warning(
                        "Codex app-server emitted stderr diagnostics",
                        extra={
                            "event": "codex.app_server.stderr_received",
                            "stderr_bytes": len(chunk),
                            "stderr_tail": redact_text(
                                chunk.decode("utf-8", errors="replace"),
                                max_length=2000,
                            ),
                        },
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Could not drain Codex app-server stderr",
                extra={"event": "codex.rpc.stderr_reader.failed", "error_type": type(exc).__name__},
            )
            return
        if self.process.returncode not in {None, 0}:
            logger.log(
                logging.WARNING if self._closed else logging.ERROR,
                "Codex app-server exited with a non-zero status",
                extra={
                    "event": "codex.app_server.process_exit",
                    "return_code": self.process.returncode,
                    "stderr_bytes": self._stderr_total_bytes,
                    "stderr_line_count": self._stderr_tail.count(b"\n"),
                    "stderr_tail": redact_text(
                        self._stderr_tail.decode("utf-8", errors="replace"),
                        max_length=3000,
                    ),
                },
            )

    def _fail_pending(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self._pending.clear()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.stdin and not self.process.stdin.is_closing():
            self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=3)
        except asyncio.TimeoutError:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=3)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        for task in (self._reader_task, self._stderr_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(self._reader_task, self._stderr_task, return_exceptions=True)
        logger.info(
            "Codex app-server process closed",
            extra={
                "event": "codex.app_server.closed",
                "return_code": self.process.returncode,
                "pending_count": len(self._pending),
            },
        )
        self._fail_pending("Codex app-server stopped")
