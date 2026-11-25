from __future__ import annotations

import contextvars
import hashlib
import hmac
import json
import logging
import os
import re
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "telecodex_log_context",
    default={},
)
_IDENTIFIER_KEY: bytes | None = None

_SAFE_FIELDS = frozenset(
    {
        "event",
        "operation",
        "stage",
        "outcome",
        "status",
        "result",
        "code",
        "route",
        "component",
        "bot_username",
        "network_access",
        "correlation_id",
        "update_id",
        "update_kind",
        "update_count",
        "access_decision",
        "has_text",
        "has_attachments",
        "output_truncated",
        "text_length",
        "input_item_count",
        "attachment_kinds",
        "is_reply",
        "group_trigger_checked",
        "group_trigger_required",
        "command_name",
        "item_count",
        "item_hash",
        "attachment_count",
        "attachment_kinds",
        "context_kind",
        "context_hash",
        "actor_hash",
        "thread_hash",
        "turn_hash",
        "request_id",
        "rpc_method",
        "interrupt_source",
        "thread_unsubscribe_source",
        "completion_reason",
        "response_status",
        "expected_thread_hash",
        "received_thread_hash",
        "expected_turn_hash",
        "received_turn_hash",
        "rpc_id_type",
        "rpc_code",
        "server_request_method",
        "error_type",
        "error_code",
        "error_message",
        "http_method",
        "http_route",
        "http_status",
        "duration_ms",
        "timeout_seconds",
        "attempt",
        "retry_delay_seconds",
        "retry_seconds",
        "chat_name_length",
        "text_part_bytes",
        "poll_error_count",
        "poll_age_seconds",
        "recovery_age_seconds",
        "last_event_age_seconds",
        "last_event_method",
        "event_loop_lag_ms",
        "pending_count",
        "queue_size",
        "return_code",
        "stderr_bytes",
        "bytes",
        "file_size",
        "file_type",
        "duration_seconds",
        "backend",
        "model",
        "account_type",
        "device",
        "compute_type",
        "effort",
        "permission_mode",
        "turn_status",
        "pending_phase",
        "device_login_type",
        "retryable",
        "was_ready",
        "was_busy",
        "transport_running",
        "container_state",
        "resolve_error",
        "stderr_tail",
        "stderr_line_count",
    }
)

_SENSITIVE_FIELD = re.compile(
    r"(?:authorization|access.?token|refresh.?token|id.?token|api.?key|"
    r"password|passwd|secret|credential|device.?code|user.?code|private.?key)",
    re.IGNORECASE,
)
_REDACTIONS = (
    (re.compile(r"\b(bot\d{6,}:)[A-Za-z0-9_-]{20,}\b"), r"\1[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[API_KEY]"),
    (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"), "[API_KEY]"),
    (re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1[REDACTED]"),
    (
        re.compile(
            r"(?i)(['\"]?(?:access[_-]?token|refresh[_-]?token|id[_-]?token|"
            r"api[_-]?key|password|passwd|secret|device[_-]?code|user[_-]?code)"
            r"['\"]?\s*[:=]\s*['\"]?)[^\s,;&}\"']+"
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"([a-z][a-z0-9+.-]*://)[^/@\s:]+(?::[^/@\s]*)?@", re.IGNORECASE),
        r"\1[REDACTED]@",
    ),
    (
        re.compile(r"([?&](?:token|key|password|code|api_key)=)[^&#\s]+", re.IGNORECASE),
        r"\1[REDACTED]",
    ),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
)


def redact_text(value: str, *, max_length: int = 4000) -> str:
    text = value[:max_length]
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    if len(value) > max_length:
        text += "…[TRUNCATED]"
    return text


def _safe_value(value: Any, key: str | None = None) -> Any:
    if key is not None and _SENSITIVE_FIELD.search(key):
        return "[REDACTED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_text(value, max_length=4000)
    if isinstance(value, dict):
        return {str(k): _safe_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    return redact_text(str(value), max_length=1000)


class _LogContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _LOG_CONTEXT.get().items():
            if key in _SAFE_FIELDS and not hasattr(record, key):
                setattr(record, key, value)
        return True


class _JsonLogFormatter(logging.Formatter):
    _standard_fields = frozenset(logging.makeLogRecord({}).__dict__) | {
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "process_id": record.process,
            "thread_name": record.threadName,
            "message": redact_text(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key in self._standard_fields or key.startswith("_"):
                continue
            if key in _SAFE_FIELDS:
                data[key] = _safe_value(value, key)
        if record.exc_info:
            exception_type = record.exc_info[0]
            traceback_frames = traceback.extract_tb(record.exc_info[2])
            data["exception"] = {
                "type": exception_type.__name__ if exception_type else "Exception",
                "stack": [
                    {
                        "file": redact_text(frame.filename, max_length=300),
                        "line": frame.lineno,
                        "function": frame.name,
                    }
                    for frame in traceback_frames[-12:]
                ],
            }
        if record.stack_info:
            data["stack"] = redact_text(record.stack_info)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


@contextmanager
def bind_log_context(**fields: Any) -> Iterator[None]:
    current = _LOG_CONTEXT.get()
    safe_fields = {
        key: _safe_value(value, key)
        for key, value in fields.items()
        if key in _SAFE_FIELDS
    }
    token = _LOG_CONTEXT.set({**current, **safe_fields})
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def current_log_context() -> dict[str, Any]:
    return dict(_LOG_CONTEXT.get())


@contextmanager
def isolated_log_context(**fields: Any) -> Iterator[None]:
    safe_fields = {
        key: _safe_value(value, key)
        for key, value in fields.items()
        if key in _SAFE_FIELDS
    }
    token = _LOG_CONTEXT.set(safe_fields)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def log_identifier(value: int | str, *, namespace: str) -> str | None:
    if _IDENTIFIER_KEY is None:
        return None
    message = f"{namespace}:{value}".encode("utf-8", errors="replace")
    return hmac.new(_IDENTIFIER_KEY, message, hashlib.sha256).hexdigest()[:12]


def log_exception(
    logger: logging.Logger,
    message: str,
    exc: BaseException,
    *,
    event: str,
    level: int = logging.ERROR,
    **fields: Any,
) -> None:
    logger.log(
        level,
        message,
        extra={"event": event, "error_type": type(exc).__name__, **fields},
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def configure_logging() -> None:
    global _IDENTIFIER_KEY

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    salt = os.environ.get("LOG_ID_SALT") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    _IDENTIFIER_KEY = (
        hmac.new(b"telecodex-log-id-v1", salt.encode("utf-8"), hashlib.sha256).digest()
        if salt
        else os.urandom(32)
    )

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.addFilter(_LogContextFilter())
    handler.setFormatter(_JsonLogFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # Telegram's Bot API embeds the bot token in request URLs.
    logging.getLogger("aiohttp.client").setLevel(logging.WARNING)
