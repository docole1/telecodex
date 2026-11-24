"""Safe MCP configuration contracts shared by Codex and Telegram layers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit
from typing import Any

_SERVER_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_ARG = re.compile(r"(?i)^--?(?:password|passwd|token|secret|api[-_]?key|credential|auth|authorization|header)(?:$|=)")
_PROTECTED_ENV_PREFIXES = ("CODEX_", "OPENAI_", "TELEGRAM_", "TELECODEX_")


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str | None
    read_only: bool | None


@dataclass(frozen=True)
class McpServer:
    name: str
    runtime_status: str | None
    auth_status: str
    tools: tuple[McpTool, ...]
    tools_error: bool
    server_name: str | None = None
    server_version: str | None = None
    http_origin: str | None = None
    plugin_id: str | None = None

    @property
    def is_plugin(self) -> bool:
        return self.plugin_id is not None


def validate_server_name(name: str) -> str:
    normalized = name.strip()
    if not _SERVER_NAME.fullmatch(normalized):
        raise ValueError("MCP server name must start with a letter, digit, or underscore and contain at most 64 characters")
    return normalized


def validate_env_name(name: str) -> str:
    normalized = name.strip()
    if len(normalized) > 128 or not _ENV_NAME.fullmatch(normalized):
        raise ValueError("Environment variable name is invalid")
    if is_protected_env_name(normalized):
        raise ValueError("TeleCodex, Telegram, and Codex environment variables cannot be forwarded to MCP")
    return normalized


def is_protected_env_name(name: str) -> bool:
    return name == "TELECODEX_MCP_ENV_VARS" or name.startswith(_PROTECTED_ENV_PREFIXES)


def validate_requested_env_names(names: tuple[str, ...], allowed: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(validate_env_name(name) for name in names if name.strip()))
    disallowed = set(normalized).difference(allowed)
    if disallowed:
        raise ValueError("One or more environment variables are not allowlisted")
    return normalized


def validate_http_url(url: str) -> str:
    normalized = url.strip()
    if any(character.isspace() or ord(character) < 0x20 for character in normalized):
        raise ValueError("URL must not contain whitespace or control characters")
    try:
        parsed = urlsplit(normalized)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Enter a valid HTTP or HTTPS URL") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("URL must use HTTP or HTTPS and must not contain credentials or a fragment")
    if parsed.query:
        raise ValueError("Put credentials in environment variables; URL query parameters are not accepted")
    if len(normalized) > 2000:
        raise ValueError("URL exceeds the maximum supported length")
    return normalized


def parse_stdio_server_status(raw: dict[str, Any]) -> McpServer:
    tools_value = raw.get("tools")
    tools: list[McpTool] = []
    if isinstance(tools_value, dict):
        for name, value in tools_value.items():
            if not isinstance(name, str):
                continue
            tool = value if isinstance(value, dict) else {}
            description = tool.get("description")
            read_only = tool.get("readOnlyHint")
            tools.append(
                McpTool(
                    name=name,
                    description=description if isinstance(description, str) else None,
                    read_only=read_only if isinstance(read_only, bool) else None,
                )
            )
    tools.sort(key=lambda item: item.name.casefold())
    info = raw.get("serverInfo")
    info = info if isinstance(info, dict) else {}
    return McpServer(
        name=str(raw.get("name") or ""),
        runtime_status=(
            raw.get("runtimeStatus")
            if isinstance(raw.get("runtimeStatus"), str)
            else None
        ),
        auth_status=str(raw.get("authStatus") or "unknown"),
        tools=tuple(tools),
        tools_error=isinstance(raw.get("toolsError"), str) and bool(raw.get("toolsError")),
        server_name=info.get("name") if isinstance(info.get("name"), str) else None,
        server_version=info.get("version") if isinstance(info.get("version"), str) else None,
        http_origin=raw.get("httpOrigin") if isinstance(raw.get("httpOrigin"), str) else None,
        plugin_id=raw.get("pluginId") if isinstance(raw.get("pluginId"), str) else None,
    )


def stdio_config(
    command: str,
    args: tuple[str, ...],
    cwd: str | None,
    env_vars: tuple[str, ...],
) -> dict[str, Any]:
    executable = command.strip()
    if not executable or len(executable) > 512 or "\x00" in executable:
        raise ValueError("Executable path is empty or invalid")
    if len(args) > 64 or any(not isinstance(argument, str) or "\x00" in argument for argument in args):
        raise ValueError("Arguments are invalid or exceed the 64-item limit")
    if any(
        _SENSITIVE_ARG.match(argument)
        or argument in {"-H", "-u", "--user"}
        or "@" in argument and "://" in argument and "@" in argument.split("://", 1)[1]
        for argument in args
    ):
        raise ValueError("Put credentials in allowlisted environment variables, not command arguments")
    if len(json.dumps([executable, *args], ensure_ascii=False)) > 2200:
        raise ValueError("Arguments exceed the size limit")
    if cwd is not None and len(cwd) > 800:
        raise ValueError("Working directory exceeds the size limit")
    result: dict[str, Any] = {"command": executable, "args": list(args)}
    if cwd:
        result["cwd"] = cwd
    if env_vars:
        result["env_vars"] = list(env_vars)
    return result


def http_config(
    url: str,
    auth: str,
    bearer_token_env_var: str | None,
) -> dict[str, Any]:
    if auth not in {"none", "oauth", "bearer"}:
        raise ValueError("Unsupported MCP HTTP authentication mode")
    result: dict[str, Any] = {"url": validate_http_url(url)}
    if auth == "oauth":
        result["auth"] = "oauth"
    elif auth == "bearer":
        if not bearer_token_env_var:
            raise ValueError("Choose an allowlisted environment variable for the bearer token")
        result["bearer_token_env_var"] = validate_env_name(bearer_token_env_var)
    return result
