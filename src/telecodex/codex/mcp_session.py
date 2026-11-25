"""App-server and config operations for Codex MCP servers."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from telecodex.codex.app_server import build_app_server_environment
from telecodex.codex.mcp import (
    McpServer,
    http_config,
    parse_stdio_server_status,
    stdio_config,
    validate_requested_env_names,
    validate_server_name,
)
from telecodex.codex.rpc import RpcError
from telecodex.config import AppConfig
from telecodex.logging_config import log_identifier
from telecodex.state.models import RuntimeState

logger = logging.getLogger(__name__)


class CodexMcpSession:
    """Manage MCP config and runtime calls for one CodexSession."""

    def __init__(
        self,
        *,
        config: AppConfig,
        app_server: Any,
        ensure_app_server: Callable[[], Awaitable[None]],
        state: Callable[[], RuntimeState],
        loaded_thread_id: Callable[[], str | None],
        thread_log_fields: Callable[..., dict[str, str | None]],
        schedule_status: Callable[..., None],
    ) -> None:
        self.config = config
        self.app_server = app_server
        self._ensure_app_server = ensure_app_server
        self._state = state
        self._get_loaded_thread_id = loaded_thread_id
        self._thread_log_fields = thread_log_fields
        self._schedule_status = schedule_status
        self._mcp_oauth_logins: dict[str, str] = {}
        self._mcp_oauth_legacy_logins: list[str] = []

    @property
    def state(self) -> RuntimeState:
        return self._state()

    @property
    def _loaded_thread_id(self) -> str | None:
        return self._get_loaded_thread_id()

    async def list_mcp_servers(self) -> list[McpServer]:
        """Return configured MCP servers and the selected thread's runtime catalog."""
        await self._ensure_app_server()
        thread_id = (
            self.state.thread_id
            if self.state.thread_id and self._loaded_thread_id == self.state.thread_id
            else None
        )
        try:
            return await self._list_mcp_server_status(thread_id)
        except RpcError as exc:
            if not thread_id:
                raise
            logger.info(
                "MCP thread status is unavailable; falling back to app-level server status",
                extra={
                    "event": "codex.mcp.thread_status.fallback",
                    "rpc_code": exc.code,
                    **self._thread_log_fields(thread_id),
                },
            )
            return await self._list_mcp_server_status(None)


    async def _list_mcp_server_status(self, thread_id: str | None) -> list[McpServer]:
        servers: list[McpServer] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _ in range(100):
            params: dict[str, Any] = {
                "detail": "toolsAndAuthOnly",
                "limit": 100,
            }
            if cursor:
                params["cursor"] = cursor
            if thread_id:
                params["threadId"] = thread_id
            result = await self.app_server.request(
                "mcpServerStatus/list",
                params,
                timeout_seconds=60,
            )
            data = result.get("data")
            if not isinstance(data, list):
                raise RuntimeError("Codex app-server returned an invalid MCP server list")
            servers.extend(
                parse_stdio_server_status(item)
                for item in data
                if isinstance(item, dict)
            )
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            if next_cursor in seen_cursors:
                raise RuntimeError("Codex app-server returned a repeated MCP pagination cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        else:
            raise RuntimeError("Codex app-server MCP list exceeded the pagination limit")
        return sorted(servers, key=lambda item: item.name.casefold())


    async def add_mcp_stdio_server(
        self,
        name: str,
        command: str,
        args: tuple[str, ...],
        cwd: str | None,
        env_vars: tuple[str, ...],
    ) -> None:
        name = validate_server_name(name)
        checked_env_vars = validate_requested_env_names(env_vars, self.config.mcp_env_vars)
        await self._add_mcp_server_config(
            name,
            stdio_config(command, args, cwd, checked_env_vars),
        )


    async def add_mcp_http_server(
        self,
        name: str,
        url: str,
        auth: str,
        bearer_token_env_var: str | None = None,
    ) -> None:
        name = validate_server_name(name)
        if auth == "bearer":
            if bearer_token_env_var is None:
                raise ValueError("Choose an allowlisted environment variable for the bearer token")
            validate_requested_env_names((bearer_token_env_var,), self.config.mcp_env_vars)
        await self._add_mcp_server_config(
            name,
            http_config(url, auth, bearer_token_env_var),
        )


    async def _add_mcp_server_config(self, name: str, server_config: dict[str, Any]) -> None:
        await self._ensure_app_server()
        for attempt in range(2):
            current = await self._read_mcp_config()
            configured = current.get("config")
            configured = configured if isinstance(configured, dict) else {}
            servers = configured.get("mcp_servers")
            servers = servers if isinstance(servers, dict) else {}
            if name in servers:
                raise ValueError("An MCP server with this name already exists")
            params: dict[str, Any] = {
                "edits": [{
                    "keyPath": f"mcp_servers.{name}",
                    "mergeStrategy": "upsert",
                    "value": server_config,
                }],
                "reloadUserConfig": False,
            }
            version = self._user_config_version(current)
            if version:
                params["expectedVersion"] = version
            try:
                await self.app_server.request(
                    "config/batchWrite",
                    params,
                    timeout_seconds=30,
                )
                return
            except RpcError as exc:
                if attempt == 0 and self._is_config_version_conflict(exc):
                    continue
                raise
        raise RuntimeError("Could not safely update the shared Codex MCP configuration")


    async def set_mcp_server_enabled(self, name: str, enabled: bool) -> None:
        name = validate_server_name(name)
        await self._ensure_app_server()
        for attempt in range(2):
            current = await self._read_mcp_config()
            configured = current.get("config")
            configured = configured if isinstance(configured, dict) else {}
            servers = configured.get("mcp_servers")
            if not isinstance(servers, dict) or name not in servers:
                raise ValueError("MCP server configuration was not found")
            params: dict[str, Any] = {
                "edits": [{
                    "keyPath": f"mcp_servers.{name}.enabled",
                    "mergeStrategy": "replace",
                    "value": enabled,
                }],
                "reloadUserConfig": False,
            }
            version = self._user_config_version(current)
            if version:
                params["expectedVersion"] = version
            try:
                await self.app_server.request(
                    "config/batchWrite",
                    params,
                    timeout_seconds=30,
                )
                return
            except RpcError as exc:
                if attempt == 0 and self._is_config_version_conflict(exc):
                    continue
                raise
        raise RuntimeError("Could not safely update the shared Codex MCP configuration")


    async def remove_mcp_server(self, name: str) -> None:
        name = validate_server_name(name)
        await self._run_codex_mcp_command("remove", name)


    async def logout_mcp_server(self, name: str) -> None:
        name = validate_server_name(name)
        await self._run_codex_mcp_command("logout", name)


    async def _run_codex_mcp_command(self, action: str, name: str) -> None:
        if action not in {"remove", "logout"}:
            raise ValueError("Unsupported Codex MCP CLI operation")
        process = await asyncio.create_subprocess_exec(
            self.config.codex_bin,
            "mcp",
            action,
            name,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=build_app_server_environment(os.environ, ()),
        )
        try:
            _, _ = await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError("Codex MCP management command timed out") from None
        if process.returncode != 0:
            raise RuntimeError("Codex MCP management command failed")


    async def _read_mcp_config(self) -> dict[str, Any]:
        return await self.app_server.request(
            "config/read",
            {"includeLayers": True},
            timeout_seconds=30,
        )


    @staticmethod
    def _user_config_version(config: dict[str, Any]) -> str | None:
        layers = config.get("layers")
        if not isinstance(layers, list):
            return None
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            name = layer.get("name")
            if isinstance(name, dict) and name.get("type") == "user":
                version = layer.get("version")
                return version if isinstance(version, str) and version else None
        return None


    async def reload_mcp_config(self) -> bool:
        """Queue a config reload for loaded threads without interrupting turns."""
        if not self.app_server.is_running:
            return False
        await self.app_server.request(
            "config/mcpServer/reload",
            params_as_null=True,
            timeout_seconds=30,
        )
        return True


    async def start_mcp_oauth(self, name: str) -> tuple[str, str | None]:
        name = validate_server_name(name)
        await self._ensure_app_server()
        params: dict[str, Any] = {"name": name}
        if self.state.thread_id:
            params["threadId"] = self.state.thread_id
        result = await self.app_server.request(
            "mcpServer/oauth/login",
            params,
            timeout_seconds=30,
        )
        authorization_url = result.get("authorizationUrl")
        login_id = result.get("loginId")
        if not isinstance(authorization_url, str):
            raise RuntimeError("Codex app-server did not return an MCP authorization URL")
        parsed = urlsplit(authorization_url)
        try:
            _ = parsed.port
        except ValueError as exc:
            raise RuntimeError("Codex app-server returned an invalid MCP authorization URL") from exc
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise RuntimeError("Codex app-server returned an invalid MCP authorization URL")
        if isinstance(login_id, str) and login_id:
            self._mcp_oauth_logins[login_id] = name
        else:
            self._mcp_oauth_legacy_logins.append(name)
        return authorization_url, login_id if isinstance(login_id, str) else None


    async def mcp_oauth_uses_remote_callback(self, name: str) -> bool:
        """Return whether the configured OAuth callback can be reached from another device."""
        name = validate_server_name(name)
        await self._ensure_app_server()
        result = await self._read_mcp_config()
        config = result.get("config")
        config = config if isinstance(config, dict) else {}
        servers = config.get("mcp_servers")
        servers = servers if isinstance(servers, dict) else {}
        server = servers.get(name)
        server = server if isinstance(server, dict) else {}
        oauth = server.get("oauth")
        oauth = oauth if isinstance(oauth, dict) else {}
        callback_url = oauth.get("callback_url")
        if not isinstance(callback_url, str) or not callback_url:
            callback_url = config.get("mcp_oauth_callback_url")
        if not isinstance(callback_url, str) or not callback_url:
            return False
        try:
            parsed = urlsplit(callback_url)
            host = parsed.hostname
            _ = parsed.port
        except ValueError:
            return False
        if not parsed.scheme or not host:
            return False
        try:
            address = ipaddress.ip_address(host)
            return not address.is_loopback and not address.is_unspecified
        except ValueError:
            return host.casefold() != "localhost"


    def _handle_mcp_oauth_completed(self, params: dict[str, Any]) -> None:
        name = params.get("name")
        login_id = params.get("loginId")
        success = params.get("success") is True
        if not isinstance(name, str):
            return
        if isinstance(login_id, str):
            expected_name = self._mcp_oauth_logins.pop(login_id, None)
            if expected_name != name:
                return
        elif name not in self._mcp_oauth_legacy_logins:
            return
        else:
            self._mcp_oauth_legacy_logins.remove(name)
        self._schedule_status(
            "status.label.mcp_oauth_success" if success else "status.label.mcp_oauth_failed",
            name,
        )


    @staticmethod
    def _is_config_version_conflict(error: RpcError) -> bool:
        message = str(error).casefold()
        return "version" in message or "stale" in message
