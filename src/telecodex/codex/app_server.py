from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from telecodex import __version__
from telecodex.codex.mcp import is_protected_env_name
from telecodex.codex.permissions import (
    ASK_PERMISSION_PROFILE,
    READ_ONLY_PERMISSION_PROFILE,
    WORKSPACE_PERMISSION_PROFILE,
)
from telecodex.codex.rpc import EventHandler, JsonRpcProcess, ServerRequestHandler
from telecodex.logging_config import log_exception

logger = logging.getLogger(__name__)


class CodexAuthenticationError(RuntimeError):
    """The native Codex login is missing or does not use a ChatGPT account."""


class CodexAppServer:
    def __init__(
        self,
        codex_bin: str,
        network_access: bool,
        on_event: EventHandler,
        on_server_request: ServerRequestHandler,
        mcp_env_vars: tuple[str, ...] = (),
    ) -> None:
        self.codex_bin = codex_bin
        self.network_access = network_access
        self.on_event = on_event
        self.on_server_request = on_server_request
        self.mcp_env_vars = mcp_env_vars
        self.rpc: JsonRpcProcess | None = None
        self.account: dict[str, Any] | None = None

    @property
    def is_running(self) -> bool:
        return self.rpc is not None and self.rpc.is_running

    async def start(self) -> JsonRpcProcess:
        if self.is_running:
            assert self.rpc is not None
            logger.debug(
                "Reusing running Codex app-server",
                extra={"event": "codex.app_server.reused"},
            )
            return self.rpc
        await self.stop()
        logger.info(
            "Starting Codex app-server",
            extra={
                "event": "codex.app_server.starting",
                "network_access": self.network_access,
            },
        )

        env = build_app_server_environment(os.environ, self.mcp_env_vars)
        command = [
            self.codex_bin,
            "app-server",
            "--listen",
            "stdio://",
            "-c",
            'shell_environment_policy.inherit="core"',
            "-c",
            "shell_environment_policy.ignore_default_excludes=false",
            "-c",
            "thread_unload_delay_secs=0",
            "-c",
            f'default_permissions="{WORKSPACE_PERMISSION_PROFILE}"',
            "-c",
            f'permissions.{WORKSPACE_PERMISSION_PROFILE}.extends=":workspace"',
            "-c",
            self._filesystem_profile_override(),
            "-c",
            f"permissions.{WORKSPACE_PERMISSION_PROFILE}.network.enabled={str(self.network_access).lower()}",
        ]
        for override in self._permission_profile_overrides():
            command.extend(["-c", override])
        self.rpc = await JsonRpcProcess.start(
            command,
            env,
            on_event=self.on_event,
            on_server_request=self.on_server_request,
        )
        try:
            await self.rpc.request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "telecodex",
                        "title": "TeleCodex",
                        "version": __version__,
                    },
                    "capabilities": {"experimentalApi": True},
                },
                timeout_seconds=60,
            )
            await self.rpc.notify("initialized", {})
            account_result = await self.rpc.request(
                "account/read",
                {"refreshToken": False},
                timeout_seconds=30,
            )
            account = account_result.get("account")
            account_type = account.get("type") if isinstance(account, dict) else None
            if account_type == "apiKey":
                raise CodexAuthenticationError(
                    "Codex is using an API key. TeleCodex requires ChatGPT sign-in to avoid API billing; "
                    "run `codex login` as the TeleCodex service user and choose ChatGPT sign-in."
                )
            if account_type != "chatgpt":
                raise CodexAuthenticationError(
                    "No ChatGPT sign-in is available to TeleCodex. Run `codex login status` and, if needed, "
                    "`codex login` as the same Unix user that runs TeleCodex."
                )
            if account_result.get("requiresOpenaiAuth") is not True:
                raise CodexAuthenticationError(
                    "The active Codex model provider is not using ChatGPT authentication. "
                    "Check the provider settings in the same user's Codex configuration."
                )
            self.account = account
        except Exception as exc:
            log_exception(
                logger,
                "Codex app-server initialization failed",
                exc,
                event="codex.app_server.initialize_failed",
            )
            await self.stop()
            raise
        logger.info(
            "Codex app-server initialized with ChatGPT authentication",
            extra={"event": "codex.app_server.initialized"},
        )
        return self.rpc

    def _filesystem_profile_override(self) -> str:
        rules = {
            ":root": "deny",
            ":minimal": "read",
            str(Path(self.codex_bin).resolve()): "read",
        }
        entries = ", ".join(f"{json.dumps(path)} = {json.dumps(access)}" for path, access in rules.items())
        return f"permissions.{WORKSPACE_PERMISSION_PROFILE}.filesystem={{ {entries} }}"

    def _permission_profile_overrides(self) -> list[str]:
        executable = str(Path(self.codex_bin).resolve())
        ask_rules = {
            ":root": "deny",
            ":minimal": "read",
            executable: "read",
        }
        read_only_rules = {
            ":root": "deny",
            ":minimal": "read",
            ":workspace": "read",
            ":tmpdir": "deny",
            ":slash_tmp": "deny",
            executable: "read",
        }
        ask_entries = ", ".join(
            f"{json.dumps(path)} = {json.dumps(access)}" for path, access in ask_rules.items()
        )
        read_only_entries = ", ".join(
            f"{json.dumps(path)} = {json.dumps(access)}" for path, access in read_only_rules.items()
        )
        return [
            f'permissions.{ASK_PERMISSION_PROFILE}.extends=":workspace"',
            f"permissions.{ASK_PERMISSION_PROFILE}.filesystem={{ {ask_entries} }}",
            f"permissions.{ASK_PERMISSION_PROFILE}.network.enabled={str(self.network_access).lower()}",
            f'permissions.{READ_ONLY_PERMISSION_PROFILE}.extends=":workspace"',
            f"permissions.{READ_ONLY_PERMISSION_PROFILE}.filesystem={{ {read_only_entries} }}",
            f"permissions.{READ_ONLY_PERMISSION_PROFILE}.network.enabled={str(self.network_access).lower()}",
        ]

    async def stop(self) -> None:
        rpc = self.rpc
        self.rpc = None
        self.account = None
        if rpc is not None:
            await rpc.close()

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_seconds: float = 120,
        *,
        params_as_null: bool = False,
    ) -> dict[str, Any]:
        if not self.is_running or self.rpc is None:
            raise RuntimeError("Codex app-server has not been started")
        return await self.rpc.request(
            method,
            params,
            timeout_seconds=timeout_seconds,
            params_as_null=params_as_null,
        )


def build_app_server_environment(
    source: dict[str, str],
    allowed_mcp_env_vars: tuple[str, ...] = (),
) -> dict[str, str]:
    """Keep sensitive variables only when explicitly allowlisted for MCP use."""
    allowed = set(allowed_mcp_env_vars)
    env = dict(source)
    for variable in tuple(env):
        if is_protected_env_name(variable):
            env.pop(variable, None)
        elif any(
            marker in variable.upper()
            for marker in ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")
        ):
            if variable not in allowed:
                env.pop(variable, None)
    return env
