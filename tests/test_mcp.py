from __future__ import annotations

import asyncio
import ast
import html
import itertools
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from multi_user_test_support import FakeSentMessage, callback_event, message_event

from telecodex.codex.app_server import build_app_server_environment
from telecodex.codex.mcp import (
    McpServer,
    http_config,
    parse_stdio_server_status,
    stdio_config,
    validate_requested_env_names,
    validate_server_name,
)
from telecodex.codex.rpc import JsonRpcProcess
from telecodex.codex.rpc import RpcError
from telecodex.codex.session import CodexSession
from telecodex.config import AppConfig, _parse_mcp_env_vars, load_config
from telecodex.state import RuntimeStateStore
from telecodex.telegram.controller import TelegramController
from telecodex.telegram.user_controller import TelegramUserController


_HTML_FORMAT_TAG = re.compile(r"</?(?:strong|code|pre)>")


def _plain_html(text: str) -> str:
    return html.unescape(_HTML_FORMAT_TAG.sub("", text))


class McpConfigTests(unittest.TestCase):
    def test_mcp_ui_uses_existing_russian_text_keys(self) -> None:
        texts = json.loads(Path("src/telecodex/texts/ru.json").read_text(encoding="utf-8"))

        def contains(key: str) -> bool:
            value: Any = texts
            for part in key.split("."):
                if not isinstance(value, dict) or part not in value:
                    return False
                value = value[part]
            return isinstance(value, str)

        missing: set[str] = set()
        for filename in (
            "src/telecodex/telegram/mcp.py",
            "src/telecodex/telegram/user_controller.py",
        ):
            tree = ast.parse(Path(filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_text"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value.startswith("mcp.")
                    and not contains(node.args[0].value)
                ):
                    missing.add(node.args[0].value)
        self.assertEqual(missing, set())

    def test_server_names_are_toml_path_safe(self) -> None:
        self.assertEqual(validate_server_name("docs_server-1"), "docs_server-1")
        for value in ("", "bad.name", "two words", "x/y", "a" * 65):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_server_name(value)
        with self.assertRaises(ValueError):
            validate_server_name("-leading-hyphen")

    def test_stdio_config_keeps_executable_and_arguments_separate(self) -> None:
        result = stdio_config("python3", ("-m", "fixture"), "/tmp", ("MCP_FIXTURE_TOKEN",))
        self.assertEqual(
            result,
            {
                "command": "python3",
                "args": ["-m", "fixture"],
                "cwd": "/tmp",
                "env_vars": ["MCP_FIXTURE_TOKEN"],
            },
        )

    def test_stdio_config_rejects_inline_credentials(self) -> None:
        for args in (("--token", "sensitive-value"), ("-H", "Authorization: Bearer x"), ("https://user:pass@example.test" ,)):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "environment variables"):
                stdio_config("python3", args, None, ())

    def test_http_config_never_accepts_inline_credentials_or_query_secrets(self) -> None:
        self.assertEqual(
            http_config("https://example.test/mcp", "oauth", None),
            {"url": "https://example.test/mcp", "auth": "oauth"},
        )
        self.assertEqual(
            http_config("https://example.test/mcp", "none", None),
            {"url": "https://example.test/mcp"},
        )
        for url in (
            "ftp://example.test/mcp",
            "https://user:password@example.test/mcp",
            "https://example.test/mcp?token=secret",
            "https://example.test/mcp#fragment",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                http_config(url, "none", None)

    def test_http_bearer_requires_an_allowlisted_variable_name(self) -> None:
        with self.assertRaises(ValueError):
            http_config("https://example.test/mcp", "bearer", None)
        with self.assertRaises(ValueError):
            validate_requested_env_names(("MCP_SECRET",), ("OTHER_SECRET",))
        self.assertEqual(
            validate_requested_env_names(("MCP_SECRET", "MCP_SECRET"), ("MCP_SECRET",)),
            ("MCP_SECRET",),
        )

    def test_status_parser_keeps_only_public_server_and_tool_metadata(self) -> None:
        server = parse_stdio_server_status(
            {
                "name": "docs",
                "runtimeStatus": "connected",
                "authStatus": "oAuth",
                "httpOrigin": "https://example.test",
                "serverInfo": {"name": "Docs", "version": "1.2"},
                "tools": {
                    "search": {"description": "Search docs", "readOnlyHint": True},
                    "write": {"description": "Write docs"},
                },
                "toolsError": None,
                "resources": [{"uri": "https://example.test/private", "name": "secret"}],
            }
        )
        self.assertEqual(server.name, "docs")
        self.assertEqual(server.runtime_status, "connected")
        self.assertEqual(server.auth_status, "oAuth")
        self.assertEqual([tool.name for tool in server.tools], ["search", "write"])
        self.assertTrue(server.tools[0].read_only)
        self.assertFalse(hasattr(server, "resources"))

    def test_environment_allowlist_preserves_only_explicit_sensitive_values(self) -> None:
        source = {
            "PATH": "/usr/bin",
            "BOT_TOKEN": "bot-secret",
            "MCP_FIXTURE_SECRET": "fixture-secret",
            "TELECODEX_MCP_ENV_VARS": "MCP_FIXTURE_SECRET",
            "NORMAL_SETTING": "visible",
        }
        without_allowlist = build_app_server_environment(source)
        self.assertNotIn("BOT_TOKEN", without_allowlist)
        self.assertNotIn("MCP_FIXTURE_SECRET", without_allowlist)
        self.assertNotIn("TELECODEX_MCP_ENV_VARS", without_allowlist)
        with_allowlist = build_app_server_environment(source, ("MCP_FIXTURE_SECRET",))
        self.assertEqual(with_allowlist["MCP_FIXTURE_SECRET"], "fixture-secret")
        self.assertNotIn("BOT_TOKEN", with_allowlist)
        self.assertNotIn("TELECODEX_MCP_ENV_VARS", with_allowlist)
        self.assertEqual(with_allowlist["NORMAL_SETTING"], "visible")
        self.assertNotIn(
            "TELEGRAM_BOT_TOKEN",
            build_app_server_environment(source | {"TELEGRAM_BOT_TOKEN": "telegram-secret"}, ("TELEGRAM_BOT_TOKEN",)),
        )

    def test_env_allowlist_parser_validates_names_and_deduplicates(self) -> None:
        self.assertEqual(_parse_mcp_env_vars("MCP_TOKEN, OTHER_KEY, MCP_TOKEN"), ("MCP_TOKEN", "OTHER_KEY"))
        for value in ("MCP_TOKEN,,OTHER", "BAD-NAME", "2BAD"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _parse_mcp_env_vars(value)
        with self.assertRaisesRegex(ValueError, "cannot include"):
            _parse_mcp_env_vars("TELEGRAM_BOT_TOKEN")

    def test_load_config_exposes_only_the_mcp_env_name_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "101",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
                "TELECODEX_MCP_ENV_VARS": "DOCS_TOKEN, OTHER_SECRET",
                "DOCS_TOKEN": "do-not-print",
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                config = load_config()
        self.assertEqual(config.mcp_env_vars, ("DOCS_TOKEN", "OTHER_SECRET"))
        self.assertNotIn("do-not-print", repr(config))


class FakeMcpAppServer:
    def __init__(self) -> None:
        self.is_running = True
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.config: dict[str, Any] = {"mcp_servers": {}}
        self.version = "user-v1"
        self.status_pages: list[dict[str, Any]] = []
        self.oauth_result = {"authorizationUrl": "https://example.test/authorize?client=codex", "loginId": "login-1"}

    async def start(self) -> None:
        self.is_running = True

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_seconds: float = 120,
    ) -> dict[str, Any]:
        params = params or {}
        self.requests.append((method, params))
        if method == "mcpServerStatus/list":
            return self.status_pages.pop(0) if self.status_pages else {"data": [], "nextCursor": None}
        if method == "config/read":
            return {
                "config": self.config,
                "layers": [{"name": {"type": "user"}, "version": self.version}],
                "origins": {},
            }
        if method == "config/batchWrite":
            for edit in params.get("edits", []):
                path = edit["keyPath"].split(".")
                target = self.config
                for part in path[:-1]:
                    target = target.setdefault(part, {})
                target[path[-1]] = edit["value"]
            self.version += "x"
            return {}
        if method == "mcpServer/oauth/login":
            return self.oauth_result
        return {}


class CodexMcpSessionTests(unittest.IsolatedAsyncioTestCase):
    def _session(self, workspace: Path, app_server: FakeMcpAppServer, on_status: Any = None) -> CodexSession:
        config = AppConfig(
            telegram_bot_token="token",
            telegram_allowed_user_ids=(101,),
            workspace_root=workspace,
            codex_bin="codex",
            codex_model=None,
            network_access=False,
            approval_policy="on-request",
            shell_timeout_seconds=60,
            database_file=workspace / "state.sqlite3",
            mcp_env_vars=("MCP_FIXTURE_TOKEN",),
        )
        store = RuntimeStateStore(config.database_file).for_user(101)

        async def no_status(*_args: Any) -> None:
            return None

        async def no_callback(*_args: Any) -> None:
            return None

        async def no_prompt(*_args: Any) -> dict[str, Any]:
            return {}

        return CodexSession(
            config,
            store,
            on_status or no_status,
            no_callback,
            no_callback,
            no_prompt,
            app_server_factory=lambda *_args: app_server,
        )

    async def test_list_mcp_servers_paginates_without_loading_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_server = FakeMcpAppServer()
            app_server.status_pages = [
                {
                    "data": [{"name": "one", "authStatus": "unknown", "tools": {}, "resources": [{"uri": "secret"}]}],
                    "nextCursor": "cursor-2",
                },
                {
                    "data": [{"name": "two", "authStatus": "notLoggedIn", "runtimeStatus": "failed", "tools": {"a": {}}}],
                    "nextCursor": None,
                },
            ]
            session = self._session(Path(temp_dir), app_server)
            session.state.thread_id = "thread-1"
            session._loaded_thread_id = "thread-1"

            servers = await session.list_mcp_servers()

        self.assertEqual([item.name for item in servers], ["one", "two"])
        self.assertEqual(app_server.requests[0][1]["detail"], "toolsAndAuthOnly")
        self.assertEqual(app_server.requests[0][1]["threadId"], "thread-1")
        self.assertEqual(app_server.requests[1][1]["cursor"], "cursor-2")

    async def test_list_falls_back_to_app_scope_when_selected_thread_is_busy_elsewhere(self) -> None:
        class ThreadRejectingAppServer(FakeMcpAppServer):
            async def request(self, method, params=None, timeout_seconds=120):
                if method == "mcpServerStatus/list" and params and params.get("threadId"):
                    self.requests.append((method, params))
                    raise RpcError(-32600, "thread is active in another app-server")
                return await super().request(method, params, timeout_seconds)

        with tempfile.TemporaryDirectory() as temp_dir:
            app_server = ThreadRejectingAppServer()
            app_server.status_pages = [{
                "data": [{"name": "shared", "authStatus": "unknown", "tools": {}}],
                "nextCursor": None,
            }]
            session = self._session(Path(temp_dir), app_server)
            session.state.thread_id = "busy-thread"
            session._loaded_thread_id = "busy-thread"

            servers = await session.list_mcp_servers()

        self.assertEqual([server.name for server in servers], ["shared"])
        self.assertIn("threadId", app_server.requests[0][1])
        self.assertNotIn("threadId", app_server.requests[1][1])

    async def test_list_does_not_attach_to_a_selected_but_unloaded_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_server = FakeMcpAppServer()
            app_server.status_pages = [{"data": [], "nextCursor": None}]
            session = self._session(Path(temp_dir), app_server)
            session.state.thread_id = "unloaded-thread"

            await session.list_mcp_servers()

        self.assertNotIn("threadId", app_server.requests[0][1])

    async def test_add_and_toggle_write_only_the_named_server(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_server = FakeMcpAppServer()
            session = self._session(Path(temp_dir), app_server)

            await session.add_mcp_stdio_server(
                "fixture",
                "python3",
                ("-m", "fixture"),
                None,
                ("MCP_FIXTURE_TOKEN",),
            )
            await session.set_mcp_server_enabled("fixture", False)

        self.assertEqual(
            app_server.config["mcp_servers"]["fixture"],
            {
                "command": "python3",
                "args": ["-m", "fixture"],
                "env_vars": ["MCP_FIXTURE_TOKEN"],
                "enabled": False,
            },
        )
        writes = [params for method, params in app_server.requests if method == "config/batchWrite"]
        self.assertEqual(writes[0]["edits"][0]["keyPath"], "mcp_servers.fixture")
        self.assertEqual(writes[0]["expectedVersion"], "user-v1")

    async def test_mcp_tool_and_oauth_events_report_only_safe_metadata(self) -> None:
        statuses: list[tuple[str, object | None, str | None]] = []

        async def capture_status(key: str, detail: object | None, item_id: str | None) -> None:
            statuses.append((key, detail, item_id))

        with tempfile.TemporaryDirectory() as temp_dir:
            app_server = FakeMcpAppServer()
            session = self._session(Path(temp_dir), app_server, capture_status)
            url, login_id = await session.start_mcp_oauth("docs")
            self.assertTrue(url.startswith("https://example.test/authorize"))
            session._on_event({
                "method": "mcpServer/oauthLogin/completed",
                "params": {"name": "docs", "loginId": login_id, "success": False, "error": "private-token"},
            })
            session._on_event({
                "method": "item/started",
                "params": {"item": {"id": "item-1", "type": "mcpToolCall", "server": "docs", "tool": "search", "arguments": {"token": "secret"}}},
            })
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        self.assertEqual(statuses[0], ("status.label.mcp_oauth_failed", "docs", None))
        self.assertEqual(statuses[1], ("detailed_log.values.mcp_started", {"server": "docs", "tool": "search"}, "item-1"))
        self.assertNotIn("private-token", repr(statuses))
        self.assertNotIn("secret", repr(statuses))

    async def test_app_server_can_send_null_params_for_mcp_reload_schema(self) -> None:
        class Process:
            returncode = None

        class ReaderTask:
            @staticmethod
            def done() -> bool:
                return False

        rpc = object.__new__(JsonRpcProcess)
        rpc.process = Process()
        rpc._closed = False
        rpc._reader_task = ReaderTask()
        rpc._request_ids = itertools.count(1)
        rpc._pending = {}
        rpc._write_lock = asyncio.Lock()
        written: list[dict[str, Any]] = []

        async def write(message: dict[str, Any]) -> None:
            written.append(message)
            rpc._pending[message["id"]].set_result({"result": {}})

        rpc._write = write
        await rpc.request("config/mcpServer/reload", params_as_null=True)

        self.assertIsNone(written[0]["params"])

    async def test_remove_uses_argv_without_a_shell(self) -> None:
        class Process:
            returncode = 0

            async def communicate(self) -> tuple[bytes, bytes]:
                return b"", b""

        with tempfile.TemporaryDirectory() as temp_dir:
            session = self._session(Path(temp_dir), FakeMcpAppServer())
            with patch(
                "telecodex.codex.session.asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=Process()),
            ) as create_process:
                await session.remove_mcp_server("fixture")

        args, kwargs = create_process.call_args
        self.assertEqual(args[:4], ("codex", "mcp", "remove", "fixture"))
        self.assertNotIn("shell", kwargs)

    async def test_oauth_callback_detection_distinguishes_loopback_and_remote_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_server = FakeMcpAppServer()
            session = self._session(Path(temp_dir), app_server)

            self.assertFalse(await session.mcp_oauth_uses_remote_callback("fixture"))
            app_server.config["mcp_oauth_callback_url"] = "http://127.0.0.1/callback"
            self.assertFalse(await session.mcp_oauth_uses_remote_callback("fixture"))
            app_server.config["mcp_oauth_callback_url"] = "https://oauth.example.test/callback"
            self.assertTrue(await session.mcp_oauth_uses_remote_callback("fixture"))
            app_server.config["mcp_servers"] = {
                "fixture": {"oauth": {"callback_url": "http://localhost/callback"}}
            }
            self.assertFalse(await session.mcp_oauth_uses_remote_callback("fixture"))


class FakeTelegram:
    def __init__(self) -> None:
        self.id = 999
        self.messages: list[tuple[int, str]] = []
        self.markups: list[dict[str, Any] | None] = []
        self.edits: list[tuple[int, int, str]] = []
        self.callbacks: list[tuple[str, str]] = []
        self.next_message_id = 1
        self.bot_user_id = 999

    async def send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> FakeSentMessage:
        self.messages.append((chat_id, text))
        self.markups.append(reply_markup)
        message_id = self.next_message_id
        self.next_message_id += 1
        return FakeSentMessage(message_id=message_id)

    async def send_rich_message(
        self,
        chat_id: int,
        rich_message: Any,
        *,
        reply_parameters: Any = None,
        disable_notification: bool = False,
    ) -> FakeSentMessage:
        markdown = (
            rich_message.get("markdown", "")
            if isinstance(rich_message, dict)
            else getattr(rich_message, "markdown", "")
        )
        reply_to_message_id = reply_parameters.message_id if reply_parameters is not None else None
        return await self.send_message(
            chat_id,
            markdown,
            reply_to_message_id=reply_to_message_id,
            disable_notification=disable_notification,
        )

    async def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        self.edits.append((chat_id, message_id, text))
        self.markups.append(reply_markup)

    async def answer_callback_query(self, callback_query_id: str, text: str | None = None) -> None:
        self.callbacks.append((callback_query_id, text or ""))


class FakeMcpCodex:
    def __init__(self) -> None:
        self.state = type("State", (), {"thread_id": None, "output_mode": "detailed"})()
        self.is_busy = False
        self.compaction_barrier_pending = False
        self.servers = []
        self.added: list[tuple[Any, ...]] = []
        self.http_added: list[tuple[Any, ...]] = []
        self.enabled_calls: list[tuple[str, bool]] = []
        self.remote_oauth_callback = True
        self.oauth_calls = 0
        self.working_directory = Path("/tmp")

    async def list_mcp_servers(self) -> list[Any]:
        return self.servers

    async def add_mcp_stdio_server(self, *args: Any) -> None:
        self.added.append(args)

    async def add_mcp_http_server(self, *args: Any) -> None:
        self.http_added.append(args)

    async def set_mcp_server_enabled(self, name: str, enabled: bool) -> None:
        self.enabled_calls.append((name, enabled))

    async def start_mcp_oauth(self, _name: str) -> tuple[str, str]:
        self.oauth_calls += 1
        return "https://example.test/oauth/authorize?state=opaque", "login-test"

    async def mcp_oauth_uses_remote_callback(self, _name: str) -> bool:
        return self.remote_oauth_callback


class McpTelegramTests(unittest.IsolatedAsyncioTestCase):
    def _controller(self, workspace: Path) -> tuple[TelegramUserController, FakeTelegram, FakeMcpCodex]:
        config = AppConfig(
            telegram_bot_token="token",
            telegram_allowed_user_ids=(101,),
            workspace_root=workspace,
            codex_bin="codex",
            codex_model=None,
            network_access=False,
            approval_policy="on-request",
            shell_timeout_seconds=60,
            database_file=workspace / "state.sqlite3",
        )
        telegram = FakeTelegram()
        controller = TelegramUserController(config, telegram, object(), 101)  # type: ignore[arg-type]
        codex = FakeMcpCodex()
        controller.bind_codex(codex)  # type: ignore[arg-type]
        return controller, telegram, codex

    async def test_add_wizard_requires_explicit_confirmation_and_uses_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            controller, telegram, codex = self._controller(workspace)
            await controller._handle_message(message_event(1, 1, 101, 101, "private", "/mcp add"))
            picker = controller.mcp_picker
            self.assertIsNotNone(picker)
            assert picker is not None
            self.assertEqual(controller.mcp_add_wizard.stage, "transport")  # type: ignore[union-attr]

            await controller._handle_callback(
                callback_event(2, "transport", 101, 101, picker.message_id or 1, f"mc:{picker.token}:stdio")
            )
            wizard = controller.mcp_add_wizard
            assert wizard is not None
            for value, expected_stage in (
                ("fixture", "command"),
                ("python3", "args"),
                ('["-m", "fixture"]', "cwd"),
                ("-", "env_vars"),
                ("-", "confirm"),
            ):
                await controller._handle_message(
                    message_event(3, 3, 101, 101, "private", value)
                )
                self.assertEqual(wizard.stage, expected_stage)
                self.assertEqual(codex.added, [])

            await controller._handle_callback(
                callback_event(4, "confirm", 101, 101, picker.message_id or 1, f"mc:{picker.token}:add_confirm")
            )

        self.assertEqual(codex.added, [("fixture", "python3", ("-m", "fixture"), None, ())])
        self.assertTrue(any("общей" in text for _, _, text in telegram.edits))

    async def test_direct_group_add_wizard_tracks_the_operator_not_the_group_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, _, _ = self._controller(Path(temp_dir))
            controller.is_group = True
            controller.chat_id = -100123
            controller.user_id = -100123
            await controller._handle_message(
                message_event(1, 1, 101, -100123, "group", "/mcp add")
            )

        self.assertEqual(controller.mcp_add_wizard.owner_user_id, 101)  # type: ignore[union-attr]

    async def test_mcp_login_command_returns_an_authorization_url_button(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, _ = self._controller(Path(temp_dir))

            await controller._handle_message(
                message_event(1, 1, 101, 101, "private", "/mcp login fixture")
            )

        self.assertIn("Открой страницу авторизации", telegram.messages[0][1])
        markup = telegram.markups[0]
        assert markup is not None
        button = markup["inline_keyboard"][0][0]
        self.assertEqual(button["url"], "https://example.test/oauth/authorize?state=opaque")

    async def test_mcp_login_uses_cli_fallback_for_loopback_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, codex = self._controller(Path(temp_dir))
            codex.remote_oauth_callback = False

            await controller._handle_message(
                message_event(1, 1, 101, 101, "private", "/mcp login fixture")
            )

        self.assertIn("codex mcp login fixture --no-browser", telegram.messages[0][1])
        self.assertIn(
            "Не отправляй redirect URL или код в Telegram",
            _plain_html(telegram.messages[0][1]),
        )
        self.assertEqual(codex.oauth_calls, 0)

    async def test_http_wizard_collects_oauth_configuration_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, codex = self._controller(Path(temp_dir))
            await controller._handle_message(message_event(1, 1, 101, 101, "private", "/mcp add"))
            picker = controller.mcp_picker
            assert picker is not None and picker.message_id is not None
            transport_rows = telegram.markups[-1]["inline_keyboard"]
            self.assertEqual(
                [row[0]["text"] for row in transport_rows[:2]],
                ["Локальный процесс (STDIO)", "Streamable HTTP"],
            )
            self.assertEqual([len(row) for row in transport_rows[:2]], [1, 1])
            await controller._handle_callback(
                callback_event(2, "http", 101, 101, picker.message_id, f"mc:{picker.token}:http")
            )
            wizard = controller.mcp_add_wizard
            assert wizard is not None
            await controller._handle_message(message_event(3, 2, 101, 101, "private", "docs"))
            await controller._handle_message(message_event(4, 3, 101, 101, "private", "https://example.test/mcp"))
            await controller._handle_callback(
                callback_event(5, "oauth", 101, 101, picker.message_id, f"mc:{picker.token}:auth:oauth")
            )
            self.assertEqual(wizard.stage, "confirm")
            self.assertNotIn("token=", controller._mcp_add_summary(wizard))
            await controller._handle_callback(
                callback_event(6, "confirm", 101, 101, picker.message_id, f"mc:{picker.token}:add_confirm")
            )

        self.assertEqual(codex.http_added, [("docs", "https://example.test/mcp", "oauth", None)])
        self.assertTrue(
            any("Сервер docs добавлен" in _plain_html(text) for _, text in telegram.messages)
        )

    async def test_server_disable_requires_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, codex = self._controller(Path(temp_dir))
            codex.servers = [McpServer("docs", "connected", "unknown", (), False)]
            await controller._handle_message(message_event(1, 1, 101, 101, "private", "/mcp"))
            picker = controller.mcp_picker
            assert picker is not None and picker.message_id is not None
            await controller._handle_callback(
                callback_event(2, "server", 101, 101, picker.message_id, f"mc:{picker.token}:server:0")
            )
            await controller._handle_callback(
                callback_event(3, "disable", 101, 101, picker.message_id, f"mc:{picker.token}:disable")
            )
            self.assertEqual(codex.enabled_calls, [])
            self.assertIn("Отключить MCP-сервер docs", _plain_html(telegram.edits[-1][2]))
            await controller._handle_callback(
                callback_event(4, "confirm", 101, 101, picker.message_id, f"mc:{picker.token}:action_confirm")
            )

        self.assertEqual(codex.enabled_calls, [("docs", False)])

    async def test_mcp_form_elicitation_validates_input_and_returns_structured_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, _ = self._controller(Path(temp_dir))
            task = asyncio.create_task(controller._ask_mcp_elicitation({
                "mode": "form",
                "serverName": "fixture",
                "message": "Enter project name",
                "requestedSchema": {
                    "type": "object",
                    "properties": {"project": {"type": "string", "minLength": 2}},
                    "required": ["project"],
                },
            }))
            await asyncio.sleep(0)
            await controller._handle_message(message_event(1, 2, 101, 101, "private", "{}"))
            self.assertFalse(task.done())
            await controller._handle_message(
                message_event(2, 3, 101, 101, "private", '{"project":"demo"}')
            )
            response = await task

        self.assertEqual(response, {"action": "accept", "content": {"project": "demo"}})
        self.assertTrue(any("не подошёл" in text for _, text in telegram.messages))

    async def test_empty_mcp_form_is_an_explicit_confirm_or_decline_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, _ = self._controller(Path(temp_dir))
            task = asyncio.create_task(controller._ask_mcp_elicitation({
                "mode": "form",
                "serverName": "openrouter",
                "message": "Allow this request?",
                "requestedSchema": {"type": "object", "properties": {}, "required": []},
            }))
            await asyncio.sleep(0)
            pending = next(iter(controller.pending_mcp_elicitations.values()))
            self.assertEqual(pending.mode, "form_confirmation")
            callback = next(
                button["callback_data"]
                for row in telegram.markups[0]["inline_keyboard"]
                for button in row
                if button["callback_data"].endswith(":accept")
            )
            await controller._handle_mcp_elicitation_callback(
                callback_event(1, "confirm", 101, 101, pending.message_id, callback),
                callback.split(":"),
            )
            response = await task

        self.assertEqual(response, {"action": "accept", "content": {}})

    async def test_mcp_url_elicitation_requires_a_user_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram, _ = self._controller(Path(temp_dir))
            task = asyncio.create_task(controller._ask_mcp_elicitation({
                "mode": "url",
                "serverName": "fixture",
                "message": "Approve connection",
                "url": "https://example.test/approve?request=abc",
            }))
            await asyncio.sleep(0)
            pending = next(iter(controller.pending_mcp_elicitations.values()))
            await controller._handle_mcp_elicitation_callback(
                callback_event(1, "approve", 101, 101, pending.message_id, f"me:{pending.token}:accept"),
                ["me", pending.token, "accept"],
            )
            response = await task

        self.assertEqual(response, {"action": "accept"})
        self.assertIn("MCP-сервер fixture", telegram.messages[0][1])


class McpReloadBroadcastTests(unittest.IsolatedAsyncioTestCase):
    async def test_reload_broadcasts_to_all_active_contexts_and_counts_failures(self) -> None:
        class Context:
            def __init__(self, raises: bool = False) -> None:
                self.raises = raises

            def _codex(self) -> Any:
                return self

            async def reload_mcp_config(self) -> bool:
                if self.raises:
                    raise RuntimeError("private rpc detail")
                return True

        controller = object.__new__(TelegramController)
        controller.users = {101: Context()}
        controller.groups = {-1001: Context(raises=True)}

        result = await controller.reload_mcp_config()

        self.assertEqual(result, (1, 1))
