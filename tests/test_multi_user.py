from __future__ import annotations
from pathlib import Path
from typing import Any
from unittest.mock import patch
import asyncio
import json
import tempfile
import unittest
from telecodex.config import AppConfig, ProxySettings, load_config
from telecodex.state import PendingTurnState, RuntimeState, RuntimeStateStore, ThreadRuntimeState
from telecodex.telegram.controller import TelegramController
from telecodex.telegram.user_controller import PendingApproval, TelegramUserController

from multi_user_test_support import (
    dispatch_controller_update,
    BOT_TRIGGER_CYRILLIC,
    CODEX_TRIGGER_CYRILLIC,
    FailingCodex,
    FakeCodex,
    FakeTelegram,
    HELPER_TRIGGER_CYRILLIC,
    KAWAII_TRIGGER_CYRILLIC,
    _catalog_text,
)

class MultiUserConfigTests(unittest.TestCase):
    def test_load_config_reads_local_audio_transcription_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "101",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
                "TELECODEX_TRANSCRIPTION_MODEL": "base",
                "TELECODEX_TRANSCRIPTION_DEVICE": "cpu",
                "TELECODEX_TRANSCRIPTION_COMPUTE_TYPE": "int8",
                "TELECODEX_AUDIO_MAX_DURATION_SECONDS": "120",
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                config = load_config()

        self.assertEqual(config.transcription_model, "base")
        self.assertEqual(config.transcription_device, "cpu")
        self.assertEqual(config.transcription_compute_type, "int8")
        self.assertEqual(config.max_audio_duration_seconds, 120)

    def test_load_config_reads_telegram_socks_proxy_and_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "101",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
                "TELEGRAM_PROXY_URL": "socks5://127.0.0.1:2080",
                "TELEGRAM_PROXY_USERNAME": "proxy-user",
                "TELEGRAM_PROXY_PASSWORD": "proxy-secret",
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                config = load_config()

        self.assertEqual(
            config.telegram_proxy,
            ProxySettings("socks5://127.0.0.1:2080", "proxy-user", "proxy-secret"),
        )
        self.assertNotIn("proxy-secret", repr(config))

    def test_load_config_leaves_telegram_proxy_unset_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "101",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                config = load_config()

        self.assertIsNone(config.telegram_proxy)

    def test_load_config_rejects_proxy_uri_with_embedded_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "101",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
                "TELEGRAM_PROXY_URL": "http://proxy-user:proxy-secret@127.0.0.1:2080",
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                with self.assertRaises(ValueError) as raised:
                    load_config()

        self.assertIn("TELEGRAM_PROXY_URL", str(raised.exception))
        self.assertNotIn("proxy-user", str(raised.exception))
        self.assertNotIn("proxy-secret", str(raised.exception))

    def test_load_config_parses_multiple_ids_and_removes_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "202, 101,303",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                config = load_config()
        self.assertEqual(config.telegram_allowed_user_ids, (202, 101, 303))

    def test_load_config_rejects_empty_id_in_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_ALLOWED_USER_IDS": "101,,202",
                "CODEX_WORKSPACE_ROOT": temp_dir,
                "CODEX_BIN": "/bin/true",
                "XDG_STATE_HOME": temp_dir,
            }
            with patch("telecodex.config.load_dotenv"), patch.dict("os.environ", environment, clear=True):
                with self.assertRaisesRegex(ValueError, "TELEGRAM_ALLOWED_USER_IDS"):
                    load_config()


class MultiUserStateTests(unittest.TestCase):
    def test_migrates_legacy_state_and_keeps_user_threads_and_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            legacy_path = Path(temp_dir) / "state.json"
            legacy_path.write_text(json.dumps({"thread_id": "thread-old", "last_update_id": 55}))
            store = RuntimeStateStore(path, migration_user_id=101, migration_path=legacy_path)

            self.assertEqual(store.load_last_update_id(), 55)
            self.assertEqual(store.for_user(101).load(), RuntimeState("thread-old"))
            store.for_user(202).save(RuntimeState("thread-new", cwd="/project/child"))
            store.save_last_update_id(56)

            reloaded = RuntimeStateStore(path)
            self.assertEqual(reloaded.load_last_update_id(), 56)
            self.assertEqual(reloaded.for_user(101).load().thread_id, "thread-old")
            self.assertEqual(reloaded.for_user(202).load().thread_id, "thread-new")
            self.assertEqual(reloaded.for_user(202).load().cwd, "/project/child")
            self.assertEqual(json.loads(legacy_path.read_text()), {"thread_id": "thread-old", "last_update_id": 55})

    def test_deleting_threads_clears_metadata_and_all_user_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            user_one = root_store.for_user(101)
            user_two = root_store.for_user(202)
            user_one.save(RuntimeState("thread-shared", cwd="/project/one"))
            user_two.save(RuntimeState("thread-shared", cwd="/project/two"))
            user_one.save_thread(
                "thread-shared",
                ThreadRuntimeState(permission_mode="ask", token_usage={"context_tokens": 10}),
            )

            user_one.delete_threads({"thread-shared"})

            self.assertIsNone(user_one.load().thread_id)
            self.assertIsNone(user_two.load().thread_id)
            self.assertEqual(user_one.load().cwd, "/project/one")
            self.assertEqual(user_two.load().cwd, "/project/two")
            self.assertEqual(user_one.load_thread("thread-shared"), ThreadRuntimeState())

    def test_selected_effort_survives_state_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3").for_user(101)
            store.save(RuntimeState(selected_model="gpt-6-luna", selected_effort="xhigh"))

            self.assertEqual(store.load().selected_model, "gpt-6-luna")
            self.assertEqual(store.load().selected_effort, "xhigh")

    def test_output_mode_is_persisted_per_user_and_old_states_default_to_detailed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            legacy_path = Path(temp_dir) / "state.json"
            legacy_path.write_text(
                json.dumps(
                    {
                        "last_update_id": 0,
                        "users": {
                            "101": {"thread_id": "one"},
                            "202": {"thread_id": "two", "output_mode": "answers"},
                        },
                        "threads": {},
                    }
                )
            )
            store = RuntimeStateStore(path, migration_path=legacy_path)

            self.assertEqual(store.for_user(101).load().output_mode, "detailed")
            self.assertEqual(store.for_user(202).load().output_mode, "answers")
            store.for_user(101).save(RuntimeState(output_mode="answers"))

            reloaded = RuntimeStateStore(path)
            self.assertEqual(reloaded.for_user(101).load().output_mode, "answers")
            self.assertEqual(reloaded.for_user(202).load().output_mode, "answers")

    def test_pending_turn_survives_reload_and_is_isolated_by_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            user_one = root.for_user(101)
            user_two = root.for_user(202)
            pending = PendingTurnState(
                thread_id="thread-one",
                turn_id="turn-one",
                prompt_message_id=41,
                started_at=123,
                phase="running",
                prompt_message_ids=[42],
            )
            user_one.save(RuntimeState(pending_turn=pending))
            user_two.save(RuntimeState(pending_turn=PendingTurnState(
                thread_id="thread-two",
                turn_id="turn-two",
                prompt_message_id=51,
                started_at=124,
            )))

            self.assertEqual(user_one.load().pending_turn, pending)
            self.assertEqual(user_two.load().pending_turn.thread_id, "thread-two")  # type: ignore[union-attr]

            user_one.delete_threads({"thread-one"})

            self.assertIsNone(user_one.load().pending_turn)
            self.assertEqual(user_two.load().pending_turn.thread_id, "thread-two")  # type: ignore[union-attr]

    def test_thread_permissions_and_token_usage_are_shared(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            thread_state = ThreadRuntimeState(
                permission_mode="read_only",
                managed_by_telecodex=True,
                token_usage={"context_tokens": 12000, "model_context_window": 200000},
            )
            store.for_user(101).save_thread("shared-thread", thread_state)

            self.assertEqual(store.for_user(202).load_thread("shared-thread"), thread_state)

    def test_group_context_has_independent_state_and_answers_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            store = RuntimeStateStore(path)
            self.assertFalse(store.attach_group(-100123, "Private team"))

            group = store.for_group(-100123)
            self.assertEqual(group.load().output_mode, "answers")
            group.save(RuntimeState(thread_id="group-thread", cwd="/project/team", output_mode="detailed"))
            store.for_user(101).save(RuntimeState(thread_id="private-thread", cwd="/project/personal"))

            reloaded = RuntimeStateStore(path)
            self.assertEqual(reloaded.list_attached_groups(), {-100123: "Private team"})
            self.assertEqual(reloaded.for_group(-100123).load().thread_id, "group-thread")
            self.assertEqual(reloaded.for_group(-100123).load().output_mode, "detailed")
            self.assertEqual(reloaded.for_user(101).load().thread_id, "private-thread")

            self.assertTrue(reloaded.detach_group(-100123))
            self.assertFalse(reloaded.is_group_attached(-100123))
            self.assertEqual(reloaded.for_group(-100123).load().thread_id, "group-thread")

    def test_group_trigger_setup_persists_and_updates_without_replacing_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            store = RuntimeStateStore(path)
            self.assertEqual(store.begin_group_setup(-100124, "Team", 101), "started")
            self.assertFalse(store.is_group_attached(-100124))
            self.assertEqual(store.pending_group_setup_user(-100124), 101)

            reloaded = RuntimeStateStore(path)
            self.assertEqual(reloaded.pending_group_setup_user(-100124), 101)
            self.assertIsNone(reloaded.complete_group_setup(-100124, 202, ("codex",)))
            self.assertEqual(
                reloaded.complete_group_setup(
                    -100124, 101, (CODEX_TRIGGER_CYRILLIC, KAWAII_TRIGGER_CYRILLIC)
                ),
                "attached",
            )
            self.assertTrue(reloaded.is_group_attached(-100124))
            russian_triggers = (CODEX_TRIGGER_CYRILLIC, KAWAII_TRIGGER_CYRILLIC)
            self.assertEqual(reloaded.group_trigger_words(-100124), russian_triggers)

            reloaded.for_group(-100124).save(RuntimeState(thread_id="shared-thread"))
            self.assertEqual(reloaded.begin_group_setup(-100124, "Team", 202), "started")
            self.assertEqual(reloaded.group_trigger_words(-100124), russian_triggers)
            self.assertTrue(reloaded.cancel_group_setup(-100124))
            self.assertTrue(reloaded.is_group_attached(-100124))
            self.assertEqual(reloaded.for_group(-100124).load().thread_id, "shared-thread")
            self.assertEqual(reloaded.group_trigger_words(-100124), russian_triggers)

            self.assertEqual(reloaded.begin_group_setup(-100124, "Team", 202), "started")
            self.assertEqual(
                reloaded.complete_group_setup(-100124, 202, ("help",)),
                "updated",
            )
            self.assertEqual(reloaded.group_trigger_words(-100124), ("help",))
            self.assertEqual(reloaded.for_group(-100124).load().thread_id, "shared-thread")

    def test_old_group_state_loads_without_trigger_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            legacy_path = Path(temp_dir) / "state.json"
            legacy_path.write_text(
                json.dumps(
                    {
                        "last_update_id": 8,
                        "users": {},
                        "groups": {
                            "-100125": {
                                "title": "Older group",
                                "attached": True,
                                "thread_id": "old-group-thread",
                            }
                        },
                        "threads": {},
                        "pending_media_groups": {},
                    }
                ),
                encoding="utf-8",
            )
            store = RuntimeStateStore(path, migration_path=legacy_path)
            self.assertTrue(store.is_group_attached(-100125))
            self.assertEqual(store.group_trigger_words(-100125), ())
            self.assertEqual(store.for_group(-100125).load().thread_id, "old-group-thread")
            self.assertEqual(store.begin_group_setup(-100125, "Older group", 101), "started")
            self.assertEqual(
                store.complete_group_setup(-100125, 101, ("codex",)),
                "updated",
            )
            self.assertEqual(store.group_trigger_words(-100125), ("codex",))
            self.assertEqual(store.for_group(-100125).load().thread_id, "old-group-thread")

    def test_group_state_rejects_non_group_chat_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            with self.assertRaisesRegex(ValueError, "group chat_id"):
                store.attach_group(101, "not a group")


class MultiUserRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_messages_and_approval_callbacks_are_isolated_by_user(self) -> None:
        state_directory = tempfile.TemporaryDirectory(prefix="telecodex-routing-")
        config = AppConfig(
            telegram_bot_token="token",
            telegram_allowed_user_ids=(101, 202),
            workspace_root=Path("/project"),
            codex_bin="codex",
            codex_model=None,
            network_access=False,
            approval_policy="on-request",
            shell_timeout_seconds=60,
            database_file=Path(state_directory.name) / "state.sqlite3",
        )
        telegram = FakeTelegram()
        sessions: dict[int, FakeCodex] = {}

        def make_session(user_id: int, _controller: Any) -> FakeCodex:
            sessions[user_id] = FakeCodex()
            return sessions[user_id]

        controller = TelegramController(config, telegram, object(), make_session)  # type: ignore[arg-type]
        await dispatch_controller_update(controller, self._message_update(1, 101, "first task"))
        await dispatch_controller_update(controller, self._message_update(2, 202, "second task"))
        await dispatch_controller_update(controller, self._message_update(3, 303, "ignored"))

        self.assertEqual(sessions[101].messages, ["first task"])
        self.assertEqual(sessions[202].messages, ["second task"])
        self.assertEqual(set(sessions), {101, 202})
        await asyncio.sleep(0)
        self.assertEqual({chat_id for chat_id, _ in telegram.actions}, {101, 202})

        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        controller.users[101].pending_approvals["approval-token"] = PendingApproval(
            future=future,
            message_id=9,
            chat_id=101,
        )
        await dispatch_controller_update(controller, self._callback_update(4, 202, "approval-token", "once"))
        self.assertFalse(future.done())
        await dispatch_controller_update(controller, self._callback_update(5, 101, "approval-token", "once"))
        self.assertEqual(await future, "accept")

        await controller.close()
        state_directory.cleanup()

    async def test_rpc_failure_detail_is_reported_in_chat(self) -> None:
        state_directory = tempfile.TemporaryDirectory(prefix="telecodex-routing-")
        config = AppConfig(
            telegram_bot_token="token",
            telegram_allowed_user_ids=(101,),
            workspace_root=Path("/project"),
            codex_bin="codex",
            codex_model=None,
            network_access=False,
            approval_policy="on-request",
            shell_timeout_seconds=60,
            database_file=Path(state_directory.name) / "state.sqlite3",
        )
        telegram = FakeTelegram()
        controller = TelegramController(
            config,
            telegram,
            object(),
            lambda _user_id, _controller: FailingCodex(),
        )  # type: ignore[arg-type]

        await dispatch_controller_update(controller, self._message_update(1, 101, "task"))

        self.assertTrue(
            any(
                "RPC -32602: Invalid request: unsupported request field" in text
                for _chat_id, text in telegram.messages
            )
        )
        await controller.close()
        state_directory.cleanup()

    async def test_group_is_one_shared_context_and_mode_is_available_to_members(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101, 202),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=Path(temp_dir) / "state.sqlite3",
            )
            telegram = FakeTelegram()
            sessions: dict[int, FakeCodex] = {}

            def make_session(context_id: int, _controller: Any) -> FakeCodex:
                codex = FakeCodex()
                if context_id < 0:
                    codex.state.output_mode = state_store.for_group(context_id).load().output_mode
                sessions[context_id] = codex
                return codex

            controller = TelegramController(
                config,
                telegram,
                object(),
                make_session,
                state_store=state_store,
            )  # type: ignore[arg-type]
            await dispatch_controller_update(controller, self._group_message_update(1, -100900, 303, "/group_add"))
            self.assertFalse(state_store.is_group_attached(-100900))
            self.assertEqual(telegram.messages, [])

            await dispatch_controller_update(controller, self._group_message_update(2, -100900, 101, "/group_add", "Dev team"))
            self.assertFalse(state_store.is_group_attached(-100900))
            self.assertEqual(state_store.pending_group_setup_user(-100900), 101)
            self.assertEqual(telegram.messages[-1][1], _catalog_text("groups.label.setup_prompt"))
            await dispatch_controller_update(controller, self._group_message_update(25, -100900, 101, "several words, bad!"))
            self.assertFalse(state_store.is_group_attached(-100900))
            self.assertEqual(state_store.pending_group_setup_user(-100900), 101)
            await dispatch_controller_update(controller, self._group_message_update(3, -100900, 202, "other, list"))
            self.assertFalse(state_store.is_group_attached(-100900))
            await dispatch_controller_update(controller,
                self._group_message_update(
                    4,
                    -100900,
                    101,
                    f"{CODEX_TRIGGER_CYRILLIC.capitalize()}, {KAWAII_TRIGGER_CYRILLIC}",
                )
            )
            self.assertTrue(state_store.is_group_attached(-100900))
            codex_triggers = (CODEX_TRIGGER_CYRILLIC, KAWAII_TRIGGER_CYRILLIC)
            self.assertEqual(state_store.group_trigger_words(-100900), codex_triggers)
            await dispatch_controller_update(controller, self._group_message_update(5, -100900, 101, "plain text is ignored"))
            await dispatch_controller_update(controller,
                self._group_message_update(6, -100900, 101, f"{CODEX_TRIGGER_CYRILLIC.upper()}, first task")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(7, -100900, 202, f"{KAWAII_TRIGGER_CYRILLIC} second task")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(8, -100900, 202, f"{KAWAII_TRIGGER_CYRILLIC}ny third task")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(9, -100900, 303, f"{CODEX_TRIGGER_CYRILLIC} ignored")
            )
            await dispatch_controller_update(controller, self._message_update(10, 101, "private task"))
            await dispatch_controller_update(controller,
                self._group_message_update(26, -100900, 101, CODEX_TRIGGER_CYRILLIC.upper())
            )
            self.assertEqual(telegram.messages[-1][1], _catalog_text("groups.label.trigger_empty"))

            self.assertEqual(sessions[-100900].messages, ["first task", "second task"])
            self.assertEqual(sessions[101].messages, ["private task"])
            self.assertNotIn(303, sessions)

            await dispatch_controller_update(controller, self._group_message_update(11, -100900, 202, "/group_add"))
            self.assertTrue(state_store.is_group_attached(-100900))
            self.assertEqual(state_store.group_trigger_words(-100900), codex_triggers)
            await dispatch_controller_update(controller,
                self._group_message_update(12, -100900, 101, f"{KAWAII_TRIGGER_CYRILLIC} old list still works")
            )
            helper_triggers = (BOT_TRIGGER_CYRILLIC, HELPER_TRIGGER_CYRILLIC)
            await dispatch_controller_update(controller,
                self._group_message_update(13, -100900, 202, ", ".join(helper_triggers))
            )
            self.assertEqual(state_store.group_trigger_words(-100900), helper_triggers)
            await dispatch_controller_update(controller,
                self._group_message_update(14, -100900, 101, f"{CODEX_TRIGGER_CYRILLIC} old list stopped")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(15, -100900, 202, f"{BOT_TRIGGER_CYRILLIC.capitalize()}: next task")
            )
            self.assertEqual(
                sessions[-100900].messages,
                ["first task", "second task", "old list still works", "next task"],
            )

            await dispatch_controller_update(controller, self._group_message_update(16, -100900, 202, "/mode"))
            group_controller = controller.groups[-100900]
            self.assertEqual(group_controller._output_mode(), "answers")
            picker = group_controller.mode_picker
            self.assertIsNotNone(picker)
            await dispatch_controller_update(controller,
                self._group_callback_update(
                    17,
                    -100900,
                    303,
                    picker.message_id,  # type: ignore[union-attr]
                    f"om:{picker.token}:detailed",  # type: ignore[union-attr]
                )
            )
            self.assertEqual(group_controller._output_mode(), "answers")
            await dispatch_controller_update(controller,
                self._group_callback_update(
                    18,
                    -100900,
                    202,
                    picker.message_id,  # type: ignore[union-attr]
                    f"om:{picker.token}:detailed",  # type: ignore[union-attr]
                )
            )
            self.assertEqual(group_controller._output_mode(), "detailed")

            sessions[-100900].working_directory = Path(temp_dir)
            await controller._handle_media_group(
                [
                    self._group_attachment_update(19, -100900, 101, "album-1", "one.txt"),
                    self._group_attachment_update(
                        20,
                        -100900,
                        101,
                        "album-2",
                        "two.txt",
                        caption=f"{HELPER_TRIGGER_CYRILLIC.upper()}, inspect these files",
                    ),
                ]
            )
            self.assertEqual(len(sessions[-100900].messages), 5)
            self.assertIn("one.txt", sessions[-100900].messages[-1])
            self.assertIn("two.txt", sessions[-100900].messages[-1])

            await dispatch_controller_update(controller, self._group_message_update(21, -100900, 101, "/group_remove"))
            self.assertFalse(state_store.is_group_attached(-100900))
            self.assertTrue(sessions[-100900].closed)
            await dispatch_controller_update(controller, self._group_message_update(22, -100900, 202, "after detach"))
            self.assertEqual(len(sessions[-100900].messages), 5)
            await dispatch_controller_update(controller, self._group_message_update(23, -100900, 202, "/group_add"))
            self.assertEqual(state_store.pending_group_setup_user(-100900), 202)
            await dispatch_controller_update(controller, self._group_message_update(24, -100900, 101, "/group_remove"))
            self.assertIsNone(state_store.pending_group_setup_user(-100900))
            self.assertFalse(state_store.is_group_attached(-100900))
            await controller.close()

    async def test_group_all_messages_mode_is_opt_in_and_keeps_the_group_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "state.sqlite3"
            state_store = RuntimeStateStore(database)
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101, 202),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=database,
            )
            telegram = FakeTelegram()
            sessions: dict[int, FakeCodex] = {}

            def make_session(context_id: int, _controller: Any) -> FakeCodex:
                codex = FakeCodex()
                sessions[context_id] = codex
                return codex

            controller = TelegramController(
                config,
                telegram,
                object(),
                make_session,
                state_store=state_store,
            )  # type: ignore[arg-type]
            await dispatch_controller_update(controller,
                self._group_message_update(1, -100920, 101, "/group_add", "Work chat")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(2, -100920, 101, CODEX_TRIGGER_CYRILLIC)
            )

            self.assertFalse(state_store.group_all_messages_to_codex(-100920))
            await dispatch_controller_update(controller,
                self._group_message_update(3, -100920, 101, "small talk stays in Telegram")
            )
            self.assertEqual(sessions[-100920].messages, [])

            await dispatch_controller_update(controller,
                self._group_message_update(4, -100920, 101, "/group_mode")
            )
            group_controller = controller.groups[-100920]
            picker = group_controller.group_mode_picker
            self.assertIsNotNone(picker)
            assert picker is not None and picker.message_id is not None
            mode_markup = telegram.reply_markups[-1]
            all_callback = mode_markup["inline_keyboard"][1][0]["callback_data"]

            await dispatch_controller_update(controller,
                self._group_callback_update(5, -100920, 303, picker.message_id, all_callback)
            )
            self.assertFalse(state_store.group_all_messages_to_codex(-100920))
            await dispatch_controller_update(controller,
                self._group_callback_update(6, -100920, 202, picker.message_id, all_callback)
            )
            self.assertTrue(state_store.group_all_messages_to_codex(-100920))
            self.assertTrue(RuntimeStateStore(database).group_all_messages_to_codex(-100920))

            await dispatch_controller_update(controller,
                self._group_message_update(7, -100920, 101, "first ordinary task")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(8, -100920, 202, "second ordinary task")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(9, -100920, 303, "not on the allowlist")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(10, -100920, 101, f"{CODEX_TRIGGER_CYRILLIC}, stripped trigger")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(11, -100920, 101, "/status")
            )

            self.assertEqual(
                sessions[-100920].messages,
                ["first ordinary task", "second ordinary task", "stripped trigger"],
            )
            self.assertIn("Codex: ready", telegram.messages[-1][1])

            reply_callback = telegram.edit_markups[-1]
            stop_all_callback = reply_callback["inline_keyboard"][0][0]["callback_data"]
            await dispatch_controller_update(controller,
                self._group_callback_update(12, -100920, 101, picker.message_id, stop_all_callback)
            )
            self.assertFalse(state_store.group_all_messages_to_codex(-100920))
            await dispatch_controller_update(controller,
                self._group_message_update(13, -100920, 101, "ordinary message again ignored")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(14, -100920, 101, f"{CODEX_TRIGGER_CYRILLIC} trigger still works")
            )
            self.assertEqual(
                sessions[-100920].messages,
                [
                    "first ordinary task",
                    "second ordinary task",
                    "stripped trigger",
                    "trigger still works",
                ],
            )
            await controller.close()

    async def test_group_all_messages_mode_forwards_media_without_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "state.sqlite3"
            state_store = RuntimeStateStore(database)
            state_store.attach_group(-100921, "Work media", ("codex",))
            self.assertTrue(state_store.set_group_all_messages_to_codex(-100921, True))
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101,),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=database,
            )
            telegram = FakeTelegram()
            codex = FakeCodex()
            codex.working_directory = Path(temp_dir)
            user_controller = TelegramUserController(
                config,
                telegram,
                object(),
                -100921,
                is_group=True,
                group_trigger_words=lambda: state_store.group_trigger_words(-100921),
                group_all_messages_to_codex=lambda: state_store.group_all_messages_to_codex(-100921),
            )  # type: ignore[arg-type]
            user_controller.attached = True
            user_controller.bind_codex(codex)  # type: ignore[arg-type]

            await user_controller._handle_media_group(
                [
                    self._group_attachment_update(1, -100921, 101, "media-1", "one.txt"),
                    self._group_attachment_update(2, -100921, 101, "media-2", "two.txt"),
                ]
            )

            self.assertEqual(len(codex.messages), 1)
            self.assertIn("one.txt", codex.messages[0])
            self.assertIn("two.txt", codex.messages[0])
            await user_controller._finish_typing()

    async def test_group_access_is_limited_to_global_operators(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            state_store.attach_group(-100910, "Operator group", ("codex",))
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101,),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=Path(temp_dir) / "state.sqlite3",
            )
            telegram = FakeTelegram()
            sessions: dict[int, FakeCodex] = {}

            def make_session(context_id: int, _controller: Any) -> FakeCodex:
                session = FakeCodex()
                sessions[context_id] = session
                return session

            controller = TelegramController(
                config,
                telegram,
                object(),
                make_session,
                state_store=state_store,
            )  # type: ignore[arg-type]

            await dispatch_controller_update(controller,
                self._group_message_update(1, -100910, 101, "codex allowed task")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(2, -100910, 303, "codex ignored task")
            )
            await dispatch_controller_update(controller,
                self._group_message_update(3, -100910, 303, "/add_user 303")
            )
            await dispatch_controller_update(controller, self._message_update(4, 303, "private attempt"))
            self.assertEqual(sessions[-100910].messages, ["allowed task"])
            self.assertEqual(set(sessions), {-100910})
            self.assertFalse(state_store.is_group_attached(-100911))
            await controller.close()

    async def test_start_restores_attached_group_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.sqlite3"
            state_store = RuntimeStateStore(path)
            state_store.attach_group(-100901, "Persistent group")
            state_store.for_group(-100901).save(
                RuntimeState(thread_id="persisted-group-thread", output_mode="detailed")
            )
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101,),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=path,
            )
            telegram = FakeTelegram()
            sessions: dict[int, FakeCodex] = {}

            def make_session(context_id: int, _controller: Any) -> FakeCodex:
                codex = FakeCodex()
                codex.state = (
                    state_store.for_group(context_id).load()
                    if context_id < 0
                    else state_store.for_user(context_id).load()
                )
                sessions[context_id] = codex
                return codex

            controller = TelegramController(
                config,
                telegram,
                object(),
                make_session,
                state_store=state_store,
            )  # type: ignore[arg-type]
            await controller.start()

            self.assertIn(-100901, controller.groups)
            self.assertTrue(sessions[-100901].started)
            self.assertEqual(sessions[-100901].state.thread_id, "persisted-group-thread")
            self.assertEqual(sessions[-100901].state.output_mode, "detailed")
            await controller.close()

    @staticmethod
    def _message_update(update_id: int, user_id: int, text: str) -> dict[str, Any]:
        return {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "from": {"id": user_id},
                "chat": {"id": user_id, "type": "private"},
                "text": text,
            },
        }

    @staticmethod
    def _callback_update(update_id: int, user_id: int, token: str, choice: str) -> dict[str, Any]:
        return {
            "update_id": update_id,
            "callback_query": {
                "id": f"callback-{update_id}",
                "from": {"id": user_id},
                "message": {
                    "message_id": 9,
                    "chat": {"id": user_id, "type": "private"},
                },
                "data": f"ap:{token}:{choice}",
            },
        }

    @staticmethod
    def _group_message_update(
        update_id: int,
        chat_id: int,
        user_id: int,
        text: str,
        title: str = "Dev team",
    ) -> dict[str, Any]:
        return {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "from": {"id": user_id},
                "chat": {"id": chat_id, "type": "supergroup", "title": title},
                "text": text,
            },
        }

    @staticmethod
    def _group_callback_update(
        update_id: int,
        chat_id: int,
        user_id: int,
        message_id: int,
        data: str,
    ) -> dict[str, Any]:
        return {
            "update_id": update_id,
            "callback_query": {
                "id": f"callback-{update_id}",
                "from": {"id": user_id},
                "message": {
                    "message_id": message_id,
                    "chat": {"id": chat_id, "type": "supergroup"},
                },
                "data": data,
            },
        }

    @staticmethod
    def _group_attachment_update(
        update_id: int,
        chat_id: int,
        user_id: int,
        file_id: str,
        filename: str,
        caption: str = "",
    ) -> dict[str, Any]:
        update = MultiUserRoutingTests._group_message_update(
            update_id,
            chat_id,
            user_id,
            "",
        )
        message = update["message"]
        message.pop("text")
        message["media_group_id"] = "test-album"
        message["document"] = {
            "file_id": file_id,
            "file_size": 12,
            "file_name": filename,
            "mime_type": "text/plain",
        }
        if caption:
            message["caption"] = caption
        return update
