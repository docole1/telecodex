from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from telecodex.config import state_paths_from_env
from telecodex.state.sqlite import export_state_json
from telecodex.state import PendingTurnState, RuntimeState, RuntimeStateStore, ThreadRuntimeState

KAWAII_TRIGGER_CYRILLIC = "\u043a\u0430\u0432\u0430\u0438"


class SQLiteStateMigrationTests(unittest.TestCase):
    @staticmethod
    def _downgrade_v3_database(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TABLE service_metadata")
        connection.execute("ALTER TABLE group_contexts DROP COLUMN all_messages_to_codex")
        for table in ("user_contexts", "group_contexts"):
            for column in (
                "quiet_progress",
                "selected_service_tier",
                "selected_memory_mode",
                "pending_skill_name",
                "pending_skill_path",
            ):
                connection.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        connection.execute("ALTER TABLE threads DROP COLUMN memory_mode")
        connection.execute("PRAGMA user_version = 3")

    def test_schema_v3_migrates_new_context_preferences_and_thread_memory_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "state.sqlite3"
            store = RuntimeStateStore(database_path)
            store.for_user(101).save(
                RuntimeState(
                    thread_id="v3-thread",
                    cwd="/project/v3",
                    selected_model="gpt-6-luna",
                    selected_effort="high",
                    pending_turn=PendingTurnState(
                        thread_id="v3-thread",
                        turn_id="v3-turn",
                        prompt_message_id=77,
                        started_at=123,
                        phase="running",
                    ),
                )
            )
            store.for_user(101).save_thread(
                "v3-thread",
                ThreadRuntimeState(permission_mode="ask", managed_by_telecodex=True),
            )
            store.attach_group(-100990, "V3 group", ("codex",))
            store.for_group(-100990).save(RuntimeState(thread_id="v3-thread", cwd="/project/v3"))

            with sqlite3.connect(database_path) as connection:
                self._downgrade_v3_database(connection)

            migrated = RuntimeStateStore(database_path)
            self.assertEqual(migrated.for_user(101).load().selected_model, "gpt-6-luna")
            self.assertEqual(migrated.for_user(101).load().pending_turn.turn_id, "v3-turn")
            self.assertIsNone(migrated.for_user(101).load().selected_service_tier)
            self.assertIsNone(migrated.for_user(101).load().pending_skill_name)
            self.assertIsNone(migrated.for_user(101).load_thread("v3-thread").memory_mode)

            migrated.for_user(101).save(
                RuntimeState(
                    thread_id="v3-thread",
                    cwd="/project/v3",
                    selected_service_tier="priority",
                    selected_memory_mode="enabled",
                    pending_skill_name="reviewer",
                    pending_skill_path="/root/.codex/skills/reviewer/SKILL.md",
                )
            )
            thread_state = migrated.for_user(101).load_thread("v3-thread")
            thread_state.memory_mode = "disabled"
            migrated.for_user(101).save_thread("v3-thread", thread_state)

            reopened = RuntimeStateStore(database_path)
            user = reopened.for_user(101).load()
            self.assertEqual(user.selected_service_tier, "priority")
            self.assertEqual(user.selected_memory_mode, "enabled")
            self.assertEqual(user.pending_skill_name, "reviewer")
            self.assertEqual(user.pending_skill_path, "/root/.codex/skills/reviewer/SKILL.md")
            self.assertEqual(reopened.for_user(101).load_thread("v3-thread").memory_mode, "disabled")
            self.assertEqual(reopened.group_trigger_words(-100990), ("codex",))
            self.assertFalse(reopened.group_all_messages_to_codex(-100990))
            self.assertFalse(reopened.for_user(101).load().quiet_progress)
            self.assertFalse(reopened.for_group(-100990).load().quiet_progress)
            with sqlite3.connect(database_path) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 7)
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

            user = reopened.for_user(101).load()
            user.output_mode = "answers"
            user.quiet_progress = True
            reopened.for_user(101).save(user)
            group = reopened.for_group(-100990).load()
            group.output_mode = "answers"
            group.quiet_progress = True
            reopened.for_group(-100990).save(group)
            persisted = RuntimeStateStore(database_path)
            self.assertTrue(persisted.for_user(101).load().quiet_progress)
            self.assertTrue(persisted.for_group(-100990).load().quiet_progress)

    def test_schema_v4_migrates_group_routing_mode_as_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "state.sqlite3"
            store = RuntimeStateStore(database_path)
            store.attach_group(-100991, "V4 group", ("codex",))
            store.for_group(-100991).save(RuntimeState(thread_id="v4-thread"))

            with sqlite3.connect(database_path) as connection:
                connection.execute("DROP TABLE service_metadata")
                connection.execute("ALTER TABLE group_contexts DROP COLUMN all_messages_to_codex")
                connection.execute("ALTER TABLE user_contexts DROP COLUMN quiet_progress")
                connection.execute("ALTER TABLE group_contexts DROP COLUMN quiet_progress")
                connection.execute("PRAGMA user_version = 4")

            migrated = RuntimeStateStore(database_path)
            self.assertEqual(migrated.group_trigger_words(-100991), ("codex",))
            self.assertFalse(migrated.group_all_messages_to_codex(-100991))
            self.assertTrue(migrated.set_group_all_messages_to_codex(-100991, True))

            reopened = RuntimeStateStore(database_path)
            self.assertTrue(reopened.group_all_messages_to_codex(-100991))
            with sqlite3.connect(database_path) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 7)

    def test_group_message_mode_survives_detach_and_reattach(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            store.attach_group(-100992, "Work group", ("codex",))
            self.assertTrue(store.set_group_all_messages_to_codex(-100992, True))

            self.assertTrue(store.detach_group(-100992))
            self.assertTrue(store.group_all_messages_to_codex(-100992))
            store.attach_group(-100992, "Work group", ("assistant",))

            self.assertTrue(store.is_group_attached(-100992))
            self.assertTrue(store.group_all_messages_to_codex(-100992))
            self.assertEqual(store.group_trigger_words(-100992), ("assistant",))

    def test_imports_complete_multicontext_json_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "state.sqlite3"
            json_path = root / "state.json"
            legacy_state = {
                "last_update_id": 456,
                "users": {
                    "101": {
                        "thread_id": "user-thread",
                        "cwd": "/project/user",
                        "selected_model": "gpt-6-luna",
                        "selected_effort": "high",
                        "default_permission_mode": "ask",
                        "language": "ru",
                        "output_mode": "detailed",
                        "pending_turn": {
                            "thread_id": "user-thread",
                            "turn_id": "turn-101",
                            "prompt_message_id": 71,
                            "started_at": 1234,
                            "phase": "running",
                            "result_status": None,
                            "error": None,
                            "error_key": None,
                            "prompt_message_ids": [72, 73],
                        },
                    }
                },
                "groups": {
                    "-100123": {
                        "thread_id": "group-thread",
                        "cwd": "/project/team",
                        "selected_model": "gpt-6-luna",
                        "selected_effort": "medium",
                        "default_permission_mode": "read_only",
                        "language": "ru",
                        "output_mode": "answers",
                        "pending_turn": None,
                        "title": "Team",
                        "attached": True,
                        "trigger_words": ["codex", KAWAII_TRIGGER_CYRILLIC],
                        "pending_setup_user_id": 202,
                        "pending_setup_title": "Team title",
                    }
                },
                "threads": {
                    "user-thread": {
                        "permission_mode": "ask",
                        "managed_by_telecodex": True,
                        "token_usage": {"context_tokens": 321, "model_context_window": 200000},
                    },
                    "group-thread": {
                        "permission_mode": "read_only",
                        "managed_by_telecodex": False,
                        "token_usage": None,
                    },
                },
                "pending_media_groups": {
                    "-100123:album-9": [
                        {"update_id": 457, "message": {"message_id": 91, "caption": "one"}},
                        {"update_id": 458, "message": {"message_id": 92, "photo": [{"file_id": "f"}]}},
                    ]
                },
            }
            original = json.dumps(legacy_state, ensure_ascii=False, indent=2).encode()
            json_path.write_bytes(original)

            store = RuntimeStateStore(database_path, migration_user_id=101, migration_path=json_path)

            self.assertEqual(store.load_last_update_id(), 456)
            self.assertEqual(store.for_user(101).load().thread_id, "user-thread")
            self.assertEqual(
                store.for_user(101).load().pending_turn,
                PendingTurnState(
                    thread_id="user-thread",
                    turn_id="turn-101",
                    prompt_message_id=71,
                    started_at=1234,
                    phase="running",
                    prompt_message_ids=[72, 73],
                ),
            )
            self.assertTrue(store.is_group_attached(-100123))
            self.assertEqual(store.group_trigger_words(-100123), ("codex", KAWAII_TRIGGER_CYRILLIC))
            self.assertFalse(store.group_all_messages_to_codex(-100123))
            self.assertEqual(store.pending_group_setup_user(-100123), 202)
            self.assertEqual(
                store.for_user(101).load_thread("user-thread"),
                ThreadRuntimeState(
                    permission_mode="ask",
                    managed_by_telecodex=True,
                    token_usage={"context_tokens": 321, "model_context_window": 200000},
                ),
            )
            self.assertEqual(len(store.load_pending_media_groups()["-100123:album-9"]), 2)
            self.assertEqual(json_path.read_bytes(), original)
            self.assertEqual(json_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(database_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)

            with sqlite3.connect(database_path) as connection:
                tables = {
                    row[0]
                    for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                self.assertIn("user_pending_prompt_messages", tables)
                self.assertIn("thread_token_usage", tables)
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM group_trigger_words").fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM pending_media_groups").fetchone()[0],
                    2,
                )

    def test_imports_legacy_single_user_state_and_retains_json_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_path = root / "legacy.json"
            database_path = root / "database.sqlite3"
            raw = b'{"thread_id":"legacy-thread","last_update_id":89}'
            legacy_path.write_bytes(raw)

            store = RuntimeStateStore(database_path, migration_user_id=303, migration_path=legacy_path)

            self.assertEqual(store.for_user(303).load(), RuntimeState("legacy-thread"))
            self.assertEqual(store.load_last_update_id(), 89)
            self.assertEqual(legacy_path.read_bytes(), raw)

    def test_existing_database_wins_over_stale_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "state.sqlite3"
            json_path = root / "state.json"
            first = RuntimeStateStore(database_path)
            first.for_user(101).save(RuntimeState(thread_id="database-thread"))
            json_path.write_text(
                json.dumps({"thread_id": "stale-thread", "last_update_id": 999}),
                encoding="utf-8",
            )

            restarted = RuntimeStateStore(database_path, migration_user_id=101, migration_path=json_path)

            self.assertEqual(restarted.for_user(101).load().thread_id, "database-thread")
            self.assertEqual(restarted.load_last_update_id(), 0)
            self.assertEqual(json.loads(json_path.read_text())["thread_id"], "stale-thread")
            restarted.save_last_update_id(1000)
            self.assertEqual(json.loads(json_path.read_text())["last_update_id"], 999)

    def test_invalid_json_fails_before_creating_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "state.sqlite3"
            json_path = root / "state.json"
            json_path.write_text('{"users": {"101": {"language": "unknown"}}}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unsupported language"):
                RuntimeStateStore(database_path, migration_path=json_path)

            self.assertFalse(database_path.exists())

    def test_non_file_legacy_source_does_not_create_an_empty_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "state.sqlite3"
            legacy_directory = root / "state.json"
            legacy_directory.mkdir()

            with self.assertRaisesRegex(ValueError, "not a regular JSON file"):
                RuntimeStateStore(database_path, migration_path=legacy_directory)

            self.assertFalse(database_path.exists())

    def test_corrupt_or_unknown_database_never_falls_back_to_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "state.sqlite3"
            json_path = root / "state.json"
            json_path.write_text('{"thread_id":"must-not-load"}', encoding="utf-8")
            database_path.write_bytes(b"not a sqlite database")
            with self.assertRaisesRegex(ValueError, "corrupt or unreadable"):
                RuntimeStateStore(database_path, migration_user_id=101, migration_path=json_path)

            database_path.unlink()
            store = RuntimeStateStore(database_path, migration_user_id=101, migration_path=json_path)
            with sqlite3.connect(database_path) as connection:
                connection.execute("PRAGMA user_version = 99")
            with self.assertRaisesRegex(ValueError, "Unsupported TeleCodex SQLite database"):
                RuntimeStateStore(database_path, migration_user_id=101, migration_path=json_path)
            self.assertEqual(store.for_user(101).load().thread_id, "must-not-load")

    def test_export_round_trip_and_refuse_overwriting_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "state.sqlite3"
            export_path = root / "rollback.json"
            round_trip_path = root / "round-trip.sqlite3"
            store = RuntimeStateStore(database_path)
            store.attach_group(-100456, "Exported", ("codex",))
            store.for_group(-100456).save(RuntimeState(thread_id="shared-thread", cwd="/workspace/team"))
            store.for_user(101).save(RuntimeState(thread_id="shared-thread", cwd="/workspace/user"))
            store.for_user(101).save_thread(
                "shared-thread",
                ThreadRuntimeState(
                    permission_mode="auto_review",
                    managed_by_telecodex=True,
                    token_usage={"context_tokens": 44},
                ),
            )
            store.save_pending_media_group("album", [{"update_id": 1, "message": {"caption": "x"}}])
            before = store._load_data()

            export_state_json(database_path, export_path)
            imported = RuntimeStateStore(round_trip_path, migration_path=export_path)

            self.assertEqual(imported._load_data(), before)
            self.assertEqual(export_path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                export_state_json(database_path, export_path)

    def test_export_preserves_permissions_of_an_existing_destination_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "state.sqlite3"
            export_directory = root / "shared"
            export_directory.mkdir()
            export_directory.chmod(0o755)
            RuntimeStateStore(database_path)

            export_state_json(database_path, export_directory / "backup.json")

            self.assertEqual(export_directory.stat().st_mode & 0o777, 0o755)

    def test_config_uses_database_default_and_derives_database_for_legacy_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict("os.environ", {"XDG_STATE_HOME": temp_dir}, clear=True):
                database_path, migration_path = state_paths_from_env()
            self.assertEqual(database_path, Path(temp_dir) / "telecodex" / "state.sqlite3")
            self.assertEqual(migration_path, Path(temp_dir) / "telecodex" / "state.json")

            custom_json = Path(temp_dir) / "custom" / "bot-state.json"
            with patch.dict(
                "os.environ",
                {"TELECODEX_STATE_FILE": str(custom_json), "XDG_STATE_HOME": temp_dir},
                clear=True,
            ):
                database_path, migration_path = state_paths_from_env()
            self.assertEqual(database_path, custom_json.with_suffix(".sqlite3"))
            self.assertEqual(migration_path, custom_json)

            explicit_database = Path(temp_dir) / "custom.sqlite3"
            with patch.dict(
                "os.environ",
                {
                    "TELECODEX_DATABASE_FILE": str(explicit_database),
                    "TELECODEX_STATE_FILE": str(custom_json),
                    "XDG_STATE_HOME": temp_dir,
                },
                clear=True,
            ):
                database_path, migration_path = state_paths_from_env()
            self.assertEqual(database_path, explicit_database)
            self.assertEqual(migration_path, custom_json)

    def test_separate_store_instances_do_not_overwrite_concurrent_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "state.sqlite3"
            stores = [RuntimeStateStore(database_path), RuntimeStateStore(database_path)]

            def save_user(index: int) -> None:
                stores[index % 2].for_user(100 + index).save(
                    RuntimeState(thread_id=f"thread-{index}", cwd=f"/project/{index}")
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(save_user, range(16)))

            reloaded = RuntimeStateStore(database_path)
            for index in range(16):
                state = reloaded.for_user(100 + index).load()
                self.assertEqual(state.thread_id, f"thread-{index}")
                self.assertEqual(state.cwd, f"/project/{index}")

    def test_failed_state_update_rolls_back_all_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            store.for_user(101).save(RuntimeState(thread_id="preserved"))
            before = store._load_data()

            def fail_after_mutation(state: dict[str, Any]) -> dict[str, Any]:
                state["last_update_id"] = 1000
                state["users"]["202"] = {"thread_id": "must-rollback"}
                raise RuntimeError("simulated failure before commit")

            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                store._backend.update_state(fail_after_mutation)

            self.assertEqual(store._load_data(), before)

    def test_process_exit_rolls_back_uncommitted_sqlite_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "state.sqlite3"
            store = RuntimeStateStore(database_path)
            original_offset = store.load_last_update_id()
            crash_code = (
                "import os, sqlite3, sys; "
                "connection=sqlite3.connect(sys.argv[1], timeout=5, isolation_level=None); "
                "connection.execute('BEGIN IMMEDIATE'); "
                "connection.execute('UPDATE service_state SET last_update_id=? WHERE singleton=1', "
                "(int(sys.argv[2]),)); "
                "os._exit(23)"
            )

            crashed = subprocess.run(
                [sys.executable, "-c", crash_code, str(database_path), str(original_offset + 999)],
                check=False,
            )

            self.assertEqual(crashed.returncode, 23)
            recovered = RuntimeStateStore(database_path)
            self.assertEqual(recovered.load_last_update_id(), original_offset)
            with sqlite3.connect(database_path) as connection:
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_database_and_wal_sidecars_remain_private(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "state.sqlite3"
            store = RuntimeStateStore(database_path)
            anchor = store._backend._connect()
            try:
                store.save_last_update_id(1)
                self.assertEqual(database_path.stat().st_mode & 0o777, 0o600)
                for sidecar in (Path(f"{database_path}-wal"), Path(f"{database_path}-shm")):
                    if sidecar.exists():
                        self.assertEqual(sidecar.stat().st_mode & 0o777, 0o600)
            finally:
                anchor.close()

    def test_delete_threads_clears_user_and_group_contexts_in_one_state_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RuntimeStateStore(Path(temp_dir) / "state.sqlite3")
            store.attach_group(-100789, "Shared")
            store.for_user(101).save(
                RuntimeState(
                    thread_id="doomed",
                    pending_turn=PendingTurnState(
                        thread_id="doomed",
                        turn_id="turn",
                        prompt_message_id=10,
                        started_at=1,
                        phase="running",
                    ),
                )
            )
            store.for_group(-100789).save(
                RuntimeState(
                    thread_id="doomed",
                    pending_turn=PendingTurnState(
                        thread_id="doomed",
                        turn_id="group-turn",
                        prompt_message_id=20,
                        started_at=2,
                        phase="running",
                    ),
                    output_mode="answers",
                )
            )
            store.for_user(101).save_thread("doomed", ThreadRuntimeState(permission_mode="ask"))

            store.delete_threads({"doomed"})

            self.assertIsNone(store.for_user(101).load().thread_id)
            self.assertIsNone(store.for_user(101).load().pending_turn)
            self.assertIsNone(store.for_group(-100789).load().thread_id)
            self.assertIsNone(store.for_group(-100789).load().pending_turn)
            self.assertEqual(store.for_user(101).load_thread("doomed"), ThreadRuntimeState())


if __name__ == "__main__":
    unittest.main()
