from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 7
_APPLICATION_ID = 0x54434F44  # "TCOD"

_SCHEMA = """
CREATE TABLE service_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    last_update_id INTEGER NOT NULL CHECK (last_update_id >= 0)
);

CREATE TABLE service_metadata (
    metadata_key TEXT PRIMARY KEY CHECK (length(metadata_key) > 0),
    metadata_value TEXT NOT NULL CHECK (length(metadata_value) > 0)
);

CREATE TABLE threads (
    thread_id TEXT PRIMARY KEY CHECK (length(thread_id) > 0),
    permission_mode TEXT CHECK (permission_mode IS NULL OR permission_mode IN ('ask', 'auto_review', 'full_access', 'read_only')),
    memory_mode TEXT CHECK (memory_mode IS NULL OR memory_mode IN ('enabled', 'disabled')),
    managed_by_telecodex INTEGER NOT NULL CHECK (managed_by_telecodex IN (0, 1))
);

CREATE TABLE thread_token_usage (
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    usage_key TEXT NOT NULL CHECK (length(usage_key) > 0),
    usage_value INTEGER CHECK (usage_value IS NULL OR usage_value >= 0),
    PRIMARY KEY (thread_id, usage_key)
);

CREATE TABLE user_contexts (
    user_id INTEGER PRIMARY KEY CHECK (user_id > 0),
    thread_id TEXT REFERENCES threads(thread_id) ON DELETE SET NULL,
    cwd TEXT,
    selected_model TEXT,
    selected_effort TEXT,
    selected_service_tier TEXT,
    selected_memory_mode TEXT CHECK (selected_memory_mode IS NULL OR selected_memory_mode IN ('enabled', 'disabled')),
    pending_skill_name TEXT,
    pending_skill_path TEXT,
    default_permission_mode TEXT CHECK (default_permission_mode IS NULL OR default_permission_mode IN ('ask', 'auto_review', 'full_access', 'read_only')),
    language TEXT NOT NULL CHECK (language IN ('ru', 'en')),
    output_mode TEXT NOT NULL CHECK (output_mode IN ('answers', 'detailed')),
    quiet_progress INTEGER NOT NULL DEFAULT 0 CHECK (quiet_progress IN (0, 1))
);

CREATE TABLE group_contexts (
    chat_id INTEGER PRIMARY KEY CHECK (chat_id < 0),
    thread_id TEXT REFERENCES threads(thread_id) ON DELETE SET NULL,
    cwd TEXT,
    selected_model TEXT,
    selected_effort TEXT,
    selected_service_tier TEXT,
    selected_memory_mode TEXT CHECK (selected_memory_mode IS NULL OR selected_memory_mode IN ('enabled', 'disabled')),
    pending_skill_name TEXT,
    pending_skill_path TEXT,
    default_permission_mode TEXT CHECK (default_permission_mode IS NULL OR default_permission_mode IN ('ask', 'auto_review', 'full_access', 'read_only')),
    language TEXT NOT NULL CHECK (language IN ('ru', 'en')),
    output_mode TEXT NOT NULL CHECK (output_mode IN ('answers', 'detailed')),
    quiet_progress INTEGER NOT NULL DEFAULT 0 CHECK (quiet_progress IN (0, 1)),
    title TEXT NOT NULL,
    attached INTEGER NOT NULL CHECK (attached IN (0, 1)),
    all_messages_to_codex INTEGER NOT NULL DEFAULT 0 CHECK (all_messages_to_codex IN (0, 1)),
    pending_setup_user_id INTEGER CHECK (pending_setup_user_id IS NULL OR pending_setup_user_id > 0),
    pending_setup_title TEXT,
    CHECK ((pending_setup_user_id IS NULL) = (pending_setup_title IS NULL))
);

CREATE INDEX attached_groups ON group_contexts(attached, chat_id);

CREATE TABLE group_trigger_words (
    chat_id INTEGER NOT NULL REFERENCES group_contexts(chat_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    word TEXT NOT NULL CHECK (length(word) > 0),
    PRIMARY KEY (chat_id, position),
    UNIQUE (chat_id, word)
);

CREATE TABLE user_pending_turns (
    user_id INTEGER PRIMARY KEY REFERENCES user_contexts(user_id) ON DELETE CASCADE,
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    turn_id TEXT,
    prompt_message_id INTEGER CHECK (prompt_message_id IS NULL OR prompt_message_id > 0),
    started_at INTEGER NOT NULL CHECK (started_at >= 0),
    phase TEXT NOT NULL CHECK (phase IN ('starting', 'running', 'delivery_pending')),
    result_status TEXT CHECK (result_status IS NULL OR result_status IN ('completed', 'interrupted', 'failed', 'unknown')),
    error TEXT,
    error_key TEXT,
    CHECK (phase != 'delivery_pending' OR result_status IS NOT NULL)
);

CREATE TABLE user_pending_prompt_messages (
    user_id INTEGER NOT NULL REFERENCES user_pending_turns(user_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    message_id INTEGER NOT NULL CHECK (message_id > 0),
    PRIMARY KEY (user_id, position),
    UNIQUE (user_id, message_id)
);

CREATE TABLE group_pending_turns (
    chat_id INTEGER PRIMARY KEY REFERENCES group_contexts(chat_id) ON DELETE CASCADE,
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    turn_id TEXT,
    prompt_message_id INTEGER CHECK (prompt_message_id IS NULL OR prompt_message_id > 0),
    started_at INTEGER NOT NULL CHECK (started_at >= 0),
    phase TEXT NOT NULL CHECK (phase IN ('starting', 'running', 'delivery_pending')),
    result_status TEXT CHECK (result_status IS NULL OR result_status IN ('completed', 'interrupted', 'failed', 'unknown')),
    error TEXT,
    error_key TEXT,
    CHECK (phase != 'delivery_pending' OR result_status IS NOT NULL)
);

CREATE TABLE group_pending_prompt_messages (
    chat_id INTEGER NOT NULL REFERENCES group_pending_turns(chat_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    message_id INTEGER NOT NULL CHECK (message_id > 0),
    PRIMARY KEY (chat_id, position),
    UNIQUE (chat_id, message_id)
);

CREATE TABLE pending_media_groups (
    group_key TEXT NOT NULL CHECK (length(group_key) > 0),
    position INTEGER NOT NULL CHECK (position >= 0),
    update_json TEXT NOT NULL CHECK (json_valid(update_json)),
    PRIMARY KEY (group_key, position)
);
"""


class SQLiteStateBackend:
    """Persist the complete TeleCodex runtime state in one private SQLite file."""

    def __init__(self, path: Path, initial_state: dict[str, Any] | None = None) -> None:
        self.path = path.expanduser().resolve()
        if self.path.exists():
            self._migrate_existing_database()
            self._verify_existing_database()
        else:
            if initial_state is None:
                initial_state = _empty_state()
            self._create_atomically(initial_state)
            self._verify_existing_database()

    def load_state(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            state = _read_state(connection)
            connection.commit()
            return state
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_sidecars()

    def load_service_metadata(self, key: str) -> str | None:
        if not isinstance(key, str) or not key:
            raise ValueError("Runtime service metadata key must be a non-empty string")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT metadata_value FROM service_metadata WHERE metadata_key = ?",
                (key,),
            ).fetchone()
            return row[0] if row is not None else None
        finally:
            connection.close()
            self._secure_sidecars()

    def save_service_metadata(self, key: str, value: str) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("Runtime service metadata key must be a non-empty string")
        if not isinstance(value, str) or not value:
            raise ValueError("Runtime service metadata value must be a non-empty string")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO service_metadata(metadata_key, metadata_value) VALUES (?, ?) "
                "ON CONFLICT(metadata_key) DO UPDATE SET metadata_value = excluded.metadata_value",
                (key, value),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_sidecars()

    def _migrate_existing_database(self) -> None:
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            try:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                application_id = connection.execute("PRAGMA application_id").fetchone()[0]
            except sqlite3.DatabaseError:
                return
            if application_id != _APPLICATION_ID:
                return
            if version == 3:
                self._migrate_v3_to_v4(connection)
                version = 4
            if version == 4:
                self._migrate_v4_to_v5(connection)
                version = 5
            if version == 5:
                self._migrate_v5_to_v6(connection)
                version = 6
            if version == 6:
                self._migrate_v6_to_v7(connection)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate_v3_to_v4(self, connection: sqlite3.Connection) -> None:
        self._create_pre_migration_backup(3)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "ALTER TABLE threads ADD COLUMN memory_mode TEXT "
            "CHECK (memory_mode IS NULL OR memory_mode IN ('enabled', 'disabled'))"
        )
        for table in ("user_contexts", "group_contexts"):
            connection.execute(f"ALTER TABLE {table} ADD COLUMN selected_service_tier TEXT")
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN selected_memory_mode TEXT "
                "CHECK (selected_memory_mode IS NULL OR selected_memory_mode IN ('enabled', 'disabled'))"
            )
            connection.execute(f"ALTER TABLE {table} ADD COLUMN pending_skill_name TEXT")
            connection.execute(f"ALTER TABLE {table} ADD COLUMN pending_skill_path TEXT")
        connection.execute("PRAGMA user_version = 4")
        connection.commit()

    def _migrate_v4_to_v5(self, connection: sqlite3.Connection) -> None:
        self._create_pre_migration_backup(4)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "ALTER TABLE group_contexts ADD COLUMN all_messages_to_codex INTEGER "
            "NOT NULL DEFAULT 0 CHECK (all_messages_to_codex IN (0, 1))"
        )
        connection.execute("PRAGMA user_version = 5")
        connection.commit()

    def _migrate_v5_to_v6(self, connection: sqlite3.Connection) -> None:
        self._create_pre_migration_backup(5)
        connection.execute("BEGIN IMMEDIATE")
        for table in ("user_contexts", "group_contexts"):
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN quiet_progress INTEGER "
                "NOT NULL DEFAULT 0 CHECK (quiet_progress IN (0, 1))"
            )
        connection.execute("PRAGMA user_version = 6")
        connection.commit()

    def _migrate_v6_to_v7(self, connection: sqlite3.Connection) -> None:
        self._create_pre_migration_backup(6)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE service_metadata ("
            "metadata_key TEXT PRIMARY KEY CHECK (length(metadata_key) > 0), "
            "metadata_value TEXT NOT NULL CHECK (length(metadata_value) > 0))"
        )
        connection.execute("PRAGMA user_version = 7")
        connection.commit()

    def _create_pre_migration_backup(self, schema_version: int) -> None:
        import time

        backup_path = self.path.with_name(
            f"{self.path.name}.schema-v{schema_version}-{time.time_ns()}.bak"
        )
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.migration-backup.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            source = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            destination = sqlite3.connect(temporary_path, timeout=5, isolation_level=None)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()
            temporary_path.chmod(0o600)
            os.link(temporary_path, backup_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def update_state(self, update: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            state = _read_state(connection)
            updated = update(state)
            _replace_state(connection, updated)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_sidecars()

    def _create_atomically(self, initial_state: dict[str, Any]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            connection = sqlite3.connect(temporary_path, timeout=5, isolation_level=None)
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA journal_mode = DELETE")
                connection.execute("PRAGMA synchronous = FULL")
                connection.executescript(_SCHEMA)
                connection.execute("BEGIN IMMEDIATE")
                _replace_state(connection, initial_state)
                connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
            temporary_path.chmod(0o600)
            os.replace(temporary_path, self.path)
            self.path.chmod(0o600)
            if os.name == "posix":
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            for candidate in (temporary_path, Path(f"{temporary_path}-journal")):
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass

    def _verify_existing_database(self) -> None:
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            self.path.chmod(0o600)
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            application_id = connection.execute("PRAGMA application_id").fetchone()[0]
            if version != SCHEMA_VERSION or application_id != _APPLICATION_ID:
                raise ValueError(
                    f"Unsupported TeleCodex SQLite database at {self.path} "
                    f"(schema version {version}, application id {application_id})"
                )
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ValueError(f"SQLite state database failed integrity check: {integrity}")
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            required = {
                "service_state",
                "service_metadata",
                "threads",
                "thread_token_usage",
                "user_contexts",
                "group_contexts",
                "group_trigger_words",
                "user_pending_turns",
                "user_pending_prompt_messages",
                "group_pending_turns",
                "group_pending_prompt_messages",
                "pending_media_groups",
            }
            if not required.issubset(tables):
                raise ValueError(f"SQLite state database is missing tables: {', '.join(sorted(required - tables))}")
            foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise ValueError("SQLite state database has broken foreign-key references")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA busy_timeout = 5000")
            self._secure_sidecars()
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"SQLite state database is corrupt or unreadable: {self.path}") from exc
        finally:
            connection.close()
            self._secure_sidecars()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA synchronous = FULL")
            self._secure_sidecars()
            return connection
        except Exception:
            connection.close()
            raise

    def _secure_sidecars(self) -> None:
        for path in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
            try:
                path.chmod(0o600)
            except FileNotFoundError:
                pass


def export_state_json(database_path: Path, destination: Path) -> None:
    """Export a validated SQLite database to the legacy JSON interchange format."""
    database_path = database_path.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"SQLite state database does not exist: {database_path}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {destination}")

    backend = SQLiteStateBackend(database_path)
    state = backend.load_state()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        Path(temporary_name).chmod(0o600)
        os.replace(temporary_name, destination)
        destination.chmod(0o600)
    finally:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass


def _empty_state() -> dict[str, Any]:
    return {
        "last_update_id": 0,
        "users": {},
        "groups": {},
        "threads": {},
        "pending_media_groups": {},
    }


def _read_state(connection: sqlite3.Connection) -> dict[str, Any]:
    service_row = connection.execute(
        "SELECT last_update_id FROM service_state WHERE singleton = 1"
    ).fetchone()
    if service_row is None:
        raise ValueError("SQLite state database is missing service state")

    users: dict[str, dict[str, Any]] = {}
    for row in connection.execute(
        "SELECT user_id, thread_id, cwd, selected_model, selected_effort, selected_service_tier, "
        "selected_memory_mode, pending_skill_name, pending_skill_path, "
        "default_permission_mode, language, output_mode, quiet_progress FROM user_contexts ORDER BY user_id"
    ):
        (
            user_id,
            thread_id,
            cwd,
            model,
            effort,
            service_tier,
            memory_mode,
            skill_name,
            skill_path,
            permission,
            language,
            output_mode,
            quiet_progress,
        ) = row
        users[str(user_id)] = _context_from_row(
            connection,
            "user",
            user_id,
            thread_id,
            cwd,
            model,
            effort,
            service_tier,
            memory_mode,
            skill_name,
            skill_path,
            permission,
            language,
            output_mode,
            bool(quiet_progress),
        )

    groups: dict[str, dict[str, Any]] = {}
    for row in connection.execute(
        "SELECT chat_id, thread_id, cwd, selected_model, selected_effort, selected_service_tier, "
        "selected_memory_mode, pending_skill_name, pending_skill_path, "
        "default_permission_mode, language, output_mode, quiet_progress, title, attached, "
        "all_messages_to_codex, pending_setup_user_id, pending_setup_title "
        "FROM group_contexts ORDER BY chat_id"
    ):
        (
            chat_id,
            thread_id,
            cwd,
            model,
            effort,
            service_tier,
            memory_mode,
            skill_name,
            skill_path,
            permission,
            language,
            output_mode,
            quiet_progress,
            title,
            attached,
            all_messages_to_codex,
            pending_user_id,
            pending_title,
        ) = row
        group = _context_from_row(
            connection,
            "group",
            chat_id,
            thread_id,
            cwd,
            model,
            effort,
            service_tier,
            memory_mode,
            skill_name,
            skill_path,
            permission,
            language,
            output_mode,
            bool(quiet_progress),
        )
        group.update(
            {
                "title": title,
                "attached": bool(attached),
                "all_messages_to_codex": bool(all_messages_to_codex),
                "trigger_words": [
                    value[0]
                    for value in connection.execute(
                        "SELECT word FROM group_trigger_words WHERE chat_id = ? ORDER BY position",
                        (chat_id,),
                    )
                ],
                "pending_setup_user_id": pending_user_id,
                "pending_setup_title": pending_title,
            }
        )
        groups[str(chat_id)] = group

    threads: dict[str, dict[str, Any]] = {}
    for thread_id, permission_mode, memory_mode, managed in connection.execute(
        "SELECT thread_id, permission_mode, memory_mode, managed_by_telecodex FROM threads ORDER BY thread_id"
    ):
        usage = {
            key: value
            for key, value in connection.execute(
                "SELECT usage_key, usage_value FROM thread_token_usage WHERE thread_id = ? ORDER BY usage_key",
                (thread_id,),
            )
        }
        threads[thread_id] = {
            "permission_mode": permission_mode,
            "memory_mode": memory_mode,
            "managed_by_telecodex": bool(managed),
            "token_usage": usage or None,
        }

    pending_media_groups: dict[str, list[dict[str, Any]]] = {}
    for group_key, update_json in connection.execute(
        "SELECT group_key, update_json FROM pending_media_groups ORDER BY group_key, position"
    ):
        pending_media_groups.setdefault(group_key, []).append(json.loads(update_json))

    return {
        "last_update_id": service_row[0],
        "users": users,
        "groups": groups,
        "threads": threads,
        "pending_media_groups": pending_media_groups,
    }


def _context_from_row(
    connection: sqlite3.Connection,
    kind: str,
    context_id: int,
    thread_id: str | None,
    cwd: str | None,
    model: str | None,
    effort: str | None,
    service_tier: str | None,
    memory_mode: str | None,
    skill_name: str | None,
    skill_path: str | None,
    permission: str | None,
    language: str,
    output_mode: str,
    quiet_progress: bool,
) -> dict[str, Any]:
    turn_table = "user_pending_turns" if kind == "user" else "group_pending_turns"
    prompt_table = "user_pending_prompt_messages" if kind == "user" else "group_pending_prompt_messages"
    id_column = "user_id" if kind == "user" else "chat_id"
    row = connection.execute(
        f"SELECT thread_id, turn_id, prompt_message_id, started_at, phase, result_status, error, error_key "
        f"FROM {turn_table} WHERE {id_column} = ?",
        (context_id,),
    ).fetchone()
    pending_turn = None
    if row is not None:
        turn_thread_id, turn_id, prompt_message_id, started_at, phase, result_status, error, error_key = row
        prompt_message_ids = [
            item[0]
            for item in connection.execute(
                f"SELECT message_id FROM {prompt_table} WHERE {id_column} = ? ORDER BY position",
                (context_id,),
            )
        ]
        pending_turn = {
            "thread_id": turn_thread_id,
            "turn_id": turn_id,
            "prompt_message_id": prompt_message_id,
            "started_at": started_at,
            "phase": phase,
            "result_status": result_status,
            "error": error,
            "error_key": error_key,
            "prompt_message_ids": prompt_message_ids,
        }
    return {
        "thread_id": thread_id,
        "cwd": cwd,
        "selected_model": model,
        "selected_service_tier": service_tier,
        "selected_memory_mode": memory_mode,
        "pending_skill_name": skill_name,
        "pending_skill_path": skill_path,
        "selected_effort": effort,
        "default_permission_mode": permission,
        "language": language,
        "pending_turn": pending_turn,
        "output_mode": output_mode,
        "quiet_progress": quiet_progress,
    }


def _replace_state(connection: sqlite3.Connection, state: dict[str, Any]) -> None:
    threads = dict(state["threads"])
    for collection in (state["users"].values(), state["groups"].values()):
        for context in collection:
            for thread_id in (
                context.get("thread_id"),
                (context.get("pending_turn") or {}).get("thread_id"),
            ):
                if thread_id and thread_id not in threads:
                    threads[thread_id] = {
                        "permission_mode": None,
                        "memory_mode": None,
                        "managed_by_telecodex": False,
                        "token_usage": None,
                    }

    connection.execute("DELETE FROM pending_media_groups")
    connection.execute("DELETE FROM group_trigger_words")
    connection.execute("DELETE FROM user_contexts")
    connection.execute("DELETE FROM group_contexts")
    connection.execute("DELETE FROM threads")
    connection.execute("DELETE FROM service_state")
    connection.execute("INSERT INTO service_state(singleton, last_update_id) VALUES (1, ?)", (state["last_update_id"],))

    for thread_id, thread in threads.items():
        connection.execute(
            "INSERT INTO threads(thread_id, permission_mode, memory_mode, managed_by_telecodex) "
            "VALUES (?, ?, ?, ?)",
            (
                thread_id,
                thread["permission_mode"],
                thread.get("memory_mode"),
                int(thread["managed_by_telecodex"]),
            ),
        )
        for usage_key, usage_value in (thread["token_usage"] or {}).items():
            connection.execute(
                "INSERT INTO thread_token_usage(thread_id, usage_key, usage_value) VALUES (?, ?, ?)",
                (thread_id, usage_key, usage_value),
            )

    for user_id_text, context in state["users"].items():
        user_id = int(user_id_text)
        connection.execute(
            "INSERT INTO user_contexts(user_id, thread_id, cwd, selected_model, selected_effort, "
            "selected_service_tier, selected_memory_mode, pending_skill_name, pending_skill_path, "
            "default_permission_mode, language, output_mode, quiet_progress) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _context_values(user_id, context),
        )
        _insert_pending_turn(connection, "user", user_id, context.get("pending_turn"))

    for chat_id_text, context in state["groups"].items():
        chat_id = int(chat_id_text)
        connection.execute(
            "INSERT INTO group_contexts(chat_id, thread_id, cwd, selected_model, selected_effort, "
            "selected_service_tier, selected_memory_mode, pending_skill_name, pending_skill_path, "
            "default_permission_mode, language, output_mode, quiet_progress, title, attached, "
            "all_messages_to_codex, "
            "pending_setup_user_id, pending_setup_title) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *_context_values(chat_id, context),
                context["title"],
                int(context["attached"]),
                int(context["all_messages_to_codex"]),
                context["pending_setup_user_id"],
                context["pending_setup_title"],
            ),
        )
        for position, word in enumerate(context["trigger_words"]):
            connection.execute(
                "INSERT INTO group_trigger_words(chat_id, position, word) VALUES (?, ?, ?)",
                (chat_id, position, word),
            )
        _insert_pending_turn(connection, "group", chat_id, context.get("pending_turn"))

    for group_key, updates in state["pending_media_groups"].items():
        for position, update in enumerate(updates):
            connection.execute(
                "INSERT INTO pending_media_groups(group_key, position, update_json) VALUES (?, ?, ?)",
                (group_key, position, json.dumps(update, ensure_ascii=False, separators=(",", ":"))),
            )


def _context_values(context_id: int, context: dict[str, Any]) -> tuple[Any, ...]:
    return (
        context_id,
        context["thread_id"],
        context["cwd"],
        context["selected_model"],
        context["selected_effort"],
        context["selected_service_tier"],
        context["selected_memory_mode"],
        context["pending_skill_name"],
        context["pending_skill_path"],
        context["default_permission_mode"],
        context["language"],
        context["output_mode"],
        int(context["quiet_progress"]),
    )


def _insert_pending_turn(
    connection: sqlite3.Connection,
    kind: str,
    context_id: int,
    pending_turn: dict[str, Any] | None,
) -> None:
    if pending_turn is None:
        return
    turn_table = "user_pending_turns" if kind == "user" else "group_pending_turns"
    prompt_table = "user_pending_prompt_messages" if kind == "user" else "group_pending_prompt_messages"
    id_column = "user_id" if kind == "user" else "chat_id"
    connection.execute(
        f"INSERT INTO {turn_table}({id_column}, thread_id, turn_id, prompt_message_id, started_at, "
        "phase, result_status, error, error_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            context_id,
            pending_turn["thread_id"],
            pending_turn["turn_id"],
            pending_turn["prompt_message_id"],
            pending_turn["started_at"],
            pending_turn["phase"],
            pending_turn["result_status"],
            pending_turn["error"],
            pending_turn["error_key"],
        ),
    )
    for position, message_id in enumerate(pending_turn["prompt_message_ids"]):
        connection.execute(
            f"INSERT INTO {prompt_table}({id_column}, position, message_id) VALUES (?, ?, ?)",
            (context_id, position, message_id),
        )
