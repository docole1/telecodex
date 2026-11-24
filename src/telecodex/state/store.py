from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from telecodex.codex.permissions import PERMISSION_MODES
from telecodex.group_triggers import normalize_stored_trigger_words, parse_trigger_words
from telecodex.i18n import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
from telecodex.state.legacy_json import load_legacy_json_object
from telecodex.state.models import PendingTurnState, RuntimeState, ThreadRuntimeState
from telecodex.state.sqlite import SQLiteStateBackend


RECOVERY_PHASES = {"starting", "running", "delivery_pending"}
RECOVERY_RESULT_STATUSES = {"completed", "interrupted", "failed", "unknown"}
OUTPUT_MODES = {"answers", "detailed"}
OUTPUT_MODE_CHOICES = {"answers", "quiet_answers", "detailed"}
_GROUP_STATE_FIELDS = {
    "thread_id",
    "cwd",
    "selected_model",
    "selected_effort",
    "selected_service_tier",
    "selected_memory_mode",
    "pending_skill_name",
    "pending_skill_path",
    "default_permission_mode",
    "language",
    "pending_turn",
    "output_mode",
    "quiet_progress",
    "title",
    "attached",
    "all_messages_to_codex",
    "trigger_words",
    "pending_setup_user_id",
    "pending_setup_title",
}


class RuntimeStateStore:
    """Store polling progress and all dialog state in one private SQLite database."""

    def __init__(
        self,
        path: Path,
        migration_user_id: int | None = None,
        migration_path: Path | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        if migration_path is not None:
            migration_path = migration_path.expanduser().resolve()
        self.migration_user_id = migration_user_id
        if self.path.exists():
            # An existing database is authoritative; never fall back to stale JSON.
            self._backend = SQLiteStateBackend(self.path)
        else:
            initial_data = _empty_state()
            source_exists = migration_path is not None and migration_path.exists()
            if source_exists and not migration_path.is_file():
                raise ValueError(f"Legacy runtime state is not a regular JSON file: {migration_path}")
            if source_exists:
                initial_data = self._normalize_data(load_legacy_json_object(migration_path))
            self._backend = SQLiteStateBackend(self.path, initial_data)
            if source_exists:
                migration_path.chmod(0o600)
        self._load_data()

    def for_user(self, user_id: int) -> UserRuntimeStateStore:
        return UserRuntimeStateStore(self, user_id)

    def for_group(self, chat_id: int) -> GroupRuntimeStateStore:
        return GroupRuntimeStateStore(self, chat_id)

    def attach_group(
        self,
        chat_id: int,
        title: str,
        trigger_words: tuple[str, ...] | list[str] | None = None,
    ) -> bool:
        """Attach a Telegram group while retaining its prior Codex context."""
        _validate_group_chat_id(chat_id)
        if not isinstance(title, str):
            raise ValueError("Runtime state has an invalid group title")
        key = str(chat_id)

        def update(data: dict[str, Any]) -> bool:
            group = data["groups"].get(key)
            already_attached = isinstance(group, dict) and group["attached"]
            if not isinstance(group, dict):
                group = _new_group_record()
                data["groups"][key] = group
            group["title"] = title.strip()[:128] or key
            group["attached"] = True
            if trigger_words is not None:
                group["trigger_words"] = list(parse_trigger_words(",".join(trigger_words)))
            group["pending_setup_user_id"] = None
            group["pending_setup_title"] = None
            return already_attached

        return self._update_data(update)

    def begin_group_setup(self, chat_id: int, title: str, user_id: int) -> str:
        """Start or resume one trusted user's trigger-word setup for a group."""
        _validate_group_chat_id(chat_id)
        if not isinstance(title, str):
            raise ValueError("Runtime state has an invalid group title")
        if type(user_id) is not int or user_id < 1:
            raise ValueError("Runtime state has an invalid group setup user_id")
        key = str(chat_id)

        def update(data: dict[str, Any]) -> str:
            group = data["groups"].get(key)
            if not isinstance(group, dict):
                group = _new_group_record()
                data["groups"][key] = group

            pending_user_id = group["pending_setup_user_id"]
            if pending_user_id is not None:
                return "pending_self" if pending_user_id == user_id else "pending_other"

            group["pending_setup_user_id"] = user_id
            group["pending_setup_title"] = title.strip()[:128] or group["title"] or key
            return "started"

        result = self._update_data(update)
        return result

    def complete_group_setup(
        self,
        chat_id: int,
        user_id: int,
        trigger_words: tuple[str, ...],
    ) -> str | None:
        """Save trigger words and attach the group, or update its existing list."""
        _validate_group_chat_id(chat_id)
        if type(user_id) is not int or user_id < 1:
            raise ValueError("Runtime state has an invalid group setup user_id")
        normalized_words = parse_trigger_words(",".join(trigger_words))
        def update(data: dict[str, Any]) -> str | None:
            group = data["groups"].get(str(chat_id))
            if not isinstance(group, dict) or group["pending_setup_user_id"] != user_id:
                return None

            was_attached = group["attached"]
            pending_title = group["pending_setup_title"]
            if isinstance(pending_title, str) and pending_title.strip():
                group["title"] = pending_title.strip()[:128]
            group["attached"] = True
            group["trigger_words"] = list(normalized_words)
            group["pending_setup_user_id"] = None
            group["pending_setup_title"] = None
            return "updated" if was_attached else "attached"

        return self._update_data(update)

    def cancel_group_setup(self, chat_id: int) -> bool:
        """Clear an unfinished setup without detaching an already active group."""
        _validate_group_chat_id(chat_id)
        def update(data: dict[str, Any]) -> bool:
            group = data["groups"].get(str(chat_id))
            if not isinstance(group, dict) or group["pending_setup_user_id"] is None:
                return False
            group["pending_setup_user_id"] = None
            group["pending_setup_title"] = None
            return True

        return self._update_data(update)

    def pending_group_setup_user(self, chat_id: int) -> int | None:
        _validate_group_chat_id(chat_id)
        group = self._load_data()["groups"].get(str(chat_id))
        return group["pending_setup_user_id"] if isinstance(group, dict) else None

    def group_trigger_words(self, chat_id: int) -> tuple[str, ...]:
        _validate_group_chat_id(chat_id)
        group = self._load_data()["groups"].get(str(chat_id))
        return tuple(group["trigger_words"]) if isinstance(group, dict) else ()

    def group_all_messages_to_codex(self, chat_id: int) -> bool:
        _validate_group_chat_id(chat_id)
        group = self._load_data()["groups"].get(str(chat_id))
        return bool(group["all_messages_to_codex"]) if isinstance(group, dict) else False

    def set_group_all_messages_to_codex(self, chat_id: int, enabled: bool) -> bool:
        _validate_group_chat_id(chat_id)
        if not isinstance(enabled, bool):
            raise ValueError("Group all-messages mode must be true or false")

        def update(data: dict[str, Any]) -> bool:
            group = data["groups"].get(str(chat_id))
            if not isinstance(group, dict) or not group["attached"]:
                return False
            group["all_messages_to_codex"] = enabled
            return True

        return self._update_data(update)

    def detach_group(self, chat_id: int) -> bool:
        _validate_group_chat_id(chat_id)
        def update(data: dict[str, Any]) -> bool:
            group = data["groups"].get(str(chat_id))
            if not isinstance(group, dict) or not group["attached"]:
                return False
            group["attached"] = False
            group["pending_setup_user_id"] = None
            group["pending_setup_title"] = None
            return True

        return self._update_data(update)

    def is_group_attached(self, chat_id: int) -> bool:
        _validate_group_chat_id(chat_id)
        group = self._load_data()["groups"].get(str(chat_id))
        return isinstance(group, dict) and group["attached"]

    def list_attached_groups(self) -> dict[int, str]:
        return {
            int(chat_id): group["title"]
            for chat_id, group in self._load_data()["groups"].items()
            if group["attached"]
        }

    def load_last_update_id(self) -> int:
        return self._load_data()["last_update_id"]

    def save_last_update_id(self, last_update_id: int) -> None:
        if not isinstance(last_update_id, int) or last_update_id < 0:
            raise ValueError("Runtime state has an invalid last_update_id")
        self._update_data(lambda data: data.__setitem__("last_update_id", last_update_id))

    def load_pending_media_groups(self) -> dict[str, list[dict[str, Any]]]:
        return self._load_data()["pending_media_groups"]

    def save_pending_media_group(self, key: str, updates: list[dict[str, Any]]) -> None:
        if not key or not isinstance(key, str):
            raise ValueError("Runtime state has an invalid pending media group key")
        if not updates or any(not isinstance(update, dict) for update in updates):
            raise ValueError("Runtime state has invalid pending media group updates")
        def update(data: dict[str, Any]) -> None:
            data["pending_media_groups"][key] = updates

        self._update_data(update)

    def delete_pending_media_group(self, key: str) -> None:
        def update(data: dict[str, Any]) -> None:
            data["pending_media_groups"].pop(key, None)

        self._update_data(update)

    def _load_user_state(self, user_id: int) -> RuntimeState:
        _validate_user_id(user_id)
        data = self._load_data()
        raw_state = data["users"].get(str(user_id), {})
        return _runtime_state_from_data(raw_state, "detailed")

    def _load_group_state(self, chat_id: int) -> RuntimeState:
        _validate_group_chat_id(chat_id)
        raw_state = self._load_data()["groups"].get(str(chat_id), {})
        return _runtime_state_from_data(raw_state, "answers")

    def _save_user_state(self, user_id: int, state: RuntimeState) -> None:
        _validate_user_id(user_id)
        serialized = _runtime_state_to_data(state)

        def update(data: dict[str, Any]) -> None:
            data["users"].setdefault(str(user_id), {}).update(serialized)

        self._update_data(update)

    def _save_group_state(self, chat_id: int, state: RuntimeState) -> None:
        _validate_group_chat_id(chat_id)
        serialized = _runtime_state_to_data(state)

        def update(data: dict[str, Any]) -> None:
            group = data["groups"].get(str(chat_id))
            if not isinstance(group, dict):
                # A detached session may finish while the service is shutting down.
                # Do not recreate a group's attachment or context after it was removed.
                return
            group.update(serialized)

        self._update_data(update)

    def _load_thread_state(self, thread_id: str) -> ThreadRuntimeState:
        raw_state = self._load_data()["threads"].get(thread_id, {})
        return ThreadRuntimeState(
            permission_mode=raw_state.get("permission_mode"),
            memory_mode=raw_state.get("memory_mode"),
            managed_by_telecodex=raw_state.get("managed_by_telecodex", False),
            token_usage=raw_state.get("token_usage"),
        )

    def _save_thread_state(self, thread_id: str, state: ThreadRuntimeState) -> None:
        if not thread_id:
            raise ValueError("Thread runtime state has an invalid thread_id")
        if state.permission_mode is not None and state.permission_mode not in PERMISSION_MODES:
            raise ValueError("Thread runtime state has an invalid permission_mode")
        if state.memory_mode is not None and state.memory_mode not in {"enabled", "disabled"}:
            raise ValueError("Thread runtime state has an invalid memory_mode")
        if not isinstance(state.managed_by_telecodex, bool):
            raise ValueError("Thread runtime state has an invalid managed_by_telecodex")
        if state.token_usage is not None:
            if not isinstance(state.token_usage, dict) or any(
                not isinstance(key, str)
                or not key
                or (value is not None and (type(value) is not int or value < 0))
                for key, value in state.token_usage.items()
            ):
                raise ValueError("Thread runtime state has invalid token usage")
        def update(data: dict[str, Any]) -> None:
            data["threads"][thread_id] = {
                "permission_mode": state.permission_mode,
                "memory_mode": state.memory_mode,
                "managed_by_telecodex": state.managed_by_telecodex,
                "token_usage": state.token_usage,
            }

        self._update_data(update)

    def delete_threads(self, thread_ids: set[str]) -> None:
        """Remove TeleCodex metadata and user selections for deleted Codex threads."""
        valid_thread_ids = {thread_id for thread_id in thread_ids if thread_id}
        if not valid_thread_ids:
            return

        def update(data: dict[str, Any]) -> None:
            for thread_id in valid_thread_ids:
                data["threads"].pop(thread_id, None)
            for context in (*data["users"].values(), *data["groups"].values()):
                if context.get("thread_id") in valid_thread_ids:
                    context["thread_id"] = None
                pending_turn = context.get("pending_turn")
                if isinstance(pending_turn, dict) and pending_turn.get("thread_id") in valid_thread_ids:
                    context["pending_turn"] = None

        self._update_data(update)

    def _load_data(self) -> dict[str, Any]:
        return self._normalize_data(self._backend.load_state())

    def _update_data(self, update: Callable[[dict[str, Any]], Any]) -> Any:
        result: Any = None

        def apply(data: dict[str, Any]) -> dict[str, Any]:
            nonlocal result
            normalized = self._normalize_data(data)
            result = update(normalized)
            return self._normalize_data(normalized)

        self._backend.update_state(apply)
        return result

    def _normalize_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        last_update_id = raw.get("last_update_id", 0)
        if type(last_update_id) is not int or last_update_id < 0:
            raise ValueError("Runtime state has an invalid last_update_id")

        users = raw.get("users")
        if users is None:
            # Import the former single-user format without changing its source file.
            thread_id = raw.get("thread_id")
            if thread_id is not None and not isinstance(thread_id, str):
                raise ValueError("Runtime state has an invalid thread_id")
            users = {}
            if thread_id is not None and self.migration_user_id is not None:
                _validate_user_id(self.migration_user_id)
                users[str(self.migration_user_id)] = {
                    "thread_id": thread_id,
                    "cwd": None,
                    "selected_model": None,
                    "selected_effort": None,
                    "default_permission_mode": None,
                    "language": DEFAULT_LANGUAGE,
                    "output_mode": "detailed",
                }
        if not isinstance(users, dict):
            raise ValueError("Runtime state has an invalid users object")
        normalized_users: dict[str, dict[str, Any]] = {}
        for user_id, user_state in users.items():
            if not isinstance(user_id, str) or not isinstance(user_state, dict):
                raise ValueError("Runtime state has an invalid user entry")
            try:
                numeric_user_id = int(user_id)
            except ValueError as exc:
                raise ValueError("Runtime state has an invalid user ID") from exc
            _validate_user_id(numeric_user_id)
            if str(numeric_user_id) != user_id:
                raise ValueError("Runtime state has an invalid user ID")
            normalized_users[user_id] = _normalize_runtime_state_data(user_state, "detailed")
        raw_groups = raw.get("groups", {})
        if not isinstance(raw_groups, dict):
            raise ValueError("Runtime state has an invalid groups object")
        normalized_groups: dict[str, dict[str, Any]] = {}
        for chat_id, group_state in raw_groups.items():
            if not isinstance(chat_id, str) or not isinstance(group_state, dict):
                raise ValueError("Runtime state has an invalid group entry")
            if not set(group_state).issubset(_GROUP_STATE_FIELDS):
                raise ValueError("Runtime state has unsupported group fields")
            try:
                numeric_chat_id = int(chat_id)
            except ValueError as exc:
                raise ValueError("Runtime state has an invalid group chat_id") from exc
            _validate_group_chat_id(numeric_chat_id)
            if str(numeric_chat_id) != chat_id:
                raise ValueError("Runtime state has an invalid group chat_id")
            title = group_state.get("title", chat_id)
            attached = group_state.get("attached", False)
            all_messages_to_codex = group_state.get("all_messages_to_codex", False)
            trigger_words = normalize_stored_trigger_words(group_state.get("trigger_words", []))
            pending_setup_user_id = group_state.get("pending_setup_user_id")
            pending_setup_title = group_state.get("pending_setup_title")
            if (
                not isinstance(title, str)
                or not isinstance(attached, bool)
                or not isinstance(all_messages_to_codex, bool)
            ):
                raise ValueError("Runtime state has invalid group metadata")
            if pending_setup_user_id is not None and (
                type(pending_setup_user_id) is not int or pending_setup_user_id < 1
            ):
                raise ValueError("Runtime state has an invalid group setup user_id")
            if pending_setup_title is not None and not isinstance(pending_setup_title, str):
                raise ValueError("Runtime state has an invalid group setup title")
            if (pending_setup_user_id is None) != (pending_setup_title is None):
                raise ValueError("Runtime state has incomplete group setup metadata")
            context_state = _normalize_runtime_state_data(group_state, "answers")
            normalized_groups[chat_id] = {
                **context_state,
                "title": title[:128] or chat_id,
                "attached": attached,
                "all_messages_to_codex": all_messages_to_codex,
                "trigger_words": list(trigger_words),
                "pending_setup_user_id": pending_setup_user_id,
                "pending_setup_title": pending_setup_title[:128] if pending_setup_title else None,
            }
        raw_threads = raw.get("threads", {})
        if not isinstance(raw_threads, dict):
            raise ValueError("Runtime state has an invalid threads object")
        normalized_threads: dict[str, dict[str, Any]] = {}
        for thread_id, thread_state in raw_threads.items():
            if not isinstance(thread_id, str) or not thread_id or not isinstance(thread_state, dict):
                raise ValueError("Runtime state has an invalid thread entry")
            permission_mode = thread_state.get("permission_mode")
            if permission_mode is not None and permission_mode not in PERMISSION_MODES:
                raise ValueError("Runtime state has an invalid thread permission_mode")
            memory_mode = thread_state.get("memory_mode")
            if memory_mode is not None and memory_mode not in {"enabled", "disabled"}:
                raise ValueError("Runtime state has an invalid thread memory_mode")
            managed = thread_state.get("managed_by_telecodex", False)
            if not isinstance(managed, bool):
                raise ValueError("Runtime state has an invalid managed_by_telecodex")
            token_usage = thread_state.get("token_usage")
            if token_usage is not None and (
                not isinstance(token_usage, dict)
                or any(
                    not isinstance(key, str)
                    or not key
                    or (value is not None and (type(value) is not int or value < 0))
                    for key, value in token_usage.items()
                )
            ):
                raise ValueError("Runtime state has invalid token usage")
            normalized_threads[thread_id] = {
                "permission_mode": permission_mode,
                "memory_mode": memory_mode,
                "managed_by_telecodex": managed,
                "token_usage": token_usage,
            }
        for context in (*normalized_users.values(), *normalized_groups.values()):
            referenced_ids = [context.get("thread_id")]
            pending_turn = context.get("pending_turn")
            if isinstance(pending_turn, dict):
                referenced_ids.append(pending_turn.get("thread_id"))
            for thread_id in referenced_ids:
                if thread_id and thread_id not in normalized_threads:
                    # A selected thread can predate TeleCodex's per-thread metadata.
                    # Keep a default row so SQLite foreign keys remain valid.
                    normalized_threads[thread_id] = {
                        "permission_mode": None,
                        "memory_mode": None,
                        "managed_by_telecodex": False,
                        "token_usage": None,
                    }
        raw_media_groups = raw.get("pending_media_groups", {})
        if not isinstance(raw_media_groups, dict):
            raise ValueError("Runtime state has an invalid pending_media_groups object")
        pending_media_groups: dict[str, list[dict[str, Any]]] = {}
        for group_key, updates in raw_media_groups.items():
            if (
                not isinstance(group_key, str)
                or not group_key
                or not isinstance(updates, list)
                or not updates
                or any(not isinstance(update, dict) for update in updates)
            ):
                raise ValueError("Runtime state has an invalid pending media group")
            pending_media_groups[group_key] = updates
        return {
            "last_update_id": last_update_id,
            "users": normalized_users,
            "groups": normalized_groups,
            "threads": normalized_threads,
            "pending_media_groups": pending_media_groups,
        }


def _pending_turn_from_data(value: Any) -> PendingTurnState | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Runtime state has an invalid pending_turn")
    thread_id = value.get("thread_id")
    turn_id = value.get("turn_id")
    prompt_message_id = value.get("prompt_message_id")
    started_at = value.get("started_at")
    phase = value.get("phase", "starting")
    result_status = value.get("result_status")
    error = value.get("error")
    error_key = value.get("error_key")
    prompt_message_ids = value.get("prompt_message_ids", [])
    if not isinstance(thread_id, str) or not thread_id:
        raise ValueError("Runtime state has an invalid pending_turn thread_id")
    if turn_id is not None and (not isinstance(turn_id, str) or not turn_id):
        raise ValueError("Runtime state has an invalid pending_turn turn_id")
    if prompt_message_id is not None and (type(prompt_message_id) is not int or prompt_message_id <= 0):
        raise ValueError("Runtime state has an invalid pending_turn prompt_message_id")
    if type(started_at) is not int or started_at < 0:
        raise ValueError("Runtime state has an invalid pending_turn started_at")
    if not isinstance(phase, str) or phase not in RECOVERY_PHASES:
        raise ValueError("Runtime state has an invalid pending_turn phase")
    if result_status is not None and (
        not isinstance(result_status, str) or result_status not in RECOVERY_RESULT_STATUSES
    ):
        raise ValueError("Runtime state has an invalid pending_turn result_status")
    if error is not None and not isinstance(error, str):
        raise ValueError("Runtime state has an invalid pending_turn error")
    if error_key is not None and not isinstance(error_key, str):
        raise ValueError("Runtime state has an invalid pending_turn error_key")
    if not isinstance(prompt_message_ids, list) or any(
        type(message_id) is not int or message_id <= 0 for message_id in prompt_message_ids
    ):
        raise ValueError("Runtime state has invalid pending_turn prompt_message_ids")
    if phase == "delivery_pending" and result_status is None:
        raise ValueError("Runtime state is missing pending_turn result_status")
    return PendingTurnState(
        thread_id=thread_id,
        turn_id=turn_id,
        prompt_message_id=prompt_message_id,
        started_at=started_at,
        phase=phase,
        result_status=result_status,
        error=error,
        error_key=error_key,
        prompt_message_ids=list(dict.fromkeys(prompt_message_ids)),
    )


def _pending_turn_to_data(value: PendingTurnState | None) -> dict[str, Any] | None:
    if value is None:
        return None
    # Reuse the decoder to keep records written by this process within the same schema.
    checked = _pending_turn_from_data(
        {
            "thread_id": value.thread_id,
            "turn_id": value.turn_id,
            "prompt_message_id": value.prompt_message_id,
            "started_at": value.started_at,
            "phase": value.phase,
            "result_status": value.result_status,
            "error": value.error,
            "error_key": value.error_key,
            "prompt_message_ids": value.prompt_message_ids,
        }
    )
    assert checked is not None
    return {
        "thread_id": checked.thread_id,
        "turn_id": checked.turn_id,
        "prompt_message_id": checked.prompt_message_id,
        "started_at": checked.started_at,
        "phase": checked.phase,
        "result_status": checked.result_status,
        "error": checked.error,
        "error_key": checked.error_key,
        "prompt_message_ids": checked.prompt_message_ids,
    }


class UserRuntimeStateStore:
    """CodexSession-compatible view of one user's thread state."""

    def __init__(self, store: RuntimeStateStore, user_id: int) -> None:
        self.store = store
        self.user_id = user_id

    def load(self) -> RuntimeState:
        return self.store._load_user_state(self.user_id)

    def save(self, state: RuntimeState) -> None:
        self.store._save_user_state(self.user_id, state)

    def load_thread(self, thread_id: str) -> ThreadRuntimeState:
        return self.store._load_thread_state(thread_id)

    def save_thread(self, thread_id: str, state: ThreadRuntimeState) -> None:
        self.store._save_thread_state(thread_id, state)

    def delete_threads(self, thread_ids: set[str]) -> None:
        self.store.delete_threads(thread_ids)


class GroupRuntimeStateStore:
    """CodexSession-compatible view of one shared Telegram group context."""

    def __init__(self, store: RuntimeStateStore, chat_id: int) -> None:
        _validate_group_chat_id(chat_id)
        self.store = store
        self.chat_id = chat_id

    def load(self) -> RuntimeState:
        return self.store._load_group_state(self.chat_id)

    def save(self, state: RuntimeState) -> None:
        self.store._save_group_state(self.chat_id, state)

    def load_thread(self, thread_id: str) -> ThreadRuntimeState:
        return self.store._load_thread_state(thread_id)

    def save_thread(self, thread_id: str, state: ThreadRuntimeState) -> None:
        self.store._save_thread_state(thread_id, state)

    def delete_threads(self, thread_ids: set[str]) -> None:
        self.store.delete_threads(thread_ids)


def _validate_group_chat_id(chat_id: int) -> None:
    if type(chat_id) is not int or chat_id >= 0:
        raise ValueError("Runtime state has an invalid Telegram group chat_id")


def _validate_user_id(user_id: int) -> None:
    if type(user_id) is not int or user_id < 1:
        raise ValueError("Runtime state has an invalid Telegram user_id")


def _empty_state() -> dict[str, Any]:
    return {
        "last_update_id": 0,
        "users": {},
        "groups": {},
        "threads": {},
        "pending_media_groups": {},
    }


def _new_group_record() -> dict[str, Any]:
    record = _runtime_state_to_data(RuntimeState(output_mode="answers"))
    record.update(
        {
            "title": "",
            "attached": False,
            "all_messages_to_codex": False,
            "trigger_words": [],
            "pending_setup_user_id": None,
            "pending_setup_title": None,
        }
    )
    return record


def _runtime_state_from_data(raw_state: dict[str, Any], default_output_mode: str) -> RuntimeState:
    return RuntimeState(
        thread_id=raw_state.get("thread_id"),
        cwd=raw_state.get("cwd"),
        selected_model=raw_state.get("selected_model"),
        selected_service_tier=raw_state.get("selected_service_tier"),
        selected_memory_mode=raw_state.get("selected_memory_mode"),
        pending_skill_name=raw_state.get("pending_skill_name"),
        pending_skill_path=raw_state.get("pending_skill_path"),
        default_permission_mode=raw_state.get("default_permission_mode"),
        language=raw_state.get("language", DEFAULT_LANGUAGE),
        selected_effort=raw_state.get("selected_effort"),
        pending_turn=_pending_turn_from_data(raw_state.get("pending_turn")),
        output_mode=raw_state.get("output_mode", default_output_mode),
        quiet_progress=raw_state.get("quiet_progress", False),
    )


def _runtime_state_to_data(state: RuntimeState) -> dict[str, Any]:
    if state.thread_id is not None and (not isinstance(state.thread_id, str) or not state.thread_id):
        raise ValueError("Runtime state has an invalid thread_id")
    if state.cwd is not None and not isinstance(state.cwd, str):
        raise ValueError("Runtime state has an invalid cwd")
    if state.selected_model is not None and not isinstance(state.selected_model, str):
        raise ValueError("Runtime state has an invalid selected_model")
    if state.selected_effort is not None and (not isinstance(state.selected_effort, str) or not state.selected_effort):
        raise ValueError("Runtime state has an invalid selected_effort")
    if state.selected_service_tier is not None and (
        not isinstance(state.selected_service_tier, str) or not state.selected_service_tier
    ):
        raise ValueError("Runtime state has an invalid selected_service_tier")
    if state.selected_memory_mode is not None and state.selected_memory_mode not in {"enabled", "disabled"}:
        raise ValueError("Runtime state has an invalid selected_memory_mode")
    if (state.pending_skill_name is None) != (state.pending_skill_path is None):
        raise ValueError("Runtime state has an incomplete pending skill selection")
    if state.pending_skill_name is not None and (
        not isinstance(state.pending_skill_name, str)
        or not state.pending_skill_name
        or not isinstance(state.pending_skill_path, str)
        or not state.pending_skill_path
    ):
        raise ValueError("Runtime state has an invalid pending skill selection")
    if state.default_permission_mode is not None and state.default_permission_mode not in PERMISSION_MODES:
        raise ValueError("Runtime state has an invalid default_permission_mode")
    if state.language not in SUPPORTED_LANGUAGES:
        raise ValueError("Runtime state has an unsupported language")
    if not isinstance(state.output_mode, str) or state.output_mode not in OUTPUT_MODES:
        raise ValueError("Runtime state has an invalid output_mode")
    if not isinstance(state.quiet_progress, bool):
        raise ValueError("Runtime state has an invalid quiet_progress setting")
    if state.quiet_progress and state.output_mode != "answers":
        raise ValueError("Quiet progress notifications require answers output mode")
    return {
        "thread_id": state.thread_id,
        "cwd": state.cwd,
        "selected_model": state.selected_model,
        "selected_effort": state.selected_effort,
        "selected_service_tier": state.selected_service_tier,
        "selected_memory_mode": state.selected_memory_mode,
        "pending_skill_name": state.pending_skill_name,
        "pending_skill_path": state.pending_skill_path,
        "default_permission_mode": state.default_permission_mode,
        "language": state.language,
        "pending_turn": _pending_turn_to_data(state.pending_turn),
        "output_mode": state.output_mode,
        "quiet_progress": state.quiet_progress,
    }


def _normalize_runtime_state_data(
    raw_state: dict[str, Any], default_output_mode: str
) -> dict[str, Any]:
    state = _runtime_state_from_data(raw_state, default_output_mode)
    return _runtime_state_to_data(state)
