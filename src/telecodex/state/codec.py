"""Pure conversion helpers for the runtime-state schema."""

from __future__ import annotations

from typing import Any

from telecodex.codex.permissions import PERMISSION_MODES
from telecodex.i18n import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
from telecodex.state.models import NewChatDefaults, PendingTurnState, RuntimeState

RECOVERY_PHASES = {"starting", "running", "delivery_pending"}


RECOVERY_RESULT_STATUSES = {"completed", "interrupted", "failed", "unknown"}


OUTPUT_MODES = {"answers", "detailed"}
NEW_CHAT_OUTPUT_MODES = {"answers", "quiet_answers", "detailed"}


def _new_chat_defaults_from_data(raw: dict[str, Any]) -> NewChatDefaults:
    if not isinstance(raw, dict):
        raise ValueError("Runtime state has an invalid operator defaults entry")
    allowed_fields = {
        "selected_model",
        "selected_effort",
        "default_permission_mode",
        "selected_memory_mode",
        "output_mode",
        "group_all_messages_to_codex",
    }
    if not set(raw).issubset(allowed_fields):
        raise ValueError("Runtime state has unsupported operator defaults fields")
    defaults = NewChatDefaults(
        selected_model=raw.get("selected_model"),
        selected_effort=raw.get("selected_effort"),
        default_permission_mode=raw.get("default_permission_mode"),
        selected_memory_mode=raw.get("selected_memory_mode"),
        output_mode=raw.get("output_mode"),
        group_all_messages_to_codex=raw.get("group_all_messages_to_codex"),
    )
    _new_chat_defaults_to_data(defaults)
    return defaults


def _new_chat_defaults_to_data(defaults: NewChatDefaults) -> dict[str, Any]:
    for field_name in ("selected_model", "selected_effort"):
        value = getattr(defaults, field_name)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"New-chat defaults have an invalid {field_name}")
    if defaults.selected_effort is not None and defaults.selected_model is None:
        raise ValueError("New-chat defaults cannot set reasoning effort without a model")
    if defaults.default_permission_mode is not None and (
        not isinstance(defaults.default_permission_mode, str)
        or defaults.default_permission_mode not in PERMISSION_MODES
    ):
        raise ValueError("New-chat defaults have an invalid default_permission_mode")
    if (
        defaults.selected_memory_mode is not None
        and (
            not isinstance(defaults.selected_memory_mode, str)
            or defaults.selected_memory_mode not in {"enabled", "disabled"}
        )
    ):
        raise ValueError("New-chat defaults have an invalid selected_memory_mode")
    if defaults.output_mode is not None and (
        not isinstance(defaults.output_mode, str)
        or defaults.output_mode not in NEW_CHAT_OUTPUT_MODES
    ):
        raise ValueError("New-chat defaults have an invalid output_mode")
    if defaults.group_all_messages_to_codex is not None and not isinstance(
        defaults.group_all_messages_to_codex, bool
    ):
        raise ValueError("New-chat defaults have an invalid group mode")
    return {
        "selected_model": defaults.selected_model,
        "selected_effort": defaults.selected_effort,
        "default_permission_mode": defaults.default_permission_mode,
        "selected_memory_mode": defaults.selected_memory_mode,
        "output_mode": defaults.output_mode,
        "group_all_messages_to_codex": defaults.group_all_messages_to_codex,
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


def _empty_state() -> dict[str, Any]:
    return {
        "last_update_id": 0,
        "users": {},
        "groups": {},
        "threads": {},
        "pending_media_groups": {},
        "operator_defaults": {},
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
