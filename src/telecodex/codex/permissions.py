from __future__ import annotations

from typing import Any

WORKSPACE_PERMISSION_PROFILE = "telecodex_workspace"
ASK_PERMISSION_PROFILE = "telecodex_ask"
READ_ONLY_PERMISSION_PROFILE = "telecodex_read_only"

PERMISSION_MODES = ("ask", "auto_review", "full_access", "read_only")


def thread_permission_params(mode: str | None) -> dict[str, Any]:
    if mode == "ask":
        return {
            "approvalPolicy": "on-request",
            "approvalsReviewer": "user",
            "permissions": ASK_PERMISSION_PROFILE,
        }
    if mode == "auto_review":
        return {
            "approvalPolicy": "on-request",
            "approvalsReviewer": "auto_review",
            "permissions": ASK_PERMISSION_PROFILE,
        }
    if mode == "full_access":
        return {"approvalPolicy": "never", "sandbox": "danger-full-access"}
    if mode == "read_only":
        return {
            "approvalPolicy": "on-request",
            "approvalsReviewer": "user",
            "permissions": READ_ONLY_PERMISSION_PROFILE,
        }
    return {}


def turn_permission_params(mode: str | None) -> dict[str, Any]:
    if mode == "full_access":
        return {
            "approvalPolicy": "never",
            "sandboxPolicy": {"type": "dangerFullAccess"},
        }
    return thread_permission_params(mode)
