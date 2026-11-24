from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def format_activity_lines(lines: list[tuple[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    """Format Codex activity lines and bold only their labels with Telegram entities."""
    rendered: list[str] = []
    entities: list[dict[str, Any]] = []
    offset = 0
    for label, detail in lines:
        line = f"• {label}"
        if detail:
            line += f" {detail}"
        rendered.append(line)
        entities.append(
            {
                "type": "bold",
                "offset": offset + utf16_length("• "),
                "length": utf16_length(label),
            }
        )
        offset += utf16_length(line) + utf16_length("\n")
    return "\n".join(rendered), entities


def utf16_length(value: str) -> int:
    """Telegram entity offsets and lengths use UTF-16 code units."""
    return len(value.encode("utf-16-le")) // 2


def file_change_groups(item: object) -> dict[str, list[str]]:
    groups = {"add": [], "update": [], "delete": []}
    if not isinstance(item, dict):
        return groups
    changes = item.get("changes")
    if not isinstance(changes, list):
        return groups

    for change in changes:
        if not isinstance(change, dict):
            continue
        path = change.get("path")
        if not isinstance(path, str) or not path:
            continue
        kind = change.get("kind")
        if isinstance(kind, dict):
            kind = kind.get("type")
        if not isinstance(kind, str) or kind not in groups:
            kind = "update"
        diff = change.get("diff")
        if isinstance(diff, str):
            added, removed = _diff_line_counts(diff)
            if added or removed:
                path = f"{path} (+{added} -{removed})"
        groups[kind].append(path)
    return {kind: list(dict.fromkeys(paths)) for kind, paths in groups.items()}


def explored_action_summaries(
    actions: object,
    cwd: object = None,
    *,
    render: Callable[..., str],
) -> list[str]:
    if not isinstance(actions, list):
        return []
    working_directory = cwd.strip() if isinstance(cwd, str) else ""

    summaries: list[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = action.get("type")
        path = action.get("path")
        path = path.strip() if isinstance(path, str) else ""
        if action_type == "read":
            if path:
                summaries.append(path)
        elif action_type == "listFiles":
            target = path or working_directory
            if target:
                summaries.append(render("detailed_log.values.explored_action.files_in", target=target))
        elif action_type == "search":
            query = action.get("query")
            query = query.strip() if isinstance(query, str) else ""
            target = path or working_directory
            if query and target:
                summaries.append(
                    render("detailed_log.values.explored_action.search_in", query=query, target=target)
                )
            elif query:
                summaries.append(render("detailed_log.values.explored_action.search", query=query))
            elif target:
                summaries.append(render("detailed_log.values.explored_action.search_directory", target=target))
    return list(dict.fromkeys(summaries))


def web_search_detail(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    action = item.get("action")
    if isinstance(action, dict):
        action_type = action.get("type")
        if action_type == "search":
            queries = action.get("queries")
            if isinstance(queries, list):
                values = [value.strip() for value in queries if isinstance(value, str) and value.strip()]
                if values:
                    return "; ".join(dict.fromkeys(values))
            query = action.get("query")
            if isinstance(query, str) and query.strip():
                return query.strip()
        elif action_type == "openPage":
            url = action.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip()
        elif action_type == "findInPage":
            pattern = action.get("pattern")
            url = action.get("url")
            details = [value.strip() for value in (pattern, url) if isinstance(value, str) and value.strip()]
            if details:
                return " — ".join(details)

    query = item.get("query")
    return query.strip() if isinstance(query, str) else ""


def image_saved_path(item: object, working_directory: object) -> str:
    """Return a short image path without exposing unrelated host directories."""
    if not isinstance(item, dict):
        return ""
    path_value = item.get("savedPath")
    if not isinstance(path_value, str) or not path_value.strip():
        return ""
    path = Path(path_value.strip())
    if isinstance(working_directory, (str, Path)) and str(working_directory):
        try:
            base = Path(working_directory).resolve(strict=False)
            candidate = path if path.is_absolute() else base / path
            relative_path = candidate.resolve(strict=False).relative_to(base)
            return relative_path.as_posix()
        except (OSError, ValueError):
            return path.name
    if not path.is_absolute():
        return path.as_posix()
    return path.name


def _diff_line_counts(diff: str) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed
