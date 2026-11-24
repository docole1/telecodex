from __future__ import annotations

import re
from typing import Any

from telecodex.telegram.formatting import TELEGRAM_TEXT_LIMIT

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADING = re.compile(r"^ {0,3}#{1,6}(?:\s|$)")
_RULE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_LIST = re.compile(r"^\s*(?:[-*+]|\d{1,9}[.)])\s")
_RESERVE = 64


def format_long_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Split a plain Telegram reply at natural boundaries and keep Markdown blocks balanced.

    This adapts Waveo's `_split` algorithm for TeleCodex's plain `sendMessage` path. The
    character limit is per Telegram message; code fences and math blocks are closed and
    reopened across chunks, while table headers are repeated when a table is split.
    """
    if limit <= _RESERVE:
        raise ValueError(f"limit must be greater than {_RESERVE}")
    parts = _split(text, _Limits(limit, None), None)
    return _format_parts(parts)


def _kind(line: str) -> str:
    if _HEADING.match(line):
        return "heading"
    if _RULE.match(line):
        return "rule"
    stripped = line.lstrip()
    if stripped.startswith("|"):
        return "table"
    if stripped.startswith(">"):
        return "quote"
    if _LIST.match(line):
        return "list"
    return "text"


class _Scan:
    """Track open Markdown structures while scanning a reply line by line."""

    __slots__ = ("fence", "fence_line", "math", "kind", "blocks", "table_head")

    def __init__(self) -> None:
        self.fence: tuple[str, int] | None = None
        self.fence_line = ""
        self.math = False
        self.kind: str | None = None
        self.blocks = 0
        self.table_head: list[str] = []

    def copy(self) -> _Scan:
        other = _Scan()
        other.fence = self.fence
        other.fence_line = self.fence_line
        other.math = self.math
        other.kind = self.kind
        other.blocks = self.blocks
        other.table_head = list(self.table_head)
        return other

    def feed(self, line: str) -> None:
        match = _FENCE.match(line)
        if self.fence is not None:
            if _closes(self.fence, match):
                self.fence = None
                self.kind = "fence"
            return
        if match:
            self.fence = (match.group(1)[0], len(match.group(1)))
            self.fence_line = line.strip()
            self.blocks += 1
            self.table_head = []
            return
        if line.strip() == "$$":
            self.math = not self.math
            if self.math:
                self.blocks += 1
            self.kind = "math"
            self.table_head = []
            return
        if self.math:
            return
        if not line.strip():
            self.kind = None
            self.table_head = []
            return
        kind = _kind(line)
        if (
            kind in ("heading", "rule")
            or self.kind in (None, "fence", "math")
            or (kind in ("table", "quote", "list") and kind != self.kind)
        ):
            self.blocks += 1
        if kind == "table":
            if self.kind != "table":
                self.table_head = [line]
            elif len(self.table_head) < 2:
                self.table_head.append(line)
        else:
            self.table_head = []
        self.kind = kind


class _Limits:
    __slots__ = ("chars", "bytes_")

    def __init__(self, chars: int | None, bytes_: int | None) -> None:
        self.chars = chars
        self.bytes_ = bytes_

    def measure(self, text: str) -> tuple[int, int]:
        return len(text), len(text.encode("utf-8")) if self.bytes_ is not None else 0

    def fits(self, used: tuple[int, int], extra: tuple[int, int]) -> bool:
        if self.chars is not None and used[0] + extra[0] + 1 > self.chars - _RESERVE:
            return False
        return self.bytes_ is None or used[1] + extra[1] + 1 <= self.bytes_ - _RESERVE

    def prefix(self, text: str, used: tuple[int, int]) -> str:
        if self.chars is not None:
            text = text[: max(self.chars - _RESERVE - used[0], 1)]
        if self.bytes_ is not None:
            room = max(self.bytes_ - _RESERVE - used[1], 1)
            text = text.encode("utf-8")[:room].decode("utf-8", errors="ignore")
        return text


def _closes(fence: tuple[str, int], match: re.Match[str] | None) -> bool:
    return bool(match) and match.group(1)[0] == fence[0] and len(match.group(1)) >= fence[1] and not match.group(2).strip()


def _reopen(scan: _Scan) -> tuple[str, str, str]:
    if scan.fence is not None:
        char, length = scan.fence
        return "\n" + char * length, scan.fence_line, ""
    if scan.math:
        return "\n$$", "$$", ""
    if scan.kind == "table" and len(scan.table_head) == 2:
        return "", "", "\n".join(scan.table_head)
    return "", "", ""


def _split(text: str, limits: _Limits, max_blocks: int | None) -> list[dict[str, Any]]:
    lines = text.split("\n")
    sizes = [limits.measure(line) for line in lines]
    parts: list[dict[str, Any]] = []
    pos = 0
    opener = ""
    table_head = ""
    while pos < len(lines):
        if table_head and _kind(lines[pos]) == "table":
            opener = table_head
        table_head = ""
        scan = _Scan()
        base = (0, 0)
        if opener:
            for opener_line in opener.split("\n"):
                scan.feed(opener_line)
            base = limits.measure(opener)
            base = (base[0] + 1, base[1] + 1)
        used = base
        accepted = 0
        safe = soft = None
        overflow = False
        while pos + accepted < len(lines):
            line = lines[pos + accepted]
            probe = scan.copy()
            probe.feed(line)
            if not limits.fits(used, sizes[pos + accepted]) or (
                max_blocks is not None and probe.blocks > max_blocks
            ):
                overflow = True
                break
            scan = probe
            used = (
                used[0] + sizes[pos + accepted][0] + 1,
                used[1] + sizes[pos + accepted][1] + 1,
            )
            accepted += 1
            following = lines[pos + accepted] if pos + accepted < len(lines) else ""
            if scan.fence is None and not scan.math and not (
                scan.kind == "table" and _kind(following) == "table"
            ):
                if not line.strip():
                    safe = (accepted, used)
                else:
                    soft = (accepted, used)
        if not overflow:
            parts.append({"body": "\n".join(lines[pos:]), "opener": opener, "closer": ""})
            break

        if accepted == 0:
            line = lines[pos]
            piece = limits.prefix(line, base) or line[:1]
            space = piece.rfind(" ")
            if space > len(piece) // 3:
                piece = piece[: space + 1]
            check = scan.copy()
            check.feed(piece)
            closer, next_opener, table_head = _reopen(check)
            parts.append({"body": piece, "opener": opener, "closer": closer})
            lines[pos] = line[len(piece):]
            sizes[pos] = limits.measure(lines[pos])
            opener = next_opener
            continue

        def weight(point: tuple[int, tuple[int, int]] | None) -> int:
            return point[1][0] if point else -1

        if safe is not None and (soft is None or weight(safe) >= (weight(soft) + base[0]) / 2):
            keep = safe[0]
        elif soft is not None:
            keep = soft[0]
        else:
            keep = accepted
        while keep > 1 and (not lines[pos + keep - 1].strip() or _HEADING.match(lines[pos + keep - 1])):
            keep -= 1
        end = _Scan()
        if opener:
            for opener_line in opener.split("\n"):
                end.feed(opener_line)
        for kept in lines[pos : pos + keep]:
            end.feed(kept)
        closer, next_opener, table_head = _reopen(end)
        parts.append({
            "body": "\n".join(lines[pos : pos + keep]),
            "opener": opener,
            "closer": closer,
        })
        pos += keep
        opener = next_opener
    return parts


def _format_parts(parts: list[dict[str, Any]]) -> list[str]:
    chunks: list[str] = []
    for part in parts:
        content = (part["opener"] + "\n" if part["opener"] else "") + part["body"]
        content = content.rstrip("\n") + part["closer"] if part["closer"] else content.rstrip("\n")
        if content.strip():
            chunks.append(content)
    return chunks
