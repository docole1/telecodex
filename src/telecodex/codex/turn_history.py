"""Read Codex turn and item history from the app-server."""

from __future__ import annotations

from typing import Any


class CodexTurnHistory:
    """Read persisted turn records and reconstruct user-visible turn text."""

    def __init__(self, app_server: Any) -> None:
        self.app_server = app_server

    async def _read_persisted_turn(self, thread_id: str, turn_id: str) -> dict[str, Any] | None:
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _ in range(20):
            params: dict[str, Any] = {
                "threadId": thread_id,
                "limit": 100,
                "sortDirection": "desc",
                "itemsView": "notLoaded",
            }
            if cursor is not None:
                params["cursor"] = cursor
            result = await self.app_server.request("thread/turns/list", params, timeout_seconds=60)
            data = result.get("data")
            if not isinstance(data, list):
                raise RuntimeError("Codex app-server returned invalid turn history")
            for turn in data:
                if isinstance(turn, dict) and turn.get("id") == turn_id:
                    return turn
            next_cursor = result.get("nextCursor")
            if next_cursor is None:
                return None
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
                raise RuntimeError("Codex app-server returned an invalid turn-history cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        raise RuntimeError("Codex app-server turn history exceeded the recovery page limit")


    async def last_assistant_message(self, thread_id: str) -> str:
        """Return the latest user-visible assistant message without resuming the thread."""
        async with asyncio.timeout(20):
            await self._ensure_app_server()
            result = await self.app_server.request(
                "thread/turns/list",
                {
                    "threadId": thread_id,
                    "limit": 20,
                    "sortDirection": "desc",
                    "itemsView": "notLoaded",
                },
                timeout_seconds=15,
            )
            turns = result.get("data")
            if not isinstance(turns, list):
                raise RuntimeError("Codex app-server returned invalid turn history")
            for turn in turns:
                turn_id = turn.get("id") if isinstance(turn, dict) else None
                if not isinstance(turn_id, str) or not turn_id:
                    continue
                text = await self._read_turn_text(
                    thread_id,
                    turn_id,
                    timeout_seconds=15,
                )
                if text:
                    return text
        return ""


    async def _read_turn_text(
        self,
        thread_id: str,
        turn_id: str | None,
        *,
        timeout_seconds: int = 60,
    ) -> str:
        if not turn_id:
            return ""
        cursor: str | None = None
        seen_cursors: set[str] = set()
        messages: list[dict[str, Any]] = []
        plans: list[dict[str, Any]] = []
        reviews: list[dict[str, Any]] = []
        for _ in range(20):
            params: dict[str, Any] = {
                "threadId": thread_id,
                "turnId": turn_id,
                "limit": 100,
                "sortDirection": "asc",
            }
            if cursor is not None:
                params["cursor"] = cursor
            result = await self.app_server.request(
                "thread/items/list",
                params,
                timeout_seconds=timeout_seconds,
            )
            data = result.get("data")
            if not isinstance(data, list):
                raise RuntimeError("Codex app-server returned invalid turn items")
            for entry in data:
                item = entry.get("item") if isinstance(entry, dict) else None
                if isinstance(item, dict) and item.get("type") == "agentMessage":
                    messages.append(item)
                elif isinstance(item, dict) and item.get("type") == "plan":
                    plans.append(item)
                elif isinstance(item, dict) and item.get("type") == "exitedReviewMode":
                    reviews.append(item)
            next_cursor = result.get("nextCursor")
            if next_cursor is None:
                break
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
                raise RuntimeError("Codex app-server returned an invalid item-history cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        else:
            raise RuntimeError("Codex app-server item history exceeded the recovery page limit")

        final_plan = next(
            (
                item["text"].strip()
                for item in reversed(plans)
                if isinstance(item.get("text"), str) and item["text"].strip()
            ),
            None,
        )
        if final_plan:
            return final_plan

        final_messages = [item for item in messages if item.get("phase") == "final_answer"]
        if final_messages:
            text = final_messages[-1].get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        final_review = next(
            (
                item["review"].strip()
                for item in reversed(reviews)
                if isinstance(item.get("review"), str) and item["review"].strip()
            ),
            None,
        )
        if final_review:
            return final_review
        text = messages[-1].get("text") if messages else None
        return text.strip() if isinstance(text, str) else ""


    @staticmethod
    def _turn_error_message(turn: dict[str, Any]) -> str | None:
        error = turn.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        return message if isinstance(message, str) and message else None
