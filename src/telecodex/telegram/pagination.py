from __future__ import annotations

from typing import Any

PAGE_SIZE = 10
INLINE_BUTTON_TEXT_LIMIT = 64
MAX_PAIRED_BUTTON_TEXT_LENGTH = 18
MAX_PAIRED_ROW_TEXT_LENGTH = 32


def page_count(item_count: int) -> int:
    return max(1, (item_count + PAGE_SIZE - 1) // PAGE_SIZE)


def two_column_rows(buttons: list[dict[str, str]], page: int) -> list[list[dict[str, str]]]:
    """Keep short labels in pairs and give longer labels a full-width row."""
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, len(buttons))
    rows: list[list[dict[str, str]]] = []
    for index in range(start, end, 2):
        pair = buttons[index : min(index + 2, end)]
        pair_texts = [button["text"] for button in pair]
        fits_paired_row = (
            len(pair) == 2
            and all(len(text) <= MAX_PAIRED_BUTTON_TEXT_LENGTH for text in pair_texts)
            and sum(len(text) for text in pair_texts) <= MAX_PAIRED_ROW_TEXT_LENGTH
        )
        if fits_paired_row:
            rows.append(pair)
        else:
            rows.extend([[button] for button in pair])
    return rows


def navigation_row(
    page: int,
    count: int,
    previous_text: str,
    page_text: str,
    next_text: str,
    callback_prefix: str,
) -> list[dict[str, str]]:
    return [
        {"text": previous_text, "callback_data": f"{callback_prefix}:p:{page - 1}"},
        {"text": page_text.format(current=page + 1, total=count), "callback_data": f"{callback_prefix}:i"},
        {"text": next_text, "callback_data": f"{callback_prefix}:p:{page + 1}"},
    ]


def keyboard(rows: list[list[dict[str, str]]]) -> dict[str, Any]:
    return {"inline_keyboard": rows}
