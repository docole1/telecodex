from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_legacy_json_object(path: Path) -> dict[str, Any]:
    """Read the one-time JSON state import without accepting non-objects."""
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value
