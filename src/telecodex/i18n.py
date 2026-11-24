from __future__ import annotations

import html
import json
import re
from importlib.resources import files
from typing import Any

DEFAULT_LANGUAGE = "ru"
SUPPORTED_LANGUAGES = ("ru", "en")
_HTML_TAG = re.compile(r"<\s*/?\s*(?:strong|code|pre)\b", re.IGNORECASE)


class _HtmlText(str):
    """A rendered catalog string that contains trusted HTML formatting."""

    def __add__(self, other: object) -> _HtmlText:
        if not isinstance(other, str):
            return NotImplemented
        suffix = (
            str(other)
            if isinstance(other, _HtmlText)
            else html.escape(other, quote=False)
        )
        return _HtmlText(str(self) + suffix)

    def __radd__(self, other: object) -> _HtmlText:
        if not isinstance(other, str):
            return NotImplemented
        prefix = (
            str(other)
            if isinstance(other, _HtmlText)
            else html.escape(other, quote=False)
        )
        return _HtmlText(prefix + str(self))


class TextCatalog:
    """Load and format the bot's user-facing text from its packaged JSON catalog."""

    def __init__(self) -> None:
        resource_dir = files("telecodex").joinpath("texts")
        texts: dict[str, dict[str, Any]] = {}
        for language in SUPPORTED_LANGUAGES:
            content = resource_dir.joinpath(f"{language}.json").read_text(encoding="utf-8")
            language_texts = json.loads(content)
            if not isinstance(language_texts, dict):
                raise ValueError(f"texts/{language}.json must contain an object")
            texts[language] = language_texts
        system_content = resource_dir.joinpath("system.json").read_text(encoding="utf-8")
        system = json.loads(system_content)
        if not isinstance(system, dict):
            raise ValueError("texts/system.json must contain an object")
        self._texts: dict[str, dict[str, Any]] = texts
        self._system: dict[str, Any] = system

    def render(self, language: str, key: str, **values: Any) -> str:
        template = self._lookup(self._texts.get(language, {}), key)
        if not isinstance(template, str):
            return self.system("label.missing_translation", language=language, key=key)
        try:
            return _format_template(template, values)
        except (KeyError, ValueError, IndexError):
            return self.system("label.missing_translation", language=language, key=key)

    def system(self, system_key: str, **values: Any) -> str:
        template = self._lookup(self._system, system_key)
        if not isinstance(template, str):
            raise ValueError(f"texts/system.json is missing system text '{system_key}'")
        return _format_template(template, values)

    @staticmethod
    def _lookup(source: dict[str, Any], key: str) -> Any:
        value: Any = source
        for part in key.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value


def _format_template(template: str, values: dict[str, Any]) -> str:
    """Escape interpolated data in HTML templates and preserve nested catalog markup."""
    has_html = bool(_HTML_TAG.search(template))
    formatted_values: dict[str, Any] = {}
    contains_html = has_html
    for key, value in values.items():
        if isinstance(value, _HtmlText):
            formatted_values[key] = str(value)
            contains_html = True
        elif has_html:
            formatted_values[key] = html.escape(str(value), quote=False)
        else:
            formatted_values[key] = value
    rendered = template.format(**formatted_values)
    return _HtmlText(rendered) if contains_html else rendered


def load_text_catalog() -> TextCatalog:
    return TextCatalog()
