from __future__ import annotations

import json
import re
import string
import unittest
from pathlib import Path

from telecodex.i18n import TextCatalog


class TextCatalogHtmlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = TextCatalog()

    def test_html_templates_escape_dynamic_values(self) -> None:
        text = self.catalog.render(
            "ru",
            "mcp.label.confirm_disable",
            name="docs <strong>operator</strong>",
        )

        self.assertIn(
            "<strong>docs &lt;strong&gt;operator&lt;/strong&gt;</strong>",
            text,
        )

    def test_nested_catalog_html_survives_template_composition(self) -> None:
        mode = self.catalog.render("ru", "permissions.buttons.mode_names.ask")
        current = self.catalog.render("ru", "permissions.values.current", mode=mode)
        current += self.catalog.render("ru", "permissions.values.next_turn")
        prompt = self.catalog.render(
            "ru",
            "permissions.label.prompt",
            scope=self.catalog.render("ru", "permissions.values.current_thread"),
            current=current,
        )

        self.assertIn("Сейчас: <strong>Спрашивать при необходимости</strong>.", prompt)
        self.assertNotIn("&lt;strong&gt;Спрашивать", prompt)

    def test_english_catalog_translates_html_templates_and_escapes_values(self) -> None:
        text = self.catalog.render(
            "en",
            "mcp.label.confirm_disable",
            name="docs <strong>operator</strong>",
        )

        self.assertIn(
            "Disable MCP server <strong>docs &lt;strong&gt;operator&lt;/strong&gt;</strong>",
            text,
        )

    def test_language_catalogs_have_matching_keys_and_formatting(self) -> None:
        root = Path("src/telecodex/texts")
        russian = json.loads((root / "ru.json").read_text(encoding="utf-8"))
        english = json.loads((root / "en.json").read_text(encoding="utf-8"))

        self.assertEqual(type(russian), type(english))
        russian_values = _flatten(russian)
        english_values = _flatten(english)
        self.assertEqual(russian_values.keys(), english_values.keys())

        for path in russian_values:
            with self.subTest(path=".".join(path)):
                source = russian_values[path]
                translated = english_values[path]
                self.assertEqual(type(source), type(translated))
                if not isinstance(source, str):
                    continue
                self.assertTrue(translated.strip())
                self.assertEqual(_placeholders(source), _placeholders(translated))
                self.assertEqual(_html_tags(source), _html_tags(translated))
                self.assertEqual(source.count("\n"), translated.count("\n"))
                self.assertIsNone(re.search(r"[А-Яа-яЁё]", translated))

    def test_system_catalog_is_loaded_from_separate_file(self) -> None:
        system = json.loads(
            Path("src/telecodex/texts/system.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            self.catalog.system("buttons.pager.page", current=1, total=2),
            system["buttons"]["pager"]["page"].format(current=1, total=2),
        )


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, dict):
        flattened: dict[tuple[str, ...], object] = {}
        for key, item in value.items():
            flattened.update(_flatten(item, prefix + (key,)))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten(item, prefix + (str(index),)))
        return flattened
    return {prefix: value}


def _placeholders(value: str) -> list[str]:
    return [field for _, field, _, _ in string.Formatter().parse(value) if field is not None]


def _html_tags(value: str) -> list[str]:
    return re.findall(r"</?(?:strong|code|pre)\b[^>]*>", value, re.IGNORECASE)
