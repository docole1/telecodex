from __future__ import annotations

import unittest

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
