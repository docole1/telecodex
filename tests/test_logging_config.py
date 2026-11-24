from __future__ import annotations

import json
import logging
import unittest

from telecodex.logging_config import _JsonLogFormatter
from telecodex.logging_config import _LogContextFilter
from telecodex.logging_config import bind_log_context


class TeleCodexLoggingTests(unittest.TestCase):
    def test_json_formatter_redacts_tokens_and_emails(self) -> None:
        record = logging.LogRecord(
            "telecodex.test",
            logging.WARNING,
            __file__,
            10,
            'request failed https://api.telegram.org/bot123456:abcdefghijklmnopqrstuvwxyz123456?access_token=secret-value {"access_token":"json-secret"} sk-123456789012345678901234 ghp_abcdefghijklmnopqrstuvwxyz123456 contact dev@example.com',
            (),
            None,
        )
        record.event = "test.failure"
        record.http_status = 401

        rendered = _JsonLogFormatter().format(record)

        self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", rendered)
        self.assertNotIn("secret-value", rendered)
        self.assertNotIn("json-secret", rendered)
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz123456", rendered)
        self.assertNotIn("dev@example.com", rendered)
        self.assertIn("test.failure", rendered)
        self.assertEqual(json.loads(rendered)["http_status"], 401)

    def test_context_fields_attach_to_log_record(self) -> None:
        record = logging.LogRecord("telecodex.test", logging.INFO, __file__, 20, "step", (), None)
        with bind_log_context(correlation_id="update-123", update_id=123, raw_prompt="private text"):
            _LogContextFilter().filter(record)

        rendered = _JsonLogFormatter().format(record)
        data = json.loads(rendered)
        self.assertEqual(data["correlation_id"], "update-123")
        self.assertEqual(data["update_id"], 123)
        self.assertNotIn("raw_prompt", data)
        self.assertNotIn("private text", rendered)

    def test_exception_output_does_not_include_exception_message(self) -> None:
        try:
            raise ValueError("private user message")
        except ValueError:
            record = logging.LogRecord("telecodex.test", logging.ERROR, __file__, 30, "operation failed", (), None)
            record.exc_info = __import__("sys").exc_info()

        rendered = _JsonLogFormatter().format(record)
        data = json.loads(rendered)
        self.assertEqual(data["exception"]["type"], "ValueError")
        self.assertNotIn("private user message", rendered)
        self.assertTrue(data["exception"]["stack"])


if __name__ == "__main__":
    unittest.main()
