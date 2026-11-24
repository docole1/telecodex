from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from telecodex.shell.runner import ShellRunner


class ShellRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_selected_directory_inside_configured_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            runner = ShellRunner(root)

            result = await runner.run("pwd", working_directory=project)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(result.output.strip(), str(project))

    async def test_rejects_selected_directory_outside_configured_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ShellRunner(Path(temp_dir))

            with self.assertRaises(OSError):
                await runner.run("pwd", working_directory=Path("/tmp"))
