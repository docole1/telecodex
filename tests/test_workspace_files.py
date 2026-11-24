from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from telecodex.workspace_files import (
    WorkspaceFileBrowser,
    WorkspaceFileTooLarge,
    WorkspaceFilesError,
    WorkspacePreviewUnavailable,
)


class WorkspaceFileBrowserTests(unittest.TestCase):
    def test_lists_only_regular_files_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            (project / "src").mkdir()
            (project / "note.txt").write_text("hello", encoding="utf-8")
            try:
                (project / "outside-link").symlink_to(root, target_is_directory=True)
            except OSError:
                self.skipTest("Symbolic links are not available")

            entries = WorkspaceFileBrowser(root).list_entries(project)

            self.assertEqual([(item.name, item.kind) for item in entries], [
                ("src", "directory"),
                ("note.txt", "file"),
            ])

    def test_rejects_paths_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            outside = Path(temp_dir) / "outside"
            root.mkdir()
            outside.mkdir()

            with self.assertRaises(WorkspaceFilesError):
                WorkspaceFileBrowser(root).list_entries(outside)

    def test_preview_is_bounded_and_requires_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text_file = root / "long.txt"
            text_file.write_text("x" * (32 * 1024 + 10), encoding="utf-8")
            binary_file = root / "binary.dat"
            binary_file.write_bytes(b"\xff\xfe")
            browser = WorkspaceFileBrowser(root)

            preview = browser.read_text_preview(text_file)

            self.assertEqual(len(preview.text.encode("utf-8")), 32 * 1024)
            self.assertTrue(preview.truncated)
            with self.assertRaises(WorkspacePreviewUnavailable):
                browser.read_text_preview(binary_file)

    def test_download_copies_file_and_enforces_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "note.txt"
            source.write_text("content", encoding="utf-8")
            destination = root / "copy"
            browser = WorkspaceFileBrowser(root)

            name, copied = browser.copy_file(source, destination)

            self.assertEqual(name, "note.txt")
            self.assertEqual(copied.read_text(encoding="utf-8"), "content")
            with patch("telecodex.workspace_files.MAX_DOWNLOAD_BYTES", 3):
                with self.assertRaises(WorkspaceFileTooLarge):
                    browser.copy_file(source, root / "too-large")
            self.assertFalse((root / "too-large").exists())
