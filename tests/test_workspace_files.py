from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from multi_user_test_support import callback_event
from telecodex.telegram.pickers import FilesPicker
from telecodex.telegram.user_files import TelegramWorkspaceFilesController
from telecodex.workspace_files import (
    WorkspaceArchiveTooLarge,
    WorkspaceFileBrowser,
    WorkspaceFileTooLarge,
    WorkspaceFilesError,
    WorkspacePreviewUnavailable,
)


class _ArchiveTelegram:
    def __init__(self) -> None:
        self.document: tuple[int, str, bytes, int | None] | None = None
        self.messages: list[tuple[int, str]] = []

    async def answer_callback_query(self, _callback_id: str, _text: str | None = None) -> None:
        return None

    async def send_document(
        self,
        *,
        chat_id: int,
        document,
        reply_to_message_id: int | None = None,
    ) -> None:
        path = Path(document.path)
        self.document = (chat_id, document.filename, path.read_bytes(), reply_to_message_id)

    async def send_message(self, chat_id: int, text: str, **_kwargs) -> None:
        self.messages.append((chat_id, text))


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

    def test_archives_nested_files_and_empty_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            nested = project / "src"
            empty = project / "empty"
            nested.mkdir(parents=True)
            empty.mkdir()
            (nested / "app.py").write_text("print('ok')", encoding="utf-8")
            destination = root / "project.zip"

            filename, archive_path = WorkspaceFileBrowser(root).archive_directory(
                project,
                destination,
            )

            self.assertEqual(filename, "project.zip")
            self.assertEqual(archive_path, destination)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {"project/", "project/src/", "project/empty/", "project/src/app.py"},
                )
                self.assertEqual(archive.read("project/src/app.py"), b"print('ok')")

    def test_archive_skips_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            outside = root / "outside.txt"
            outside.write_text("secret", encoding="utf-8")
            try:
                (project / "outside-link.txt").symlink_to(outside)
                (project / "outside-directory").symlink_to(root, target_is_directory=True)
            except OSError:
                self.skipTest("Symbolic links are not available")
            destination = root / "project.zip"

            WorkspaceFileBrowser(root).archive_directory(project, destination)

            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(archive.namelist(), ["project/"])

    def test_archive_stops_at_telegram_upload_limit_and_removes_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            (project / "payload.bin").write_bytes(bytes(range(256)) * 16)
            destination = root / "project.zip"

            with patch("telecodex.workspace_files.MAX_DOWNLOAD_BYTES", 100):
                with self.assertRaises(WorkspaceArchiveTooLarge):
                    WorkspaceFileBrowser(root).archive_directory(project, destination)

            self.assertFalse(destination.exists())

    def test_archive_rejects_excessive_uncompressed_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            (project / "payload.txt").write_text("payload", encoding="utf-8")
            destination = root / "project.zip"

            with patch("telecodex.workspace_files.MAX_ARCHIVE_SOURCE_BYTES", 3):
                with self.assertRaises(WorkspaceArchiveTooLarge):
                    WorkspaceFileBrowser(root).archive_directory(project, destination)

            self.assertFalse(destination.exists())


class TelegramWorkspaceArchiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_files_menu_sends_current_folder_as_a_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            (project / "readme.txt").write_text("hello", encoding="utf-8")
            telegram = _ArchiveTelegram()
            controller = TelegramWorkspaceFilesController(
                telegram=telegram,  # type: ignore[arg-type]
                codex=lambda: None,
                workspace_root=lambda: root,
                text=lambda key, **_values: key,
                texts=SimpleNamespace(system=lambda _key: "…"),
                edit_message=lambda *_args, **_kwargs: None,
                log_failure=lambda *_args, **_kwargs: None,
            )
            picker = FilesPicker(
                token="files-token",
                owner_user_id=101,
                current_path=project,
                entries=[],
                message_id=77,
            )
            controller.picker = picker

            menu = controller._files_picker_keyboard(picker)
            self.assertIn(
                "fb:files-token:z",
                [button["callback_data"] for row in menu["inline_keyboard"] for button in row],
            )
            incoming = callback_event(
                1,
                "archive-folder",
                101,
                101,
                77,
                "fb:files-token:z",
            )
            await controller._handle_files_callback(incoming, ["fb", "files-token", "z"])

            self.assertIsNotNone(telegram.document)
            assert telegram.document is not None
            chat_id, filename, payload, reply_to = telegram.document
            self.assertEqual((chat_id, filename, reply_to), (101, "project.zip", 77))
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                self.assertEqual(archive.read("project/readme.txt"), b"hello")
