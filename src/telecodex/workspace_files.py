from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

MAX_PREVIEW_BYTES = 32 * 1024
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


class WorkspaceFilesError(RuntimeError):
    """A safe error while browsing files within the configured workspace."""


class WorkspaceFileTooLarge(WorkspaceFilesError):
    """The selected file exceeds Telegram's upload limit."""


class WorkspacePreviewUnavailable(WorkspaceFilesError):
    """The selected file cannot be shown as a UTF-8 text preview."""


@dataclass(frozen=True)
class WorkspaceEntry:
    name: str
    path: Path
    kind: str
    size: int | None


@dataclass(frozen=True)
class TextPreview:
    text: str
    truncated: bool


class WorkspaceFileBrowser:
    max_download_bytes = MAX_DOWNLOAD_BYTES

    def __init__(self, root: Path) -> None:
        try:
            self.root = root.expanduser().resolve(strict=True)
        except OSError as exc:
            raise WorkspaceFilesError("The configured workspace is unavailable") from exc
        if not self.root.is_dir():
            raise WorkspaceFilesError("The configured workspace is not a directory")

    def resolve_directory(self, directory: Path) -> Path:
        path = self._resolve(directory)
        if not path.is_dir():
            raise WorkspaceFilesError("The selected path is not a directory")
        return path

    def list_entries(self, directory: Path) -> list[WorkspaceEntry]:
        path = self.resolve_directory(directory)

        entries: list[WorkspaceEntry] = []
        try:
            with os.scandir(path) as children:
                for child in children:
                    try:
                        metadata = child.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISDIR(metadata.st_mode):
                        kind = "directory"
                        size = None
                    elif stat.S_ISREG(metadata.st_mode):
                        kind = "file"
                        size = metadata.st_size
                    else:
                        continue
                    entries.append(
                        WorkspaceEntry(
                            name=child.name,
                            path=path / child.name,
                            kind=kind,
                            size=size,
                        )
                    )
        except OSError as exc:
            raise WorkspaceFilesError("Could not list the selected directory") from exc
        return sorted(entries, key=lambda entry: (entry.kind != "directory", entry.name.casefold()))

    def read_text_preview(self, path: Path) -> TextPreview:
        descriptor = self._open_regular_file(path)
        try:
            with os.fdopen(descriptor, "rb") as source:
                payload = source.read(MAX_PREVIEW_BYTES + 1)
        except OSError as exc:
            raise WorkspaceFilesError("Could not read the selected file") from exc
        truncated = len(payload) > MAX_PREVIEW_BYTES
        try:
            text = payload[:MAX_PREVIEW_BYTES].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspacePreviewUnavailable from exc
        return TextPreview(text=text, truncated=truncated)

    def copy_file(self, path: Path, destination: Path) -> tuple[str, Path]:
        descriptor = self._open_regular_file(path)
        try:
            source_stat = os.fstat(descriptor)
            if source_stat.st_size > MAX_DOWNLOAD_BYTES:
                raise WorkspaceFileTooLarge
            filename = Path(path).name
            if not filename or filename in {".", ".."}:
                raise WorkspaceFilesError("The selected file has an invalid name")
            total = 0
            created = False
            try:
                with os.fdopen(descriptor, "rb") as source:
                    target = destination.open("xb")
                    created = True
                    with target:
                        while chunk := source.read(64 * 1024):
                            total += len(chunk)
                            if total > MAX_DOWNLOAD_BYTES:
                                raise WorkspaceFileTooLarge
                            target.write(chunk)
            except Exception:
                if created:
                    destination.unlink(missing_ok=True)
                raise
            return filename, destination
        except OSError as exc:
            raise WorkspaceFilesError("Could not copy the selected file") from exc
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _open_regular_file(self, path: Path) -> int:
        validated = self._resolve(path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(validated, flags)
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise WorkspaceFilesError("The selected file is unavailable") from exc
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise WorkspaceFilesError("The selected path is not a regular file")
        return descriptor

    def _resolve(self, path: Path) -> Path:
        candidate = path.expanduser()
        if candidate.is_symlink():
            raise WorkspaceFilesError("Symbolic links cannot be browsed")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise WorkspaceFilesError("The selected path is unavailable") from exc
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceFilesError("The selected path is outside the configured workspace") from exc
        return resolved
