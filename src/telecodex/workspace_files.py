from __future__ import annotations

import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

MAX_PREVIEW_BYTES = 32 * 1024
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_SOURCE_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 10_000
ARCHIVE_COPY_CHUNK_BYTES = 64 * 1024


class WorkspaceFilesError(RuntimeError):
    """A safe error while browsing files within the configured workspace."""


class WorkspaceFileTooLarge(WorkspaceFilesError):
    """The selected file exceeds Telegram's upload limit."""


class WorkspaceArchiveTooLarge(WorkspaceFilesError):
    """The selected folder archive exceeds safe size or entry limits."""


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

    def archive_directory(self, directory: Path, destination: Path) -> tuple[str, Path]:
        """Create a bounded ZIP containing regular files under a workspace folder."""
        selected = self.resolve_directory(directory)
        directory_name = selected.name or "workspace"
        self._validate_archive_parts((directory_name,))
        archive_name = f"{directory_name}.zip"
        entry_count = 1
        source_bytes = 0

        def on_walk_error(error: OSError) -> None:
            raise WorkspaceFilesError("Could not read the selected folder") from error

        try:
            with destination.open("xb", buffering=0) as output:
                with zipfile.ZipFile(
                    output,
                    mode="w",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                ) as archive:
                    root_info = zipfile.ZipInfo(f"{directory_name}/")
                    root_info.external_attr = (stat.S_IFDIR | 0o755) << 16
                    archive.writestr(root_info, b"")

                    for current, directory_names, file_names in os.walk(
                        selected,
                        topdown=True,
                        followlinks=False,
                        onerror=on_walk_error,
                    ):
                        current_path = Path(current)
                        safe_directory_names: list[str] = []
                        for name in sorted(directory_names):
                            child = current_path / name
                            try:
                                metadata = child.stat(follow_symlinks=False)
                            except OSError as exc:
                                raise WorkspaceFilesError(
                                    "Could not inspect a folder in the selected directory"
                                ) from exc
                            if not stat.S_ISDIR(metadata.st_mode):
                                continue
                            safe_directory_names.append(name)
                            entry_count += 1
                            if entry_count > MAX_ARCHIVE_ENTRIES:
                                raise WorkspaceArchiveTooLarge
                            member = self._archive_member_name(
                                selected,
                                child,
                                directory_name,
                                is_directory=True,
                            )
                            directory_info = zipfile.ZipInfo(member)
                            directory_info.external_attr = (stat.S_IFDIR | 0o755) << 16
                            archive.writestr(directory_info, b"")
                            if output.tell() > MAX_DOWNLOAD_BYTES:
                                raise WorkspaceArchiveTooLarge

                        directory_names[:] = safe_directory_names
                        for name in sorted(file_names):
                            child = current_path / name
                            try:
                                metadata = child.stat(follow_symlinks=False)
                            except OSError as exc:
                                raise WorkspaceFilesError(
                                    "Could not inspect a file in the selected directory"
                                ) from exc
                            if not stat.S_ISREG(metadata.st_mode):
                                continue
                            entry_count += 1
                            if entry_count > MAX_ARCHIVE_ENTRIES:
                                raise WorkspaceArchiveTooLarge
                            if source_bytes + metadata.st_size > MAX_ARCHIVE_SOURCE_BYTES:
                                raise WorkspaceArchiveTooLarge
                            member = self._archive_member_name(
                                selected,
                                child,
                                directory_name,
                            )
                            descriptor = self._open_regular_file(child)
                            with os.fdopen(descriptor, "rb") as source:
                                with archive.open(member, "w") as archived_file:
                                    while chunk := source.read(ARCHIVE_COPY_CHUNK_BYTES):
                                        source_bytes += len(chunk)
                                        if source_bytes > MAX_ARCHIVE_SOURCE_BYTES:
                                            raise WorkspaceArchiveTooLarge
                                        archived_file.write(chunk)
                                        if output.tell() > MAX_DOWNLOAD_BYTES:
                                            raise WorkspaceArchiveTooLarge

            if destination.stat().st_size > MAX_DOWNLOAD_BYTES:
                raise WorkspaceArchiveTooLarge
        except WorkspaceFilesError:
            destination.unlink(missing_ok=True)
            raise
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            destination.unlink(missing_ok=True)
            raise WorkspaceFilesError("Could not create a ZIP for the selected folder") from exc
        return archive_name, destination

    @staticmethod
    def _validate_archive_parts(parts: tuple[str, ...]) -> None:
        if any(
            not part
            or part in {".", ".."}
            or any(character in part for character in ("/", "\\", "\0", ":"))
            for part in parts
        ):
            raise WorkspaceFilesError("The selected folder contains an unsupported archive path")

    def _archive_member_name(
        self,
        selected: Path,
        path: Path,
        directory_name: str,
        *,
        is_directory: bool = False,
    ) -> str:
        try:
            relative_parts = path.relative_to(selected).parts
        except ValueError as exc:
            raise WorkspaceFilesError("A folder entry is outside the selected directory") from exc
        parts = (directory_name, *relative_parts)
        self._validate_archive_parts(parts)
        member = "/".join(parts)
        return member + "/" if is_directory else member

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
