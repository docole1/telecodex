"""Read-only checks for newer TeleCodex releases on the GitHub origin."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from telecodex import __version__

UPDATE_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 8
_GITHUB_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_STABLE_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class GitHubRelease:
    tag_name: str
    version: str
    html_url: str


def check_for_update(repository_root: Path | None = None) -> GitHubRelease | None:
    """Return the latest public GitHub release if it is newer than this install."""
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[2]
    repository = _github_origin(repository_root)
    if repository is None:
        return None

    owner, name = repository
    endpoint = f"https://api.github.com/repos/{owner}/{name}/releases/latest"
    request = Request(
        endpoint,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"TeleCodex/{__version__}",
        },
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            # GitHub uses 404 when the public repository has no latest release.
            return None
        raise

    if not isinstance(payload, dict):
        raise ValueError("GitHub returned an invalid release response")
    tag_name = payload.get("tag_name")
    html_url = payload.get("html_url")
    if not isinstance(tag_name, str) or not isinstance(html_url, str):
        raise ValueError("GitHub release response is missing its tag or URL")

    latest_version = _version_tuple(tag_name)
    current_version = _version_tuple(__version__)
    if latest_version is None or current_version is None:
        raise ValueError("TeleCodex release tags must use MAJOR.MINOR.PATCH versions")
    if latest_version <= current_version:
        return None

    parsed_url = urlsplit(html_url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "github.com":
        raise ValueError("GitHub returned an invalid release URL")

    return GitHubRelease(
        tag_name=tag_name,
        version=tag_name.removeprefix("v"),
        html_url=html_url,
    )


def _version_tuple(value: str) -> tuple[int, int, int] | None:
    match = _STABLE_VERSION.fullmatch(value)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _github_origin(repository_root: Path) -> tuple[str, str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "remote", "get-url", "origin"],
            check=False,
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    remote = result.stdout.strip()
    if remote.startswith("git@github.com:"):
        path = remote.removeprefix("git@github.com:")
    else:
        parsed_remote = urlsplit(remote)
        if parsed_remote.hostname != "github.com":
            return None
        path = parsed_remote.path.lstrip("/")

    path = path.removesuffix(".git").rstrip("/")
    parts = path.split("/")
    if len(parts) != 2 or not all(_GITHUB_SEGMENT.fullmatch(part) for part in parts):
        return None
    return parts[0], parts[1]
