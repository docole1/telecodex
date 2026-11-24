"""Small dependency-free helper for systemd service notifications."""

from __future__ import annotations

import asyncio
import os
import socket


class SystemdNotifier:
    """Send readiness and watchdog datagrams when launched by systemd."""

    def __init__(self) -> None:
        value = os.environ.get("NOTIFY_SOCKET", "")
        self._address: str | None = None
        if value:
            self._address = "\0" + value[1:] if value.startswith("@") else value
        try:
            watchdog_usec = int(os.environ.get("WATCHDOG_USEC", "0"))
        except ValueError:
            watchdog_usec = 0
        self.interval_seconds = max(1.0, watchdog_usec / 2_000_000) if watchdog_usec > 0 else None

    @property
    def enabled(self) -> bool:
        return self._address is not None

    def send(self, *fields: str) -> None:
        if self._address is None:
            return
        payload = "\n".join(field for field in fields if field).encode()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                client.connect(self._address)
                client.sendall(payload)
        except OSError:
            # Losing a notifier must not terminate the actual service.
            return

    def ready(self, status: str | None = None) -> None:
        self.send("READY=1", f"STATUS={status}" if status else "")

    def watchdog(self, status: str | None = None) -> None:
        self.send("WATCHDOG=1", f"STATUS={status}" if status else "")

    async def sleep_interval(self) -> None:
        if self.interval_seconds is not None:
            await asyncio.sleep(self.interval_seconds)
