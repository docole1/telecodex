"""Attach a child process to the PTY already created by the parent."""

from __future__ import annotations

import fcntl
import os
import sys
import termios


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "--":
        return 2

    command = sys.argv[2:]
    try:
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    except OSError:
        return 3

    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
