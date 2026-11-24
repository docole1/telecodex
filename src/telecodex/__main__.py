from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from pathlib import Path

from dotenv import load_dotenv

from telecodex.config import load_config, state_paths_from_env
from telecodex.logging_config import configure_logging
from telecodex.service import TeleCodexService
from telecodex.state.sqlite import export_state_json

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="telecodex", description="Telegram interface for Codex on your Linux system")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("run", help="run the Telegram bot")
    state_parser = commands.add_parser("state", help="manage local runtime state")
    state_commands = state_parser.add_subparsers(dest="state_command", required=True)
    export_parser = state_commands.add_parser(
        "export-json", help="export SQLite state to a JSON rollback file"
    )
    export_parser.add_argument("path", help="destination path for the JSON export")
    return parser


async def _run_service() -> None:
    config = load_config()
    service = TeleCodexService(config)
    task = asyncio.create_task(service.run())
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    except (NotImplementedError, RuntimeError):
        pass
    try:
        await task
    except asyncio.CancelledError:
        pass


def main() -> None:
    load_dotenv()
    configure_logging()
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "state" and args.state_command == "export-json":
            database_file, _ = state_paths_from_env()
            export_state_json(database_file, Path(args.path))
            print(f"State exported: {Path(args.path).expanduser().resolve()}")
            return
        if args.command not in {None, "run"}:
            parser.error("unknown command")
        asyncio.run(_run_service())
    except KeyboardInterrupt:
        return
    except Exception as exc:
        logger.exception(
            "TeleCodex CLI command failed",
            extra={"event": "telecodex.cli.failed", "error_type": type(exc).__name__},
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
