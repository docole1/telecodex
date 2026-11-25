from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiogram import Bot
    from telecodex.audio.transcriber import LocalAudioTranscriber
    from telecodex.codex.session import CodexSession


@dataclass(frozen=True)
class DoctorReport:
    telegram_ok: bool
    telegram_error: str | None
    codex_process_running: bool
    codex_rpc_ok: bool | None
    codex_error: str | None
    chatgpt_auth: str
    telegram_proxy_configured: bool
    thread_selected: bool
    task_busy: bool
    audio_transcription_status: str = "missing"
    poll_error_count: int = 0
    poll_age_seconds: int | None = None


async def collect_doctor_report(
    telegram: Bot,
    codex: CodexSession,
    telegram_proxy_configured: bool,
    audio_transcriber: LocalAudioTranscriber | None = None,
    poll_health: dict[str, int | None] | None = None,
) -> DoctorReport:
    checks = [
        _check_telegram(telegram),
        _check_codex(codex),
    ]
    results = await asyncio.gather(*checks)
    telegram_result, codex_result = results[:2]
    telegram_ok, telegram_error = telegram_result
    process_running, rpc_ok, codex_error, chatgpt_auth = codex_result
    return DoctorReport(
        telegram_ok=telegram_ok,
        telegram_error=telegram_error,
        codex_process_running=process_running,
        codex_rpc_ok=rpc_ok,
        codex_error=codex_error,
        chatgpt_auth=chatgpt_auth,
        telegram_proxy_configured=telegram_proxy_configured,
        thread_selected=bool(codex.state.thread_id),
        task_busy=codex.is_busy,
        audio_transcription_status=(audio_transcriber.status if audio_transcriber else "missing"),
        poll_error_count=int((poll_health or {}).get("error_count") or 0),
        poll_age_seconds=(poll_health or {}).get("age_seconds"),
    )


async def _check_telegram(telegram: Bot) -> tuple[bool, str | None]:
    try:
        await telegram.get_me(request_timeout=8)
    except Exception as exc:
        cause = exc.__cause__
        detail = type(cause).__name__ if cause is not None else type(exc).__name__
        return False, detail
    return True, None


async def _check_codex(
    codex: CodexSession,
) -> tuple[bool, bool | None, str | None, str]:
    if not codex.app_server.is_running:
        return False, None, None, "unknown"

    try:
        result = await codex.app_server.request(
            "account/read",
            {"refreshToken": False},
            timeout_seconds=8,
        )
    except Exception as exc:
        detail = type(exc).__name__
        return codex.app_server.is_running, False, detail, "unknown"

    account = result.get("account")
    account_type = account.get("type") if isinstance(account, dict) else None
    if account_type == "chatgpt" and result.get("requiresOpenaiAuth") is True:
        auth = "ok"
    elif account_type == "apiKey":
        auth = "api_key"
    elif account_type is None:
        auth = "missing"
    else:
        auth = "unverified"
    return True, True, None, auth
