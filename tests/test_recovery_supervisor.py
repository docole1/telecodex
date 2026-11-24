import asyncio
import os
import socket
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from telecodex.recovery_supervisor import RecoverySupervisor
from telecodex.systemd import SystemdNotifier


class FakeCodex:
    def __init__(self, outcome: str, *, transport_running: bool = False) -> None:
        self.pending_turn = SimpleNamespace(
            phase="running",
            thread_id="thread-test",
            turn_id="turn-test",
        )
        self.transport_running = transport_running
        self.is_busy = True
        self.seconds_since_last_event = 120.0
        self.last_event_method = None
        self.has_active_command_execution = False
        self.outcome = outcome
        self.calls = 0

    async def reconcile_pending_turn(self) -> str:
        self.calls += 1
        if self.outcome in {"delivered", "none"}:
            self.pending_turn = None
        return self.outcome

    def pending_turn_age_seconds(self) -> float:
        return 600.0


class FakeContext:
    def __init__(self, chat_id: int, codex: FakeCodex) -> None:
        self.chat_id = chat_id
        self._codex_instance = codex
        self.notices: list[str] = []
        self.switch_offers = 0
        self.pending_approvals: dict[str, object] = {}
        self.pending_user_input = None
        self.pending_mcp_elicitations: dict[str, object] = {}

    def _codex(self) -> FakeCodex:
        return self._codex_instance

    async def send_recovery_notice(self, key: str) -> None:
        self.notices.append(key)

    async def send_recovery_switch_offer(self) -> None:
        self.switch_offers += 1


class FakeController:
    def __init__(self, contexts: list[FakeContext]) -> None:
        self.contexts = contexts

    def active_context_controllers(self) -> tuple[FakeContext, ...]:
        return tuple(self.contexts)


class RecoverySupervisorTest(unittest.IsolatedAsyncioTestCase):
    async def test_closed_transport_reports_and_recovers_active_turn_once(self) -> None:
        codex = FakeCodex("active", transport_running=False)
        context = FakeContext(-1001, codex)
        heartbeat_count = 0

        def heartbeat() -> None:
            nonlocal heartbeat_count
            heartbeat_count += 1

        supervisor = RecoverySupervisor(FakeController([context]), heartbeat)
        await supervisor._check_once()

        self.assertEqual(codex.calls, 1)
        self.assertEqual(context.notices, ["recovery.label.checking", "recovery.label.running"])
        self.assertEqual(heartbeat_count, 0)

    async def test_retry_backoff_prevents_a_second_immediate_probe(self) -> None:
        codex = FakeCodex("retry", transport_running=False)
        context = FakeContext(-1002, codex)
        supervisor = RecoverySupervisor(FakeController([context]), lambda: None)

        await supervisor._check_once()
        await supervisor._check_once()

        self.assertEqual(codex.calls, 1)
        self.assertEqual(context.notices, ["recovery.label.checking"])

    async def test_pending_approval_suppresses_stale_turn_notices(self) -> None:
        codex = FakeCodex("active", transport_running=False)
        context = FakeContext(-1004, codex)
        context.pending_approvals = {"approval": object()}
        supervisor = RecoverySupervisor(FakeController([context]), lambda: None)

        await supervisor._check_once()
        supervisor._states[-1004].next_attempt_at = 0
        await supervisor._check_once()

        self.assertEqual(codex.calls, 2)
        self.assertEqual(context.notices, [])
        self.assertEqual(context.switch_offers, 0)

    async def test_quiet_active_command_never_gets_an_automatic_switch_offer(self) -> None:
        codex = FakeCodex("active", transport_running=True)
        codex.seconds_since_last_event = 600
        codex.has_active_command_execution = True
        context = FakeContext(-1003, codex)
        supervisor = RecoverySupervisor(FakeController([context]), lambda: None)

        await supervisor._check_once()

        self.assertEqual(context.switch_offers, 0)
        self.assertEqual(codex.calls, 1)
        self.assertTrue(codex.is_busy)

    async def test_quiet_active_turn_without_command_gets_one_manual_switch_offer(self) -> None:
        codex = FakeCodex("active", transport_running=True)
        codex.seconds_since_last_event = 600
        context = FakeContext(-1004, codex)
        supervisor = RecoverySupervisor(FakeController([context]), lambda: None)

        await supervisor._check_once()
        await supervisor._check_once()

        self.assertEqual(context.switch_offers, 1)
        self.assertEqual(codex.calls, 1)
        self.assertTrue(codex.is_busy)

    async def test_new_pending_turn_gets_a_fresh_quiet_turn_offer(self) -> None:
        codex = FakeCodex("active", transport_running=True)
        codex.seconds_since_last_event = 600
        context = FakeContext(-1005, codex)
        supervisor = RecoverySupervisor(FakeController([context]), lambda: None)

        await supervisor._check_once()
        codex.pending_turn = SimpleNamespace(
            phase="running",
            thread_id="thread-test",
            turn_id="turn-next",
        )
        await supervisor._check_once()

        self.assertEqual(context.switch_offers, 2)


class SystemdNotifierTest(unittest.TestCase):
    def test_notifier_sends_datagram_when_socket_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "notify.sock")
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
                receiver.bind(path)
                receiver.settimeout(1)
                with patch.dict(os.environ, {"NOTIFY_SOCKET": path}, clear=False):
                    notifier = SystemdNotifier()
                    notifier.ready("ready")
                self.assertEqual(receiver.recv(1024), b"READY=1\nSTATUS=ready")

    def test_notifier_is_noop_outside_systemd(self) -> None:
        with patch.dict(os.environ, {"NOTIFY_SOCKET": ""}, clear=False):
            notifier = SystemdNotifier()
            notifier.ready("ignored")
        self.assertFalse(notifier.enabled)
