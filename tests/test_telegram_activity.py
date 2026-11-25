from __future__ import annotations
from pathlib import Path
from unittest.mock import patch
import asyncio
import tempfile
import unittest
from telecodex.codex.session import CodexMemorySettings, CodexBackgroundTerminal, CodexGoal, CodexSkill
from telecodex.config import AppConfig
from telecodex.telegram.activity import explored_action_summaries, file_change_groups, image_saved_path, web_search_detail
from telecodex.telegram.formatting import TELEGRAM_TEXT_LIMIT, split_plain_text
from telecodex.telegram.updates import IncomingCallback, IncomingMessage, parse_update
from telecodex.telegram.user_controller import RunningCommand, TelegramUserController

from multi_user_test_support import FakeCodex, FakeTelegram, _catalog_text, _plain_html

class CodexActivityFormattingTests(unittest.TestCase):
    def test_plain_text_chunks_respect_limit_without_markdown_fences(self) -> None:
        source = ("word " * 80).strip()
        chunks = split_plain_text(source, limit=64)

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 64 for chunk in chunks))
        self.assertEqual(" ".join(chunks), source)

    def test_file_change_groups_include_added_edited_and_deleted_line_counts(self) -> None:
        groups = file_change_groups(
            {
                "changes": [
                    {"path": "src/new.py", "kind": {"type": "add"}, "diff": "+one\n+two"},
                    {
                        "path": "src/changed.py",
                        "kind": {"type": "update"},
                        "diff": "--- old\n+++ new\n-old\n+new\n+extra",
                    },
                    {"path": "src/old.py", "kind": {"type": "delete"}, "diff": "-one\n-two"},
                ]
            }
        )

        self.assertEqual(groups["add"], ["src/new.py (+2 -0)"])
        self.assertEqual(groups["update"], ["src/changed.py (+2 -1)"])
        self.assertEqual(groups["delete"], ["src/old.py (+0 -2)"])

    def test_explored_actions_report_read_paths_and_search_targets(self) -> None:
        actions = [
            {"type": "read", "path": "src/main.py", "name": "read_file", "command": "cat"},
            {"type": "listFiles", "path": "tests", "command": "ls"},
            {"type": "search", "path": "src", "query": "turn/start", "command": "rg"},
        ]

        self.assertEqual(
            explored_action_summaries(actions, render=lambda key, **values: _catalog_text(key, **values)),
            [
                "src/main.py",
                _catalog_text("detailed_log.values.explored_action.files_in", target="tests"),
                _catalog_text("detailed_log.values.explored_action.search_in", query="turn/start", target="src"),
            ],
        )

    def test_web_search_detail_uses_action_specific_query_or_url(self) -> None:
        self.assertEqual(
            web_search_detail(
                {"query": "fallback", "action": {"type": "search", "queries": ["one", "two"]}}
            ),
            "one; two",
        )
        self.assertEqual(
            web_search_detail({"query": "open", "action": {"type": "openPage", "url": "https://example.com"}}),
            "https://example.com",
        )
        self.assertEqual(
            web_search_detail(
                {"query": "find", "action": {"type": "findInPage", "pattern": "release", "url": "https://example.com"}}
            ),
            "release — https://example.com",
        )

    def test_image_saved_path_is_relative_and_hides_unrelated_host_paths(self) -> None:
        self.assertEqual(
            image_saved_path(
                {"savedPath": "/workspace/project/assets/generated.png"},
                "/workspace/project",
            ),
            "assets/generated.png",
        )
        self.assertEqual(
            image_saved_path({"savedPath": "/tmp/generated.png"}, "/workspace/project"),
            "generated.png",
        )
        self.assertEqual(
            image_saved_path({"savedPath": "../../private/generated.png"}, "/workspace/project"),
            "generated.png",
        )


class CodexActivityHandlerTests(unittest.IsolatedAsyncioTestCase):
    def _controller(self, workspace: Path) -> tuple[TelegramUserController, FakeTelegram]:
        config = AppConfig(
            telegram_bot_token="token",
            telegram_allowed_user_ids=(101,),
            workspace_root=workspace,
            codex_bin="codex",
            codex_model=None,
            network_access=False,
            approval_policy="on-request",
            shell_timeout_seconds=60,
            database_file=workspace / "state.sqlite3",
        )
        telegram = FakeTelegram()
        controller = TelegramUserController(config, telegram, object(), 101)  # type: ignore[arg-type]
        controller.bind_codex(FakeCodex())  # type: ignore[arg-type]
        return controller, telegram

    async def test_recap_uses_a_normal_turn_without_compacting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/recap")
            )

            self.assertEqual(len(codex.messages), 1)
            self.assertIn("without compacting", codex.messages[0])
            self.assertFalse(codex.compacted)
            self.assertEqual(telegram.messages, [])
            await controller._finish_typing()

    async def test_recap_requires_thread_and_steers_an_active_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/recap")
            )
            self.assertEqual(codex.messages, [])
            self.assertIn("Нет активного диалога", telegram.messages[-1][1])

            codex.state.thread_id = "active-thread"
            codex.is_busy = True
            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", "/recap")
            )
            self.assertEqual(len(codex.messages), 1)
            self.assertIn("Summarize this Codex conversation", codex.messages[0])
            self.assertFalse(
                any("Дождись завершения" in text for _, text in telegram.messages)
            )
            await controller._finish_typing()

    async def test_plan_command_toggles_the_active_thread_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/plan")
            )
            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", "/plan")
            )

            self.assertEqual(codex.collaboration_mode_calls, [("plan", False), ("default", False)])
            self.assertIn("режим планирования", telegram.messages[0][1].lower())
            self.assertIn("обычный режим", telegram.messages[1][1].lower())

    async def test_plan_inline_task_starts_and_busy_plan_task_steers_active_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            task = "Спланируй обновление интерфейса"

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", f"/plan {task}")
            )

            self.assertEqual(codex.collaboration_mode_calls, [("plan", True)])
            self.assertEqual(codex.messages, [task])

            codex.is_busy = True
            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", "/plan another task")
            )
            self.assertEqual(codex.messages, [task, "another task"])
            self.assertEqual(
                codex.collaboration_mode_calls,
                [("plan", True), ("plan", True)],
            )
            self.assertFalse(
                any("Дождись завершения" in text for _, text in telegram.messages)
            )
            await controller._finish_typing()

    async def test_plan_and_goal_arguments_keep_telegram_formatting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, _telegram = self._controller(Path(temp_dir))
            codex = controller._codex()

            plan = parse_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 77,
                        "from": {"id": 101},
                        "chat": {"id": 101, "type": "private"},
                        "text": "/plan Write a plan",
                        "entities": [{"type": "bold", "offset": 6, "length": 5}],
                    },
                }
            )
            goal = parse_update(
                {
                    "update_id": 2,
                    "message": {
                        "message_id": 78,
                        "from": {"id": 101},
                        "chat": {"id": 101, "type": "private"},
                        "text": "/goal Finish the task",
                        "entities": [{"type": "italic", "offset": 6, "length": 6}],
                    },
                }
            )

            self.assertIsInstance(plan, IncomingMessage)
            self.assertIsInstance(goal, IncomingMessage)
            await controller._handle_message(plan)  # type: ignore[arg-type]
            await controller._handle_message(goal)  # type: ignore[arg-type]

            self.assertEqual(codex.messages, ["**Write** a plan"])
            self.assertEqual(codex.goal.objective, "*Finish* the task")  # type: ignore[union-attr]

            await controller._handle_message(
                IncomingMessage(3, 79, 101, 101, "private", "/goal edit")
            )
            edited_goal = parse_update(
                {
                    "update_id": 4,
                    "message": {
                        "message_id": 80,
                        "from": {"id": 101},
                        "chat": {"id": 101, "type": "private"},
                        "text": "Revise objective",
                        "entities": [{"type": "bold", "offset": 0, "length": 6}],
                    },
                }
            )
            self.assertIsInstance(edited_goal, IncomingMessage)
            await controller._handle_message(edited_goal)  # type: ignore[arg-type]
            self.assertEqual(codex.goal.objective, "**Revise** objective")  # type: ignore[union-attr]
            await controller._finish_typing()

    async def test_messages_steer_active_turn_and_queue_after_compaction_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"

            codex.is_busy = True
            await controller._send_to_codex(101, "steer this turn", 77)
            self.assertEqual(codex.messages, ["steer this turn"])

            codex.compaction_barrier_pending = True
            codex.compaction_barrier_event.clear()
            await controller._send_to_codex(101, "after compact one", 78)
            await controller._send_to_codex(101, "after compact two", 79)

            self.assertEqual(codex.messages, ["steer this turn"])
            self.assertIn("позиция 1", telegram.messages[-2][1])
            self.assertIn("позиция 2", telegram.messages[-1][1])

            codex.is_busy = False
            codex.compaction_barrier_pending = False
            codex.compaction_barrier_event.set()
            task = controller._compaction_input_queue_task
            self.assertIsNotNone(task)
            await task  # type: ignore[arg-type]

            self.assertEqual(
                codex.messages,
                ["steer this turn", "after compact one", "after compact two"],
            )
            self.assertEqual(len(controller._compaction_input_queue), 0)

    async def test_compact_command_is_accepted_while_a_turn_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.is_busy = True

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/compact")
            )

            self.assertTrue(codex.compacted)
            self.assertEqual(telegram.messages, [])

    async def test_compact_queue_status_is_sent_when_the_barrier_reports_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "waiting-thread"
            codex.compaction_barrier_pending = True

            await controller._on_codex_status("status.compaction_queued")

            self.assertEqual(len(telegram.messages), 1)
            self.assertIn("сжатие поставлено в очередь", telegram.messages[0][1].lower())

    async def test_stop_cancels_compaction_barrier_and_discards_queued_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.compaction_barrier_pending = True
            codex.compaction_barrier_event.clear()
            controller._start_typing()

            await controller._send_to_codex(101, "queued prompt", 78)
            self.assertEqual(len(controller._compaction_input_queue), 1)

            await controller._stop_codex(IncomingMessage(1, 79, 101, 101, "private", "/stop"))

            self.assertFalse(codex.compaction_barrier_pending)
            self.assertEqual(len(controller._compaction_input_queue), 0)
            self.assertIn("удалено сообщений — 1", telegram.messages[-1][1].lower())
            self.assertFalse(controller._typing_enabled)
            task = controller._compaction_input_queue_task
            if task is not None:
                await task

    async def test_goal_command_reads_sets_pauses_resumes_and_clears_native_goal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            objective = "Исправь обработку повторных Telegram updates и проверь тесты"

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", f"/goal {objective}")
            )
            self.assertIsNotNone(codex.goal)
            self.assertEqual(codex.goal.objective, objective)
            self.assertEqual(codex.goal.status, "active")
            self.assertIn("Цель установлена", _plain_html(telegram.messages[-1][1]))

            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", "/goal")
            )
            self.assertIn(objective, telegram.messages[-1][1])
            self.assertIn("выполняется", telegram.messages[-1][1])

            codex.is_busy = True
            await controller._handle_message(
                IncomingMessage(3, 79, 101, 101, "private", "/goal pause")
            )
            self.assertEqual(codex.goal.status, "paused")
            self.assertIn(
                "цель поставлена на паузу",
                _plain_html(telegram.messages[-1][1]).lower(),
            )

            await controller._handle_message(
                IncomingMessage(4, 80, 101, 101, "private", "/goal resume")
            )
            self.assertEqual(codex.goal.status, "active")
            self.assertIn("Цель возобновлена", telegram.messages[-1][1])

            await controller._handle_message(
                IncomingMessage(5, 81, 101, 101, "private", "/goal clear")
            )
            self.assertIsNone(codex.goal)
            self.assertIn("Цель очищена", telegram.messages[-1][1])
            await controller._finish_typing()

    async def test_goal_objective_starts_a_chat_when_none_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/goal Подготовь проект и проверь сборку")
            )

            self.assertEqual(codex.state.thread_id, "thread-goal")
            self.assertIsNotNone(codex.goal)
            self.assertIn("Цель установлена", _plain_html(telegram.messages[-1][1]))

    async def test_goal_edit_collects_a_replacement_and_supports_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.goal = CodexGoal("Old objective", "paused", None, 20, 4)

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/goal edit")
            )
            self.assertIn("новым текстом цели", telegram.messages[-1][1].lower())
            self.assertIsNotNone(controller.pending_goal_edit)

            new_objective = "Finish the new objective"
            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", new_objective)
            )
            self.assertEqual(codex.goal.objective, new_objective)
            self.assertEqual(codex.goal.status, "active")
            self.assertIsNone(controller.pending_goal_edit)

            await controller._handle_message(
                IncomingMessage(3, 79, 101, 101, "private", "/goal edit")
            )
            await controller._handle_message(
                IncomingMessage(4, 80, 101, 101, "private", "/cancel")
            )
            self.assertIsNone(controller.pending_goal_edit)
            self.assertEqual(codex.goal.objective, new_objective)
            self.assertIn("изменение цели отменено", telegram.messages[-1][1].lower())
            await controller._finish_typing()

    async def test_fast_command_only_toggles_without_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()

            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/fast"))
            self.assertTrue(codex.fast_enabled)
            self.assertIn("Быстрый режим включён", _plain_html(telegram.messages[-1][1]))

            await controller._handle_message(IncomingMessage(2, 78, 101, 101, "private", "/fast off"))
            self.assertTrue(codex.fast_enabled)
            self.assertIn("не принимает аргументы", telegram.messages[-1][1])

            await controller._handle_message(IncomingMessage(3, 79, 101, 101, "private", "/fast"))
            self.assertFalse(codex.fast_enabled)
            self.assertIn("стандартная скорость", telegram.messages[-1][1])

    async def test_skills_command_selects_skill_for_the_next_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.skills = [
                CodexSkill("reviewer", "Review changes", "/tmp/skills/reviewer/SKILL.md", True, "Reviewer")
            ]

            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/skills"))
            picker = controller.skill_picker
            self.assertIsNotNone(picker)
            assert picker is not None

            await controller._handle_callback(
                IncomingCallback(2, "callback-use", 101, 101, picker.message_id or 0,
                                f"sk:{picker.token}:a:use")
            )
            await controller._handle_callback(
                IncomingCallback(3, "callback-skill", 101, 101, picker.message_id or 0,
                                f"sk:{picker.token}:use:0")
            )

            self.assertEqual(codex.state.pending_skill_name, "reviewer")
            self.assertEqual(codex.state.pending_skill_path, "/tmp/skills/reviewer/SKILL.md")
            self.assertIn("Reviewer", telegram.edits[-1][2])

    async def test_skills_command_can_disable_a_codex_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.skills = [CodexSkill("reviewer", "Review changes", "/skills/reviewer/SKILL.md", True)]

            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/skills"))
            picker = controller.skill_picker
            self.assertIsNotNone(picker)
            assert picker is not None
            await controller._handle_callback(
                IncomingCallback(2, "callback-manage", 101, 101, picker.message_id or 0,
                                f"sk:{picker.token}:a:manage")
            )
            await controller._handle_callback(
                IncomingCallback(3, "callback-disable", 101, 101, picker.message_id or 0,
                                f"sk:{picker.token}:toggle:0")
            )

            self.assertEqual(codex.skill_enabled_calls, [("reviewer", False)])
            self.assertFalse(codex.skills[0].enabled)
            self.assertIn("общей", telegram.edits[-1][2].lower())

    async def test_memories_command_scopes_use_to_thread_and_confirms_global_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()

            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/memories"))
            picker = controller.memory_picker
            self.assertIsNotNone(picker)
            assert picker is not None

            await controller._handle_callback(
                IncomingCallback(2, "callback-use-memory", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:use:enabled")
            )
            self.assertEqual(codex.state.selected_memory_mode, "enabled")
            memory_markup = telegram.edit_markups[-1]
            memory_labels = [
                button["text"]
                for row in memory_markup["inline_keyboard"]
                for button in row
            ]
            self.assertEqual(
                memory_markup["inline_keyboard"][0:2],
                [
                    [{"text": "› Использовать в этом диалоге ‹", "callback_data": f"mm:{picker.token}:use:enabled"}],
                    [{"text": "Не использовать в этом диалоге", "callback_data": f"mm:{picker.token}:use:disabled"}],
                ],
            )
            self.assertTrue(
                all(len(row) == 1 for row in memory_markup["inline_keyboard"])
            )
            self.assertIn("› Использовать в этом диалоге ‹", memory_labels)

            await controller._handle_callback(
                IncomingCallback(3, "callback-generate", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:generate")
            )
            self.assertFalse(codex.memory_settings.generate_memories)
            await controller._handle_callback(
                IncomingCallback(4, "callback-confirm-generate", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:confirm")
            )
            self.assertTrue(codex.memory_settings.generate_memories)
            self.assertTrue(any("общую настройку" in text.lower() for _chat, _message_id, text in telegram.edits))

    async def test_memories_can_enable_the_codex_feature_for_future_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.memory_settings = CodexMemorySettings(False, False, False, "config-v1")

            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/memories"))
            picker = controller.memory_picker
            self.assertIsNotNone(picker)
            assert picker is not None
            await controller._handle_callback(
                IncomingCallback(2, "callback-enable", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:enable")
            )

            self.assertTrue(codex.memory_settings.feature_enabled)
            self.assertIn("новых диалогов", telegram.edits[-1][2].lower())

    async def test_memories_reset_requires_a_second_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            await controller._handle_message(IncomingMessage(1, 77, 101, 101, "private", "/memories"))
            picker = controller.memory_picker
            self.assertIsNotNone(picker)
            assert picker is not None

            await controller._handle_callback(
                IncomingCallback(2, "reset-start", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:reset")
            )
            self.assertFalse(codex.memories_reset)
            self.assertIn("отменить действие нельзя", _plain_html(telegram.edits[-1][2]).lower())

            await controller._handle_callback(
                IncomingCallback(3, "reset-confirm", 101, 101, picker.message_id or 0,
                                f"mm:{picker.token}:reset_confirm")
            )
            self.assertTrue(codex.memories_reset)

    async def test_init_prompts_codex_to_create_only_a_missing_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            controller, _telegram = self._controller(project)
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.working_directory = project

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/init")
            )

            self.assertEqual(len(codex.messages), 1)
            self.assertIn(str(project / "AGENTS.md"), codex.messages[0])
            self.assertIn("Create only this AGENTS.md", codex.messages[0])
            self.assertFalse((project / "AGENTS.md").exists())
            await controller._finish_typing()

    async def test_init_does_not_overwrite_an_existing_agents_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            agents_file = project / "AGENTS.md"
            agents_file.write_text("keep this file", encoding="utf-8")
            controller, telegram = self._controller(project)
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.working_directory = project

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/init")
            )

            self.assertEqual(codex.messages, [])
            self.assertIn("AGENTS.md", telegram.messages[-1][1])
            self.assertEqual(agents_file.read_text(encoding="utf-8"), "keep this file")
            await controller._finish_typing()

    async def test_init_refuses_a_symlink_target_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            outside_file = Path(temp_dir) / "outside.md"
            outside_file.write_text("keep this content", encoding="utf-8")
            (project / "AGENTS.md").symlink_to(outside_file)
            controller, telegram = self._controller(project)
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.working_directory = project

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/init")
            )

            self.assertEqual(codex.messages, [])
            self.assertIn("уже существует", telegram.messages[-1][1])
            self.assertEqual(outside_file.read_text(encoding="utf-8"), "keep this content")
            await controller._finish_typing()

    async def test_ps_lists_background_terminals_and_paginates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.background_terminals = [
                CodexBackgroundTerminal(
                    process_id=f"terminal-{index}",
                    command=f"python3 server_{index}.py",
                    cwd=temp_dir,
                    os_pid=2000 + index,
                    cpu_percent=1.25,
                    rss_kb=2048,
                )
                for index in range(11)
            ]

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/ps")
            )

            self.assertIn("python3 server_0.py", telegram.messages[-1][1])
            rows = telegram.reply_markups[-1]["inline_keyboard"]
            self.assertEqual(len(rows), 11)
            self.assertTrue(all(len(row) == 1 for row in rows[:10]))
            navigation = rows[-1]
            next_callback = navigation[2]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(2, "next", 101, 101, 1, next_callback)
            )
            self.assertIn("2/2", telegram.edits[-1][2])
            self.assertIn("server_10.py", telegram.edits[-1][2])
            self.assertNotIn("server_0.py", telegram.edits[-1][2])
            await controller._finish_typing()

    async def test_ps_stops_only_the_selected_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()
            codex.state.thread_id = "active-thread"
            codex.background_terminals = [
                CodexBackgroundTerminal(
                    process_id="terminal-1",
                    command="python3 -m http.server --password private-test-value 8080",
                    cwd=temp_dir,
                    os_pid=4242,
                    cpu_percent=0.5,
                    rss_kb=1024,
                )
            ]

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/ps")
            )
            view_callback = telegram.reply_markups[-1]["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(2, "view", 101, 101, 1, view_callback)
            )
            self.assertIn("PID 4242", telegram.edits[-1][2])
            self.assertIn("[REDACTED]", telegram.edits[-1][2])
            self.assertNotIn("private-test-value", telegram.edits[-1][2])
            stop_callback = telegram.edit_markups[-1]["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(3, "stop", 101, 101, 1, stop_callback)
            )

            self.assertEqual(codex.terminated_background_terminals, ["terminal-1"])
            self.assertEqual(codex.background_terminals, [])
            self.assertEqual(telegram.edits[-1][2], _catalog_text("ps.label.empty"))
            await controller._finish_typing()

    async def test_ps_requires_a_thread_and_rejects_stale_stop_buttons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            codex = controller._codex()

            await controller._handle_message(
                IncomingMessage(1, 77, 101, 101, "private", "/ps")
            )
            self.assertEqual(codex.background_terminal_list_calls, 0)
            self.assertIn("выбери диалог", telegram.messages[-1][1].lower())

            codex.state.thread_id = "thread-before-switch"
            codex.background_terminals = [
                CodexBackgroundTerminal(
                    process_id="terminal-stale",
                    command="sleep 60",
                    cwd=temp_dir,
                    os_pid=9090,
                    cpu_percent=0,
                    rss_kb=128,
                )
            ]
            await controller._handle_message(
                IncomingMessage(2, 78, 101, 101, "private", "/ps")
            )
            view_callback = telegram.reply_markups[-1]["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(3, "view", 101, 101, 2, view_callback)
            )
            stop_callback = telegram.edit_markups[-1]["inline_keyboard"][0][0]["callback_data"]
            codex.state.thread_id = "thread-after-switch"

            await controller._handle_callback(
                IncomingCallback(4, "stale-stop", 101, 101, 2, stop_callback)
            )

            self.assertEqual(codex.terminated_background_terminals, [])
            self.assertIn("устарел", telegram.callbacks[-1][1])
            await controller._finish_typing()

    async def test_assistant_commentary_is_sent_in_both_output_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for mode in ("answers", "detailed"):
                with self.subTest(mode=mode):
                    controller, telegram = self._controller(Path(temp_dir) / mode)
                    controller._codex().state.output_mode = mode
                    controller._turn_prompt_message_id = 55

                    await controller._on_codex_assistant_message(
                        "commentary-item",
                        "I will check the task now.",
                    )

                    self.assertEqual(
                        telegram.messages,
                        [(101, "I will check the task now.")],
                    )
                    self.assertEqual(
                        telegram.rich_messages,
                        [(101, "I will check the task now.", False)],
                    )
                    self.assertEqual(telegram.replies, [55])
            self.assertEqual(telegram.notification_flags, [False])

    async def test_quiet_answers_silences_progress_and_notifies_on_final_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            state = controller._codex().state
            state.output_mode = "answers"
            state.quiet_progress = True

            await controller._on_codex_assistant_message(
                "commentary-item",
                "Working through the task.",
            )
            await controller._on_codex_finished(
                {"status": "completed", "text": "Final answer. " * TELEGRAM_TEXT_LIMIT}
            )

            self.assertGreater(len(telegram.messages), 2)
            self.assertTrue(telegram.notification_flags[0])
            self.assertTrue(all(telegram.notification_flags[1:-1]))
            self.assertFalse(telegram.notification_flags[-1])

    async def test_quiet_answers_keeps_terminal_errors_audible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            state = controller._codex().state
            state.output_mode = "answers"
            state.quiet_progress = True

            await controller._on_codex_finished(
                {"status": "failed", "text": "Codex failed", "error": "request failed"}
            )

            self.assertEqual(len(telegram.messages), 1)
            self.assertEqual(telegram.notification_flags, [False])

    async def test_quiet_answers_silences_compaction_progress_but_not_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            state = controller._codex().state
            state.output_mode = "answers"
            state.quiet_progress = True

            await controller._on_codex_status("status.label.compaction_started")
            await controller._on_codex_status("status.label.compaction_failed")

            self.assertEqual(telegram.notification_flags, [True, False])

    async def test_review_result_is_delivered_as_final_text_in_answers_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._codex().state.output_mode = "answers"

            await controller._on_codex_finished(
                {
                    "status": "completed",
                    "text": "Review finding: non-owners can delete the project.",
                }
            )

            self.assertEqual(
                telegram.messages,
                [(101, "Review finding: non-owners can delete the project.")],
            )
            self.assertEqual(
                telegram.rich_messages,
                [(101, "Review finding: non-owners can delete the project.", False)],
            )

    async def test_long_assistant_commentary_is_split_at_telegram_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._turn_prompt_message_id = 77
            long_text = "Start.\n\n```python\n" + "print('progress')\n" * 700 + "```\nDone."

            await controller._on_codex_assistant_message("long-commentary", long_text)

            self.assertGreater(len(telegram.messages), 1)
            self.assertTrue(all(len(text) <= TELEGRAM_TEXT_LIMIT for _, text in telegram.messages))
            self.assertTrue(all(reply == 77 for reply in telegram.replies))
            self.assertTrue(telegram.messages[0][1].startswith("Start."))
            self.assertTrue(telegram.messages[-1][1].endswith("Done."))

    async def test_group_commentary_stays_ordered_and_separate_from_activity_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(
                telegram_bot_token="token",
                telegram_allowed_user_ids=(101,),
                workspace_root=Path(temp_dir),
                codex_bin="codex",
                codex_model=None,
                network_access=False,
                approval_policy="on-request",
                shell_timeout_seconds=60,
                database_file=Path(temp_dir) / "state.sqlite3",
            )
            telegram = FakeTelegram()
            controller = TelegramUserController(
                config, telegram, object(), -1001, is_group=True
            )  # type: ignore[arg-type]
            controller.bind_codex(FakeCodex())  # type: ignore[arg-type]
            controller._turn_prompt_message_id = 88

            with patch("telecodex.telegram.user_controller.GROUP_ACTIVITY_INTERVAL_SECONDS", 0):
                await controller._send_codex_activity_lines(
                    [("Ran", "pwd")],
                    reply_to_message_id=88,
                )
                await controller._on_codex_assistant_message(
                    "group-commentary",
                    "Found the requested directory.",
                )
                await controller._on_codex_finished(
                    {"status": "completed", "text": "Done."}
                )

            self.assertEqual(len(telegram.messages), 3)
            self.assertIn("• Ran pwd", telegram.messages[0][1])
            self.assertEqual(telegram.messages[1][1], "Found the requested directory.")
            self.assertIn("Done.", telegram.messages[2][1])
            self.assertEqual(telegram.replies, [88, 88, 88])
            self.assertEqual(
                telegram.rich_messages,
                [
                    (-1001, "Found the requested directory.", False),
                    (-1001, "Done.", False),
                ],
            )
            await controller._stop_group_activity()

    async def test_file_change_statuses_separate_added_edited_and_deleted_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))

            await controller._on_codex_status(
                "status.file_change",
                {
                    "status": "completed",
                    "changes": [
                        {"path": "added.py", "kind": {"type": "add"}, "diff": "+line"},
                        {"path": "edited.py", "kind": {"type": "update"}, "diff": "-old\n+new"},
                        {"path": "deleted.py", "kind": {"type": "delete"}, "diff": "-line"},
                    ],
                },
            )

            self.assertIn("• Added added.py (+1 -0)", telegram.messages[0][1])
            self.assertIn("• Edited edited.py (+1 -1)", telegram.messages[0][1])
            self.assertIn("• Deleted deleted.py (+0 -1)", telegram.messages[0][1])

    async def test_web_search_emits_started_and_completed_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            item = {"query": "default", "action": {"type": "search", "query": "Codex updates"}}

            await controller._on_codex_status("detailed_log.values.web_search_started", item)
            await controller._on_codex_status("detailed_log.values.web_search_completed", item)

            self.assertIn("• Browsing the web Codex updates", telegram.messages[0][1])
            self.assertIn("• Searched the web Codex updates", telegram.messages[1][1])

    async def test_web_search_start_without_query_still_emits_activity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))

            await controller._on_codex_status(
                "detailed_log.values.web_search_started",
                {"id": "web-1", "type": "webSearch", "query": "", "action": None},
            )

            self.assertEqual("• Browsing the web...", telegram.messages[0][1])

    async def test_answers_mode_shows_automatic_compaction_start_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._codex().state.output_mode = "answers"

            await controller._on_codex_status("status.label.compaction_started")
            await controller._on_codex_status("status.label.compaction_completed")

            self.assertEqual(
                [text for _, text in telegram.messages],
                [
                    "Codex сжимает контекст...",
                    "Контекст сжат. Codex продолжает задачу.",
                ],
            )

    async def test_typing_action_is_refreshed_during_long_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))

            with patch("telecodex.telegram.user_controller.TYPING_ACTION_INTERVAL_SECONDS", 0.01):
                controller._start_typing()
                await asyncio.sleep(0.04)
                await controller._finish_typing()

            self.assertGreaterEqual(len(telegram.actions), 2)

    async def test_skill_review_image_and_subagent_activity_obey_output_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for mode in ("detailed", "answers"):
                with self.subTest(mode=mode):
                    controller, telegram = self._controller(Path(temp_dir) / mode)
                    controller._codex().state.output_mode = mode
                    controller._codex().working_directory = Path(temp_dir) / mode

                    updates = [
                        ("detailed_log.values.skill_attached", "OpenAI Docs"),
                        ("detailed_log.values.review_started", None),
                        ("detailed_log.values.review_completed", None),
                        ("detailed_log.values.image_generation_started", None),
                        (
                            "detailed_log.values.image_generation_completed",
                            {"savedPath": str(Path(temp_dir) / mode / "assets/generated.png")},
                        ),
                        (
                            "detailed_log.values.image_generation_failed",
                            {"failureType": "usageLimitExceeded"},
                        ),
                        ("detailed_log.values.subagent_started", {"agentPath": "root/researcher"}),
                        ("detailed_log.values.subagent_completed", {"agentPath": "root/researcher"}),
                        ("detailed_log.values.subagent_interrupted", None),
                        ("detailed_log.values.subagent_failed", None),
                    ]
                    for key, details in updates:
                        await controller._on_codex_status(key, details)

                    if mode == "detailed":
                        rendered = [text for _, text in telegram.messages]
                        self.assertEqual(
                            rendered,
                            [
                                "• Skill attached: OpenAI Docs",
                                "• Code review started",
                                "• Code review finished",
                                "• Generating image",
                                "• Image generated assets/generated.png",
                                "• Image generation failed usage limit exceeded",
                                "• Subagent started root/researcher",
                                "• Subagent completed root/researcher",
                                "• Subagent interrupted",
                                "• Subagent failed",
                            ],
                        )
                        self.assertEqual(telegram.message_entities[0][0]["type"], "bold")
                        self.assertEqual(telegram.message_entities[0][0]["offset"], 2)
                        self.assertEqual(
                            telegram.message_entities[0][0]["length"],
                            len("Skill attached:"),
                        )
                    else:
                        self.assertEqual(telegram.messages, [])

    async def test_completed_command_reports_explored_file_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._command_messages["command-1"] = RunningCommand(
                message_id=55,
                started_at=0,
                command="cat src/main.py",
            )

            await controller._on_codex_command_completed(
                {
                    "id": "command-1",
                    "command": "cat src/main.py",
                    "status": "completed",
                    "exitCode": 0,
                    "durationMs": 50,
                    "aggregatedOutput": "source",
                    "commandActions": [
                        {"type": "read", "path": "src/main.py", "name": "read_file", "command": "cat"}
                    ],
                }
            )

            self.assertEqual(len(telegram.messages), 2)
            expected_header = _catalog_text(
                "detailed_log.label.command_result_header",
                state=_catalog_text("detailed_log.values.command_succeeded"),
                duration=_catalog_text("detailed_log.values.duration_seconds", seconds="0.1"),
                exit_code="0",
            )
            self.assertIn(expected_header, telegram.messages[0][1])
            self.assertIn("• Explored src/main.py", telegram.messages[1][1])
            self.assertEqual(telegram.replies[1], 55)

    async def test_answers_mode_hides_codex_activity_but_keeps_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._codex().state.output_mode = "answers"

            await controller._on_codex_status(
                "detailed_log.label.command_running",
                "pwd",
                item_id="command-1",
            )
            await controller._on_codex_status(
                "detailed_log.values.web_search_started",
                {"query": "Codex"},
            )
            await controller._on_codex_status(
                "status.file_change",
                {"status": "completed", "changes": [{"path": "main.py", "kind": {"type": "add"}}]},
            )
            self.assertIsNone(controller._command_messages["command-1"].message_id)

            await controller._on_codex_command_completed(
                {
                    "id": "command-1",
                    "command": "pwd",
                    "status": "completed",
                    "exitCode": 0,
                    "durationMs": 1,
                    "aggregatedOutput": "/project",
                }
            )
            await controller._on_codex_finished({"status": "completed", "text": "Done."})

            self.assertNotIn("command-1", controller._command_messages)
            self.assertEqual([text for _, text in telegram.messages], ["Done."])

    async def test_mode_picker_updates_same_message_and_persists_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))

            await controller._handle_message(
                IncomingMessage(1, 1, 101, 101, "private", "/mode")
            )

            self.assertEqual(len(telegram.messages), 1)
            markup = telegram.reply_markups[0]
            self.assertEqual([len(row) for row in markup["inline_keyboard"]], [1, 1, 1])
            callback_data = markup["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(2, "callback-1", 101, 101, 1, callback_data)
            )

            self.assertEqual(controller._codex().state.output_mode, "answers")
            self.assertEqual(len(telegram.messages), 1)
            self.assertEqual(telegram.edits[0][1], 1)
            self.assertIn(_catalog_text("mode.buttons.names.answers"), telegram.edits[0][2])
            self.assertEqual(
                telegram.edit_markups[-1]["inline_keyboard"][0][0]["text"],
                "› Только ответы ‹",
            )

    async def test_mode_picker_selects_quiet_answers_without_changing_content_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            await controller._handle_message(
                IncomingMessage(1, 1, 101, 101, "private", "/mode")
            )
            callback_data = telegram.reply_markups[0]["inline_keyboard"][1][0]["callback_data"]

            await controller._handle_callback(
                IncomingCallback(2, "quiet-callback", 101, 101, 1, callback_data)
            )

            state = controller._codex().state
            self.assertEqual(state.output_mode, "answers")
            self.assertTrue(state.quiet_progress)
            self.assertIn(_catalog_text("mode.buttons.names.quiet_answers"), telegram.edits[-1][2])

    async def test_quiet_answers_keeps_codex_question_audible_and_hides_delivery_ack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._codex().state.output_mode = "answers"
            controller._codex().state.quiet_progress = True
            task = asyncio.create_task(
                controller._ask_user_input(
                    {"questions": [{"id": "q1", "question": "Which format?"}]}
                )
            )
            await asyncio.sleep(0)

            self.assertEqual(len(telegram.messages), 1)
            self.assertIn("Which format?", telegram.messages[0][1])
            await controller._resolve_user_input("JSON")

            self.assertEqual(await task, {"answers": {"q1": {"answers": ["JSON"]}}})
            self.assertEqual(len(telegram.messages), 1)
            self.assertEqual(telegram.notification_flags, [False])

    async def test_quiet_answers_keeps_permission_confirmation_audible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller, telegram = self._controller(Path(temp_dir))
            controller._codex().state.output_mode = "answers"
            controller._codex().state.quiet_progress = True
            task = asyncio.create_task(
                controller._ask_approval(
                    "item/commandExecution/requestApproval",
                    {"command": "npm test"},
                )
            )
            await asyncio.sleep(0)

            self.assertEqual(len(telegram.messages), 1)
            markup = telegram.reply_markups[0]
            self.assertEqual([len(row) for row in markup["inline_keyboard"]], [1, 1, 1])
            callback_data = markup["inline_keyboard"][0][0]["callback_data"]
            await controller._handle_callback(
                IncomingCallback(1, "approval-callback", 101, 101, 1, callback_data)
            )

            self.assertEqual(await task, "accept")
            self.assertIn("npm test", telegram.messages[0][1])
            self.assertEqual(telegram.notification_flags, [False])
            self.assertEqual(telegram.edits, [])
            self.assertEqual(
                telegram.reply_markup_edits,
                [(101, 1, {"inline_keyboard": []})],
            )
