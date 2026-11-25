"""Model, skill, memory, and collaboration settings for a Codex context."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from telecodex.config import AppConfig
from telecodex.codex.rpc import RpcError
from telecodex.codex.types import (
    CodexMemorySettings,
    CodexModel,
    CodexServiceTier,
    CodexSkill,
    CollaborationModeUnavailableError,
    FastTierUnavailableError,
    MemoryFeatureUnavailableError,
    SelectedSkillUnavailableError,
)
from telecodex.logging_config import log_exception
from telecodex.state.models import RuntimeState

logger = logging.getLogger(__name__)


class CodexSettings:
    """Own Codex settings APIs and transient settings for the selected thread."""

    def __init__(
        self,
        *,
        config: AppConfig,
        app_server: Any,
        state: RuntimeState,
        state_store: Any,
        ensure_app_server: Callable[[], Awaitable[None]],
        ensure_thread: Callable[..., Awaitable[None]],
        unsubscribe_thread: Callable[..., Awaitable[None]],
        is_busy: Callable[[], bool],
        thread_log_fields: Callable[..., dict[str, str | None]],
        working_directory: Callable[[], Path],
        sync_persisted_thread_selection: Callable[[], None],
        is_config_version_conflict: Callable[[RpcError], bool],
    ) -> None:
        self.config = config
        self.app_server = app_server
        self.state = state
        self.state_store = state_store
        self._ensure_app_server = ensure_app_server
        self._ensure_thread = ensure_thread
        self._unsubscribe_thread = unsubscribe_thread
        self._is_busy = is_busy
        self._thread_log_fields = thread_log_fields
        self._get_working_directory = working_directory
        self._sync_persisted_thread_selection = sync_persisted_thread_selection
        self._is_config_version_conflict = is_config_version_conflict
        self._collaboration_mode = "default"
        self._thread_model: str | None = None
        self._thread_effort: str | None = None
        self._thread_service_tier: str | None = None
        self._collaboration_mode_presets: dict[str, dict[str, Any]] | None = None

    @property
    def is_busy(self) -> bool:
        return self._is_busy()

    @property
    def working_directory(self) -> Path:
        return self._get_working_directory()

    def snapshot_thread_settings(self) -> tuple[str, str | None, str | None, str | None]:
        return (
            self._collaboration_mode,
            self._thread_model,
            self._thread_effort,
            self._thread_service_tier,
        )

    def restore_thread_settings(
        self,
        snapshot: tuple[str, str | None, str | None, str | None],
    ) -> None:
        (
            self._collaboration_mode,
            self._thread_model,
            self._thread_effort,
            self._thread_service_tier,
        ) = snapshot

    def reset_thread_settings(self, *, clear_presets: bool = False) -> None:
        self._collaboration_mode = "default"
        self._thread_model = None
        self._thread_effort = None
        self._thread_service_tier = None
        if clear_presets:
            self._collaboration_mode_presets = None

    def reset_collaboration_mode(self) -> None:
        self._collaboration_mode = "default"

    async def list_models(self) -> list[CodexModel]:
        await self._ensure_app_server()
        models: list[CodexModel] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            params: dict[str, Any] = {"includeHidden": False, "limit": 100}
            if cursor is not None:
                params["cursor"] = cursor
            result = await self.app_server.request("model/list", params, timeout_seconds=60)
            data = result.get("data", [])
            if not isinstance(data, list):
                raise RuntimeError("Codex app-server returned an invalid model list")
            for raw_model in data:
                if not isinstance(raw_model, dict) or raw_model.get("hidden") is True:
                    continue
                model = raw_model.get("model")
                display_name = raw_model.get("displayName")
                if not isinstance(model, str) or not model:
                    continue
                if not isinstance(display_name, str) or not display_name:
                    display_name = model
                raw_efforts = raw_model.get("supportedReasoningEfforts", [])
                efforts: list[str] = []
                if isinstance(raw_efforts, list):
                    for option in raw_efforts:
                        effort = option.get("reasoningEffort") if isinstance(option, dict) else None
                        if isinstance(effort, str) and effort and effort not in efforts:
                            efforts.append(effort)
                default_effort = raw_model.get("defaultReasoningEffort")
                service_tiers: list[CodexServiceTier] = []
                raw_service_tiers = raw_model.get("serviceTiers", [])
                if isinstance(raw_service_tiers, list):
                    for tier in raw_service_tiers:
                        if not isinstance(tier, dict):
                            continue
                        tier_id = tier.get("id")
                        tier_name = tier.get("name")
                        if not isinstance(tier_id, str) or not tier_id:
                            continue
                        if not isinstance(tier_name, str) or not tier_name:
                            tier_name = tier_id
                        description = tier.get("description")
                        service_tiers.append(
                            CodexServiceTier(
                                id=tier_id,
                                name=tier_name,
                                description=description if isinstance(description, str) else "",
                            )
                        )
                default_service_tier = raw_model.get("defaultServiceTier")
                models.append(
                    CodexModel(
                        model=model,
                        display_name=display_name,
                        is_default=raw_model.get("isDefault") is True,
                        supported_reasoning_efforts=tuple(efforts),
                        default_reasoning_effort=(
                            default_effort if isinstance(default_effort, str) and default_effort else None
                        ),
                        service_tiers=tuple(service_tiers),
                        default_service_tier=(
                            default_service_tier
                            if isinstance(default_service_tier, str) and default_service_tier
                            else None
                        ),
                    )
                )
            next_cursor = result.get("nextCursor")
            if next_cursor is None:
                return models
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
                raise RuntimeError("Codex app-server returned an invalid model-list cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor


    async def current_model_settings(
        self,
        models: list[CodexModel],
    ) -> tuple[str | None, str | None]:
        """Return the effective model and effort for the selected thread or user."""
        self._sync_persisted_thread_selection()
        await self._ensure_app_server()

        thread_id = self.state.thread_id
        if thread_id:
            try:
                response = await self.app_server.request(
                    "thread/read",
                    {"threadId": thread_id},
                    timeout_seconds=30,
                )
            except Exception as exc:
                log_exception(
                    logger,
                    "Could not refresh model settings for the selected Codex thread",
                    exc,
                    event="codex.model_picker.thread_read_failed",
                    level=logging.WARNING,
                    **self._thread_log_fields(thread_id),
                )
            else:
                self._capture_thread_settings(response)
            model = self._thread_model or self.state.selected_model or self.config.codex_model
            effort = self._thread_effort or self.state.selected_effort
        else:
            model = self.state.selected_model or self.config.codex_model
            effort = self.state.selected_effort

        if not model:
            model = self.default_model(models)
        return model, effort


    @property
    def selected_model(self) -> str | None:
        return self.state.selected_model


    @property
    def selected_effort(self) -> str | None:
        return self.state.selected_effort

    @property
    def selected_memory_mode(self) -> str | None:
        return self.state.selected_memory_mode


    @property
    def selected_service_tier(self) -> str | None:
        return self._thread_service_tier if self.state.thread_id else self.state.selected_service_tier

    @property
    def pending_skill_name(self) -> str | None:
        return self.state.pending_skill_name

    @property
    def pending_skill_path(self) -> str | None:
        return self.state.pending_skill_path


    async def toggle_fast_mode(self) -> bool:
        """Toggle the catalog-provided Fast tier for this thread/context."""
        models = await self.list_models()
        await self._ensure_app_server()
        if self.state.thread_id:
            await self._ensure_thread(create_if_missing=False)
            model_name = self._thread_model or self.state.selected_model or self.config.codex_model
        else:
            model_name = self.state.selected_model or self.config.codex_model or self.default_model(models)
        model = next((item for item in models if item.model == model_name), None)
        if model is None:
            raise RuntimeError("Codex could not resolve the active model")
        fast_tier = next(
            (tier for tier in model.service_tiers if tier.name.casefold() == "fast"),
            None,
        )
        if fast_tier is None:
            raise FastTierUnavailableError("The active Codex model does not advertise Fast mode")

        current_tier = self.selected_service_tier or model.default_service_tier
        current_fast = current_tier == fast_tier.id
        target_enabled = not current_fast
        target_tier = fast_tier.id if target_enabled else "default"
        thread_id = self.state.thread_id
        try:
            if thread_id:
                await self.app_server.request(
                    "thread/settings/update",
                    {"threadId": thread_id, "serviceTier": target_tier},
                    timeout_seconds=30,
                )
            self.state.selected_service_tier = target_tier
            self.state_store.save(self.state)
            self._thread_service_tier = target_tier
        finally:
            if thread_id and not self.is_busy:
                await self._unsubscribe_thread()
        logger.info(
            "Codex service tier updated",
            extra={
                "event": "codex.thread.service_tier.updated",
                "service_tier": target_tier,
                "model": model.model,
                **self._thread_log_fields(thread_id),
            },
        )
        return target_enabled


    async def list_skills(self, *, force_reload: bool = False) -> list[CodexSkill]:
        await self._ensure_app_server()
        cwd = str(self.working_directory)
        result = await self.app_server.request(
            "skills/list",
            {"cwds": [cwd], "forceReload": force_reload},
            timeout_seconds=60,
        )
        entries = result.get("data")
        if not isinstance(entries, list):
            raise RuntimeError("Codex app-server returned an invalid skills list")
        entry = next(
            (item for item in entries if isinstance(item, dict) and item.get("cwd") == cwd),
            None,
        )
        if entry is None:
            return []
        raw_skills = entry.get("skills")
        if not isinstance(raw_skills, list):
            raise RuntimeError("Codex app-server returned invalid skill metadata")
        skills: list[CodexSkill] = []
        for item in raw_skills:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            path = item.get("path")
            if not isinstance(name, str) or not name or not isinstance(path, str) or not path:
                continue
            interface = item.get("interface")
            display_name = interface.get("displayName") if isinstance(interface, dict) else None
            short_description = interface.get("shortDescription") if isinstance(interface, dict) else None
            description = item.get("description")
            if isinstance(short_description, str) and short_description:
                description = short_description
            skills.append(
                CodexSkill(
                    name=name,
                    description=description if isinstance(description, str) else "",
                    path=path,
                    enabled=item.get("enabled") is True,
                    display_name=display_name if isinstance(display_name, str) and display_name else None,
                )
            )
        return sorted(skills, key=lambda skill: (skill.name.casefold(), skill.path))


    async def set_skill_enabled(self, skill: CodexSkill, enabled: bool) -> None:
        await self._ensure_app_server()
        await self.app_server.request(
            "skills/config/write",
            {"name": skill.name, "path": skill.path, "enabled": enabled},
            timeout_seconds=30,
        )


    def select_skill_for_next_turn(self, skill: CodexSkill) -> None:
        if not skill.enabled:
            raise ValueError("Disabled skills cannot be selected for a turn")
        self.state.pending_skill_name = skill.name
        self.state.pending_skill_path = skill.path
        self.state_store.save(self.state)


    def clear_selected_skill(self) -> None:
        if self.state.pending_skill_name is None and self.state.pending_skill_path is None:
            return
        self.state.pending_skill_name = None
        self.state.pending_skill_path = None
        self.state_store.save(self.state)


    def current_memory_mode(self) -> str | None:
        thread_id = self.state.thread_id
        if thread_id:
            return self.state_store.load_thread(thread_id).memory_mode
        return self.state.selected_memory_mode


    async def read_memory_settings(self) -> CodexMemorySettings:
        await self._ensure_app_server()
        result = await self.app_server.request(
            "config/read",
            {"includeLayers": True},
            timeout_seconds=30,
        )
        config = result.get("config")
        config = config if isinstance(config, dict) else {}
        features = config.get("features")
        features = features if isinstance(features, dict) else {}
        memories = config.get("memories")
        memories = memories if isinstance(memories, dict) else {}
        version: str | None = None
        layers = result.get("layers")
        if isinstance(layers, list):
            for layer in layers:
                source = layer.get("name") if isinstance(layer, dict) else None
                if isinstance(source, dict) and source.get("type") == "user":
                    candidate = layer.get("version")
                    if isinstance(candidate, str) and candidate:
                        version = candidate
                    break
        return CodexMemorySettings(
            feature_enabled=features.get("memories") is True,
            use_memories=memories.get("use_memories") is True,
            generate_memories=memories.get("generate_memories") is True,
            user_config_version=version,
        )


    async def set_memory_mode(self, mode: str) -> None:
        if mode not in {"enabled", "disabled"}:
            raise ValueError("Unsupported memory mode")
        settings = await self.read_memory_settings()
        if not settings.feature_enabled:
            raise MemoryFeatureUnavailableError("The Codex memories feature is disabled")

        thread_id = self.state.thread_id
        if thread_id:
            await self._ensure_thread(create_if_missing=False)
            thread_id = self.state.thread_id
            if not thread_id:
                raise RuntimeError("The active Codex chat changed")
            await self.app_server.request(
                "thread/memoryMode/set",
                {"threadId": thread_id, "mode": mode},
                timeout_seconds=30,
            )
            thread_state = self.state_store.load_thread(thread_id)
            thread_state.memory_mode = mode
            self.state_store.save_thread(thread_id, thread_state)
        self.state.selected_memory_mode = mode
        self.state_store.save(self.state)
        if thread_id and not self.is_busy:
            await self._unsubscribe_thread(source="memory_mode_update")


    async def set_memory_generation(self, enabled: bool) -> None:
        settings = await self.read_memory_settings()
        if not settings.feature_enabled:
            raise MemoryFeatureUnavailableError("The Codex memories feature is disabled")
        if settings.generate_memories is enabled:
            return
        try:
            await self._write_global_memory_setting("memories.generate_memories", enabled, settings)
        except RpcError as exc:
            if not self._is_config_version_conflict(exc):
                raise
            refreshed = await self.read_memory_settings()
            if not refreshed.feature_enabled:
                raise MemoryFeatureUnavailableError("The Codex memories feature is disabled") from exc
            if refreshed.generate_memories is not enabled:
                await self._write_global_memory_setting("memories.generate_memories", enabled, refreshed)
        updated = await self.read_memory_settings()
        if updated.generate_memories is not enabled:
            raise RuntimeError("Codex policy prevented changing the global memory generation setting")


    async def toggle_memory_generation(self) -> bool:
        settings = await self.read_memory_settings()
        if not settings.feature_enabled:
            raise MemoryFeatureUnavailableError("The Codex memories feature is disabled")
        target = not settings.generate_memories
        try:
            await self._write_global_memory_setting("memories.generate_memories", target, settings)
        except RpcError as exc:
            if not self._is_config_version_conflict(exc):
                raise
            refreshed = await self.read_memory_settings()
            if not refreshed.feature_enabled:
                raise MemoryFeatureUnavailableError("The Codex memories feature is disabled") from exc
            target = not refreshed.generate_memories
            await self._write_global_memory_setting(
                "memories.generate_memories",
                target,
                refreshed,
            )
        updated = await self.read_memory_settings()
        if updated.generate_memories is not target:
            raise RuntimeError("Codex policy prevented changing the global memory generation setting")
        return target


    async def enable_memories_feature(self) -> None:
        settings = await self.read_memory_settings()
        if settings.feature_enabled:
            return
        try:
            await self._write_global_memory_setting("features.memories", True, settings)
        except RpcError as exc:
            if not self._is_config_version_conflict(exc):
                raise
            refreshed = await self.read_memory_settings()
            if not refreshed.feature_enabled:
                await self._write_global_memory_setting("features.memories", True, refreshed)
        updated = await self.read_memory_settings()
        if not updated.feature_enabled:
            raise RuntimeError("Codex policy prevents enabling memories for new threads")


    async def reset_memories(self) -> None:
        settings = await self.read_memory_settings()
        if not settings.feature_enabled:
            raise MemoryFeatureUnavailableError("The Codex memories feature is disabled")
        await self.app_server.request("memory/reset", timeout_seconds=60)


    async def _write_global_memory_setting(
        self,
        key_path: str,
        value: bool,
        settings: CodexMemorySettings,
        *,
        reload_user_config: bool = True,
    ) -> None:
        params: dict[str, Any] = {
            "edits": [{"keyPath": key_path, "mergeStrategy": "replace", "value": value}],
            "reloadUserConfig": reload_user_config,
        }
        if settings.user_config_version:
            params["expectedVersion"] = settings.user_config_version
        await self.app_server.request("config/batchWrite", params, timeout_seconds=30)


    def default_model(self, models: list[CodexModel]) -> str | None:
        if self.config.codex_model:
            return self.config.codex_model
        return next((model.model for model in models if model.is_default), None)


    async def select_model(self, model: str | None, effort: str | None = None) -> None:
        # A turn uses the model parameters captured at turn/start. Updating the
        # thread settings while it runs therefore changes the next turn only.
        await self._ensure_app_server()
        if self.state.thread_id:
            await self._ensure_thread()
        service_tier = self._thread_service_tier or self.state.selected_service_tier
        configured_model = model if model is not None else self.config.codex_model
        if service_tier and service_tier != "default":
            models = await self.list_models()
            target_model = configured_model or self.default_model(models)
            target_info = next((item for item in models if item.model == target_model), None)
            if target_info is None or not any(tier.id == service_tier for tier in target_info.service_tiers):
                service_tier = None
        if self.state.thread_id:
            settings: dict[str, Any] = {
                "threadId": self.state.thread_id,
                "model": configured_model,
                "effort": effort,
            }
            if (self._thread_service_tier or self.state.selected_service_tier) != service_tier:
                settings["serviceTier"] = service_tier
            await self.app_server.request(
                "thread/settings/update",
                settings,
                timeout_seconds=30,
            )
            self._thread_model = configured_model
            self._thread_effort = effort
            self._thread_service_tier = service_tier
            await self._unsubscribe_thread()
        self.state.selected_model = model
        self.state.selected_effort = effort
        self.state.selected_service_tier = service_tier
        self.state_store.save(self.state)


    def _effective_model(self) -> str | None:
        return self.state.selected_model or self.config.codex_model


    @property
    def collaboration_mode(self) -> str:
        return self._collaboration_mode

    @collaboration_mode.setter
    def collaboration_mode(self, value: str) -> None:
        self._collaboration_mode = value

    @property
    def thread_model(self) -> str | None:
        return self._thread_model

    @thread_model.setter
    def thread_model(self, value: str | None) -> None:
        self._thread_model = value

    @property
    def thread_effort(self) -> str | None:
        return self._thread_effort

    @thread_effort.setter
    def thread_effort(self, value: str | None) -> None:
        self._thread_effort = value

    @property
    def thread_service_tier(self) -> str | None:
        return self._thread_service_tier

    @thread_service_tier.setter
    def thread_service_tier(self, value: str | None) -> None:
        self._thread_service_tier = value


    async def toggle_plan_mode(self) -> str:
        if not self.state.thread_id:
            raise RuntimeError("There is no active Codex chat")
        await self._ensure_thread(create_if_missing=False)
        target_mode = "default" if self._collaboration_mode == "plan" else "plan"
        await self._update_collaboration_mode(target_mode)
        return target_mode


    async def set_collaboration_mode(
        self,
        mode: str,
        *,
        create_if_missing: bool = False,
    ) -> None:
        if mode not in {"plan", "default"}:
            raise ValueError("Unsupported collaboration mode")
        if not self.state.thread_id and not create_if_missing:
            raise RuntimeError("There is no active Codex chat")
        if create_if_missing and not self.state.thread_id:
            await self._ensure_app_server()
            presets = await self._list_collaboration_mode_presets()
            if mode not in presets:
                raise CollaborationModeUnavailableError(
                    f"Codex app-server does not provide the {mode} collaboration mode"
                )
        previous_thread_id = self.state.thread_id
        await self._ensure_thread(create_if_missing=create_if_missing)
        created_thread = self.state.thread_id != previous_thread_id
        await self._update_collaboration_mode(mode, unsubscribe=not created_thread)


    async def _update_collaboration_mode(self, mode: str, *, unsubscribe: bool = True) -> None:
        thread_id = self.state.thread_id
        if not thread_id:
            raise RuntimeError("There is no active Codex chat")

        try:
            presets = await self._list_collaboration_mode_presets()
            preset = presets.get(mode)
            if preset is None:
                raise CollaborationModeUnavailableError(
                    f"Codex app-server does not provide the {mode} collaboration mode"
                )

            model = (
                self._thread_model
                or self._effective_model()
                or self.config.codex_model
                or preset.get("model")
            )
            if not isinstance(model, str) or not model:
                raise RuntimeError("Codex app-server did not report the current model")

            effort = self._thread_effort or self.state.selected_effort
            collaboration_settings: dict[str, Any] = {
                "model": model,
                "developer_instructions": None,
            }
            if effort:
                collaboration_settings["reasoning_effort"] = effort

            await self.app_server.request(
                "thread/settings/update",
                {
                    "threadId": thread_id,
                    "collaborationMode": {
                        "mode": mode,
                        "settings": collaboration_settings,
                    },
                },
                timeout_seconds=30,
            )
        finally:
            if unsubscribe and not self.is_busy:
                await self._unsubscribe_thread(source="collaboration_mode_update")

        self._collaboration_mode = mode
        self._thread_model = model
        self._thread_effort = effort
        logger.info(
            "Codex collaboration mode updated",
            extra={
                "event": "codex.thread.collaboration_mode.updated",
                "collaboration_mode": mode,
                "model": model,
                "effort": effort,
                **self._thread_log_fields(thread_id),
            },
        )


    async def _list_collaboration_mode_presets(self) -> dict[str, dict[str, Any]]:
        if self._collaboration_mode_presets is not None:
            return self._collaboration_mode_presets
        try:
            result = await self.app_server.request(
                "collaborationMode/list",
                {},
                timeout_seconds=30,
            )
        except RpcError as exc:
            raise CollaborationModeUnavailableError(
                "Codex app-server does not support collaboration modes"
            ) from exc

        data = result.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Codex app-server returned an invalid collaboration mode list")
        presets: dict[str, dict[str, Any]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            mode = item.get("mode")
            if mode in {"plan", "default"}:
                presets[mode] = item
        if not {"plan", "default"}.issubset(presets):
            raise CollaborationModeUnavailableError(
                "Codex app-server does not provide both Plan and Default modes"
            )
        self._collaboration_mode_presets = presets
        return presets


    def _capture_thread_settings(self, response: dict[str, Any]) -> None:
        thread = response.get("thread")
        thread = thread if isinstance(thread, dict) else {}
        thread_settings = response.get("threadSettings")
        if isinstance(thread_settings, dict):
            thread = {**thread, **thread_settings}
        model = response.get("model") or thread.get("model")
        effort = response.get("reasoningEffort") or thread.get("reasoningEffort")
        collaboration_mode = response.get("collaborationMode")
        if not isinstance(collaboration_mode, dict):
            collaboration_mode = thread.get("collaborationMode")
        settings = collaboration_mode.get("settings") if isinstance(collaboration_mode, dict) else None
        settings = settings if isinstance(settings, dict) else {}
        model = model or settings.get("model")
        effort = effort or settings.get("reasoning_effort")
        if isinstance(model, str) and model:
            self._thread_model = model
        if isinstance(effort, str) and effort:
            self._thread_effort = effort
        if "serviceTier" in response:
            tier = response.get("serviceTier")
            self._thread_service_tier = tier if isinstance(tier, str) and tier else None
        elif "serviceTier" in thread:
            tier = thread.get("serviceTier")
            self._thread_service_tier = tier if isinstance(tier, str) and tier else None

        mode = collaboration_mode.get("mode") if isinstance(collaboration_mode, dict) else None
        if mode in {"plan", "default"}:
            self._collaboration_mode = mode
        elif "collaborationMode" in response or "collaborationMode" in thread:
            self._collaboration_mode = "default"
