from __future__ import annotations

import json


RECAP_PROMPT = """Summarize this Codex conversation without compacting or otherwise modifying it.
Return a concise progress recap with these sections:
- Goal
- Decisions and constraints
- Completed work
- Open questions or blockers
- Next step
Use only information present in the conversation. If a section has no known information, say so. Do not run commands or modify files."""


GOAL_CONTINUATION_PROMPT = """Continue working toward the active Goal using the existing thread context.
Do not restate, narrow, or replace the Goal. Check what remains, take the next useful step, and continue until the Goal is complete or blocked. If it is already complete, verify that from the current evidence and mark the Goal complete without doing unrelated work."""


def init_prompt(project_root: str) -> str:
    target_path = json.dumps(f"{project_root}/AGENTS.md", ensure_ascii=False)
    return f"""Initialize the current project by creating a concise English AGENTS.md at:
{target_path}

Inspect the project and include only verified, useful instructions for future coding agents: project purpose, source layout, commands that actually work, and important implementation constraints. Keep it short and specific to this repository.

Before writing, check again that the target does not exist and is not a symlink. If it exists, do not modify it or any other file. Create only this AGENTS.md in the project root. Do not run installation, build, or other mutating commands. Do not follow symlinks while inspecting files."""
