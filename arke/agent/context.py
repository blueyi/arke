# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — Context file loader.

Auto-loads agent context files (AGENTS.md, IDENTITY.md, SOUL.md, TOOLS.md)
from the project root and injects them into the LLM system prompt.

Usage:
    from arke.agent.context import load_agent_context, format_context_prompt

    context = load_agent_context()
    system_prompt = format_context_prompt(context) + "\n\n" + optimization_prompt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Context files loaded from project root, in order
CONTEXT_FILES = [
    "AGENTS.md",
    "IDENTITY.md",
    "SOUL.md",
    "TOOLS.md",
]

# Max size per file to prevent excessive token usage (32KB)
MAX_FILE_SIZE = 32 * 1024


def _find_project_root(start: Optional[Path] = None) -> Path:
    """Find the Arke project root by walking up from start dir.

    Looks for markers: pyproject.toml, AGENTS.md, or arke/ directory.
    Falls back to the arke package location if traversal fails.
    """
    if start is None:
        # Start from this file's location: arke/agent/context.py -> arke/ -> root
        start = Path(__file__).resolve().parent.parent.parent

    current = start
    for _ in range(10):  # max 10 levels up
        if (current / "pyproject.toml").exists() and (current / "arke").is_dir():
            return current
        if (current / "AGENTS.md").exists() and (current / "arke").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Fallback: assume arke/agent/context.py is at <root>/arke/agent/context.py
    fallback = Path(__file__).resolve().parent.parent.parent
    logger.warning("Could not find project root, using fallback: %s", fallback)
    return fallback


@dataclass
class AgentContext:
    """Loaded agent context from project root files."""

    project_root: Path
    files: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def agents_md(self) -> str:
        return self.files.get("AGENTS.md", "")

    @property
    def identity_md(self) -> str:
        return self.files.get("IDENTITY.md", "")

    @property
    def soul_md(self) -> str:
        return self.files.get("SOUL.md", "")

    @property
    def tools_md(self) -> str:
        return self.files.get("TOOLS.md", "")

    def has(self, filename: str) -> bool:
        return filename in self.files and len(self.files[filename]) > 0


def load_agent_context(
    root: Optional[Path] = None,
    files: Optional[list[str]] = None,
    max_file_size: int = MAX_FILE_SIZE,
) -> AgentContext:
    """Load agent context files from the project root.

    Args:
        root: Project root directory. Auto-detected if None.
        files: List of filenames to load. Defaults to CONTEXT_FILES.
        max_file_size: Maximum bytes per file (truncated with warning).

    Returns:
        AgentContext with loaded file contents.
    """
    if root is None:
        root = _find_project_root()

    if files is None:
        files = CONTEXT_FILES

    ctx = AgentContext(project_root=root)

    for filename in files:
        filepath = root / filename
        if not filepath.exists():
            logger.debug("Context file not found: %s", filepath)
            continue

        try:
            content = filepath.read_text(encoding="utf-8")
            if len(content) > max_file_size:
                logger.warning(
                    "Context file %s exceeds %d bytes (%d), truncating",
                    filename,
                    max_file_size,
                    len(content),
                )
                content = content[:max_file_size] + "\n\n... (truncated)"

            ctx.files[filename] = content
            logger.debug("Loaded context file: %s (%d bytes)", filename, len(content))

        except Exception as e:
            msg = f"Failed to load {filename}: {e}"
            logger.warning(msg)
            ctx.errors.append(msg)

    loaded = list(ctx.files.keys())
    logger.info(
        "Agent context loaded from %s: %d files (%s)",
        root,
        len(loaded),
        ", ".join(loaded) or "none",
    )

    return ctx


def format_context_prompt(context: AgentContext) -> str:
    """Format loaded context files into a system prompt section.

    Returns a markdown string suitable for prepending to the LLM system prompt.
    Files are included in order with clear section headers.
    """
    if not context.files:
        return ""

    sections = []
    sections.append("# Agent Context\n")
    sections.append("The following project context files were auto-loaded from the Arke repository.\n")

    for filename in CONTEXT_FILES:
        if filename not in context.files:
            continue
        content = context.files[filename]
        sections.append(f"## {filename}\n\n{content}\n")

    return "\n".join(sections)
