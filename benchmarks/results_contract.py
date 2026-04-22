# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Generic benchmark result-tree contract helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


DEFAULT_LAYER_NAMES = ("l1", "l2")


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def check_result_tree_artifacts(
    root: Path,
    *,
    layer_artifacts: Iterable[str],
    root_artifacts: Iterable[str],
    tree_name: str,
    layers: Iterable[str] = DEFAULT_LAYER_NAMES,
) -> tuple[bool, str]:
    found: list[str] = []
    missing: list[str] = []

    for artifact in root_artifacts:
        path = root / artifact
        if path.exists():
            found.append(f"root:{artifact}")
        else:
            missing.append(f"root:{artifact} missing")

    for layer in layers:
        layer_dir = root / layer
        if not layer_dir.exists():
            missing.append(f"{layer}:{_display_path(layer_dir, root)} absent")
            continue
        layer_missing = [artifact for artifact in layer_artifacts if not (layer_dir / artifact).exists()]
        if layer_missing:
            missing.append(f"{layer}:{', '.join(layer_missing)} missing")
        else:
            found.append(f"{layer}:{_display_path(layer_dir, root)}")

    detail = "; ".join(found + missing)
    if tree_name:
        detail = f"{tree_name}: {detail}" if detail else tree_name
    return not missing, detail
