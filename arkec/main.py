# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke CLI — compatibility shim.

The canonical CLI lives in :mod:`arke.cli` (the ``arke`` console-script entry
point declared in ``pyproject.toml``). This module previously carried a second,
stub-only ``click`` CLI whose ``optimize`` / ``codegen`` / ``verify`` commands
only printed ``[TODO]`` — a misleading dead duplicate.

It now delegates to the real CLI so there is exactly one implementation. Kept
for backward compatibility with anything invoking ``python -m arkec.main``.
"""

from __future__ import annotations

from arke.cli import main as _arke_main


def cli() -> None:
    """Delegate to the canonical Arke CLI (:func:`arke.cli.main`)."""
    _arke_main()


# Backward-compatible alias.
main = cli


if __name__ == "__main__":
    cli()
