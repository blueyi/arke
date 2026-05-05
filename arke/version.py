# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Central version constants and schema validation helpers for Arke."""

PACKAGE_VERSION = "0.2.0.dev0"
LANG_SCHEMA_VERSION = "2.0.0"
IR_SCHEMA_VERSION = "2.0.0"


def resolve_ir_schema_version(version: str | None, *, artifact: str) -> str:
    """Return the active IR schema version, rejecting unsupported explicit versions."""
    if version is None:
        return IR_SCHEMA_VERSION
    if version != IR_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {artifact} version: expected {IR_SCHEMA_VERSION!r}, got {version!r}"
        )
    return version
