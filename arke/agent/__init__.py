# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent package."""

from arke.agent.facade import (
    FACADE_CONTRACT_ID,
    FACADE_LOCKED_ON,
    FACADE_V1_SCHEMA_PATH,
    FACADE_V1_TOOLS,
    FACADE_VERSION,
    load_facade_v1_schema,
)
from arke.agent.optimize import HeuristicStrategyGenerator, OptimizeResult, optimize_file

__all__ = [
    "HeuristicStrategyGenerator",
    "OptimizeResult",
    "optimize_file",
    # Façade v1.0 public contract
    "FACADE_VERSION",
    "FACADE_CONTRACT_ID",
    "FACADE_LOCKED_ON",
    "FACADE_V1_TOOLS",
    "FACADE_V1_SCHEMA_PATH",
    "load_facade_v1_schema",
]
