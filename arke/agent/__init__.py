# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent package."""

from arke.agent.events import (
    EVENT_KINDS_V1,
    EVENT_PAYLOADS_V1,
    EVENTS_CONTRACT_ID,
    EVENTS_LOCKED_ON,
    EVENTS_V1_SCHEMA_PATH,
    EVENTS_VERSION,
    EventKind,
    OptimizationEvent,
    PayloadField,
    load_events_v1_schema,
    make_event,
    validate_payload,
)
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
    # Façade v1.0 public contract — tools
    "FACADE_VERSION",
    "FACADE_CONTRACT_ID",
    "FACADE_LOCKED_ON",
    "FACADE_V1_TOOLS",
    "FACADE_V1_SCHEMA_PATH",
    "load_facade_v1_schema",
    # Façade v1.0 public contract — event stream
    "EVENTS_VERSION",
    "EVENTS_CONTRACT_ID",
    "EVENTS_LOCKED_ON",
    "EventKind",
    "EVENT_KINDS_V1",
    "PayloadField",
    "EVENT_PAYLOADS_V1",
    "OptimizationEvent",
    "make_event",
    "validate_payload",
    "EVENTS_V1_SCHEMA_PATH",
    "load_events_v1_schema",
]
