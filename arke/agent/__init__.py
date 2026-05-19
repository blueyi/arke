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

# Lazy re-export of the heavy ``optimize`` module to avoid a circular
# import chain (``arke.agent`` ← ``arke.agent.optimize`` ←
# ``arke.learn.trajectory`` ← ``arke.learn.trajectory_schema`` ←
# ``arke.agent.events`` ← ``arke.agent``). Contract modules
# (``events``, ``facade``) stay eager because they are leaf-light and
# constitute the Façade public surface.

def __getattr__(name: str):  # PEP 562 module-level lazy attribute hook
    if name in {"HeuristicStrategyGenerator", "OptimizeResult", "optimize_file"}:
        from arke.agent import optimize as _optimize
        value = getattr(_optimize, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
