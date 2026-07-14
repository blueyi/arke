# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for MCP resource subscription (audit item C4).

Covers:
- resources/subscribe + resources/unsubscribe JSON-RPC methods
- Notification delivery after state-mutating tools (apply_decision, checkpoint, rollback)
- Capabilities advertising subscribe: true
- Unknown URI rejection
- Multiple subscriptions with targeted notification delivery
"""

from __future__ import annotations

import json

import pytest

from arke.agent.mcp_server import ArkeMCPServer


def _server():
    return ArkeMCPServer("matmul", {"A": [64, 64], "B": [64, 64]})


def _req(method, params=None, req_id=1):
    r = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        r["params"] = params
    return r


# ── Capabilities ──────────────────────────────────────────────────────


class TestCapabilities:
    def test_capabilities_advertise_subscribe_true(self):
        """_initialize() must return subscribe: true after C4 implementation."""
        resp = _server().handle(_req("initialize"))
        caps = resp["result"]["capabilities"]
        assert caps["resources"]["subscribe"] is True


# ── Subscribe / Unsubscribe ──────────────────────────────────────────


class TestSubscribe:
    def test_subscribe_to_resource(self):
        """Subscribe to a known resource URI → success (empty result)."""
        s = _server()
        resp = s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))
        assert "result" in resp
        assert resp["result"] == {}

    def test_subscribe_stores_subscription(self):
        """After subscribing, the URI is tracked in _subscriptions."""
        s = _server()
        s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))
        assert "arke://strategy/current" in s._subscriptions.get("_default", set())

    def test_subscribe_unknown_uri_returns_error(self):
        """Subscribe to a nonexistent URI → JSON-RPC error."""
        s = _server()
        resp = s.handle(_req("resources/subscribe", {"uri": "arke://nope/bad"}))
        assert "error" in resp
        assert resp["error"]["code"] == -32603

    def test_unsubscribe(self):
        """Subscribe then unsubscribe → URI removed from tracking."""
        s = _server()
        s.handle(_req("resources/subscribe", {"uri": "arke://actions/legal"}))
        assert "arke://actions/legal" in s._subscriptions["_default"]

        resp = s.handle(_req("resources/unsubscribe", {"uri": "arke://actions/legal"}))
        assert "result" in resp
        assert resp["result"] == {}
        assert "arke://actions/legal" not in s._subscriptions.get("_default", set())

    def test_unsubscribe_unknown_uri_returns_error(self):
        """Unsubscribe from a nonexistent URI → JSON-RPC error."""
        s = _server()
        resp = s.handle(_req("resources/unsubscribe", {"uri": "arke://nope"}))
        assert "error" in resp
        assert resp["error"]["code"] == -32603

    def test_unsubscribe_when_not_subscribed_is_noop(self):
        """Unsubscribe from a known URI when never subscribed → success (no-op)."""
        s = _server()
        resp = s.handle(_req("resources/unsubscribe", {"uri": "arke://strategy/current"}))
        assert "result" in resp
        assert resp["result"] == {}

    def test_idempotent_subscribe(self):
        """Subscribing twice to the same URI does not duplicate the entry."""
        s = _server()
        s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))
        s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))
        # It's a set, so only one entry.
        assert len(s._subscriptions["_default"]) == 1


# ── Notification delivery ────────────────────────────────────────────


class TestNotifications:
    def test_notify_resource_changed_with_subscriber(self):
        """notify_resource_changed() produces notifications for subscribed URIs."""
        s = _server()
        s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))
        notes = s.notify_resource_changed("arke://strategy/current")
        assert len(notes) == 1
        assert notes[0]["method"] == "notifications/resources/updated"
        assert notes[0]["params"]["uri"] == "arke://strategy/current"
        assert "id" not in notes[0]  # notifications have no id

    def test_notify_resource_changed_without_subscriber(self):
        """notify_resource_changed() returns empty list when no subscribers."""
        s = _server()
        notes = s.notify_resource_changed("arke://strategy/current")
        assert notes == []

    def test_notifications_accumulate_in_pending(self):
        """Notifications are appended to _pending_notifications."""
        s = _server()
        s.handle(_req("resources/subscribe", {"uri": "arke://actions/legal"}))
        s.notify_resource_changed("arke://actions/legal")
        s.notify_resource_changed("arke://actions/legal")
        assert len(s._pending_notifications) == 2

    def test_notification_after_apply_decision(self):
        """After apply_decision succeeds, subscribers get strategy+actions notifications."""
        s = _server()
        # Subscribe to both mutable resources.
        s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))
        s.handle(_req("resources/subscribe", {"uri": "arke://actions/legal"}))
        assert len(s._pending_notifications) == 0

        # apply_decision (must be legal — use a tile decision).
        resp = s.handle(_req("tools/call", {
            "name": "apply_decision",
            "arguments": {
                "kind": "tile",
                "params": {"loop": "i", "factors": [16]},
                "rationale": "test notification delivery",
            },
        }))
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert payload["success"] is True

        # Should have 2 notifications (strategy + actions).
        assert len(s._pending_notifications) == 2
        uris = {n["params"]["uri"] for n in s._pending_notifications}
        assert uris == {"arke://strategy/current", "arke://actions/legal"}

    def test_notification_after_checkpoint(self):
        """After checkpoint succeeds, subscribers get notifications."""
        s = _server()
        s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))

        resp = s.handle(_req("tools/call", {
            "name": "checkpoint",
            "arguments": {"label": "snap1"},
        }))
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert payload["success"] is True
        # checkpoint fires notifications for strategy + actions.
        assert len(s._pending_notifications) >= 1
        assert any(n["params"]["uri"] == "arke://strategy/current"
                    for n in s._pending_notifications)

    def test_notification_after_rollback(self):
        """After checkpoint → apply_decision → rollback, subscribers get notifications."""
        s = _server()
        s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))
        s.handle(_req("resources/subscribe", {"uri": "arke://actions/legal"}))

        # Create a checkpoint.
        s.handle(_req("tools/call", {
            "name": "checkpoint",
            "arguments": {"label": "before_explore"},
        }))
        # Apply a decision.
        s.handle(_req("tools/call", {
            "name": "apply_decision",
            "arguments": {
                "kind": "tile",
                "params": {"loop": "i", "factors": [16]},
                "rationale": "exploration branch",
            },
        }))
        # Clear pending to isolate rollback notifications.
        s._pending_notifications.clear()

        # Rollback.
        resp = s.handle(_req("tools/call", {
            "name": "rollback",
            "arguments": {"label": "before_explore"},
        }))
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert payload["success"] is True

        # Rollback mutates state → notifications fired.
        assert len(s._pending_notifications) == 2
        uris = {n["params"]["uri"] for n in s._pending_notifications}
        assert uris == {"arke://strategy/current", "arke://actions/legal"}

    def test_no_notification_on_failed_tool(self):
        """Failed tool calls should NOT trigger notifications."""
        s = _server()
        s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))

        # apply_decision without rationale → fail.
        resp = s.handle(_req("tools/call", {
            "name": "apply_decision",
            "arguments": {
                "kind": "tile",
                "params": {"loop": "i", "factors": [16]},
                # No rationale — triggers contract error.
            },
        }))
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert payload["success"] is False

        # No notifications should have been fired.
        assert len(s._pending_notifications) == 0

    def test_no_notification_for_read_only_tool(self):
        """Read-only tools (get_hw_profile) should NOT trigger notifications."""
        s = _server()
        s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))

        resp = s.handle(_req("tools/call", {
            "name": "get_hw_profile",
            "arguments": {},
        }))
        assert resp["result"]["isError"] is False
        assert len(s._pending_notifications) == 0


# ── Multiple subscriptions ───────────────────────────────────────────


class TestMultipleSubscriptions:
    def test_subscribe_to_multiple_uris(self):
        """Subscribing to 2 different URIs tracks both."""
        s = _server()
        s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))
        s.handle(_req("resources/subscribe", {"uri": "arke://hw/profile"}))
        subs = s._subscriptions["_default"]
        assert subs == {"arke://strategy/current", "arke://hw/profile"}

    def test_only_matching_uri_gets_notification(self):
        """When subscribed to 2 URIs, only the changed one gets a notification."""
        s = _server()
        s.handle(_req("resources/subscribe", {"uri": "arke://strategy/current"}))
        s.handle(_req("resources/subscribe", {"uri": "arke://context/op"}))

        notes = s.notify_resource_changed("arke://strategy/current")
        assert len(notes) == 1
        assert notes[0]["params"]["uri"] == "arke://strategy/current"
        # No notification for arke://context/op.
        assert all(n["params"]["uri"] != "arke://context/op"
                    for n in s._pending_notifications)

    def test_subscribe_all_four_uris(self):
        """Can subscribe to all 4 known resource URIs."""
        s = _server()
        for uri in [
            "arke://context/op", "arke://hw/profile",
            "arke://strategy/current", "arke://actions/legal",
        ]:
            resp = s.handle(_req("resources/subscribe", {"uri": uri}))
            assert "result" in resp
        assert len(s._subscriptions["_default"]) == 4
