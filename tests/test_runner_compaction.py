# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for C3 — context compaction helpers.

Verifies _messages_chars sizing and _compact_messages folding: preamble
(system + first user) preserved, middle elided to a single digest, last-N
turns kept, and a no-op on short logs. Pure-logic (no network).
"""

from __future__ import annotations

from arke.agent.runner import _compact_messages, _messages_chars


def _log(n_middle: int):
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "optimize matmul"},
    ]
    for k in range(n_middle):
        msgs.append({"role": "assistant", "content": f"turn {k}"})
        msgs.append({"role": "user", "content": f"tool result {k}"})
    return msgs


def test_messages_chars_grows_with_log():
    small = _messages_chars(_log(1))
    big = _messages_chars(_log(20))
    assert big > small > 0


def test_compact_preserves_preamble_and_tail():
    msgs = _log(10)  # 2 preamble + 20 middle msgs
    out, did = _compact_messages(msgs, "openai", keep_last_turns=4)
    assert did is True
    # system + first user preserved at the head
    assert out[0]["role"] == "system"
    assert out[1]["content"] == "optimize matmul"
    # a single digest follows
    assert "context compacted" in out[2]["content"]
    # last 4 messages preserved verbatim at the tail
    assert out[-4:] == msgs[-4:]
    # compaction actually shrank the log
    assert len(out) < len(msgs)


def test_compact_noop_on_short_log():
    msgs = _log(1)  # too short to benefit
    out, did = _compact_messages(msgs, "openai", keep_last_turns=4)
    assert did is False
    assert out == msgs


def test_compact_anthropic_no_system_msg():
    # anthropic keeps system out-of-band; first message is the user intro
    msgs = [{"role": "user", "content": "optimize matmul"}]
    for k in range(8):
        msgs.append({"role": "assistant", "content": f"a{k}"})
        msgs.append({"role": "user", "content": f"r{k}"})
    out, did = _compact_messages(msgs, "anthropic", keep_last_turns=4)
    assert did is True
    assert out[0]["content"] == "optimize matmul"  # intro preserved
    assert "context compacted" in out[1]["content"]
