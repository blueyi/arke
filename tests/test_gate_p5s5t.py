# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P5-S5-T mechanism tests — extraction, rule evaluation, criteria logic,
prompt no-leak assertions, gate dry-run smoke. NO GPU, no live LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.gate_p5s5t import (
    C3_GEOMEAN_MAX,
    HELDOUT_MATRIX,
    build_decisions,
    dims_from_l3_case,
    eval_c1,
    eval_c2,
    eval_c3,
    eval_c4,
    eval_c5,
    l3_sweep_key,
)
from benchmarks.live.generalize_p5s5t import (
    apply_rule,
    build_prompt,
    load_explore_strategies,
)
from benchmarks.live.run_p5s5t import (
    EXPLORE_MATRIX,
    case_key,
    extract_strategy,
    shape_label,
    write_strategy_file,
)

REPO = Path(__file__).resolve().parent.parent
LIVE_STATE = REPO / "benchmarks" / "results" / "phase5" / "s5" / "live_run_state.json"


# ── explore matrix sanity ───────────────────────────────────────────────────

def test_explore_matrix_shape():
    assert len(EXPLORE_MATRIX) == 8
    ops = {op for op, _ in EXPLORE_MATRIX}
    assert ops == {"rmsnorm", "softmax", "layernorm", "matmul"}
    for op, dims in EXPLORE_MATRIX:
        assert len(dims) == (3 if op == "matmul" else 2)
    assert case_key("matmul", [1024, 1024, 1024]) == "matmul@1024x1024x1024"
    assert shape_label([32, 4096]) == "32x4096"


# ── strategy extraction ─────────────────────────────────────────────────────

class TestExtractStrategy:
    def test_real_live_state_sample(self):
        """The real matmul live-run sample: best checkpoint wins."""
        state = json.loads(LIVE_STATE.read_text())
        decisions, source = extract_strategy(state, str(LIVE_STATE))
        # best correct latency in the sample is 0.41176 ms
        # (checkpoint best_wmma_4x4_WTM1_WTN1 == top-level best_result)
        assert source["best_latency_ms"] == pytest.approx(0.41176)
        assert source["baseline_ratio"] == pytest.approx(0.4875)
        # the reproducing set is the single wmma decision (the top-level
        # decision_log has a trailing never-profiled pipeline_stages probe,
        # strategy_decisions=1)
        assert len(decisions) == 1
        assert decisions[0]["kind"] == "wmma_tile"
        assert decisions[0]["params"] == {"WM": 4, "WN": 4, "WTM": 1, "WTN": 1}
        assert decisions[0]["level"] == 3
        assert decisions[0]["rationale"].strip()
        # ties prefer checkpoints (exact decision_log snapshot)
        assert source["extracted_from"].endswith(
            "::checkpoint:best_wmma_4x4_WTM1_WTN1")

    def test_empty_decision_log_keeps_default(self):
        state = {
            "decision_log": [],
            "best_result": {"correct": True, "latency_ms": 1.0,
                            "baseline_ratio": 1.0,
                            "metadata": {"strategy_decisions": 0}},
            "checkpoints": {},
        }
        decisions, source = extract_strategy(state, "s.json")
        assert decisions == []
        assert source["best_latency_ms"] == 1.0

    def test_all_rolled_back_keeps_default(self):
        """Baseline checkpoint (empty log) is the best correct result."""
        state = {
            "decision_log": [
                {"kind": "block_threads", "params": {"n": 1024}, "level": 3,
                 "rationale": {"text": "try wide blocks"}},
            ],
            # the applied decision made things worse
            "best_result": {"correct": True, "latency_ms": 2.0,
                            "baseline_ratio": 0.5,
                            "metadata": {"strategy_decisions": 1}},
            "checkpoints": {
                "baseline": {"decision_log": [],
                             "best_result": {"correct": True,
                                             "latency_ms": 1.0,
                                             "baseline_ratio": 1.0}},
            },
        }
        decisions, source = extract_strategy(state, "s.json")
        assert decisions == []
        assert source["extracted_from"].endswith("::checkpoint:baseline")

    def test_top_level_better_than_checkpoints(self):
        state = {
            "decision_log": [
                {"kind": "block_threads", "params": {"n": 512}, "level": 3,
                 "rationale": "faster"},
                {"kind": "block_threads", "params": {"n": 128}, "level": 3,
                 "rationale": "post-best probe, never profiled"},
            ],
            "best_result": {"correct": True, "latency_ms": 0.5,
                            "baseline_ratio": 1.4,
                            "metadata": {"strategy_decisions": 1}},
            "checkpoints": {
                "ck1": {"decision_log": [
                            {"kind": "block_threads", "params": {"n": 256},
                             "level": 3, "rationale": "r"}],
                        "best_result": {"correct": True, "latency_ms": 0.9}},
            },
        }
        decisions, source = extract_strategy(state, "s.json")
        assert source["extracted_from"].endswith("::top_level")
        # strategy_decisions=1 prunes the never-profiled trailing probe
        assert len(decisions) == 1
        assert decisions[0]["params"] == {"n": 512}

    def test_incorrect_results_excluded(self):
        state = {
            "decision_log": [{"kind": "k", "params": {}, "level": 3,
                              "rationale": "r"}],
            "best_result": {"correct": False, "latency_ms": 0.1},
            "checkpoints": {
                "ok": {"decision_log": [],
                       "best_result": {"correct": True, "latency_ms": 1.0}},
            },
        }
        decisions, source = extract_strategy(state, "s.json")
        assert decisions == []
        assert source["extracted_from"].endswith("::checkpoint:ok")

    def test_no_correct_result_at_all(self):
        decisions, source = extract_strategy(
            {"decision_log": [], "best_result": None, "checkpoints": {}},
            "s.json")
        assert decisions == []
        assert source["best_latency_ms"] is None
        # Empty top-level decision_log short-circuits to "keep default"
        # (the agent's final verdict; rollbacks prune the log).
        assert source["extracted_from"].endswith("::final_empty_decision_log")

    def test_rolled_back_checkpoint_not_resurrected(self):
        # Agent explored bt(1024) (checkpointed with a latency) but rolled
        # back and finished with an empty decision_log -> keep default,
        # even though the checkpoint has the only latency_ms.
        state = {
            "decision_log": [],
            "best_result": {"correct": True, "latency_ms": 0.013,
                            "baseline_ratio": 1.1},
            "checkpoints": {
                "bt1024_v1": {
                    "decision_log": [{"kind": "block_threads",
                                      "params": {"n": 1024}, "level": 3,
                                      "rationale": "r"}],
                    "best_result": {"correct": True, "latency_ms": 0.013},
                },
            },
        }
        decisions, source = extract_strategy(state, "s.json")
        assert decisions == []
        assert source["extracted_from"].endswith("::final_empty_decision_log")

    def test_write_strategy_file_schema(self, tmp_path):
        path = write_strategy_file(
            "rmsnorm", [32, 4096],
            [{"kind": "block_threads", "params": {"n": 512}, "level": 3,
              "rationale": "r"}],
            {"state_json": "s.json", "best_latency_ms": 0.01,
             "baseline_ratio": 1.2, "extracted_from": "s.json::top_level"},
            out_dir=tmp_path)
        rec = json.loads(path.read_text())
        assert path.name == "rmsnorm_32x4096.json"
        assert rec["op"] == "rmsnorm"
        assert rec["shape"] == "32x4096"
        assert rec["role"] == "explore"
        assert rec["decisions"][0]["kind"] == "block_threads"
        assert rec["source"]["extracted_from"] == "s.json::top_level"


# ── apply_rule ──────────────────────────────────────────────────────────────

DEC_512 = [{"kind": "block_threads", "params": {"n": 512}, "level": 3,
            "rationale": "small rows"}]
DEC_1024 = [{"kind": "block_threads", "params": {"n": 1024}, "level": 3,
             "rationale": "big rows"}]


class TestApplyRule:
    def _rule(self, rules, fallback=None, op="rmsnorm"):
        return {"op": op, "rules": rules,
                "fallback_decisions": fallback or []}

    @pytest.mark.parametrize("cmp,value,dims,expect_match", [
        ("<=", 64, [64, 4096], True),
        ("<=", 64, [65, 4096], False),
        ("<", 64, [64, 4096], False),
        ("<", 64, [63, 4096], True),
        (">=", 64, [64, 4096], True),
        (">=", 64, [63, 4096], False),
        (">", 64, [64, 4096], False),
        (">", 64, [65, 4096], True),
        ("==", 64, [64, 4096], True),
        ("==", 64, [65, 4096], False),
    ])
    def test_cmp_branches(self, cmp, value, dims, expect_match):
        rule = self._rule([{"when": {"var": "M", "cmp": cmp, "value": value},
                            "decisions": DEC_512}])
        got = apply_rule(rule, dims)
        assert got == (DEC_512 if expect_match else [])

    def test_fallback(self):
        rule = self._rule([{"when": {"var": "M", "cmp": "<=", "value": 64},
                            "decisions": DEC_512}], fallback=DEC_1024)
        assert apply_rule(rule, [256, 4096]) == DEC_1024

    def test_rule_order_first_match_wins(self):
        rule = self._rule([
            {"when": {"var": "M", "cmp": "<=", "value": 1024},
             "decisions": DEC_512},
            {"when": {"var": "M", "cmp": "<=", "value": 64},
             "decisions": DEC_1024},
        ])
        assert apply_rule(rule, [32, 4096]) == DEC_512

    def test_and_conditions(self):
        rule = self._rule([{
            "when": [{"var": "M", "cmp": "<=", "value": 64},
                     {"var": "N", "cmp": ">=", "value": 8192}],
            "decisions": DEC_1024}])
        assert apply_rule(rule, [64, 8192]) == DEC_1024
        assert apply_rule(rule, [64, 4096]) == []

    def test_matmul_vocabulary(self):
        wmma = [{"kind": "wmma_tile",
                 "params": {"WM": 2, "WN": 4, "WTM": 4, "WTN": 2},
                 "level": 3, "rationale": "reuse"}]
        rule = self._rule([{"when": {"var": "K", "cmp": ">=", "value": 1024},
                            "decisions": wmma}], op="matmul")
        assert apply_rule(rule, [1536, 1536, 1536]) == wmma
        assert apply_rule(rule, [512, 512, 512]) == []

    def test_bad_var_raises(self):
        rule = self._rule([{"when": {"var": "K", "cmp": "<=", "value": 1},
                            "decisions": DEC_512}])  # K invalid for rowwise
        with pytest.raises(ValueError):
            apply_rule(rule, [32, 4096])

    def test_bad_cmp_raises(self):
        rule = self._rule([{"when": {"var": "M", "cmp": "!=", "value": 1},
                            "decisions": DEC_512}])
        with pytest.raises(ValueError):
            apply_rule(rule, [32, 4096])

    def test_bad_dims_raises(self):
        with pytest.raises(ValueError):
            apply_rule({"op": "matmul", "rules": []}, [32, 4096])
        with pytest.raises(ValueError):
            apply_rule({"op": "rmsnorm", "rules": []}, [1, 2, 3])

    def test_empty_rule_table_keeps_default(self):
        assert apply_rule({"op": "softmax", "rules": [],
                           "fallback_decisions": []}, [256, 4096]) == []


# ── prompt no-leak (criteria-4 iron rule) ──────────────────────────────────

class TestPromptNoLeak:
    STRATS = [
        {"op": "rmsnorm", "shape": "32x4096", "role": "explore",
         "decisions": DEC_512,
         "source": {"state_json": "t/rmsnorm_32x4096/state.json",
                    "best_latency_ms": 0.008, "baseline_ratio": 1.23,
                    "extracted_from": "t/rmsnorm_32x4096/state.json::top_level"}},
        {"op": "rmsnorm", "shape": "1024x4096", "role": "explore",
         "decisions": DEC_1024,
         "source": {"state_json": "t/rmsnorm_1024x4096/state.json",
                    "best_latency_ms": 0.110, "baseline_ratio": 1.06,
                    "extracted_from": "t/rmsnorm_1024x4096/state.json::top_level"}},
    ]

    def test_prompt_contains_agent_artifacts_only(self):
        p = build_prompt("rmsnorm", self.STRATS)
        assert "32x4096" in p and "1024x4096" in p
        assert "block_threads" in p
        assert "small rows" in p  # rationale carried through

    def test_prompt_has_no_sweep_leak(self):
        p = build_prompt("rmsnorm", self.STRATS)
        # no sweep file / sweep fields / sweep best labels
        for banned in ("l3_sweep", "best_label", "headroom",
                       "default_ratio", "cudac_us", "sweep"):
            assert banned not in p, f"sweep leak: {banned!r} in prompt"

    def test_prompt_never_reads_l3_sweep_even_if_loaded(self):
        # the sweep's known best labels for rmsnorm gate cases must not appear
        p = build_prompt("rmsnorm", self.STRATS)
        assert "block_threads(512)" not in p  # sweep best_label formatting
        assert "block_threads(1024)" not in p

    def test_load_explore_strategies_skips_rule_files(self, tmp_path):
        (tmp_path / "rmsnorm_32x4096.json").write_text(
            json.dumps(self.STRATS[0]))
        (tmp_path / "rmsnorm_rule.json").write_text(
            json.dumps({"op": "rmsnorm", "rules": []}))
        got = load_explore_strategies("rmsnorm", strategies_dir=tmp_path)
        assert len(got) == 1
        assert got[0]["shape"] == "32x4096"

    def test_generalize_module_never_imports_sweep(self):
        src = (REPO / "benchmarks" / "live" / "generalize_p5s5t.py").read_text()
        assert "l3_sweep" not in src


# ── criteria evaluators ─────────────────────────────────────────────────────

class TestC1:
    def test_pass_all(self):
        rows = [
            {"key": "a", "default_us": 10.0, "agent_us": 9.5,
             "decisions_empty": False},
            {"key": "b", "default_us": 10.0, "agent_us": 10.0,
             "decisions_empty": False},   # equal counts as PASS
            {"key": "c", "default_us": 10.0, "agent_us": 99.0,
             "decisions_empty": True},    # empty decisions PASS regardless
        ]
        r = eval_c1(rows)
        assert r["pass"] is True
        assert all(c["pass"] for c in r["cases"])

    def test_fail_when_agent_slower(self):
        r = eval_c1([{"key": "a", "default_us": 10.0, "agent_us": 10.01,
                      "decisions_empty": False}])
        assert r["pass"] is False

    def test_fail_on_error_row(self):
        r = eval_c1([{"key": "a", "error": "strategies file missing"}])
        assert r["pass"] is False


class TestC2:
    def test_boundary_1p05(self):
        r = eval_c2([{"key": "a", "sweep_default_ratio": 1.285,
                      "agent_us": 105.0, "cudac_us": 100.0}])
        assert r["pass"] is True  # exactly 1.05 passes
        r = eval_c2([{"key": "a", "sweep_default_ratio": 1.285,
                      "agent_us": 105.1, "cudac_us": 100.0}])
        assert r["pass"] is False

    def test_error_fails(self):
        r = eval_c2([{"key": "a", "sweep_default_ratio": 1.285,
                      "error": "not measured"}])
        assert r["pass"] is False


class TestC3:
    def _rows(self, gm_target):
        # single l3 row + single non_l3 row, equal weights -> geomean is
        # sqrt(r1*r2); pick r1=r2=gm_target
        l3 = [{"key": "a", "ratio": gm_target, "weight_us": 100.0}]
        nl3 = [{"key": "b", "ratio": gm_target, "weight_us": 100.0}]
        return l3, nl3

    def test_exactly_0948_passes(self):
        l3, nl3 = self._rows(0.948)
        r = eval_c3(l3, nl3)
        assert r["weighted_geomean"] == pytest.approx(0.948)
        assert r["pass"] is True

    def test_0949_fails(self):
        l3, nl3 = self._rows(0.949)
        assert eval_c3(l3, nl3)["pass"] is False

    def test_weighting_matches_l3_sweep(self):
        # heavy fast case dominates: w1=1000 r=0.9, w2=10 r=1.5
        from math import exp, log
        l3 = [{"key": "a", "ratio": 0.9, "weight_us": 1000.0}]
        nl3 = [{"key": "b", "ratio": 1.5, "weight_us": 10.0}]
        expect = exp((1000 * log(0.9) + 10 * log(1.5)) / 1010)
        r = eval_c3(l3, nl3)
        assert r["weighted_geomean"] == pytest.approx(expect, abs=1e-4)

    def test_missing_row_fails(self):
        l3 = [{"key": "a", "ratio": 0.9, "weight_us": 100.0},
              {"key": "bad", "ratio": None, "weight_us": None,
               "error": "compile fail"}]
        nl3 = [{"key": "b", "ratio": 0.9, "weight_us": 100.0}]
        assert eval_c3(l3, nl3)["pass"] is False

    def test_empty_fails(self):
        assert eval_c3([], [])["pass"] is False

    def test_threshold_constant_locked(self):
        # Leon-approved recalibration 2026-07-22 (Discord ack "2"):
        # 0.940 -> 0.948 after the sweep's rmsnorm@32x4096 headroom was
        # proven a same-cubin phantom. See gate_p5s5t.py threshold block
        # + docs/roadmap/plan.md footnote [double-dagger].
        assert abs(C3_GEOMEAN_MAX - 0.948) < 1e-12

    def test_c2_case_list_locked(self):
        from benchmarks.gate_p5s5t import C2_CASE_KEYS
        # rmsnorm@32x4096 removed (phantom); softmax@1024x4096 remains.
        assert C2_CASE_KEYS == ("softmax@1024x4096",)


class TestC4:
    def _rec(self, decisions, extracted="state.json::top_level"):
        return {"decisions": decisions,
                "source": {"extracted_from": extracted}}

    def test_pass_with_rationale_and_real_state(self):
        recs = [{"key": "a", "record": self._rec(DEC_512)}]
        r = eval_c4(recs, exists_fn=lambda p: True)
        assert r["pass"] is True
        assert r["audit"][0]["n_decisions"] == 1

    def test_empty_decisions_pass(self):
        recs = [{"key": "a", "record": self._rec([])}]
        assert eval_c4(recs, exists_fn=lambda p: True)["pass"] is True

    def test_missing_rationale_fails(self):
        bad = [{"kind": "block_threads", "params": {"n": 512}, "level": 3,
                "rationale": "  "}]
        recs = [{"key": "a", "record": self._rec(bad)}]
        r = eval_c4(recs, exists_fn=lambda p: True)
        assert r["pass"] is False
        assert r["audit"][0]["decisions_missing_rationale"] == [0]

    def test_missing_state_json_fails(self):
        recs = [{"key": "a", "record": self._rec(DEC_512)}]
        assert eval_c4(recs, exists_fn=lambda p: False)["pass"] is False

    def test_missing_strategies_file_fails(self):
        recs = [{"key": "a", "record": None}]
        assert eval_c4(recs, exists_fn=lambda p: True)["pass"] is False

    def test_real_sample_state_exists(self):
        # sanity: existence check against a real file works
        recs = [{"key": "a", "record": self._rec(
            DEC_512, extracted=f"{LIVE_STATE}::top_level")}]
        assert eval_c4(recs)["pass"] is True


class TestC5:
    def test_pass_and_boundary(self):
        rows = [
            {"key": "a", "default_us": 10.0, "agent_us": 10.0,
             "decisions_empty": False},   # exactly 1.00x passes
            {"key": "b", "default_us": 10.0, "agent_us": 5.0,
             "decisions_empty": False},
            {"key": "c", "default_us": 10.0, "agent_us": 999.0,
             "decisions_empty": True},    # keep-default rule = PASS
        ]
        assert eval_c5(rows)["pass"] is True

    def test_slower_fails(self):
        rows = [{"key": "a", "default_us": 10.0, "agent_us": 10.001,
                 "decisions_empty": False}]
        assert eval_c5(rows)["pass"] is False

    def test_rule_missing_is_fail_not_skip(self):
        rows = [{"key": "a", "rule_missing": True}]
        assert eval_c5(rows)["pass"] is False

    def test_error_fails(self):
        rows = [{"key": "a", "error": "apply_rule: bad var"}]
        assert eval_c5(rows)["pass"] is False

    def test_heldout_matrix_locked(self):
        assert ("matmul", [1536, 1536, 1536]) in HELDOUT_MATRIX
        rowwise = [(op, d) for op, d in HELDOUT_MATRIX if op != "matmul"]
        assert len(rowwise) == 6
        assert {tuple(d) for _, d in rowwise} == {(256, 4096), (64, 8192)}


# ── gate plumbing helpers ───────────────────────────────────────────────────

class TestGateHelpers:
    def test_l3_sweep_key_mapping(self):
        assert l3_sweep_key("matmul", [1024, 1024, 1024]) == "matmul@1024x1024"
        assert l3_sweep_key("rmsnorm", [32, 4096]) == "rmsnorm@32x4096"

    def test_dims_from_l3_case(self):
        assert dims_from_l3_case(
            {"op": "matmul", "shape": "1024x1024"}) == [1024, 1024, 1024]
        assert dims_from_l3_case(
            {"op": "softmax", "shape": "32x4096"}) == [32, 4096]

    def test_build_decisions(self):
        decs = build_decisions([
            {"kind": "wmma_tile",
             "params": {"WM": 2, "WN": 4, "WTM": 4, "WTN": 2},
             "level": 3, "rationale": "agent reasoning"},
            {"kind": "block_threads", "params": {"n": 512}, "level": 3,
             "rationale": {"text": "dict-shaped rationale", "lang": "en"}},
        ])
        from arke.ir.strategy import Decision
        assert all(isinstance(d, Decision) for d in decs)
        assert decs[0].kind == "wmma_tile"
        assert decs[0].level == 3
        assert decs[0].rationale.text == "agent reasoning"
        assert decs[1].params == {"n": 512}
        assert decs[1].rationale.text == "dict-shaped rationale"


# ── dry-run smoke (uses a fully-populated fake strategies dir) ──────────────

class TestDryRunSmoke:
    def _populate(self, strategies_dir: Path, state_path: Path):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{}")
        strategies_dir.mkdir(parents=True, exist_ok=True)
        for op, dims in EXPLORE_MATRIX:
            label = "x".join(str(d) for d in dims)
            rec = {"op": op, "shape": label, "role": "explore",
                   "decisions": [] if op == "layernorm" else DEC_512,
                   "source": {"state_json": str(state_path),
                              "best_latency_ms": 0.01, "baseline_ratio": 1.1,
                              "extracted_from": f"{state_path}::top_level"}}
            (strategies_dir / f"{op}_{label}.json").write_text(
                json.dumps(rec))
        for op in ("rmsnorm", "softmax", "layernorm", "matmul"):
            (strategies_dir / f"{op}_rule.json").write_text(json.dumps(
                {"op": op, "rules": [], "fallback_decisions": []}))

    def test_dry_run_ok_with_complete_files(self, tmp_path, monkeypatch):
        import benchmarks.gate_p5s5t as gate

        strategies = tmp_path / "strategies"
        state = tmp_path / "t" / "x" / "state.json"
        self._populate(strategies, state)
        monkeypatch.setattr(gate, "STRATEGIES_DIR", strategies)

        evidence = gate.run_gate(skip_live_measure=True)
        assert evidence["mode"] == "dry-run"
        assert evidence["criteria"]["C4"]["pass"] is True
        assert evidence["structure"]["strategies_missing"] == []
        assert evidence["structure"]["rules_missing"] == []
        assert evidence["dry_run_ok"] is True
        assert evidence["overall"] is None  # no perf verdict in dry-run

    def test_dry_run_flags_missing_rule(self, tmp_path, monkeypatch):
        import benchmarks.gate_p5s5t as gate

        strategies = tmp_path / "strategies"
        state = tmp_path / "t" / "x" / "state.json"
        self._populate(strategies, state)
        (strategies / "matmul_rule.json").unlink()
        monkeypatch.setattr(gate, "STRATEGIES_DIR", strategies)

        evidence = gate.run_gate(skip_live_measure=True)
        assert evidence["structure"]["rules_missing"] == ["matmul"]
        assert evidence["dry_run_ok"] is False

    def test_dry_run_flags_missing_strategy(self, tmp_path, monkeypatch):
        import benchmarks.gate_p5s5t as gate

        strategies = tmp_path / "strategies"
        state = tmp_path / "t" / "x" / "state.json"
        self._populate(strategies, state)
        (strategies / "layernorm_32x4096.json").unlink()
        monkeypatch.setattr(gate, "STRATEGIES_DIR", strategies)

        evidence = gate.run_gate(skip_live_measure=True)
        assert "layernorm@32x4096" in evidence["structure"]["strategies_missing"]
        assert evidence["dry_run_ok"] is False
        assert evidence["criteria"]["C4"]["pass"] is False


# ── run_p5s5t driver plumbing (mocked run_backend) ─────────────────────────

class TestDriverMocked:
    def test_run_case_writes_strategy_from_state(self, tmp_path, monkeypatch):
        import benchmarks.live.run_p5s5t as drv

        runs = tmp_path / "t"
        strategies = tmp_path / "strategies"
        monkeypatch.setattr(drv, "RUNS_DIR", runs)
        monkeypatch.setattr(drv, "STRATEGIES_DIR", strategies)

        state = json.loads(LIVE_STATE.read_text())

        class FakeResult:
            success = True
            message = "ok"

        def fake_run_backend(backend, **kw):
            assert backend == "builtin"
            assert kw["op_name"] == "matmul"
            assert kw["shapes"] == {"A": [1024, 1024], "B": [1024, 1024]}
            assert kw["target_hw"] == "nvidia_ampere"
            out = Path(kw["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "state.json").write_text(json.dumps(state))
            return FakeResult()

        import arke.agent.backends as backends
        monkeypatch.setattr(backends, "run_backend", fake_run_backend)

        path = drv.run_case("matmul", [1024, 1024, 1024],
                            max_turns=5, timeout=60)
        rec = json.loads(path.read_text())
        assert rec["op"] == "matmul"
        assert rec["shape"] == "1024x1024x1024"
        assert rec["role"] == "explore"
        assert len(rec["decisions"]) == 1
        assert rec["decisions"][0]["kind"] == "wmma_tile"
        assert rec["source"]["best_latency_ms"] == pytest.approx(0.41176)

    def test_resume_skips_existing(self, tmp_path, monkeypatch, capsys):
        import benchmarks.live.run_p5s5t as drv

        strategies = tmp_path / "strategies"
        strategies.mkdir(parents=True)
        for op, dims in EXPLORE_MATRIX:
            label = "x".join(str(d) for d in dims)
            (strategies / f"{op}_{label}.json").write_text("{}")
        monkeypatch.setattr(drv, "STRATEGIES_DIR", strategies)

        rc = drv.main([])   # everything skipped, no live calls
        assert rc == 0
        out = capsys.readouterr().out
        assert out.count("skip ") == len(EXPLORE_MATRIX)
