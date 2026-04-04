# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""G6: Arke Lang & IR Completeness gate runner.

Imported by benchmarks/gate.py — kept in a separate module to avoid
bloating the main gate file.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.gate import GateResult, GateSummary


def run_g6(tier: int = 2) -> GateSummary:  # noqa: ARG001
    """G6: Arke Lang & IR Completeness (Key Features).

    G6.1  .ak → SemanticIR + StrategyIR → Triton → GPU E2E pipeline
    G6.2  ast_to_strategy() converter implemented
    G6.3  @rationale preserved through full pipeline
    G6.4  Token efficiency: .ak code lines < Triton code lines
    G6.5  Python interop: IR ↔ JSON round-trip (all OP_CATALOG ops)
    G6.6  IR-MLIR mapping documented (docs/spec/ir-mlir-mapping.md)
    G6.7  Grammar: all .ak files parse without errors
    G6.8  Cat A+B+C+D operators: .ak expressible + GPU correctness verified
    G6.9  Language Spec v1.0 + IR Spec v1.0 present
    """
    from arke.compiler.ast_to_strategy import ast_to_strategy
    from arke.ir.builder import KernelBuilder
    from arke.ir.ops.catalog import OP_CATALOG
    from arke.ir.semantic import SemanticIR
    from arke.ir.strategy import Decision, Rationale, StrategyIR
    from arke.parser.parser import parse_file
    from arke.pipeline import ArkePipeline
    results: list[GateResult] = []
    repo_root = Path(__file__).parent.parent
    ak_dir = repo_root / "docs" / "examples"
    ak_files = sorted(ak_dir.glob("*.ak"))

    # ── G6.7: All .ak files parse ────────────────────────────────────────
    ak_pass = 0
    ak_fail_msgs: list[str] = []
    for f in ak_files:
        try:
            prog = parse_file(str(f))
            assert prog.kernels, f"No kernels in {f.name}"
            ak_pass += 1
        except Exception as e:
            ak_fail_msgs.append(f"{f.name}: {e}")

    results.append(GateResult(
        "G6", "G6.7", "Grammar: all .ak files parse", "function",
        not ak_fail_msgs,
        f"{ak_pass}/{len(ak_files)} parse OK"
        + (f" FAIL: {ak_fail_msgs[:2]}" if ak_fail_msgs else ""),
    ))

    # ── G6.2: ast_to_strategy() converter ───────────────────────────────
    strat_ok = 0
    strat_total = 0
    strat_fail_msgs: list[str] = []
    for f in ak_files:
        prog = parse_file(str(f))
        if not prog.strategies:
            continue
        strat_total += 1
        try:
            ir = ast_to_strategy(prog.strategies[0])
            assert ir.decision_count >= 1
            strat_ok += 1
        except Exception as e:
            strat_fail_msgs.append(f"{f.name}: {e}")

    results.append(GateResult(
        "G6", "G6.2", "ast_to_strategy() converter", "function",
        not strat_fail_msgs and strat_total > 0,
        f"{strat_ok}/{strat_total} strategy defs converted OK",
    ))

    # ── G6.3: @rationale preserved through pipeline ──────────────────────
    rat_ok = False
    rat_msg = "No @rationale found in any .ak file"
    for f in ak_files:
        prog = parse_file(str(f))
        if not prog.strategies:
            continue
        for action in prog.strategies[0].actions:
            if action.annotation and action.annotation.key == "rationale":
                ir = ast_to_strategy(prog.strategies[0])
                for d in ir.decisions:
                    if d.rationale and d.rationale.text:
                        rat_ok = True
                        rat_msg = f"@rationale preserved: '{d.rationale.text[:50]}'"
                        break
                if rat_ok:
                    break
        if rat_ok:
            break

    results.append(GateResult(
        "G6", "G6.3", "@rationale preserved through pipeline", "function",
        rat_ok, rat_msg,
    ))

    # ── G6.1 / G6.8: E2E pipeline + OP_CATALOG full coverage ──────────────
    # Build a map: op_name → .ak file that covers it
    op_to_file: dict[str, str] = {}
    e2e_ok = 0
    e2e_fail_msgs: list[str] = []
    # Also track token efficiency per file
    tok_pass = 0
    tok_total = 0
    tok_fail_msgs: list[str] = []

    for f in ak_files:
        try:
            # Parse to find which ops this file covers
            prog = parse_file(str(f))
            for kernel in prog.kernels:
                for stmt in kernel.body:
                    if hasattr(stmt, 'op_call'):
                        op_name = stmt.op_call.name
                        if op_name in OP_CATALOG:
                            op_to_file[op_name] = f.name
            # Also scan nodes from SemanticIR
            res = ArkePipeline.from_ak_file(str(f), target_hw="nvidia_ampere",
                                            codegen=True)
            assert res.correct, "numerical check failed"
            e2e_ok += 1
            # Register ops from SemanticIR
            sem = res.semantic_ir
            for node in sem.get("nodes", []):
                op_name = node.get("op", "")
                if op_name in OP_CATALOG:
                    op_to_file[op_name] = f.name
            # Token efficiency
            if res.codegen_source:
                ak_code = sum(
                    1 for l in f.read_text().splitlines()
                    if l.strip() and not l.strip().startswith("//")
                )
                triton_code = sum(
                    1 for l in res.codegen_source.splitlines()
                    if l.strip() and not l.strip().startswith("#")
                )
                tok_total += 1
                if ak_code < triton_code:
                    tok_pass += 1
                else:
                    tok_fail_msgs.append(
                        f"{f.name}: .ak={ak_code} >= triton={triton_code}"
                    )
        except Exception as e:
            e2e_fail_msgs.append(f"{f.name}: {str(e)[:60]}")

    results.append(GateResult(
        "G6", "G6.1", ".ak → SemanticIR+StrategyIR → GPU E2E", "function",
        e2e_ok == len(ak_files) and not e2e_fail_msgs,
        f"{e2e_ok}/{len(ak_files)} .ak files E2E OK, correct=True"
        + (f" FAIL: {e2e_fail_msgs[:2]}" if e2e_fail_msgs else ""),
    ))

    # G6.8: ALL OP_CATALOG ops covered by some .ak file
    missing_ops = [op for op in OP_CATALOG if op not in op_to_file]
    results.append(GateResult(
        "G6", "G6.8",
        "ALL OP_CATALOG ops: .ak expressible + GPU correct",
        "accuracy",
        not missing_ops,
        f"{len(op_to_file)}/{len(OP_CATALOG)} ops covered"
        + (f" MISSING: {missing_ops}" if missing_ops else
           f" — all {len(OP_CATALOG)} ops verified correct"),
    ))

    # G6.4: ALL .ak files: .ak code lines < generated Triton code lines
    results.append(GateResult(
        "G6", "G6.4",
        "Token efficiency: .ak lines < Triton lines (all ops)",
        "performance",
        tok_pass == tok_total and tok_total > 0 and not tok_fail_msgs,
        f"{tok_pass}/{tok_total} files: .ak lines < generated Triton lines"
        + (f" FAIL: {tok_fail_msgs[:2]}" if tok_fail_msgs else ""),
    ))

    # ── G6.5: IR ↔ JSON round-trip (all OP_CATALOG ops) ──────────────────
    _OP_SHAPES: dict[str, dict[str, list[int]]] = {
        "batch_matmul": {"A": [4, 32, 64], "B": [4, 64, 32]},
        "layernorm":    {"X": [32, 64], "W": [64]},
        "rmsnorm":      {"X": [32, 64], "W": [64]},
        "transpose":    {"X": [32, 64]},
        "matmul":       {"A": [32, 64], "B": [64, 32]},
    }
    rt_ok = 0
    rt_fail = 0
    for op_name in sorted(OP_CATALOG.keys()):
        try:
            b = KernelBuilder(f"test_{op_name}")
            op_def = OP_CATALOG[op_name]
            custom = _OP_SHAPES.get(op_name, {})
            kwargs: dict = {}
            for inp in op_def.inputs:
                b.param(inp, custom.get(inp, [64, 64]), "f16")
                kwargs[inp] = inp
            nid = b.op(op_name, **kwargs)
            b.returns(nid, b._params[0].shape, "f16")
            ir = b.build()
            d1 = ir.to_dict()
            ir2 = SemanticIR.from_dict(json.loads(json.dumps(d1)))
            d2 = ir2.to_dict()
            assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
            rt_ok += 1
        except Exception:
            rt_fail += 1

    # StrategyIR round-trip
    try:
        s = StrategyIR(kernel_id="k", target_hw="nvidia_ampere")
        s.tile("M", [64], "test")
        s.add_decision(Decision(kind="fuse", params={"ops": ["a", "b"]},
                                rationale=Rationale("saves mem")))
        s2 = StrategyIR.from_dict(json.loads(json.dumps(s.to_dict())))
        assert s2.decisions[1].rationale is not None
        assert s2.decisions[1].rationale.text == "saves mem"
        rt_ok += 1
    except Exception:
        rt_fail += 1

    results.append(GateResult(
        "G6", "G6.5", "IR ↔ JSON round-trip (all OP_CATALOG ops)", "function",
        rt_fail == 0,
        f"{rt_ok}/{rt_ok+rt_fail} round-trip OK "
        f"(SemanticIR×{len(OP_CATALOG)} + StrategyIR)",
    ))

    # ── G6.6: IR-MLIR mapping documented ─────────────────────────────────
    mlir_doc = repo_root / "docs" / "spec" / "ir-mlir-mapping.md"
    mlir_ok = mlir_doc.exists() and mlir_doc.stat().st_size > 1000
    results.append(GateResult(
        "G6", "G6.6", "IR-MLIR mapping document", "function",
        mlir_ok,
        f"docs/spec/ir-mlir-mapping.md "
        f"({'present' if mlir_ok else 'MISSING or empty'}, "
        f"{mlir_doc.stat().st_size if mlir_doc.exists() else 0}B)",
    ))

    # ── G6.9: Language Spec v1.0 + IR Spec v1.0 ──────────────────────────
    lang_spec = repo_root / "docs" / "spec" / "arke-lang-spec-v1.md"
    ir_spec = repo_root / "docs" / "spec" / "arke-ir-spec-v1.md"
    spec_ok = (
        lang_spec.exists() and lang_spec.stat().st_size > 1000
        and ir_spec.exists() and ir_spec.stat().st_size > 1000
    )
    results.append(GateResult(
        "G6", "G6.9", "Language Spec v1.0 + IR Spec v1.0 frozen", "function",
        spec_ok,
        "arke-lang-spec-v1.md + arke-ir-spec-v1.md: "
        + ("both present" if spec_ok else "one or more MISSING"),
    ))

    return GateSummary(
        gate="G6",
        results=results,
    )
