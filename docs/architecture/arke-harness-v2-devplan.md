# Arke Harness v2 — 最佳目标态开发计划 (Development Plan)

> **Status:** 🚧 IN PROGRESS — Leon 授权 "不考虑 Token，按最佳目标态拆解开发和测试任务" (2026-06-26)
> **Companion:** `arke-harness-v2-rfc.md` (proposal) · `arke-harness-handbook.md` (manual)
> **Governance:** Façade v1.0 stays frozen (Locked Principle #4). Breaking
> items (B1/B2) remain held for an explicit v2.0 decision — the target state
> below is reached **without** breaking the frozen contract, using additive
> 1.x surfaces and Substrate-only changes.

---

## 0. 目标态定义 (Definition of "best target state")

A Harness where an LLM agent can **autonomously generate and tune real Triton
GPU kernels end-to-end**, with production-grade runtime ergonomics matching
the four mature agent harnesses (Hermes / OpenClaw / Codex / Claude Code):

1. **Legal actions are compiler/HW-truth** (no illegal move ever surfaces).
2. **Live LLM loop is robust** — fallback chain, retry, resumable, degrades to
   heuristic floor, never produces *no* kernel.
3. **Scales** — async streaming, concurrent read tools, context compaction.
4. **Reaches the ecosystem** — MCP server so any MCP client drives Arke.
5. **Extensible at runtime** — skills, hooks, subagents wired, not just specced.
6. **Proven on real GPU** — live trajectories with real `baseline_ratio` on
   matmul / rmsnorm / (stretch) flash_attention, recorded as evidence.
7. **Documented** — handbook reflects every shipped capability with real numbers.

---

## 1. 任务依赖图 (Dependency DAG)

```
P0 (robustness)         P1 (scale/reach)        P2 (extension runtimes)
─────────────           ────────────────        ───────────────────────
S1 ✅ HW-legal           N1 AsyncGen loop ──┐     N4 Skills runtime
S3 ✅ fallback           N2 concurrent tools │     N5 Hooks runtime
S4 ✅ rationale          S5 compaction ──────┼──►  N6 Subagent sweep
S2 ⬜ resume  ───────────┘                    │
                         N3 MCP server ◄──────┘  (streams via N1)

LIVE (evidence, cross-cutting — needs P0 done):
  L1 live matmul     L2 live rmsnorm    L3 (stretch) live flash_attention
  L4 @rationale KB refresh from live trajectories
```

**Ordering rule:** P0 fully closed → LIVE evidence (validates P0 on real GPU) →
P1 (scale, informed by live-loop pain points) → P2 (extension runtimes) →
final handbook + KB refresh.

---

## 2. 任务卡 (Task cards — dev + test, with acceptance gates)

Each card: **Dev** (what to build) · **Test** (how it's verified) · **Done**
(acceptance gate). Status: ✅ done · 🚧 active · ⬜ todo.

### Phase A — P0 closure

#### A1. S1 HW-legal actions — ✅ DONE (`36bb4b2`)
- Dev: `place(shared)` capacity legality via `_tensor_bytes` × `hw.shared_memory_bytes`.
- Test: `test_legal_actions_shape_aware.py` +3 (oversized dropped / fits kept / unknown both).
- Done: 8/8 pass.

#### A2. S4 @rationale enforced — ✅ DONE (`36bb4b2`)
- Dev: `ApplyDecisionTool.execute` rejects level≥1 empty-rationale.
- Test: `test_facade_tools_f12.py` +2; 6 existing updated.
- Done: 115/115 pass.

#### A3. S3 provider fallback + retry — ✅ DONE (`6b2684a`)
- Dev: `LLMConfig.fallback` + `provider_chain` + `_call_llm_resilient`.
- Test: `test_runner_resilience.py` 16.
- Done: 16/16 pass.

#### A4. S2 resumable trajectory — ✅ DONE (`<pending>`)
- Dev: `OptimizationState.from_dict` (inverse of to_dict, reuses ScheduleIR.from_dict / _parse_decision / CompileResult rebuild); `LLMRunner.optimize(resume_from=, state_out=)` — dumps `state.json`, rehydrates spent budget, records resume provenance in session_summary.
- Test: `test_runner_resume.py` 5 (round-trip / partial-dict / state_out write / resume rehydrates budget / missing-file fresh).
- Done: 5/5 pass.

#### A5. S4b @rationale trajectory assertion — ✅ DONE (`<pending>`)
- Dev: `trajectory.audit_decision_rationales(path)` — additive gate-style audit (every `decision` record has non-empty rationale); does NOT touch frozen events.validate_payload.
- Test: `test_trajectory_rationale_contract.py` 5 (clean / missing / empty / non-decision-ignored / missing-file).
- Done: 5/5 pass.

### Phase B — LIVE evidence (real GPU, real LLM)

> Needs P0 done. Uses yunwu `/v1` (token cost OK per Leon). 6 GB VRAM → small shapes.

#### B1. L1 live matmul autotuning — ✅ DONE (`019ff7f`)
- Dev: `benchmarks/live/run_live_optimize.py` op-agnostic driver (state/result/trajectory/evidence).
- Result (matmul 512³, yunwu/claude-sonnet-4-6): 10 decisions 10/10 @rationale (A5 clean), 25 tool calls, **2 real GPU profiles — best latency 0.0797 ms (baseline_ratio 0.4035), backend=triton, correct=True**. S2 state.json dumped + verified rehydratable (10 dec / 3 compiles).
- Done: evidence card under `benchmarks/results/phase1/harness_v2/live/matmul/`.

#### B2. L2 live rmsnorm autotuning — ✅ DONE (`<pending>`)
- Dev: same driver. **Pitfall found + fixed:** rmsnorm needs `X[M,N]` + `W[N]`; the driver's `_shapes_for` initially only passed X → ArkeEnv default-filled W=[4,8] → SemanticInterpreter ref mismatch (4096 vs 8) → verify failed, baseline_ratio=None despite a real 0.21 ms profile. Fixed `_shapes_for` to emit W=[N] (and layernorm B=[N]).
- Result (rmsnorm 2048², yunwu/claude-sonnet-4-6): 1 decision 1/1 @rationale (A5 clean), 25 tool calls, **3 real GPU profiles — best latency 0.0680 ms, backend=triton, correct=True**.
- Done: evidence card under `benchmarks/results/phase1/harness_v2/live/rmsnorm/`.

#### B3. L3 (stretch) live flash_attention — ⬜ TODO
- Same for `flash_attention`, small head/seq to fit 6 GB; OOM recorded non-blocking.
- Done: evidence card OR documented OOM with the shape that fit.

#### B4. L4 @rationale KB refresh from live — ⬜ TODO
- Dev: extend `benchmarks/build_rationale_kb` to also mine live trajectories
  (real baseline_ratio paired rationales), append to `data/rationale_kb.jsonl`.
- Test: KB entry count grows; each live entry has source="live".
- Done: KB has ≥1 live-sourced entry per live op.

#### B5. Harness fixes found during live (from B1/B2) — ⬜ TODO
- **F1 best_result has no latency:** in B1, `best_result` came from the verify
  path (correct=True, V1_triton) but carried no `latency_ms`/`baseline_ratio`,
  so `session_summary["best_performance"]` couldn't surface the winning ratio
  (evidence card now extracts best from the action trajectory as a workaround).
  Fix: `OptimizationState.record_compile` should prefer a profile result with
  latency over a verify-only result when updating best_result.
- **F2 LLM never self-terminates:** both B1/B2 hit `max_turns` (LLM keeps
  applying decisions, never stops). Fix: strengthen the system prompt to make
  the model compare `baseline_ratio` after each profile and STOP once it has a
  best + N non-improving cycles (already half-specced in the prompt; tighten).

### Phase C — P1 scale & reach

#### C1. N1 streaming event callback — ✅ DONE (`<pending>`)
- Dev: `optimize(on_event=callback)` — each action is streamed to the consumer as it lands (real-time progress / live dashboards / hook substrate). Non-breaking: sync `optimize` signature preserved; on_event optional, errors in callback isolated. (Full AsyncGenerator deferred — the callback covers the streaming-consumption need without rewriting the proven sync loop.)
- Test: `test_runner_concurrency.py::test_on_event_callback_receives_each_action`.
- Done: green.

#### C2. N2 concurrent read-tool partitioning — ✅ DONE (`<pending>`)
- Dev: the optimize loop now runs `registry.partition_for_execution` over each turn's tool calls — `concurrent_safe` batches (get_hw_profile / analyze_compute / list_legal_actions) execute together in a ThreadPoolExecutor; mutating/compile tools force their own serial batch. `concurrent_tools=False` opt-out. Trajectory ordering preserved.
- Test: `test_runner_concurrency.py` (partition groups concurrent-then-serial; order preserved with concurrency; opt-out path).
- Done: 5 tests green.

#### C3. S5 context compaction + delta observe — ⬜ TODO
- Dev: predictive (token-threshold) + reactive message-log compaction; an
  `observe(delta=true)`-style result-shrinking for repeated state reads.
- Test: `test_compaction.py` — a long synthetic message log compacts below
  threshold while preserving the last-N turns + the ground-truth state pointer.
- Done: green; a 30-turn live run stays within context (manual check in B-phase).

#### C4. N3 MCP server (Mode C) — ✅ DONE (`<pending>`)
- Dev: `arke/agent/mcp_server.py` — **zero-dependency** JSON-RPC-2.0-over-stdio MCP server (no `mcp` SDK dep). Methods: initialize / tools/list / tools/call / ping / notifications. Exposes the 8 frozen Façade tools env-bound. CLI: `arke mcp serve --kernel <op> [--shape] [--target]`.
- Test: `test_mcp_server.py` 8 (initialize / tools-list=8 / tools-call hw_profile + legal_actions / unknown-tool error / unknown-method -32601 / notification→None / serve_stdio round-trip) + real subprocess stdio smoke.
- Done: 8/8 pass; any MCP client (Hermes/Cline/Claude Desktop) can now drive Arke's 8 tools directly.

### Phase D — P2 extension runtimes (`arke/agent/extensions.py`)

#### D1. N4 Skills runtime — ✅ DONE (`<pending>`)
- Dev: `Skill` + `load_skill` / `load_skills_dir` (dependency-free SKILL.md frontmatter+body parse) + `skills_prompt_block` (renders recipes into a system-prompt addendum, truncates long bodies). `optimize(skills=[...])` injects them.
- Test: `test_extensions.py` (parse frontmatter+body / load dir skips bad / prompt block renders+truncates / end-to-end injected into system prompt).
- Done: green.

#### D2. N5 Hooks runtime — ✅ DONE (`<pending>`)
- Dev: `HookRegistry` with `PreDecision / PostCompile / PostProfile / OnRollback`. PreDecision MAY veto an apply_decision (return False → tool result `vetoed`); Post* are observation; hook exceptions isolated. `optimize(hooks=registry)` wires them into the tool-exec path.
- Test: `test_extensions.py` (predecision veto / observation never vetoes / errors isolated / unknown point rejected / end-to-end veto of apply_decision via runner → decisions==0).
- Done: green.

#### D3. N6 Subagent design-space sweep — ✅ DONE (`<pending>`)
- Dev: `sweep_design_space(op, shapes, variants)` — each variant runs on its OWN ArkeEnv (isolated budget/state) in a ThreadPoolExecutor; a failing fork can't corrupt siblings; returns `(best, all)` = lowest-latency correct variant. Caller replays `best.decisions`.
- Test: `test_extensions.py` (ranking picks lowest-latency correct + failed fork isolated; no-correct→None). **Real GPU verified manually:** matmul 256² tile128 0.147ms beats tile64 0.230ms, both correct.
- Done: green + real-GPU confirmation.

### Phase E — finalization

#### E1. Handbook real-number refresh — ⬜ TODO
- Dev: fill §6/§9 with real live `baseline_ratio` from B-phase; add Mode C, skills,
  hooks, subagents sections from C/D; refresh troubleshooting with live findings.
- Done: handbook has zero "proposed/planned" placeholders for shipped features.

#### E2. Full regression + RFC status sync — ⬜ TODO
- Dev: run full `pytest tests/` (minus known-long); mark every landed RFC item ✅;
  update `arke-harness.md §18` status matrix.
- Done: full suite green; RFC + arke-harness.md + this plan all consistent.

---

## 3. 验收总闸 (Master acceptance)

The target state is reached when **all** hold:
- Phase A–E cards ✅.
- Full `pytest tests/` green (0 regressions vs current 451+ agent-slice baseline).
- ≥2 live evidence cards (matmul + rmsnorm) with real `baseline_ratio`.
- Façade v1.0 frozen contract test still passes unchanged (no breaking change).
- Handbook documents every shipped capability with real numbers.
- `docs/architecture/{arke-harness.md, arke-harness-v2-rfc.md}` status synced.

---

## 4. 执行顺序 (Execution order — one commit per card, push each)

```
A4 → A5 → B1 → B2 → B3 → B4 → C1 → C2 → C3 → C4 → D1 → D2 → D3 → E1 → E2
```

Reversible work proceeds autonomously. The only HOLD points (need Leon) remain
the one-way doors: a Façade major bump (B1/B2 breaking items — NOT in this
plan) and any locked-Gate threshold change (none here).

---

*Dev plan v1 — 2026-06-26. Tracks the RFC's target state to closure.*
