# Arke Spec Completeness Audit — Stage 7 Track 2

> **Date:** 2026-04-26  
> **Purpose:** Audit README capability claims against current spec/docs completeness  
> **Status:** ✅ Mostly complete — 7/8 core capabilities explicitly documented; 1 capability remains Phase 2 expansion work

---

## Executive Summary

README 声称 8 个核心能力。当前 spec/docs 覆盖情况如下：

| # | 能力 | 状态 | 文档 | 备注 |
|:--|:-----|:----:|:-----|:-----|
| 1 | LLM-Native Language | ✅ 完整 | `arke-lang-spec.md` | — |
| 2 | Semantic/Strategy Separation | ✅ 完整 | `arke-ir-spec.md` | — |
| 3 | Minimal-Token E2E | ✅ 完整 | `e2e-flow.md` + `token-efficiency-analysis.md` | 已补足 token 量化与预算系统 |
| 4 | Bounded Action Space | ✅ 完整 | `agent-design.md` | — |
| 5 | @rationale Annotations | ✅ 完整 | `arke-lang-spec.md` + `arke-ir-spec.md` | — |
| 6 | Compiler-as-Verifier | ✅ 完整 | `arke-compiler-infrastructure.md` + pass/validator docs | — |
| 7 | Structured LLM-Compiler Protocol | ✅ 完整 | `agent-design.md` | — |
| 8 | Multi-Hardware | 🟨 继续扩展 | `arke-ir-spec.md` + backend notes | Phase 2 / 3 的后续设计项 |

**结论：**
- ✅ Stage 7 所需的核心 Lang / IR / Compiler / Benchmark 文档已经齐备
- ✅ Minimal-Token 卖点现在有定量依据
- ✅ Pass / Symbolic-Dimension 规范已补齐
- 🟨 Multi-Hardware 仍然保留为后续 Phase 2 / 3 的扩展工作，不阻塞 Stage 7 关闭

---

## 1. LLM-Native Language ✅

**README 声称：**
> Semantic/Strategy Separation — "What to compute" and "how to optimize" are independent layers

**现有文档：**
- `docs/spec/arke-lang-spec.md`
- `docs/architecture/naming-system.md`
- `docs/spec/arke-lang-vs-python-triton.md`

**覆盖度：** 100%

---

## 2. Semantic/Strategy Separation ✅

**README 声称：**
> enabling LLMs to explore strategies without risking correctness

**现有文档：**
- `docs/spec/arke-ir-spec.md`
- `docs/architecture/e2e-flow.md`
- `docs/phase1/dynamic-shape-feasibility.md`

**覆盖度：** 100%

---

## 3. Minimal-Token End-to-End ✅

**README 声称：**
> The entire pipeline consumes an order of magnitude fewer tokens than direct code generation

**现有文档：**
- `docs/architecture/e2e-flow.md`
- `docs/architecture/agent-design.md`
- `docs/architecture/token-efficiency-analysis.md`

**补充完成的内容：**
- baseline vs Arke token 消耗分解
- 实例化的对比分析
- `OptimizationBudget` token 预算系统

**覆盖度：** 100%

---

## 4. Bounded Action Space ✅

**README 声称：**
> LLMs select from compiler-enumerated legal actions, not free-form code

**现有文档：**
- `docs/architecture/agent-design.md`
- `docs/spec/arke-ir-spec.md`
- `docs/phase1/stage7-plan.md`

**覆盖度：** 100%

---

## 5. @rationale Annotations ✅

**README 声称：**
> Every optimization decision carries a natural language explanation

**现有文档：**
- `docs/spec/arke-lang-spec.md`
- `docs/spec/arke-ir-spec.md`
- `docs/architecture/agent-design.md`

**覆盖度：** 100%

---

## 6. Compiler-as-Verifier ✅

**README 声称：**
> The compiler validates every LLM decision through progressive checks

**现有文档：**
- `docs/architecture/arke-compiler-infrastructure.md`
- `docs/spec/pass-infrastructure-spec.md`
- `docs/spec/symbolic-dimension-spec.md`
- `arke/compiler/validator.py`
- `arke/compiler/semantic_passes.py`

**覆盖度：** 100%

---

## 7. Structured LLM-Compiler Protocol ✅

**README 声称：**
> LLM and compiler interact through a closed-loop tool-use API

**现有文档：**
- `docs/architecture/agent-design.md`
- `docs/architecture/e2e-flow.md`
- `docs/architecture/naming-system.md`

**覆盖度：** 100%

---

## 8. Multi-Hardware 🟨

**README 声称：**
> Single kernel definition targets NVIDIA, Ascend, and beyond; strategy adapts per hardware

**现有文档：**
- `docs/spec/arke-ir-spec.md`
- `docs/architecture/arke-compiler-infrastructure.md`
- backend abstraction notes in the compiler stack

**当前结论：**
- Stage 7 需要的 backend boundary 已经明确
- Phase 2 / 3 的 Ascend / AMD-specific design 仍属于后续扩展
- 这不阻塞 G7 / Stage 7 关闭，但保留为下一阶段的设计工作

**覆盖度：** 仍在扩展中

---

## Stage 7 影响判断

### 已满足的 Stage 7 关键文档面

- Lang v0.1.0 语法与示例
- IR v0.1.0 分层与 symbolic shape
- Pass / validator 规范
- Token efficiency 量化
- BL5 coverage ledger / audit / dashboard
- Gate evidence 与 Track 6 artifact contract
- Stage 7 conformance checklist

### 仍需保留的 Phase 2 / 3 扩展

- backend abstraction 的更完整跨硬件设计
- Ascend backend design
- AMD backend design
- MLIR / LLVM deeper interoperability design

---

## 结论

Stage 7 相关文档与实现现在已经足够支撑：

- G7 gate evidence 记录
- BL5 L1 / L2 benchmark artifact contract
- Lang / IR v0.1.0 的主线开发
- 后续 S8 / S9 的继续推进

**本次审视结论：Stage 7 的核心文档闭环已完成；Multi-Hardware 仍作为后续阶段的扩展项。**

---

*版本：v0.1.0 | 创建：2026-04-26 | 审视者：Kitty*
