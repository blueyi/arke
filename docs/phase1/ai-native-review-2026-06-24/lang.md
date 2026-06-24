# Arke-Lang (.ak) AI-Native / LLM-友好度 架构审视

> 2026-06-24 AI-Native 审视批次之一(Lang 层)。透镜:假设 .ak 主要使用者是 LLM/Agent 而非人类。
> 配套:`ir-compiler.md`(IR+Compiler)、`harness.md`(Harness/Agent)。

## 审视方法

读取文法(`arke/lang/arke.lark`)、parser/transformer(`arke/lang/grammar.py`)、错误处理调用点(`arke/compiler/pipeline.py`、`scripts/check_backend_agnostic.py`)、多个 `.ak` 示例(elementwise/matmul/layernorm/flash-attn/MLA/production ST4/L2 fused),及 spec(`docs/spec/arke-lang-spec.md`、`arke-lang-vs-python-triton.md`)。并在 venv 中**实跑 parser** 取得真实报错文本作为 Q3 证据。

项目自我宣称(`docs/spec/arke-lang-spec.md:65-66`):"**LLM-Native** — Regular, simple, unambiguous grammar." / "**Token efficient** — Shorter than equivalent Triton."

## Q1. token 密度是否够紧凑?—— 部分不合理
**证据**:`00_relu.ak:9-16` 单算子核需 `kernel(签名)+where+{let Y=relu(X=X); return Y;}`;`tensor_type`(`arke.lark:48`)用 `Tensor<[...],dtype>` 三重包装;入参/返回重复写同一 shape(`00_relu.ak:10` vs `:11`),而 `_` 推断(`arke.lark:30`)几乎无人用;全命名参在单参算子产生 `X=X` 冗余。
**诊断**:比 Triton 短(spec 对比成立),但仍有纯样板/重复 token 可削。
**建议**(均**不动 Gate/Benchmark**,纯语法糖/文档层):默认 `_` 返回推断、引入"表达式核"糖免去 `let/return`。

## Q2. 是否易被 LLM 可靠生成?—— 基本合理,一处隐患
**证据(正面)**:LALR(1) 无二义(`grammar.py:58`)、全命名参消除位置歧义(`arke.lark:88`)、dtype 封闭枚举(`arke.lark:57-70`)、strategy 统一 `directive(kwargs)@rationale` 模式。
**证据(隐患)**:`where` 维度名与签名 shape **无绑定**——`00_relu.ak:12` 声明 `where B/S/D`,签名却是字面量 `[128,3072]`,三个维度名根本没出现在任何 shape,仍能 parse 通过(`where_clause`/`dim_list` 解耦,`arke.lark:33,50-52`)。"沉默成功但语义错位"是 Agent 陷阱。
**建议**:parser/converter 加语义校验(where 维度名↔shape 符号一致性)。benchmark 示例已满足,**不动 Gate**。

## Q3. parser 报错对 LLM 友好吗?—— 【最不合理】
**证据**:`grammar.py:478-480` 裸抛 Lark 异常,调用方仅 `str(e)`(`pipeline.py:239-240`)。实跑三类典型 Agent 错误:
- 漏分号 → `Unexpected token Token('RETURN','return')... Expected: SEMICOLON`
- 位置参 `relu(X)` → `Unexpected token Token('RPAR',')')... Expected: EQUAL`
- 错 dtype `float16` → 裸列 `U64 I8 BF16 I32...F16` 14 个枚举,不提示"应为 f16"

**诊断**:纯传统 LALR 报错——用内部 token 名(`RPAR/SEMICOLON/F16`)而非源面量、无源码行+caret、无修正建议、无近似匹配。直接掐断 Agent 的 compile→fix 自纠环,与 spec:65 "LLM-Native" 落差最大。
**建议**(**不动 Gate**):包 `ArkeSyntaxError`,输出 `{line,col,源码片段+caret,expected(回译为字面量),got,suggestion}`;内置 fix-hint 表(漏 `;`、需命名参、`float16→f16` 近似)。

## Q4. 是否沿用人类语言习惯?—— 部分不合理
**证据**:spec 定位 "**human- and LLM-facing**"(`arke-lang-spec.md:31`,人类与 LLM 并列而非 Agent-first);人类修辞关键字 `for target(...)`(`arke.lark:100`)、`when/otherwise`(`:122-124`)、`and/or`(`:126-127`)、`as/where`;示例头部大量自然语言注释(`00_relu.ak:1-7`)会被模型当生成样式模仿;`@rationale` 强制自由文本(`01_matmul.ak:24`)是有意人类可读优先,但自由文本不可机器校验。
**建议**:`for` 设为可选、连词允许符号别名(纯糖,**不动 Gate**);`@rationale` 若要机器可校验需引入受控标签——**会触及 StrategyIR rationale 语义/可能的 Gate 校验,需先确认**,建议作可选扩展。

## Top-3 不合理点排序
1. **parser 错误信息纯传统 LALR 风格、对 Agent 不可操作**(Q3)—— 内部 token 名、无源码片段、无修正建议;掐断 compile→fix 自纠环。修复成本低、收益最高,**不动 Gate**。
2. **`where` 与签名 shape 无绑定校验,制造沉默语义错位**(Q2)—— Agent 能生成过 parse 但语义错的核,隐蔽高危。加语义校验,**不动 Gate**。
3. **token 冗余:返回重复 shape + 单算子 `let/return` 样板 + 人类糖关键字**(Q1/Q4)—— 与 spec:66 "Token efficient" 有量化差距。全为语法糖层,**不动 Gate/Benchmark**。

## 值得肯定
LALR(1) 确定性文法 + 全命名参数 + 封闭 dtype 枚举(可生成性/无歧义已做对);semantic/strategy 分块 + `directive(kwargs)@rationale` 统一模式(利于模式化生成)——是好的 AI-native 基础。

> 注:任务上下文称 parser 在 `arke/parser/`,实际不存在;Lark parser 与 AST 均在 `arke/lang/`。
