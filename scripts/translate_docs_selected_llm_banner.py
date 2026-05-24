#!/usr/bin/env python3
"""Translate selected docs/*.md to docs_zh/ with LLM-style banner and term protection."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from deep_translator import GoogleTranslator, MyMemoryTranslator

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS_ZH = ROOT / "docs_zh"

BANNER = (
    "> 中文译本（LLM 翻译），仅供阅读参考；规范与验收以 `docs/` 英文原文为准。"
    "专有名词（Arke、Harness、Token、Claude Code、Semantic IR、Strategy IR、"
    "Bounded Action Space、@rationale 等）保留英文。\n\n"
)

# Order matters: longer phrases first
TERM_PLACEHOLDERS: list[tuple[str, str]] = [
    ("Arke Harness", "⟦AH⟧"),
    ("Claude Code", "⟦CC⟧"),
    ("Bounded Action Space", "⟦BAS⟧"),
    ("Semantic IR", "⟦SIR⟧"),
    ("Strategy IR", "⟦STRIR⟧"),
    ("SemanticIR", "⟦SEMIR⟧"),
    ("StrategyIR", "⟦STIR⟧"),
    ("ScheduleIR", "⟦SCHIR⟧"),
    ("InstructionIR", "⟦IIR⟧"),
    ("@rationale", "⟦RAT⟧"),
    ("HeuristicStrategyGenerator", "⟦HSG⟧"),
    ("OptimizationState", "⟦OST⟧"),
    ("OptimizationEvent", "⟦OEV⟧"),
    ("OptimizationBudget", "⟦OBU⟧"),
    ("list_legal_actions", "⟦LLA⟧"),
    ("apply_decision", "⟦APD⟧"),
    ("verify_correctness", "⟦VFC⟧"),
    ("compile_and_profile", "⟦CAP⟧"),
    ("analyze_compute", "⟦ANC⟧"),
    ("get_hw_profile", "⟦GHP⟧"),
    ("checkpoint", "⟦CHK⟧"),
    ("rollback", "⟦RBK⟧"),
    ("compaction", "⟦CMPN⟧"),
    ("compact", "⟦CMP⟧"),
    ("ToolMeta", "⟦TM⟧"),
    ("ArkeEnv", "⟦AE⟧"),
    ("OpRegistry", "⟦OR⟧"),
    ("PassContext", "⟦PC⟧"),
    ("ShapeInferenceEngine", "⟦SIE⟧"),
    ("SemanticInterpreter", "⟦SI⟧"),
    ("ArkeBackend", "⟦AB⟧"),
    ("BackendRegistry", "⟦BR⟧"),
    ("Harness", "⟦HNS⟧"),
    ("Arke", "⟦ARK⟧"),
    ("Token", "⟦TOK⟧"),
    ("Triton", "⟦TRI⟧"),
    ("MLIR", "⟦MLI⟧"),
    ("LLVM IR", "⟦LLVM⟧"),
]

CHUNK_CHARS = 3500
MYMEMORY_CHARS = 450
PLACEHOLDER_FMT = "⟦{idx}⟧"

FENCE_RE = re.compile(r"(```[^\n]*\n.*?```)", re.DOTALL)
INLINE_CODE_RE = re.compile(r"(`[^`\n]+`)")
URL_RE = re.compile(r"(https?://[^\s\)\]>]+)")


def protect_terms(text: str) -> str:
    for term, ph in TERM_PLACEHOLDERS:
        text = text.replace(term, ph)
    return text


def restore_terms(text: str) -> str:
    for term, ph in reversed(TERM_PLACEHOLDERS):
        text = text.replace(ph, term)
    return text


def protect_segments(text: str) -> tuple[str, list[str]]:
    protected: list[str] = []

    def stash(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return PLACEHOLDER_FMT.format(idx=len(protected) - 1)

    for pattern in (FENCE_RE, INLINE_CODE_RE, URL_RE):
        text = pattern.sub(stash, text)
    return protect_terms(text), protected


def restore_segments(text: str, protected: list[str]) -> str:
    for idx, original in enumerate(protected):
        text = text.replace(PLACEHOLDER_FMT.format(idx=idx), original)
    return restore_terms(text)


def chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in text.split("\n\n"):
        if len(para) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for start in range(0, len(para), max_chars):
                chunks.append(para[start : start + max_chars])
            continue
        piece = para if not current else "\n\n" + para
        if current_len + len(piece) > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            if not current:
                current = [para]
            else:
                current.append(para)
            current_len += len(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _translate_with_engine(chunk: str, translator, max_len: int, retries: int) -> str:
    last_exc: Exception | None = None
    pieces = chunk_text(chunk, max_len) if len(chunk) > max_len else [chunk]
    out_parts: list[str] = []
    for piece in pieces:
        for attempt in range(retries):
            try:
                out_parts.append(translator.translate(piece))
                time.sleep(0.35)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"translate failed: {last_exc}") from last_exc
    return "\n\n".join(out_parts) if len(out_parts) > 1 else out_parts[0]


def _translate_chunk(chunk: str, translators: list, retries: int = 6) -> str:
    last_exc: Exception | None = None
    for i, tr in enumerate(translators):
        max_len = CHUNK_CHARS if i == 0 else MYMEMORY_CHARS
        try:
            return _translate_with_engine(chunk, tr, max_len, retries)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(f"translate failed: {last_exc}") from last_exc


def translate_text(text: str, translators: list) -> str:
    protected_text, protected = protect_segments(text)
    chunks = chunk_text(protected_text, CHUNK_CHARS)
    translated_parts: list[str] = []
    for chunk in chunks:
        if not chunk.strip():
            translated_parts.append(chunk)
            continue
        translated_parts.append(_translate_chunk(chunk, translators))
    return restore_segments("\n\n".join(translated_parts), protected)


def translate_file(src: Path, dst: Path, translators: list) -> None:
    raw = src.read_text(encoding="utf-8")
    body = translate_text(raw, translators)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(BANNER + body, encoding="utf-8")


def main() -> int:
    rel_paths = [
        "architecture/arke-lang-spec-design.md",
        "spec/arke-ir-spec.md",
        "architecture/arke-compiler-infrastructure.md",
    ]
    if len(sys.argv) > 1:
        rel_paths = [p for p in sys.argv[1:] if p.endswith(".md")]

    translators = [
        GoogleTranslator(source="en", target="zh-CN"),
        MyMemoryTranslator(source="english", target="chinese simplified"),
    ]
    for rel in rel_paths:
        src = DOCS / rel
        dst = DOCS_ZH / rel
        if not src.is_file():
            print(f"skip missing: {src}", file=sys.stderr)
            continue
        print(f"translating {rel} ...")
        translate_file(src, dst, translators)
        print(f"  -> {dst} ({sum(1 for _ in dst.open())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
