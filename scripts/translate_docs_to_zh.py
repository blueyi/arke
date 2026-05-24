#!/usr/bin/env python3
"""[DEPRECATED] Machine-translate docs/*.md to docs_zh/.

Prefer full LLM translation (see docs_zh banner workflow). MT causes
systematic errors (Pass→通行证, LLM→法学硕士, dim→昏暗). Kept only for
emergency regen with manual terminology fixes afterward.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from deep_translator import GoogleTranslator, MyMemoryTranslator

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS_ZH = ROOT / "docs_zh"

# Google Translate practical chunk size
CHUNK_CHARS = 3500
MYMEMORY_CHARS = 450

PLACEHOLDER_FMT = "⟦{idx}⟧"

FENCE_RE = re.compile(r"(```[^\n]*\n.*?```)", re.DOTALL)
INLINE_CODE_RE = re.compile(r"(`[^`\n]+`)")
URL_RE = re.compile(r"(https?://[^\s\)\]>]+)")


def protect_segments(text: str) -> tuple[str, list[str]]:
    """Replace non-translatable segments with placeholders."""
    protected: list[str] = []

    def stash(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return PLACEHOLDER_FMT.format(idx=len(protected) - 1)

    for pattern in (FENCE_RE, INLINE_CODE_RE, URL_RE):
        text = pattern.sub(stash, text)
    return text, protected


def restore_segments(text: str, protected: list[str]) -> str:
    for idx, original in enumerate(protected):
        text = text.replace(PLACEHOLDER_FMT.format(idx=idx), original)
    return text


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
                time.sleep(0.4)
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
    header = (
        f"> 本文由 `scripts/translate_docs_to_zh.py` 从 "
        f"`{src.relative_to(ROOT)}` 自动生成，仅供阅读参考；"
        f"规范以英文原文为准。\n\n"
    )
    body = translate_text(raw, translators)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(header + body, encoding="utf-8")


def main() -> int:
    if not DOCS.is_dir():
        print(f"Missing docs dir: {DOCS}", file=sys.stderr)
        return 1

    md_files = sorted(DOCS.rglob("*.md"))
    if not md_files:
        print("No markdown files under docs/", file=sys.stderr)
        return 1

    resume = "--resume" in sys.argv
    translators = [
        GoogleTranslator(source="en", target="zh-CN"),
        MyMemoryTranslator(source="english", target="chinese simplified"),
    ]
    total = len(md_files)
    for i, src in enumerate(md_files, 1):
        rel = src.relative_to(DOCS)
        dst = DOCS_ZH / rel
        if resume and dst.is_file():
            print(f"[{i}/{total}] skip {rel}")
            continue
        print(f"[{i}/{total}] {rel}")
        translate_file(src, dst, translators)

    print(f"Done: {total} files -> {DOCS_ZH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
