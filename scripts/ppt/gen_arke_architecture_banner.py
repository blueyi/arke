"""
Arke architecture banner PPT — single slide, 570×120 px.

Layout (left → right pipeline):
  [Input · 用户 / Agent 意图]
      → ① Language (.ak)
      → ② Multi-Layer IR
      → ③ Compiler-as-Verifier
      → ④ Agent Runtime
      → ⑤ Benchmark (横切度量层)
      → [Output · 多硬件 GPU/NPU]

Output: docs/sharing/ai-native-arke-architecture-banner.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt


# ============================================================
# Slide size = 570 × 120 pixels @ 96 DPI
# 1 px = 9525 EMU at 96 DPI (≈ 1/96 inch)
# ============================================================
PX = 9525
SLIDE_W = Emu(570 * PX)   # 5.9375 in
SLIDE_H = Emu(120 * PX)   # 1.25 in


# ============================================================
# Theme (consistent with the main insight deck)
# ============================================================
BG_DARK = RGBColor(0x0B, 0x1B, 0x2E)
BG_PANEL = RGBColor(0x12, 0x2A, 0x43)
BG_PANEL_ALT = RGBColor(0x18, 0x35, 0x53)
FG = RGBColor(0xEA, 0xF2, 0xFA)
FG_MUTED = RGBColor(0x9F, 0xB5, 0xC9)
FG_DIM = RGBColor(0x6E, 0x86, 0x9F)

ACCENT = RGBColor(0x38, 0xD1, 0xB8)         # teal — Language
ACCENT_BLUE = RGBColor(0x7A, 0xB8, 0xFF)    # IR
ACCENT_ALT = RGBColor(0xF2, 0xC5, 0x5C)     # gold — Compiler / Benchmark
ACCENT_PURPLE = RGBColor(0xB8, 0x8A, 0xF0)  # Agent
ACCENT_RED = RGBColor(0xEF, 0x6E, 0x6E)
STROKE = RGBColor(0x24, 0x44, 0x66)

FONT_BODY = "Microsoft YaHei"
FONT_MONO = "Consolas"


# ============================================================
# Helpers
# ============================================================
def _set_fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def _no_line(shape):
    shape.line.fill.background()


def _set_line(shape, color, width_pt=0.6):
    shape.line.color.rgb = color
    shape.line.width = Pt(width_pt)


def add_text(slide, left, top, width, height, text, *,
             size=8, bold=False, italic=False, color=FG, font=FONT_BODY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb


def emu(px_value):
    """Convert pixel value to EMU (96 DPI)."""
    return Emu(int(px_value * PX))


# ============================================================
# Build the single banner slide
# ============================================================
def build(out_path: Path) -> None:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    s = prs.slides.add_slide(prs.slide_layouts[6])

    # background
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    _set_fill(bg, BG_DARK)
    _no_line(bg)
    sp_tree = bg._element.getparent()
    sp_tree.remove(bg._element)
    sp_tree.insert(2, bg._element)

    # top accent bar
    top_bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 0, 0, SLIDE_W, emu(2))
    _set_fill(top_bar, ACCENT)
    _no_line(top_bar)
    bot_bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 0, SLIDE_H - emu(2),
                                 SLIDE_W, emu(2))
    _set_fill(bot_bar, ACCENT)
    _no_line(bot_bar)

    # ============================================================
    # Top kicker — title row
    # ============================================================
    add_text(s, emu(8), emu(5), emu(380), emu(14),
             "ARKE · AI-NATIVE OPERATOR COMPILER STACK",
             size=7, bold=True, color=ACCENT, font=FONT_MONO,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, emu(360), emu(5), emu(202), emu(14),
             "让 LLM 写 kernel · 让编译器把关数学",
             size=6.5, bold=False, color=FG_MUTED, font=FONT_MONO,
             align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)

    # ============================================================
    # Pipeline row (the main flow)
    # ============================================================
    # 7 boxes: input + 5 components + output
    # widths balanced; arrows in between
    # 总宽 570 - 左右各留 6px = 558px
    PAD_X = 6
    GAP = 4   # arrow gap between boxes
    ARROW_W = 8

    # box widths (px) — sum + 6 gaps = 558 - 6*8 = 510 → boxes ≈ 73 each
    box_widths_px = [70, 78, 78, 78, 78, 78, 70]   # 530 + 6*8 = 578... adjust
    # recompute: total = sum(box) + 6*ARROW_W = 558 → sum(box) = 510
    box_widths_px = [70, 76, 76, 76, 76, 76, 60]  # =510

    # Top of box row
    BOX_TOP_PX = 26
    BOX_H_PX = 60

    box_y = emu(BOX_TOP_PX)
    box_h = emu(BOX_H_PX)

    # Component definitions — each box has:
    #   id_glyph, header text, sub text, color
    boxes = [
        ("⎘",     "User / Agent",
         "kernel semantics\n+ optimization intent",
         ACCENT_RED, BG_PANEL_ALT),

        ("①",     "Arke Language",
         ".ak  ·  kernel { } + strategy { }",
         ACCENT, BG_PANEL),

        ("②",     "Multi-Layer IR",
         "Semantic · Strategy ·\nSchedule · Instruction",
         ACCENT_BLUE, BG_PANEL),

        ("③",     "Compiler-as-Verifier",
         "OpRegistry · Pass\n静态 / 数值 / 性能 三级验证",
         ACCENT_ALT, BG_PANEL),

        ("④",     "Agent Runtime",
         "Bounded action · tool-use\nbudget · checkpoint/rollback",
         ACCENT_PURPLE, BG_PANEL),

        ("⑤",     "Benchmark",
         "BL × OT × ST × L · 6 baselines\n45 ops · ~350 shapes",
         ACCENT_ALT, BG_PANEL_ALT),

        ("▶",     "Hardware",
         "NVIDIA · Ascend · AMD\n→ 多 NPU / DSA",
         ACCENT_RED, BG_PANEL_ALT),
    ]

    cur_x_px = PAD_X
    centers = []  # for arrow connectors
    for i, ((g, head, sub, col, fill_bg), w_px) in enumerate(
            zip(boxes, box_widths_px)):
        x = emu(cur_x_px)
        w = emu(w_px)
        # rounded box
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 x, box_y, w, box_h)
        box.adjustments[0] = 0.18
        _set_fill(box, fill_bg)
        _set_line(box, col, 0.75)

        # left accent bar
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 x, box_y, emu(3), box_h)
        _set_fill(bar, col)
        _no_line(bar)

        # glyph
        add_text(s, x + emu(4), box_y + emu(3), emu(14), emu(14),
                 g, size=9, bold=True, color=col, font=FONT_MONO,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

        # header text
        add_text(s, x + emu(4), box_y + emu(3),
                 w - emu(8), emu(14),
                 head, size=6.5, bold=True, color=col,
                 align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.TOP)

        # sub text
        add_text(s, x + emu(4), box_y + emu(20),
                 w - emu(8), emu(BOX_H_PX - 22),
                 sub, size=5.5, color=FG_MUTED,
                 font=FONT_MONO, align=PP_ALIGN.LEFT,
                 anchor=MSO_ANCHOR.TOP)

        centers.append((cur_x_px, cur_x_px + w_px))
        cur_x_px += w_px
        if i < len(boxes) - 1:
            # arrow between this and next box
            ax_px = cur_x_px
            # right-pointing arrow shape
            arr = s.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW,
                emu(ax_px), box_y + emu(BOX_H_PX // 2 - 4),
                emu(ARROW_W), emu(8))
            _set_fill(arr, FG_MUTED)
            _no_line(arr)
            cur_x_px += ARROW_W

    # ============================================================
    # Bottom caption row — pillars / scenarios
    # ============================================================
    cap_y_px = BOX_TOP_PX + BOX_H_PX + 4
    add_text(s, emu(8), emu(cap_y_px),
             emu(560), emu(12),
             "解决场景： LLM 驱动 kernel 生成 / 优化  ·  跨硬件统一表达 (SIMT ↔ SIMD ↔ NPU)  ·  最小 token 预算下达到厂商库级性能",
             size=6, color=FG, bold=True,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, emu(8), emu(cap_y_px + 11),
             emu(560), emu(11),
             "技术主张： 语义 / 策略分离  ·  有界动作空间  ·  多层 IR + 多级验证  ·  硬件信号闭环  ·  结构化经验沉淀  ·  可复现评测",
             size=5.5, color=FG_DIM, italic=True,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    print(f"wrote: {out_path}")
    print(f"slide size: {prs.slide_width / PX:.0f} × "
          f"{prs.slide_height / PX:.0f} px "
          f"({prs.slide_width / 914400:.3f} × "
          f"{prs.slide_height / 914400:.3f} in)")


if __name__ == "__main__":
    out = (Path(__file__).resolve().parents[2]
           / "docs" / "sharing"
           / "ai-native-arke-architecture-banner.pptx")
    build(out)
