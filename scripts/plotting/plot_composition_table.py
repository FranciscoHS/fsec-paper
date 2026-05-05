"""Composition figure: Gender x Tense on "Once upon a time, the prince".

Cloned from scripts/plotting/plot_composition_formal_pt.py — same renderer
(rounded box, monospace, label inline with first body line) and the same
orange/blue/purple palette. Only the feature labels and content change:

  C_GEN  = orange  (female-gendered)   <- slot 1, was Formal
  C_FUT  = blue    (future tense)      <- slot 2, was PT
  C_BOTH = purple  (both)
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_PNG = "results/figures/composition_table_gender_tense.png"
OUT_PDF = "results/figures/composition_table_gender_tense.pdf"

# Colours (slot-matched to plot_composition_formal_pt.py)
C_TEXT = "#2a2a2a"
C_LABEL_NEUTRAL = "#222222"
C_GEN  = "#D9822B"   # orange — female-gendered (slot 1, was Formal)
C_FUT  = "#1F77B4"   # blue   — future tense    (slot 2, was Portuguese)
C_BOTH = "#7B3294"   # purple — both

PROMPT = "Once upon a time, the prince"

# Each row: (label_segments, body_segments).
# label_segments is a list of (text, colour) tuples — same structure as
# body, so the angle can be coloured to match its direction.
ROWS = [
    (
        [("Unsteered", C_LABEL_NEUTRAL)],
        [
            ("went out into the world to learn the craft of his father, the "
             "king. The prince traveled to the edge of the world, and from "
             "there he traveled on to a land where the sun never set. Here "
             "he found a wise old man. “What is the highest art of "
             "kingship?” asked the prince. “To love and s…",
             C_TEXT),
        ],
    ),
    (
        [("+ ", C_LABEL_NEUTRAL), ("Female 20°", C_GEN)],
        [
            ("was walking through the woods, and ", C_TEXT),
            ("she", C_GEN),
            (" came across a fairy who asked ", C_TEXT),
            ("her", C_GEN),
            (" what ", C_TEXT),
            ("she", C_GEN),
            (" wanted. And the ", C_TEXT),
            ("princess", C_GEN),
            (" replied, “I want to be loved.” This fairy is very "
             "old and has lived in these woods for hundreds of years. ",
             C_TEXT),
            ("She", C_GEN),
            (" is a very powerful fairy, and when ", C_TEXT),
            ("she", C_GEN),
            (" heard the", C_TEXT),
        ],
    ),
    (
        [("+ ", C_LABEL_NEUTRAL), ("Future 30°", C_FUT)],
        [
            ("was sent to a land where the king is having a problem. The "
             "prince ", C_TEXT),
            ("will try", C_FUT),
            (" to solve the problem, but he ", C_TEXT),
            ("will fail", C_FUT),
            (". So he ", C_TEXT),
            ("will go", C_FUT),
            (" to a tower to get help from a fairy, but he ", C_TEXT),
            ("will fail", C_FUT),
            (". Then he ", C_TEXT),
            ("will go", C_FUT),
            (" to a tower to get help from a wizard, but he ", C_TEXT),
            ("will fail", C_FUT),
            (". Finally, he ", C_TEXT),
            ("will go", C_FUT),
            ("…", C_TEXT),
        ],
    ),
    (
        [("+ ", C_LABEL_NEUTRAL),
         ("Female 20°", C_GEN),
         (" + ", C_LABEL_NEUTRAL),
         ("Future 30°", C_FUT)],
        [
            ("of an Indian tribe was forced to marry the prince of an "
             "African tribe. After they married each other, the ", C_TEXT),
            ("queen", C_GEN),
            (" of the Indian tribe ", C_TEXT),
            ("will give", C_FUT),
            (" birth to two twins. One twin has long hair and one has short "
             "hair. The ", C_TEXT),
            ("mother", C_GEN),
            (" of the twin ", C_TEXT),
            ("will take", C_FUT),
            (" ", C_TEXT),
            ("her", C_GEN),
            (" short hair and send ", C_TEXT),
            ("her", C_GEN),
            (" to another", C_TEXT),
        ],
    ),
]

# ---- Layout (matches plot_composition_formal_pt.py) ----
FIG_W_IN = 11.5
FONT_PROMPT_LABEL = 11
FONT_LABEL = 10.5
FONT_BODY = 10
FONT_FAMILY = "monospace"
WRAP_CHARS = 88
LABEL_COL_W = 3.5
LEFT_MARGIN = 0.45
RIGHT_MARGIN = 0.45
TOP_MARGIN = 0.55
BOTTOM_MARGIN = 0.45
PROMPT_GAP = 0.30
ROW_GAP = 0.30
LINE_HEIGHT = FONT_BODY * 1.4 / 72
PARA_GAP = 0.05


def fragment_to_tokens(text, color):
    tokens = []; buf = ""
    for ch in text:
        if ch == "\n":
            if buf: tokens.append((buf, color)); buf = ""
            tokens.append(("\n", color))
        elif ch == " ":
            if buf: tokens.append((buf, color)); buf = ""
            tokens.append((" ", color))
        else:
            buf += ch
    if buf: tokens.append((buf, color))
    return tokens


def wrap_tokens(tokens, max_chars):
    lines = []; current = []; cur_len = 0; i = 0
    while i < len(tokens):
        tok, col = tokens[i]
        if tok == "\n":
            n = 1
            while i + 1 < len(tokens) and tokens[i + 1][0] == "\n":
                n += 1; i += 1
            if current: lines.append(current); current = []; cur_len = 0
            if n >= 2: lines.append("PARA")
            i += 1; continue
        if tok == " ":
            if current: current.append((" ", col)); cur_len += 1
            i += 1; continue
        if cur_len + len(tok) > max_chars and current:
            while current and current[-1][0] == " ": current.pop()
            lines.append(current); current = []; cur_len = 0
        current.append((tok, col)); cur_len += len(tok); i += 1
    if current: lines.append(current)
    return lines


def main():
    prompt_lines = wrap_tokens(fragment_to_tokens(PROMPT, C_TEXT), WRAP_CHARS)
    rows_lines = []
    for _label_segs, segs in ROWS:
        toks = []
        for t, c in segs:
            toks.extend(fragment_to_tokens(t, c))
        rows_lines.append(wrap_tokens(toks, WRAP_CHARS))

    total_h = TOP_MARGIN + BOTTOM_MARGIN + LINE_HEIGHT
    for ln in prompt_lines:
        total_h += PARA_GAP if ln == "PARA" else LINE_HEIGHT
    total_h += PROMPT_GAP
    for lines in rows_lines:
        for ln in lines:
            total_h += PARA_GAP if ln == "PARA" else LINE_HEIGHT
        total_h += ROW_GAP
    total_h -= ROW_GAP

    fig = plt.figure(figsize=(FIG_W_IN, total_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W_IN); ax.set_ylim(0, total_h); ax.invert_yaxis()
    ax.axis("off")

    box = FancyBboxPatch(
        (0.18, 0.18), FIG_W_IN - 0.36, total_h - 0.36,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        linewidth=1.2, edgecolor="#888888", facecolor="white")
    ax.add_patch(box)

    label_x = LEFT_MARGIN
    body_x = LEFT_MARGIN + LABEL_COL_W
    y = TOP_MARGIN

    ax.text(label_x, y, "Prompt:", fontsize=FONT_PROMPT_LABEL,
            fontfamily=FONT_FAMILY, fontweight="bold",
            color=C_LABEL_NEUTRAL, va="top")
    x = body_x
    for tok, col in fragment_to_tokens(PROMPT, C_TEXT):
        if tok == " ":
            t = ax.text(x, y, " ", fontsize=FONT_BODY,
                        fontfamily=FONT_FAMILY, color=col, va="top")
            fig.canvas.draw(); x += t.get_window_extent().width / fig.dpi
        elif tok == "\n":
            y += LINE_HEIGHT; x = body_x
        else:
            t = ax.text(x, y, tok, fontsize=FONT_BODY,
                        fontfamily=FONT_FAMILY, color=col, va="top")
            fig.canvas.draw(); x += t.get_window_extent().width / fig.dpi
    y += LINE_HEIGHT
    y += PROMPT_GAP

    for (label_segs, _body), lines in zip(ROWS, rows_lines):
        # multi-coloured label, all on one line, ending with ':'
        x = label_x
        for tok, col in label_segs:
            t = ax.text(x, y, tok, fontsize=FONT_LABEL,
                        fontfamily=FONT_FAMILY, fontweight="bold",
                        color=col, va="top")
            fig.canvas.draw(); x += t.get_window_extent().width / fig.dpi
        ax.text(x, y, ":", fontsize=FONT_LABEL,
                fontfamily=FONT_FAMILY, fontweight="bold",
                color=C_LABEL_NEUTRAL, va="top")
        for ln in lines:
            if ln == "PARA":
                y += PARA_GAP; continue
            x = body_x
            for tok, col in ln:
                t = ax.text(x, y, tok if tok != " " else " ",
                            fontsize=FONT_BODY,
                            fontfamily=FONT_FAMILY, color=col, va="top")
                fig.canvas.draw(); x += t.get_window_extent().width / fig.dpi
            y += LINE_HEIGHT
        y += ROW_GAP

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
