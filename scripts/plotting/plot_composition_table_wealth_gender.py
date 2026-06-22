"""Composition figure: Wealth x Gender on a gender-neutral, wealth-silent prompt.

Cloned from plot_composition_table.py (same renderer). Steering directions are
applied as norm-matched rotations at Gemma-2-9B layer 2 (see scripts/steer_test.py).
Completions are representative greedy/sampled generations selected from
logs/steer_wg_combos.txt (singles, 20 deg) and logs/steer_wg_combo_angle.txt
(composite, 30 deg). Labels name the *effect*: +Wealth steers toward poverty,
+Gender toward feminine.

  C_WEALTH = orange  (poverty;  Wealth axis)   <- slot 1
  C_GENDER = blue    (feminine; Gender axis)   <- slot 2
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_PNG = "results/figures/composition_table_wealth_gender.png"
OUT_PDF = "results/figures/composition_table_wealth_gender.pdf"

C_TEXT = "#2a2a2a"
C_LABEL_NEUTRAL = "#222222"
C_WEALTH = "#D9822B"   # orange — poverty  (Wealth axis, slot 1)
C_GENDER = "#1F77B4"   # blue   — feminine (Gender axis, slot 2)

PROMPT = "The other day I met someone who"

ROWS = [
    (
        [("Unsteered", C_LABEL_NEUTRAL)],
        [
            ("was interested in the idea of moving to Austin, TX. He wanted a "
             "bit more detail about what makes it such an attractive place for "
             "professionals and students from out-of-town. What are some "
             "reasons we love living here? Here's my…", C_TEXT),
        ],
    ),
    (
        [("+ ", C_LABEL_NEUTRAL), ("Poverty 30°", C_WEALTH)],
        [
            ("was ", C_TEXT),
            ("so poor", C_WEALTH),
            (" that he ", C_TEXT),
            ("had no money to buy food", C_WEALTH),
            (". He asked me for ", C_TEXT),
            ("some bread and water", C_WEALTH),
            (", but all the while my heart bled with him as it could not give "
             "anything more than a little bit of pity…", C_TEXT),
        ],
    ),
    (
        [("+ ", C_LABEL_NEUTRAL), ("Female 20°", C_GENDER)],
        [
            ("was wearing a pair of earrings that caught my eye. ", C_TEXT),
            ("The girl", C_GENDER),
            (" explained they were ", C_TEXT),
            ("her", C_GENDER),
            (" engagement ring and wedding bands all in one, so ", C_TEXT),
            ("she", C_GENDER),
            (" didn't have to take off the earring set when taking them on…",
             C_TEXT),
        ],
    ),
    (
        [("+ ", C_LABEL_NEUTRAL),
         ("Poverty", C_WEALTH),
         (" + ", C_LABEL_NEUTRAL),
         ("Female 30°", C_GENDER)],
        [
            ("was ", C_TEXT),
            ("homeless", C_WEALTH),
            (". ", C_TEXT),
            ("She", C_GENDER),
            (" told me ", C_TEXT),
            ("she", C_GENDER),
            (" had ", C_TEXT),
            ("nowhere to go", C_WEALTH),
            (" and ", C_TEXT),
            ("no money", C_WEALTH),
            (" in ", C_TEXT),
            ("her", C_GENDER),
            (" pocket, so ", C_TEXT),
            ("she", C_GENDER),
            (" slept out ", C_TEXT),
            ("on the street", C_WEALTH),
            (" every night because of ", C_TEXT),
            ("poverty", C_WEALTH),
            ("…", C_TEXT),
        ],
    ),
]

# ---- Layout (matches plot_composition_table.py) ----
FIG_W_IN = 11.5
FONT_PROMPT_LABEL = 11
FONT_LABEL = 10.5
FONT_BODY = 10
FONT_FAMILY = "monospace"
WRAP_CHARS = 80
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
