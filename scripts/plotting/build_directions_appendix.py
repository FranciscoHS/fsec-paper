"""Build the appendix artifacts describing the 33 contrastive directions:

  1. A 33x33 pairwise |cos| heatmap, ordered by family
     (semantic / language / code), with the |cos| < 0.10 inclusion
     mask outlined and the family blocks separated by black gridlines.
  2. A LaTeX longtable with one row per direction giving family, P
     (number of pairs), the sign convention, and one example pair
     (positive prompt / negative prompt).

Outputs:
  results/figures/appendix_directions_overlap.pdf
  results/figures/appendix_directions_overlap.png
  results/figures/appendix_directions_table.tex

Pass --paper_dir <dir> to redirect outputs elsewhere (e.g. into a
paper-build tree).

Usage:
  python scripts/plotting/build_directions_appendix.py
"""
from __future__ import annotations
import os, sys, pickle, argparse
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scripts.lib import registry

PAPER_DIR = "results"
EXCLUDED = {"Formal", "HonestyShort", "TensePresent"}
THRESH = 0.10


def latex_escape(s: str) -> str:
    """Escape LaTeX-special characters in a free-text prompt."""
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
            "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


# Names that use non-Latin scripts (pdflatex without xelatex/lualatex
# can't typeset their glyphs). Show a placeholder for the b-side example.
NON_LATIN_LANGS = {"Arabic", "Chinese", "Japanese", "Russian"}


def is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def truncate(s: str, n: int = 60) -> str:
    s = s.strip().replace("\n", " ")
    if len(s) <= n: return s
    return s[: n - 1].rstrip() + "\\ldots"


def family_label(fam: str) -> str:
    return {"semantic": "Semantic",
            "language": "Language",
            "code":     "Code"}[fam]


def example_pair(name: str) -> tuple[str, str]:
    """Return (a_prompt, b_prompt) for a representative pair.

    Sign is dir = b - a; we keep the same orientation in the table so
    column 'positive (b)' shows the +-side prompt and column 'negative
    (a)' shows the --side prompt.
    """
    a, b = registry.get_prompts(name)
    if not a or not b:
        return "", ""
    return a[0], b[0]


def load_directions(target: str = "gemma", layer: int = 2):
    fp = f"results/directions/dirs_{target}_L{layer}.pkl"
    with open(fp, "rb") as f:
        blob = pickle.load(f)
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--paper_dir", default=PAPER_DIR)
    args = ap.parse_args()

    blob = load_directions(args.target, args.layer)
    raw = blob["directions"]
    hashes = blob.get("prompt_set_hashes", {})

    # Collect the 33 directions and order by family then alphabetical.
    keep = [n for n in raw if n not in EXCLUDED]
    fam_of = {n: registry.family(n) for n in keep}
    fam_order = {"semantic": 0, "language": 1, "code": 2}
    keep.sort(key=lambda n: (fam_order[fam_of[n]], n.lower()))
    print(f"n_directions = {len(keep)}")
    counts = {f: sum(1 for n in keep if fam_of[n] == f)
              for f in fam_order}
    print(f"  by family: {counts}")
    assert len(keep) == 33, f"expected 33, got {len(keep)}"

    # ----- (1) overlap heatmap -----
    D = np.stack([
        raw[n].cpu().numpy().astype(np.float64)
        / max(float(np.linalg.norm(raw[n].cpu().numpy())), 1e-12)
        for n in keep
    ], axis=0)
    cos = np.abs(D @ D.T)
    np.fill_diagonal(cos, 0.0)

    n_pairs_total = len(keep) * (len(keep) - 1) // 2
    iu, ju = np.triu_indices(len(keep), k=1)
    pair_cos = cos[iu, ju]
    n_kept = int((pair_cos < THRESH).sum())
    print(f"pairs total      = {n_pairs_total}  (= C({len(keep)}, 2))")
    print(f"|cos| <  {THRESH:.2f}    = {n_kept}  ({100*n_kept/n_pairs_total:.1f}%)")

    # family block boundaries (between sorted groups)
    boundaries = []
    for i in range(1, len(keep)):
        if fam_of[keep[i]] != fam_of[keep[i - 1]]:
            boundaries.append(i)

    fig, ax = plt.subplots(figsize=(9.0, 8.4))
    im = ax.imshow(cos, cmap="magma", vmin=0.0, vmax=0.6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|\langle d_i, d_j\rangle|$  (raw cosine overlap)",
                   fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # 0.10 inclusion contour: hatched overlay where |cos| >= 0.10.
    bad = cos >= THRESH
    np.fill_diagonal(bad, False)
    ys, xs = np.where(bad)
    ax.scatter(xs, ys, s=6, c="white", marker="x", linewidths=0.6,
               alpha=0.65, zorder=3)

    ax.set_xticks(range(len(keep)))
    ax.set_yticks(range(len(keep)))
    ax.set_xticklabels(keep, rotation=90, fontsize=8)
    ax.set_yticklabels(keep, fontsize=8)
    ax.tick_params(axis="both", which="both", length=0)

    for b in boundaries:
        ax.axhline(b - 0.5, color="black", lw=1.0, alpha=0.85)
        ax.axvline(b - 0.5, color="black", lw=1.0, alpha=0.85)

    # Family block labels along the diagonal of each block.
    block_starts = [0] + boundaries + [len(keep)]
    for s, e in zip(block_starts[:-1], block_starts[1:]):
        fam = fam_of[keep[s]]
        ax.text(e - 0.6, s + 0.4, family_label(fam),
                fontsize=10, color="white", fontweight="semibold",
                ha="right", va="top",
                bbox=dict(facecolor="black", alpha=0.55, pad=2.0,
                          edgecolor="none"))

    ax.set_title(
        rf"33 contrastive directions: pairwise $|\cos|$ (Gemma-2-9B, L=2). "
        rf"White $\times$ marks pairs dropped by the $|\cos|<{THRESH:.2f}$ "
        rf"filter ({n_pairs_total - n_kept}/{n_pairs_total}; "
        rf"kept {n_kept}/{n_pairs_total}, "
        rf"{100*n_kept/n_pairs_total:.0f}\%).",
        fontsize=10)

    plt.tight_layout()
    fig_dir = os.path.join(args.paper_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_pdf = os.path.join(fig_dir, "appendix_directions_overlap.pdf")
    out_png = os.path.join(fig_dir, "appendix_directions_overlap.png")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")

    # ----- (2) LaTeX longtable -----
    rows = []
    for n in keep:
        fam = family_label(fam_of[n])
        sign = registry.get_sign(n)
        n_pairs = int(hashes.get(n, {}).get("n_pairs", 0))
        a_ex, b_ex = example_pair(n)
        # For non-Latin-script languages, swap the b-side example for a
        # placeholder so pdflatex (no xelatex / no CJK fonts) compiles.
        if n in NON_LATIN_LANGS or not is_ascii(b_ex):
            b_disp = rf"$\langle$ {latex_escape(n)} translation $\rangle$"
        else:
            b_disp = truncate(latex_escape(b_ex))
        if not is_ascii(a_ex):
            a_disp = rf"$\langle$ non-Latin source $\rangle$"
        else:
            a_disp = truncate(latex_escape(a_ex))
        rows.append((n, fam, n_pairs, sign, b_disp, a_disp))

    tex_lines = []
    tex_lines.append(r"% auto-generated by build_directions_appendix.py;")
    tex_lines.append(r"% input from paper.tex with \input{appendix_directions_table.tex}")
    tex_lines.append(r"\begin{table*}[t]")
    tex_lines.append(r"  \centering")
    tex_lines.append(r"  \caption{The 33 contrastive direction-of-the-mean "
                     r"directions used throughout the paper. Sign convention "
                     r"is $\mathbf{d} = \mathbf{a}(b) - \mathbf{a}(a)$, so "
                     r"the +$\alpha$ direction rotates from the negative "
                     r"prompt toward the positive prompt. $P$ is the number "
                     r"of contrastive prompt pairs averaged over to form "
                     r"each direction. For multi-class concepts (languages, "
                     r"programming) each direction is the difference from "
                     r"English text aligned by prompt template.}")
    tex_lines.append(r"  \label{tab:contrastive-directions}")
    tex_lines.append(r"  \footnotesize")
    tex_lines.append(r"  \setlength{\tabcolsep}{4pt}")
    tex_lines.append(r"  \begin{tabular}{l l c l p{4.4cm} p{4.4cm}}")
    tex_lines.append(r"    \toprule")
    tex_lines.append(r"    Direction & Family & $P$ & Sign "
                     r"& Example positive prompt ($b$) "
                     r"& Example negative prompt ($a$) \\")
    tex_lines.append(r"    \midrule")
    last_fam = None
    for (name, fam, P, sign, pos, neg) in rows:
        if last_fam is not None and fam != last_fam:
            tex_lines.append(r"    \midrule")
        last_fam = fam
        tex_lines.append(
            rf"    {latex_escape(name)} & {fam} & {P} & "
            rf"{latex_escape(sign)} & {pos} & {neg} \\")
    tex_lines.append(r"    \bottomrule")
    tex_lines.append(r"  \end{tabular}")
    tex_lines.append(r"\end{table*}")
    tex_path = os.path.join(args.paper_dir, "appendix_directions_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"saved {tex_path}")

    # Quick cross-check: report retention at threshold variants for the
    # caption.
    for t in (0.05, 0.10, 0.15, 0.20):
        k = int((pair_cos < t).sum())
        print(f"  |cos| < {t:.2f}: kept {k}/{n_pairs_total}  "
              f"({100*k/n_pairs_total:.0f}%)")


if __name__ == "__main__":
    main()
