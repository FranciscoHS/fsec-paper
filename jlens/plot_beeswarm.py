#!/usr/bin/env python3
"""Beeswarm of the superellipse exponent p for J-lens vs contrastive
directions on Qwen3.6-27B (perturb L36, measure L62).

Reads only the cached fits in jlens/data/ — no GPU, no model, no network.
Self-contained (numpy + matplotlib) so this directory can be lifted out of
the repo as a standalone artifact.

Usage (from the repo root, or from anywhere):
    python jlens/plot_beeswarm.py
Output: jlens/qwen36_p_beeswarm.{png,pdf}
"""
from __future__ import annotations
import os, pickle, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Per-pair p = the headline cell of the paper's protocol: the single fit at
# the per-pair threshold 1.0xT, exact-geodesic normalization. Mirrors
# CANONICAL_CELL / median_p_for_pair in
# scripts/plotting/plot_robustness_beeswarm.py (inlined to keep this
# directory dependency-free). The angle window differs by family only
# because the sweeps do: contrastive runs to 60 deg, J-lens to 20 deg.
N_BOOT = 200_000
CI_PCT = (0.5, 99.5)        # 99% interval
SEED = 0


def headline_ps(path, window):
    """Per-pair p at cell ("1.0xT", window) from a protocol fits pkl."""
    with open(path, "rb") as f:
        full = pickle.load(f)
    cell_key = ("1.0xT", window)
    out = []
    for fit in full.values():
        cell = fit.get(cell_key)
        if cell is None:
            continue
        p = cell.get("p_median", np.nan)
        if np.isfinite(p):
            out.append(float(p))
    return out


def boot_ci(ps, rng):
    ps = np.asarray(ps, dtype=float)
    meds = np.median(rng.choice(ps, size=(N_BOOT, len(ps)), replace=True),
                     axis=1)
    lo, hi = np.percentile(meds, CI_PCT)
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "qwen36_p_beeswarm"))
    args = ap.parse_args()

    cols = [
        ("Contrastive",
         headline_ps(os.path.join(DATA, "fits_contrastive_thrpair_exact.pkl"),
                     60.0), "#e41a1c"),
        ("J-Lens ZH",
         headline_ps(os.path.join(DATA, "fits_jlens_zh100_thrpair_exact.pkl"),
                     20.0), "#377eb8"),
        ("J-Lens EN",
         headline_ps(os.path.join(DATA,
                                  "fits_jlens_en_concrete_thrpair_exact.pkl"),
                     20.0), "#984ea3"),
    ]

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (label, ps, c) in enumerate(cols):
        ps = np.asarray(ps, dtype=float)
        x = i + (rng.random(len(ps)) - 0.5) * 0.25
        ax.scatter(x, ps, s=34, color=c, alpha=0.75, edgecolor="k",
                   linewidth=0.4, zorder=3)
        med = float(np.median(ps))
        lo, hi = boot_ci(ps, rng)
        xc = i + 0.34
        ax.plot([xc, xc], [lo, hi], color="k", lw=1.6, zorder=4)
        for y in (lo, hi):
            ax.plot([xc - 0.05, xc + 0.05], [y, y], color="k", lw=1.6, zorder=4)
        ax.plot(xc, med, "o", color="k", ms=5, zorder=5)
        ax.plot([i - 0.28, i + 0.28], [med, med], color="k", lw=2, zorder=4)
        ax.annotate(f"med {med:.2f}\n[{lo:.2f}, {hi:.2f}]\nn={len(ps)}",
                    (xc, med), textcoords="offset points", xytext=(12, 0),
                    fontsize=8.5, va="center")
        print(f"{label.replace(chr(10), ' '):22s} n={len(ps):3d}  "
              f"median={med:.3f}  99% CI=[{lo:.3f}, {hi:.3f}]")

    ax.axhline(2.0, color="gray", ls="--", lw=1, zorder=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols])
    ax.set_ylabel("superellipse exponent p")
    ax.set_title("Qwen3.6-27B  L36→L62\n"
                 "bars: bootstrap 99% CI on the median", fontsize=11)
    ax.set_xlim(-0.5, len(cols) - 1 + 0.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150)
    print(f"Saved: {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
