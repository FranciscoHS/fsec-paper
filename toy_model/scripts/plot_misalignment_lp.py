"""Replot Phase 4 (Lp exponent vs feature-axis misalignment) from saved npz.

Reads runs/.../lp_vs_angle[_jumprelu].npz and emits a stripped-down plot:
  * y-axis: p
  * x-axis: cosine similarity to the feature directions (cos α)
  * legend: short, one entry per d/H ratio
  * no title

Single panel by default. For NPZ files saved with multiple τ values
(--tau_fracs / --taus_abs sweeps), pass --tau_index N to pick which τ to plot.

Usage:
    python -m scripts.plot_misalignment_lp \
        --npz runs/misalignment_sweep_H1024/lp_vs_angle.npz \
        --out runs/misalignment_sweep_H1024/lp_vs_angle_clean.png
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau_index", type=int, default=0,
                    help="which τ to plot when the npz holds a τ-sweep")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    angles = z["angles"]
    cos_alpha_param = np.cos(angles)
    H = int(z["H"])
    ds = list(z["ds"])

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    used_empirical = False
    for ix, d in enumerate(ds):
        ns = z[f"ns_d{int(d)}"]
        if ns.ndim == 3:
            ns = ns[args.tau_index]
        med = np.array([np.nanmedian(ns[ai]) for ai in range(ns.shape[0])])
        q25 = np.array([np.nanpercentile(ns[ai], 25)
                        for ai in range(ns.shape[0])])
        q75 = np.array([np.nanpercentile(ns[ai], 75)
                        for ai in range(ns.shape[0])])

        # x-axis: prefer the realized cosine ((|cos(u,e_j)| + |cos(v,e_k)|)/2,
        # median over pairs) if the NPZ has it; otherwise fall back to the
        # parametric cos(α) input grid.
        cos_uj_key = f"cos_uj_d{int(d)}"
        cos_vk_key = f"cos_vk_d{int(d)}"
        if cos_uj_key in z.files and cos_vk_key in z.files:
            cos_uj = np.asarray(z[cos_uj_key])      # (n_angles, n_pairs)
            cos_vk = np.asarray(z[cos_vk_key])
            cos_per_pair = 0.5 * (np.abs(cos_uj) + np.abs(cos_vk))
            x = np.array([np.nanmedian(cos_per_pair[ai])
                          for ai in range(cos_per_pair.shape[0])])
            used_empirical = True
        else:
            x = cos_alpha_param

        ratio = int(d) // H
        color = palette[ix % len(palette)]
        ax.fill_between(x, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, med, color=color, marker="o", linewidth=1.6,
                label=f"{ratio}×")
    ax.axhline(2, color="k", ls="--", alpha=0.5, linewidth=0.8)
    xlabel = ("cosine similarity to feature axes (median over pairs)"
              if used_empirical else "cosine similarity to feature axes")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"fitted superellipse exponent $p$")
    ax.set_xlim(-0.02, 1.02)
    ax.invert_xaxis()  # left = aligned (cos=1), right = orthogonal (cos=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=10)

    # Mirror the bottom (cosine) axis on top as the misalignment angle in
    # degrees: forward = arccos(c), inverse = cos(deg).
    def _cos_to_deg(c):
        return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))

    def _deg_to_cos(deg):
        return np.cos(np.radians(deg))

    secax = ax.secondary_xaxis("top", functions=(_cos_to_deg, _deg_to_cos))
    secax.set_xlabel(r"misalignment angle $\alpha$ (deg)")
    secax.set_xticks([15, 30, 45, 60, 75, 90])

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
