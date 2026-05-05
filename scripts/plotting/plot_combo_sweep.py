"""1D sweep curves along d1, d2, their equal-weight combination, and a
random-direction baseline.

Reads a 2D sweep pkl (FineWeb anchors) and extracts three 1D curves
from the same activation set, so the combination curve and the two
single-direction curves are directly comparable:

  - l2[:, :, 0]            -> sweep along d1_perp at alpha_2 = 0
  - l2[:, 0, :]            -> sweep along d2_perp at alpha_1 = 0
  - diag l2[:, k, k] with
    x = sqrt(2) * angles[k] -> sweep along (d1_perp + d2_perp)/sqrt(2)

Then reads up to ``--n_random`` of the random-direction sweep pkls
(``sweep2d_<target>_L<layer>_*_fineweb_60deg_dirrandom.pkl``); each
contributes its first axis (``l2[:, :, 0]``, response along one random
unit direction). Per-direction curve = median over anchors; the random
band shows median + IQR across the ``n_random`` per-direction curves.

Plots median + IQR for each, a horizontal threshold line T, and dashed
verticals at each curve's plateau-breaking angle (linear interp of the
median curve across T).

Usage:
  python scripts/plotting/plot_combo_sweep.py \\
      --target gemma --d1 Gender --d2 Tense --layer 2 --max_angle 25
"""
from __future__ import annotations
import os, sys, glob, pickle, argparse
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results/figures"
DATA_DIR = "results/sweeps_2d"


def crossing(angles, curve, threshold):
    """Linear-interpolated angle (deg) where curve first crosses threshold.

    Returns nan if the curve never reaches the threshold within the
    plotted range.
    """
    if curve.max() < threshold:
        return float("nan")
    for i in range(1, len(curve)):
        if (curve[i - 1] - threshold) * (curve[i] - threshold) <= 0:
            x0, x1 = float(angles[i - 1]), float(angles[i])
            y0, y1 = float(curve[i - 1]), float(curve[i])
            if y1 == y0:
                return x0
            t = (threshold - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--d1", default="Gender")
    ap.add_argument("--d2", default="Tense")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--max_angle", type=float, default=25.0,
                    help="x-axis upper limit in degrees")
    ap.add_argument("--threshold", type=float, default=None,
                    help="L^2 threshold for plateau-breaking. Default = "
                         "0.5 * min(max(L2[:,0]), max(L2[0,:])) on the "
                         "median grid.")
    ap.add_argument("--threshold_frac", type=float, default=0.50,
                    help="when --threshold is omitted, threshold = "
                         "frac * min(axis maxes).")
    ap.add_argument("--n_random", type=int, default=10,
                    help="how many random-direction sweep pkls to "
                         "include for the random baseline curve "
                         "(median + IQR across the per-direction "
                         "anchor-medians). 0 disables the random "
                         "baseline.")
    ap.add_argument("--out_tag", default="")
    args = ap.parse_args()

    pkl_path = (f"{DATA_DIR}/sweep2d_{args.target}_L{args.layer}_"
                f"{args.d1}__{args.d2}_fineweb_60deg.pkl")
    if not os.path.exists(pkl_path):
        # try the swapped pair (file naming follows extraction order)
        pkl_path = (f"{DATA_DIR}/sweep2d_{args.target}_L{args.layer}_"
                    f"{args.d2}__{args.d1}_fineweb_60deg.pkl")
    if not os.path.exists(pkl_path):
        raise SystemExit(f"no sweep2d pkl for {args.d1} x {args.d2} at "
                         f"{args.target} L{args.layer}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    angles = np.asarray(d["angles_deg"], dtype=float)   # (n,)
    grid = np.asarray(d["l2"], dtype=float)             # (n_anchors, n, n)
    a, b = d["direction_labels"]
    # The order in the filename / direction_labels is (d1, d2). axis 1 of
    # `l2` is alpha_1 (along a), axis 2 is alpha_2 (along b). Match the
    # user's --d1 / --d2 to the file order so the displayed curves are
    # correctly named even when we fell back to the swapped filename.
    if {a, b} != {args.d1, args.d2}:
        raise SystemExit(f"label mismatch: pkl has ({a}, {b}), user asked "
                         f"({args.d1}, {args.d2})")
    if (a, b) == (args.d1, args.d2):
        gender_axis_grid = grid[:, :, 0]                # along d1 (= --d1)
        tense_axis_grid  = grid[:, 0, :]                # along d2 (= --d2)
    else:
        gender_axis_grid = grid[:, 0, :]
        tense_axis_grid  = grid[:, :, 0]

    # equal-weight combination: l2 along (d1_perp + d2_perp)/sqrt(2)
    diag = np.stack([grid[:, k, k] for k in range(len(angles))], axis=1)
    diag_x = np.sqrt(2.0) * angles                       # geodesic radius

    median_grid_full = np.median(grid, axis=0)
    base = float(min(median_grid_full[:, 0].max(),
                     median_grid_full[0, :].max()))
    if args.threshold is not None:
        T = float(args.threshold)
    else:
        T = args.threshold_frac * base
    print(f"axis maxes (median grid): "
          f"{median_grid_full[:, 0].max():.2f} / "
          f"{median_grid_full[0, :].max():.2f}  "
          f"(base = min = {base:.2f})")
    print(f"threshold T = {T:.2f}")

    # Per-curve median + IQR, restricted to the requested x range for plotting.
    def _stats(g):
        return (np.median(g, axis=0),
                np.percentile(g, 25, axis=0),
                np.percentile(g, 75, axis=0))
    g_med, g_lo, g_hi = _stats(gender_axis_grid)
    t_med, t_lo, t_hi = _stats(tense_axis_grid)
    c_med, c_lo, c_hi = _stats(diag)

    # Random-direction baseline. Each random sweep pkl contributes its
    # axis-1 marginal `l2[:, :, 0]` (response along one random unit
    # direction at alpha_2=0). Per-direction curve = median over anchors;
    # IQR across the n_random per-direction curves.
    rand_curve = None
    if args.n_random > 0:
        rand_pat = (f"{DATA_DIR}/sweep2d_{args.target}_L{args.layer}_"
                    f"*__*_fineweb_60deg_dirrandom.pkl")
        rand_files = sorted(glob.glob(rand_pat))[: args.n_random]
        rand_per_dir = []
        for fp in rand_files:
            with open(fp, "rb") as f:
                dr = pickle.load(f)
            rg = np.asarray(dr["l2"], dtype=float)            # (n_anchors, n, n)
            rand_per_dir.append(np.median(rg[:, :, 0], axis=0))
        if rand_per_dir:
            rand_stack = np.stack(rand_per_dir, axis=0)        # (n_dirs, n_angles)
            r_med = np.median(rand_stack, axis=0)
            r_lo  = np.percentile(rand_stack, 25, axis=0)
            r_hi  = np.percentile(rand_stack, 75, axis=0)
            rand_curve = (r_med, r_lo, r_hi, len(rand_per_dir))
            print(f"random baseline: pooled {len(rand_per_dir)} sweeps "
                  f"from {rand_pat}")
        else:
            print(f"no random sweeps found at {rand_pat}; "
                  f"skipping random baseline")

    # Plateau-breaking angles from each median curve.
    a_g = crossing(angles, g_med, T)
    a_t = crossing(angles, t_med, T)
    a_c = crossing(diag_x, c_med, T)
    a_r = crossing(angles, rand_curve[0], T) if rand_curve else float("nan")
    print(f"plateau-breaking angles:  {args.d1}={a_g:.2f}  "
          f"{args.d2}={a_t:.2f}  combo={a_c:.2f}  "
          f"random={a_r:.2f}  (deg)")

    # Plot.
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    C_D1 = "#ff7f00"        # gender / d1: orange
    C_D2 = "#377eb8"        # tense  / d2: blue
    C_CB = "#3aa54a"        # combo: green
    C_RD = "#888888"        # random baseline: grey
    C_T  = "#444444"

    def _plot(x, med, lo, hi, color, label, lw=2.4, ls="-"):
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        ax.plot(x, med, ls, color=color, lw=lw, label=label)

    _plot(angles, g_med, g_lo, g_hi, C_D1, args.d1)
    _plot(angles, t_med, t_lo, t_hi, C_D2, args.d2)
    _plot(diag_x, c_med, c_lo, c_hi, C_CB,
          f"{args.d1} + {args.d2}")
    if rand_curve is not None:
        r_med, r_lo, r_hi, _n_dirs = rand_curve
        _plot(angles, r_med, r_lo, r_hi, C_RD,
              "Random", lw=2.0, ls="--")

    ax.axhline(T, color=C_T, ls=":", lw=1.3,
               label=rf"Threshold ($L^2 = {T:.0f}$)")

    # Plateau-breaking dashed verticals + small annotation tags.
    breakers = [(a_g, C_D1), (a_t, C_D2), (a_c, C_CB)]
    if rand_curve is not None:
        breakers.append((a_r, C_RD))
    for ang_break, color in breakers:
        if np.isfinite(ang_break) and ang_break <= args.max_angle:
            ax.axvline(ang_break, color=color, ls="--", lw=1.4, alpha=0.85)

    ax.set_xlim(0, args.max_angle)
    # cap y just above the threshold + a margin so the post-plateau
    # response stays visible without dominating the panel
    y_top_candidates = [T * 1.6,
                         float(np.nanmax([g_hi[angles <= args.max_angle].max(),
                                          t_hi[angles <= args.max_angle].max(),
                                          c_hi[diag_x <= args.max_angle].max()
                                          if np.any(diag_x <= args.max_angle)
                                          else 0.0]))]
    ax.set_ylim(0, max(y_top_candidates) * 1.05)

    ax.set_xlabel(r"Perturbation angle $\alpha$ (deg)", fontsize=13)
    ax.set_ylabel(r"$L^2$ distance at penultimate layer", fontsize=13)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc="upper left")

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    stem = (f"combo_sweep_{args.target}_L{args.layer}_"
            f"{args.d1}_{args.d2}_to{int(args.max_angle)}deg{tag}")
    out_png = os.path.join(OUT_DIR, f"{stem}.png")
    out_pdf = os.path.join(OUT_DIR, f"{stem}.pdf")
    plt.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
