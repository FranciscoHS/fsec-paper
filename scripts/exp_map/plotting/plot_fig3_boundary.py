"""Fig 3 style iso-plateau boundary plot for an exp-map sweep2d pair.

Cloned from scripts/perturbation/plot_fig3_formal_pt.py — same layout
(p=2 circle in blue, p=infinity square in dashed orange, data as small
grey circles, polar (theta cos phi / theta_1, theta sin phi / theta_2)
axes) but reads the new exp_map sweep2d PKLs and uses the new
lib/superellipse fit code.

Usage:
  python scripts/exp_map/plotting/plot_fig3_boundary.py \\
      --target gemma --d1 Gender --d2 Tense --layer 2 --thresh 50
"""
from __future__ import annotations
import os, sys, argparse, pickle
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.exp_map.lib.superellipse import (
    extract_contour, axis_intercept, fit_superellipse, superellipse_curve,
)

OUT_DIR = "results/exp_map/figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--d1", required=True)
    ap.add_argument("--d2", required=True)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--thresh", type=float, default=None,
                    help="absolute L^2 threshold; if omitted, uses 50%% of "
                         "min(max(L2[:,0]), max(L2[0,:])) on the mean grid")
    ap.add_argument("--max_alpha", type=float, default=60.0)
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--exact", action="store_true",
                    help="use the exact-geodesic normalization "
                         "(sin(alpha) cos phi / sin(alpha_1), sin(alpha) "
                         "sin phi / sin(alpha_2)) instead of the small-"
                         "angle approximation (alpha cos phi / alpha_1). "
                         "Adds '_exact' to the default output filename.")
    args = ap.parse_args()

    pkl_path = (f"results/exp_map/data/sweeps_2d/sweep2d_{args.target}_L"
                f"{args.layer}_{args.d1}__{args.d2}_fineweb_"
                f"{int(args.max_alpha)}deg.pkl")
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    angles = d["angles_deg"]
    grid = d["l2"]                    # (n_anchors, n, n)
    median_grid = np.median(grid, axis=0)

    # Threshold
    if args.thresh is not None:
        thresh = float(args.thresh)
    else:
        base = min(median_grid[:, 0].max(), median_grid[0, :].max())
        thresh = 0.5 * base
    print(f"threshold = {thresh:.3f}  (axis maxes: "
          f"{median_grid[:, 0].max():.2f} / {median_grid[0, :].max():.2f})")

    theta1 = axis_intercept(angles, median_grid[:, 0], thresh)
    theta2 = axis_intercept(angles, median_grid[0, :], thresh)
    print(f"theta_{args.d1} = {theta1:.2f} deg, "
          f"theta_{args.d2} = {theta2:.2f} deg")
    if not (np.isfinite(theta1) and np.isfinite(theta2)
            and theta1 > 0 and theta2 > 0):
        raise RuntimeError("axis intercepts not in range; pick a smaller "
                           "threshold or check the L2 grid")

    # Contour points + normalize. Exact-geodesic mode matches the
    # protocol in lib/superellipse.fit_p_one_pass(exact_geodesic=True).
    contour = extract_contour(angles, median_grid, thresh)
    if contour.size == 0:
        raise RuntimeError("no contour at this threshold")
    cn = contour.astype(float).copy()
    if args.exact:
        r_deg = np.hypot(cn[:, 0], cn[:, 1])
        r_safe = np.where(r_deg > 1e-12, r_deg, 1.0)
        r_rad = np.deg2rad(r_deg)
        t1_rad = np.deg2rad(theta1)
        t2_rad = np.deg2rad(theta2)
        cn[:, 0] = (np.sin(r_rad) / np.sin(t1_rad)) * (cn[:, 0] / r_safe)
        cn[:, 1] = (np.sin(r_rad) / np.sin(t2_rad)) * (cn[:, 1] / r_safe)
    else:
        cn[:, 0] /= theta1; cn[:, 1] /= theta2
    # filter to first quadrant, away from axes (matches fit_superellipse)
    keep = (cn[:, 0] > 0.05) & (cn[:, 1] > 0.05) \
        & (cn[:, 0] < 1.5) & (cn[:, 1] < 1.5)
    cn = cn[keep]

    fit = fit_superellipse(cn[:, 0], cn[:, 1])
    p_fit = fit["p"]
    print(f"Fitted p = {p_fit:.3f}   "
          f"mean_radial_frac = {fit['mean_radial_frac']:.3f}   "
          f"n_pts = {fit['n_pts']}")

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.set_xticks(np.arange(0, 1.21, 0.2))
    ax.set_yticks(np.arange(0, 1.21, 0.2))
    ax.grid(True, which="major", color="#bbbbbb", lw=0.5, alpha=0.45)

    xi, yi = superellipse_curve(20)
    ax.plot(xi, yi, ls=(0, (1.8, 2.5)), color="#D9822B", lw=1.6,
            alpha=0.95,
            label=r"Per-feature threshold ($p\to\infty$)")
    xc, yc = superellipse_curve(2)
    ax.plot(xc, yc, "-", color="#1F77B4", lw=2.6, alpha=0.95,
            label=r"Euclidean combination ($p=2$)")

    cont_angle = np.arctan2(cn[:, 1], cn[:, 0])
    order = np.argsort(cont_angle)
    res_pct = 100.0 * fit["mean_radial_frac"]
    # Fitted super-ellipse curve, drawn under the data points so the
    # markers visually sit on the line they're being fit to.
    xf, yf = superellipse_curve(p_fit)
    ax.plot(xf, yf, "-", color="#3a7d3a", lw=2.0, alpha=0.95,
            label=rf"Super-ellipse fit ($p_{{\mathrm{{fit}}}} = {p_fit:.2f}$)",
            zorder=4)
    ax.plot(cn[order, 0], cn[order, 1], "o",
            color="#3a3a3a", markersize=4.0, alpha=0.85,
            markeredgecolor="none",
            label="Data",
            zorder=5)
    # Residual reported separately for the caption; printed to stdout.
    print(f"Caption stat: residual = {res_pct:.2f}%  (n_pts = "
          f"{fit['n_pts']})")

    ax.set_xlim(-0.03, 1.15); ax.set_ylim(-0.03, 1.15)
    ax.set_aspect("equal")
    if args.exact:
        x_lab = (rf"$\sin\alpha(\varphi)\cos\varphi / \sin\alpha_1$  "
                 rf"({args.d1})")
        y_lab = (rf"$\sin\alpha(\varphi)\sin\varphi / \sin\alpha_2$  "
                 rf"({args.d2})")
    else:
        x_lab = rf"$\alpha(\varphi)\cos\varphi / \alpha_1$  ({args.d1})"
        y_lab = rf"$\alpha(\varphi)\sin\varphi / \alpha_2$  ({args.d2})"
    ax.set_xlabel(x_lab, fontsize=12)
    ax.set_ylabel(y_lab, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="lower left", frameon=True, framealpha=0.95)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    if args.exact: tag += "_exact"
    out_png = os.path.join(OUT_DIR,
        f"fig3_boundary_{args.d1}_{args.d2}_{args.target}_L{args.layer}{tag}.png")
    out_pdf = out_png.replace(".png", ".pdf")
    plt.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
