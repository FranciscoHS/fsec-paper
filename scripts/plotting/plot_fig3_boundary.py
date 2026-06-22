"""Fig 3 style iso-plateau boundary plot for an exp-map sweep2d pair.

Plots the iso-response contour (one point per polar angle phi_i) in
raw (alpha_1, alpha_2) degrees, with the p=2 ellipse and p=infinity
rectangle (semi-axes theta_1, theta_2) as references and the fitted
superellipse overlaid.

Usage:
  python scripts/plotting/plot_fig3_boundary.py \\
      --target gemma --d1 Gender --d2 Tense --layer 2 --thresh 50
"""
from __future__ import annotations
import os, sys, argparse, pickle
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.lib.superellipse import (
    extract_contour, axis_intercept, fit_superellipse, superellipse_curve,
)

OUT_DIR = "results/figures"


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

    pkl_path = (f"results/sweeps_2d/sweep2d_{args.target}_L"
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

    # Contour points in raw (alpha_1, alpha_2) degrees.
    contour = extract_contour(angles, median_grid, thresh)
    if contour.size == 0:
        raise RuntimeError("no contour at this threshold")
    cn_raw = contour.astype(float).copy()           # raw degrees, for plotting

    # Fit superellipse in normalised (theta_1, theta_2) coords; the fit
    # routine assumes unit reference, so we still divide by theta1, theta2
    # here. The plotting axes below stay in raw degrees.
    cn_norm = cn_raw.copy()
    if args.exact:
        r_deg = np.hypot(cn_norm[:, 0], cn_norm[:, 1])
        r_safe = np.where(r_deg > 1e-12, r_deg, 1.0)
        r_rad = np.deg2rad(r_deg)
        t1_rad = np.deg2rad(theta1)
        t2_rad = np.deg2rad(theta2)
        cn_norm[:, 0] = (np.sin(r_rad) / np.sin(t1_rad)) * (cn_norm[:, 0] / r_safe)
        cn_norm[:, 1] = (np.sin(r_rad) / np.sin(t2_rad)) * (cn_norm[:, 1] / r_safe)
    else:
        cn_norm[:, 0] /= theta1; cn_norm[:, 1] /= theta2
    keep = (cn_norm[:, 0] > 0.05) & (cn_norm[:, 1] > 0.05) \
        & (cn_norm[:, 0] < 1.5) & (cn_norm[:, 1] < 1.5)
    cn_norm = cn_norm[keep]
    cn_raw  = cn_raw[keep]

    fit = fit_superellipse(cn_norm[:, 0], cn_norm[:, 1])
    p_fit = fit["p"]
    print(f"Fitted p = {p_fit:.3f}   "
          f"mean_radial_frac = {fit['mean_radial_frac']:.3f}   "
          f"n_pts = {fit['n_pts']}")

    # Plot in the SAME space the super-ellipse was fit in. For --exact the
    # fit uses sin-geodesic coords (sin(alpha)cos phi / sin(theta_i)), so the
    # displayed axes, reference curves, data and fitted curve must all go
    # through sin() too -- otherwise the green fit (a super-ellipse in
    # sin-space) would be drawn in raw-degree space where it is no longer
    # that super-ellipse. The display semi-axes are sin(theta_i); for the
    # small-angle (non-exact) fit sin() is dropped and the axes stay in raw
    # degrees. In both cases the displayed points are cn_norm * (A1, A2),
    # which reduces to cn_raw exactly when non-exact (A_i = theta_i).
    if args.exact:
        A1 = float(np.sin(np.deg2rad(theta1)))
        A2 = float(np.sin(np.deg2rad(theta2)))
    else:
        A1, A2 = theta1, theta2
    disp = cn_norm * np.array([A1, A2])

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    if args.exact:
        # sin-space axes span ~[0, sin(theta)] < 1; let matplotlib pick
        # round sin-unit ticks rather than the degree spacing below.
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    else:
        # tick spacing chosen to give ~5–7 ticks across each raw-degree axis
        def _tickspacing(span):
            targets = [0.5, 1, 2, 5, 10, 15, 20, 25, 50]
            return next((t for t in targets if span / t <= 8), 50)
        sp1 = _tickspacing(A1 * 1.15)
        sp2 = _tickspacing(A2 * 1.15)
        ax.set_xticks(np.arange(0, A1 * 1.18 + 1e-6, sp1))
        ax.set_yticks(np.arange(0, A2 * 1.18 + 1e-6, sp2))
    ax.grid(True, which="major", color="#bbbbbb", lw=0.5, alpha=0.45)

    # p=infinity rectangle with semi-axes (A1, A2)
    xi, yi = superellipse_curve(20)
    ax.plot(A1 * xi, A2 * yi, ls=(0, (1.8, 2.5)),
            color="#D9822B", lw=1.6, alpha=0.95,
            label=r"Per-feature threshold ($p\to\infty$)")
    # p=2 ellipse with semi-axes (A1, A2)
    xc, yc = superellipse_curve(2)
    ax.plot(A1 * xc, A2 * yc, "-",
            color="#1F77B4", lw=2.6, alpha=0.95,
            label=r"Euclidean combination ($p=2$)")

    cont_angle = np.arctan2(disp[:, 1], disp[:, 0])
    order = np.argsort(cont_angle)
    res_pct = 100.0 * fit["mean_radial_frac"]
    # Fitted super-ellipse, scaled to (A1, A2).
    xf, yf = superellipse_curve(p_fit)
    ax.plot(A1 * xf, A2 * yf, "-",
            color="#3a7d3a", lw=2.0, alpha=0.95,
            label=rf"Super-ellipse fit ($p_{{\mathrm{{fit}}}} = {p_fit:.2f}$)",
            zorder=4)
    ax.plot(disp[order, 0], disp[order, 1], "o",
            color="#3a3a3a", markersize=4.0, alpha=0.85,
            markeredgecolor="none",
            label="Data",
            zorder=5)
    # Residual reported separately for the caption; printed to stdout.
    print(f"Caption stat: residual = {res_pct:.2f}%  (n_pts = "
          f"{fit['n_pts']})")

    ax.set_xlim(-0.03 * A1, 1.15 * A1)
    ax.set_ylim(-0.03 * A2, 1.15 * A2)
    ax.set_aspect("equal")
    if args.exact:
        x_lab = rf"$\sin\alpha(\varphi_i)\cos\varphi_i$  ({args.d1})"
        y_lab = rf"$\sin\alpha(\varphi_i)\sin\varphi_i$  ({args.d2})"
    else:
        x_lab = rf"$\alpha(\varphi_i)\cos\varphi_i$  ({args.d1}, deg)"
        y_lab = rf"$\alpha(\varphi_i)\sin\varphi_i$  ({args.d2}, deg)"
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
