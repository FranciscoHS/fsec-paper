"""Methodology check (Stefan): does fitting the superellipse exponent p
PER ANCHOR and then aggregating give tighter error bars than the current
"fit once on the median-over-anchors grid" procedure?

For each contrastive pair we compute, with the exact (sin-geodesic) fit
matching plot_fig3_boundary.py:
  A) p_median   : fit on the median-over-anchors L2 grid (current method)
  B) p_anchor   : fit each anchor's own grid, then mean over anchors
We then compare, across pairs, the bootstrap CI of the mean p under each
method (the beeswarm error bar), and report the typical within-pair
anchor spread that method B exposes.

Run: python scripts/pfit_anchor_vs_median.py
"""
from __future__ import annotations
import sys, glob, re, pickle
sys.path.insert(0, ".")
import numpy as np
from scripts.lib.superellipse import extract_contour, axis_intercept, fit_superellipse

LANG = {"Arabic","Chinese","Dutch","French","German","Italian","Japanese",
        "Portuguese","Russian","Spanish"}
PROG = {"Cpp","Go","Haskell","Java","JavaScript","Lisp","Php","Python","Rust",
        "TypeScript","C"}


def fit_p(grid2d, angles, frac=0.5):
    """Exact-geodesic superellipse p for one (n,n) L2 grid, or None."""
    base = min(grid2d[:, 0].max(), grid2d[0, :].max())
    thr = frac * base
    t1 = axis_intercept(angles, grid2d[:, 0], thr)
    t2 = axis_intercept(angles, grid2d[0, :], thr)
    if not (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0):
        return None
    cn = extract_contour(angles, grid2d, thr).astype(float)
    if cn.size == 0:
        return None
    r = np.hypot(cn[:, 0], cn[:, 1]); rs = np.where(r > 1e-12, r, 1.0)
    rr = np.deg2rad(r)
    n = cn.copy()
    n[:, 0] = (np.sin(rr) / np.sin(np.deg2rad(t1))) * (cn[:, 0] / rs)
    n[:, 1] = (np.sin(rr) / np.sin(np.deg2rad(t2))) * (cn[:, 1] / rs)
    keep = (n[:, 0] > 0.05) & (n[:, 1] > 0.05) & (n[:, 0] < 1.5) & (n[:, 1] < 1.5)
    n = n[keep]
    if len(n) < 8:
        return None
    p = fit_superellipse(n[:, 0], n[:, 1])["p"]
    return float(p) if np.isfinite(p) else None


def boot_ci(vals, n_iter=10000, seed=0):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)])
    rng = np.random.RandomState(seed)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_iter)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return vals.mean(), lo, hi


def main():
    files = sorted(glob.glob(
        "results/sweeps_2d/sweep2d_gemma_L2_*_fineweb_60deg.pkl"))
    rows = []
    for f in files:
        m = re.search(r"L2_(.+?)__(.+?)_fineweb", f)
        a, b = m.group(1), m.group(2)
        d = pickle.load(open(f, "rb"))
        grid = np.asarray(d["l2"], float)           # (n_anchors, n, n)
        angles = np.asarray(d["angles_deg"], float)
        p_med = fit_p(np.median(grid, axis=0), angles)
        if p_med is None:
            continue
        pa = [fit_p(grid[i], angles) for i in range(grid.shape[0])]
        pa = [x for x in pa if x is not None]
        if len(pa) < 5:
            continue
        rows.append({"pair": f"{a}x{b}", "a": a, "b": b,
                     "p_med": p_med, "p_anchor_mean": float(np.mean(pa)),
                     "p_anchor_std": float(np.std(pa, ddof=1)),
                     "n_ok": len(pa), "n_tot": grid.shape[0]})
    print(f"pairs with valid fits: {len(rows)}\n")

    # Per-pair comparison summary
    dmed = np.array([r["p_med"] for r in rows])
    danc = np.array([r["p_anchor_mean"] for r in rows])
    stds = np.array([r["p_anchor_std"] for r in rows])
    print("PER-PAIR estimator agreement (median-grid vs per-anchor-mean):")
    print(f"  corr = {np.corrcoef(dmed, danc)[0,1]:.3f}")
    print(f"  mean |p_med - p_anchor_mean| = {np.mean(np.abs(dmed-danc)):.3f}")
    print(f"  median within-pair anchor std of p = {np.median(stds):.3f}")
    print(f"  -> per-anchor gives a within-pair CI of ~+/-{1.96*np.median(stds)/np.sqrt(np.median([r['n_ok'] for r in rows])):.3f} (median)\n")

    # Across-pair beeswarm error bar under each method
    for name, vals in [("median-grid (current)", dmed),
                       ("per-anchor-mean", danc)]:
        mean, lo, hi = boot_ci(vals)
        print(f"ACROSS-PAIR mean p [{name}]: {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"half-width {0.5*(hi-lo):.4f}")

    # A few biggest disagreements
    rows.sort(key=lambda r: abs(r["p_med"]-r["p_anchor_mean"]), reverse=True)
    print("\nLargest median-vs-anchor disagreements:")
    for r in rows[:8]:
        print(f"  {r['pair']:24} p_med={r['p_med']:.2f}  "
              f"p_anchor={r['p_anchor_mean']:.2f}+/-{r['p_anchor_std']:.2f}  "
              f"(n_ok={r['n_ok']}/{r['n_tot']})")


if __name__ == "__main__":
    main()
