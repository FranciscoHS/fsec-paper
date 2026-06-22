"""Methodology check (Stefan): does the superellipse fit better in
sin(alpha) coordinates (exact geodesic, current) or in raw-alpha
coordinates (small-angle)? Compares the fit residual (mean radial
fraction) per pair on the median-over-anchors grid.

Run: python scripts/pfit_sin_vs_raw.py
"""
from __future__ import annotations
import sys, glob, re, pickle
sys.path.insert(0, ".")
import numpy as np
from scripts.lib.superellipse import extract_contour, axis_intercept, fit_superellipse


def fit_resid(grid2d, angles, mode, frac=0.5):
    """Return (p, mean_radial_frac) for 'sin' or 'raw' normalization."""
    base = min(grid2d[:, 0].max(), grid2d[0, :].max())
    thr = frac * base
    t1 = axis_intercept(angles, grid2d[:, 0], thr)
    t2 = axis_intercept(angles, grid2d[0, :], thr)
    if not (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0):
        return None
    cn = extract_contour(angles, grid2d, thr).astype(float)
    if cn.size == 0:
        return None
    n = cn.copy()
    if mode == "sin":
        r = np.hypot(cn[:, 0], cn[:, 1]); rs = np.where(r > 1e-12, r, 1.0)
        rr = np.deg2rad(r)
        n[:, 0] = (np.sin(rr) / np.sin(np.deg2rad(t1))) * (cn[:, 0] / rs)
        n[:, 1] = (np.sin(rr) / np.sin(np.deg2rad(t2))) * (cn[:, 1] / rs)
    else:  # raw alpha (linear)
        n[:, 0] = cn[:, 0] / t1
        n[:, 1] = cn[:, 1] / t2
    keep = (n[:, 0] > 0.05) & (n[:, 1] > 0.05) & (n[:, 0] < 1.5) & (n[:, 1] < 1.5)
    n = n[keep]
    if len(n) < 8:
        return None
    f = fit_superellipse(n[:, 0], n[:, 1])
    return float(f["p"]), float(f["mean_radial_frac"])


def main():
    files = sorted(glob.glob(
        "results/sweeps_2d/sweep2d_gemma_L2_*_fineweb_60deg.pkl"))
    sin_res, raw_res, sin_p, raw_p, sin_wins = [], [], [], [], 0
    n = 0
    for f in files:
        d = pickle.load(open(f, "rb"))
        grid = np.median(np.asarray(d["l2"], float), axis=0)
        angles = np.asarray(d["angles_deg"], float)
        rs = fit_resid(grid, angles, "sin")
        rr = fit_resid(grid, angles, "raw")
        if rs is None or rr is None:
            continue
        n += 1
        sin_p.append(rs[0]); sin_res.append(rs[1])
        raw_p.append(rr[0]); raw_res.append(rr[1])
        if rs[1] < rr[1]:
            sin_wins += 1
    sin_res, raw_res = np.array(sin_res), np.array(raw_res)
    print(f"pairs: {n}\n")
    print(f"mean fit residual  sin = {sin_res.mean()*100:.3f}%   "
          f"raw = {raw_res.mean()*100:.3f}%")
    print(f"median fit residual sin = {np.median(sin_res)*100:.3f}%   "
          f"raw = {np.median(raw_res)*100:.3f}%")
    print(f"sin has the lower residual in {sin_wins}/{n} pairs "
          f"({100*sin_wins/n:.0f}%)")
    print(f"\nmean fitted p   sin = {np.mean(sin_p):.3f}   raw = {np.mean(raw_p):.3f}")


if __name__ == "__main__":
    main()
