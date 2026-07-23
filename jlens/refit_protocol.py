#!/usr/bin/env python3
"""Re-fit both direction families under the paper's per-pair threshold
protocol, so the beeswarm columns are aggregated identically.

Protocol (matches `fit_pairs.py --per_pair_threshold --exact`):
    T_pair = 0.50 * min(axis-1 max, axis-2 max) on the pair's own median grid
    robust_p_fit_fixed_l2 with factors (0.5, 1.0, 2.0), exact_geodesic=True
    headline cell = ("1.0xT", <window>)

The one thing that cannot be matched is the angle window: contrastive sweeps
run to 60 deg, the J-lens sweeps only to 20 deg (finer grid, narrower cone),
so J-lens is fit at its own 20 deg limit. Everything else is identical.

Run from the repo root (needs scripts/lib/superellipse.py):
    python jlens/refit_protocol.py
"""
from __future__ import annotations
import os, sys, glob, pickle, time
sys.path.insert(0, ".")
import numpy as np

from scripts.lib.superellipse import robust_p_fit_fixed_l2

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEPS = os.path.join(HERE, "data", "sweeps")
OUT = os.path.join(HERE, "data")

PER_PAIR_F = 0.50
N_BOOT = 200


def t_pair(grid):
    mg = np.median(grid, axis=0)
    base = min(float(mg[:, 0].max()), float(mg[0, :].max()))
    return PER_PAIR_F * base


def fit_one(angles, grid, window):
    return robust_p_fit_fixed_l2(angles, grid, t_pair(grid),
                                 max_alphas=(window,),
                                 n_bootstrap=N_BOOT,
                                 exact_geodesic=True)


def refit_contrastive():
    out = {}
    files = sorted(glob.glob(os.path.join(SWEEPS, "sweep2d_qwen36_*.pkl")))
    for fp in files:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        a, b = d["direction_labels"]
        out[(a, b)] = fit_one(d["angles_deg"], d["l2"], 60.0)
    path = os.path.join(OUT, "fits_contrastive_thrpair_exact.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"  contrastive: {len(out)} pairs -> {os.path.basename(path)}")
    return out


def refit_jlens(tag, glob_pat):
    out = {}
    files = sorted(glob.glob(os.path.join(SWEEPS, glob_pat)))
    for fp in files:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        key = (d["category"], int(d["pair_idx"]))
        out[key] = fit_one(d["angles_deg"], d["l2"], 20.0)
    path = os.path.join(OUT, f"fits_jlens_{tag}_thrpair_exact.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"  jlens {tag}: {len(out)} pairs -> {os.path.basename(path)}")
    return out


def summarize(fits, label, window):
    cell = ("1.0xT", window)
    ps = [f[cell]["p_median"] for f in fits.values()
          if cell in f and np.isfinite(f[cell].get("p_median", np.nan))]
    ps = np.asarray(ps)
    print(f"  {label:22s} n={len(ps):3d}  median={np.median(ps):.3f}")


def main():
    t0 = time.time()
    print("Re-fitting under per-pair threshold + exact geodesic...")
    con = refit_contrastive()
    zh = refit_jlens("zh100", "sweep_zh100_*.pkl")
    en = refit_jlens("en_concrete", "sweep_en_concrete_*.pkl")
    print("\nHeadline cell (1.0xT):")
    summarize(con, "Contrastive", 60.0)
    summarize(zh, "J-Lens ZH", 20.0)
    summarize(en, "J-Lens EN (concrete)", 20.0)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
