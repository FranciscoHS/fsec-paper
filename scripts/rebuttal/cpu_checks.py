"""Rebuttal checks that only need cached fits / sweeps (no GPU).

E1 (R1 Q3): contrastive pairs kept vs dropped by the |cos| < 0.1 filter.
E2 (R1 Q2): contrastive exponents per anchor source (fineweb, wiki_en,
            wiki_zh, code), overlap-filtered.
E3 (R2 Q2): refit p from the cached sweeps at per-pair threshold fractions
            f in {0.05, 0.1, ..., 0.5}, for contrastive pairs and the PCA,
            random and random-difference baselines.

Canonical protocol throughout: Gemma-2-9B L2, per-pair threshold, exact
geodesic normalization, 60-degree window, point estimate. The 33 paper
directions exclude HonestyShort, TensePresent, Formal (as in
render_figures.sh). CIs are the direction-level cluster bootstrap used in the
paper (scripts/plotting/plot_robustness_beeswarm._bootstrap_pooled_mean_ci),
extended here to also return a median CI.

Usage: python scripts/rebuttal/cpu_checks.py [--e1] [--e2] [--e3]
Outputs: results/rebuttal/*.json (and stdout tables).
"""
from __future__ import annotations
import argparse, glob, json, os, pickle, sys
import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, ".")
from scripts.lib.superellipse import robust_p_fit_fixed_l2
from scripts.plotting.plot_robustness_beeswarm import _load_dom_overlaps

EXCLUDE = {"HonestyShort", "TensePresent", "Formal"}
CELL = ("1.0xT", 60.0)
OUT = "results/rebuttal"
OVERLAP_MAX = 0.1


# ----------------------------------------------------------------- stats
def cluster_bootstrap(pairs: dict, n_iter: int = 5000, seed: int = 0):
    """Direction-level cluster bootstrap (resample directions, rebuild all
    pairs, drop self-pairs). Returns point/CI for mean and median."""
    rng = np.random.RandomState(seed)
    pairs = {k: v for k, v in pairs.items() if np.isfinite(v)}
    dirs = sorted({d for k in pairs for d in k})
    idx = {d: i for i, d in enumerate(dirs)}
    n = len(dirs)
    P = np.full((n, n), np.nan)
    for k, p in pairs.items():
        a, b = sorted(k)
        P[idx[a], idx[b]] = P[idx[b], idx[a]] = p
    means, meds = np.empty(n_iter), np.empty(n_iter)
    for it in range(n_iter):
        r = rng.randint(0, n, size=n)
        ii, jj = np.triu_indices(n, k=1)
        a, b = r[ii], r[jj]
        v = P[a, b][a != b]
        v = v[np.isfinite(v)]
        means[it], meds[it] = v.mean(), np.median(v)
    vals = np.array(list(pairs.values()))
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "mean_ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "median": float(np.median(vals)),
        "median_ci": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
        "frac_gt_2": float(np.mean(vals > 2)),
    }


def fmt(s):
    return (f"n={s['n']:4d}  median {s['median']:.2f} [{s['median_ci'][0]:.2f}, {s['median_ci'][1]:.2f}]"
            f"  mean {s['mean']:.2f} [{s['mean_ci'][0]:.2f}, {s['mean_ci'][1]:.2f}]"
            f"  frac>2 {s['frac_gt_2']:.2f}")


# ----------------------------------------------------------------- loading
def load_fits(path):
    with open(path, "rb") as f:
        fits = pickle.load(f)
    out = {}
    for k, v in fits.items():
        if not isinstance(k, tuple) or set(k) & EXCLUDE or CELL not in v:
            continue
        p = v[CELL]["p_median"]
        if p is not None and np.isfinite(p):
            out[frozenset(k)] = float(p)
    return out


def split_by_overlap(pairs, ov):
    kept = {k: p for k, p in pairs.items() if ov[k] < OVERLAP_MAX}
    dropped = {k: p for k, p in pairs.items() if ov[k] >= OVERLAP_MAX}
    return kept, dropped


# ----------------------------------------------------------------- E1
def e1(ov):
    pairs = load_fits("results/fits/fits_gemma_L2_thrpair_exact.pkl")
    kept, dropped = split_by_overlap(pairs, ov)
    res = {"kept": cluster_bootstrap(kept), "dropped": cluster_bootstrap(dropped)}
    u = mannwhitneyu(list(kept.values()), list(dropped.values()), alternative="two-sided")
    res["mannwhitney_two_sided_p"] = float(u.pvalue)
    # dropped pairs binned by overlap: does p fall as overlap (distortion) grows?
    bins = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01)]
    res["dropped_by_overlap"] = {}
    for lo, hi in bins:
        v = np.array([p for k, p in dropped.items() if lo <= ov[k] < hi])
        if len(v):
            res["dropped_by_overlap"][f"{lo}-{hi}"] = {
                "n": int(len(v)), "median": float(np.median(v)),
                "mean": float(v.mean()), "frac_gt_2": float(np.mean(v > 2))}
    print("\n=== E1: overlap filter (contrastive, Gemma L2, default setup) ===")
    print(f"kept    (|cos| < 0.1):  {fmt(res['kept'])}")
    print(f"dropped (|cos| >= 0.1): {fmt(res['dropped'])}")
    print(f"Mann-Whitney two-sided p = {res['mannwhitney_two_sided_p']:.3g}")
    for b, s in res["dropped_by_overlap"].items():
        print(f"  dropped, |cos| in {b}: n={s['n']:3d} median {s['median']:.2f} "
              f"mean {s['mean']:.2f} frac>2 {s['frac_gt_2']:.2f}")
    return res


# ----------------------------------------------------------------- E2
def e2(ov):
    srcs = {"fineweb": "", "wiki_en": "_srcwiki_en", "wiki_zh": "_srcwiki_zh", "code": "_srccode"}
    res = {}
    print("\n=== E2: anchor source (contrastive, overlap-filtered) ===")
    for name, suf in srcs.items():
        kept, _ = split_by_overlap(load_fits(f"results/fits/fits_gemma_L2{suf}_thrpair_exact.pkl"), ov)
        res[name] = cluster_bootstrap(kept)
        print(f"{name:8s} {fmt(res[name])}")
    return res


# ----------------------------------------------------------------- E3
FAMILIES = {
    # family: (sweep glob suffix, overlap filter applies?)
    "contrastive": ("_fineweb_60deg.pkl", True),
    "pca": ("_fineweb_60deg_dirpca_fineweb.pkl", False),
    "random": ("_fineweb_60deg_dirrandom.pkl", False),
    "random_diff": ("_fineweb_60deg_dirrandomdiffavg_fineweb.pkl", "self"),
}
FS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def e3(ov, rd_overlap_path="results/directions/dirs_gemma_L2_randomdiffavg_fineweb.pkl"):
    import torch  # noqa: F401  (direction pickles hold tensors)
    # overlaps for random-diff directions (paper filters them at |cos| < 0.1)
    with open(rd_overlap_path, "rb") as f:
        rd = pickle.load(f)
    rd_vecs = {k: np.asarray(v, dtype=np.float64) for k, v in rd["directions"].items()} \
        if isinstance(rd, dict) and "directions" in rd else None

    res = {}
    print("\n=== E3: fitted p vs threshold fraction f (median grid, per-pair T = f * min axis max) ===")
    for fam, (suf, filt) in FAMILIES.items():
        files = sorted(glob.glob(f"results/sweeps_2d/sweep2d_gemma_L2_*{suf}"))
        per_f = {f: {} for f in FS}
        for fp in files:
            with open(fp, "rb") as fh:
                d = pickle.load(fh)
            a, b = d["direction_labels"]
            key = frozenset((a, b))
            if fam == "contrastive":
                if {a, b} & EXCLUDE or key not in ov or ov[key] >= OVERLAP_MAX:
                    continue
            if fam == "random_diff" and rd_vecs is not None:
                va, vb = rd_vecs[a], rd_vecs[b]
                if abs(va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb)) >= OVERLAP_MAX:
                    continue
            grid = d["l2"]
            mg = np.median(grid, axis=0)
            base = min(float(mg[:, 0].max()), float(mg[0, :].max()))
            for f in FS:
                fit = robust_p_fit_fixed_l2(d["angles_deg"], grid, f * base,
                                            threshold_factors=(1.0,), max_alphas=(60.0,),
                                            n_bootstrap=0, exact_geodesic=True)
                c = fit[("1.0xT", 60.0)]
                if np.isfinite(c["p_median"]):
                    per_f[f][key] = (float(c["p_median"]), int(c["n_pts"]))
        res[fam] = {}
        for f in FS:
            pairs = {k: v[0] for k, v in per_f[f].items()}
            if len(pairs) < 10:
                continue
            s = cluster_bootstrap(pairs, n_iter=2000)
            s["median_contour_pts"] = float(np.median([v[1] for v in per_f[f].values()]))
            res[fam][str(f)] = s
            print(f"{fam:12s} f={f:<5} {fmt(s)}  contour pts {s['median_contour_pts']:.0f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", action="store_true")
    ap.add_argument("--e2", action="store_true")
    ap.add_argument("--e3", action="store_true")
    args = ap.parse_args()
    run_all = not (args.e1 or args.e2 or args.e3)
    os.makedirs(OUT, exist_ok=True)
    ov = _load_dom_overlaps("gemma", 2)
    for flag, fn, name in [(args.e1, e1, "e1_overlap_filter"),
                           (args.e2, e2, "e2_anchor_source"),
                           (args.e3, e3, "e3_threshold_sweep")]:
        if flag or run_all:
            res = fn(ov)
            with open(f"{OUT}/{name}.json", "w") as f:
                json.dump(res, f, indent=1)
            print(f"saved {OUT}/{name}.json")


if __name__ == "__main__":
    main()
