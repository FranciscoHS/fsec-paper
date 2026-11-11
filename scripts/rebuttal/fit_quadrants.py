"""E4 (R2 Q1): fit the four-quadrant sweeps.

For each of the 318 overlap-filtered contrastive pairs (Gemma-2-9B L2, default
setup) we have the original (+d1, +d2) sweep (paper) and new (+,-), (-,+),
(-,-) sweeps from the pod (transformers 4.57, reproducibility-checked).
Each quadrant is fit with the canonical protocol (per-pair threshold f = 0.5,
exact geodesic normalization, 60-degree window).

Why this separates the hypotheses per pair: a tilted ellipse
x^2 + 2*delta*x*y + y^2 = 1 has its cross term flip sign between adjacent
quadrants (x -> -x flips delta), so it bulges outward (looks like p > 2) in
two opposite quadrants and inward (p < 2) in the other two. A superellipse
bulges outward in all four.

Outputs results/rebuttal/e4_quadrants.json and a stdout summary.
"""
from __future__ import annotations
import glob, json, pickle, sys
import numpy as np
sys.path.insert(0, ".")
from scripts.lib.superellipse import robust_p_fit_fixed_l2
from scripts.rebuttal.cpu_checks import cluster_bootstrap, fmt

QDIR = "results/rebuttal/sweeps_quadrants"


def fit(d):
    mg = np.median(d["l2"], axis=0)
    base = min(float(mg[:, 0].max()), float(mg[0, :].max()))
    c = robust_p_fit_fixed_l2(d["angles_deg"], d["l2"], 0.5 * base,
                              threshold_factors=(1.0,), max_alphas=(60.0,),
                              n_bootstrap=0, exact_geodesic=True)[("1.0xT", 60.0)]
    return float(c["p_median"]), float(c["mean_radial_frac"])


def main():
    pairs = {}
    for line in open("results/rebuttal/pairs_quadrants.txt"):
        if line.startswith("#"):
            continue
        a, b, q = line.strip().split(",")
        base = (a.replace("_neg", ""), b.replace("_neg", ""))
        pairs.setdefault(base, {})[q] = (a, b)

    res = {q: {} for q in ("++", "+-", "-+", "--")}
    resid = {q: [] for q in res}
    for (a, b), qs in pairs.items():
        d = pickle.load(open(f"results/sweeps_2d/sweep2d_gemma_L2_{a}__{b}_fineweb_60deg.pkl", "rb"))
        p, r = fit(d)
        res["++"][frozenset((a, b))] = p; resid["++"].append(r)
        for q, (x, y) in qs.items():
            d = pickle.load(open(f"{QDIR}/sweep2d_gemma_L2_{x}__{y}_fineweb_60deg_dirsigned.pkl", "rb"))
            p, r = fit(d)
            res[q][frozenset((a, b))] = p; resid[q].append(r)

    out = {"per_quadrant": {}, "per_pair": {}}
    print("=== E4: four quadrants, 318 contrastive pairs ===")
    for q in res:
        s = cluster_bootstrap(res[q])
        s["median_radial_resid"] = float(np.median(resid[q]))
        out["per_quadrant"][q] = s
        print(f"({q})  {fmt(s)}  median radial resid {100 * s['median_radial_resid']:.2f}%")

    keys = sorted(res["++"], key=sorted)
    P = np.array([[res[q][k] for q in ("++", "+-", "-+", "--")] for k in keys])
    ok = np.all(np.isfinite(P), axis=1)
    P = P[ok]
    gt2 = P > 2
    # per-pair mean over quadrants
    pm = P.mean(1)
    s_all = cluster_bootstrap({k: float(v) for k, v in zip([k for k, o in zip(keys, ok) if o], pm)})
    out["per_pair"] = {
        "n_pairs": int(len(P)),
        "frac_all4_gt2": float(gt2.all(1).mean()),
        "frac_ge3_gt2": float((gt2.sum(1) >= 3).mean()),
        "n_quadrants_gt2_hist": {int(k): int((gt2.sum(1) == k).sum()) for k in range(5)},
        "quadrant_mean_p": s_all,
    }
    # Tilted-ellipse signature: p(++) and p(--) share delta; p(+-) and p(-+)
    # have -delta. Under a tilted ellipse, (p_diag - 2) and (p_anti - 2) have
    # opposite signs; under a superellipse both are > 0.
    diag = P[:, [0, 3]].mean(1) - 2
    anti = P[:, [1, 2]].mean(1) - 2
    out["per_pair"]["frac_diag_and_anti_both_gt2"] = float(((diag > 0) & (anti > 0)).mean())
    out["per_pair"]["frac_opposite_signs"] = float((np.sign(diag) != np.sign(anti)).mean())
    out["per_pair"]["spearman_diag_anti"] = float(
        __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(diag, anti).correlation)
    pp = out["per_pair"]
    print(f"\nper pair (n={pp['n_pairs']}): p>2 in all 4 quadrants {pp['frac_all4_gt2']:.2f}, "
          f"in >=3 {pp['frac_ge3_gt2']:.2f}; histogram of #quadrants with p>2: {pp['n_quadrants_gt2_hist']}")
    print(f"mean over the 4 quadrants: {fmt(s_all)}")
    print(f"diagonal (++,--) and anti-diagonal (+-,-+) means both > 2: {pp['frac_diag_and_anti_both_gt2']:.2f}; "
          f"opposite signs (tilted-ellipse pattern): {pp['frac_opposite_signs']:.2f}; "
          f"Spearman(diag, anti) = {pp['spearman_diag_anti']:.2f}")
    with open("results/rebuttal/e4_quadrants.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/rebuttal/e4_quadrants.json")


if __name__ == "__main__":
    main()
