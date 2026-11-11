"""E5: fit the MLP-neuron direction family (Gemma-2-9B L2, default setup).

Sweeps come from the pod run (transformers 4.57, reproducibility-checked) and
live in results/rebuttal/sweeps_mlpneuron/. Canonical protocol (per-pair
threshold f = 0.5, exact geodesic, 60 degrees); pairs filtered at intra-pair
|cos| < 0.1 as for every other family.

Writes results/fits/fits_gemma_L2_dirmlpneuron_thrpair_exact.pkl (same
structure as fit_pairs.py output, 0.5/1/2 x T cells) and
results/rebuttal/e5_mlpneuron.json.
"""
from __future__ import annotations
import glob, json, pickle, sys
import numpy as np
sys.path.insert(0, ".")
from scripts.lib.superellipse import robust_p_fit_fixed_l2
from scripts.rebuttal.cpu_checks import cluster_bootstrap, fmt

CELL = ("1.0xT", 60.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mlpneuron", choices=["mlpneuron", "mlpneuronrand"])
    fam = ap.parse_args().family
    tag = {"mlpneuron": "e5_mlpneuron", "mlpneuronrand": "e5_mlpneuronrand"}[fam]
    dirs = pickle.load(open(f"results/directions/dirs_gemma_L2_{fam}.pkl", "rb"))["directions"]
    vec = {k: np.asarray(v, dtype=np.float64) for k, v in dirs.items()}
    full, kept, allp, resid = {}, {}, {}, []
    for fp in sorted(glob.glob(f"results/rebuttal/sweeps_{fam}/*_dir{fam}.pkl")):
        d = pickle.load(open(fp, "rb"))
        a, b = d["direction_labels"]
        mg = np.median(d["l2"], axis=0)
        base = min(float(mg[:, 0].max()), float(mg[0, :].max()))
        fit = robust_p_fit_fixed_l2(d["angles_deg"], d["l2"], 0.5 * base, max_alphas=(60.0,),
                                    n_bootstrap=0, exact_geodesic=True)
        full[(a, b)] = fit
        p = fit[CELL]["p_median"]
        if not np.isfinite(p):
            continue
        cos = abs(vec[a] @ vec[b]) / (np.linalg.norm(vec[a]) * np.linalg.norm(vec[b]))
        allp[frozenset((a, b))] = float(p)
        if cos < 0.1:
            kept[frozenset((a, b))] = float(p)
            resid.append(fit[CELL]["mean_radial_frac"])
    with open(f"results/fits/fits_gemma_L2_dir{fam}_thrpair_exact.pkl", "wb") as f:
        pickle.dump(full, f)
    res = {"kept": cluster_bootstrap(kept), "all": cluster_bootstrap(allp),
           "median_radial_resid_kept": float(np.median(resid))}
    print(f"{fam}, |cos|<0.1: {fmt(res['kept'])}  median radial resid {100*res['median_radial_resid_kept']:.2f}%")
    print(f"{fam}, all pairs: {fmt(res['all'])}")
    json.dump(res, open(f"results/rebuttal/{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
