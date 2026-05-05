"""Run the robustness protocol (threshold x range x bootstrap) on every
sweep2d pkl matching `--target`, `--layer`, and the variant tag
encoded in `--variant_suffix` (default "" = canonical config).

Output:
  results/exp_map/data/fits/fits_<target>_L<layer>[<suffix>].pkl
  also prints a compact table to stdout.

Variant suffix examples:
  ""              canonical (geodesic, l2, measure=penult)
  "_M-7"          measure layer = penult-5
  "_additive"     additive perturbation
  "_metric_cos"   fit on cosine grid instead of L^2
  "_metric_kl"    fit on KL grid
"""
from __future__ import annotations
import os, sys, pickle, glob, argparse
sys.path.insert(0, ".")
import numpy as np
from scripts.exp_map.lib.superellipse import robust_p_fit, robust_p_fit_fixed_l2

OUT_DIR = "results/exp_map/data/fits"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--variant_suffix", default="",
                    help="filename suffix to match input sweep pkls; e.g. "
                         "'_M-7', '_additive', '_cos+l2' (matches "
                         "sweep_2d.py's --measure_offset / --mode / "
                         "--metrics flags)")
    ap.add_argument("--anchor_source", default="fineweb",
                    choices=["fineweb", "wiki_en", "wiki_zh", "code"],
                    help="match sweep pkls produced with this anchor source. "
                         "Default 'fineweb' keeps the canonical filename. "
                         "Other sources append '_src<label>' to both the "
                         "input glob and the output fits pkl.")
    ap.add_argument("--metric", default="l2", choices=["l2", "cos", "kl"],
                    help="which grid to fit on (must be present in the pkl)")
    ap.add_argument("--n_bootstrap", type=int, default=200,
                    help="bootstrap resamples per cell. 0 = use point "
                         "estimate (much faster; OK when each pair is "
                         "one dot in a 528-dot beeswarm and the dot "
                         "spread already reflects population uncertainty).")
    ap.add_argument("--out_suffix", default=None,
                    help="suffix on the output fits pkl. Default: input "
                         "variant_suffix + ('_metric_<m>' if metric != l2)")
    ap.add_argument("--thresh_l2", type=float, default=None,
                    help="if set, use fixed-L^2 protocol "
                         "(threshold = f * thresh_l2 for f in {0.5,1,2}, "
                         "max_alpha in {30,45,60} = 9 cells). Default "
                         "output suffix becomes '_thrfixed' (overridable "
                         "via --out_suffix).")
    ap.add_argument("--exact", action="store_true",
                    help="use exact-geodesic normalization "
                         "(sin alpha cos phi / sin alpha_1, sin alpha sin "
                         "phi / sin alpha_2) instead of the small-angle "
                         "approximation alpha cos phi / alpha_1. Adds "
                         "'_exact' to the default output suffix.")
    args = ap.parse_args()

    # anchor_source != fineweb tacks an extra _src<label> suffix on both
    # the input sweep glob and the output fits filename.
    src_suffix = ""
    if args.anchor_source != "fineweb":
        src_suffix = f"_src{args.anchor_source}"
    match_suffix = args.variant_suffix + src_suffix

    pat = (f"results/exp_map/data/sweeps_2d/sweep2d_{args.target}_L"
           f"{args.layer}_*deg{match_suffix}.pkl")
    files = sorted(glob.glob(pat))
    if not files:
        print(f"no files match {pat}"); return

    rows = []
    full = {}
    for fp in files:
        with open(fp, "rb") as f: d = pickle.load(f)
        ang = d["angles_deg"]
        if args.metric not in d:
            print(f"  skip {os.path.basename(fp)}: missing '{args.metric}'")
            continue
        grid = d[args.metric]   # (n_anchors, n, n)
        a, b = d["direction_labels"]
        if args.thresh_l2 is not None:
            fit = robust_p_fit_fixed_l2(ang, grid, args.thresh_l2,
                                         n_bootstrap=args.n_bootstrap,
                                         exact_geodesic=args.exact)
        else:
            fit = robust_p_fit(ang, grid, n_bootstrap=args.n_bootstrap,
                                exact_geodesic=args.exact)
        full[(a, b)] = fit
        # collapse cells into a row: low/high across all (threshold x range)
        ps = []
        for k, v in fit.items():
            if k == "__levels__": continue
            if np.isfinite(v["p_lo"]) and np.isfinite(v["p_hi"]):
                ps.append(v["p_lo"]); ps.append(v["p_hi"])
        if ps:
            row = {
                "pair": f"{a} x {b}",
                "p_lo_overall": float(min(ps)),
                "p_hi_overall": float(max(ps)),
                "n_cells_ok": int(sum(1 for k, v in fit.items()
                                       if k != "__levels__" and
                                       np.isfinite(v["p_median"]))),
            }
        else:
            row = {"pair": f"{a} x {b}", "p_lo_overall": np.nan,
                   "p_hi_overall": np.nan, "n_cells_ok": 0}
        rows.append(row)

    # sort by mid-p
    def mid(r):
        if np.isnan(r["p_lo_overall"]): return 99
        return 0.5 * (r["p_lo_overall"] + r["p_hi_overall"])
    rows = sorted(rows, key=mid)

    print(f"\n=== {args.target} L{args.layer}  {len(rows)} pairs ===")
    print(f"{'pair':40s}  p_range (across thresh x range x bootstrap)  cells_ok")
    for r in rows:
        if np.isnan(r["p_lo_overall"]):
            rng = "no fit"
        else:
            rng = f"[{r['p_lo_overall']:.2f}, {r['p_hi_overall']:.2f}]"
        print(f"{r['pair']:40s}  {rng:20s}  {r['n_cells_ok']}/6")

    if args.out_suffix is not None:
        suffix = args.out_suffix
    else:
        suffix = args.variant_suffix
        if args.metric != "l2":
            suffix += f"_metric_{args.metric}"
        suffix += src_suffix
        if args.exact:
            suffix += "_exact"
    out_pkl = os.path.join(OUT_DIR,
                            f"fits_{args.target}_L{args.layer}{suffix}.pkl")
    with open(out_pkl, "wb") as f: pickle.dump(full, f)
    print(f"\nsaved {out_pkl}")


if __name__ == "__main__":
    main()
