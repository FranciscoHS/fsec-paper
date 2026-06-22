"""Run fit_pairs.py with a per-pair threshold for every condition.

For each (target, layer, variant_suffix, anchor_source, metric) condition,
call ``fit_pairs.py --per_pair_threshold``. Each sweep computes its own
threshold T_pair = f * min(axis-1 max, axis-2 max) from its median-over-anchors
grid, so there is no random-direction reference scan (that apparatus, and
``recommend_fixed_threshold.py`` / ``run_random_ref_sweeps.sh``, are retired by
the per-pair switch). The threshold cancels in the superellipse derivation, so
this measures the same scale-invariant exponent everywhere while guaranteeing
both axes cross -- which fixes the KL coverage problem the global threshold had.

Saves to fits/fits_<target>_L<layer><suffix>_thrpair[_exact].pkl. Mirrors
refit_thrfixed_all.py's CONDITIONS list (imported, single source of truth).
"""
from __future__ import annotations
import os, sys, subprocess, argparse
sys.path.insert(0, ".")

from scripts.refit_thrfixed_all import CONDITIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float, default=0.50,
                    help="T_pair = f * min(axis maxima) per sweep. Default "
                         "0.50 (locked headline). The 0.5/1/2 x T_pair "
                         "threshold-robustness ablation is layered on top.")
    ap.add_argument("--dry_run", action="store_true",
                    help="just print the planned commands; don't run.")
    ap.add_argument("--only", default="",
                    help="comma-sep filter on target names (e.g. 'gemma').")
    ap.add_argument("--exact", action="store_true",
                    help="forward --exact to fit_pairs.py (exact-geodesic "
                         "normalization). Output suffix becomes "
                         "'_thrpair_exact'.")
    args = ap.parse_args()
    only = {n for n in args.only.split(",") if n}

    print(f"per-pair threshold, f = {args.f}\n")
    for tgt, L, vsuf, asrc, metric in CONDITIONS:
        if only and tgt not in only:
            continue

        out_suf = vsuf
        if metric != "l2":
            out_suf += f"_metric_{metric}"
        if asrc != "fineweb":
            out_suf += f"_src{asrc}"
        out_suf += "_thrpair"
        if args.exact:
            out_suf += "_exact"

        cmd = [
            "python", "scripts/fit_pairs.py",
            "--target", tgt, "--layer", str(L),
            "--variant_suffix", vsuf,
            "--anchor_source", asrc,
            "--metric", metric,
            "--n_bootstrap", "0",
            "--per_pair_threshold",
            "--per_pair_f", f"{args.f:.6f}",
            "--out_suffix", out_suf,
        ]
        if args.exact:
            cmd.append("--exact")
        print(f"{tgt} L={L} vsuf='{vsuf}' src={asrc} metric={metric} "
              f"-> {out_suf}")
        if args.dry_run:
            print("  $ " + " ".join(cmd))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED: {r.stderr[-500:]}")
        else:
            tail = r.stdout.strip().split("\n")[-2:]
            for line in tail:
                print(f"  {line}")


if __name__ == "__main__":
    main()
