"""Run fit_pairs.py with a per-condition fixed-L^2 (or fixed-metric)
threshold. For each condition (target, layer, variant_suffix, metric,
anchor_source), compute the recommended threshold as
f × median(random-direction plateau height) over the matching random x random
reference sweeps, and call fit_pairs.py.

Saves to fits/fits_<target>_L<layer><suffix>_thrfixed.pkl.
"""
from __future__ import annotations
import os, sys, re, subprocess, argparse
sys.path.insert(0, ".")
import numpy as np

from scripts.recommend_fixed_threshold import scan_target_layer


def ref_random_suffix(vsuf: str, asrc: str) -> str:
    """Full variant suffix of the random reference sweep for this condition,
    matching sweep_2d.py's filename order: [M][mode][metric] _dirrandom
    [_pos] [_src]. The reference shares the perturbation/measurement settings
    but swaps the direction family to random (every direction-family cell
    therefore anchors on the same main random sweep)."""
    v = re.sub(r"_dir[A-Za-z0-9_]+", "", vsuf)        # drop the dir family
    m = re.search(r"_pos-?\d+", v)
    pos = m.group(0) if m else ""                     # _pos goes after dirrandom
    pre = v.replace(pos, "")                          # M / mode / metric: before
    src = f"_src{asrc}" if asrc != "fineweb" else ""  # _src goes after dirrandom
    return f"{pre}_dirrandom{pos}{src}"

# (target, layer, variant_suffix, anchor_source, metric)
# variant_suffix matches the sweep filename suffix appended after the
# 60deg tag. anchor_source = "fineweb" → no extra suffix.
CONDITIONS = [
    # Model column
    ("gemma",   2, "",          "fineweb", "l2"),
    ("llama",   2, "",          "fineweb", "l2"),
    ("qwen",    2, "",          "fineweb", "l2"),
    ("mistral", 2, "",          "fineweb", "l2"),
    ("aya",     2, "",          "fineweb", "l2"),
    ("yi",      2, "",          "fineweb", "l2"),
    # Perturb-layer column (gemma)
    ("gemma",   5,  "",         "fineweb", "l2"),
    ("gemma",  10,  "",         "fineweb", "l2"),
    ("gemma",  20,  "",         "fineweb", "l2"),
    # Measure-layer column (gemma)
    ("gemma",   2, "_M-7",      "fineweb", "l2"),
    ("gemma",   2, "_M-12",     "fineweb", "l2"),
    # Method column (gemma) — additive
    ("gemma",   2, "_additive", "fineweb", "l2"),
    # Anchor-source column (gemma)
    ("gemma",   2, "",          "wiki_en", "l2"),
    ("gemma",   2, "",          "wiki_zh", "l2"),
    ("gemma",   2, "",          "code",    "l2"),
    # Metric column (gemma) — cos/kl on the cos+kl+l2 sweep set
    ("gemma",   2, "_cos+kl+l2", "fineweb", "cos"),
    ("gemma",   2, "_cos+kl+l2", "fineweb", "kl"),
    # Direction-family column (gemma) — SAE / MELBO / random
    ("gemma",   2, "_dirsae_random", "fineweb", "l2"),
    ("gemma",   2, "_dirmelbo",      "fineweb", "l2"),
    ("gemma",   2, "_dirrandom",     "fineweb", "l2"),
    # Direction-family — better baselines
    ("gemma",   2, "_dirsae_eval",    "fineweb", "l2"),
    ("gemma",   2, "_dirsae_fineweb", "fineweb", "l2"),
    ("gemma",   2, "_dirpca_fineweb", "fineweb", "l2"),
    # Token-position ablation (gemma) — perturb at non-final positions.
    # pos=-5 with seq_len=5 lands on Gemma's BOS (an attention-sink
    # position with anomalous geometry); the kept clean far-back data
    # point is pos=-3.
    ("gemma",   2, "_pos-2", "fineweb", "l2"),
    ("gemma",   2, "_pos-3", "fineweb", "l2"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float, default=0.50,
                    help="threshold = f × median(random plateau height). "
                         "Default 0.50; ablated downstream.")
    ap.add_argument("--f_kl", type=float, default=0.50,
                    help="separate f for the KL metric cell. Kept uniform "
                         "with --f by default: the random reference is "
                         "measured in KL's own units, so no special-casing "
                         "is needed (override if a KL contour fails to close).")
    ap.add_argument("--dry_run", action="store_true",
                    help="just print the planned commands; don't run.")
    ap.add_argument("--only", default="",
                    help="comma-sep filter on target names (e.g. 'gemma').")
    ap.add_argument("--exact", action="store_true",
                    help="forward --exact to fit_pairs.py (use exact-"
                         "geodesic normalization instead of the small-"
                         "angle approximation). Output suffix becomes "
                         "'_thrfixed_exact'.")
    args = ap.parse_args()
    only = {n for n in args.only.split(",") if n}

    print(f"f = {args.f}\n")
    for tgt, L, vsuf, asrc, metric in CONDITIONS:
        if only and tgt not in only: continue
        # random-direction plateau heights for this exact condition
        vals = scan_target_layer(tgt, L, ref_random_suffix(vsuf, asrc), metric)
        if not vals:
            print(f"[skip] {tgt} L={L} vsuf='{vsuf}' src={asrc} metric={metric}: "
                  f"no random reference sweeps")
            continue
        med = float(np.median(vals))
        f = args.f_kl if metric == "kl" else args.f
        T = f * med
        cond = (f"{tgt} L={L} vsuf='{vsuf}' src={asrc} metric={metric}: "
                f"n_pairs={len(vals)} med={med:.3f} T={T:.3f}")
        print(cond)

        out_suf = vsuf
        if metric != "l2": out_suf += f"_metric_{metric}"
        if asrc != "fineweb": out_suf += f"_src{asrc}"
        out_suf += "_thrfixed"
        if args.exact: out_suf += "_exact"

        cmd = [
            "python", "scripts/fit_pairs.py",
            "--target", tgt, "--layer", str(L),
            "--variant_suffix", vsuf,
            "--anchor_source", asrc,
            "--metric", metric,
            "--n_bootstrap", "0",
            "--thresh_l2", f"{T:.6f}",
            "--out_suffix", out_suf,
        ]
        if args.exact: cmd.append("--exact")
        if args.dry_run:
            print("  $ " + " ".join(cmd))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED: {r.stderr[-500:]}")
        else:
            tail = r.stdout.strip().split("\n")[-2:]
            for line in tail: print(f"  {line}")


if __name__ == "__main__":
    main()
