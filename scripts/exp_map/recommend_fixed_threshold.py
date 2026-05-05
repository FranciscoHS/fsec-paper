"""For each (target, layer), scan canonical DoM sweep2d pkls and report
the recommended fixed-L^2 threshold = f * median(axis_max), where
axis_max per pair = min(max(L2[:, 0]), max(L2[0, :])).

Defaults to f = 0.40 (Gemma's pick of 150 corresponds to f ≈ 0.39 of
its median axis-max).

Usage:
  python scripts/exp_map/recommend_fixed_threshold.py
  python scripts/exp_map/recommend_fixed_threshold.py --f 0.35
"""
from __future__ import annotations
import os, glob, pickle, argparse, re
import numpy as np

SWEEP_DIR = "results/exp_map/data/sweeps_2d"


def axis_max_per_pair(pkl_path: str, metric: str = "l2") -> float:
    """Per-pair denominator: min of the two 1D axis maxes."""
    with open(pkl_path, "rb") as f: d = pickle.load(f)
    if metric not in d: return float("nan")
    g = d[metric].mean(axis=0)
    return float(min(g[:, 0].max(), g[0, :].max()))


def scan_target_layer(target: str, layer: int, suffix: str = "",
                       metric: str = "l2"):
    sfx = f"_{suffix}" if suffix else ""
    pat = f"{SWEEP_DIR}/sweep2d_{target}_L{layer}_*_fineweb_60deg{sfx}.pkl"
    files = list(glob.glob(pat))
    if not suffix:
        # canonical scan: exclude variant-suffix files like _M-7, _additive,
        # _srcwiki_en. With an explicit suffix this filter would wrongly
        # drop the file we asked for.
        files = [p for p in files
                 if not re.search(r"_fineweb_\d+deg_[a-zA-Z]",
                                  os.path.basename(p))]
    vals = [axis_max_per_pair(p, metric) for p in files]
    return [v for v in vals if v == v]  # drop NaN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float, default=0.40,
                    help="threshold = f * median(axis_max). Gemma "
                         "L=2 choice 150 corresponds to f ≈ 0.39.")
    ap.add_argument("--targets", default="gemma,llama,qwen,mistral,aya,yi",
                    help="comma-sep target list")
    ap.add_argument("--layer", type=int, default=2)
    args = ap.parse_args()

    print(f"Recommended fixed L^2 thresholds (f = {args.f} × median axis-max):")
    print(f"{'target':<10s} {'layer':>5s} {'n_pairs':>8s} "
          f"{'med_axmax':>10s} {'thresh':>8s}")
    for target in args.targets.split(","):
        target = target.strip()
        vals = scan_target_layer(target, args.layer)
        if not vals:
            print(f"{target:<10s} {args.layer:>5d}   no sweeps")
            continue
        med = float(np.median(vals))
        thresh = args.f * med
        print(f"{target:<10s} {args.layer:>5d} {len(vals):>8d} "
              f"{med:>10.2f} {thresh:>8.1f}")


if __name__ == "__main__":
    main()
