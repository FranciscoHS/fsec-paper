"""Recommended threshold = the random-direction reference scale:

    T = f * median_i [ max_alpha  mean_anchors  metric(alpha; random_i) ]

i.e. f (default 0.5) times the median, over a set of isotropic random unit
directions, of the 1D plateau height each random direction reaches when we
perturb activations along it. The plateau height of one random direction is
read off the *edge* of a random x random 2D sweep: with anchor-mean grid
g = sweep[metric].mean(axis=0), g[:, 0] is the 1D response along direction 1
(direction 2 pinned at alpha=0) and g[0, :] the response along direction 2.
Each random x random sweep therefore contributes two single-direction plateau
samples, one per axis.

This replaces the older per-feature-pair definition (min of the two single-
axis maxima, median over feature pairs), which dragged in feature pairs
unnecessarily: the threshold is only a response *level* at which to slice the
already-collected 2D feature surfaces, and a level is a 1D quantity. Anchoring
on random directions makes that explicit and removes the feature pairs. The
fraction f is ablated downstream, so 0.5 is only a starting point.

Usage:
  python scripts/recommend_fixed_threshold.py
  python scripts/recommend_fixed_threshold.py --f 0.5
"""
from __future__ import annotations
import glob, pickle, argparse
from collections import defaultdict
import numpy as np

SWEEP_DIR = "results/sweeps_2d"


def random_sweep_files(target: str, layer: int,
                       full_suffix: str = "_dirrandom") -> list[str]:
    """Random x random reference sweeps for this condition. ``full_suffix`` is
    the COMPLETE variant suffix as sweep_2d.py writes it, including the
    ``_dirrandom`` direction-family tag in its correct position (after
    metric, before _src/_pos) -- e.g. '_dirrandom', '_M-7_dirrandom',
    '_dirrandom_srcwiki_en', '_dirrandom_pos-2'. See ref_random_suffix() in
    refit_thrfixed_all.py, which builds it from a condition."""
    pat = (f"{SWEEP_DIR}/sweep2d_{target}_L{layer}_random_*__random_*"
           f"_fineweb_60deg{full_suffix}.pkl")
    return sorted(glob.glob(pat))


def plateau_heights(files: list[str], metric: str = "l2") -> dict[str, float]:
    """Per random direction, its 1D plateau height = max over angle of the
    anchor-mean response along that direction. Read from both grid edges and
    averaged across the partners it happened to be swept against (each edge is
    a pure 1D sweep of that direction, so partners agree up to anchor noise).
    Returns {dir_name: height}."""
    per_dir: dict[str, list[float]] = defaultdict(list)
    for p in files:
        with open(p, "rb") as f: d = pickle.load(f)
        if metric not in d: continue
        g = d[metric].mean(axis=0)                       # (n_alpha, n_alpha)
        labels = d.get("direction_labels")
        if not labels: continue
        n1, n2 = labels
        per_dir[n1].append(float(g[:, 0].max()))         # axis-1 edge
        per_dir[n2].append(float(g[0, :].max()))         # axis-2 edge
    return {k: float(np.mean(v)) for k, v in per_dir.items()}


def scan_target_layer(target: str, layer: int, full_suffix: str = "_dirrandom",
                       metric: str = "l2") -> list[float]:
    """Per-direction plateau heights for the random reference set.
    refit_thrfixed_all.py takes f * median(.) of the returned list. Returns []
    when no random reference sweep exists for the condition."""
    files = random_sweep_files(target, layer, full_suffix)
    heights = plateau_heights(files, metric)
    return [v for v in heights.values() if v == v]  # drop NaN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float, default=0.50,
                    help="threshold = f * median(random plateau height).")
    ap.add_argument("--targets", default="gemma,llama,qwen,mistral,aya,yi",
                    help="comma-sep target list")
    ap.add_argument("--layer", type=int, default=2)
    args = ap.parse_args()

    print(f"Recommended thresholds (f = {args.f} x median random plateau):")
    print(f"{'target':<10s} {'layer':>5s} {'n_dirs':>7s} "
          f"{'med_h':>10s} {'thresh':>8s}")
    for target in args.targets.split(","):
        target = target.strip()
        vals = scan_target_layer(target, args.layer)
        if not vals:
            print(f"{target:<10s} {args.layer:>5d}   no random sweeps")
            continue
        med = float(np.median(vals))
        print(f"{target:<10s} {args.layer:>5d} {len(vals):>7d} "
              f"{med:>10.2f} {args.f * med:>8.1f}")


if __name__ == "__main__":
    main()
