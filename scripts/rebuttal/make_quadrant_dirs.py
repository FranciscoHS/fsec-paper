"""E4 (R2 Q1) setup: directions + pair lists for the four-quadrant sweeps.

The paper sweeps only the (+d1, +d2) quadrant. Here we add, for each of the
318 overlap-filtered contrastive pairs (Gemma-2-9B, L2), the three remaining
sign combinations (+d1, -d2), (-d1, +d2), (-d1, -d2). Negated directions are
stored as '<name>_neg'. Gram-Schmidt in the sweep maps (a, -b) to
(a_perp, -b_perp), so each sweep is exactly the corresponding quadrant of the
original pair's plane.

Also writes a small reproducibility set of (+, +) pairs whose sweeps are
already cached, so we can check the pod reproduces the paper's grids.

Outputs:
  results/directions/dirs_gemma_L2_signed.pkl   (family 'signed')
  results/rebuttal/pairs_quadrants.txt          (954 pairs)
  results/rebuttal/pairs_quadrants_check.txt    (8 (+,+) pairs)

Run locally (CPU): python scripts/rebuttal/make_quadrant_dirs.py
"""
from __future__ import annotations
import os, pickle, sys
sys.path.insert(0, ".")
from scripts.plotting.plot_robustness_beeswarm import _load_dom_overlaps

EXCLUDE = {"HonestyShort", "TensePresent", "Formal"}
OVERLAP_MAX = 0.1
N_CHECK = 8


def main():
    with open("results/directions/dirs_gemma_L2.pkl", "rb") as f:
        blob = pickle.load(f)
    names = sorted(n for n in blob["directions"] if n not in EXCLUDE)
    assert len(names) == 33, len(names)

    dirs, signs = {}, {}
    for n in names:
        d = blob["directions"][n]
        dirs[n] = d.clone()
        dirs[f"{n}_neg"] = (-d).clone()
        signs[n] = blob["signs"].get(n)
        signs[f"{n}_neg"] = f"negated({blob['signs'].get(n)})"
    out = {"directions": dirs, "signs": signs, "family": "signed",
           "layer": 2, "model": "gemma",
           "source": "results/directions/dirs_gemma_L2.pkl"}
    with open("results/directions/dirs_gemma_L2_signed.pkl", "wb") as f:
        pickle.dump(out, f)

    ov = _load_dom_overlaps("gemma", 2)
    kept = sorted((a, b) for i, a in enumerate(names) for b in names[i + 1:]
                  if ov[frozenset((a, b))] < OVERLAP_MAX)
    assert len(kept) == 318, len(kept)
    # Use the (d1, d2) order of the original paper sweeps: the sweep keeps d1
    # exact and Gram-Schmidts d2 against it, so order changes the axes.
    def orig_order(a, b):
        if os.path.exists(f"results/sweeps_2d/sweep2d_gemma_L2_{a}__{b}_fineweb_60deg.pkl"):
            return a, b
        assert os.path.exists(f"results/sweeps_2d/sweep2d_gemma_L2_{b}__{a}_fineweb_60deg.pkl"), (a, b)
        return b, a
    kept = [orig_order(a, b) for a, b in kept]

    os.makedirs("results/rebuttal", exist_ok=True)
    with open("results/rebuttal/pairs_quadrants.txt", "w") as f:
        f.write("# a,b,quadrant\n")
        for a, b in kept:
            f.write(f"{a},{b}_neg,+-\n")
            f.write(f"{a}_neg,{b},-+\n")
            f.write(f"{a}_neg,{b}_neg,--\n")
    step = len(kept) // N_CHECK
    with open("results/rebuttal/pairs_quadrants_check.txt", "w") as f:
        f.write("# a,b,quadrant  ((+,+) pairs already cached, for reproducibility)\n")
        for a, b in kept[::step][:N_CHECK]:
            f.write(f"{a},{b},++\n")
    print(f"33 directions -> 66 signed; {len(kept)} pairs -> {3 * len(kept)} quadrant sweeps")


if __name__ == "__main__":
    main()
