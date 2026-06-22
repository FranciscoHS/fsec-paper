"""Fit the per-pair-threshold exponent for every rotated-contrastive sweep
produced for the LLM misalignment sweep (item 1).

Each (theta, seed) cell computes its own per-pair threshold
T_pair = f * min(axis-1 max, axis-2 max) from its median-over-anchors grid
(no shared random-direction reference scan), matching ``refit_thrpair_all.py``
everywhere else. We call ``fit_pairs.py --per_pair_threshold`` per cell.

Discovers cells by globbing the rotated sweep pkls, so it stays in sync with
whatever ``rotated_contrastive_directions.py`` + ``sweep_2d.py`` actually
produced. Output: results/fits/fits_<target>_L<layer>_dirrotcontrastive_
th<deg>_s<seed>_thrpair[_exact].pkl per cell.

Usage:
  python scripts/fit_misalignment_thrfixed.py --target gemma --layer 2 --exact
"""
from __future__ import annotations
import os, sys, re, glob, subprocess, argparse
sys.path.insert(0, ".")

SWEEP_DIR = "results/sweeps_2d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--f", type=float, default=0.50,
                    help="per-pair: T_pair = f * min(axis maxima). Default "
                         "0.50, matching refit_thrpair_all.py.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--exact", action="store_true",
                    help="forward --exact to fit_pairs.py (exact-geodesic "
                         "normalization, matching the paper's thrpair_exact "
                         "protocol). Output suffix becomes "
                         "'..._thrpair_exact'.")
    args = ap.parse_args()

    # Per-pair threshold: each rotated-contrastive cell computes its own
    # T_pair from its grid (no shared random-direction reference scan).
    print(f"per-pair threshold, f = {args.f}\n")

    # Discover the distinct (theta, seed) variant suffixes from the sweeps.
    pat = (f"{SWEEP_DIR}/sweep2d_{args.target}_L{args.layer}_*deg"
           f"_dirrotcontrastive_th*_s*.pkl")
    suffixes = set()
    rx = re.compile(r"(_dirrotcontrastive_th\d+_s\d+)\.pkl$")
    for fp in glob.glob(pat):
        m = rx.search(os.path.basename(fp))
        if m:
            suffixes.add(m.group(1))
    if not suffixes:
        print(f"no rotated sweeps match {pat}")
        return

    def _key(sfx):
        m = re.search(r"th(\d+)_s(\d+)", sfx)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

    for vsuf in sorted(suffixes, key=_key):
        out_suf = vsuf + "_thrpair" + ("_exact" if args.exact else "")
        cmd = [
            "python", "scripts/fit_pairs.py",
            "--target", args.target, "--layer", str(args.layer),
            "--variant_suffix", vsuf,
            "--metric", "l2",
            "--n_bootstrap", "0",
            "--per_pair_threshold",
            "--per_pair_f", f"{args.f:.6f}",
            "--out_suffix", out_suf,
        ]
        if args.exact:
            cmd.append("--exact")
        if args.dry_run:
            print("  $ " + " ".join(cmd))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED {vsuf}: {r.stderr[-400:]}")
        else:
            print(f"  ok {vsuf} -> fits_{args.target}_L{args.layer}"
                  f"{out_suf}.pkl")


if __name__ == "__main__":
    main()
