#!/usr/bin/env python3
"""Select ~30 cross-family contrastive pairs for the Qwen3.6-27B run.

Extracts all 33 DoM directions at (qwen36, L36) if the cache is missing
(loads the 27B — run on pod), computes the 33x33 overlap matrix, keeps
cross-family pairs with |cos| < 0.1 (paper filter), and picks N_PAIRS
balanced across the three family combinations (semantic x language,
semantic x code, language x code), preferring low overlap and capping how
often any single direction is reused.

Outputs (results/exp_map/data/):
  qwen36_pairs_L36.csv       'a,b' lines for sweep_2d.py --pairs_file
  qwen36_pairs_L36_report.txt  overlap stats + chosen pairs
"""
from __future__ import annotations
import os, sys, argparse, itertools
sys.path.insert(0, ".")
import numpy as np
import torch

from scripts.exp_map.lib import registry, directions as dirlib

TARGET = "qwen36"
COS_MAX = 0.1
N_PAIRS = 30
PER_COMBO = 10          # target 10 pairs per family combination
MAX_USES = 4            # cap per direction so no direction dominates

OUT_DIR = "results/exp_map/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int,
                    default=registry.TARGETS[TARGET]["perturb_layer"])
    args = ap.parse_args()
    layer = args.layer

    names = registry.all_direction_names()   # semantic + language + code (33)
    cache = os.path.join(dirlib.OUT_DIR, f"dirs_{TARGET}_L{layer}.pkl")
    if os.path.exists(cache):
        import pickle
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        missing = [n for n in names if n not in blob["directions"]]
    else:
        blob, missing = None, names
    if missing:
        from src.model import load_model
        cfg = registry.TARGETS[TARGET]
        print(f"Extracting {len(missing)} directions "
              f"(loading {cfg['model']})...", flush=True)
        model, tok, dev = load_model(cfg["model"], dtype=torch.bfloat16)
        blob = dirlib.extract_all(model, tok, dev, TARGET, layer, names=names)

    dirs = blob["directions"]
    vecs = {n: dirs[n].float().numpy() for n in names}

    # overlap matrix + cross-family candidates
    report = [f"Qwen3.6-27B contrastive pair selection — L{layer}, "
              f"|cos| < {COS_MAX}, {N_PAIRS} pairs\n"]
    cands = {"semantic-language": [], "semantic-code": [], "language-code": []}
    n_cross, n_pass = 0, 0
    for a, b in itertools.combinations(names, 2):
        fa, fb = registry.family(a), registry.family(b)
        if fa == fb:
            continue
        n_cross += 1
        c = float(vecs[a] @ vecs[b])
        if abs(c) >= COS_MAX:
            continue
        n_pass += 1
        combo = "-".join(sorted((fa, fb),
                                key=["semantic", "language", "code"].index))
        cands[combo].append((abs(c), a, b))
    report.append(f"cross-family pairs: {n_cross}, passing filter: {n_pass}")
    for combo, lst in cands.items():
        report.append(f"  {combo}: {len(lst)} candidates")

    # greedy pick: per combo, lowest |cos| first, with per-direction use cap
    uses: dict[str, int] = {}
    chosen = []
    for combo in cands:
        picked = 0
        for c, a, b in sorted(cands[combo]):
            if picked >= PER_COMBO:
                break
            if uses.get(a, 0) >= MAX_USES or uses.get(b, 0) >= MAX_USES:
                continue
            chosen.append((combo, c, a, b))
            uses[a] = uses.get(a, 0) + 1
            uses[b] = uses.get(b, 0) + 1
            picked += 1
        report.append(f"{combo}: picked {picked}")
    # top-up from any combo if short of N_PAIRS
    if len(chosen) < N_PAIRS:
        taken = {(a, b) for _, _, a, b in chosen}
        rest = sorted(x for lst in cands.values() for x in lst
                      if (x[1], x[2]) not in taken)
        for c, a, b in rest:
            if len(chosen) >= N_PAIRS:
                break
            chosen.append(("topup", c, a, b))

    report.append(f"\nchosen {len(chosen)} pairs:")
    for combo, c, a, b in chosen:
        report.append(f"  {a:>12s} x {b:<12s} |cos|={c:.3f}  ({combo})")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"qwen36_pairs_L{layer}.csv")
    with open(csv_path, "w") as f:
        f.write("a,b\n")
        for _, _, a, b in chosen:
            f.write(f"{a},{b}\n")
    rep_path = os.path.join(OUT_DIR, f"qwen36_pairs_L{layer}_report.txt")
    with open(rep_path, "w") as f:
        f.write("\n".join(report))
    print("\n".join(report))
    print(f"\nSaved: {csv_path}\n       {rep_path}")


if __name__ == "__main__":
    main()
