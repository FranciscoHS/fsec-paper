"""Pre-filter pair list to |cos(d_a, d_b)| <= 0.10 for the token-
position ablation sweep. Writes the surviving 318-of-528 pairs over
the canonical 33-direction set to a CSV that sweep_2d.py
``--pairs_file`` consumes.

Usage (from repo root):
  python scripts/prefilter_pos_pairs.py
"""
import sys, os, pickle, csv, itertools
sys.path.insert(0, ".")
import torch, torch.nn.functional as F

from scripts.lib import registry

CACHE = "results/directions/dirs_gemma_L2.pkl"
OUT = "results/pos_pairs_gemma_L2.csv"
THR = 0.10

# Canonical 33-direction list from the paper: 14 binary semantic + 10
# natural language + 9 programming language. Excludes TensePresent,
# HonestyShort, Formal (added to DIRECTIONS later as variants/dupes
# but not part of the canonical set).
SEMANTIC_14 = [
    "Age", "Certainty", "Era", "Gender", "Health", "Honesty", "Literary",
    "Number", "Person", "Refusal", "Sentiment", "Status", "Tense", "Wealth",
]


def main():
    with open(CACHE, "rb") as f:
        dirs = pickle.load(f)["directions"]
    names = SEMANTIC_14 + list(registry.LANG_KEYS) + list(registry.CODE_KEYS)
    assert len(names) == 33, f"expected 33 canonical directions, got {len(names)}"
    missing = [n for n in names if n not in dirs]
    if missing:
        print(f"warning: {len(missing)} names missing from cache: {missing}")
        names = [n for n in names if n in dirs]
    print(f"using {len(names)} directions")
    survivors = []
    high = 0
    for a, b in itertools.combinations(names, 2):
        d1 = F.normalize(dirs[a].float(), dim=0)
        d2 = F.normalize(dirs[b].float(), dim=0)
        c = float((d1 * d2).sum().abs())
        if c <= THR:
            survivors.append((a, b, c))
        else:
            high += 1
    print(f"C({len(names)},2)={len(names)*(len(names)-1)//2} total, "
          f"{len(survivors)} survive |cos|<={THR}, {high} dropped")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "abs_cos"])
        for a, b, c in survivors:
            w.writerow([a, b, f"{c:.4f}"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
