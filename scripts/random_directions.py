"""Build a cache of 33 isotropic random unit directions at gemma-2-9b's
residual width (D=3584). For the direction-type column of the exp_map
robustness beeswarm.

Output:
  results/directions/dirs_gemma_L2_random.pkl
  {
    'directions': {f'random_{i:03d}': Tensor[D]},
    'family':     'random',
    'signs':      {name: 'isotropic'},
    'prompt_set_hashes': {name: {...}},  # mirrors DoM cache schema
    'seed':       42,
    'layer':      2,
    'model':      'gemma',
  }

No model load needed.

Usage:
  python scripts/random_directions.py
"""
from __future__ import annotations
import os, sys, pickle, argparse
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

OUT_DIR = "results/directions"
os.makedirs(OUT_DIR, exist_ok=True)

# gemma-2-9b residual width (matches scripts/lib/registry.py target=gemma)
D_MODEL = 3584
N_DIRS = 33
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DIRS)
    ap.add_argument("--d", type=int, default=D_MODEL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    raw = torch.randn(args.n, args.d, generator=g)
    dirs = F.normalize(raw, dim=-1)

    out = {
        "directions": {f"random_{i:03d}": dirs[i].clone()
                       for i in range(args.n)},
        "family": "random",
        "signs": {f"random_{i:03d}": "isotropic" for i in range(args.n)},
        "prompt_set_hashes": {f"random_{i:03d}":
                              {"family": "random", "seed": args.seed,
                               "index": i}
                              for i in range(args.n)},
        "seed": args.seed,
        "layer": args.layer,
        "model": args.target,
        "d_model": args.d,
        "n_dirs": args.n,
    }
    out_path = os.path.join(
        OUT_DIR, f"dirs_{args.target}_L{args.layer}_random.pkl")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, out_path)
    print(f"saved {args.n} random unit vectors (D={args.d}) -> {out_path}")
    print(f"  ||d|| range: "
          f"[{dirs.norm(dim=-1).min():.6f}, {dirs.norm(dim=-1).max():.6f}]")
    print(f"  pairwise |cos| max (off-diag): "
          f"{(dirs @ dirs.T).fill_diagonal_(0).abs().max():.4f}")


if __name__ == "__main__":
    main()
