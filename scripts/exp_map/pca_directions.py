"""Build a cache of 33 PCA directions at L=2 by taking the top-33 right
singular vectors of (centred) last-token activations from a 10k-prompt
FineWeb sample. Doesn't reference the SAE basis at all — if the
SAE-feature axis is itself wrong, PCA gives the right reference for the
direction-family beeswarm.

Output:
  results/exp_map/data/directions/dirs_<target>_L<L>_pca_fineweb.pkl
  schema: matches the existing direction-cache schema (33 unit vectors,
  ``family='pca_fineweb'``).

Activations are cached in
  results/exp_map/data/activations/acts_<target>_L<L>_fineweb_<n>.pkl
via ``actlib.fineweb_acts_n`` so this can be re-run cheaply.

Usage:
  python -u scripts/exp_map/pca_directions.py --target gemma --layer 2
"""
from __future__ import annotations
import os, sys, pickle, argparse, time
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model
from scripts.exp_map.lib import activations as actlib

OUT_DIR = "results/exp_map/data/directions"
os.makedirs(OUT_DIR, exist_ok=True)

N_DIRS = 33
N_FINEWEB = 10000
SEED = 42


def ts(): return time.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DIRS,
                    help="number of top components to keep")
    ap.add_argument("--n_fineweb", type=int, default=N_FINEWEB,
                    help="size of the FineWeb sample for PCA")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--model", default="gemma-2-9b",
                    help="HF model id used by src.model.load_model")
    ap.add_argument("--fwd_batch", type=int, default=32)
    args = ap.parse_args()

    print(f"[{ts()}] loading {args.model}", flush=True)
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)

    print(f"[{ts()}] extracting / loading {args.n_fineweb} fineweb "
          f"activations (seed={args.seed})", flush=True)
    fw = actlib.fineweb_acts_n(
        model, tokenizer, device, args.target, args.layer,
        n=args.n_fineweb, seed=args.seed, batch_size=args.fwd_batch)
    acts = fw["activations"].float()       # (N, d)
    print(f"  acts: {tuple(acts.shape)}", flush=True)
    N, D = acts.shape

    # Free the model now — PCA is a pure linear-algebra job.
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"[{ts()}] centring + SVD on ({N}, {D}) matrix", flush=True)
    mean = acts.mean(dim=0, keepdim=True)
    X = acts - mean

    # SVD on GPU when available; (10k, 3584) fits comfortably.
    svd_device = "cuda" if torch.cuda.is_available() else "cpu"
    Xd = X.to(svd_device)
    t0 = time.time()
    # full_matrices=False -> economy SVD: U (N, k), S (k,), Vh (k, D),
    # k = min(N, D). Right singular vectors are rows of Vh.
    U, S, Vh = torch.linalg.svd(Xd, full_matrices=False)
    print(f"  SVD done ({time.time()-t0:.1f}s)  "
          f"S range [{S.min():.4f}, {S.max():.4f}]", flush=True)

    # Total variance for the explained-variance summary.
    var_total = float((S.double() ** 2).sum() / max(N - 1, 1))
    var_per = (S.double() ** 2 / max(N - 1, 1)).cpu()

    raw = Vh[:args.n].cpu().float()        # (n, D)
    # Vh rows are already unit-norm by SVD construction, but renormalise
    # against any fp roundoff for safety.
    dirs = F.normalize(raw, dim=-1)

    var_top = float(var_per[:args.n].sum())
    print(f"[{ts()}] top-{args.n} variance ratio: "
          f"{var_top/var_total:.4f}  ({var_top:.2f} / {var_total:.2f})",
          flush=True)
    print(f"  per-component variance (top 5): "
          f"{[float(v) for v in var_per[:5]]}")
    print(f"  per-component variance (#28..#33): "
          f"{[float(v) for v in var_per[args.n - 5:args.n]]}")

    names = [f"pca_{i:03d}" for i in range(args.n)]
    out = {
        "directions": {n: dirs[i].clone() for i, n in enumerate(names)},
        "family": "pca_fineweb",
        "signs": {n: "pca" for n in names},
        "prompt_set_hashes": {n: {"family": "pca_fineweb",
                                   "component_index": i,
                                   "n_fineweb": args.n_fineweb,
                                   "fineweb_seed": args.seed,
                                   "singular_value": float(S[i].item()),
                                   "explained_variance": float(var_per[i])}
                              for i, n in enumerate(names)},
        "n_fineweb": args.n_fineweb,
        "fineweb_seed": args.seed,
        "singular_values": S[:args.n].cpu().tolist(),
        "explained_variance": var_per[:args.n].tolist(),
        "explained_variance_ratio": float(var_top / var_total),
        "total_variance": var_total,
        "mean_act": mean.squeeze(0).cpu(),
        "seed": args.seed,
        "layer": args.layer,
        "model": args.target,
        "model_full": args.model,
        "d_model": D,
        "n_dirs": args.n,
    }
    out_path = os.path.join(
        OUT_DIR, f"dirs_{args.target}_L{args.layer}_pca_fineweb.pkl")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, out_path)
    print(f"[{ts()}] saved -> {out_path}", flush=True)
    print(f"  ||d|| range: "
          f"[{dirs.norm(dim=-1).min():.6f}, {dirs.norm(dim=-1).max():.6f}]")
    print(f"  pairwise |cos| max (off-diag): "
          f"{(dirs @ dirs.T).fill_diagonal_(0).abs().max():.4e}")


if __name__ == "__main__":
    main()
